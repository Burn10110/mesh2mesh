"""3D view: model in one of the view modes (shading.py) with base colour maps in MATERIAL,
optional wireframe of polygon edges, ground grid, axis triad (pyqtgraph.opengl). ViewBar — the
mode buttons above it.

The model is shown rotated as it will be saved, centred and normalised to a unit diagonal:
scaling does not change the picture, only the grid cell label.
"""

import traceback

import numpy as np
import pyqtgraph.opengl as gl
from OpenGL import GL
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QActionGroup, QColor, QOpenGLContext, QVector3D
from PySide6.QtOpenGL import QOpenGLBuffer
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QLabel, QMenu, QToolButton, QWidget

from mesh2mesh.i18n import tr

from . import display, shading, teximage


class CornerNormalsMeshData(gl.MeshData):
    """Face-indexed MeshData with our own normal per triangle corner: GLMeshItem(smooth=False)
    draws face-indexed data with faceNormals(indexed='faces'), which is (t, 3, 3) — so the
    file's / auto-smoothed normals go to the shader as they are."""

    def __init__(self, normals, **kwds):
        super().__init__(**kwds)
        self.corner_normals = normals

    def faceNormals(self, indexed=None):
        if indexed == "faces" and self.corner_normals is not None:
            return self.corner_normals
        return super().faceNormals(indexed)


class TexturedMeshItem(gl.GLMeshItem):
    """GLMeshItem (face-indexed, no edges) that also feeds a_texcoord and draws the triangles in
    groups, one base colour map bound per group. The caller orders the triangles by map so a
    group is a contiguous range: [(first triangle, count, image index or -1)]."""

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.m_vbo_uv = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self.uv = None              # (t * 3, 2) float32 corner UVs, or None
        self.groups = []
        self.images = []            # teximage.Image per map index
        self.use_textures = False   # the current mode samples maps
        self._uv_dirty = False
        self._tex = {}              # map index -> GL texture id (0 = cannot be shown)
        self._stale = []            # ids to delete once the context is current

    def set_textures(self, uv, groups, images):
        self._stale += [t for t in self._tex.values() if t]
        self._tex = {}
        self.uv = None if uv is None else np.ascontiguousarray(uv, dtype=np.float32).reshape(-1, 2)
        self.groups, self.images = list(groups), list(images)
        self._uv_dirty = True
        self.update()

    def _upload_uv(self):
        vbo = self.m_vbo_uv
        if self.uv is None:
            vbo.destroy()
            return
        if not vbo.isCreated():
            vbo.create()
        vbo.bind()
        if vbo.size() != self.uv.nbytes:
            vbo.allocate(self.uv, self.uv.nbytes)
        else:
            vbo.write(0, self.uv, self.uv.nbytes)
        vbo.release()

    def _texture(self, i):
        """GL texture of map i, uploaded on first use; 0 when it cannot be shown."""
        if i < 0 or i >= len(self.images):
            return 0
        if i not in self._tex:
            self._tex[i] = 0
            img = self.images[i]
            if img.pixels is not None:
                try:
                    self._tex[i] = _upload_texture(img.pixels)
                except Exception as e:  # driver refused (size, memory): material colour instead
                    print(f"texture {img.name!r} not uploaded: {e}")
        return self._tex[i]

    def paint(self):
        self.setupGLState()
        if dirty := self.parseMeshData():
            self.upload_vertex_buffers(dirty)
        if self._uv_dirty:
            self._upload_uv()
            self._uv_dirty = False
        if self._stale:
            GL.glDeleteTextures(self._stale)
            self._stale = []
        if self.vertexes is None or not self.opts["drawFaces"]:
            return
        mvp = np.array(self.mvpMatrix().data(), dtype=np.float32)
        normal = np.array(self.modelViewMatrix().normalMatrix().data(), dtype=np.float32)
        es2 = QOpenGLContext.currentContext().hasExtension(b"GL_ARB_ES2_compatibility")
        shader = self.shader()
        program = shader.program(es2_compat=es2)

        enabled = []
        for name, vbo, size, data in (("a_position", self.m_vbo_position, 3, self.vertexes),
                                      ("a_normal", self.m_vbo_normal, 3, self.normals),
                                      ("a_color", self.m_vbo_color, 4, self.colors),
                                      ("a_texcoord", self.m_vbo_uv, 2, self.uv)):
            loc = GL.glGetAttribLocation(program, name)
            if loc == -1:            # not used by this mode's shader (optimised out)
                continue
            if data is None:
                GL.glVertexAttrib4f(loc, *((1, 1, 1, 1) if name == "a_color" else (0, 0, 1, 1)))
                continue
            vbo.bind()
            GL.glVertexAttribPointer(loc, size, GL.GL_FLOAT, False, 0, None)
            vbo.release()
            GL.glEnableVertexAttribArray(loc)
            enabled.append(loc)

        n = int(np.prod(self.vertexes.shape[:-1]))
        with shader:
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(program, "u_mvp"), 1, False, mvp)
            if (loc := GL.glGetUniformLocation(program, "u_normal")) != -1:
                GL.glUniformMatrix3fv(loc, 1, False, normal)
            use = GL.glGetUniformLocation(program, "u_use_tex")
            if use == -1 or not self.use_textures or self.uv is None or not self.groups:
                if use != -1:
                    GL.glUniform1f(use, 0.0)
                GL.glDrawArrays(GL.GL_TRIANGLES, 0, n)
            else:
                GL.glActiveTexture(GL.GL_TEXTURE0)
                if (loc := GL.glGetUniformLocation(program, "u_tex")) != -1:
                    GL.glUniform1i(loc, 0)
                for first, count, i in self.groups:
                    tex = self._texture(i)
                    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
                    GL.glUniform1f(use, 1.0 if tex else 0.0)
                    GL.glDrawArrays(GL.GL_TRIANGLES, 3 * first, 3 * count)
                GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        for loc in enabled:
            GL.glDisableVertexAttribArray(loc)


def _upload_texture(px):
    """(h, w, 4) uint8 RGBA, bottom row first -> GL texture id (repeat, mipmapped)."""
    h, w = px.shape[:2]
    tex = int(GL.glGenTextures(1))
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    try:
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, w, h, 0, GL.GL_RGBA,
                        GL.GL_UNSIGNED_BYTE, px)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        try:
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER,
                               GL.GL_LINEAR_MIPMAP_LINEAR)
        except Exception:      # no glGenerateMipmap (GL < 3.0 without the extension)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    except Exception:
        GL.glDeleteTextures([tex])
        raise
    finally:
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    return tex


class MeshView(gl.GLViewWidget):
    # No usable OpenGL (remote desktop, VM, ancient driver): pyqtgraph raises on every paint.
    # Emitted once, after the paint; the window then swaps the view for a notice.
    gl_failed = Signal(str)

    def __init__(self, parent=None):
        self.gl_error = None  # set before super(): Qt may paint during construction
        super().__init__(parent)
        self.setBackgroundColor(QColor(46, 48, 54))
        self.disp = None
        self.selected = None     # set of part indices, or None
        self.rotation = np.eye(3)
        self.mode = shading.MATERIAL
        self.wireframe = True
        self.smoothing = display.SMOOTH_FILE
        self._md = None
        self._programs = {}      # mode -> ShaderProgram of this view's GL context
        # smooth=False: face-indexed data, normals are ours (CornerNormalsMeshData)
        self.mesh = TexturedMeshItem(smooth=False, glOptions=shading.gl_options(self.mode),
                                     shader=self._program(self.mode))
        self.mesh.use_textures = shading.uses_textures(self.mode)
        self._order = None  # triangle order of the GL arrays (grouped by map), None = as is
        self.edges = gl.GLLinePlotItem(mode="lines", color=shading.edge_color(self.mode),
                                       width=1.0, antialias=True, glOptions="translucent")
        self.grid = gl.GLGridItem(color=(255, 255, 255, 40))
        self.triad = gl.GLLinePlotItem(mode="lines", width=3.0, antialias=True)
        self.labels = [gl.GLTextItem(text=t, color=c) for t, c in
                       (("X", (240, 90, 90)), ("Y", (110, 210, 110)), ("Z", (110, 150, 255)))]
        for item in self._items():
            item.setVisible(False)
            self.addItem(item)
        # orbit and wheel are pyqtgraph's; panning is ours (mouseMoveEvent)
        self.hint = QLabel(self)
        self.hint.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.hint.setStyleSheet("color: rgba(255,255,255,110); background: transparent;"
                                " font-size: 8pt;")
        self.retranslate()
        self.reset_camera()

    def retranslate(self):
        """Texts in the current language (mesh2mesh.i18n)."""
        self.labels[2].setData(text=tr("Z вверх"))
        self.hint.setText(tr("ЛКМ — вращать · колесо — зум · СКМ / Ctrl+ЛКМ — сдвиг "
                             "(+Shift — по полу)"))
        self.hint.adjustSize()
        self._place_hint()

    def _place_hint(self):
        h = self.hint
        h.move((self.width() - h.width()) // 2, self.height() - h.height() - 6)

    def _program(self, mode):
        if mode not in self._programs:
            self._programs[mode] = shading.program(mode)
        return self._programs[mode]

    def _items(self):
        return (self.grid, self.mesh, self.edges, self.triad, *self.labels)

    def showEvent(self, ev):
        super().showEvent(ev)
        # no context at all (Qt only logs "Failed to create context"): nothing raises, the view
        # just stays black. initializeGL runs on the first expose, well within this delay.
        if not hasattr(self, "_gl_watch"):
            self._gl_watch = True
            QTimer.singleShot(2000, self._check_initialized)

    def _check_initialized(self):
        if self.gl_error is None and not getattr(self, "_gl_ready", False) and self.isVisible():
            self.gl_error = "OpenGL context was not created"
            print(self.gl_error)
            self.gl_failed.emit(self.gl_error)

    def initializeGL(self):
        self._gl_ready = True
        self._guard(super().initializeGL)

    def paintGL(self):
        if self.gl_error is None:
            self._guard(super().paintGL)

    def _guard(self, fn):
        try:
            fn()
        except Exception as e:
            traceback.print_exc()
            desc = getattr(e, "description", None)  # OpenGL.error.GLError: b'invalid operation'
            desc = desc.decode(errors="replace") if isinstance(desc, bytes) else str(e)
            self.gl_error = f"{type(e).__name__}: {desc.strip().splitlines()[0] if desc else ''}"
            QTimer.singleShot(0, lambda: self.gl_failed.emit(self.gl_error))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._place_hint()

    def mouseMoveEvent(self, ev):
        """pyqtgraph pans along the floor only (middle button). Ours: middle / Ctrl+left — in the
        screen plane, so up/down reaches Z; with Shift — along the floor as before."""
        buttons, mods = ev.buttons(), ev.modifiers()
        middle = buttons == Qt.MiddleButton
        ctrl_left = buttons == Qt.LeftButton and mods & Qt.ControlModifier
        if not (middle or ctrl_left):
            super().mouseMoveEvent(ev)
            return
        pos = ev.position()
        last = getattr(self, "mousePos", pos)
        self.mousePos = pos
        o = self.opts
        scale = o["distance"] * 2 * np.tan(np.radians(o["fov"]) / 2) / max(self.width(), 1)
        d = display.pan_offset(o["azimuth"], o["elevation"], pos.x() - last.x(),
                               pos.y() - last.y(), scale, floor=bool(mods & Qt.ShiftModifier))
        o["center"] = o["center"] + QVector3D(*map(float, d))
        self.update()

    def reset_camera(self):
        self.setCameraPosition(pos=QVector3D(0, 0, 0), distance=1.9, elevation=22, azimuth=-60)
        self.update()

    @property
    def _has(self):
        return self.disp is not None and not self.disp.empty

    def set_display(self, disp, rotation, reset_camera=True, images=None):
        """New model (or None to clear). images: teximage.Image per disp.textures (decoded off
        the UI thread by the loader); None = decode here."""
        self.disp, self.selected = disp, None
        for item in self._items():
            item.setVisible(self._has)
        self._md = None
        self._set_textures(images)
        self._show_edges()
        if self._has:
            self.set_rotation(rotation)
            if reset_camera:
                self.reset_camera()
        self.update()

    def _set_textures(self, images):
        """Order the triangles by map (groups = contiguous ranges) and hand the maps over."""
        self._order, groups, uv = None, [], None
        if self._has and self.disp.corner_uv is not None:
            if images is None:
                images = teximage.decode_all(self.disp.textures)
            self._order, groups = display.texture_groups(self.disp.tri_texture)
            uv = self.disp.corner_uv[self._order]
        self.mesh.set_textures(uv, groups, images or [])

    def _perm(self, a):
        """Per-triangle array in the GL order."""
        return a if self._order is None else a[self._order]

    def set_rotation(self, rotation):
        self.rotation = np.asarray(rotation, dtype=float)
        if not self._has:
            return
        v, lo, hi = display.placed(self.disp.verts, self.rotation)
        self._v = v.astype(np.float32)
        self._md = CornerNormalsMeshData(self._normals(),
                                         vertexes=self._v[self._perm(self.disp.tris)],
                                         vertexColors=self._colors())
        self.mesh.setMeshData(meshdata=self._md)
        if len(self.disp.edges):
            self.edges.setData(pos=self._v[self.disp.edges.reshape(-1)])
        self._lo, self._hi = lo, hi
        self._place_triad(lo, hi)
        if self.selected:
            self.select(self.selected)  # keep the focused part in view after a turn
        self.update()

    def set_grid(self, step_normalised):
        """Ground grid under the model; step in the normalised view units."""
        if not self._has or step_normalised <= 0:
            return
        lo, hi = self._lo, self._hi
        n = np.ceil((hi[:2] - lo[:2]) / step_normalised) + 2
        if n.max() > 400:  # never draw thousands of lines
            step_normalised *= np.ceil(n.max() / 400)
            n = np.ceil((hi[:2] - lo[:2]) / step_normalised) + 2
        self.grid.setSpacing(step_normalised, step_normalised, 1)
        self.grid.setSize(n[0] * step_normalised, n[1] * step_normalised, 1)
        self.grid.resetTransform()
        self.grid.translate((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2])
        self.update()

    def select(self, parts, focus=True):
        """Highlight parts (iterable of part indices; None/empty = none); focus: aim the camera
        at them — parts far apart (a whole-scene file) are dots in the overall view."""
        self.selected = set(parts) if parts else None
        if not self._has:
            return
        self._recolor()
        if focus and self.selected:
            mask = self._mask()
            if mask.any():
                p = self._v[self._perm(self.disp.tris)[mask]].reshape(-1, 3)
                lo, hi = p.min(0), p.max(0)
                c = (lo + hi) / 2
                self.setCameraPosition(pos=QVector3D(*map(float, c)),
                                       distance=max(1.9 * float(np.linalg.norm(hi - lo)), 1e-6))
        self.update()

    def _mask(self):
        """Highlighted triangles, GL order."""
        return np.isin(self._perm(self.disp.tri_part), list(self.selected))

    def _colors(self):
        """(t, 3, 4) corner colours per the shading.py contract, GL order."""
        c = np.ones((len(self.disp.tris), 3, 4), dtype=np.float32)
        if shading.uses_model_colors(self.mode):
            c[..., :3] = self._perm(self.disp.corner_colors)
        if self.selected:
            mask = self._mask()
            c[mask, :, :3] = display.SELECT_COLOR
            c[mask, :, 3] = 0.0     # no map under the highlight
        return c

    def _normals(self):
        """World corner normals of the current smoothing, turned with the model (inverse
        transpose; the placement's uniform scale does not matter)."""
        n = self._perm(self.disp.normals(self.smoothing)).reshape(-1, 3) @ np.linalg.inv(self.rotation)
        return n.reshape(-1, 3, 3).astype(np.float32)

    def set_smoothing(self, mode):
        if mode not in display.SMOOTHING:
            return
        self.smoothing = mode
        if self._md is not None:
            self._md.corner_normals = self._normals()
            self.mesh.meshDataChanged()
            self.update()

    def _recolor(self):
        """Colours only: normals and positions stay cached in the MeshData."""
        if self._md is None:
            return
        self._md.setVertexColors(self._colors(), indexed="faces")
        self.mesh.meshDataChanged()
        self.update()

    def _show_edges(self):
        self.edges.setVisible(self._has and self.wireframe and len(self.disp.edges) > 0)

    @property
    def has_edges(self):
        return self.disp is None or len(self.disp.edges) > 0

    def set_mode(self, mode):
        if mode not in shading.MODES:
            return
        old, self.mode = self.mode, mode
        self.mesh.setShader(self._program(mode))
        self.mesh.setGLOptions(shading.gl_options(mode))
        self.mesh.use_textures = shading.uses_textures(mode)
        self.edges.setData(color=shading.edge_color(mode))
        if shading.uses_model_colors(old) != shading.uses_model_colors(mode):
            self._recolor()
        self.update()

    def set_wireframe(self, on):
        self.wireframe = bool(on)
        self._show_edges()
        self.update()

    def _place_triad(self, lo, hi):
        o = np.array([lo[0], lo[1], lo[2]])
        L = 0.18
        ends = o + np.eye(3) * L
        pos = np.array([o, ends[0], o, ends[1], o, ends[2]], dtype=np.float32)
        col = np.array([(0.94, 0.35, 0.35, 1)] * 2 + [(0.43, 0.82, 0.43, 1)] * 2
                       + [(0.43, 0.59, 1.0, 1)] * 2, dtype=np.float32)
        self.triad.setData(pos=pos, color=col)
        for lab, e in zip(self.labels, ends):
            lab.setData(pos=e + (e - o) * 0.15)


class ViewBar(QWidget):
    """Mode buttons, wireframe toggle and smoothing menu, a dark strip above the 3D view."""
    mode_changed = Signal(str)
    wireframe_changed = Signal(bool)
    smoothing_changed = Signal(str)

    STYLE = ("QWidget#viewbar { background: rgb(36,38,43); }"
             "QToolButton { color: #d8d8d8; background: transparent; border: 1px solid transparent;"
             " border-radius: 3px; padding: 3px 10px; }"
             "QToolButton:hover { background: rgb(58,61,68); }"
             "QToolButton:checked { background: rgb(74,110,170); color: white; }"
             "QToolButton:disabled { color: #777; }")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("viewbar")
        self.setAttribute(Qt.WA_StyledBackground)
        self.setStyleSheet(self.STYLE)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 3, 6, 3)
        lay.setSpacing(2)
        self.group = QButtonGroup(self)
        self.buttons = {}
        for mode in shading.MODES:
            b = QToolButton()
            b.setCheckable(True)
            self.group.addButton(b)
            self.buttons[mode] = b
            lay.addWidget(b)
            b.toggled.connect(lambda on, m=mode: on and self.mode_changed.emit(m))
        sep = QLabel("|")
        sep.setStyleSheet("color: #555; background: transparent; padding: 0 6px")
        lay.addWidget(sep)
        self.wire = QToolButton()
        self.wire.setCheckable(True)
        self.wire.toggled.connect(self.wireframe_changed)
        lay.addWidget(self.wire)
        # smoothing: one drop-down button, the current choice in its text
        self.smooth = QToolButton()
        self.smooth.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self.smooth)
        group = QActionGroup(menu)
        self.smooth_actions = {}
        for mode in display.SMOOTHING:
            a = menu.addAction("")
            a.setCheckable(True)
            group.addAction(a)
            self.smooth_actions[mode] = a
            a.triggered.connect(lambda _=False, m=mode: self._smoothing(m))
        menu.setToolTipsVisible(True)
        self.smooth.setMenu(menu)
        lay.addWidget(self.smooth)
        self.smooth_note = ""
        self.smoothing = display.SMOOTH_FILE
        self.retranslate()
        lay.addStretch()

    def retranslate(self):
        """Texts in the current language (mesh2mesh.i18n). The wireframe tooltip depends on
        the model: the window sets it."""
        for mode, b in self.buttons.items():
            text, tip = shading.MODES[mode]
            b.setText(tr(text))
            b.setToolTip(tr(tip))
        self.wire.setText(tr("Каркас"))
        self.wire.setToolTip(tr("Рёбра многоугольников поверх любого режима"))
        for mode, a in self.smooth_actions.items():
            text, tip = display.smoothing_texts(mode)
            a.setText(text)
            a.setToolTip(tip)
        self._show_smoothing(self.smoothing)

    def set_state(self, mode, wireframe, smoothing=display.SMOOTH_FILE):
        self.buttons.get(mode, self.buttons[shading.MATERIAL]).setChecked(True)
        self.wire.setChecked(wireframe)
        self._show_smoothing(smoothing if smoothing in display.SMOOTHING else display.SMOOTH_FILE)

    def _smoothing(self, mode):
        self._show_smoothing(mode)
        self.smoothing_changed.emit(mode)

    def _show_smoothing(self, mode):
        self.smoothing = mode
        self.smooth_actions[mode].setChecked(True)
        text, tip = display.smoothing_texts(mode)
        self.smooth.setText(tr("Сглаживание: {mode}", mode=text))
        self.smooth.setToolTip(tip + (f"\n{self.smooth_note}" if self.smooth_note else ""))

    def set_smoothing_note(self, note):
        """Extra tooltip line about the loaded model (e.g. meshes without normals)."""
        self.smooth_note = note
        self._show_smoothing(self.smoothing)
