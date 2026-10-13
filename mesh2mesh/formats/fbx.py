"""FBX (Autodesk), versions 7.x: binary (7100..7700, 32/64-bit records) and ASCII are read,
binary 7400 is written (the file skeleton Maya 2026 accepts, D8).

Axes: the file's GlobalSettings (Coord/Up/Front axis + sign) -> the Scene's Z up; the front axis
goes to -Y (as Blender does). Node matrices are conjugated A*M*A^-1, mesh points mapped by A, so
the hierarchy keeps its own transforms. Units: UnitScaleFactor = centimetres per file unit.

Reader: Model tree (FBX transform stack with pivots, offsets, pre/post rotation, rotation
order), Geometry shared between Models kept as instancing, the Geometric* offset baked into the
mesh, n-gons, first UV set, normals, vertex colours, materials (DiffuseColor + opacity, kept as
stored in the file, D9) with the base colour texture (embedded Content or the file beside).
Everything else (animation, skin, blend shapes, NURBS, cameras, lights, other texture maps)
is dropped with a warning.
Writer: Y up (Maya's default), UnitScaleFactor = output unit, Lambert materials, Null models for
empty nodes, a Geometry per mesh shared by its instances. Node matrices are written as
T * R(XYZ) * S; what does not fit (shear, non-orthogonal CAD rotations) is pushed into the
mesh points and the children, exactly.
"""

import base64
import datetime
import math
import os
import re
import struct
import zlib

import numpy as np

from .. import geom, textures
from ..model import NO_MATERIAL, Material, Mesh, Node, Scene, Texture
from ..i18n import tr

BINARY_MAGIC = b"Kaydara FBX Binary  \x00"
C3 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)   # scene -> written file
ROT_ORDERS = ["XYZ", "XZY", "YZX", "YXZ", "ZXY", "ZYX"]
DERIVED_LAYERS = {"LayerElementTangent", "LayerElementBinormal", "LayerElementSmoothing"}


# --- parser: binary and ASCII -> _Node tree --------------------------------------------------

class _Node:
    __slots__ = ("name", "props", "children")

    def __init__(self, name, props=None, children=None):
        self.name, self.props, self.children = name, props or [], children or []

    def find(self, name):
        for c in self.children:
            if c.name == name:
                return c
        return None

    def findall(self, name):
        return [c for c in self.children if c.name == name]

    def value(self, name, default=None):
        c = self.find(name)
        return c.props[0] if c is not None and c.props else default


_SCALAR = {b"Y": "<h", b"C": "<?", b"I": "<i", b"F": "<f", b"D": "<d", b"L": "<q"}
_ARRAY = {b"f": "<f4", b"d": "<f8", b"l": "<i8", b"i": "<i4", b"b": "u1", b"c": "u1"}


def _parse_binary(data):
    version = struct.unpack_from("<I", data, 23)[0]
    head = struct.Struct("<QQQB" if version >= 7500 else "<IIIB")

    def read_prop(pos):
        t = data[pos:pos + 1]
        pos += 1
        if t in _SCALAR:
            fmt = _SCALAR[t]
            return struct.unpack_from(fmt, data, pos)[0], pos + struct.calcsize(fmt)
        if t in _ARRAY:
            n, enc, clen = struct.unpack_from("<III", data, pos)
            pos += 12
            raw = data[pos:pos + clen]
            if enc == 1:
                raw = zlib.decompress(raw)
            elif enc != 0:
                raise ValueError(tr("FBX: неизвестное сжатие массива ({enc})", enc=enc))
            return np.frombuffer(raw, dtype=_ARRAY[t], count=n), pos + clen
        if t in (b"S", b"R"):
            n = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            raw = bytes(data[pos:pos + n])
            return (raw.decode("utf-8", "replace") if t == b"S" else raw), pos + n
        raise ValueError(tr("FBX: неизвестный тип значения {t!r} по смещению {v}", t=t, v=pos - 1))

    def read_node(pos):
        end, nprops, _plen, nlen = head.unpack_from(data, pos)
        if end == 0:
            return None, pos + head.size + nlen
        if end > len(data):
            raise ValueError(tr("FBX: файл обрезан"))
        pos += head.size
        node = _Node(bytes(data[pos:pos + nlen]).decode("ascii", "replace"))
        pos += nlen
        for _ in range(nprops):
            value, pos = read_prop(pos)
            node.props.append(value)
        while pos < end:
            child, pos = read_node(pos)
            if child is None:
                break
            node.children.append(child)
        return node, end

    root = _Node("")
    pos = 27
    while pos + head.size <= len(data):
        node, pos = read_node(pos)
        if node is None:
            break
        root.children.append(node)
    return version, root


_TOKEN = re.compile(r"""
    \s+ | ;[^\n]* |                              # whitespace, comment
    (?P<str>"[^"]*") |
    (?P<key>[A-Za-z_][\w|]*)\s*: |               # node name
    (?P<count>\*\d+) |
    (?P<num>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?) |
    (?P<word>[A-Za-z_][\w|]*) |
    (?P<punct>[{},])
""", re.VERBOSE)


class _ArrayCount(int):
    pass


def _parse_ascii(text):
    toks = []
    pos, n = 0, len(text)
    while pos < n:
        m = _TOKEN.match(text, pos)
        if not m:
            raise ValueError(tr("FBX (ASCII): не разобрать текст у смещения {pos}: {v!r}",
                                pos=pos, v=text[pos:pos + 30]))
        pos = m.end()
        if m.lastgroup:
            toks.append((m.lastgroup, m.group(m.lastgroup)))
    i = 0

    def number(s):
        try:
            return int(s)
        except ValueError:
            return float(s)

    def body(node):
        nonlocal i
        expect = True
        while i < len(toks):
            kind, val = toks[i]
            if expect and kind in ("str", "num", "word", "count"):
                node.props.append(val[1:-1] if kind == "str" else number(val) if kind == "num"
                                  else _ArrayCount(val[1:]) if kind == "count" else val)
                i += 1
                expect = False
            elif kind == "punct" and val == ",":
                i += 1
                expect = True
            else:
                break
        if i < len(toks) and toks[i] == ("punct", "{"):
            i += 1
            children(node, closing=True)
        if node.props and isinstance(node.props[0], _ArrayCount):   # Name: *N { a: ... }
            a = node.find("a")
            vals = a.props if a else []
            node.props = [np.array(vals, dtype=np.float64 if any(isinstance(v, float)
                                                                 for v in vals) else np.int64)]
            node.children = []

    def children(parent, closing):
        nonlocal i
        while i < len(toks):
            kind, val = toks[i]
            if kind == "punct" and val == "}":
                if not closing:
                    raise ValueError(tr("FBX (ASCII): лишняя '}'"))
                i += 1
                return
            if kind != "key":
                raise ValueError(tr("FBX (ASCII): ожидалось имя узла, а не {val!r}", val=val))
            i += 1
            node = _Node(val)
            body(node)
            parent.children.append(node)
        if closing:
            raise ValueError(tr("FBX (ASCII): не хватает '}'"))

    root = _Node("")
    children(root, closing=False)
    hdr = root.find("FBXHeaderExtension")
    return (hdr.value("FBXVersion", 0) if hdr else 0), root


def parse(path):
    """-> (version, root _Node) of a binary or ASCII FBX file."""
    with open(path, "rb") as fh:
        data = fh.read()
    if data.startswith(BINARY_MAGIC):
        return _parse_binary(memoryview(data))
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    if "FBXHeaderExtension" not in text[:4096]:
        raise ValueError(tr("FBX: не FBX-файл (нет заголовка)"))
    return _parse_ascii(text)


# --- reader ----------------------------------------------------------------------------------

def _split_name(s):
    """Binary 'Name\\x00\\x01Class' / ASCII 'Class::Name' -> Name."""
    s = str(s)
    if "\x00\x01" in s:
        return s.split("\x00\x01", 1)[0]
    if "::" in s:
        return s.split("::", 1)[1]
    return s


def _content(video):
    """Embedded file bytes of a Video object (binary: raw 'R'; ASCII: base64 strings) or None."""
    c = video.find("Content")
    if c is None or not c.props:
        return None
    if isinstance(c.props[0], (bytes, bytearray, memoryview)):
        data = bytes(c.props[0])
    else:
        try:
            data = base64.b64decode("".join(str(p) for p in c.props))
        except ValueError:
            return None
    return data or None


def _props70(node, into=None):
    """Properties70 / P rows -> {name: value or tuple of values}."""
    out = dict(into or {})
    p70 = node.find("Properties70") if node is not None else None
    for p in p70.findall("P") if p70 else []:
        if len(p.props) >= 4:
            vals = p.props[4:]
            out[p.props[0]] = vals[0] if len(vals) == 1 else tuple(vals)
    return out


def _templates(root):
    """Definitions -> {ObjectType: default Properties70} (objects store only differences)."""
    out = {}
    defs = root.find("Definitions")
    for ot in defs.findall("ObjectType") if defs else []:
        for pt in ot.findall("PropertyTemplate"):
            if ot.props:
                out[ot.props[0]] = _props70(pt)
    return out


def euler(deg, order="XYZ"):
    """Euler angles in degrees -> 3x3; order 'XYZ' = about X first: Rz @ Ry @ Rx."""
    mats = {}
    for axis, a in zip("XYZ", np.radians(np.asarray(deg, dtype=np.float64))):
        c, s = math.cos(a), math.sin(a)
        m = np.eye(3)
        i, j = {"X": (1, 2), "Y": (2, 0), "Z": (0, 1)}[axis]
        m[i, i], m[i, j], m[j, i], m[j, j] = c, -s, s, c
        mats[axis] = m
    return mats[order[2]] @ mats[order[1]] @ mats[order[0]]


def _m4(r=None, t=None, s=None):
    m = np.eye(4)
    if r is not None:
        m[:3, :3] = r
    if s is not None:
        m[:3, :3] = m[:3, :3] @ np.diag(s)
    if t is not None:
        m[:3, 3] = t
    return m


def _v3(p, key, default=(0.0, 0.0, 0.0)):
    v = p.get(key, default)
    v = tuple(v) if isinstance(v, (tuple, list, np.ndarray)) else (v,)
    return np.asarray(v[:3] if len(v) >= 3 else default, dtype=np.float64)


def local_matrix(p):
    """FBX SDK transform of a Model: T * Roff * Rp * Rpre * R * Rpost^-1 * Rp^-1 * Soff * Sp * S
    * Sp^-1. With RotationActive 0 the order and pre/post rotation are ignored (as Maya does)."""
    active = int(p.get("RotationActive", 0))
    order_i = int(p.get("RotationOrder", 0))
    order = ROT_ORDERS[order_i] if active and order_i < len(ROT_ORDERS) else "XYZ"
    pre = _m4(euler(_v3(p, "PreRotation"))) if active else np.eye(4)
    post = _m4(euler(_v3(p, "PostRotation"))) if active else np.eye(4)
    rp, sp = _v3(p, "RotationPivot"), _v3(p, "ScalingPivot")
    return (_m4(t=_v3(p, "Lcl Translation")) @ _m4(t=_v3(p, "RotationOffset")) @ _m4(t=rp)
            @ pre @ _m4(euler(_v3(p, "Lcl Rotation"), order)) @ np.linalg.inv(post)
            @ _m4(t=-rp) @ _m4(t=_v3(p, "ScalingOffset")) @ _m4(t=sp)
            @ _m4(s=_v3(p, "Lcl Scaling", (1.0, 1.0, 1.0))) @ _m4(t=-sp))


def geometric_matrix(p):
    """Geometric* offset: moves the shape only, not inherited by children."""
    return _m4(euler(_v3(p, "GeometricRotation")), _v3(p, "GeometricTranslation"),
               _v3(p, "GeometricScaling", (1.0, 1.0, 1.0)))


def axis_matrix(gs):
    """File axes -> Scene (right +X, up +Z, front -Y), 3x3 (a mirror for left-handed files)."""
    def axis(key, default_axis, default_sign):
        v = np.zeros(3)
        v[int(gs.get(key + "Axis", default_axis)) % 3] = 1.0 if float(
            gs.get(key + "AxisSign", default_sign)) >= 0 else -1.0
        return v
    src = np.column_stack([axis("Coord", 0, 1), axis("Up", 1, 1), axis("Front", 2, 1)])
    if abs(np.linalg.det(src)) < 0.5:              # two roles on one axis: broken header
        src = np.column_stack([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    dst = np.column_stack([(1, 0, 0), (0, 0, 1), (0, -1, 0)])
    return dst @ src.T                             # src is a signed permutation: inv = T


def reverse_faces(mesh):
    """Flip the winding of every face (corners and per-corner attributes), in place."""
    starts = np.repeat(mesh.face_starts, mesh.face_sizes)
    local = np.arange(len(mesh.corners)) - starts
    order = starts + np.repeat(mesh.face_sizes, mesh.face_sizes) - 1 - local
    mesh.corners = mesh.corners[order]
    for a in ("uv", "normals", "colors"):
        if getattr(mesh, a) is not None:
            setattr(mesh, a, getattr(mesh, a)[order])


class _Reader:
    def __init__(self, path):
        self.path, self.folder = path, os.path.dirname(os.path.abspath(path))
        self.version, self.root = parse(path)
        if self.version and self.version < 7000:
            raise ValueError(tr("FBX {v:.1f}: поддерживаются FBX 7.x (2011+) — пересохраните "
                                "файл в новой версии", v=self.version / 1000))
        self.scene = Scene()
        self.tmpl = _templates(self.root)
        gs = _props70(self.root.find("GlobalSettings"))
        self.scene.unit_m = float(gs.get("UnitScaleFactor", 1.0)) / 100.0
        self.A = axis_matrix(gs)
        self.A4 = _m4(self.A)
        self.mesh_of, self.mat_of = {}, {}

    def warn(self, msg):
        self.scene.warn(msg)

    def run(self):
        objs = self.root.find("Objects")
        if objs is None:
            raise ValueError(tr("FBX: нет раздела Objects"))
        self.obj, kind, order = {}, {}, {}
        for i, n in enumerate(objs.children):
            if n.props:
                self.obj[n.props[0]] = n
                kind[n.props[0]] = n.name
                order[n.props[0]] = i
        parent, self.geos, self.mats, self.attr = {}, {}, {}, {}
        self.mat_tex, self.tex_video = {}, {}
        deformed = anim = False
        conns = self.root.find("Connections")
        for c in conns.findall("C") if conns else []:
            if len(c.props) < 3:
                continue
            src, dst = c.props[1], c.props[2]
            ks, kd = kind.get(src), kind.get(dst)
            if ks == "Model" and kd == "Model":
                parent[src] = dst
            elif ks == "Geometry" and kd == "Model":
                self.geos.setdefault(dst, []).append(src)
            elif ks == "Material" and kd == "Model":
                self.mats.setdefault(dst, []).append(src)
            elif ks == "NodeAttribute" and kd == "Model":
                self.attr.setdefault(dst, src)
            elif ks in ("Texture", "LayeredTexture") and kd == "Material":
                prop = str(c.props[3]) if len(c.props) > 3 else ""
                self.mat_tex.setdefault(dst, []).append((src, prop))
            elif ks == "Video" and kd == "Texture":
                self.tex_video.setdefault(dst, []).append(src)
            elif ks == "Deformer" and kd == "Geometry":
                deformed = True
            elif ks == "AnimationCurveNode" and kd == "Model":
                anim = True
        # embedded images by file name: exporters (Mixamo) store a file's bytes once and leave
        # the other Video objects of the same file without Content
        self.embedded = {}
        for u in self.obj:
            if kind[u] == "Video":
                data = _content(self.obj[u])
                if data:
                    for k in ("RelativeFilename", "Filename"):
                        ref = self.obj[u].value(k)
                        if ref:
                            self.embedded.setdefault(textures.base_name(ref).lower(), data)
        self.tex_of, self.tex_by_data = {}, {}
        if deformed:
            self.warn(tr("FBX: скиннинг и blend shapes не переносятся (взята исходная форма)"))
        if anim or any(k == "AnimationCurve" and len(np.atleast_1d(n.value("KeyTime", []))) > 1
                       for k, n in ((kind[u], self.obj[u]) for u in self.obj)):
            self.warn(tr("FBX: анимация не переносится (взята поза из файла)"))
        models = sorted((u for u in self.obj if kind[u] == "Model"), key=order.get)
        kids = {}
        roots = []
        for u in models:
            p = parent.get(u)
            (kids.setdefault(p, []) if p in self.obj and kind[p] == "Model" else roots).append(u)
        self.kids = kids
        self.mprops = {u: _props70(self.obj[u], self.tmpl.get("Model")) for u in models}
        self.scene.nodes = [self.node(u) for u in roots]
        return self.scene

    def node(self, uid, stack=()):
        if uid in stack:
            raise ValueError(tr("FBX: циклическая иерархия объектов"))
        n = self.obj[uid]
        p = self.mprops[uid]
        name = _split_name(n.props[1]) if len(n.props) > 1 else ""
        mtype = n.props[2] if len(n.props) > 2 else ""
        it = int(p.get("InheritType", 1))
        if it != 1 and stack:
            s = np.linalg.norm(local_matrix(self.mprops[stack[-1]])[:3, :3], axis=0)
            if (it == 0 and np.ptp(s) > 1e-9 * s.max()) or (it == 2 and np.abs(s - 1).max() > 1e-9):
                self.warn(tr("FBX: «{name}»: наследование масштаба InheritType {it} заменено "
                             "обычным (RSrs) — форма под неравномерным масштабом родителя может "
                             "отличаться", name=name, it=it))
        node = Node(name or "node", self.A4 @ local_matrix(p) @ self.A4.T)
        geos = [g for g in self.geos.get(uid, []) if g in self.obj]
        meshes = [g for g in geos if self.obj[g].props[2:3] == ["Mesh"]]
        other = {str(self.obj[g].props[2]) for g in geos if g not in meshes and
                 len(self.obj[g].props) > 2 and self.obj[g].props[2] != "Shape"}
        if other:
            self.warn(tr("FBX: не меши не переносятся: {v}", v=', '.join(sorted(other))))
        elif not geos and mtype in ("Camera", "Light"):
            self.warn(tr("FBX: {v} не переносятся (оставлены пустыми узлами)",
                         v=tr('камеры') if mtype == 'Camera' else tr('источники света')))
        if len(meshes) > 1:
            self.warn(tr("FBX: «{name}»: у объекта несколько геометрий — взята первая", name=name))
        if meshes:
            node.mesh = self.mesh(meshes[0], uid, name)
        node.children = [self.node(c, stack + (uid,)) for c in self.kids.get(uid, [])]
        return node

    def material(self, uid):
        if uid not in self.mat_of:
            n = self.obj[uid]
            p = _props70(n, self.tmpl.get("Material"))
            col = p.get("DiffuseColor", p.get("Diffuse"))
            rgb = tuple(float(x) for x in col[:3]) if isinstance(col, tuple) and len(col) >= 3 \
                else None
            if "Opacity" in p:
                alpha = float(p["Opacity"])
            else:
                tc = p.get("TransparentColor", (0.0, 0.0, 0.0))
                tc = tc if isinstance(tc, tuple) and len(tc) >= 3 else (0.0, 0.0, 0.0)
                alpha = 1.0 - float(p.get("TransparencyFactor", 0.0)) * float(np.mean(tc[:3]))
            alpha = min(max(alpha, 0.0), 1.0)
            name = _split_name(n.props[1]) if len(n.props) > 1 else f"material_{uid}"
            tex = None
            for tid, prop in self.mat_tex.get(uid, []):
                if tid not in self.obj:
                    continue
                if prop not in ("DiffuseColor", "Diffuse", "Maya|baseColor", "Maya|TEX_color_map",
                                "3dsMax|Parameters|base_color_map"):
                    self.warn(tr("FBX: переносится только текстура цвета (карты нормалей, "
                                 "блеска, прозрачности и прочие — нет)"))
                elif self.obj[tid].name == "LayeredTexture":
                    self.warn(tr("FBX: «{name}»: многослойная текстура не переносится", name=name))
                elif tex is None:
                    tex = self.texture(tid)
            self.mat_of[uid] = len(self.scene.materials)
            self.scene.materials.append(Material(
                name, (*rgb, alpha) if rgb is not None else None, tex))
        return self.mat_of[uid]

    def texture(self, tid):
        """Texture object -> model Texture (embedded Content, or the file it names), or None
        with a warning. Shared by every material that uses it."""
        if tid not in self.tex_of:
            t = self.obj[tid]
            tname = _split_name(t.props[1]) if len(t.props) > 1 else "texture"
            data, refs = None, []
            for v in self.tex_video.get(tid, []):
                if v in self.obj:
                    data = data or _content(self.obj[v])
                    refs += [self.obj[v].value(k) for k in ("RelativeFilename", "Filename")]
            refs = [str(r) for r in refs + [t.value("RelativeFilename"), t.value("FileName")] if r]
            for r in refs:
                data = data or self.embedded.get(textures.base_name(r).lower())
            name = textures.base_name(refs[0]) if refs else tname
            tex = Texture(name, data) if data else None
            stem = os.path.splitext(os.path.basename(self.path))[0]
            for r in refs if tex is None else ():
                tex = textures.load_file(r, self.folder, [os.path.join(self.folder, stem + ".fbm")])
                if tex is not None:
                    break
            if tex is None:
                self.warn(tr("FBX: текстура «{name}» не вшита и не найдена рядом с файлом — "
                             "отброшена", name=name))
            else:
                if "." not in tex.name:
                    k = textures.kind(tex.data, tex.name)
                    tex.name += textures.KINDS[k][0] if k else ""
                # one image under several Texture objects (Mixamo) -> one Texture
                tex = self.tex_by_data.setdefault((tex.name, tex.data), tex)
            p = _props70(t)
            moved = (any(abs(float(x)) > 1e-9 for x in np.atleast_1d(p.get("Translation", 0)))
                     or any(abs(float(x)) > 1e-9 for x in np.atleast_1d(p.get("Rotation", 0)))
                     or any(abs(float(x) - 1) > 1e-9 for x in np.atleast_1d(p.get("Scaling", 1))))
            if tex is not None and moved:
                self.warn(tr("FBX: сдвиг/поворот/масштаб UV текстуры «{name}» не переносится",
                             name=name))
            self.tex_of[tid] = tex
        return self.tex_of[tid]

    def layer(self, geo, elem, values, index, width, name):
        """LayerElement -> (c, width) per corner, or None (with a warning) if unusable."""
        mapping = str(elem.value("MappingInformationType", "ByPolygonVertex"))
        ref = str(elem.value("ReferenceInformationType", "Direct"))
        raw = elem.value(values)
        if raw is None:
            return None
        vals = np.asarray(raw, dtype=np.float64)
        if len(vals) % width:
            self.warn(tr("FBX: «{name}»: слой {name2} повреждён — отброшен",
                         name=name, name2=elem.name))
            return None
        vals = vals.reshape(-1, width)
        corners, sizes = geo
        if mapping == "ByPolygonVertex":
            key = np.arange(len(corners))
        elif mapping in ("ByVertice", "ByVertex"):
            key = corners
        elif mapping == "ByPolygon":
            key = np.repeat(np.arange(len(sizes)), sizes)
        elif mapping == "AllSame":
            key = np.zeros(len(corners), dtype=np.int64)
        else:
            self.warn(tr("FBX: «{name}»: {name2} с привязкой {mapping} не переносится",
                         name=name, name2=elem.name, mapping=mapping))
            return None
        if ref in ("IndexToDirect", "Index"):
            idx = elem.value(index)
            if idx is not None:
                idx = np.asarray(idx, dtype=np.int64)
                if len(key) and key.max() >= len(idx):
                    self.warn(tr("FBX: «{name}»: слой {name2} короче меша — отброшен",
                                 name=name, name2=elem.name))
                    return None
                key = idx[key]
        if len(key) and (key.min() < 0 or key.max() >= len(vals)):
            self.warn(tr("FBX: «{name}»: слой {name2} ссылается за пределы — отброшен",
                         name=name, name2=elem.name))
            return None
        return vals[key]

    def mesh(self, gid, model_uid, model_name):
        p = self.mprops[model_uid]
        g = geometric_matrix(p)
        mats = tuple(m for m in self.mats.get(model_uid, []) if m in self.obj)
        key = (gid, mats, g.tobytes())
        if key in self.mesh_of:
            return self.mesh_of[key]
        n = self.obj[gid]
        name = (_split_name(n.props[1]) if len(n.props) > 1 else "") or model_name or "mesh"
        v = np.asarray(n.value("Vertices", np.zeros(0)), dtype=np.float64)
        pvi = np.asarray(n.value("PolygonVertexIndex", np.zeros(0)), dtype=np.int64)
        if len(v) % 3:
            raise ValueError(tr("FBX: «{name}»: число координат вершин не кратно 3", name=name))
        pos = v.reshape(-1, 3)
        if len(pvi) and pvi[-1] >= 0:
            pvi = np.r_[pvi[:-1], ~pvi[-1]]                  # unterminated last polygon
        ends = np.flatnonzero(pvi < 0)
        sizes = np.diff(np.r_[-1, ends])
        corners = np.where(pvi < 0, ~pvi, pvi)
        if len(corners) and (corners.max() >= len(pos)):
            raise ValueError(tr("FBX: «{name}»: индекс вершины вне диапазона", name=name))
        geo = (corners, sizes)
        attrs = {}
        for tag, values, index, width, what in (
                ("LayerElementNormal", "Normals", "NormalsIndex", 3, tr("нормалей")),
                ("LayerElementUV", "UV", "UVIndex", 2, tr("UV-каналов")),
                ("LayerElementColor", "Colors", "ColorIndex", 4, tr("наборов цветов вершин"))):
            elems = n.findall(tag)
            if len(elems) > 1:
                self.warn(tr("FBX: «{name}»: {n_elems} {what} — переносится только первый",
                             name=name, n_elems=len(elems), what=what))
            if elems:
                attrs[tag] = self.layer(geo, elems[0], values, index, width, name)
        nrm, uv, col = (attrs.get(k) for k in ("LayerElementNormal", "LayerElementUV",
                                                 "LayerElementColor"))
        known = {"LayerElementNormal", "LayerElementUV", "LayerElementColor",
                 "LayerElementMaterial"} | DERIVED_LAYERS
        extra = sorted({c.name for c in n.children if c.name.startswith("LayerElement")} - known)
        if extra:
            self.warn(tr("FBX: слои не переносятся: {v}", v=', '.join(extra)))
        sm = n.find("LayerElementSmoothing")
        if nrm is None and sm is not None and np.any(np.asarray(sm.value("Smoothing", [0]))):
            # all-zero groups = faceted, which is what a mesh without normals means here
            self.warn(tr("FBX: «{name}»: группы сглаживания без нормалей не переносятся", name=name))
        fm = None
        if mats:
            local = np.zeros(len(sizes), dtype=np.int64)
            lem = n.find("LayerElementMaterial")
            if lem is not None:
                arr = np.asarray(lem.value("Materials", np.zeros(1, int)), dtype=np.int64)
                mapping = str(lem.value("MappingInformationType", "AllSame"))
                if mapping == "ByPolygon" and len(arr) >= len(sizes):
                    local = arr[:len(sizes)]
                elif len(arr):
                    if mapping != "AllSame":
                        self.warn(tr("FBX: «{name}»: материалы с привязкой {mapping} — взят "
                                     "первый на весь меш", name=name, mapping=mapping))
                    local[:] = arr[0]
            table = np.array([self.material(m) for m in mats] + [-1], dtype=np.int64)
            fm = table[np.where((local >= 0) & (local < len(mats)), local, len(mats))]
        keep = sizes >= 3
        if not keep.all():
            self.warn(tr("FBX: «{name}»: рёбра и точки без граней отброшены", name=name))
            cmask = np.repeat(keep, sizes)
            corners, sizes = corners[cmask], sizes[keep]
            nrm, uv, col = (a[cmask] if a is not None else None for a in (nrm, uv, col))
            fm = fm[keep] if fm is not None else None
        if col is not None:
            col = np.clip(col, 0.0, 1.0)
        mesh = Mesh(name, pos, sizes.astype(np.int64), corners.astype(np.int64),
                    face_material=fm if fm is not None and np.any(fm >= 0) else None,
                    uv=uv, normals=nrm, colors=col)
        if not np.allclose(g, np.eye(4)):
            mesh.positions = geom.apply(g, mesh.positions)
            if mesh.normals is not None:
                mesh.normals = geom.apply_normals(g, mesh.normals)
        mesh.positions = mesh.positions @ self.A.T
        if mesh.normals is not None:
            mesh.normals = mesh.normals @ self.A.T
        # a mirror baked into the points (geometric offset, left-handed file) keeps the faces
        # outward only with the winding reversed; mirrors left in node matrices stay as they are
        if np.linalg.det(g[:3, :3]) * np.linalg.det(self.A) < 0:
            reverse_faces(mesh)
        self.mesh_of[key] = len(self.scene.meshes)
        self.scene.meshes.append(mesh)
        return self.mesh_of[key]


def read(path):
    return _Reader(path).run()


# --- serializer ------------------------------------------------------------------------------

class Int64(int):
    """Written as FBX 'L' (uids, KTime) instead of 'I'."""


class Float32(float):
    """Written as FBX 'F'."""


def _pack(v):
    if isinstance(v, bool):
        return b"C" + struct.pack("<B", v)
    if isinstance(v, Int64):
        return b"L" + struct.pack("<q", v)
    if isinstance(v, (int, np.integer)):
        v = int(v)
        return b"I" + struct.pack("<i", v) if -2 ** 31 <= v < 2 ** 31 else \
            b"L" + struct.pack("<q", v)
    if isinstance(v, Float32):
        return b"F" + struct.pack("<f", v)
    if isinstance(v, (float, np.floating)):
        return b"D" + struct.pack("<d", float(v))
    if isinstance(v, str):
        enc = v.encode("utf-8")
        return b"S" + struct.pack("<I", len(enc)) + enc
    if isinstance(v, bytes):
        return b"R" + struct.pack("<I", len(v)) + v
    if isinstance(v, np.ndarray):
        code, dt = (b"d", "<f8") if v.dtype.kind == "f" else (b"i", "<i4")
        raw = np.ascontiguousarray(v.ravel(), dtype=dt).tobytes()
        enc = 0
        if len(raw) > 128:
            raw, enc = zlib.compress(raw), 1
        return code + struct.pack("<III", v.size, enc, len(raw)) + raw
    raise TypeError(tr("FBX: не записать значение типа {v}", v=type(v)))


class Rec:
    """A node record to write (FBX 7400: 32-bit offsets, 13-byte null record)."""
    __slots__ = ("name", "props", "nprops", "children", "_size")

    def __init__(self, name, props=(), children=()):
        self.name = name.encode("utf-8")
        self.props = b"".join(_pack(p) for p in props)
        self.nprops = len(props)
        self.children = list(children)
        self._size = None

    def size(self):
        if self._size is None:
            self._size = (13 + len(self.name) + len(self.props)
                          + sum(c.size() for c in self.children)
                          + (13 if self.children else 0))
        return self._size

    def emit(self, out):
        end = len(out) + self.size()
        if end >= 2 ** 32:
            raise ValueError(tr("FBX 7400: файл больше 4 ГБ не записать"))
        out += struct.pack("<IIIB", end, self.nprops, len(self.props), len(self.name))
        out += self.name + self.props
        for c in self.children:
            c.emit(out)
        if self.children:
            out += b"\0" * 13


FBX_VERSION = 7400
FOOTER_MAGIC = bytes.fromhex("f85a8c6adef5d97eece90ce3758f290b")
# FileId, CreationTime and the footer id depend on each other (the FBX SDK derives the footer);
# one consistent triple written by Maya is reused, the real time goes to CreationTimeStamp.
FILE_ID = bytes.fromhex("29ba22e3be2ac0cfb1ceb526a328fcf6")
FILE_CREATION_TIME = "2019-11-20 16:37:18:382"
FOOTER_ID = bytes.fromhex("fabcaa00d9c9d569bc74f8871aff2a75")
KTIME_SECOND = 46186158000

P_INT32 = {"int", "Integer", "enum", "bool", "Bool", "Visibility Inheritance"}
P_INT64 = {"KTime", "ULongLong"}
P_STRING = {"KString", "DateTime", "Url"}
P_NONE = {"object", "Compound"}


def P(name, ptype, label, flags, *values):
    if ptype in P_NONE:
        typed = []
    elif ptype in P_INT32:
        typed = [int(v) for v in values]
    elif ptype in P_INT64:
        typed = [Int64(v) for v in values]
    elif ptype in P_STRING:
        typed = [str(v) for v in values]
    else:
        typed = [float(v) for v in values]
    return Rec("P", [name, ptype, label, flags, *typed])


def P70(rows):
    return Rec("Properties70", [], [P(*r) for r in rows])


def serialize(top):
    out = bytearray(BINARY_MAGIC + b"\x1a\x00" + struct.pack("<I", FBX_VERSION))
    for rec in top:
        rec.emit(out)
    out += b"\0" * 13 + FOOTER_ID
    out += b"\0" * ((16 - len(out) % 16) % 16 + 4)
    out += struct.pack("<I", FBX_VERSION) + b"\0" * 120 + FOOTER_MAGIC
    return bytes(out)


# --- writer ----------------------------------------------------------------------------------

# property templates: defaults of every property an object does not store (Maya 2018 values)
MODEL_TEMPLATE = [
    ("QuaternionInterpolate", "enum", "", "", 0),
    ("RotationOffset", "Vector3D", "Vector", "", 0, 0, 0),
    ("RotationPivot", "Vector3D", "Vector", "", 0, 0, 0),
    ("ScalingOffset", "Vector3D", "Vector", "", 0, 0, 0),
    ("ScalingPivot", "Vector3D", "Vector", "", 0, 0, 0),
    ("TranslationActive", "bool", "", "", 0),
    ("RotationOrder", "enum", "", "", 0),
    ("RotationSpaceForLimitOnly", "bool", "", "", 0),
    ("AxisLen", "double", "Number", "", 10),
    ("PreRotation", "Vector3D", "Vector", "", 0, 0, 0),
    ("PostRotation", "Vector3D", "Vector", "", 0, 0, 0),
    ("RotationActive", "bool", "", "", 0),
    ("InheritType", "enum", "", "", 0),
    ("ScalingActive", "bool", "", "", 0),
    ("ScalingMax", "Vector3D", "Vector", "", 1, 1, 1),
    ("GeometricTranslation", "Vector3D", "Vector", "", 0, 0, 0),
    ("GeometricRotation", "Vector3D", "Vector", "", 0, 0, 0),
    ("GeometricScaling", "Vector3D", "Vector", "", 1, 1, 1),
    ("LookAtProperty", "object", "", ""),
    ("UpVectorProperty", "object", "", ""),
    ("Show", "bool", "", "", 1),
    ("NegativePercentShapeSupport", "bool", "", "", 1),
    ("DefaultAttributeIndex", "int", "Integer", "", -1),
    ("Freeze", "bool", "", "", 0),
    ("LODBox", "bool", "", "", 0),
    ("Lcl Translation", "Lcl Translation", "", "A", 0, 0, 0),
    ("Lcl Rotation", "Lcl Rotation", "", "A", 0, 0, 0),
    ("Lcl Scaling", "Lcl Scaling", "", "A", 1, 1, 1),
    ("Visibility", "Visibility", "", "A", 1),
    ("Visibility Inheritance", "Visibility Inheritance", "", "", 1),
]
GEOMETRY_TEMPLATE = [
    ("Color", "ColorRGB", "Color", "", 0.8, 0.8, 0.8),
    ("BBoxMin", "Vector3D", "Vector", "", 0, 0, 0),
    ("BBoxMax", "Vector3D", "Vector", "", 0, 0, 0),
    ("Primary Visibility", "bool", "", "", 1),
    ("Casts Shadows", "bool", "", "", 1),
    ("Receive Shadows", "bool", "", "", 1),
]
NULL_TEMPLATE = [
    ("Color", "ColorRGB", "Color", "", 0.8, 0.8, 0.8),
    ("Size", "double", "Number", "", 100),
    ("Look", "enum", "", "", 1),
]
MATERIAL_TEMPLATE = [
    ("ShadingModel", "KString", "", "", "Lambert"),
    ("MultiLayer", "bool", "", "", 0),
    ("EmissiveColor", "Color", "", "A", 0, 0, 0),
    ("EmissiveFactor", "Number", "", "A", 1),
    ("AmbientColor", "Color", "", "A", 0.2, 0.2, 0.2),
    ("AmbientFactor", "Number", "", "A", 1),
    ("DiffuseColor", "Color", "", "A", 0.8, 0.8, 0.8),
    ("DiffuseFactor", "Number", "", "A", 1),
    ("Bump", "Vector3D", "Vector", "", 0, 0, 0),
    ("NormalMap", "Vector3D", "Vector", "", 0, 0, 0),
    ("BumpFactor", "double", "Number", "", 1),
    ("TransparentColor", "Color", "", "A", 0, 0, 0),
    ("TransparencyFactor", "Number", "", "A", 0),
    ("DisplacementColor", "ColorRGB", "Color", "", 0, 0, 0),
    ("DisplacementFactor", "double", "Number", "", 1),
    ("VectorDisplacementColor", "ColorRGB", "Color", "", 0, 0, 0),
    ("VectorDisplacementFactor", "double", "Number", "", 1),
]
TEXTURE_TEMPLATE = [
    ("TextureTypeUse", "enum", "", "", 0),
    ("Texture alpha", "Number", "", "A", 1),
    ("CurrentMappingType", "enum", "", "", 0),
    ("WrapModeU", "enum", "", "", 0),
    ("WrapModeV", "enum", "", "", 0),
    ("UVSwap", "bool", "", "", 0),
    ("PremultiplyAlpha", "bool", "", "", 1),
    ("Translation", "Vector", "", "A", 0, 0, 0),
    ("Rotation", "Vector", "", "A", 0, 0, 0),
    ("Scaling", "Vector", "", "A", 1, 1, 1),
    ("TextureRotationPivot", "Vector3D", "Vector", "", 0, 0, 0),
    ("TextureScalingPivot", "Vector3D", "Vector", "", 0, 0, 0),
    ("CurrentTextureBlendMode", "enum", "", "", 1),
    ("UVSet", "KString", "", "", "default"),
    ("UseMaterial", "bool", "", "", 0),
    ("UseMipMap", "bool", "", "", 0),
]
VIDEO_TEMPLATE = [
    ("ImageSequence", "bool", "", "", 0),
    ("ImageSequenceOffset", "int", "Integer", "", 0),
    ("FrameRate", "double", "Number", "", 0),
    ("LastFrame", "int", "Integer", "", 0),
    ("Width", "int", "Integer", "", 0),
    ("Height", "int", "Integer", "", 0),
    ("Path", "KString", "XRefUrl", "", ""),
    ("StartFrame", "int", "Integer", "", 0),
    ("StopFrame", "int", "Integer", "", 0),
    ("PlaySpeed", "double", "Number", "", 0),
    ("Offset", "KTime", "Time", "", 0),
    ("InterlaceMode", "enum", "", "", 0),
    ("FreeRunning", "bool", "", "", 0),
    ("Loop", "bool", "", "", 0),
    ("AccessMode", "enum", "", "", 0),
]
ANIM_STACK_TEMPLATE = [
    ("Description", "KString", "", "", ""),
    ("LocalStart", "KTime", "Time", "", 0),
    ("LocalStop", "KTime", "Time", "", 0),
    ("ReferenceStart", "KTime", "Time", "", 0),
    ("ReferenceStop", "KTime", "Time", "", 0),
]
ANIM_LAYER_TEMPLATE = [
    ("Weight", "Number", "", "A", 100),
    ("Mute", "bool", "", "", 0),
    ("Solo", "bool", "", "", 0),
    ("Lock", "bool", "", "", 0),
    ("Color", "ColorRGB", "Color", "", 0.8, 0.8, 0.8),
    ("BlendMode", "enum", "", "", 0),
    ("RotationAccumulationMode", "enum", "", "", 0),
    ("ScaleAccumulationMode", "enum", "", "", 0),
    ("BlendModeBypass", "ULongLong", "", "", 0),
]


def euler_xyz(r):
    """Proper rotation 3x3 -> degrees (x, y, z) with r = Rz @ Ry @ Rx."""
    sy = -r[2, 0]
    if abs(sy) < 1 - 1e-12:
        x = math.atan2(r[2, 1], r[2, 2])
        y = math.asin(max(-1.0, min(1.0, sy)))
        z = math.atan2(r[1, 0], r[0, 0])
    else:                                          # gimbal lock: put everything into x
        y = math.copysign(math.pi / 2, sy)
        x = math.atan2(-r[1, 2], r[1, 1])
        z = 0.0
    return np.degrees([x, y, z])


def decompose(m3):
    """3x3 -> (rotation 3x3, scale (3,), residual 3x3 or None) with m3 = R @ diag(S) @ K.
    K is None when R @ diag(S) reproduces m3 to round-off."""
    s = np.linalg.norm(m3, axis=0)
    if np.any(s < 1e-300):
        return np.eye(3), s, None                  # degenerate (zero scale): nothing to keep
    r = m3 / s
    if np.linalg.det(r) < 0:
        s[0], r[:, 0] = -s[0], -r[:, 0]
    u, _, vt = np.linalg.svd(r)
    ro = u @ vt
    if np.linalg.det(ro) < 0:                      # cannot happen for det(r) > 0; be safe
        u[:, -1] = -u[:, -1]
        ro = u @ vt
    tr = ro @ np.diag(s)
    back = euler(euler_xyz(ro)) @ np.diag(s)
    if np.abs(back - m3).max() <= 1e-12 * max(np.abs(m3).max(), 1e-300):
        return ro, s, None
    return ro, s, np.linalg.solve(tr, m3)


class _Writer:
    def __init__(self, scene, scale):
        self.scene, self.scale = scene, scale
        self.uid = 0
        self.objects, self.conns, self.counts = [], [], {}
        self.geo_of, self.mat_uid, self.tex_uid = {}, {}, {}
        self.tex_names = textures.file_names(textures.used_textures(scene))
        self.losses = []

    def loss(self, msg):
        if msg not in self.losses:
            self.losses.append(msg)

    def next_uid(self):
        self.uid += 1
        return Int64(2_000_000_000_000 + self.uid)

    def add(self, type_name, rec):
        self.objects.append(rec)
        self.counts[type_name] = self.counts.get(type_name, 0) + 1

    def connect(self, child, parent):
        self.conns.append(Rec("C", ["OO", child, parent]))

    @staticmethod
    def safe(name, fallback):
        return (str(name).replace("\x00", "").replace("\x01", "") or fallback)

    # -- materials --------------------------------------------------------------------------

    def material(self, mi):
        if mi not in self.mat_uid:
            if mi < 0:
                m = Material(NO_MATERIAL, None)
            else:
                m = self.scene.materials[mi]
            rows = [("ShadingModel", "KString", "", "", "lambert")]
            color = m.color if m.color is not None else (0.8, 0.8, 0.8, 1.0)
            rows += [("DiffuseColor", "Color", "", "A", *color[:3]),
                     ("DiffuseFactor", "Number", "", "A", 1.0),
                     ("Diffuse", "Vector3D", "Vector", "", *color[:3])]
            alpha = float(color[3]) if len(color) > 3 else 1.0
            if alpha < 1.0:
                rows += [("TransparentColor", "Color", "", "A", 1.0, 1.0, 1.0),
                         ("TransparencyFactor", "Number", "", "A", 1.0 - alpha)]
            rows.append(("Opacity", "double", "Number", "", alpha))
            uid = self.next_uid()
            self.add("Material", Rec("Material", [uid, self.safe(m.name, "material") +
                                                  "\x00\x01Material", ""], [
                Rec("Version", [102]), Rec("ShadingModel", ["lambert"]), Rec("MultiLayer", [0]),
                P70(rows)]))
            if m.texture is not None:
                self.conns.append(Rec("C", ["OP", self.texture(m.texture), uid, "DiffuseColor"]))
            self.mat_uid[mi] = uid
        return self.mat_uid[mi]

    def texture(self, tex):
        """Texture + Video (file bytes embedded as Content, as Maya's "Embed Media") -> uid of
        the Texture; one pair per image, shared by materials."""
        if id(tex) not in self.tex_uid:
            fname = self.tex_names[id(tex)]
            stem = self.safe(os.path.splitext(fname)[0], "texture")
            vid, tid = self.next_uid(), self.next_uid()
            self.add("Video", Rec("Video", [vid, stem + "\x00\x01Video", "Clip"], [
                Rec("Type", ["Clip"]),
                P70([("Path", "KString", "XRefUrl", "", fname)]),
                Rec("UseMipMap", [0]), Rec("Filename", [fname]), Rec("RelativeFilename", [fname]),
                Rec("Content", [tex.data])]))
            self.add("Texture", Rec("Texture", [tid, stem + "\x00\x01Texture", ""], [
                Rec("Type", ["TextureVideoClip"]), Rec("Version", [202]),
                Rec("TextureName", [stem + "\x00\x01Texture"]),
                P70([("CurrentTextureBlendMode", "enum", "", "", 0),
                     ("UVSet", "KString", "", "", "map1"),
                     ("UseMaterial", "bool", "", "", 1)]),
                Rec("Media", [stem + "\x00\x01Video"]),
                Rec("FileName", [fname]), Rec("RelativeFilename", [fname]),
                Rec("ModelUVTranslation", [0.0, 0.0]), Rec("ModelUVScaling", [1.0, 1.0]),
                Rec("Texture_Alpha_Source", ["None"]), Rec("Cropping", [0, 0, 0, 0])]))
            self.connect(vid, tid)
            self.tex_uid[id(tex)] = tid
        return self.tex_uid[id(tex)]

    # -- geometry ---------------------------------------------------------------------------

    def geometry(self, mi, residual):
        """Geometry record for mesh `mi` (points through `residual` 3x3 if any), shared by every
        instance with the same residual -> (uid, material list)."""
        key = (mi, None if residual is None else residual.tobytes())
        if key in self.geo_of:
            return self.geo_of[key]
        m = self.scene.meshes[mi]
        mesh = Mesh(m.name, m.positions, m.face_sizes, m.corners, m.face_material, m.uv,
                    m.normals, m.colors)
        if residual is not None:
            mesh.positions = m.positions @ residual.T
            if m.normals is not None:
                mesh.normals = geom.apply_normals(_m4(residual), m.normals)
            if np.linalg.det(residual) < 0:        # mirror pushed into the points
                reverse_faces(mesh)
        pos, corners, sizes = mesh.positions, mesh.corners, mesh.face_sizes
        pvi = corners.astype(np.int64).copy()
        ends = mesh.face_starts + sizes - 1
        pvi[ends] = ~pvi[ends]
        # Edges: the polygon-vertex each unique edge starts at, in first-seen order (Maya)
        nxt = np.arange(1, len(corners) + 1)
        nxt[ends] = mesh.face_starts
        pairs = np.sort(np.c_[corners, corners[nxt]], axis=1)
        _, first = geom.unique_rows(pairs, return_index=True)
        recs = [Rec("Vertices", [(pos * self.scale @ C3.T).ravel()]),
                Rec("PolygonVertexIndex", [pvi.astype(np.int32)]),
                Rec("Edges", [np.sort(first).astype(np.int32)]),
                Rec("GeometryVersion", [124])]
        layer0 = []
        if mesh.normals is not None:
            recs.append(Rec("LayerElementNormal", [0], [
                Rec("Version", [101]), Rec("Name", [""]),
                Rec("MappingInformationType", ["ByPolygonVertex"]),
                Rec("ReferenceInformationType", ["Direct"]),
                Rec("Normals", [(mesh.normals @ C3.T).ravel()])]))
            layer0.append("LayerElementNormal")
        if mesh.colors is not None and not geom.colors_redundant(mesh, self.scene.materials):
            uniq, inv = geom.unique_rows(np.clip(mesh.colors, 0, 1), return_inverse=True)
            recs.append(Rec("LayerElementColor", [0], [
                Rec("Version", [101]), Rec("Name", ["colorSet1"]),
                Rec("MappingInformationType", ["ByPolygonVertex"]),
                Rec("ReferenceInformationType", ["IndexToDirect"]),
                Rec("Colors", [uniq.ravel()]), Rec("ColorIndex", [inv.ravel().astype(np.int32)])]))
            layer0.append("LayerElementColor")
        if mesh.uv is not None:
            uniq, inv = geom.unique_rows(mesh.uv, return_inverse=True)
            recs.append(Rec("LayerElementUV", [0], [
                Rec("Version", [101]), Rec("Name", ["map1"]),
                Rec("MappingInformationType", ["ByPolygonVertex"]),
                Rec("ReferenceInformationType", ["IndexToDirect"]),
                Rec("UV", [uniq.ravel()]), Rec("UVIndex", [inv.ravel().astype(np.int32)])]))
            layer0.append("LayerElementUV")
        if mesh.normals is None:
            # no normals: faceted (no smoothing group), the way CAD / print meshes look
            recs.append(Rec("LayerElementSmoothing", [0], [
                Rec("Version", [102]), Rec("Name", [""]),
                Rec("MappingInformationType", ["ByPolygon"]),
                Rec("ReferenceInformationType", ["Direct"]),
                Rec("Smoothing", [np.zeros(len(sizes), dtype=np.int32)])]))
            layer0.append("LayerElementSmoothing")
        mats = []
        if mesh.face_material is not None:
            fm = mesh.face_material
            mats = list(dict.fromkeys(fm.tolist()))           # order of first use
            uniq, inv = np.unique(fm, return_inverse=True)
            local = np.array([mats.index(u) for u in uniq.tolist()])[inv.ravel()]
            single = len(mats) == 1
            recs.append(Rec("LayerElementMaterial", [0], [
                Rec("Version", [101]), Rec("Name", [""]),
                Rec("MappingInformationType", ["AllSame" if single else "ByPolygon"]),
                Rec("ReferenceInformationType", ["IndexToDirect"]),
                Rec("Materials", [(np.zeros(1) if single else local).astype(np.int32)])]))
            layer0.append("LayerElementMaterial")
        order = ["LayerElementNormal", "LayerElementColor", "LayerElementMaterial",
                 "LayerElementSmoothing", "LayerElementUV"]
        recs.append(Rec("Layer", [0], [Rec("Version", [100])] + [
            Rec("LayerElement", [], [Rec("Type", [t]), Rec("TypedIndex", [0])])
            for t in order if t in layer0]))
        uid = self.next_uid()
        self.add("Geometry", Rec("Geometry", [uid, self.safe(m.name, "") + "\x00\x01Geometry",
                                              "Mesh"], recs))
        self.geo_of[key] = (uid, mats)
        return self.geo_of[key]

    # -- models -----------------------------------------------------------------------------

    def model(self, node, matrix, parent_uid):
        """Write `node` with local `matrix` (scene axes, file units) under parent_uid."""
        f = _m4(C3) @ matrix @ _m4(C3.T)
        r, s, k = decompose(f[:3, :3])
        residual = None
        if k is not None:
            residual = C3.T @ k @ C3                       # back to scene axes
            nonortho = np.abs(k - np.eye(3)).max()
            if nonortho > 1e-4:
                self.loss(tr("FBX: трансформ со сдвигом (shear) перенесён в точки геометрии и в "
                             "дочерние узлы"))
        rows = [("RotationActive", "bool", "", "", 1), ("InheritType", "enum", "", "", 1),
                ("ScalingMax", "Vector3D", "Vector", "", 0, 0, 0),
                ("DefaultAttributeIndex", "int", "Integer", "", 0)]
        t = f[:3, 3] * self.scale
        if np.any(t):
            rows.append(("Lcl Translation", "Lcl Translation", "", "A", *t))
        deg = euler_xyz(r)
        deg[np.abs(deg) < 1e-12] = 0.0
        if np.any(deg):
            rows.append(("Lcl Rotation", "Lcl Rotation", "", "A", *deg))
        if not np.array_equal(s, np.ones(3)):
            rows.append(("Lcl Scaling", "Lcl Scaling", "", "A", *s))
        geo = None
        if node.mesh is not None:
            m = self.scene.meshes[node.mesh]
            if m.n_faces:
                geo = self.geometry(node.mesh, residual)
            else:
                self.loss(tr("FBX: пустые меши (без граней) записаны пустыми узлами"))
        if geo is not None:
            rows.append(("currentUVSet", "KString", "", "U", "map1"))
        uid = self.next_uid()
        self.add("Model", Rec("Model", [uid, self.safe(node.name, "node") + "\x00\x01Model",
                                        "Mesh" if geo else "Null"], [
            Rec("Version", [232]), P70(rows), Rec("Shading", [True]),
            Rec("Culling", ["CullingOff"])]))
        self.connect(uid, parent_uid)
        if geo is not None:
            gid, mats = geo
            self.connect(gid, uid)
            for mi in mats:
                self.connect(self.material(mi), uid)
        else:
            aid = self.next_uid()
            self.add("NodeAttribute", Rec("NodeAttribute", [aid, "\x00\x01NodeAttribute", "Null"], [
                P70([("Look", "enum", "", "", 0)]), Rec("TypeFlags", ["Null"])]))
            self.connect(aid, uid)
        push = np.eye(4) if residual is None else _m4(residual)
        for ch in node.children:
            self.model(ch, push @ ch.matrix, uid)

    # -- file skeleton ----------------------------------------------------------------------

    def build(self, unit_m):
        for n in self.scene.nodes:
            self.model(n, n.matrix, Int64(0))
        stack, layer = self.next_uid(), self.next_uid()
        stop = KTIME_SECOND
        self.add("AnimationStack", Rec("AnimationStack", [stack, "Take 001\x00\x01AnimStack", ""], [
            P70([("LocalStop", "KTime", "Time", "", stop),
                 ("ReferenceStop", "KTime", "Time", "", stop)])]))
        self.add("AnimationLayer", Rec("AnimationLayer", [layer, "BaseLayer\x00\x01AnimLayer", ""]))
        self.connect(layer, stack)
        now = datetime.datetime.now()
        usf = unit_m * 100.0
        templates = [("AnimationStack", "FbxAnimStack", ANIM_STACK_TEMPLATE),
                     ("AnimationLayer", "FbxAnimLayer", ANIM_LAYER_TEMPLATE),
                     ("NodeAttribute", "FbxNull", NULL_TEMPLATE),
                     ("Model", "FbxNode", MODEL_TEMPLATE),
                     ("Geometry", "FbxMesh", GEOMETRY_TEMPLATE),
                     ("Material", "FbxSurfaceLambert", MATERIAL_TEMPLATE),
                     ("Texture", "FbxFileTexture", TEXTURE_TEMPLATE),
                     ("Video", "FbxVideo", VIDEO_TEMPLATE)]
        defs = [Rec("ObjectType", ["GlobalSettings"], [Rec("Count", [1])])]
        for type_name, tname, rows in templates:
            if self.counts.get(type_name):
                defs.append(Rec("ObjectType", [type_name], [
                    Rec("Count", [self.counts[type_name]]),
                    Rec("PropertyTemplate", [tname], [P70(rows)])]))
        app = [("ApplicationVendor", "mesh2mesh"), ("ApplicationName", "mesh2mesh"),
               ("ApplicationVersion", "1")]
        date = now.strftime("%d/%m/%Y %H:%M:%S.") + f"{now.microsecond // 1000:03d}"
        top = [
            Rec("FBXHeaderExtension", [], [
                Rec("FBXHeaderVersion", [1003]), Rec("FBXVersion", [FBX_VERSION]),
                Rec("EncryptionType", [0]),
                Rec("CreationTimeStamp", [], [Rec("Version", [1000])] + [
                    Rec(k, [v]) for k, v in (("Year", now.year), ("Month", now.month),
                                             ("Day", now.day), ("Hour", now.hour),
                                             ("Minute", now.minute), ("Second", now.second),
                                             ("Millisecond", now.microsecond // 1000))]),
                Rec("Creator", ["mesh2mesh"]),
                Rec("SceneInfo", ["GlobalInfo\x00\x01SceneInfo", "UserData"], [
                    Rec("Type", ["UserData"]), Rec("Version", [100]),
                    Rec("MetaData", [], [Rec("Version", [100])] + [
                        Rec(k, [""]) for k in ("Title", "Subject", "Author", "Keywords",
                                               "Revision", "Comment")]),
                    P70([("DocumentUrl", "KString", "Url", "", ""),
                         ("SrcDocumentUrl", "KString", "Url", "", ""),
                         ("Original", "Compound", "", ""),
                         *[(f"Original|{k}", "KString", "", "", v) for k, v in app],
                         ("Original|DateTime_GMT", "DateTime", "", "", date),
                         ("Original|FileName", "KString", "", "", ""),
                         ("LastSaved", "Compound", "", ""),
                         *[(f"LastSaved|{k}", "KString", "", "", v) for k, v in app],
                         ("LastSaved|DateTime_GMT", "DateTime", "", "", date)])])]),
            Rec("FileId", [FILE_ID]),
            Rec("CreationTime", [FILE_CREATION_TIME]),
            Rec("Creator", ["mesh2mesh"]),
            Rec("GlobalSettings", [], [Rec("Version", [1000]), P70([
                ("UpAxis", "int", "Integer", "", 1), ("UpAxisSign", "int", "Integer", "", 1),
                ("FrontAxis", "int", "Integer", "", 2), ("FrontAxisSign", "int", "Integer", "", 1),
                ("CoordAxis", "int", "Integer", "", 0), ("CoordAxisSign", "int", "Integer", "", 1),
                ("OriginalUpAxis", "int", "Integer", "", 1),
                ("OriginalUpAxisSign", "int", "Integer", "", 1),
                ("UnitScaleFactor", "double", "Number", "", usf),
                ("OriginalUnitScaleFactor", "double", "Number", "", usf),
                ("AmbientColor", "ColorRGB", "Color", "", 0, 0, 0),
                ("DefaultCamera", "KString", "", "", "Producer Perspective"),
                ("TimeMode", "enum", "", "", 11), ("TimeProtocol", "enum", "", "", 2),
                ("SnapOnFrameMode", "enum", "", "", 0),
                ("TimeSpanStart", "KTime", "Time", "", 0),
                ("TimeSpanStop", "KTime", "Time", "", stop),
                ("CustomFrameRate", "double", "Number", "", -1),
                ("TimeMarker", "Compound", "", ""),
                ("CurrentTimeMarker", "int", "Integer", "", -1)])]),
            Rec("Documents", [], [Rec("Count", [1]), Rec("Document", [self.next_uid(), "", "Scene"], [
                P70([("SourceObject", "object", "", ""),
                     ("ActiveAnimStackName", "KString", "", "", "Take 001")]),
                Rec("RootNode", [Int64(0)])])]),
            Rec("References", []),
            Rec("Definitions", [], [Rec("Version", [100]),
                                    Rec("Count", [1 + sum(self.counts.values())]), *defs]),
            Rec("Objects", [], self.objects),
            Rec("Connections", [], self.conns),
            Rec("Takes", [], [Rec("Current", ["Take 001"]), Rec("Take", ["Take 001"], [
                Rec("FileName", ["Take_001.tak"]),
                Rec("LocalTime", [Int64(0), Int64(stop)]),
                Rec("ReferenceTime", [Int64(0), Int64(stop)])])]),
        ]
        return serialize(top)


def write(scene, path, scale=1.0, unit_m=1e-3):
    """scale: file units -> output units; unit_m: metres per output unit (-> UnitScaleFactor).
    Binary FBX 7400, Y up. Returns warning lines."""
    wr = _Writer(scene, scale)
    data = wr.build(unit_m)
    with open(path, "wb") as fh:
        fh.write(data)
    return wr.losses
