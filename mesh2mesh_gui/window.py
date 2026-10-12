"""Main window: open -> look (tree, size, axes) -> pick a format (losses shown up front) -> save
-> plain-language result with the read-back verdict."""

import os
import subprocess
import traceback
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QSettings, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QFileDialog, QFrame,
                               QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea,
                               QSplitter, QTextBrowser, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget)

from mesh2mesh import __url__, __version__, convert, formats, geom

from . import display, report, teximage
from .viewer import MeshView, ViewBar

APP_TITLE = "mesh2mesh — конвертер 3D-моделей"
LEVEL_COLOR = {report.OK: "#2e9e4f", report.WARN: "#d08a00", report.ERROR: "#d03a2a"}
LEVEL_MARK = {report.OK: "✔", report.WARN: "⚠", report.ERROR: "✖"}
GRAY = "color: gray"


@dataclass
class Model:
    path: str
    scene: object        # mesh2mesh.model.Scene as read; unit_m set (has_units or the choice)
    has_units: bool      # the file stores its unit (else the "units of the source" choice rules)
    disp: display.Display
    images: list         # teximage.Image per disp.textures (decoded off the UI thread)


def load_model(path, in_units):
    ext = os.path.splitext(path)[1].lower()
    if ext not in formats.READERS:
        raise ValueError(f"формат {ext or '(без расширения)'} не поддерживается")
    scene = formats.READERS[ext](path)
    has_units = scene.unit_m is not None
    if not has_units:
        scene.unit_m = convert.UNITS[in_units]
    disp = display.build(scene)
    return Model(path, scene, has_units, disp, teximage.decode_all(disp.textures))


class Job(QThread):
    """Run fn(*args, **kw) off the UI thread."""
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, parent=None, **kw):
        super().__init__(parent)
        self.fn, self.args, self.kw = fn, args, kw

    def run(self):
        try:
            self.done.emit(self.fn(*self.args, **self.kw))
        except Exception as e:  # shown to the user; details to the console / log
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _html_lines(lines):
    return "".join(f'<p style="margin:2px 0"><span style="color:{LEVEL_COLOR[lv]}">'
                   f"{LEVEL_MARK[lv]}</span> {_esc(t)}</p>" for lv, t in lines)


def _hint(text):
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet(GRAY)
    return lab


class MainWindow(QMainWindow):
    SETTINGS = ("clean_mesh", "mesh2mesh")   # tools/gui_check.py swaps it: user prefs stay intact

    def __init__(self):
        super().__init__()
        self.settings = QSettings(*self.SETTINGS)
        self.model = None
        self.job = None            # load / save: blocks the controls
        self.loss_job = None       # trial write for the losses box: runs alongside
        self.loss_key = None       # key of the losses currently shown
        self.scale = 1.0
        self.rotation = np.eye(3)
        self.last_out = None
        self.setWindowTitle(f"{APP_TITLE} {__version__}")
        self.setAcceptDrops(True)
        self.resize(1360, 860)
        self.loss_timer = QTimer(self)
        self.loss_timer.setSingleShot(True)
        self.loss_timer.setInterval(250)
        self.loss_timer.timeout.connect(self._run_losses)
        self._build_ui()
        self._refresh()

    # ---------- UI ----------
    def _build_ui(self):
        self.view = MeshView()
        side = QWidget()
        side.setMinimumWidth(420)
        lay = QVBoxLayout(side)

        # 1. file
        box = QGroupBox("1. Файл")
        b = QVBoxLayout(box)
        exts = ", ".join(e[1:].upper() for e in sorted(formats.READERS))
        self.open_btn = QPushButton(f"Открыть модель…  ({exts})")
        self.open_btn.clicked.connect(self.open_dialog)
        self.file_label = QLabel("Файл не выбран. Можно перетащить файл в окно.")
        self.file_label.setWordWrap(True)
        self.file_info = _hint("")
        row = QHBoxLayout()
        self.in_units_label = QLabel("Числа в файле — это:")
        row.addWidget(self.in_units_label)
        self.in_units = QComboBox()
        for key in convert.UNITS:
            self.in_units.addItem(report.UNIT_RU[key], key)
        self.in_units.setCurrentIndex(max(0, self.in_units.findData(
            self.settings.value("in_units", "mm"))))
        self.in_units.currentIndexChanged.connect(self._in_units_changed)
        row.addWidget(self.in_units)
        row.addStretch()
        b.addWidget(self.open_btn)
        b.addWidget(self.file_label)
        b.addWidget(self.file_info)
        b.addLayout(row)
        lay.addWidget(box)

        # 2. contents
        box = QGroupBox("2. Содержимое")
        b = QVBoxLayout(box)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Объект", "Треугольников", "Материалы"])
        self.tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        hh = self.tree.header()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.Interactive)
        hh.setStretchLastSection(False)
        self.tree.setColumnWidth(2, 120)
        self.tree.itemSelectionChanged.connect(self._tree_selected)
        self.tree.setMinimumHeight(120)
        self.tree.setToolTip("Щелчок по строке — объект крупно (с вложенными); «Показать всё» — "
                             "вся модель. ×N — один меш, повторённый N раз (экземпляры).")
        b.addWidget(self.tree, 1)
        self.read_warn = QLabel()
        self.read_warn.setWordWrap(True)
        b.addWidget(self.read_warn)
        lay.addWidget(box, 1)

        # 3. size and axes
        box = QGroupBox("3. Размер и оси")
        b = QVBoxLayout(box)
        self.size_label = QLabel()
        self.size_label.setWordWrap(True)
        b.addWidget(self.size_label)
        row = QHBoxLayout()
        row.addWidget(QLabel("Масштаб:"))
        self.down_btn = QPushButton("÷ 10")
        self.up_btn = QPushButton("× 10")
        self.scale_edit = QLineEdit("1")
        self.scale_edit.setMaximumWidth(110)
        self.scale_edit.setAlignment(Qt.AlignCenter)
        self.scale_edit.setToolTip("Любое число > 0: 25.4 — дюймы в мм, 0.1 — мм в см… "
                                   "Габарит не тот в 10/100/1000 раз — поправьте здесь")
        self.scale_edit.editingFinished.connect(self._scale_typed)
        self.down_btn.clicked.connect(lambda: self._set_scale(self.scale / 10))
        self.up_btn.clicked.connect(lambda: self._set_scale(self.scale * 10))
        for w in (self.down_btn, self.scale_edit, self.up_btn):
            row.addWidget(w)
        row.addStretch()
        b.addLayout(row)
        row = QHBoxLayout()
        self.rot_btns = []
        for axis in "xyz":
            btn = QPushButton(f"↻ {axis.upper()} 90°")
            btn.setToolTip(f"Повернуть модель на 90° вокруг оси {axis.upper()}. Лежит на боку — "
                           "крутите, пока синяя ось Z не смотрит вверх модели")
            btn.clicked.connect(lambda _=False, a=axis: self._rotate(a))
            row.addWidget(btn)
            self.rot_btns.append(btn)
        self.rot_reset = QPushButton("Как в файле")
        self.rot_reset.clicked.connect(self._rotate_reset)
        row.addWidget(self.rot_reset)
        b.addLayout(row)
        row = QHBoxLayout()
        self.view_reset = QPushButton("Показать всё")
        self.view_reset.setToolTip("Вся модель в кадре, без подсветки")
        self.view_reset.clicked.connect(self._show_all)
        row.addWidget(self.view_reset)
        self.rot_label = _hint("")
        row.addWidget(self.rot_label, 1)
        b.addLayout(row)
        lay.addWidget(box)

        # 4. save
        box = QGroupBox("4. Сохранить")
        b = QVBoxLayout(box)
        row = QHBoxLayout()
        row.addWidget(QLabel("Формат:"))
        self.fmt = QComboBox()
        for ext, label in report.FORMATS.items():
            self.fmt.addItem(label, ext)
        self.fmt.setCurrentIndex(max(0, self.fmt.findData(self.settings.value("format", ".3mf"))))
        self.fmt.currentIndexChanged.connect(self._format_changed)
        row.addWidget(self.fmt, 1)
        b.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Единицы:"))
        self.out_units = QComboBox()
        self.out_units.addItem("как у исходника", None)
        for key in convert.UNITS:
            self.out_units.addItem(report.UNIT_RU[key], key)
        self.out_units.currentIndexChanged.connect(self._options_changed)
        row.addWidget(self.out_units)
        self.ascii = QCheckBox("текстом (ASCII)")
        self.ascii.setToolTip("STL/PLY текстом: больше файл; STL так сохраняет имена объектов")
        self.ascii.setChecked(self.settings.value("ascii", "false") == "true")
        self.ascii.toggled.connect(self._options_changed)
        row.addWidget(self.ascii)
        row.addStretch()
        b.addLayout(row)
        self.units_note = _hint("")
        b.addWidget(self.units_note)
        self.losses = QLabel()
        self.losses.setWordWrap(True)
        self.losses.setTextInteractionFlags(Qt.TextSelectableByMouse)
        b.addWidget(self.losses)
        self.save_btn = QPushButton("Сохранить")
        self.save_btn.setStyleSheet("font-size: 14px; font-weight: bold; padding: 8px")
        self.save_btn.clicked.connect(self.save_dialog)
        b.addWidget(self.save_btn)
        self.result = QTextBrowser()
        self.result.setOpenLinks(False)
        self.result.anchorClicked.connect(self._result_link)
        self.result.setMinimumHeight(140)
        self.result.setVisible(False)
        b.addWidget(self.result)
        lay.addWidget(box)
        author = QLabel(f'<a href="{__url__}" style="color: gray">автор: '
                        f'{__url__.split("//")[1].rstrip("/")}</a>')
        author.setOpenExternalLinks(True)
        author.setToolTip(__url__)
        author.setAlignment(Qt.AlignRight)
        author.setStyleSheet("font-size: 8pt")
        self.author = author
        lay.addWidget(author)

        scroll = QScrollArea()  # small screens / long result: scroll, never squash the tree
        scroll.setWidget(side)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view_bar = ViewBar()
        self.view_bar.set_state(self.settings.value("view_mode", "material"),
                                self.settings.value("wireframe", "true") == "true",
                                self.settings.value("smoothing", display.SMOOTH_FILE))
        self.view.set_mode(self.settings.value("view_mode", "material"))
        self.view.set_wireframe(self.view_bar.wire.isChecked())
        self.view.set_smoothing(self.view_bar.smoothing)
        self.view_bar.mode_changed.connect(self._view_mode)
        self.view_bar.wireframe_changed.connect(self._wireframe)
        self.view_bar.smoothing_changed.connect(self._smoothing)
        view_box = QWidget()
        vb = QVBoxLayout(view_box)
        vb.setContentsMargins(0, 0, 0, 0)
        vb.setSpacing(0)
        vb.addWidget(self.view_bar)
        vb.addWidget(self.view, 1)
        split = QSplitter()
        split.addWidget(view_box)
        self.split = split
        self.view.gl_failed.connect(self._gl_failed)
        split.addWidget(scroll)
        split.setStretchFactor(0, 1)
        split.setSizes([880, 480])
        self.setCentralWidget(split)

        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.setMaximumWidth(160)
        self.busy.setVisible(False)
        self.statusBar().addPermanentWidget(self.busy)
        for key, slot in ((QKeySequence.Open, self.open_dialog), (QKeySequence.Save, self.save_dialog)):
            act = QAction(self)
            act.setShortcut(key)
            act.triggered.connect(slot)
            self.addAction(act)

    # ---------- state ----------
    @property
    def has(self):
        return self.model is not None and not self.model.disp.empty

    @property
    def out_ext(self):
        return self.fmt.currentData()

    def _write_opts(self):
        return {"binary": not self.ascii.isChecked()} if self.out_ext in formats.TEXT_OPTION else {}

    def _units_choice(self):
        return self.out_units.currentData()

    def _effective_units(self):
        return convert.out_units(self.model.scene, self.out_ext, self._units_choice())[0]

    def _busy(self, text):
        self.statusBar().showMessage(text)
        self.busy.setVisible(bool(text))
        self._refresh()

    def _refresh(self):
        has, idle = self.has, self.job is None
        self.open_btn.setEnabled(idle)
        no_units = self.model is not None and not self.model.has_units
        self.in_units.setEnabled(idle and no_units)
        self.in_units.setVisible(self.model is None or no_units)
        self.in_units_label.setVisible(self.model is None or no_units)
        for w in (self.down_btn, self.up_btn, self.scale_edit, self.rot_reset, self.view_reset,
                  *self.rot_btns):
            w.setEnabled(idle and has)
        fixed = formats.FIXED_UNITS.get(self.out_ext)
        self.out_units.setEnabled(idle and not fixed)
        self.ascii.setEnabled(idle and self.out_ext in formats.TEXT_OPTION)
        self.fmt.setEnabled(idle)
        self.save_btn.setEnabled(idle and has)
        self.save_btn.setText(f"Сохранить как {self.out_ext[1:].upper()}")
        if not has:
            self.size_label.setText("" if self.model is None else "В файле нет геометрии — "
                                    "сохранять нечего.")
            self.rot_label.setText("")
            self.units_note.setText("")
            return
        units = self._effective_units()
        m = self.model
        size = report.size_in(m.disp, m.scene.unit_m, units, self.scale, self.rotation)
        changed = self.scale != 1.0
        self.size_label.setText(report.size_line(size, units))
        self.size_label.setStyleSheet("font-size: 15px; font-weight: bold; padding: 4px 0"
                                      + ("; color: #d08a00" if changed else ""))
        step = display.grid_step(size)
        diag = float(np.linalg.norm(size)) or 1.0
        self.view.set_grid(step / diag)
        turned = not np.allclose(self.rotation, np.eye(3))
        self.rot_label.setText(("Повёрнута относительно файла" if turned else "Оси как в файле")
                               + f" · клетка сетки {report.fmt_len(step)} "
                               f"{report.UNIT_RU[units]}")
        if fixed:
            note = f"{self.out_ext[1:].upper()} по спецификации всегда в метрах."
        elif self._units_choice() is None:
            note = f"Будет записано в {report.UNIT_RU[units]}."
        else:
            note = ""
        if self.out_ext in (".obj", ".stl", ".ply"):
            note += (" Единица в этот формат не записывается — получатель должен знать, что "
                     f"числа в {report.UNIT_RU[units]}.")
        self.units_note.setText(note.strip())

    # ---------- open ----------
    def open_dialog(self):
        if self.job:
            return
        start = self.settings.value("last_dir", "")
        path, _ = QFileDialog.getOpenFileName(self, "Открыть модель", start, report.INPUT_FILTER)
        if path:
            self.open_file(path)

    def open_file(self, path):
        if self.job:
            return
        if os.path.splitext(path)[1].lower() not in formats.READERS:
            exts = ", ".join(e[1:].upper() for e in sorted(formats.READERS))
            QMessageBox.warning(self, APP_TITLE, f"Этот формат не поддерживается.\nЕсть: {exts}.")
            return
        self.settings.setValue("last_dir", os.path.dirname(path))
        self._pending = path
        self.job = Job(load_model, path, self.in_units.currentData(), parent=self)
        self.job.done.connect(self._loaded)
        self.job.failed.connect(self._load_failed)
        self.job.finished.connect(self._job_finished)
        self._busy(f"Читаю {os.path.basename(path)}…")
        self.job.start()

    def _loaded(self, model):
        self.model = model
        self.scale = 1.0
        self.scale_edit.setText("1")
        self.rotation = np.eye(3)
        self.result.setVisible(False)
        self.last_out = None
        name = os.path.basename(model.path)
        self.setWindowTitle(f"{name} — {APP_TITLE} {__version__}")
        self.file_label.setText(f"<b>{_esc(name)}</b>")
        self.file_info.setText("<br>".join(_esc(s) for s in report.file_summary(
            model.scene, model.path, model.has_units, model.disp, model.images)))
        warns = model.scene.warnings
        self.read_warn.setText(
            "" if not warns else
            _html_lines([(report.WARN, "При чтении: " + w) for w in warns]))
        self.view.set_display(model.disp, self.rotation, images=model.images)
        self.view_bar.wire.setEnabled(self.view.has_edges)
        self.view_bar.wire.setToolTip("Рёбра многоугольников поверх любого режима" if
                                      self.view.has_edges else
                                      f"Рёбер больше {display.EDGE_LIMIT:,} — каркас не рисуется"
                                      .replace(",", " "))
        self.view_bar.set_smoothing_note(report.normals_note(model.disp))
        self._fill_tree()
        self.loss_key = None
        self._schedule_losses()

    def _load_failed(self, msg):
        QMessageBox.critical(self, APP_TITLE, f"Не удалось прочитать файл\n{self._pending}\n\n{msg}")

    def _job_finished(self):
        self.job = None
        self._busy("")

    def _fill_tree(self):
        m = self.model
        self.tree.clear()
        scene, disp = m.scene, m.disp
        part_of = {id(n): i for i, n in enumerate(disp.part_nodes)}
        tri_count = np.bincount(disp.tri_part, minlength=len(disp.part_nodes))
        uses = np.bincount([n.mesh for n in disp.part_nodes], minlength=len(scene.meshes)) \
            if disp.part_nodes else np.zeros(len(scene.meshes), int)

        def mats(mesh):
            if mesh.face_material is None:
                return ""
            idx = [i for i in np.unique(mesh.face_material) if i >= 0]
            names = [scene.materials[i].name for i in idx]
            return ", ".join(names[:3]) + (f" … ({len(names)})" if len(names) > 3 else "")

        def add(node, parent):
            item = QTreeWidgetItem(parent)
            name = node.name or "(без имени)"
            parts = []
            if node.mesh is not None:
                mesh = scene.meshes[node.mesh]
                pi = part_of.get(id(node))
                if pi is not None:
                    parts.append(pi)
                    item.setText(1, report.fmt_int(tri_count[pi]))
                    item.setTextAlignment(1, Qt.AlignRight | Qt.AlignVCenter)
                if uses[node.mesh] > 1:
                    name += f"  ×{uses[node.mesh]}"
                    item.setToolTip(0, f"Меш «{mesh.name}» повторён {uses[node.mesh]} раз "
                                       "(экземпляры)")
                    item.setForeground(0, QColor("#4a7fc0"))
                item.setText(2, mats(mesh))
                item.setToolTip(2, item.text(2))
            item.setText(0, name)
            if not item.toolTip(0):
                item.setToolTip(0, name)
            for ch in node.children:
                parts += add(ch, item)
            item.setData(0, Qt.UserRole, parts)
            return parts

        for root in scene.nodes:
            add(root, self.tree)
        if self.tree.topLevelItemCount() <= 3:
            self.tree.expandToDepth(1 if len(disp.part_nodes) < 200 else 0)

    def _tree_selected(self):
        items = self.tree.selectedItems()
        self.view.select(items[0].data(0, Qt.UserRole) if items else None)

    def _show_all(self):
        self.tree.clearSelection()
        self.view.select(None)
        self.view.reset_camera()

    def _view_mode(self, mode):
        self.settings.setValue("view_mode", mode)
        self.view.set_mode(mode)

    def _wireframe(self, on):
        self.settings.setValue("wireframe", "true" if on else "false")
        self.view.set_wireframe(on)

    def _smoothing(self, mode):
        self.settings.setValue("smoothing", mode)
        QApplication.setOverrideCursor(Qt.WaitCursor)   # auto smoothing of a big scene: ~1 s
        try:
            self.view.set_smoothing(mode)
        finally:
            QApplication.restoreOverrideCursor()

    # ---------- units / size / axes ----------
    def _in_units_changed(self):
        self.settings.setValue("in_units", self.in_units.currentData())
        m = self.model
        if m is not None and not m.has_units:
            m.scene.unit_m = convert.UNITS[self.in_units.currentData()]
            self.file_info.setText("<br>".join(_esc(s) for s in report.file_summary(
                m.scene, m.path, m.has_units, m.disp, m.images)))
            self._schedule_losses()
        self._refresh()

    def _set_scale(self, value):
        self.scale = float(f"{value:.12g}")  # ×10 ÷10 round trips stay exact-looking
        self.scale_edit.setText(report.fmt_scale(self.scale))
        self._refresh()

    def _scale_typed(self):
        v = report.parse_scale(self.scale_edit.text())
        if v is None:
            self.statusBar().showMessage("Масштаб — положительное число, например 10 или 0.001",
                                         6000)
            v = self.scale
        self._set_scale(v)

    def _rotate(self, axis):
        self.rotation = geom.quarter_turn(axis, 1) @ self.rotation
        self.view.set_rotation(self.rotation)
        self._refresh()

    def _rotate_reset(self):
        self.rotation = np.eye(3)
        self.view.set_rotation(self.rotation)
        self._refresh()

    # ---------- format and losses ----------
    def _format_changed(self):
        self.settings.setValue("format", self.out_ext)
        self._options_changed()

    def _options_changed(self):
        self.settings.setValue("ascii", "true" if self.ascii.isChecked() else "false")
        self._refresh()
        self._schedule_losses()

    def _current_loss_key(self):
        m = self.model
        return (id(m), m.scene.unit_m, self.out_ext, self._units_choice(),
                tuple(sorted(self._write_opts().items())))

    def _schedule_losses(self):
        if not self.has:
            self.losses.setText("")
            return
        if self.loss_key != self._current_loss_key():
            self.losses.setText(f'<span style="{GRAY}">Проверяю, что вместит '
                                f'{self.out_ext[1:].upper()}…</span>')
        self.loss_timer.start()

    def _run_losses(self):
        if not self.has or self.loss_job is not None:
            return  # a running trial re-checks the key when it finishes
        key = self._current_loss_key()
        if key == self.loss_key:
            return
        self._loss_running = key
        self.loss_job = Job(convert.losses, self.model.scene, self.out_ext, self._units_choice(),
                            parent=self, **self._write_opts())
        self.loss_job.done.connect(self._losses_done)
        self.loss_job.failed.connect(self._losses_failed)
        self.loss_job.finished.connect(self._losses_finished)
        self.loss_job.start()

    def _losses_done(self, losses):
        if self.has and self._loss_running == self._current_loss_key():
            self.loss_key = self._loss_running
            self.losses.setText(_html_lines(report.losses_summary(losses, self.out_ext)))

    def _losses_failed(self, msg):
        if self.has and self._loss_running == self._current_loss_key():
            self.loss_key = self._loss_running
            self.losses.setText(_html_lines([(report.ERROR, f"Пробная запись не удалась: {msg}")]))

    def _losses_finished(self):
        self.loss_job = None
        if self.has and self.loss_key != self._current_loss_key():
            self.loss_timer.start()

    # ---------- save ----------
    def save_dialog(self):
        if self.job or not self.has:
            return
        m = self.model
        ext = self.out_ext
        start = report.default_out_path(m.path, ext, self.settings.value("out_dir", None))
        filt = f"{ext[1:].upper()} (*{ext})"
        path, _ = QFileDialog.getSaveFileName(self, f"Сохранить как {ext[1:].upper()}", start, filt)
        if not path:
            return
        path, typed = report.fix_out_path(path, ext)
        if typed != ext:  # the user typed another known extension: that is the format now
            self.fmt.setCurrentIndex(self.fmt.findData(typed))
        if os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(m.path)):
            QMessageBox.warning(self, APP_TITLE, "Нельзя записать поверх исходного файла — "
                                                 "выберите другое имя.")
            return
        self.settings.setValue("out_dir", os.path.dirname(path))
        self.job = Job(convert.convert, m.path, path, self._units_choice(), "mm", self.scale,
                       self.rotation.copy(), scene=m.scene, parent=self, **self._write_opts())
        self.job.done.connect(self._saved)
        self.job.failed.connect(self._save_failed)
        self.job.finished.connect(self._job_finished)
        self._busy("Сохраняю и проверяю чтением…")
        self.job.start()

    def _saved(self, res):
        self.last_out = res
        html = _html_lines(report.result_summary(res, self.model.scene.warnings))
        html += ("<p style='margin:6px 0'><a href='show'>Показать в папке</a> · "
                 "<a href='details'>Технический отчёт</a></p>")
        self.result.setHtml(html)
        self.result.setVisible(True)
        self.statusBar().showMessage("Сохранено" if res.ok else
                                     "Сохранено, но проверка чтением не прошла", 10000)

    def _save_failed(self, msg):
        QMessageBox.critical(self, APP_TITLE, f"Не удалось сохранить.\n\n{msg}")

    def _result_link(self, url):
        res = self.last_out
        if not res:
            return
        if url.toString() == "show":
            path = os.path.normpath(res.out_path)
            if os.name == "nt" and os.path.exists(path):
                subprocess.Popen(["explorer", "/select,", path])
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(path)))
        else:
            box = QMessageBox(QMessageBox.Information, "Технический отчёт",
                              "Отчёт как в командной строке (можно скопировать).", parent=self)
            box.setDetailedText("\n".join(res.lines + ["WARNING: " + w for w in res.warnings]
                                          + ["MISMATCH: " + w for w in res.mismatches]))
            box.exec()

    def _gl_failed(self, detail):
        """No usable OpenGL: drop the 3D view, keep everything else (loading, size, saving
        need no graphics card)."""
        note = QLabel("<p style='font-size:12pt'><b>3D-просмотр недоступен</b></p>"
                      "<p>Видеокарта или драйвер не дают OpenGL 2.0 (так бывает в удалённом "
                      "рабочем столе и виртуальных машинах).</p>"
                      "<p>Открыть модель, посмотреть содержимое и сохранить в другой формат "
                      "можно как обычно — не будет только картинки. Поможет обновление драйвера "
                      "видеокарты.</p>"
                      f"<p style='color:gray; font-size:8pt'>{_esc(detail)}</p>")
        note.setWordWrap(True)
        note.setAlignment(Qt.AlignCenter)
        note.setMargin(40)
        note.setStyleSheet("background: #2e3036; color: #e6e6e6;")
        note.setMinimumWidth(300)
        self.gl_note = note  # PySide's replaceWidget does not take ownership: keep it alive
        sizes = self.split.sizes()
        self.split.replaceWidget(0, note)
        self.view.hide()
        self.split.setCollapsible(0, False)
        self.split.setSizes([max(sizes[0], 600), sizes[1]])  # the view may have had width 0

    # ---------- drag & drop ----------
    def dragEnterEvent(self, e):
        urls = e.mimeData().urls()
        if urls and urls[0].toLocalFile().lower().endswith(tuple(formats.READERS)):
            e.acceptProposedAction()

    def dropEvent(self, e):
        self.open_file(e.mimeData().urls()[0].toLocalFile())

    def closeEvent(self, e):
        if self.job and self.job.isRunning():
            if QMessageBox.question(self, APP_TITLE, "Идёт работа. Всё равно закрыть?") \
                    != QMessageBox.Yes:
                e.ignore()
                return
            self.job.wait()
        if self.loss_job:
            self.loss_job.wait()
        super().closeEvent(e)
