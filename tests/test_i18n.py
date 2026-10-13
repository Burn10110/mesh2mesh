"""Guards of the two-language interface (mesh2mesh/i18n.py, i18n_en.py).

Static: every tr()/N_() literal has an English entry with the same placeholders, tr() keyword
arguments name exactly those placeholders, no dead entries, no Cyrillic text left outside
tr()/N_(). Runtime: in English nothing a user sees (losses on every format pair, reader
errors on broken files, CLI output and --help) contains Cyrillic."""

import ast
import os
import re
import string
import subprocess
import sys
from pathlib import Path

import pytest

from mesh2mesh import convert, formats, i18n
from mesh2mesh.i18n import tr
from mesh2mesh.i18n_en import EN

ROOT = Path(__file__).resolve().parents[1]
CODE = [p for d in ("mesh2mesh", "mesh2mesh_gui") for p in sorted((ROOT / d).rglob("*.py"))
        if p.name != "i18n_en.py"]
CYR = re.compile("[Ѐ-ӿ]")


def _fields(s):
    try:
        return sorted({(f, spec, conv) for _, f, spec, conv in string.Formatter().parse(s)
                       if f is not None})
    except ValueError:      # not a format string (tr() without arguments never formats it)
        return [("raw", s.count("{"), s.count("}"))]


def _calls():
    """(file:line, key, kwarg names or None when there are none) of tr()/N_() with a literal."""
    out = []
    for p in CODE:
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if not (isinstance(n, ast.Call) and n.args):
                continue
            f = n.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            if name in ("tr", "N_") and isinstance(n.args[0], ast.Constant):
                kws = {k.arg for k in n.keywords} or None
                out.append((f"{p.relative_to(ROOT)}:{n.lineno}", n.args[0].value, kws))
    return out


CALLS = _calls()


def test_every_key_translated_with_same_placeholders():
    bad = [f"{where}: no English for {key!r}" for where, key, _ in CALLS if key not in EN]
    bad += [f"placeholders differ: {k!r} -> {v!r}" for k, v in EN.items() if _fields(k) != _fields(v)]
    assert not bad, "\n".join(bad)


def test_keyword_arguments_match_placeholders():
    bad = []
    for where, key, kws in CALLS:
        names = {f.split(".")[0].split("[")[0] for f, _, _ in _fields(key)}
        if kws is not None and kws != names:
            bad.append(f"{where}: {sorted(kws)} vs {sorted(names)} in {key!r}")
    assert not bad, "\n".join(bad)


def test_no_duplicate_entries():
    """A repeated key in the dict literal silently keeps the last one."""
    tree = ast.parse((ROOT / "mesh2mesh" / "i18n_en.py").read_text(encoding="utf-8"))
    keys = [k.value for n in ast.walk(tree) if isinstance(n, ast.Dict) for k in n.keys]
    assert not {k for k in keys if keys.count(k) > 1}


def test_no_dead_or_untranslated_entries():
    used = {key for _, key, _ in CALLS}
    assert not [k for k in EN if k not in used], "dead entries"
    assert not [v for v in EN.values() if CYR.search(v)], "Cyrillic in English text"


NATIVE = {"Русский"}   # language names in the switch: each in its own language


def test_no_cyrillic_outside_tr():
    """String literals with Cyrillic must go through tr()/N_() (docstrings are fine)."""
    bad = []
    for p in CODE:
        tree = ast.parse(p.read_text(encoding="utf-8"))
        skip = set()
        for n in ast.walk(tree):
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant):
                    skip.add(id(n.body[0].value))
            if isinstance(n, ast.Call) and n.args and isinstance(n.args[0], ast.Constant):
                f = n.func
                if (f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)) in ("tr", "N_"):
                    skip.add(id(n.args[0]))
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in skip \
                    and CYR.search(n.value) and n.value not in NATIVE:
                bad.append(f"{p.relative_to(ROOT)}:{n.lineno}: {n.value[:60]!r}")
    assert not bad, "\n".join(bad)


def test_tr_and_language_switch():
    assert tr("STL: UV отброшены") == "STL: UV отброшены"
    i18n.set_lang("en")
    try:
        assert tr("STL: UV отброшены") == "STL: UV dropped"
        assert tr("{v:g} м", v=0.5) == "0.5 m"
        assert tr("not a key {x}", x=1) == "not a key 1"
    finally:
        i18n.set_lang("ru")
    assert i18n.normalize("EN_us") == "en" and i18n.normalize("de") is None
    assert i18n.pre_parse_lang(["a", "--lang", "en"]) == "en"
    assert i18n.pre_parse_lang(["--lang=ru", "x"]) == "ru"
    assert i18n.pre_parse_lang(["--lang"]) is None


# --- runtime: English output has no Cyrillic -------------------------------------------------

def _no_cyr(lines, what):
    bad = [s for s in lines if CYR.search(str(s))]
    assert not bad, f"{what}: " + " | ".join(map(str, bad))


@pytest.mark.parametrize("src", sorted(formats.WRITERS))
def test_english_conversion_reports(tmp_path, english, src):
    from test_formats import rich_scene
    from test_textures import tex_scene
    from mesh2mesh.model import Texture
    odd = tex_scene(Texture("paint.tga", bytes(18) + b"\xff" * 12))   # non-web image
    odd.meshes[1].face_material[:] = 0                                # textured, no UV
    for name, scene in (("rich", rich_scene()), ("tex", tex_scene()), ("odd", odd)):
        a = tmp_path / f"{name}{src}"
        formats.WRITERS[src](scene, str(a), scale=1.0, unit_m=1e-3)
        loaded = convert.load(str(a))
        _no_cyr(loaded.warnings, f"read {a.name}")
        _no_cyr(convert.describe(loaded), "describe")
        for dst in sorted(formats.WRITERS):
            res = convert.convert(str(a), str(tmp_path / f"out_{name}{dst}"), scene=loaded)
            _no_cyr(res.lines + res.warnings + res.mismatches, f"{a.name} -> {dst}")
            _no_cyr(convert.losses(loaded, dst, units="in"), f"losses {a.name} -> {dst}")


BROKEN = {
    "short.stl": b"\0" * 10,
    "count.stl": b"\0" * 80 + b"\x05\0\0\0",
    "none.stl": b"hello",
    "bad.obj": b"v 0 0 0\nf 1 2 3\n",
    "head.ply": b"plx\n",
    "prop.ply": b"ply\nformat ascii 1.0\nelement vertex 1\nproperty floaty x\nend_header\n",
    "fmt.ply": b"ply\nformat weird 1.0\nend_header\n",
    "novert.ply": b"ply\nformat ascii 1.0\nelement face 0\nproperty list uchar int vertex_indices\nend_header\n",
    "json.gltf": b"{not json",
    "ver.gltf": b'{"asset": {"version": "1.0"}}',
    "draco.gltf": b'{"asset": {"version": "2.0"}, "extensionsRequired": ["KHR_draco_mesh_compression"]}',
    "ver.glb": b"glTF" + (1).to_bytes(4, "little") + (20).to_bytes(4, "little") + b"\0" * 8,
    "junk.fbx": b"not an fbx file at all",
    "old.fbx": b"Kaydara FBX Binary  \x00\x1a\x00" + (6100).to_bytes(4, "little") + b"\0" * 200,
    "ascii.fbx": b"; FBX 7.4.0 project file\nFBXHeaderExtension: {\n}\n}\n",
}


@pytest.mark.parametrize("name", sorted(BROKEN))
def test_english_reader_errors(tmp_path, english, name):
    p = tmp_path / name
    p.write_bytes(BROKEN[name])
    try:
        scene = convert.load(str(p))
    except (ValueError, OSError) as e:
        _no_cyr([str(e)], name)
    else:
        _no_cyr(scene.warnings, name)


def test_english_3mf_errors(tmp_path, english):
    import zipfile
    p = tmp_path / "x.3mf"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("hello.txt", "no model")
    with pytest.raises(ValueError) as e:
        convert.load(str(p))
    _no_cyr([str(e.value)], "3mf")
    with pytest.raises(ValueError) as e:
        convert.load(str(tmp_path / "x.abc"))
    _no_cyr([str(e.value)], "ext")


def _cli(*args):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.pop("M2M_LANG", None)
    r = subprocess.run([sys.executable, "-m", "mesh2mesh", *args], cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding="utf-8")
    return r.returncode, r.stdout + r.stderr


def test_cli_english(tmp_path):
    from test_formats import rich_scene
    src = tmp_path / "a.3mf"
    formats.WRITERS[".3mf"](rich_scene(), str(src), scale=1.0, unit_m=1e-3)
    for args in (["--help"], ["--version"], [str(src), "--info"],
                 [str(src), "-o", str(tmp_path / "b.stl")], [str(tmp_path / "missing.obj"), "--info"],
                 [str(src)], [str(src), "-o", str(tmp_path / "b.xyz")]):
        code, out = _cli("--lang", "en", *args)
        assert out.strip(), args
        _no_cyr([out], " ".join(args))
    code, out = _cli("--lang", "ru", "--help")
    assert code == 0 and "входной файл" in out
    code, out = _cli("--lang=en", str(src), "--info")
    assert code == 0 and "units: mm" in out


# --- window ------------------------------------------------------------------------------------

def test_english_window_reports(english):
    from test_formats import CANISTER
    from test_textures import tex_scene
    from mesh2mesh.model import Texture
    from mesh2mesh_gui import display, report, shading, teximage
    scene = convert.load(CANISTER)
    d = display.build(scene)
    lines = report.file_summary(scene, CANISTER, True, d)
    assert lines == ["3MF: units from the file — mm.",
                     "2 meshes, 2 instances, 23 512 triangles, 2 materials."]
    odd = tex_scene(Texture("paint.tga", bytes(18) + b"\xff" * 12))
    odd.meshes[1].face_material[:] = 0
    od = display.build(odd)
    images = teximage.decode_all(od.textures)
    out = report.file_summary(odd, "x.obj", False, od, images) + report.texture_lines(od, images)
    out += [report.normals_note(x) for x in (d, od)] + [report.size_line(None, "mm"),
                                                        report.input_filter()]
    out += [t for _, t in report.losses_summary(["a"], ".stl") + report.losses_summary([], ".stl")]
    res = convert.Result("x.stl", warnings=["w"], mismatches=["m"])
    out += [t for _, t in report.result_summary(res, ["r"])]
    out += [report.unit(k) for k in convert.UNITS] + [tr(v) for v in report.FORMATS.values()]
    out += [tr(s) for pair in shading.MODES.values() for s in pair]
    out += [s for m in display.SMOOTHING for s in display.smoothing_texts(m)]
    _no_cyr(out, "window texts")
    for data in (b"", bytes(2) + b"\x05" + bytes(15), bytes(2) + b"\x02" + bytes(9) + b"\x01\0\x01\0\x07\0"):
        with pytest.raises(ValueError) as e:
            teximage.read_tga(data)
        _no_cyr([str(e.value)], "tga")


def _window_texts(w):
    """Every text and tooltip a user can see in the window (except the language names)."""
    from PySide6.QtWidgets import QAbstractButton, QComboBox, QGroupBox, QLabel, QTreeWidget
    out = [w.windowTitle(), w.view.hint.text()]
    for x in w.findChildren(QAbstractButton) + w.findChildren(QLabel):
        out += [x.text(), x.toolTip()]
    out += [x.title() for x in w.findChildren(QGroupBox)]
    for c in w.findChildren(QComboBox):
        if c is not w.lang:
            out += [c.itemText(i) for i in range(c.count())] + [c.toolTip()]
    for t in w.findChildren(QTreeWidget):
        out += [t.headerItem().text(i) for i in range(t.columnCount())] + [t.toolTip()]
    for a in w.view_bar.smooth.menu().actions():
        out += [a.text(), a.toolTip()]
    out.append(w.result.toPlainText())
    return out


def test_english_window_and_switch(tmp_path, english):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from test_formats import CANISTER
    from mesh2mesh_gui import window
    window.MainWindow.SETTINGS = ("clean_mesh", "mesh2mesh_pytest")
    w = window.MainWindow("en")
    try:
        w.settings.clear()
        _no_cyr(_window_texts(w), "empty window")
        w._loaded(window.load_model(CANISTER, "mm"))
        w._loss_running = w._current_loss_key()
        w._losses_done(convert.losses(w.model.scene, w.out_ext))
        w._saved(convert.convert(CANISTER, str(tmp_path / "c.stl"), scene=w.model.scene))
        w._gl_failed("GLError: test")
        app.processEvents()
        texts = _window_texts(w)
        assert w.save_btn.text() == "Save as 3MF" and w.lang.currentData() == "en"
        _no_cyr(texts, "loaded window")
        w.lang.setCurrentIndex(w.lang.findData("ru"))
        assert i18n.get_lang() == "ru" and w.save_btn.text().startswith("Сохранить как")
        assert w.settings.value("lang") == "ru" and "Габарит" in w.size_label.text()
        w.lang.setCurrentIndex(w.lang.findData("en"))
        _no_cyr(_window_texts(w), "switched back to English")
    finally:
        w.settings.clear()
        w.close()
