"""Build the delivery: dist/mesh2mesh/ (mesh2mesh.exe + mesh2mesh-cli.exe) and
dist/mesh2mesh-<ver>.zip.

    .venv/Scripts/python packaging/build.py [--no-test] [--no-zip] [--test-only]

Steps: version resources -> PyInstaller (packaging/mesh2mesh.spec) -> README_EN/RU.txt, LICENSE.txt,
licenses/ -> self-test of the built programs (not the sources) -> zip. Exit 1 if anything fails.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from mesh2mesh import __url__, __version__  # noqa: E402

DIST = os.path.join(ROOT, "dist")
APP = os.path.join(DIST, "mesh2mesh")
WORK = os.path.join(HERE, "_build")
TEST_OUT = os.path.join(ROOT, "tools", "_out", "pkg")
MIXAMO = os.path.join(ROOT, "samples", "fbx", "mixamo_bin_none.fbx")    # 5 PNG maps
CANISTER = os.path.join(ROOT, "samples", "Mata_Canister_CAP_Scaled.3mf")  # 2 parts, colors, mm

VERSION_RC = """VSVersionInfo(
  ffi=FixedFileInfo(filevers={t}, prodvers={t}, mask=0x3f, flags=0x0, OS=0x40004,
                    fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[StringFileInfo([StringTable('041904B0', [
      StringStruct('FileDescription', '{desc}'),
      StringStruct('FileVersion', '{v}'),
      StringStruct('ProductName', 'mesh2mesh'),
      StringStruct('ProductVersion', '{v}'),
      StringStruct('Comments', '{url}'),
      StringStruct('OriginalFilename', '{name}.exe')])]),
    VarFileInfo([VarStruct('Translation', [0x0419, 1200])])])
"""

# 1x1 red GIF89a: Qt can read GIF but not write it
TINY_GIF = (b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\x00\x00\x00\x00\x00"
            b",\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;")


def write_version_files():
    os.makedirs(WORK, exist_ok=True)
    nums = [int(x) for x in __version__.split(".")] + [0] * 4
    t = tuple(nums[:4])
    for name, desc in (("mesh2mesh", "Mesh format converter (window)"),
                       ("mesh2mesh-cli", "Mesh format converter (command line)")):
        with open(os.path.join(WORK, f"{name}_version.txt"), "w", encoding="utf-8") as f:
            f.write(VERSION_RC.format(t=t, v=__version__, desc=desc, name=name, url=__url__))


def write_icon():
    """The exe icon from the same drawing as the window's (mesh2mesh_gui/icon.py)."""
    from PySide6.QtGui import QGuiApplication
    from mesh2mesh_gui import icon
    app = QGuiApplication.instance() or QGuiApplication(["build", "-platform", "offscreen"])
    icon.write_ico(os.path.join(WORK, "mesh2mesh.ico"))
    return app


def run_pyinstaller():
    cmd = [sys.executable, "-m", "PyInstaller", os.path.join(HERE, "mesh2mesh.spec"),
           "--noconfirm", "--clean", "--distpath", DIST, "--workpath", WORK]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def write_readme():
    for name in ("README_EN.txt", "README_RU.txt"):
        with open(os.path.join(HERE, name), encoding="utf-8") as f:
            text = f.read().replace("{version}", __version__).replace("{url}", __url__)
        # UTF-8 with BOM: Notepad on any Windows opens it as Unicode
        with open(os.path.join(APP, name), "w", encoding="utf-8-sig", newline="\r\n") as f:
            f.write(text)


# Third-party components inside dist/: (name, pip distribution, license, home, license texts).
# A text is "dist:<file in the distribution's .dist-info/licenses>", "python:<file in the
# Python install>" or a file in packaging/licenses/ (texts the wheels do not carry).
THIRD_PARTY = [
    ("Python", None, "PSF-2.0", "https://www.python.org/", ["python:LICENSE.txt"]),
    ("Qt for Python (PySide6, shiboken6) and Qt", "PySide6", "LGPL-3.0",
     "https://www.qt.io/qt-for-python  (Qt sources: https://download.qt.io/official_releases/qt/)",
     ["LGPL-3.0.txt", "GPL-3.0.txt"]),
    ("NumPy", "numpy", "BSD-3-Clause and others (see the file)", "https://numpy.org/",
     ["dist:LICENSE.txt"]),
    ("pyqtgraph", "pyqtgraph", "MIT", "https://www.pyqtgraph.org/", ["dist:LICENSE.txt"]),
    ("PyOpenGL", "PyOpenGL", "BSD-3-Clause", "https://github.com/mcfletch/pyopengl",
     ["PyOpenGL.txt"]),
    ("PyInstaller bootloader", "pyinstaller", "GPL-2.0 with the bootloader exception",
     "https://pyinstaller.org/", ["dist:COPYING.txt"]),
]

THIRD_PARTY_HEAD = """mesh2mesh {version} - third-party components
==================================================

mesh2mesh itself is MIT-licensed: LICENSE.txt next to the programs.
The programs bundle the components below, each under its own license; the license texts
are in this folder.

Qt and Qt for Python are used under the GNU LGPL v3 as separate dynamic libraries
(_internal\\PySide6\\*.dll, *.pyd and _internal\\shiboken6): you may replace them with your
own compatible builds.
Their source code is available from the Qt project (links below).

"""


def _license_source(text, dist):
    from importlib import metadata
    if text.startswith("python:"):
        return os.path.join(sys.base_prefix, text[7:])
    if text.startswith("dist:"):
        files = metadata.distribution(dist).files or []
        name = text[5:]
        hits = [f for f in files if f.name == name and "licenses" in f.parts]
        if not hits:
            raise FileNotFoundError(f"{dist}: no {name} in its .dist-info/licenses")
        return str(hits[0].locate())
    return os.path.join(HERE, "licenses", text)


def write_licenses():
    """LICENSE.txt (ours) + licenses/ (THIRD-PARTY.txt and every component's license text)."""
    from importlib import metadata
    shutil.copy(os.path.join(ROOT, "LICENSE"), os.path.join(APP, "LICENSE.txt"))
    out = os.path.join(APP, "licenses")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    lines = [THIRD_PARTY_HEAD.format(version=__version__)]
    for name, dist, lic, home, texts in THIRD_PARTY:
        ver = (metadata.version(dist) if dist
               else ".".join(map(str, sys.version_info[:3])))
        files = []
        for text in texts:
            src = _license_source(text, dist)
            stem = (dist or name).split()[0]
            dst = os.path.basename(text.split(":")[-1])
            dst = dst if not text.startswith(("dist:", "python:")) else f"{stem}-{dst}"
            shutil.copy(src, os.path.join(out, dst))
            files.append(dst)
        lines.append(f"{name} {ver}\n   License: {lic}\n   Home: {home}\n"
                     f"   Text: {', '.join(files)}\n\n")
    with open(os.path.join(out, "THIRD-PARTY.txt"), "w", encoding="utf-8", newline="\r\n") as f:
        f.write("".join(lines))


def make_texture_obj(folder):
    """OBJ with four quads, each mapped with a picture in a format the Qt image plugins decode
    (JPEG, TIFF, WebP, GIF): in the exe they show only if the imageformats plugins came along."""
    from PySide6.QtGui import QColor, QImage
    os.makedirs(folder, exist_ok=True)
    maps = []
    for i, (ext, color) in enumerate((("jpg", "#d04020"), ("tif", "#20a040"),
                                      ("webp", "#2050d0"))):
        img = QImage(64, 64, QImage.Format_RGB32)
        img.fill(QColor(color))
        name = f"карта_{ext}.{ext}"
        if not img.save(os.path.join(folder, name)):
            raise RuntimeError(f"Qt in the build venv cannot write {ext}")
        maps.append(name)
    with open(os.path.join(folder, "карта_gif.gif"), "wb") as f:
        f.write(TINY_GIF)
    maps.append("карта_gif.gif")
    mtl = [f"newmtl m{i}\nKd 1 1 1\nmap_Kd {m}\n" for i, m in enumerate(maps)]
    obj = ["mtllib текстуры.mtl"]
    for i in range(len(maps)):
        x = i * 12
        obj += [f"v {x} 0 0", f"v {x + 10} 0 0", f"v {x + 10} 10 0", f"v {x} 10 0"]
    obj += ["vt 0 0", "vt 1 0", "vt 1 1", "vt 0 1", "vn 0 0 1", "o карты"]
    for i in range(len(maps)):
        b = i * 4
        obj += [f"usemtl m{i}",
                "f " + " ".join(f"{b + k}/{k}/1" for k in range(1, 5))]
    with open(os.path.join(folder, "текстуры.mtl"), "w", encoding="utf-8") as f:
        f.write("\n".join(mtl))
    path = os.path.join(folder, "текстуры.obj")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(obj) + "\n")
    return path, len(maps)


def self_test():
    """Run the BUILT programs from a copy in a path with Cyrillic and spaces, with a bare
    environment (no Python on PATH, no PYTHON* vars): close to a clean machine."""
    ok = True
    base = os.path.join(TEST_OUT, "Проверка сборки с пробелами")
    shutil.rmtree(base, ignore_errors=True)
    app = os.path.join(base, "mesh2mesh")
    shutil.copytree(APP, app)
    mixamo = os.path.join(base, "персонаж тест.fbx")
    canister = os.path.join(base, "канистра тест.3mf")
    shutil.copy(MIXAMO, mixamo)
    shutil.copy(CANISTER, canister)
    texobj, n_maps = make_texture_obj(os.path.join(base, "карты"))
    out = os.path.join(base, "выход")
    env = {k: v for k, v in os.environ.items()
           if k.upper() in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "USERPROFILE", "LOCALAPPDATA",
                            "APPDATA", "HOMEDRIVE", "HOMEPATH", "COMPUTERNAME", "USERNAME")}
    env["PATH"] = os.path.join(os.environ.get("SYSTEMROOT", r"C:\Windows"), "System32")
    cli = os.path.join(app, "mesh2mesh-cli.exe")
    gui = os.path.join(app, "mesh2mesh.exe")

    def check(cond, msg):
        nonlocal ok
        print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
        ok &= bool(cond)

    # licenses travel with the programs: ours and every bundled component's text
    lic = os.path.join(app, "licenses")
    third = os.path.join(lic, "THIRD-PARTY.txt")
    listed = open(third, encoding="utf-8").read() if os.path.exists(third) else ""
    named = [t.strip() for ln in listed.splitlines() if ln.strip().startswith("Text:")
             for t in ln.split(":", 1)[1].split(",")]
    check(os.path.exists(os.path.join(app, "LICENSE.txt")) and len(named) >= len(THIRD_PARTY)
          and all(os.path.getsize(os.path.join(lic, n)) > 500 for n in named),
          f"LICENSE.txt + licenses/: {len(named)} license texts")

    def run_cli(*args, expect=0):
        t = time.time()
        r = subprocess.run([cli, *args], env=env, capture_output=True, encoding="utf-8",
                           errors="replace", timeout=600)
        print(r.stdout[-800:], r.stderr[-800:], sep="")
        check(r.returncode == expect, f"mesh2mesh-cli {' '.join(os.path.basename(a) for a in args)}"
                                      f" -> exit {r.returncode} ({time.time() - t:.1f} s)")
        return r

    # command line: every writer at least once, textures carried to files next to OBJ
    r = run_cli("--version")
    check(__version__ in r.stdout and __url__ in r.stdout,
          f"--version prints {__version__} and the author link")
    run_cli(canister, "--info")
    for ext in ("glb", "fbx", "obj", "stl", "ply", "3mf"):
        run_cli(canister, "-o", os.path.join(out, f"канистра.{ext}"))
    run_cli(mixamo, "-o", os.path.join(out, "персонаж", "персонаж.obj"))
    pngs = [n for n in os.listdir(os.path.join(out, "персонаж")) if n.lower().endswith(".png")] \
        if os.path.isdir(os.path.join(out, "персонаж")) else []
    check(len(pngs) == 5, f"Mixamo -> OBJ: 5 texture files written next to it ({len(pngs)})")
    run_cli(os.path.join(base, "нет такого.fbx"), "-o", os.path.join(out, "x.obj"), expect=2)

    # window: model loads, losses computed, view draws, textures decoded by the Qt plugins
    # (in English the smoke test also checks the main texts have no Cyrillic)
    for model, maps, tag, lang in ((mixamo, 5, "mixamo", "ru"), (canister, 0, "canister", "en"),
                                   (canister, 0, "canister_ru", "ru"),
                                   (texobj, n_maps, "texformats", "en")):
        png = os.path.join(TEST_OUT, f"exe_smoke_{tag}.png")
        t = time.time()
        r = subprocess.run([gui, "--lang", lang, "--smoke-test", model, png], env=env, timeout=300)
        txt = open(png + ".txt", encoding="utf-8").read() if os.path.exists(png + ".txt") else ""
        print(txt)
        check(r.returncode == 0, f"mesh2mesh.exe --lang {lang} --smoke-test {tag}: exit "
                                 f"{r.returncode} ({time.time() - t:.1f} s), screenshot {png}")
        check(f"language: {lang}" in txt, f"{tag}: window in {lang}")
        check(f"textures shown: {maps}/{maps}" in txt, f"{tag}: all {maps} texture maps shown")
    r = subprocess.run([cli, "--lang", "en", "--help"], env=env, capture_output=True,
                       encoding="utf-8", errors="replace", timeout=60)
    check(r.returncode == 0 and not any("Ѐ" <= c <= "ӿ" for c in r.stdout),
          "mesh2mesh-cli.exe --lang en --help: English")

    # no usable OpenGL (RDP/VM): the offscreen platform gives no GL context; the window must
    # survive with a notice (exit 3 = view not drawn) and still load the model.
    png = os.path.join(TEST_OUT, "exe_smoke_nogl.png")
    r = subprocess.run([gui, "--smoke-test", canister, png],
                       env=dict(env, QT_QPA_PLATFORM="offscreen"), timeout=300)
    txt = open(png + ".txt", encoding="utf-8").read() if os.path.exists(png + ".txt") else ""
    check(r.returncode == 3 and "loaded: True" in txt and "gl error: None" not in txt,
          f"no OpenGL: window survives, model loaded (exit {r.returncode}), screenshot {png}")
    return ok


def make_zip():
    path = os.path.join(DIST, f"mesh2mesh-{__version__}.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for d, _, files in os.walk(APP):
            for n in files:
                full = os.path.join(d, n)
                z.write(full, os.path.relpath(full, DIST))
    return path


def folder_size(path):
    return sum(os.path.getsize(os.path.join(d, n)) for d, _, fs in os.walk(path) for n in fs)


def main():
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--test-only", action="store_true", help="self-test the existing dist/")
    a = ap.parse_args()

    if not a.test_only:
        write_version_files()
        _app = write_icon()  # noqa: F841 - keep the Qt app alive for make_texture_obj
        run_pyinstaller()
        write_readme()
        write_licenses()
    print(f"built {APP}: {folder_size(APP) / 2**20:.0f} MB", flush=True)
    if not a.no_test and not self_test():
        print("SELF-TEST FAILED")
        return 1
    if not a.no_zip and not a.test_only:
        z = make_zip()
        print(f"-> {z}: {os.path.getsize(z) / 2**20:.0f} MB")
    print("BUILD OK", __version__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
