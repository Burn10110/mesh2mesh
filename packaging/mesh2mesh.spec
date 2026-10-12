# PyInstaller spec: one folder, two programs sharing it.
#   mesh2mesh.exe     - the window (no console)
#   mesh2mesh-cli.exe - the command line
# Build: .venv/Scripts/python packaging/build.py  (not pyinstaller directly: build.py also
# stamps the version, writes README, self-tests the built programs and zips).
import os

ROOT = os.path.dirname(SPECPATH)

# Qt modules pyqtgraph may import optionally, check tools (lib3mf, pytest) and stdlib bits we
# never use: keeps size down.
EXCLUDES = ["tkinter", "pydoc_data", "matplotlib", "scipy", "PyQt5", "PyQt6",
            "PySide2", "IPython", "lib3mf", "pytest", "_pytest",
            "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtNetwork",
            "PySide6.QtSvg", "PySide6.QtPdf", "PySide6.QtWebEngineCore", "PySide6.QtMultimedia",
            "PySide6.Qt3DCore", "PySide6.QtCharts", "PySide6.QtDataVisualization",
            "pyqtgraph.examples", "pyqtgraph.jupyter", "pyqtgraph.canvas",
            "pyqtgraph.flowchart", "pyqtgraph.console"]


def analysis(script):
    return Analysis([os.path.join(SPECPATH, script)], pathex=[ROOT], excludes=EXCLUDES,
                    noarchive=False)


gui_a = analysis("entry_gui.py")
cli_a = analysis("entry_cli.py")
# Files the hooks collect that this app never loads:
# - PyOpenGL's GLUT/GLE DLLs (need MSVCR90; Qt makes the GL context);
# - opengl32sw.dll: Qt's software GL; PyOpenGL keeps calling the system opengl32.dll, so the
#   view fails under it anyway; the window then shows a notice instead of the view;
# - virtual keyboard (drags in Qt Quick/QML), TUIO touch (drags in Network), Direct2D platform;
#   Qt translations other than ru/en;
# - image plugins for formats no model carries: PDF (drags in Qt Pdf), macOS icons, WAP bitmaps.
#   KEPT on purpose: imageformats qjpeg/qtiff/qwebp/qgif - textures in the view (PNG/BMP are
#   built into Qt; TGA has our own reader). build.py checks JPEG/TIFF/WebP/GIF maps show.
DROP_PREFIX = ("freeglut", "gle32", "gle64", "opengl32sw", "qt6pdf", "qpdf", "qicns", "qwbmp",
               "qt6quick", "qt6qml", "qt6virtualkeyboard", "qtvirtualkeyboardplugin",
               "qtuiotouchplugin", "qt6network", "qdirect2d")


def keep(entry):
    name = os.path.basename(entry[0]).lower()
    if name.startswith(DROP_PREFIX):
        return False
    if name.endswith(".qm") and not name.endswith(("_ru.qm", "_en.qm")):
        return False
    return True


for a in (gui_a, cli_a):
    a.binaries = [b for b in a.binaries if keep(b)]
    a.datas = [d for d in a.datas if keep(d)]


def exe(a, name, console):
    return EXE(PYZ(a.pure), a.scripts, [], exclude_binaries=True, name=name, console=console,
               upx=False, version=os.path.join(SPECPATH, "_build", f"{name}_version.txt"),
               icon=os.path.join(SPECPATH, "_build", "mesh2mesh.ico"))  # build.py draws it


gui_exe = exe(gui_a, "mesh2mesh", console=False)
cli_exe = exe(cli_a, "mesh2mesh-cli", console=True)

coll = COLLECT(gui_exe, cli_exe, gui_a.binaries, gui_a.datas, cli_a.binaries, cli_a.datas,
               upx=False, name="mesh2mesh")
