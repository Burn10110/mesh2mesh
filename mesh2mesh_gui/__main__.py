"""Window: python -m mesh2mesh_gui [model]   |   --smoke-test model out.png"""

import datetime
import os
import sys
import traceback

from PySide6.QtWidgets import QApplication  # before pyqtgraph: it picks the loaded Qt binding

LOG_LIMIT = 1_000_000  # bytes; the log is started afresh past this


def log_path():
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, "mesh2mesh", "mesh2mesh.log")


def _setup_log():
    """No console (pythonw, windowed exe): stdout/stderr are None and tracebacks vanish.
    Send them to a log file the user can hand over."""
    if sys.stdout is not None and sys.stderr is not None:
        return None
    path = log_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mode = "w" if os.path.exists(path) and os.path.getsize(path) > LOG_LIMIT else "a"
        f = open(path, mode, encoding="utf-8", buffering=1)
    except OSError:
        return None
    f.write(f"\n=== {datetime.datetime.now():%Y-%m-%d %H:%M:%S} start {sys.argv!r}\n")
    sys.stdout = sys.stdout or f
    sys.stderr = sys.stderr or f
    return path


def _install_excepthook(app, log):
    """An exception in a Qt slot would otherwise only print (or vanish): tell the user."""
    def hook(tp, value, tb):
        traceback.print_exception(tp, value, tb)
        from PySide6.QtWidgets import QMessageBox
        where = f"\n\nПодробности записаны в журнал:\n{log}" if log else ""
        QMessageBox.critical(app.activeWindow(), "Ошибка программы",
                             f"Что-то пошло не так: {tp.__name__}: {value}{where}")
    sys.excepthook = hook


def _install_translator(app):
    """Qt's own buttons (Yes/No, Cancel) in Russian, like the rest of the window."""
    from PySide6.QtCore import QLibraryInfo, QTranslator
    tr = QTranslator(app)
    if tr.load("qtbase_ru", QLibraryInfo.path(QLibraryInfo.TranslationsPath)):
        app.installTranslator(tr)
        return True
    print("qtbase_ru translation not found")
    return False


def _smoke_test(app, w, model, out_png):
    """--smoke-test model out.png: open, wait for the load and the losses box, grab the window,
    exit. Exit 0 only if the model loaded, the losses were computed and the 3D view drew
    something (checks the packaged exe: Qt plugins, OpenGL, readers, writers); 3 if not.
    Summary goes to out.png + '.txt'."""
    from PySide6.QtCore import QTimer
    state = {"t": 0}

    def poll():
        state["t"] += 1
        # wait for the load + losses, and >= 2.5 s for the view's "no GL context" watchdog
        waiting = w.job is not None or w.loss_job is not None or w.loss_timer.isActive()
        if (waiting or state["t"] < 25) and state["t"] < 1200:  # up to 2 min
            return
        timer.stop()
        app.processEvents()
        drawn = 0
        if w.view.gl_error is None:
            w.view.update()
            app.processEvents()
            img = w.view.grabFramebuffer()
            bg = img.pixelColor(2, 2)
            drawn = sum(img.pixelColor(x, y) != bg for x in range(0, img.width(), 7)
                        for y in range(0, img.height(), 7))
        app.processEvents()  # let a gl_failed notice replace the view before the screenshot
        w.grab().save(out_png)
        loaded = w.model is not None
        from PySide6.QtGui import QImageReader
        images = w.model.images if loaded else []
        shown = sum(im.pixels is not None for im in images)
        lines = [f"title: {w.windowTitle()}", f"loaded: {loaded}",
                 f"textures shown: {shown}/{len(images)}",
                 *(f"texture not shown: {im.name}: {im.reason}" for im in images
                   if im.pixels is None),
                 "image formats: " + " ".join(sorted(bytes(f).decode() for f in
                                                     QImageReader.supportedImageFormats())),
                 f"tree rows: {w.tree.topLevelItemCount()}", f"losses: {w.loss_key is not None}",
                 f"gl error: {w.view.gl_error}",
                 f"qt russian: {app.property('ru_translation')}",
                 f"icon sizes: {len(app.windowIcon().availableSizes())}",
                 f"view pixels drawn (1/49 sample): {drawn}"]
        with open(out_png + ".txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        ok = loaded and w.loss_key is not None and drawn > 20 and app.property("ru_translation")
        app.exit(0 if ok else 3)

    w.open_file(model)
    timer = QTimer(w)
    timer.timeout.connect(poll)
    timer.start(100)


def _taskbar_identity():
    """Own taskbar button with our icon (from sources it would group under python.exe)."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("mesh2mesh.converter")
        except (AttributeError, OSError):
            pass


def main():
    log = _setup_log()
    _taskbar_identity()
    app = QApplication(sys.argv)
    app.setApplicationName("mesh2mesh")
    from .icon import app_icon
    app.setWindowIcon(app_icon())
    app.setProperty("ru_translation", _install_translator(app))
    _install_excepthook(app, log)
    from .window import MainWindow
    w = MainWindow()
    w.show()
    args = sys.argv[1:]
    if len(args) == 3 and args[0] == "--smoke-test":
        _smoke_test(app, w, os.path.abspath(args[1]), os.path.abspath(args[2]))
    elif args:
        w.open_file(os.path.abspath(args[0]))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
