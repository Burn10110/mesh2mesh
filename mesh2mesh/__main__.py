"""CLI: python -m mesh2mesh in.3mf -o out.obj [--units mm] [--in-units mm] [--scale 10]
[--rotate x90] [--ascii]      |      python -m mesh2mesh in.3mf --info

Converts between mesh formats, prints every piece of data the target cannot hold, then reads the
result back with its own reader and compares triangles, bounding box, area and volume.
Exit code 1 if that read-back check fails, 2 on bad input.
"""

import argparse
import os
import sys

from . import __url__, __version__, convert, formats, geom, i18n
from .i18n import tr


def _utf8_output():
    """Redirected to a file/pipe/NUL, Python writes in the ANSI code page and Cyrillic report
    text raises UnicodeEncodeError (NUL even claims to be a tty). A real console already is
    UTF-8 and stays untouched."""
    for s in (sys.stdout, sys.stderr):
        if (s is not None and hasattr(s, "reconfigure")
                and (s.encoding or "").lower().replace("-", "") != "utf8"):
            s.reconfigure(encoding="utf-8", errors="replace")


def main(argv=None):
    _utf8_output()
    argv = sys.argv[1:] if argv is None else list(argv)
    lang = i18n.pre_parse_lang(argv)
    if lang:
        i18n.set_lang(lang)                     # before argparse: --help is translated too
    prog = "mesh2mesh-cli" if getattr(sys, "frozen", False) else "mesh2mesh"  # exe vs -m
    ap = argparse.ArgumentParser(
        prog=prog, description=tr("Конвертирует mesh-форматы, перечисляет всё, что целевой формат "
                                  "не может хранить, затем читает результат своим ридером и "
                                  "сверяет треугольники, габарит, площадь и объём. Код выхода 1 — "
                                  "проверка чтением не прошла, 2 — неверный ввод."))
    ap.add_argument("--version", action="version",
                    version=tr("{prog} {version} · автор: {url}", prog=prog, version=__version__,
                               url=__url__))
    ap.add_argument("input", help=tr("входной файл: {exts}", exts=", ".join(sorted(formats.READERS))))
    ap.add_argument("-o", "--out",
                    help=tr("выходной файл: {exts}", exts=", ".join(sorted(formats.WRITERS))))
    ap.add_argument("--info", action="store_true", help=tr("только показать содержимое файла"))
    ap.add_argument("--units", choices=convert.UNITS,
                    help=tr("единицы координат в OBJ/STL на выходе (по умолчанию — как у исходника)"))
    ap.add_argument("--in-units", choices=convert.UNITS, default="mm",
                    help=tr("единицы исходника без единиц в формате (OBJ/STL), по умолчанию mm"))
    ap.add_argument("--scale", type=float, default=1.0,
                    help=tr("умножить модель (от начала координат)"))
    ap.add_argument("--rotate", default="",
                    help=tr("четверть-обороты вокруг мировых осей слева направо: x90 или x-90,z180"))
    ap.add_argument("--ascii", "--stl-ascii", dest="ascii", action="store_true",
                    help=tr("STL/PLY текстом (STL так сохраняет имена объектов), иначе binary"))
    ap.add_argument("--lang", choices=i18n.LANGS,
                    help=tr("язык сообщений: ru, en (по умолчанию — язык Windows или M2M_LANG)"))
    a = ap.parse_args(argv)

    try:
        rotation = geom.rotation_matrix(a.rotate)
    except ValueError as e:
        ap.error(str(e))
    if a.scale <= 0:
        ap.error(tr("--scale должен быть положительным"))
    if not os.path.isfile(a.input):
        ap.error(tr("нет файла {path}", path=a.input))

    try:
        if a.info:
            scene = convert.transform(convert.load(a.input, a.in_units), a.scale, rotation)
            print(f"<- {a.input}")
            for s in convert.describe(scene, a.units):
                print("  ", s)
            for _, _, path, depth in scene.walk():
                print("   " + "  " * depth + path[-1])
            for w in scene.warnings:
                print("WARNING:", w)
            return 0
        if not a.out:
            ap.error(tr("укажите -o выходной файл или --info"))
        out = os.path.abspath(a.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        out_ext = os.path.splitext(out)[1].lower()
        opts = {"binary": not a.ascii} if out_ext in formats.TEXT_OPTION else {}
        res = convert.convert(os.path.abspath(a.input), out, a.units, a.in_units, a.scale,
                              rotation, **opts)
    except (ValueError, OSError) as e:
        print("ERROR:", e, file=sys.stderr)
        return 2
    for line in res.lines:
        print(line)
    for w in res.warnings:
        print("WARNING:", w)
    for m in res.mismatches:
        print("MISMATCH:", m)
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
