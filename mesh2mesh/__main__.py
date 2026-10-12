"""CLI: python -m mesh2mesh in.3mf -o out.obj [--units mm] [--in-units mm] [--scale 10]
[--rotate x90] [--ascii]      |      python -m mesh2mesh in.3mf --info

Converts between mesh formats, prints every piece of data the target cannot hold, then reads the
result back with its own reader and compares triangles, bounding box, area and volume.
Exit code 1 if that read-back check fails, 2 on bad input.
"""

import argparse
import os
import sys

from . import __url__, __version__, convert, formats, geom


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
    prog = "mesh2mesh-cli" if getattr(sys, "frozen", False) else "mesh2mesh"  # exe vs -m
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[1].strip())
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__} · автор: {__url__}")
    ap.add_argument("input", help=f"входной файл: {', '.join(sorted(formats.READERS))}")
    ap.add_argument("-o", "--out", help=f"выходной файл: {', '.join(sorted(formats.WRITERS))}")
    ap.add_argument("--info", action="store_true", help="только показать содержимое файла")
    ap.add_argument("--units", choices=convert.UNITS,
                    help="единицы координат в OBJ/STL на выходе (по умолчанию — как у исходника)")
    ap.add_argument("--in-units", choices=convert.UNITS, default="mm",
                    help="единицы исходника без единиц в формате (OBJ/STL), по умолчанию mm")
    ap.add_argument("--scale", type=float, default=1.0, help="умножить модель (от начала координат)")
    ap.add_argument("--rotate", default="",
                    help="четверть-обороты вокруг мировых осей слева направо: x90 или x-90,z180")
    ap.add_argument("--ascii", "--stl-ascii", dest="ascii", action="store_true",
                    help="STL/PLY текстом (STL так сохраняет имена объектов), иначе binary")
    a = ap.parse_args(argv)

    try:
        rotation = geom.rotation_matrix(a.rotate)
    except ValueError as e:
        ap.error(str(e))
    if a.scale <= 0:
        ap.error("--scale должен быть положительным")
    if not os.path.isfile(a.input):
        ap.error(f"нет файла {a.input}")

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
            ap.error("укажите -o выходной файл или --info")
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
