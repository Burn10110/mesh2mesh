"""Plain-language texts for the window (current language, mesh2mesh.i18n): file summary, size,
losses, save result.

No Qt here: tested offline (tests/test_gui_report.py).
"""

import os

import numpy as np

from mesh2mesh import convert, formats
from mesh2mesh.i18n import N_, tr

UNIT_NAMES = {"um": N_("мкм"), "mm": N_("мм"), "cm": N_("см"), "m": N_("м"), "in": N_("дюйм"),
              "ft": N_("фут")}

# output formats in the order the window offers them: ext -> label (tr() it to show)
FORMATS = {".3mf": N_("3MF — 3D-печать, слайсеры"), ".stl": N_("STL — 3D-печать, CAD"),
           ".obj": N_("OBJ — универсальный"), ".fbx": N_("FBX — Maya, 3ds Max, движки"),
           ".glb": N_("GLB — веб, движки (один файл)"),
           ".gltf": N_("glTF — веб, движки (.gltf + .bin)"), ".ply": N_("PLY — сканы, MeshLab")}

OK, WARN, ERROR = "ok", "warn", "error"


def unit(key):
    """'mm' -> 'мм' / 'mm' in the current language; unknown keys as they are."""
    return tr(UNIT_NAMES[key]) if key in UNIT_NAMES else key


def input_filter():
    exts = sorted(formats.READERS)
    return (tr("Модели ({patterns})", patterns=" ".join(f"*{e}" for e in exts)) + ";;"
            + ";;".join(f"{e[1:].upper()} (*{e})" for e in exts))


def _singular(n):
    """1, 21, 101… take the singular in Russian (not 11)."""
    n = abs(int(n))
    return n % 10 == 1 and n % 100 != 11


def plural(n, forms):
    """'n word' with the word agreeing with n. forms: N_() key 'one|few|many' (Russian); its
    English entry is 'one|other'."""
    n = abs(int(n))
    words = tr(forms).split("|")
    if len(words) == 2:
        word = words[0] if n == 1 else words[1]
    elif _singular(n):
        word = words[0]
    elif 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        word = words[1]
    else:
        word = words[2]
    return f"{fmt_int(n)} {word}"


def fmt_int(n):
    return f"{int(n):,}".replace(",", " ")


def fmt_len(x):
    """Length for people: whole numbers from 100 up (spaced thousands), 3-4 digits below."""
    x = float(x)
    if abs(x) >= 100:
        return f"{x:,.0f}".replace(",", " ")
    if x == 0:
        return "0"
    return f"{x:.4g}"


def fmt_scale(x):
    """Scale factor as typed by people: 1, 10, 0.001, 25.4, 1e-07."""
    return f"{float(x):.10g}"


def parse_scale(text):
    """'25,4' / '1e3' / '×10' -> float > 0, or None."""
    t = text.strip().replace(",", ".").lstrip("×xX* ").replace(" ", "")
    try:
        v = float(t)
    except ValueError:
        return None
    return v if np.isfinite(v) and v > 0 else None


def size_in(disp, unit_m, out_units, scale=1.0, rotation=None):
    """Bounding box size (x, y, z) of the result in `out_units`; None when there is no geometry."""
    if disp is None or disp.empty:
        return None
    v = disp.verts
    if rotation is not None:
        v = v @ np.asarray(rotation, dtype=float).T
    return (v.max(0) - v.min(0)) * scale * unit_m / convert.UNITS[out_units]


def size_line(size, units):
    if size is None:
        return tr("В файле нет геометрии")
    return tr("Габарит: {size} {unit}", size=" × ".join(fmt_len(s) for s in size), unit=unit(units))


def file_summary(scene, path, has_units, disp, images=None):
    """Lines under the file name: format, units, contents, textures in the view (images:
    teximage.Image per disp.textures; None = not decoded)."""
    fmt = os.path.splitext(path)[1][1:].upper()
    lines = []
    if has_units:
        lines.append(tr("{fmt}: единицы из файла — {unit}.", fmt=fmt,
                        unit=unit(convert.unit_name(scene.unit_m))))
    else:
        lines.append(tr("{fmt}: единиц в файле нет — выберите, в чём записаны числа.", fmt=fmt))
    lines.append(", ".join((plural(len(scene.meshes), N_("меш|меша|мешей")),
                            plural(len(disp.part_nodes), N_("экземпляр|экземпляра|экземпляров")),
                            plural(len(disp.tris), N_("треугольник|треугольника|треугольников")),
                            plural(len(scene.materials), N_("материал|материала|материалов"))))
                 + ".")
    return lines + texture_lines(disp, images)


def texture_lines(disp, images=None):
    """What the view shows of the base colour maps (the file keeps them all regardless)."""
    if not disp.textures and not disp.no_uv_textured:
        return []
    lines = []
    if disp.textures:
        lines.append(tr("{textures} — видны в режиме «Материал».",
                        textures=plural(len(disp.textures), N_("текстура|текстуры|текстур"))))
    bad = [im for im in images or () if im.pixels is None]
    if bad:
        shown = ", ".join(f"{im.name} ({im.reason})" for im in bad[:3])
        more = tr(" и ещё {n}", n=len(bad) - 3) if len(bad) > 3 else ""
        lines.append(tr("Не показаны в 3D-виде: {shown}{more} — там цвет материала; в "
                        "сохранённый файл переносятся.", shown=shown, more=more))
    if disp.no_uv_textured:
        lines.append(tr("У {tris} есть текстура, но нет UV — показаны цветом материала.",
                        tris=plural(disp.no_uv_textured,
                                    N_("треугольника|треугольников|треугольников"))))
    return lines


def normals_note(disp):
    """Tooltip line of the smoothing menu: how many meshes have no normals in the file."""
    without, total = disp.meshes_without_normals
    if not total:
        return ""
    if not without:
        return tr("В этом файле нормали есть у всех мешей.")
    if without == total:
        return tr("В этом файле нормалей нет — «Из файла» показывает плоско.")
    return tr("Без нормалей в файле: {n} из {total} — они плоские в «Из файла».",
              n=fmt_int(without), total=plural(total, N_("меша|мешей|мешей")))


def losses_summary(losses, ext):
    """(level, text) lines for the 'what will be lost' box."""
    name = ext[1:].upper()
    if not losses:
        return [(OK, tr("{name} вместит всё, что прочитано из файла.", name=name))]
    return [(WARN, tr("{name} не вместит — будет отброшено или изменено:", name=name))] + \
           [(WARN, w) for w in losses]


def result_summary(res, reader_warnings=()):
    """Save result as (level, text) lines, most important first."""
    out = []
    name = os.path.basename(res.out_path)
    if res.ok:
        out.append((OK, tr("Сохранено: {name}. Проверка чтением пройдена — геометрия совпадает с "
                           "исходной.", name=name)))
    else:
        out.append((ERROR, tr("Файл {name} записан, но проверка чтением не прошла: записанное не "
                              "совпадает с исходной моделью. Не используйте его — пришлите "
                              "исходный файл разработчику.", name=name)))
        out += [(ERROR, tr("Расхождение: {m}", m=m)) for m in res.mismatches]
    reader = set(reader_warnings)
    lost = [w for w in res.warnings if w not in reader]
    if lost:
        out.append((WARN, tr("Не вошло в файл ({n}):", n=len(lost))))
        out += [(WARN, w) for w in lost]
    else:
        out.append((OK, tr("Всё прочитанное перенесено без потерь.")))
    out += [(WARN, tr("При чтении исходника: {w}", w=w)) for w in reader_warnings]
    return out


def default_out_path(src, ext, out_dir=None):
    stem = os.path.splitext(os.path.basename(src))[0]
    return os.path.join(out_dir or os.path.dirname(src), stem + ext)


def fix_out_path(path, ext):
    """User-typed save path -> (path, ext). A known writable extension the user typed wins
    (it becomes the format); anything else gets the chosen format's extension appended."""
    typed = os.path.splitext(path)[1].lower()
    if typed in formats.WRITERS:
        return path, typed
    return path + ext, ext
