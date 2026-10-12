"""Plain-language (Russian) texts for the window: file summary, size, losses, save result.

No Qt here: tested offline (tests/test_gui_report.py).
"""

import os

import numpy as np

from mesh2mesh import convert, formats

UNIT_RU = {"um": "мкм", "mm": "мм", "cm": "см", "m": "м", "in": "дюйм", "ft": "фут"}

# output formats in the order the window offers them: ext -> label
FORMATS = {".3mf": "3MF — 3D-печать, слайсеры", ".stl": "STL — 3D-печать, CAD",
           ".obj": "OBJ — универсальный", ".fbx": "FBX — Maya, 3ds Max, движки",
           ".glb": "GLB — веб, движки (один файл)", ".gltf": "glTF — веб, движки (.gltf + .bin)",
           ".ply": "PLY — сканы, MeshLab"}
INPUT_FILTER = ("Модели (" + " ".join(f"*{e}" for e in sorted(formats.READERS)) + ");;"
                + ";;".join(f"{e[1:].upper()} (*{e})" for e in sorted(formats.READERS)))

OK, WARN, ERROR = "ok", "warn", "error"


def _singular(n):
    """1, 21, 101… take the singular in Russian (not 11)."""
    n = abs(int(n))
    return n % 10 == 1 and n % 100 != 11


def plural(n, one, few, many):
    n = abs(int(n))
    if _singular(n):
        word = one
    elif 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        word = few
    else:
        word = many
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
        return "В файле нет геометрии"
    return "Габарит: " + " × ".join(fmt_len(s) for s in size) + f" {UNIT_RU[units]}"


def file_summary(scene, path, has_units, disp, images=None):
    """Lines under the file name: format, units, contents, textures in the view (images:
    teximage.Image per disp.textures; None = not decoded)."""
    ext = os.path.splitext(path)[1].lower()
    lines = []
    if has_units:
        lines.append(f"{ext[1:].upper()}: единицы из файла — "
                     f"{UNIT_RU.get(convert.unit_name(scene.unit_m), convert.unit_name(scene.unit_m))}.")
    else:
        lines.append(f"{ext[1:].upper()}: единиц в файле нет — выберите, в чём записаны числа.")
    n_inst = len(disp.part_nodes)
    tris = len(disp.tris)
    text = (f"{plural(len(scene.meshes), 'меш', 'меша', 'мешей')}, "
            f"{plural(n_inst, 'экземпляр', 'экземпляра', 'экземпляров')}, "
            f"{plural(tris, 'треугольник', 'треугольника', 'треугольников')}, "
            f"{plural(len(scene.materials), 'материал', 'материала', 'материалов')}.")
    lines.append(text)
    return lines + texture_lines(disp, images)


def texture_lines(disp, images=None):
    """What the view shows of the base colour maps (the file keeps them all regardless)."""
    if not disp.textures and not disp.no_uv_textured:
        return []
    lines = []
    if disp.textures:
        lines.append(f"{plural(len(disp.textures), 'текстура', 'текстуры', 'текстур')} — "
                     "видны в режиме «Материал».")
    bad = [im for im in images or () if im.pixels is None]
    if bad:
        shown = ", ".join(f"{im.name} ({im.reason})" for im in bad[:3])
        more = f" и ещё {len(bad) - 3}" if len(bad) > 3 else ""
        lines.append(f"Не показаны в 3D-виде: {shown}{more} — там цвет материала; "
                     "в сохранённый файл переносятся.")
    if disp.no_uv_textured:
        lines.append(f"У {plural(disp.no_uv_textured, 'треугольника', 'треугольников', 'треугольников')}"
                     " есть текстура, но нет UV — показаны цветом материала.")
    return lines


def normals_note(disp):
    """Tooltip line of the smoothing menu: how many meshes have no normals in the file."""
    without, total = disp.meshes_without_normals
    if not total:
        return ""
    if not without:
        return "В этом файле нормали есть у всех мешей."
    if without == total:
        return "В этом файле нормалей нет — «Из файла» показывает плоско."
    return (f"Без нормалей в файле: {fmt_int(without)} из "
            f"{plural(total, 'меша', 'мешей', 'мешей')} — они плоские в «Из файла».")


def losses_summary(losses, ext):
    """(level, text) lines for the 'what will be lost' box."""
    name = ext[1:].upper()
    if not losses:
        return [(OK, f"{name} вместит всё, что прочитано из файла.")]
    return [(WARN, f"{name} не вместит — будет отброшено или изменено:")] + \
           [(WARN, w) for w in losses]


def result_summary(res, reader_warnings=()):
    """Save result as (level, text) lines, most important first."""
    out = []
    name = os.path.basename(res.out_path)
    if res.ok:
        out.append((OK, f"Сохранено: {name}. Проверка чтением пройдена — геометрия совпадает "
                        f"с исходной."))
    else:
        out.append((ERROR, f"Файл {name} записан, но проверка чтением не прошла: записанное не "
                           f"совпадает с исходной моделью. Не используйте его — пришлите "
                           f"исходный файл разработчику."))
        out += [(ERROR, "Расхождение: " + m) for m in res.mismatches]
    reader = set(reader_warnings)
    lost = [w for w in res.warnings if w not in reader]
    if lost:
        out.append((WARN, f"Не вошло в файл ({len(lost)}):"))
        out += [(WARN, w) for w in lost]
    else:
        out.append((OK, "Всё прочитанное перенесено без потерь."))
    out += [(WARN, "При чтении исходника: " + w) for w in reader_warnings]
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
