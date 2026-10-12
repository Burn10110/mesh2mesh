"""Pipeline: read -> (scale / rotate) -> write -> read back with our own reader and compare."""

import copy
import os
import tempfile
from dataclasses import dataclass, field

import numpy as np

from . import formats, geom

UNITS = {"um": 1e-6, "mm": 1e-3, "cm": 1e-2, "m": 1.0, "in": 0.0254, "ft": 0.3048}


def unit_name(unit_m):
    for k, v in UNITS.items():
        if np.isclose(v, unit_m, rtol=1e-9):
            return k
    return f"{unit_m:g} м"


@dataclass
class Result:
    out_path: str
    lines: list = field(default_factory=list)      # report
    warnings: list = field(default_factory=list)   # everything lost on the way
    mismatches: list = field(default_factory=list)  # read-back differences

    @property
    def ok(self):
        return not self.mismatches


def _ext(path, what):
    ext = os.path.splitext(path)[1].lower()
    table = formats.READERS if what == "read" else formats.WRITERS
    if ext not in table:
        verb = "чтения" if what == "read" else "записи"
        raise ValueError(f"формат {ext or '(без расширения)'} не поддерживается для {verb}; "
                         f"есть: {', '.join(sorted(table))}")
    return ext


def load(path, in_units="mm"):
    """Read any supported file. Formats without units get `in_units` (with a note)."""
    scene = formats.READERS[_ext(path, "read")](path)
    if scene.unit_m is None:
        scene.unit_m = UNITS[in_units]
        scene.warn(f"в файле нет единиц — принято: {in_units}")
    return scene


def transform(scene, scale=1.0, rotation=None):
    """Scale about the origin and rotate (proper rotation) every root node, in place."""
    if scale <= 0:
        raise ValueError("масштаб должен быть положительным")
    m = np.eye(4)
    m[:3, :3] = (np.eye(3) if rotation is None else np.asarray(rotation, float)) * scale
    for node in scene.nodes:
        node.matrix = m @ node.matrix
    return scene


def describe(scene, unit=None):
    """Human-readable summary lines of a scene."""
    unit = unit or unit_name(scene.unit_m)
    n_inst = sum(1 for _ in scene.instances())
    depth = max((d for *_, d in scene.walk()), default=0)
    st = geom.stats(scene)
    lines = [f"единицы: {unit_name(scene.unit_m)}; мешей {len(scene.meshes)}, "
             f"экземпляров {n_inst}, глубина иерархии {depth + 1 if scene.nodes else 0}, "
             f"материалов {len(scene.materials)}"]
    lines += st.lines(UNITS.get(unit, scene.unit_m), unit)
    return lines


def out_units(scene, out_ext, units=None):
    """Output unit key for writing `scene` as out_ext -> (key, warnings). Default: the source's
    own unit (mm when it is not one of UNITS); formats with a fixed unit override the request."""
    warns = []
    fixed = formats.FIXED_UNITS.get(out_ext)
    if fixed:
        if units and units != fixed:
            warns.append(f"{out_ext[1:]}: единицы по спецификации всегда {fixed} — "
                         f"--units {units} не применён")
        units = fixed
    units = units or unit_name(scene.unit_m)
    return (units if units in UNITS else "mm"), warns


def _write(scene, dst, out_ext, units, write_opts):
    k = scene.unit_m / UNITS[units]
    return formats.WRITERS[out_ext](scene, dst, scale=k, unit_m=UNITS[units], **write_opts)


def losses(scene, out_ext, units=None, scale=1.0, rotation=None, **write_opts):
    """What writing `scene` (not modified) as out_ext would drop: a trial write into a temporary
    folder, no read-back. Reader warnings are not included."""
    out_ext = out_ext.lower()
    if out_ext not in formats.WRITERS:
        raise ValueError(f"формат {out_ext} не поддерживается для записи")
    scene = transform(copy.deepcopy(scene), scale, rotation)
    scene.warnings = []
    units, warns = out_units(scene, out_ext, units)
    with tempfile.TemporaryDirectory(prefix="mesh2mesh_") as tmp:
        warns += _write(scene, os.path.join(tmp, "trial" + out_ext), out_ext, units, write_opts)
    return warns


def convert(src, dst, units=None, in_units="mm", scale=1.0, rotation=None, scene=None,
            **write_opts):
    """src -> dst. units: output units (written into 3MF; for OBJ/STL/PLY only the scale of the
    numbers); default = the source's own. glTF is always metres. scene: `load(src)` already done
    (not modified). Returns Result; never raises on read-back mismatch."""
    out_ext = _ext(dst, "write")
    scene = load(src, in_units) if scene is None else copy.deepcopy(scene)
    transform(scene, scale, rotation)
    res = Result(dst)
    units, res.warnings = out_units(scene, out_ext, units)
    res.lines.append(f"<- {src}")
    res.lines += ["   " + s for s in describe(scene, units)]
    res.warnings += scene.warnings
    res.warnings += _write(scene, dst, out_ext, units, write_opts)

    back = load(dst, units)
    res.warnings += [f"(read-back) {w}" for w in back.warnings
                     if not w.startswith("в файле нет единиц")]
    res.mismatches = geom.compare(geom.stats(scene), geom.stats(back),
                                  float32=out_ext in formats.FLOAT32)
    res.lines.append(f"-> {dst}")
    res.lines += ["   " + s for s in describe(back, units)]
    res.lines.append("проверка чтением: " + ("OK" if res.ok else "РАСХОЖДЕНИЕ"))
    return res
