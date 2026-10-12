"""PLY (Stanford polygon file): ascii, binary little/big endian. No units, Z up (as Blender,
MeshLab), one object per file, n-gons kept.

Reader: vertex x y z, normals nx ny nz, colours red green blue [alpha] (uchar or float),
UV s t | u v | texture_u texture_v | texture_s texture_t; face vertex_indices | vertex_index,
per-face colour, per-face `texcoord` lists (MeshLab). Other elements/properties are dropped
with a warning. Writer: everything baked into one object, per-corner attributes split into
vertices where they differ; face materials become vertex colours (PLY has no materials).
"""

import os
import re

import numpy as np

from .. import geom, textures
from ..model import Mesh, Node, Scene

TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
         "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
         "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}
UV_NAMES = (("s", "t"), ("u", "v"), ("texture_u", "texture_v"), ("texture_s", "texture_t"))
CHUNK = 100_000     # rows per formatted text block
FAST = True         # list fast paths; tools/check_ply_fast.py switches it off for the reference
COLOR_NAMES = (("red", "green", "blue", "alpha"), ("r", "g", "b", "a"),
               ("diffuse_red", "diffuse_green", "diffuse_blue", "diffuse_alpha"))


class _Element:
    def __init__(self, name, count):
        self.name, self.count = name, count
        self.props = []        # (name, type) or (name, (count type, item type)) for lists

    @property
    def has_lists(self):
        return any(isinstance(t, tuple) for _, t in self.props)


def _header(data):
    end = re.search(rb"end_header[ \t]*\r?\n", data)
    if not data.startswith(b"ply") or end is None:
        raise ValueError("PLY: нет заголовка ply … end_header")
    fmt, elements = None, []
    for line in data[:end.start()].decode("ascii", "replace").splitlines()[1:]:
        t = line.split()
        if not t or t[0] in ("comment", "obj_info"):
            continue
        if t[0] == "format":
            fmt = t[1]
        elif t[0] == "element":
            elements.append(_Element(t[1], int(t[2])))
        elif t[0] == "property" and elements:
            try:
                if t[1] == "list":
                    elements[-1].props.append((t[4], (TYPES[t[2]], TYPES[t[3]])))
                else:
                    elements[-1].props.append((t[2], TYPES[t[1]]))
            except (KeyError, IndexError):
                raise ValueError(f"PLY: непонятное свойство: {line.strip()}") from None
    if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):
        raise ValueError(f"PLY: неизвестный формат {fmt!r}")
    return fmt, elements, end.end()


class _List:
    """A list property of every record: counts (n,) and all items back to back."""

    def __init__(self, counts, flat):
        self.counts = np.asarray(counts, dtype=np.int64)
        self.flat = flat

    def __len__(self):
        return len(self.counts)

    @property
    def starts(self):
        return np.cumsum(self.counts) - self.counts


def _gather_lists(out, props):
    """Per-record values collected by the slow paths -> arrays / _List."""
    res = {}
    for (n, t), v in zip(props, out.values()):
        if not isinstance(t, tuple):
            res[n] = np.array(v)
        elif v:
            res[n] = _List([len(a) for a in v], np.concatenate(v))
        else:
            res[n] = _List([], np.zeros(0))
    return res


def _read_binary(el, data, off, bo):
    """-> ({prop: array or _List}, new offset)."""
    if not el.has_lists:
        dt = np.dtype([(n, bo + t) for n, t in el.props])
        rec = np.frombuffer(data, dt, el.count, off)
        return {n: rec[n] for n, _ in el.props}, off + dt.itemsize * el.count
    if FAST and len(el.props) == 1 and el.count:
        name, (ct, it) = el.props[0]
        cs, isz = np.dtype(ct).itemsize, np.dtype(it).itemsize
        # one list whose count is the same everywhere (all-triangle meshes)
        k = int(np.frombuffer(data, bo + ct, 1, off)[0])
        dt = np.dtype([("n", bo + ct), ("i", bo + it, (max(k, 0),))])
        if k >= 0 and off + dt.itemsize * el.count <= len(data):
            rec = np.frombuffer(data, dt, el.count, off)
            if np.all(rec["n"] == k):
                return ({name: _List(np.full(el.count, k), rec["i"].reshape(-1))},
                        off + dt.itemsize * el.count)
        # mixed counts: walk the records reading only the counts, then gather in one go
        if np.dtype(ct).kind in "iu":
            res = _walk_binary(data, off, el.count, bo + ct, isz)
            if res is not None:
                starts, end = res
                block = np.frombuffer(data, np.uint8, end - off, off)
                is_item = np.ones(len(block), bool)
                rel = starts - off
                for b in range(cs):
                    is_item[rel + b] = False
                counts = np.frombuffer(block[~is_item].tobytes(), bo + ct)
                if np.all(counts >= 0):
                    items = np.frombuffer(block[is_item].tobytes(), bo + it)
                    return {name: _List(counts, items)}, end
    out = {n: [] for n, _ in el.props}
    for _ in range(el.count):
        for n, t in el.props:
            if isinstance(t, tuple):
                k = int(np.frombuffer(data, bo + t[0], 1, off)[0])
                if k < 0:
                    raise ValueError(f"PLY: отрицательная длина списка {n} ({k})")
                off += np.dtype(t[0]).itemsize
                out[n].append(np.frombuffer(data, bo + t[1], k, off))
                off += np.dtype(t[1]).itemsize * k
            else:
                out[n].append(np.frombuffer(data, bo + t, 1, off)[0])
                off += np.dtype(t).itemsize
    return _gather_lists(out, el.props), off


def _walk_binary(data, off, count, ct, isz):
    """Record starts of a single-list element -> (starts, end offset), None if the data ends
    early or a count is negative (the record-by-record path then reports it)."""
    cs = np.dtype(ct).itemsize
    starts = np.empty(count, np.int64)
    n = len(data)
    if cs == 1 and np.dtype(ct).kind == "u":
        for i in range(count):          # bytes[i] is the count itself: the cheapest loop
            if off >= n:
                return None
            starts[i] = off
            off += 1 + data[off] * isz
    else:
        signed, order = np.dtype(ct).kind == "i", "big" if ct[0] == ">" else "little"
        frm = int.from_bytes
        for i in range(count):
            if off + cs > n:
                return None
            starts[i] = off
            k = frm(data[off:off + cs], order, signed=signed)
            if k < 0:
                return None
            off += cs + k * isz
    return (starts, off) if off <= n else None


def _read_ascii(el, tok, pos):
    if not el.has_lists:
        k = len(el.props)
        block = np.array(tok[pos:pos + k * el.count]).astype(np.float64).reshape(el.count, k)
        return {n: block[:, i] for i, (n, _) in enumerate(el.props)}, pos + k * el.count
    if FAST and len(el.props) == 1 and el.count:
        res = _walk_ascii(el, tok, pos)
        if res is not None:
            return res
    out = {n: [] for n, _ in el.props}
    for _ in range(el.count):
        for n, t in el.props:
            if isinstance(t, tuple):
                k = int(tok[pos])
                if k < 0:
                    raise ValueError(f"PLY: отрицательная длина списка {n} ({k})")
                if pos + 1 + k > len(tok):
                    raise IndexError("list runs past the end of the file")
                out[n].append(np.array(tok[pos + 1:pos + 1 + k]).astype(np.float64))
                pos += 1 + k
            else:
                out[n].append(float(tok[pos]))
                pos += 1
    return _gather_lists(out, el.props), pos


def _walk_ascii(el, tok, pos):
    """Single list property: a walk over the counts only, then all items converted at once
    (the same S -> float64 conversion as per record). None -> the record-by-record path,
    which reports a bad count or a short file."""
    name = el.props[0][0]
    counts = [0] * el.count
    p, n = pos, len(tok)
    try:
        for i in range(el.count):
            k = counts[i] = int(tok[p])
            p += 1 + k
    except (ValueError, IndexError):
        return None
    if p > n or min(counts) < 0:
        return None
    counts = np.array(counts, np.int64)
    is_item = np.ones(p - pos, bool)
    is_item[np.cumsum(counts + 1) - counts - 1] = False
    items = np.array(tok[pos:p])[is_item].astype(np.float64)
    return {name: _List(counts, items)}, p


def _colour(props, types, names):
    """Colour columns -> (n, 4) 0..1, or None. Integer types are scaled by their maximum."""
    for group in names:
        if all(n in props for n in group[:3]):
            cols = []
            for n in group:
                if n in props:
                    v = np.asarray(props[n], dtype=np.float64)
                    t = np.dtype(types[n])
                    cols.append(v / np.iinfo(t).max if t.kind in "iu" else v)
                else:
                    cols.append(np.ones(len(props[group[0]])))
            return np.stack(cols, 1), set(group)
    return None, set()


def read(path):
    with open(path, "rb") as fh:
        data = fh.read()
    fmt, elements, off = _header(data)
    scene = Scene(unit_m=None)
    bo = {"binary_little_endian": "<", "binary_big_endian": ">"}.get(fmt)
    tok = data[off:].split() if bo is None else None
    pos, parsed = 0, {}
    for el in elements:
        try:
            if bo is None:
                parsed[el.name], pos = _read_ascii(el, tok, pos)
            else:
                parsed[el.name], off = _read_binary(el, data, off, bo)
        except (ValueError, IndexError) as e:     # short file, bad numbers: numpy's words
            if str(e).startswith("PLY:"):
                raise
            raise ValueError(f"PLY: данные элемента «{el.name}» повреждены или обрезаны "
                             f"({e})") from None
    types = {el.name: {n: (t[1] if isinstance(t, tuple) else t) for n, t in el.props}
             for el in elements}
    for el in elements:
        if el.name not in ("vertex", "face"):
            scene.warn(f"PLY: элемент «{el.name}» ({el.count}) не переносится")

    v = parsed.get("vertex")
    if v is None or not all(k in v for k in "xyz"):
        raise ValueError("PLY: нет вершин с x y z")
    used = {"x", "y", "z"}
    pos_ = np.stack([np.asarray(v[k], dtype=np.float64) for k in "xyz"], 1)
    vnorm = None
    if all(k in v for k in ("nx", "ny", "nz")):
        vnorm = np.stack([np.asarray(v[k], dtype=np.float64) for k in ("nx", "ny", "nz")], 1)
        used |= {"nx", "ny", "nz"}
    vcol, cu = _colour(v, types["vertex"], COLOR_NAMES)
    used |= cu
    vuv = None
    for a, b in UV_NAMES:
        if a in v and b in v:
            vuv = np.stack([np.asarray(v[a], np.float64), np.asarray(v[b], np.float64)], 1)
            used |= {a, b}
            break
    extra = [n for n in v if n not in used]
    if extra:
        scene.warn(f"PLY: свойства вершин не переносятся: {', '.join(extra)}")

    f = parsed.get("face", {})
    key = next((k for k in ("vertex_indices", "vertex_index") if k in f), None)
    polys = f[key] if key else _List([], np.zeros(0, np.int64))
    if not isinstance(polys, _List):
        raise ValueError(f"PLY: {key} граней — не список")
    if not len(polys):
        scene.warn("PLY: нет граней — облако точек не переносится")
    sizes = polys.counts
    keep = sizes >= 3
    if not np.all(keep):
        scene.warn(f"PLY: {int(np.sum(~keep))} граней с < 3 вершинами отброшены")
    corners = polys.flat[np.repeat(keep, sizes)].astype(np.int64)
    sizes = sizes[keep]
    if len(corners) and (corners.min() < 0 or corners.max() >= len(pos_)):
        raise ValueError("PLY: индекс вершины вне диапазона")
    colors = vcol[corners] if vcol is not None else None
    uv = vuv[corners] if vuv is not None else None
    fcol, fu = _colour(f, types.get("face", {}), COLOR_NAMES)
    if fcol is not None:
        fcol = np.repeat(fcol[keep], sizes, axis=0)
        if colors is not None:
            scene.warn("PLY: есть цвета и у вершин, и у граней — взяты цвета вершин")
        else:
            colors = fcol
    ftex = f.get("texcoord")
    if ftex is not None:
        if isinstance(ftex, _List) and np.array_equal(ftex.counts[keep], 2 * sizes):
            uv = (ftex.flat[np.repeat(keep, ftex.counts)].astype(np.float64).reshape(-1, 2)
                  if len(sizes) else None)
        else:
            scene.warn("PLY: texcoord граней не совпадает с числом вершин — UV отброшены")
    extra = [n for n in f if n not in {key, "texcoord"} | fu]
    if extra:
        scene.warn(f"PLY: свойства граней не переносятся: {', '.join(extra)}")

    name = os.path.splitext(os.path.basename(path))[0]
    scene.meshes.append(Mesh(name, pos_, sizes, corners, uv=uv,
                             normals=vnorm[corners] if vnorm is not None else None,
                             colors=colors))
    scene.nodes.append(Node(name, mesh=0))
    return scene


# --- writer ----------------------------------------------------------------------------------

def _gather(scene, scale):
    """Bake all instances into one mesh's arrays. -> dict of per-corner arrays + warnings."""
    w = []
    baked = [b for b in geom.bake(scene)]
    if len(baked) > 1:
        w.append(f"PLY: объекты ({len(baked)}) слиты в один, имена потеряны")
    if not scene.is_flat():
        w.append("PLY: иерархия и инстансы запечены в вершины")
    if any(not b.mesh.n_faces for b in baked):
        w.append("PLY: пустые меши (без граней) отброшены")
    has_uv = [b.mesh.uv is not None for b in baked]
    has_n = [b.mesh.normals is not None for b in baked]
    has_c = [b.mesh.colors is not None for b in baked]
    has_m = [b.mesh.face_material is not None and any(
        scene.materials[i].color is not None for i in np.unique(b.mesh.face_material) if i >= 0)
        for b in baked]
    use_uv, use_n = bool(baked) and all(has_uv), bool(baked) and all(has_n)
    use_c = any(has_c) or any(has_m)
    if any(has_uv) and not use_uv:
        w.append("PLY: UV есть не у всех объектов — отброшены")
    if any(has_n) and not use_n:
        w.append("PLY: нормали есть не у всех объектов — отброшены")
    if any(has_m):
        w.append("PLY: материалов нет в формате — их цвета записаны цветами вершин")
    if textures.used_textures(scene):
        w.append("PLY: текстуры отброшены (формат их не хранит)")
    if any(has_c) and any(c and m for c, m in zip(has_c, has_m)) and not all(
            geom.colors_redundant(b.mesh, scene.materials) for b in baked):
        w.append("PLY: где есть и цвета вершин, и материал — взяты цвета вершин")
    pos, sizes, corners, uv, nrm, col = [], [], [], [], [], []
    base = 0
    for b in baked:
        m = b.mesh
        order = geom.corner_order(m, geom.is_mirror(b.matrix))
        pos.append(b.positions * scale)
        sizes.append(m.face_sizes)
        corners.append(m.corners[order] + base)
        base += len(m.positions)
        if use_uv:
            uv.append(m.uv[order])
        if use_n:
            nrm.append(geom.apply_normals(b.matrix, m.normals)[order])
        if use_c:
            c = (m.colors if m.colors is not None
                 else geom.material_colors(m, scene.materials))
            col.append(c[order])
    cat = lambda a, shape: np.concatenate(a) if a else np.zeros(shape)
    out = {"pos": cat(pos, (0, 3)), "sizes": cat(sizes, 0).astype(np.int64),
           "corners": cat(corners, 0).astype(np.int64),
           "uv": cat(uv, (0, 2)) if use_uv else None, "nrm": cat(nrm, (0, 3)) if use_n else None,
           "col": cat(col, (0, 4)) if use_c else None}
    return out, w


def _split(d):
    """Per-corner attributes -> PLY vertices (geom.split_vertices: one per distinct vertex and
    attributes, np.unique order, values of the first corner) and the vertex of each corner."""
    c = d["corners"]
    cols = []
    if d["uv"] is not None:
        cols.append(d["uv"])
    if d["nrm"] is not None:
        cols.append(d["nrm"])
    if d["col"] is not None:
        cols.append(np.round(np.clip(d["col"], 0, 1) * 255))
    attr = np.concatenate(cols, 1) if cols else np.zeros((len(c), 0))
    first, inv = geom.split_vertices(c, attr)
    return np.concatenate([d["pos"][c[first]], attr[first]], 1), inv


def write(scene, path, scale=1.0, unit_m=None, binary=True):
    """scale: file units -> output units. Returns warning lines."""
    d, w = _gather(scene, scale)
    vert, corners = _split(d)
    props = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if d["uv"] is not None:
        props += [("s", "f4"), ("t", "f4")]
    if d["nrm"] is not None:
        props += [("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    if d["col"] is not None:
        props += [("red", "u1"), ("green", "u1"), ("blue", "u1"), ("alpha", "u1")]
    sizes = d["sizes"]
    ctype = "uchar" if not len(sizes) or sizes.max() < 256 else "int"
    names = {"f4": "float", "u1": "uchar"}
    head = ["ply", f"format {'binary_little_endian' if binary else 'ascii'} 1.0",
            "comment mesh2mesh", f"element vertex {len(vert)}"]
    head += [f"property {names[t]} {n}" for n, t in props]
    head += [f"element face {len(sizes)}", f"property list {ctype} int vertex_indices",
             "end_header"]
    with open(path, "wb") as fh:
        fh.write(("\n".join(head) + "\n").encode("ascii"))
        if binary:
            rec = np.zeros(len(vert), [(n, "<" + t) for n, t in props])
            for i, (n, _) in enumerate(props):
                rec[n] = vert[:, i]
            fh.write(rec.tobytes())
            if len(sizes) and np.all(sizes == sizes[0]):
                k = int(sizes[0])
                frec = np.zeros(len(sizes), [("n", "<" + ("u1" if ctype == "uchar" else "i4")),
                                              ("i", "<i4", (k,))])
                frec["n"], frec["i"] = k, corners.reshape(-1, k)
                fh.write(frec.tobytes())
            elif len(sizes):
                # count + indices per face, back to back: indices fill every byte that is
                # not a count
                ct = np.dtype("<u1" if ctype == "uchar" else "<i4")
                rec = ct.itemsize + 4 * sizes
                head = np.cumsum(rec) - rec
                is_count = np.zeros(int(rec.sum()), bool)
                for b in range(ct.itemsize):
                    is_count[head + b] = True
                out = np.empty(len(is_count), np.uint8)
                out[is_count] = np.frombuffer(sizes.astype(ct).tobytes(), np.uint8)
                out[~is_count] = np.frombuffer(corners.astype("<i4").tobytes(), np.uint8)
                fh.write(out.tobytes())
        else:
            # text formatted in blocks by the % operator: the same as str.format per number
            # (%d truncates like int(), %.9g of the float32 value like {:.9g})
            fmt = " ".join("%d" if t == "u1" else "%.9g" for _, t in props) + "\n"
            cols = np.array(vert, dtype=np.float64)
            for i, (_, t) in enumerate(props):
                cols[:, i] = np.trunc(cols[:, i]) if t == "u1" else \
                    cols[:, i].astype(np.float32).astype(np.float64)
            for s in range(0, len(cols), CHUNK):
                part = cols[s:s + CHUNK]
                fh.write(((fmt * len(part)) % tuple(part.ravel().tolist())).encode("ascii"))
            templates = {}
            c = 0
            for s in range(0, len(sizes), CHUNK):
                part = sizes[s:s + CHUNK]
                n = int(part.sum())
                for k in np.unique(part).tolist():
                    templates.setdefault(k, f"{k} " + " ".join(["%d"] * k) + "\n")
                text = "".join([templates[k] for k in part.tolist()])
                fh.write((text % tuple(corners[c:c + n].tolist())).encode("ascii"))
                c += n
    if d["col"] is not None and np.any(np.abs(np.round(d["col"] * 255) / 255 - d["col"]) > 1 / 510):
        w.append("PLY: цвета округлены до 8 бит на канал")
    return w
