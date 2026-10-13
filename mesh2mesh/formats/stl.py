"""STL (binary and ASCII). No units, no materials, triangles only, Z up (as Blender/slicers).

Reader: welds exactly equal coordinates (STL stores every triangle on its own); ASCII files with
several `solid` blocks become several objects. Writer: binary by default (one anonymous body);
ASCII keeps one `solid <name>` per object. Facet normals are recomputed from the geometry.
"""

import re
import struct

import numpy as np

from .. import geom, textures
from ..model import Mesh, Node, Scene
from ..i18n import tr

BIN_DTYPE = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
FACET = ("  facet normal %.7e %.7e %.7e\n    outer loop\n"
         + "      vertex %.9e %.9e %.9e\n" * 3 + "    endloop\n  endfacet\n")
CHUNK = 50_000      # facets per formatted block


def _weld(tri_pts):
    """(t, 3, 3) -> positions, (t, 3) indices; exact-equality weld."""
    flat = tri_pts.reshape(-1, 3)
    pos, inv = geom.unique_rows(flat, return_inverse=True)
    return pos.astype(np.float64), inv.reshape(-1, 3)


def _is_binary(data):
    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        if 84 + n * 50 == len(data):
            return True
    return not data.lstrip()[:5].lower().startswith(b"solid")


def _ascii_solids(data):
    """The reference ASCII parse: [(name, (t, 3, 3) points)]."""
    text = data.decode("utf-8", "replace")
    solids = []
    for m in re.finditer(r"^\s*solid[ \t]*([^\r\n]*)(.*?)^\s*endsolid", text, re.S | re.M | re.I):
        nums = re.findall(r"^\s*vertex\s+(\S+)\s+(\S+)\s+(\S+)", m.group(2), re.M | re.I)
        v = np.array(nums, dtype=np.float64).reshape(-1, 3, 3)
        solids.append((m.group(1).strip() or "stl", v))
    return solids


_SOLID = re.compile(rb"^\s*solid[ \t]*([^\r\n]*)", re.M)
_VERTEX = re.compile(rb"\n[ \t\r\f\v]*vertex\s+(\S+\s+\S+\s+\S+)")
_WS = b" \t\n\r\f\v"


def _ascii_fast(data):
    """The same result as _ascii_solids without its regex walking every character (the lazy
    body costs ~10 ns a byte: 13 s on a 1.2 GB file). Works on the lower-cased bytes (the
    keywords are case-insensitive; float() reads 'E'/'e', 'INF'/'inf' the same) and finds
    `endsolid` / `vertex` by plain search, checking that only whitespace stands between them
    and a line start. None -> the reference parse (non-ASCII text, where \\s and re.I mean
    more, or a body opening with a lone \\r)."""
    if not data.isascii() or any(bytes([c]) in data for c in range(0x1C, 0x20)):
        return None
    low = data.lower()
    solids, pos = [], 0
    while True:
        m = _SOLID.search(low, pos)
        if m is None:
            return solids
        start = m.end()
        j = start
        while True:                    # first `endsolid` with only whitespace back to a line start
            j = low.find(b"endsolid", j)
            if j < 0:
                return solids           # no end: the reference finds no further solids either
            k = j
            while k > start and low[k - 1] in _WS:
                k -= 1
            run = low[max(k - 1, start - 1):j]
            if b"\n" in run or (k == start and _line_start(low, start)):
                break
            j += 1
        lead = len(low[start:j]) - len(low[start:j].lstrip(_WS))
        if b"\n" not in low[start:start + lead] and low.startswith(b"vertex", start + lead):
            return None                 # `\r vertex` right at the body start
        nums = b" ".join(_VERTEX.findall(low, start, j)).split()
        v = np.array(nums, dtype=np.float64).reshape(-1, 3, 3)
        solids.append((data[m.start(1):m.end(1)].decode("ascii").strip() or "stl", v))
        pos = j + len(b"endsolid")


def _line_start(b, i):
    return i == 0 or b[i - 1] == 0x0A


def read(path):
    with open(path, "rb") as fh:
        data = fh.read()
    scene = Scene(unit_m=None)
    if _is_binary(data):
        if len(data) < 84:
            raise ValueError(tr("STL: файл слишком короткий"))
        n = struct.unpack_from("<I", data, 80)[0]
        if 84 + n * 50 > len(data):
            raise ValueError(tr("STL: заявлено {n} треугольников, а данных меньше", n=n))
        rec = np.frombuffer(data, BIN_DTYPE, n, 84)
        if np.any(rec["attr"]):
            scene.warn(tr("STL: цвета в attribute-байтах (VisCAM/Materialise) не поддерживаются "
                          "— отброшены"))
        name = data[:80].split(b"\0")[0].decode("ascii", "replace").strip()
        if name.lower().startswith("solid"):
            name = ""
        solids = [(name or "stl", rec["v"].astype(np.float64))]
    else:
        solids = _ascii_fast(data)
        if solids is None:
            solids = _ascii_solids(data)
        if not solids:
            raise ValueError(tr("STL: не найдено ни одного solid … endsolid"))
    for name, tri_pts in solids:
        pos, tris = _weld(tri_pts)
        scene.nodes.append(Node(name, mesh=len(scene.meshes)))
        scene.meshes.append(Mesh.from_triangles(name, pos, tris))
    return scene


def losses(scene, binary):
    """What an STL cannot hold, as warning lines."""
    w = []
    if any(not m.is_triangles for m in scene.meshes):
        w.append(tr("STL: многоугольники разбиты на треугольники"))
    if scene.materials or any(m.face_material is not None for m in scene.meshes):
        w.append(tr("STL: материалы и цвета отброшены"))
    if textures.used_textures(scene):
        w.append(tr("STL: текстуры отброшены (формат их не хранит)"))
    if any(m.colors is not None for m in scene.meshes):
        w.append(tr("STL: цвета вершин отброшены"))
    if any(m.uv is not None for m in scene.meshes):
        w.append(tr("STL: UV отброшены"))
    if any(m.normals is not None for m in scene.meshes):
        w.append(tr("STL: нормали заменены нормалями граней"))
    if not scene.is_flat():
        w.append(tr("STL: иерархия и инстансы запечены в вершины"))
    if any(not scene.meshes[n.mesh].n_faces for n, _ in scene.instances()):
        w.append(tr("STL: пустые меши (без граней) отброшены"))
    n = sum(1 for _ in scene.instances())
    if binary and n > 1:
        w.append(tr("STL (binary): объекты ({n}) слиты в одно тело, имена потеряны (--stl-ascii "
                    "сохранит их раздельно)", n=n))
    return w


def write(scene, path, scale=1.0, unit_m=None, binary=True):
    """scale: file units -> output units. Returns warning lines."""
    baked = geom.bake(scene)
    parts = []
    for b in baked:
        t = b.tri_points * scale
        nrm = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        ln = np.linalg.norm(nrm, axis=1, keepdims=True)
        parts.append((b.name, t, np.divide(nrm, ln, out=np.zeros_like(nrm), where=ln > 0)))
    if binary:
        t = np.concatenate([p[1] for p in parts]) if parts else np.zeros((0, 3, 3))
        nrm = np.concatenate([p[2] for p in parts]) if parts else np.zeros((0, 3))
        rec = np.zeros(len(t), BIN_DTYPE)
        rec["n"], rec["v"] = nrm, t
        header = b"mesh2mesh binary STL".ljust(80, b" ")
        with open(path, "wb") as fh:
            fh.write(header + struct.pack("<I", len(t)) + rec.tobytes())
    else:
        with open(path, "w", encoding="ascii", errors="replace", newline="\n") as fh:
            for name, t, nrm in parts:
                name = re.sub(r"\s+", "_", name)
                fh.write(f"solid {name}\n")
                # facets formatted in blocks by the % operator (the same text as f-strings)
                vals = np.concatenate([nrm, t.reshape(-1, 9)], 1)
                for s in range(0, len(vals), CHUNK):
                    part = vals[s:s + CHUNK]
                    fh.write((FACET * len(part)) % tuple(part.ravel().tolist()))
                fh.write(f"endsolid {name}\n")
    w = losses(scene, binary)
    collapsed = sum(geom.float32_collapsed(p[1]) for p in parts)
    if collapsed:
        w.append(tr("STL: {collapsed} треуг. во float32 имеют совпадающие вершины — "
                    "Blender/слайсеры их выбросят (записаны как есть)", collapsed=collapsed))
    return w
