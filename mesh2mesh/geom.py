"""Geometry helpers: triangulation, transforms, scene statistics for the read-back check."""

from dataclasses import dataclass

import numpy as np

from .i18n import tr

AXES = "xyz"


# --- triangulation ---------------------------------------------------------------------------

VECTOR_MAX_SIDES = 16   # n-gons up to this size go through the batched path


def triangulate(mesh):
    """-> (t, 3) int64 indices into mesh.corners (so per-corner attributes stay addressable)
    and (t,) int64 source face index. Triangles pass through; n-gons are ear-clipped.
    Faces of one size are handled as a batch: convex and degenerate ones get exactly the
    triangles ear_clip would give them; only the rest go through ear_clip one by one."""
    sizes, starts = mesh.face_sizes, mesh.face_starts
    tri_faces = np.flatnonzero(sizes == 3)
    out = [starts[tri_faces, None] + np.arange(3)]
    src = [tri_faces]
    for k in np.unique(sizes[sizes > 3]):
        faces = np.flatnonzero(sizes == k)
        slow = faces
        if k <= VECTOR_MAX_SIDES:
            local, done = _batch_ear_clip(mesh.positions[mesh.corners[starts[faces, None]
                                                                     + np.arange(k)]])
            fast = faces[done]
            out.append((starts[fast, None, None] + local[done]).reshape(-1, 3))
            src.append(np.repeat(fast, k - 2))
            slow = faces[~done]
        for f in slow:
            s = starts[f]
            local = ear_clip(mesh.positions[mesh.corners[s:s + k]])
            out.append(local + s)
            src.append(np.full(len(local), f))
    order = np.argsort(np.concatenate(src), kind="stable")    # keep file face order
    return np.concatenate(out)[order].astype(np.int64), np.concatenate(src)[order].astype(np.int64)


def _fan(k, apex):
    """ear_clip's triangles for a k-gon cut as a fan: around vertex 0 (degenerate fallback) or
    around k-1 (what clipping ears in order does to a strictly convex polygon)."""
    if apex == 0:
        return np.array([[0, i, i + 1] for i in range(1, k - 1)])
    return np.array([[k - 1, i, i + 1] for i in range(k - 3)] + [[k - 3, k - 2, k - 1]])


def _batch_ear_clip(polys):
    """polys (F, k, 3) -> (F, k-2, 3) local triangles and (F,) bool: True where the result
    equals ear_clip (zero Newell normal -> fan at 0; strictly convex simple polygon in the
    projected plane -> fan at k-1). False rows are left for ear_clip."""
    F, k, _ = polys.shape
    nxt = np.roll(polys, -1, axis=1)
    nrm = np.stack([np.sum((polys[..., 1] - nxt[..., 1]) * (polys[..., 2] + nxt[..., 2]), 1),
                    np.sum((polys[..., 2] - nxt[..., 2]) * (polys[..., 0] + nxt[..., 0]), 1),
                    np.sum((polys[..., 0] - nxt[..., 0]) * (polys[..., 1] + nxt[..., 1]), 1)], 1)
    degenerate = ~np.any(nrm, axis=1)
    nrm[degenerate] = (0.0, 0.0, 1.0)                    # any basis; these rows use fan at 0
    a = np.argmin(np.abs(nrm), axis=1)
    u = np.cross(nrm, np.eye(3)[a])
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(nrm / np.linalg.norm(nrm, axis=1, keepdims=True), u)
    p = np.stack([np.einsum("fkc,fc->fk", polys, u), np.einsum("fkc,fc->fk", polys, v)], -1)
    # every vertex q strictly left of every edge (i, i+1) it is not on: strictly convex corners
    # and no self-overlap (a star with all-left turns has vertices on the wrong side of edges)
    e0 = p[:, :, None, :]                                # (F, k edges, 1, 2)
    e1 = np.roll(p, -1, axis=1)[:, :, None, :]
    q = p[:, None, :, :]                                 # (F, 1, k points, 2)
    side = ((e1[..., 0] - e0[..., 0]) * (q[..., 1] - e0[..., 1])
            - (e1[..., 1] - e0[..., 1]) * (q[..., 0] - e0[..., 0]))   # (F, edge, point)
    i = np.arange(k)
    off_edge = (i[None, :] != i[:, None]) & (i[None, :] != (i[:, None] + 1) % k)
    convex = np.all((side > 0) | ~off_edge, axis=(1, 2))
    local = np.where(degenerate[:, None, None], _fan(k, 0), _fan(k, k - 1))
    return local, degenerate | convex


def ear_clip(poly):
    """Triangulate one planar-ish polygon (k, 3) -> (k-2, 3) local indices, same winding.
    Projects on the Newell normal; falls back to a fan for degenerate input."""
    k = len(poly)
    if k == 3:
        return np.array([[0, 1, 2]])
    fan = np.array([[0, i, i + 1] for i in range(1, k - 1)])
    nrm = newell_normal(poly)
    if not np.any(nrm):
        return fan
    # 2D basis in the polygon plane, oriented so the polygon is counter-clockwise
    a = np.argmin(np.abs(nrm))
    u = np.cross(nrm, np.eye(3)[a])
    u /= np.linalg.norm(u)
    v = np.cross(nrm / np.linalg.norm(nrm), u)
    p = np.c_[poly @ u, poly @ v]
    idx = list(range(k))
    tris = []
    guard = 0
    while len(idx) > 3 and guard < k * k:
        guard += 1
        m = len(idx)
        for j in range(m):
            i0, i1, i2 = idx[j - 1], idx[j], idx[(j + 1) % m]
            if _cross2(p[i0], p[i1], p[i2]) <= 0:      # reflex or degenerate corner
                continue
            if any(_inside(p[q], p[i0], p[i1], p[i2]) for q in idx if q not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            del idx[j]
            break
        else:
            return fan                                 # self-intersecting / numeric trouble
    tris.append(tuple(idx))
    return np.array(tris)


def newell_normal(poly):
    nxt = np.roll(poly, -1, axis=0)
    return np.array([np.sum((poly[:, 1] - nxt[:, 1]) * (poly[:, 2] + nxt[:, 2])),
                     np.sum((poly[:, 2] - nxt[:, 2]) * (poly[:, 0] + nxt[:, 0])),
                     np.sum((poly[:, 0] - nxt[:, 0]) * (poly[:, 1] + nxt[:, 1]))])


def _cross2(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _inside(q, a, b, c):
    return _cross2(a, b, q) >= 0 and _cross2(b, c, q) >= 0 and _cross2(c, a, q) >= 0


# --- distinct rows ---------------------------------------------------------------------------

_MIX = np.uint64(0x9E3779B97F4A7C15)


def unique_rows(a, return_index=False, return_inverse=False):
    """np.unique(a, axis=0, …) for a 2-D numeric array, same results, several times faster.
    np.unique compares rows as structured records; here rows are hashed to one uint64, the
    hashes sorted, every group checked for real equality (a collision falls back to np.unique),
    and the distinct rows put in np.unique's lexicographic order. -0.0 == 0.0 as in np.unique;
    arrays with NaN go to np.unique."""
    a = np.asarray(a)
    want = [return_index, return_inverse]

    def fallback():
        r = np.unique(a, axis=0, return_index=return_index, return_inverse=return_inverse)
        if not any(want):
            return r
        r = list(r)
        if return_inverse:
            r[-1] = r[-1].reshape(-1)
        return tuple(r)

    if a.ndim != 2 or a.dtype.kind not in "fiu" or len(a) < 2 or a.shape[1] == 0:
        return fallback()
    if a.dtype.kind == "f":
        if np.isnan(a).any():
            return fallback()
        canon = a + a.dtype.type(0)                # -0.0 -> +0.0: equal rows, equal bits
        bits = canon.view(f"u{a.dtype.itemsize}")
    else:
        canon = a
        bits = a.view(f"u{a.dtype.itemsize}")
    n, k = a.shape
    h = np.zeros(n, np.uint64)
    with np.errstate(over="ignore"):
        for j in range(k):
            h ^= bits[:, j].astype(np.uint64)
            h *= _MIX
            h ^= h >> np.uint64(31)
    order = np.argsort(h)
    hs = h[order]
    start = np.empty(n, bool)
    start[0] = True
    np.not_equal(hs[1:], hs[:-1], out=start[1:])
    gid = np.cumsum(start) - 1                     # group of each sorted row
    first = np.minimum.reduceat(order, np.flatnonzero(start))   # first occurrence per group
    for j in range(k):                             # same hash but different row: give up
        col = bits[:, j]
        if not np.array_equal(col[order], col[first][gid]):
            return fallback()
    rank = np.lexsort(canon[first].T[::-1])        # np.unique order: column 0 first
    uniq = a[first[rank]]
    out = [uniq]
    if return_index:
        out.append(first[rank])
    if return_inverse:
        pos = np.empty(len(first), np.int64)
        pos[rank] = np.arange(len(first))
        inv = np.empty(n, np.int64)
        inv[order] = pos[gid]
        out.append(inv)
    return out[0] if len(out) == 1 else tuple(out)


def split_vertices(vid, attr):
    """Per-corner attributes -> output vertices, one per distinct (vertex, attribute row).
    vid (n,) vertex index of each corner, attr (n, k) float attributes (k may be 0).
    -> first (m,) the first corner of each output vertex, inv (n,) output vertex of each corner.
    Same as unique_rows(c_[vid, attr], return_index=True, return_inverse=True): order of
    np.unique over the rows, -0.0 == 0.0. The vertex index is compared first and corners of
    a vertex mostly share their attributes, so only vertices whose corners differ go through
    unique_rows."""
    vid = np.asarray(vid).astype(np.int64, copy=False)
    attr = np.asarray(attr, np.float64)
    n = len(vid)
    if not n:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    if np.isnan(attr).any():                       # NaN: np.unique semantics, the slow way
        key = np.concatenate([vid[:, None].astype(np.float64), attr], 1)
        _, first, inv = unique_rows(key, return_index=True, return_inverse=True)
        return first, inv.ravel()
    order = np.argsort(vid, kind="stable")
    cs = vid[order]
    new = np.empty(n, bool)                        # first corner of each vertex (sorted)
    new[0] = True
    np.not_equal(cs[1:], cs[:-1], out=new[1:])
    gid = np.cumsum(new) - 1
    differs = np.zeros(n, bool)                    # float != : -0.0 == 0.0, as in np.unique
    if attr.shape[1]:
        aso = attr[order]
        np.any(aso[1:] != aso[:-1], axis=1, out=differs[1:])
    differs &= ~new
    mixed = np.zeros(gid[-1] + 1, bool)            # vertices with more than one attribute row
    mixed[gid[differs]] = True
    count = np.ones(len(mixed), np.int64)
    inv = np.empty(n, np.int64)
    rank = np.zeros(n, np.int64)                   # row within its vertex, by sorted order
    pick = order[new]                              # first corner of each vertex
    if mixed.any():
        rows = np.flatnonzero(mixed[gid])          # sorted positions in mixed vertices
        sub = np.sort(order[rows])                 # original corners, in file order
        key = np.concatenate([vid[sub, None].astype(np.float64), attr[sub]], 1)
        _, sfirst, sinv = unique_rows(key, return_index=True, return_inverse=True)
        fc = vid[sub[sfirst]]                      # vertex of each distinct row, ascending
        fnew = np.empty(len(fc), bool)
        fnew[0] = True
        np.not_equal(fc[1:], fc[:-1], out=fnew[1:])
        start = np.flatnonzero(fnew)
        within = np.arange(len(fc)) - np.repeat(start, np.diff(np.append(start, len(fc))))
        mg = np.flatnonzero(mixed)                 # mixed vertices, ascending like fc[start]
        count[mg] = np.diff(np.append(start, len(fc)))
        rank[sub] = within[sinv]
    offset = np.cumsum(count) - count
    inv[order] = offset[gid]
    inv += rank
    first = np.empty(int(count.sum()), np.int64)
    first[offset] = pick
    if mixed.any():
        first[offset[np.repeat(mg, count[mg])] + within] = sub[sfirst]
    return first, inv


# --- transforms ------------------------------------------------------------------------------

def apply(matrix, points):
    """Transform (n, 3) points by a 4x4 column-vector matrix."""
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def apply_normals(matrix, normals):
    """Normals go through the inverse transpose; renormalised (zero stays zero)."""
    n = normals @ np.linalg.inv(matrix[:3, :3])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)


def is_mirror(matrix):
    return np.linalg.det(matrix[:3, :3]) < 0


def corner_order(mesh, mirror):
    """Corner indices face by face; within each face reversed for a mirrored instance."""
    n = len(mesh.corners)
    if not mirror:
        return np.arange(n)
    sizes = mesh.face_sizes
    starts = np.repeat(mesh.face_starts, sizes)
    local = np.arange(n) - starts
    return starts + np.repeat(sizes, sizes) - 1 - local


def rotation_matrix(spec):
    """'x90,z-90' -> 3x3 rotation, applied left to right about world axes. Quarter turns only,
    so the matrix is exact integers and a mirror can never sneak in."""
    m = np.eye(3)
    for step in filter(None, (s.strip().lower() for s in (spec or "").split(","))):
        axis, deg = step[0], step[1:]
        if axis not in AXES or not deg.lstrip("+-").isdigit() or int(deg) % 90:
            raise ValueError(f"bad rotation step {step!r}: expected x|y|z and a multiple of 90")
        m = quarter_turn(axis, int(deg) // 90) @ m
    return m


def quarter_turn(axis, turns):
    """Rotation by turns * 90 degrees about a world axis (right-hand rule), exact integers."""
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][turns % 4]
    i, j = [(1, 2), (2, 0), (0, 1)][AXES.index(axis)]
    m = np.eye(3)
    m[i, i], m[i, j], m[j, i], m[j, j] = c, -s, s, c
    return m


# --- colours ---------------------------------------------------------------------------------

def srgb_to_linear(c):
    """sRGB-encoded 0..1 -> linear (the transfer function only; alpha is passed separately)."""
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def material_colors(mesh, materials, default=(1.0, 1.0, 1.0, 1.0)):
    """(c, 4) per-corner colour from the face materials (faces without colour get `default`)."""
    pal = np.array([m.color or default for m in materials] + [default], dtype=np.float64)
    fm = mesh.face_material if mesh.face_material is not None else np.full(mesh.n_faces, -1)
    return pal[np.repeat(fm, mesh.face_sizes)]


def colors_redundant(mesh, materials):
    """True when the corner colours only repeat the face material colours (3MF readers produce
    that for every triangle of a mesh that has any gradient): writing both would double them."""
    if mesh.colors is None:
        return True
    if mesh.face_material is None:
        return False
    return bool(np.allclose(mesh.colors, material_colors(mesh, materials, (np.nan,) * 4),
                            atol=1 / 512))


# --- baking and statistics -------------------------------------------------------------------

@dataclass
class Baked:
    """One mesh instance in world space, triangulated."""
    name: str
    mesh: object        # source Mesh
    positions: np.ndarray   # (n, 3) world, file units
    tris: np.ndarray        # (t, 3) indices into mesh.corners, winding fixed for mirrors
    tri_face: np.ndarray    # (t,) source face
    matrix: np.ndarray      # world matrix

    @property
    def tri_points(self):
        """(t, 3, 3) world corner positions."""
        return self.positions[self.mesh.corners[self.tris]]


def bake(scene):
    """Every mesh instance -> Baked, names unique ('name', 'name.001', ...)."""
    out, seen, cache = [], {}, {}
    for node, world in scene.instances():
        mesh = scene.meshes[node.mesh]
        if node.mesh not in cache:
            cache[node.mesh] = triangulate(mesh)
        tris, src = cache[node.mesh]
        if is_mirror(world):
            tris = tris[:, ::-1]
        name = node.name or mesh.name or "mesh"
        n = seen.get(name, 0)
        seen[name] = n + 1
        if n:
            name = f"{name}.{n:03d}"
        out.append(Baked(name, mesh, apply(world, mesh.positions), tris, src, world))
    return out


def float32_collapsed(tri_points):
    """Triangles that have two corners at the same float32 point. Matters for STL only: its
    importers weld by position (Blender, slicers) and drop them; indexed float32 formats (glTF,
    PLY) keep them — checked in Blender, tools/probe_writer_edges.py."""
    f = np.asarray(tri_points, dtype=np.float32)
    return int(np.sum(np.all(f[:, 0] == f[:, 1], 1) | np.all(f[:, 1] == f[:, 2], 1)
                      | np.all(f[:, 0] == f[:, 2], 1)))


@dataclass
class Stats:
    objects: int
    triangles: int
    bbox_min: np.ndarray    # metres
    bbox_max: np.ndarray
    area: float             # m^2
    volume: float           # m^3, signed (winding-sensitive)

    def lines(self, unit_m, unit_name):
        k = 1.0 / unit_m
        size = (self.bbox_max - self.bbox_min) * k
        f = lambda v: " ".join(f"{x:.4f}" for x in v)
        return [tr("объектов {objects}, треугольников {tris}", objects=self.objects,
                   tris=self.triangles),
                tr("габарит ({unit}): [{lo}] .. [{hi}], размер [{size}]", unit=unit_name,
                   lo=f(self.bbox_min * k), hi=f(self.bbox_max * k), size=f(size))]


def stats(scene):
    """World-space statistics in metres; n-gons counted as their triangulation."""
    baked = bake(scene)
    u = scene.unit_m or 1.0
    pts = [b.tri_points * u for b in baked if len(b.tris)]
    if not pts:
        z = np.zeros(3)
        return Stats(len(baked), 0, z, z, 0.0, 0.0)
    t = np.concatenate(pts)
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    cr = np.cross(b - a, c - a)
    flat = t.reshape(-1, 3)
    return Stats(len(baked), len(t), flat.min(0), flat.max(0),
                 float(np.linalg.norm(cr, axis=1).sum() / 2),
                 float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6))


def compare(src, dst, float32=False):
    """Read-back verdict: list of mismatch descriptions (empty = OK)."""
    bad = []
    if src.triangles != dst.triangles:
        bad.append(tr("треугольников {a} -> {b}", a=src.triangles, b=dst.triangles))
    ext = max(np.abs(np.r_[src.bbox_min, src.bbox_max]).max(), 1e-12)
    tol = ext * (1e-6 if float32 else 1e-8)
    for nm, a, b in (("min", src.bbox_min, dst.bbox_min), ("max", src.bbox_max, dst.bbox_max)):
        if np.abs(a - b).max() > tol:
            bad.append(tr("габарит {nm} {a} -> {b}", nm=nm, a=a, b=b))
    rel = 1e-4 if float32 else 1e-6
    if abs(src.area - dst.area) > rel * max(src.area, 1e-30):
        bad.append(tr("площадь {a:.6g} -> {b:.6g} м²", a=src.area, b=dst.area))
    vref = max(abs(src.volume), src.area * ext * 1e-3, 1e-30)
    if abs(src.volume - dst.volume) > rel * vref:
        bad.append(tr("объём {a:.6g} -> {b:.6g} м³ (знак = ориентация граней)", a=src.volume,
                      b=dst.volume))
    return bad
