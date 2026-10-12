"""Scene -> arrays for the 3D view: world triangles, polygon edges, per-triangle colours and
corner normals per smoothing mode. No Qt."""

from dataclasses import dataclass, field

import numpy as np

from mesh2mesh import geom

GREY = (0.72, 0.74, 0.78)
SELECT_COLOR = (1.0, 0.85, 0.15)
EDGE_COLOR = (0.10, 0.10, 0.12, 0.55)
EDGE_LIMIT = 3_000_000   # more polygon edges than this: draw faces only (GL memory, clutter)

# smoothing of the view (normals); the file's data is never changed
SMOOTH_FILE, SMOOTH_AUTO, SMOOTH_FLAT = "file", "auto", "flat"
AUTO_ANGLE = 30.0
ANGLE_EPS = 1e-5   # radians: ties at exactly AUTO_ANGLE (float32 noise ~1e-6) are smooth
SMOOTHING = {   # mode -> (menu text, tooltip)
    SMOOTH_FILE: ("Из файла", "Нормали из файла: гладко там, где автор сгладил, острые рёбра"
                              " остаются. Меши без нормалей (STL, 3MF…) — плоско"),
    SMOOTH_AUTO: ("Авто 30°", f"Нормали пересчитаны по углу между гранями (до {AUTO_ANGLE:g}° —"
                              " гладко), нормали файла не используются"),
    SMOOTH_FLAT: ("Плоско", "Каждая грань плоская — видны все фасеты"),
}


@dataclass
class Display:
    verts: np.ndarray        # (n, 3) world, file units, all instances stacked
    tris: np.ndarray         # (t, 3) indices into verts
    tri_part: np.ndarray     # (t,) instance (part) index of every triangle
    edges: np.ndarray        # (e, 2) polygon edges (no triangulation diagonals); may be empty
    corner_colors: np.ndarray  # (t, 3, 3) float32 sRGB 0..1, per triangle corner
    part_nodes: list         # node of every part, scene.instances() order
    baked: list = field(default_factory=list)   # geom.Baked of every part (normals on demand)
    # base colour maps: model.Texture of the scene (each once), and per triangle an index into
    # it (-1 = none: no map, or the mesh has no UV) with the triangle's corner UVs
    textures: list = field(default_factory=list)
    tri_texture: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    corner_uv: np.ndarray | None = None    # (t, 3, 2) float32; None when nothing is textured
    no_uv_textured: int = 0  # triangles whose material has a map but whose mesh has no UV
    _normals: dict = field(default_factory=dict, repr=False)

    @property
    def empty(self):
        return not len(self.tris)

    @property
    def meshes_without_normals(self):
        """(meshes without normals in the file, meshes shown); instances count once."""
        meshes = {id(b.mesh): b.mesh for b in self.baked}
        return sum(m.normals is None for m in meshes.values()), len(meshes)

    def normals(self, mode):
        """(t, 3, 3) float32 unit world normals per triangle corner for a SMOOTHING mode
        (cached: auto smoothing of a big scene takes a moment)."""
        if mode not in SMOOTHING:
            mode = SMOOTH_FILE
        if mode not in self._normals:
            self._normals[mode] = _normals(self, mode)
        return self._normals[mode]


def polygon_edges(mesh):
    """(e, 2) unique position-index pairs along every face boundary."""
    if not len(mesh.face_sizes):
        return np.zeros((0, 2), np.int64)
    nxt, _, _ = _corner_ring(mesh)
    e = np.sort(np.stack([mesh.corners, mesh.corners[nxt]], 1), axis=1)
    return geom.unique_rows(e)


def corner_colors(mesh, tris, tri_face, materials):
    """(t, 3, 3): the file's corner colours (vertex paint) win, then the face material colour,
    then grey."""
    if mesh.colors is not None and not geom.colors_redundant(mesh, materials):
        return np.clip(mesh.colors[tris][..., :3], 0, 1).astype(np.float32)
    pal = np.array([(m.color or GREY)[:3] for m in materials] + [GREY], dtype=np.float32)
    if mesh.face_material is None:
        face = np.tile(np.float32(GREY), (len(tris), 1))
    else:
        face = pal[mesh.face_material[tri_face]]
    return np.repeat(face[:, None], 3, axis=1)


def tri_textures(mesh, tri_face, materials, tex_index):
    """(t,) index of the base colour map of every triangle's material (-1 = none); tex_index:
    id(Texture) -> index, extended with new maps."""
    if mesh.face_material is None or not len(tri_face):
        return np.full(len(tri_face), -1, np.int64)
    pal = []
    for m in materials:
        t = m.texture
        if t is not None and id(t) not in tex_index:
            tex_index[id(t)] = len(tex_index)
        pal.append(-1 if t is None else tex_index[id(t)])
    pal = np.array(pal + [-1], dtype=np.int64)   # face_material -1 -> the last entry
    return pal[mesh.face_material[tri_face]]


def texture_groups(tri_texture):
    """Triangle order with equal maps together (stable) and the draw groups of that order:
    [(first triangle, count, map index or -1)]."""
    order = np.argsort(tri_texture, kind="stable")
    tt = tri_texture[order]
    if not len(tt):
        return order, []
    starts = np.concatenate([[0], np.flatnonzero(np.diff(tt)) + 1])
    counts = np.diff(np.concatenate([starts, [len(tt)]]))
    return order, [(int(a), int(c), int(tt[a])) for a, c in zip(starts, counts)]


def build(scene):
    baked = geom.bake(scene)
    nodes = [n for n, _ in scene.instances()]
    if not baked:
        z = np.zeros((0, 3))
        return Display(z, z.astype(np.int64), np.zeros(0, np.int64), np.zeros((0, 2), np.int64),
                       np.zeros((0, 3, 3), np.float32), nodes)
    edge_cache, tex_index = {}, {}
    verts, tris, tri_part, edges, colors, tri_tex, uvs = [], [], [], [], [], [], []
    no_uv = 0
    offset = 0
    for pi, b in enumerate(baked):
        mesh = b.mesh
        key = id(mesh)
        if key not in edge_cache:
            edge_cache[key] = polygon_edges(mesh)
        verts.append(b.positions)
        tris.append(mesh.corners[b.tris] + offset)
        edges.append(edge_cache[key] + offset)
        tri_part.append(np.full(len(b.tris), pi))
        colors.append(corner_colors(mesh, b.tris, b.tri_face, scene.materials))
        tt = tri_textures(mesh, b.tri_face, scene.materials, tex_index)
        if mesh.uv is None:
            no_uv += int((tt >= 0).sum())
            tt[:] = -1
            uvs.append(np.zeros((len(b.tris), 3, 2), np.float32))
        else:
            uvs.append(mesh.uv[b.tris].astype(np.float32))
        tri_tex.append(tt)
        offset += len(b.positions)
    edges = np.vstack(edges)
    if len(edges) > EDGE_LIMIT:
        edges = np.zeros((0, 2), np.int64)
    # keep the maps some triangle shows, renumbered in order
    tri_tex = np.concatenate(tri_tex)
    by_id = {id(m.texture): m.texture for m in scene.materials if m.texture is not None}
    all_tex = [by_id[k] for k in sorted(tex_index, key=tex_index.get)]
    keep = np.unique(tri_tex[tri_tex >= 0])
    textures = [all_tex[i] for i in keep]
    remap = np.full(len(all_tex) + 1, -1, np.int64)
    remap[keep] = np.arange(len(keep))
    tri_tex = remap[tri_tex]               # -1 -> remap[-1] = -1
    used = len(keep) > 0
    return Display(np.vstack(verts), np.vstack(tris).astype(np.int64), np.concatenate(tri_part),
                   edges, np.concatenate(colors), nodes, baked, textures=textures,
                   tri_texture=tri_tex, corner_uv=np.concatenate(uvs) if used else None,
                   no_uv_textured=no_uv)


def _unit(v):
    ln = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, ln, out=np.zeros_like(v), where=ln > 0)


def face_normals(points):
    """(t, 3, 3) triangle corners -> (t, 3) unit normals (zero for degenerate triangles)."""
    return _unit(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]))


def _corner_ring(mesh):
    """Next and previous corner of every corner within its polygon, and its face index."""
    sizes, starts = mesh.face_sizes, mesh.face_starts
    face = np.repeat(np.arange(len(sizes)), sizes)
    pos = np.arange(len(mesh.corners))
    first, last = starts[face], starts[face] + sizes[face] - 1
    nxt = np.where(pos == last, first, pos + 1)
    prv = np.where(pos == first, last, pos - 1)
    return nxt, prv, face


def polygon_normals(mesh):
    """(f, 3) unit Newell normals of the polygons (as Blender: one normal per n-gon, so a
    non-planar quad is one facet, not two triangles); zero for degenerate polygons."""
    if not len(mesh.face_sizes):
        return np.zeros((0, 3))
    nxt, _, _ = _corner_ring(mesh)
    p = mesh.positions[mesh.corners]
    return _unit(np.add.reduceat(np.cross(p, p[nxt]), mesh.face_starts))


def _normals(disp, mode):
    """World corner normals of every triangle; per mesh the work is done once, per polygon
    corner in the file's own winding (bake reverses triangles of mirrored instances, the
    inverse transpose then turns the file's normals the right way)."""
    # fallback for whatever stays zero (degenerate polygons): the triangle's own facet
    out = np.repeat(face_normals(disp.verts[disp.tris])[:, None], 3, axis=1)
    start, cache = 0, {}
    for b in disp.baked:
        part = slice(start, start + len(b.tris))
        start += len(b.tris)
        mesh = b.mesh
        if id(mesh) not in cache:
            if mode == SMOOTH_AUTO:
                local = smooth_by_angle(mesh, AUTO_ANGLE)
            else:
                local = polygon_normals(mesh)[np.repeat(np.arange(mesh.n_faces), mesh.face_sizes)]
                if mode == SMOOTH_FILE and mesh.normals is not None:
                    ok = np.any(mesh.normals != 0, axis=1)   # zero normals in the file: flat
                    local[ok] = mesh.normals[ok]
            cache[id(mesh)] = local
        world = geom.apply_normals(b.matrix, cache[id(mesh)][b.tris].reshape(-1, 3)).reshape(-1, 3, 3)
        ok = np.any(world != 0, axis=-1)
        out[part][ok] = world[ok]
    return out.astype(np.float32)


def smooth_by_angle(mesh, angle):
    """Auto smooth (as Blender's Smooth by Angle) -> (c, 3) unit normals per corner.

    Points with equal coordinates are welded (glTF/OBJ split vertices at UV seams). A polygon
    edge shared by exactly two polygons whose normals differ by no more than `angle` degrees
    is smooth; corners around a vertex joined through smooth edges share one normal (their
    polygon normals weighted by corner angle). Open, non-manifold and sharper edges stay hard."""
    c = len(mesh.corners)
    if not c:
        return np.zeros((0, 3))
    nxt, prv, face = _corner_ring(mesh)
    _, vid = geom.unique_rows(mesh.positions, return_inverse=True)
    cv = vid.reshape(-1)[mesh.corners]                            # welded vertex of a corner
    p = mesh.positions[mesh.corners].astype(float)
    fn = polygon_normals(mesh)
    e1, e2 = p[nxt] - p, p[prv] - p
    weight = np.arctan2(np.linalg.norm(np.cross(e1, e2), axis=-1), np.einsum("ci,ci->c", e1, e2))

    # edge of corner k runs k -> nxt[k]; key = sorted welded ids
    a, b = cv, cv[nxt]
    nv = int(cv.max()) + 1
    key = np.minimum(a, b).astype(np.int64) * nv + np.maximum(a, b)
    collapsed = a == b
    key[collapsed] = -1 - np.arange(int(collapsed.sum()))         # never shared
    order = np.argsort(key, kind="stable")
    ks = key[order]
    same = ks[1:] == ks[:-1]
    prev = np.concatenate([[False], same[:-1]])
    after = np.concatenate([same[1:], [False]])
    pair = np.flatnonzero(same & ~prev & ~after)                  # exactly two polygons
    ei, ej = order[pair], order[pair + 1]
    # a hair over the angle: a 12-sided cylinder at 30° is a tie that float noise would split
    # into smooth and faceted sides
    smooth = (np.einsum("ij,ij->i", fn[face[ei]], fn[face[ej]])
              >= np.cos(np.radians(angle) + ANGLE_EPS))
    ei, ej = ei[smooth], ej[smooth]

    # join the corners at both ends of every smooth edge
    straight = a[ei] == a[ej]                  # same direction: neighbours with flipped winding
    u = np.concatenate([ei, nxt[ei]])
    v = np.concatenate([np.where(straight, ej, nxt[ej]), np.where(straight, nxt[ej], ej)])
    label = _components(c, u, v)

    contrib = fn[face] * weight[:, None]
    acc = np.stack([np.bincount(label, contrib[:, k], minlength=c) for k in range(3)], 1)
    n = _unit(acc[label])
    return np.where(np.any(n != 0, axis=1, keepdims=True), n, fn[face])


def _components(n, u, v):
    """Connected components of n nodes joined by edges (u, v) -> smallest node id per node.
    Min-label hooking + pointer jumping, numpy only."""
    label = np.arange(n)
    while True:
        m = np.minimum(label[u], label[v])
        new = label.copy()
        np.minimum.at(new, label[u], m)
        np.minimum.at(new, label[v], m)
        while True:                                               # flatten the trees
            jumped = new[new]
            if np.array_equal(jumped, new):
                break
            new = jumped
        if np.array_equal(new, label):
            return label
        label = new


def placed(verts, rotation):
    """Rotate, then centre and normalise to a unit bbox diagonal (GL float precision, fixed
    camera). Returns (verts, bbox min, bbox max) of the placed model."""
    v = verts @ np.asarray(rotation, dtype=float).T
    lo, hi = v.min(0), v.max(0)
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    c = (lo + hi) / 2
    v = (v - c) / diag
    return v, (lo - c) / diag, (hi - c) / diag


def camera_axes(azimuth, elevation):
    """Screen right / screen up / horizontal forward of pyqtgraph's euler camera (degrees),
    in world coordinates, Z up. From azimuth alone where possible: stable looking straight down."""
    az, el = np.radians(azimuth), np.radians(elevation)
    right = np.array([-np.sin(az), np.cos(az), 0.0])
    to_center = -np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    up = np.cross(right, to_center)
    forward = np.array([-np.cos(az), -np.sin(az), 0.0])
    return right, up, forward


def pan_offset(azimuth, elevation, dx, dy, scale, floor=False):
    """Look-at point shift for a mouse drag of (dx, dy) px (screen y down), `scale` = world units
    per pixel at the look-at point. Grab-style: the model follows the cursor. Default — in the
    screen plane (seen from the side, vertical drag moves along Z); floor=True — along the
    ground (XY), dragging down pulls the model toward the viewer."""
    right, up, forward = camera_axes(azimuth, elevation)
    return scale * (-right * dx + (forward if floor else up) * dy)


def grid_step(size_units):
    """Grid cell (in output units) as a power of ten: about 10 cells across the largest side."""
    m = float(max(size_units)) if size_units is not None and len(size_units) else 0.0
    if m <= 0:
        return 1.0
    return 10.0 ** np.floor(np.log10(m / 2))
