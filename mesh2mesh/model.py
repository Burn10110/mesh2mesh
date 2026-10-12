"""Format-neutral scene: what every reader produces and every writer consumes.

Conventions (readers convert into them, writers out of them):
- right-handed, Z up; lengths in file units, `Scene.unit_m` metres per unit;
- matrices are 4x4 for column vectors: p_parent = M @ p_local;
- faces are n-gons stored flat: `face_sizes` + `corners` (indices into `positions`);
  per-corner attributes (uv, normals, colors) are aligned with `corners`.
"""

from dataclasses import dataclass, field

import numpy as np

# name writers give faces without a material when the format cannot leave them bare
# (OBJ: usemtl persists; 3MF: object-level property)
NO_MATERIAL = "no_material"


@dataclass(eq=False)
class Texture:
    """An image file: `data` = its bytes as stored (PNG, JPEG, TGA…), `name` = file name to use
    when written beside the model (unique per scene is the writer's job)."""
    name: str
    data: bytes


@dataclass
class Material:
    name: str
    color: tuple | None = None      # RGBA 0..1 as stored in the file (sRGB for 3MF/OBJ)
    # base colour map (UV channel = Mesh.uv); replaces the RGB of `color`, alpha still applies
    texture: Texture | None = None


@dataclass
class Mesh:
    name: str
    positions: np.ndarray            # (n, 3) float64
    face_sizes: np.ndarray           # (f,) int64, each >= 3
    corners: np.ndarray              # (sum(face_sizes),) int64 -> positions
    face_material: np.ndarray | None = None   # (f,) int64 into Scene.materials, -1 = none
    uv: np.ndarray | None = None              # (c, 2) per corner
    normals: np.ndarray | None = None         # (c, 3) per corner
    colors: np.ndarray | None = None          # (c, 4) per corner, RGBA 0..1

    @property
    def n_faces(self):
        return len(self.face_sizes)

    @property
    def face_starts(self):
        """Offset of each face's first corner in `corners`."""
        return np.concatenate(([0], np.cumsum(self.face_sizes)[:-1])).astype(np.int64)

    @property
    def is_triangles(self):
        return bool(np.all(self.face_sizes == 3))

    @classmethod
    def from_triangles(cls, name, positions, tris, **attrs):
        tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
        return cls(name, np.asarray(positions, dtype=np.float64).reshape(-1, 3),
                   np.full(len(tris), 3, dtype=np.int64), tris.ravel(), **attrs)


@dataclass
class Node:
    name: str
    matrix: np.ndarray = field(default_factory=lambda: np.eye(4))  # local -> parent
    mesh: int | None = None          # index into Scene.meshes
    children: list = field(default_factory=list)


@dataclass
class Scene:
    unit_m: float | None = None      # metres per file unit; None = the format has none
    nodes: list = field(default_factory=list)       # roots
    meshes: list = field(default_factory=list)
    materials: list = field(default_factory=list)
    warnings: list = field(default_factory=list)    # what the reader could not take over

    def warn(self, msg):
        if msg not in self.warnings:
            self.warnings.append(msg)

    def walk(self):
        """Yield (node, world matrix, path of names, depth) depth-first, in file order."""
        def rec(node, parent, path, depth):
            world = parent @ node.matrix
            path = path + [node.name]
            yield node, world, path, depth
            for ch in node.children:
                yield from rec(ch, world, path, depth + 1)
        for root in self.nodes:
            yield from rec(root, np.eye(4), [], 0)

    def instances(self):
        """Yield (node, world matrix) for every node that carries a mesh."""
        for node, world, _, _ in self.walk():
            if node.mesh is not None:
                yield node, world

    def is_flat(self):
        """No hierarchy, no transforms, no mesh reuse: maps 1:1 onto OBJ/STL/PLY objects."""
        used = [n.mesh for n, _ in self.instances()]
        return (len(used) == len(set(used))
                and all(not n.children and np.allclose(n.matrix, np.eye(4)) for n in self.nodes))
