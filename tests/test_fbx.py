"""Iteration 3: FBX specifics - ASCII reader, the transform stack (pivots, rotation order,
pre/post rotation, Geometric offset), axes and handedness, writer decomposition (shear, mirror).
Expected values are worked out by hand, not with the module's own formula.
Run: .venv/Scripts/python -m pytest tests -q"""

import os

import numpy as np
import pytest

from mesh2mesh import convert, geom
from mesh2mesh.formats import fbx
from mesh2mesh.model import Mesh, Node, Scene

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FBX_SAMPLES = os.path.join(ROOT, "samples", "fbx")

CUBE_V = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
CUBE_Q = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
Y_UP = dict(UpAxis=1, UpAxisSign=1, FrontAxis=2, FrontAxisSign=1, CoordAxis=0, CoordAxisSign=1)


# --- a tiny ASCII FBX writer for hand-made cases ------------------------------------------------

def _p(name, *vals):
    kind = {"Lcl Translation": "Lcl Translation", "Lcl Rotation": "Lcl Rotation",
            "Lcl Scaling": "Lcl Scaling"}.get(name, "Vector3D" if len(vals) == 3 else "int")
    return f'\t\t\tP: "{name}", "{kind}", "", "A",' + ",".join(repr(float(v)) if len(vals) == 3
                                                               else str(v) for v in vals)


def ascii_fbx(path, models, axes=Y_UP, unit_cm=1.0, verts=CUBE_V, quads=CUBE_Q):
    """models: list of (uid, name, parent uid or 0, has_mesh, {property: value(s)})."""
    gs = "\n".join(f'\t\tP: "{k}", "int", "Integer", "",{v}' for k, v in axes.items())
    pvi = [c if i < len(q) - 1 else ~c for q in quads for i, c in enumerate(q)]
    flat = [x for v in verts for x in v]
    out = ["; FBX 7.4.0 project file", "FBXHeaderExtension:  {", "\tFBXVersion: 7400", "}",
           "GlobalSettings:  {", "\tVersion: 1000", "\tProperties70:  {", gs,
           f'\t\tP: "UnitScaleFactor", "double", "Number", "",{unit_cm}', "\t}", "}",
           "Objects:  {",
           '\tGeometry: 1000, "Geometry::cube", "Mesh" {',
           f"\t\tVertices: *{len(flat)} {{", "\t\t\ta: " + ",".join(map(str, flat)), "\t\t}",
           f"\t\tPolygonVertexIndex: *{len(pvi)} {{", "\t\t\ta: " + ",".join(map(str, pvi)),
           "\t\t}", "\t}"]
    conns = []
    for uid, name, parent, has_mesh, props in models:
        out += [f'\tModel: {uid}, "Model::{name}", "{"Mesh" if has_mesh else "Null"}" {{',
                "\t\tVersion: 232", "\t\tProperties70:  {"]
        out += [_p(k, *(v if isinstance(v, (tuple, list)) else (v,))) for k, v in props.items()]
        out += ["\t\t}", "\t}"]
        conns.append(f'\tC: "OO",{uid},{parent}')
        if has_mesh:
            conns.append(f'\tC: "OO",1000,{uid}')
    out += ["}", "Connections:  {"] + conns + ["}"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    return path


def world_points(scene, name):
    """World positions (scene axes, file units) of the instance called `name`."""
    for node, world in scene.instances():
        if node.name == name:
            return geom.apply(world, scene.meshes[node.mesh].positions)
    raise KeyError(name)


def yup(p):
    """Y-up file coordinates -> scene (Z up, front -Y): (x, y, z) -> (x, -z, y)."""
    p = np.asarray(p, dtype=np.float64)
    return np.column_stack([p[..., 0], -p[..., 2], p[..., 1]]).reshape(p.shape)


def one(tmp_path, props, **kw):
    path = ascii_fbx(str(tmp_path / "t.fbx"), [(1, "obj", 0, True, props)], **kw)
    return fbx.read(path)


# --- transform stack ----------------------------------------------------------------------------

def test_translation_and_units(tmp_path):
    s = one(tmp_path, {"Lcl Translation": (10, 20, 30)}, unit_cm=2.54)
    assert s.unit_m == pytest.approx(0.0254)
    assert np.allclose(world_points(s, "obj")[0], yup([10, 20, 30]))


def test_rotation_pivot(tmp_path):
    # 90 deg about Y around the pivot (1, 0, 0): (0,0,0) -> (1,0,1); (1,0,0) stays
    s = one(tmp_path, {"Lcl Rotation": (0, 90, 0), "RotationPivot": (1, 0, 0)})
    w = world_points(s, "obj")
    assert np.allclose(w[0], yup([1, 0, 1]), atol=1e-12)
    assert np.allclose(w[1], yup([1, 0, 0]), atol=1e-12)


def test_scaling_pivot_and_offsets(tmp_path):
    # scale 2 around (1, 1, 1), then ScalingOffset and RotationOffset just translate
    s = one(tmp_path, {"Lcl Scaling": (2, 2, 2), "ScalingPivot": (1, 1, 1),
                       "ScalingOffset": (0, 0, 5), "RotationOffset": (3, 0, 0)})
    w = world_points(s, "obj")
    assert np.allclose(w[0], yup([-1 + 3, -1, -1 + 5]))
    assert np.allclose(w[6], yup([1 + 3, 1, 1 + 5]))


@pytest.mark.parametrize("active", [0, 1])
def test_rotation_order_and_pre_rotation(tmp_path, active):
    # RotationOrder 5 = ZYX (Z first); PreRotation 90 about X. Both count only with
    # RotationActive 1 (Maya ignores them otherwise).
    s = one(tmp_path, {"RotationActive": active, "RotationOrder": 5,
                       "PreRotation": (90, 0, 0), "Lcl Rotation": (90, 90, 0)})
    p = np.array([1.0, 0, 0])                      # cube vertex 1
    rx = lambda v: np.array([v[0], -v[2], v[1]])   # +90 about X
    ry = lambda v: np.array([v[2], v[1], -v[0]])   # +90 about Y
    # XYZ: X first then Y; ZYX: Z(0) first, then Y, then X; then pre-rotation on the left
    want = rx(rx(ry(p))) if active else ry(rx(p))
    assert np.allclose(world_points(s, "obj")[1], yup(want), atol=1e-12)


def test_geometric_offset_moves_the_shape_not_the_children(tmp_path):
    path = ascii_fbx(str(tmp_path / "g.fbx"), [
        (1, "parent", 0, True, {"Lcl Translation": (5, 0, 0),
                                "GeometricTranslation": (0, 7, 0)}),
        (2, "child", 1, True, {})])
    s = fbx.read(path)
    assert np.allclose(world_points(s, "parent")[0], yup([5, 7, 0]))
    assert np.allclose(world_points(s, "child")[0], yup([5, 0, 0]))
    # the same Geometry under a different Geometric offset must not be shared
    assert len(s.meshes) == 2


def test_shared_geometry_is_instanced(tmp_path):
    path = ascii_fbx(str(tmp_path / "i.fbx"), [
        (1, "a", 0, True, {}), (2, "b", 0, True, {"Lcl Translation": (3, 0, 0)})])
    s = fbx.read(path)
    assert len(s.meshes) == 1 and [n.mesh for n in s.nodes] == [0, 0]


# --- axes, handedness, mirror ---------------------------------------------------------------------

def test_z_up_file(tmp_path):
    zup = dict(UpAxis=2, UpAxisSign=1, FrontAxis=1, FrontAxisSign=-1, CoordAxis=0, CoordAxisSign=1)
    s = one(tmp_path, {"Lcl Translation": (1, 2, 3)}, axes=zup)
    assert np.allclose(world_points(s, "obj")[0], [1, 2, 3])


def test_left_handed_file_keeps_faces_outward(tmp_path):
    lh = dict(Y_UP, CoordAxisSign=-1)
    s = one(tmp_path, {}, axes=lh)
    assert np.allclose(world_points(s, "obj")[1], [-1, 0, 0])
    assert geom.stats(s).volume > 0                # mirrored points, reversed winding


def test_mirrored_node_round_trip(tmp_path):
    m = np.diag([-1.0, 1, 1, 1])
    m[:3, 3] = (4, 0, 0)
    src = Scene(unit_m=1e-3, nodes=[Node("mirror", m, mesh=0, children=[
        Node("kid", np.eye(4), mesh=0)])],
        meshes=[Mesh("cube", np.array(CUBE_V, float), np.full(6, 4), np.array(CUBE_Q).ravel())])
    p = str(tmp_path / "m.fbx")
    fbx.write(src, p, unit_m=1e-3)
    back = fbx.read(p)
    for name in ("mirror", "kid"):
        assert np.allclose(world_points(back, name), world_points(src, name), atol=1e-12)
    assert geom.stats(back).volume == pytest.approx(geom.stats(src).volume)


def test_shear_goes_into_points_and_children(tmp_path):
    shear = np.eye(4)
    shear[0, 1] = 0.5                              # x += 0.5 y
    shear[:3, 3] = (1, 2, 3)
    child = np.eye(4)
    child[:3, 3] = (0, 2, 0)
    src = Scene(unit_m=1e-3, nodes=[Node("sheared", shear, mesh=0, children=[
        Node("kid", child, mesh=0)])],
        meshes=[Mesh("cube", np.array(CUBE_V, float), np.full(6, 4), np.array(CUBE_Q).ravel())])
    p = str(tmp_path / "s.fbx")
    fbx.write(src, p, unit_m=1e-3)
    back = fbx.read(p)
    for name in ("sheared", "kid"):
        assert np.allclose(world_points(back, name), world_points(src, name), atol=1e-12)
    # the sheared instance can no longer share the plain one's Geometry
    assert len(back.meshes) == 2


# --- rotations ----------------------------------------------------------------------------------

def random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    return q if np.linalg.det(q) > 0 else -q


def test_euler_xyz_inverts_euler():
    rng = np.random.default_rng(7)
    for _ in range(500):
        r = random_rotation(rng)
        assert np.allclose(fbx.euler(fbx.euler_xyz(r)), r, atol=1e-12)


@pytest.mark.parametrize("y", [90, -90])
def test_euler_xyz_gimbal_lock(y):
    r = fbx.euler([30, y, 40])
    assert np.allclose(fbx.euler(fbx.euler_xyz(r)), r, atol=1e-12)


def test_euler_orders_match_axis_products():
    a = [10, 20, 30]
    rx, ry, rz = (fbx.euler([a[0], 0, 0]), fbx.euler([0, a[1], 0]), fbx.euler([0, 0, a[2]]))
    want = {"XYZ": rz @ ry @ rx, "XZY": ry @ rz @ rx, "YZX": rx @ rz @ ry,
            "YXZ": rz @ rx @ ry, "ZXY": ry @ rx @ rz, "ZYX": rx @ ry @ rz}
    for order, m in want.items():
        assert np.allclose(fbx.euler(a, order), m), order


def test_decompose_random_matrices():
    rng = np.random.default_rng(11)
    for i in range(300):
        r = random_rotation(rng)
        s = rng.uniform(0.1, 10, 3) * rng.choice([-1, 1], 3)
        m = r @ np.diag(s)
        if i % 3 == 0:                             # some with shear on top
            m = m @ (np.eye(3) + np.triu(rng.normal(scale=0.3, size=(3, 3)), 1))
        ro, so, k = fbx.decompose(m)
        assert np.linalg.det(ro) == pytest.approx(1.0)
        back = ro @ np.diag(so) @ (k if k is not None else np.eye(3))
        assert np.allclose(back, m, atol=1e-12 * np.abs(m).max())
        if i % 3:
            assert k is None or np.allclose(k, np.eye(3), atol=1e-9)


# --- ASCII vs binary on Maya's own files -------------------------------------------------------

@pytest.mark.parametrize("up", ["yup", "zup"])
def test_maya_ascii_equals_binary(up):
    a, b = (os.path.join(FBX_SAMPLES, f"maya_{k}_{up}.fbx") for k in ("ascii", "bin"))
    if not (os.path.exists(a) and os.path.exists(b)):
        pytest.skip("Maya samples missing")
    sa, sb = fbx.read(a), fbx.read(b)
    names = [n.name for n, _ in sa.instances()]
    assert names == [n.name for n, _ in sb.instances()] and len(names) == 5
    for n in names:
        assert np.allclose(world_points(sa, n), world_points(sb, n), atol=1e-9)
    assert geom.stats(sa).triangles == geom.stats(sb).triangles


def test_ascii_round_trip_through_every_format(tmp_path):
    path = ascii_fbx(str(tmp_path / "a.fbx"), [(1, "obj", 0, True, {"Lcl Rotation": (0, 45, 0)})])
    for ext in (".obj", ".glb", ".3mf", ".fbx"):
        res = convert.convert(path, str(tmp_path / f"o{ext}"))
        assert res.ok, (ext, res.mismatches)
