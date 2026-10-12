"""Core tests on synthetic files: run with .venv/Scripts/python -m pytest tests"""

import os
import zipfile

import numpy as np
import pytest

from mesh2mesh import __main__ as cli
from mesh2mesh import convert, geom
from mesh2mesh.formats import obj, stl, threemf
from mesh2mesh.model import Mesh, Node, Scene

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANISTER = os.path.join(ROOT, "samples", "Mata_Canister_CAP_Scaled.3mf")

# unit cube, outward-facing triangles
CUBE_V = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
CUBE_T = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
          (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]


def make_3mf(path, unit="millimeter", item_tf="", comp_tf="", second_comp_tf=None, tri_attrs=None):
    """Cube mesh object 1 (base materials + colour group), assembly object 2 with components."""
    tri_attrs = tri_attrs or {}
    verts = "".join(f'<vertex x="{x}" y="{y}" z="{z}"/>' for x, y, z in CUBE_V)
    tris = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}" {tri_attrs.get(i, "")}/>'
                   for i, (a, b, c) in enumerate(CUBE_T))
    comps = f'<component objectid="1" transform="{comp_tf}"/>' if comp_tf else '<component objectid="1"/>'
    if second_comp_tf is not None:
        comps += f'<component objectid="1" transform="{second_comp_tf}"/>'
    model = f"""<?xml version="1.0" encoding="UTF-8"?>
<model unit="{unit}" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
 xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">
<resources>
 <basematerials id="5"><base name="Red" displaycolor="#FF0000"/><base name="Blue" displaycolor="#0000FFFF"/></basematerials>
 <m:colorgroup id="6"><m:color color="#00FF00FF"/><m:color color="#FF0000FF"/></m:colorgroup>
 <object id="1" name="cube" type="model" pid="5" pindex="0"><mesh><vertices>{verts}</vertices>
  <triangles>{tris}</triangles></mesh></object>
 <object id="2" name="assembly" type="model"><components>{comps}</components></object>
</resources>
<build><item objectid="2" transform="{item_tf}"/></build>
</model>"""
    rels = ('<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/'
            'package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("_rels/.rels", rels)
        z.writestr("3D/3dmodel.model", model)
    return path


def tf(m3x3, t=(0, 0, 0)):
    """Column-vector 3x3 + translation -> 3MF row-vector attribute string."""
    rows = np.asarray(m3x3, float).T
    return " ".join(f"{x:.17g}" for x in [*rows.ravel(), *t])


# --- 3MF ------------------------------------------------------------------------------------

def test_3mf_nested_transforms_and_units(tmp_path):
    rot = geom.quarter_turn("z", 1)
    p = make_3mf(tmp_path / "a.3mf", unit="inch",
                 comp_tf=tf(rot * 2, (10, 0, 0)), item_tf=tf(np.eye(3), (0, 0, 5)))
    s = threemf.read(p)
    assert s.unit_m == pytest.approx(0.0254)
    assert [n.name for n, *_ in s.walk()] == ["assembly", "cube"]
    st = geom.stats(s)
    # cube scaled 2, turned 90 about z (x' = -y), moved +10 x, +5 z -> inches -> metres
    assert st.bbox_min / 0.0254 == pytest.approx([8, 0, 5])
    assert st.bbox_max / 0.0254 == pytest.approx([10, 2, 7])
    assert st.volume / 0.0254 ** 3 == pytest.approx(8)


def test_3mf_mirror_keeps_volume_positive_after_bake(tmp_path):
    p = make_3mf(tmp_path / "m.3mf", comp_tf=tf(np.diag([-1, 1, 1])))
    s = threemf.read(p)
    assert geom.stats(s).volume == pytest.approx(1e-9)   # 1 mm^3, outward after winding fix


def test_3mf_instancing_shares_mesh(tmp_path):
    p = make_3mf(tmp_path / "i.3mf", comp_tf=tf(np.eye(3)), second_comp_tf=tf(np.eye(3), (3, 0, 0)))
    s = threemf.read(p)
    assert len(s.meshes) == 1 and sum(1 for _ in s.instances()) == 2
    assert not s.is_flat()


def test_3mf_materials_and_corner_colours(tmp_path):
    attrs = {0: 'pid="5" p1="1"', 1: 'pid="6" p1="0" p2="1" p3="0"', 2: 'pid="6" p1="1"'}
    s = threemf.read(make_3mf(tmp_path / "c.3mf", tri_attrs=attrs))
    names = [m.name for m in s.materials]
    # object default Red; Blue; green colour; the red colour-group colour folds into "Red"
    assert sorted(names) == ["Blue", "Red", "color_00FF00FF"]
    mesh = s.meshes[0]
    fm = [s.materials[i].name for i in mesh.face_material]
    assert fm[0] == "Blue" and fm[1] == "color_00FF00FF" and fm[2] == "Red" and fm[3] == "Red"
    assert mesh.colors is not None and mesh.colors.shape == (36, 4)
    assert tuple(mesh.colors[4]) == (1, 0, 0, 1)     # triangle 1, corner 1 = p2 red


def test_3mf_bad_transform_raises(tmp_path):
    with pytest.raises(ValueError):
        threemf.read(make_3mf(tmp_path / "b.3mf", comp_tf="1 0 0"))


# --- geometry -------------------------------------------------------------------------------

def test_ear_clip_concave_keeps_area_and_winding():
    # L-shape (concave), counter-clockwise seen from +z
    poly = np.array([(0, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (1, 2, 0), (0, 2, 0)], float)
    tris = geom.ear_clip(poly)
    assert len(tris) == 4
    t = poly[tris]
    cr = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    assert np.all(cr[:, 2] > 0)                       # same winding as the polygon
    assert cr[:, 2].sum() / 2 == pytest.approx(3.0)   # area of the L


def test_triangulate_batch_equals_ear_clip():
    # convex, concave star-shaped, star {k/2} (all left turns, self-overlapping), degenerate
    rng = np.random.default_rng(0)
    polys = []
    for i in range(3000):
        k = int(rng.integers(4, 20))
        ang = 2 * np.pi * np.arange(k) / k
        r = np.ones(k) if i % 4 == 0 else rng.uniform(0.2, 1.0, k)
        if i % 4 == 2:
            ang = 2 * np.pi * ((np.arange(k) * 2) % k) / k
        p = np.c_[r * np.cos(ang), r * np.sin(ang), np.zeros(k)]
        if i % 4 == 3:
            p = np.outer(rng.random(k), rng.normal(size=3))
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        polys.append(p @ q.T + rng.normal(0, 10, 3))
    sizes = np.array([len(p) for p in polys])
    mesh = Mesh("m", np.concatenate(polys), sizes, np.arange(sizes.sum()))
    tris, src = geom.triangulate(mesh)
    starts = mesh.face_starts
    want = np.concatenate([geom.ear_clip(mesh.positions[s:s + n]) + s
                           for s, n in zip(starts, sizes)])
    assert np.array_equal(tris, want)
    assert np.array_equal(src, np.repeat(np.arange(len(sizes)), sizes - 2))


def test_rotation_spec():
    assert np.allclose(geom.rotation_matrix("x90") @ [0, 1, 0], [0, 0, 1])
    with pytest.raises(ValueError):
        geom.rotation_matrix("x45")


# --- OBJ ------------------------------------------------------------------------------------

def test_obj_reader_ngons_negative_indices_uv_mtl(tmp_path):
    (tmp_path / "a.mtl").write_text("newmtl Wood paint\nKd 0.5 0.25 0\nd 0.5\n")
    (tmp_path / "a.obj").write_text(
        "mtllib a.mtl\no L\n"
        "v 0 0 0\nv 2 0 0\nv 2 0 -1\nv 1 0 -1\nv 1 0 -2\nv 0 0 -2\n"   # L in the OBJ XZ plane, Y up
        "vt 0 0\nvt 1 0\nvt 1 1\nvn 0 1 0\n"
        "usemtl Wood paint\nf -6/1/1 -5/2/1 -4/3/1 -3/1/1 -2/2/1 -1/3/1\n"
        "l 1 2\n")
    s = obj.read(tmp_path / "a.obj")
    m = s.meshes[0]
    assert m.face_sizes.tolist() == [6]
    assert s.materials[0].name == "Wood paint" and s.materials[0].color == (0.5, 0.25, 0, 0.5)
    assert np.allclose(m.positions[1], [2, 0, 0]) and np.allclose(m.positions[4], [1, 2, 0])  # Y up -> Z up
    assert np.allclose(m.normals, [0, 0, 1])
    assert m.uv.shape == (6, 2)
    assert any("линии" in w for w in s.warnings)
    st = geom.stats(Scene(unit_m=1.0, nodes=s.nodes, meshes=s.meshes))
    assert st.triangles == 4 and st.area == pytest.approx(3.0)


@pytest.mark.parametrize("ext", [".obj", ".stl"])
def test_round_trip_3mf(tmp_path, ext):
    src = make_3mf(tmp_path / "r.3mf", unit="centimeter",
                   comp_tf=tf(geom.quarter_turn("x", 1) * 3, (1, 2, 3)),
                   second_comp_tf=tf(np.diag([1, -1, 1]), (-5, 0, 0)))
    res = convert.convert(str(src), str(tmp_path / f"out{ext}"), units="mm")
    assert res.ok, res.mismatches
    assert any("иерархия" in w for w in res.warnings)
    back = convert.load(str(tmp_path / f"out{ext}"), "mm")
    assert geom.stats(back).volume == pytest.approx((27 + 1) * 1e-6, rel=1e-5)  # cm^3 -> m^3


def test_obj_round_trip_keeps_ngons_uv_normals(tmp_path):
    pos = np.array([(0, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (1, 2, 0), (0, 2, 0)], float)
    mesh = Mesh("L", pos, np.array([6]), np.arange(6), uv=np.random.rand(6, 2),
                normals=np.tile([0, 0, 1.0], (6, 1)))
    s = Scene(unit_m=1e-3, nodes=[Node("L", mesh=0)], meshes=[mesh])
    obj.write(s, str(tmp_path / "l.obj"))
    b = obj.read(str(tmp_path / "l.obj")).meshes[0]
    assert b.face_sizes.tolist() == [6]
    assert np.allclose(b.positions[b.corners], pos) and np.allclose(b.uv, mesh.uv)
    assert np.allclose(b.normals, mesh.normals)


# --- STL ------------------------------------------------------------------------------------

def test_stl_ascii_multi_solid_and_binary_weld(tmp_path):
    s = Scene(unit_m=1e-3, nodes=[Node("a b", mesh=0), Node("c", np.diag([1, 1, 1, 1.0]), mesh=0)],
              meshes=[Mesh.from_triangles("cube", CUBE_V, CUBE_T)])
    s.nodes[1].matrix[:3, 3] = (5, 0, 0)
    stl.write(s, str(tmp_path / "a.stl"), binary=False)
    a = stl.read(str(tmp_path / "a.stl"))
    assert [n.name for n in a.nodes] == ["a_b", "c"]
    assert [len(m.positions) for m in a.meshes] == [8, 8]          # welded back to 8 vertices
    w = stl.write(s, str(tmp_path / "b.stl"))
    assert any("слиты" in x for x in w)
    b = stl.read(str(tmp_path / "b.stl"))
    assert len(b.meshes) == 1 and len(b.meshes[0].positions) == 16


def test_binary_stl_starting_with_solid_is_binary(tmp_path):
    p = tmp_path / "s.stl"
    s = Scene(unit_m=1e-3, nodes=[Node("x", mesh=0)], meshes=[Mesh.from_triangles("x", CUBE_V, CUBE_T)])
    stl.write(s, str(p))
    data = bytearray(p.read_bytes())
    data[:80] = b"solid exported by some CAD".ljust(80)
    p.write_bytes(bytes(data))
    assert stl.read(str(p)).meshes[0].n_faces == 12


# --- pipeline -------------------------------------------------------------------------------

def test_compare_catches_flipped_winding():
    s = Scene(unit_m=1e-3, nodes=[Node("x", mesh=0)], meshes=[Mesh.from_triangles("x", CUBE_V, CUBE_T)])
    f = Scene(unit_m=1e-3, nodes=[Node("x", mesh=0)],
              meshes=[Mesh.from_triangles("x", CUBE_V, np.array(CUBE_T)[:, ::-1])])
    assert any("объём" in x for x in geom.compare(geom.stats(s), geom.stats(f)))


def test_cli_scale_rotate_units(tmp_path, capsys):
    src = make_3mf(tmp_path / "u.3mf")
    out = tmp_path / "u.stl"
    assert cli.main([str(src), "-o", str(out), "--units", "cm", "--scale", "10", "--rotate", "z90"]) == 0
    st = geom.stats(convert.load(str(out), "cm"))
    assert st.bbox_min / 0.01 == pytest.approx([-1, 0, 0], abs=1e-6)   # 10 mm cube = 1 cm, turned
    assert st.bbox_max / 0.01 == pytest.approx([0, 1, 1], abs=1e-6)
    assert cli.main([str(src), "-o", str(tmp_path / "x.dae")]) == 2
    assert "не поддерживается" in capsys.readouterr().err


@pytest.mark.skipif(not os.path.exists(CANISTER), reason="sample missing")
def test_canister_matches_reference_numbers(tmp_path):
    s = threemf.read(CANISTER)
    st = geom.stats(s)
    assert st.triangles == 3828 + 19684
    assert st.bbox_min * 1000 == pytest.approx([11.556, 12.307, 0.0], abs=1e-3)
    assert st.bbox_max * 1000 == pytest.approx([99.058, 99.749, 36.336], abs=1e-3)
    assert sorted(m.name for m in s.materials) == ["Default", "Green"]
    for ext in (".obj", ".stl"):
        assert convert.convert(CANISTER, str(tmp_path / f"c{ext}")).ok
