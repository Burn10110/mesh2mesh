"""Window logic without Qt: convert.losses / scene reuse, display arrays, report texts."""

import copy

import numpy as np
import pytest

from mesh2mesh import convert, formats
from mesh2mesh_gui import display, report

from test_formats import CANISTER, EXTS, LOSSES, rich_scene, write_read


# --- convert: trial write and a pre-loaded scene ---------------------------------------------

@pytest.mark.parametrize("ext", EXTS)
def test_losses_equal_real_write(tmp_path, ext):
    """The losses box shows exactly what the real save will report (minus reader warnings)."""
    scene = rich_scene()
    before = copy.deepcopy(scene)
    trial = convert.losses(scene, ext, "mm" if ext not in formats.FIXED_UNITS else None)
    real, _, _ = write_read(rich_scene(), tmp_path, ext)
    assert trial == real
    must, _ = LOSSES[ext]
    assert all(any(f in w for w in trial) for f in must)
    # the window's scene is left untouched
    assert [n.matrix.tolist() for n, _ in scene.instances()] == \
           [n.matrix.tolist() for n, _ in before.instances()]


def test_losses_report_fixed_units_request():
    w = convert.losses(rich_scene(), ".glb", "mm")
    assert any("всегда m" in s for s in w)


def test_convert_with_preloaded_scene_matches_file_path(tmp_path):
    scene = convert.load(CANISTER)
    keep = copy.deepcopy(scene)
    a = convert.convert(CANISTER, str(tmp_path / "a.stl"), "cm", scale=10, rotation=None,
                        scene=scene)
    b = convert.convert(CANISTER, str(tmp_path / "b.stl"), "cm", scale=10)
    assert a.ok and b.ok
    assert (tmp_path / "a.stl").read_bytes() == (tmp_path / "b.stl").read_bytes()
    assert a.warnings == b.warnings
    assert scene.nodes[0].matrix.tolist() == keep.nodes[0].matrix.tolist()  # not transformed


# --- display ----------------------------------------------------------------------------------

def test_display_rich_scene():
    scene = rich_scene()
    d = display.build(scene)
    assert len(d.part_nodes) == 3 and [n.name for n in d.part_nodes] == ["Part A", "Part B", "Part C"]
    assert np.bincount(d.tri_part).tolist() == [20, 20, 12]  # L: 2 hexagons (4+4) + 6 quads
    # polygon edges of the L prism: 6 + 6 + 6 verticals, no diagonals; cube: 12 + 6 diagonals
    assert len(d.edges) == 18 * 2 + 18
    # world placement: Part C is the cube scaled x2 under the root moved by x+10
    c = d.verts[d.tris[d.tri_part == 2]].reshape(-1, 3)
    assert c.min(0).tolist() == [10, 0, 0] and c.max(0).tolist() == [12, 2, 2]
    # L prism has gradient corner colours -> they win; the cube has no materials -> grey
    assert d.corner_colors.shape == (len(d.tris), 3, 3)
    assert not np.allclose(d.corner_colors[d.tri_part == 0], display.GREY)
    assert np.allclose(d.corner_colors[d.tri_part == 2], display.GREY)
    # vertex paint is per corner, not averaged: the L prism's gradient differs inside a triangle
    l0 = d.corner_colors[d.tri_part == 0]
    assert np.abs(l0 - l0[:, :1]).max() > 0.1


def test_display_material_colours_and_empty():
    scene = convert.load(CANISTER)
    d = display.build(scene)
    greens = np.all(np.isclose(d.corner_colors[:, 0], scene.materials[1].color[:3]), axis=1).sum()
    assert greens == 553 and len(d.tris) == 23512
    empty = display.build(type(scene)(unit_m=1e-3))
    assert empty.empty and len(empty.part_nodes) == 0


def _cylinder(seg=24, split_seam=False):
    """Closed cylinder r=1, h=2 along z: quad sides + n-gon caps, outward winding.
    split_seam: the side's first column duplicated (a UV seam as glTF/OBJ store it)."""
    from mesh2mesh.model import Mesh
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ring = np.stack([np.cos(a), np.sin(a), np.zeros(seg)], 1)
    pos = np.vstack([ring, ring + (0, 0, 2)])
    side_next = [(i + 1) % seg for i in range(seg)]
    if split_seam:
        pos = np.vstack([pos, pos[[0, seg]]])
        side_next[-1] = 2 * seg     # the last quad closes on the duplicates
    faces = [[i, side_next[i], side_next[i] + seg if side_next[i] < seg else 2 * seg + 1,
              i + seg] for i in range(seg)]
    faces += [list(range(seg))[::-1], list(range(seg, 2 * seg))]
    return Mesh("cyl", pos, np.array([len(f) for f in faces]), np.concatenate(faces))


def _cyl_scene(mesh, matrices):
    from mesh2mesh.model import Node, Scene
    return Scene(unit_m=1e-3, meshes=[mesh],
                 nodes=[Node(f"n{i}", m, mesh=0) for i, m in enumerate(matrices)])


@pytest.mark.parametrize("split", [False, True])
def test_auto_smooth_cylinder(split):
    """Auto 30°: the sides are smooth (radial normals), the caps and their rims stay sharp;
    a UV-seam split does not leave a crease."""
    d = display.build(_cyl_scene(_cylinder(split_seam=split), [np.eye(4)]))
    n = d.normals(display.SMOOTH_AUTO)
    p = d.verts[d.tris]
    assert np.allclose(np.linalg.norm(n, axis=-1), 1)
    cap = np.all(np.isclose(p[..., 2], p[:, :1, 2]), axis=1)
    side = ~cap
    radial = p[side] * (1, 1, 0)
    assert np.allclose(n[side], radial, atol=1e-9)
    assert np.allclose(np.abs(n[cap][..., 2]), 1)
    flat = d.normals(display.SMOOTH_FLAT)
    assert np.allclose(flat[:, 0], flat[:, 1]) and not np.allclose(flat[side], radial, atol=1e-3)
    # no normals in the file: "from file" is flat, honestly
    assert np.array_equal(d.normals(display.SMOOTH_FILE), flat)
    assert d.meshes_without_normals == (1, 1)
    assert "нормалей нет" in report.normals_note(d)


def test_auto_smooth_angle_threshold():
    """6 segments (60° between sides) are sharp at 30°; 12 segments — exactly 30°, a tie
    that float noise must not split — are smooth all round."""
    d = display.build(_cyl_scene(_cylinder(6), [np.eye(4)]))
    assert np.array_equal(d.normals(display.SMOOTH_AUTO), d.normals(display.SMOOTH_FLAT))
    mesh = _cylinder(12)
    mesh.positions = mesh.positions.astype(np.float32).astype(float)   # as read from a file
    d = display.build(_cyl_scene(mesh, [np.eye(4)]))
    p = d.verts[d.tris]
    side = ~np.all(np.isclose(p[..., 2], p[:, :1, 2]), axis=1)
    assert np.allclose(d.normals(display.SMOOTH_AUTO)[side], p[side] * (1, 1, 0), atol=1e-6)


def test_normals_mirror_and_file():
    """Mirrored instance: outward normals (bake flips winding); file normals go through the
    inverse transpose of a non-uniform scale."""
    mesh = _cylinder()
    mirror = np.diag([-1.0, 2.0, 1.0, 1.0])
    mirror[0, 3] = 10
    d = display.build(_cyl_scene(mesh, [np.eye(4), mirror]))
    for mode in (display.SMOOTH_FLAT, display.SMOOTH_AUTO):
        n = d.normals(mode)
        p = d.verts[d.tris]
        centre = np.array([[0, 0, 1], [10, 0, 1]])[d.tri_part]
        out = np.einsum("tki,tki->tk", n, p - centre[:, None])
        assert (out > 0).all(), mode          # every corner normal points out of its part
    # file normals: radial on the sides, ±z on the caps (as an author would smooth it)
    corners = mesh.positions[mesh.corners]
    sizes = np.repeat(mesh.face_sizes, mesh.face_sizes)
    mesh.normals = np.where(sizes[:, None] == 4, corners * (1, 1, 0),
                            np.where(corners[:, 2:] > 1, (0, 0, 1), (0, 0, -1))).astype(float)
    d = display.build(_cyl_scene(mesh, [np.eye(4), mirror]))
    n = d.normals(display.SMOOTH_FILE)
    side = ~np.all(np.isclose(d.verts[d.tris][..., 2], d.verts[d.tris][:, :1, 2]), axis=1)
    mir = side & (d.tri_part == 1)
    q = d.verts[d.tris][mir] - (10, 0, 0)
    expect = q * (1, 0.25, 0)                 # radial of an ellipse x² + (y/2)² = 1
    expect /= np.linalg.norm(expect, axis=-1, keepdims=True)
    assert np.allclose(n[mir], expect)
    assert np.allclose(n[side & (d.tri_part == 0)], d.verts[d.tris][side & (d.tri_part == 0)] * (1, 1, 0))
    assert d.meshes_without_normals == (0, 1)
    assert "есть у всех" in report.normals_note(d)


def test_polygon_edges_quad():
    from mesh2mesh.model import Mesh
    m = Mesh("q", np.zeros((4, 3)), np.array([4]), np.array([0, 1, 2, 3]))
    assert display.polygon_edges(m).tolist() == [[0, 1], [0, 3], [1, 2], [2, 3]]


def test_grid_step():
    assert display.grid_step(np.array([87.5, 87.4, 36.3])) == 10
    assert display.grid_step(np.array([0.0875, 0.08, 0.03])) == pytest.approx(0.01)
    assert display.grid_step(None) == 1.0


# --- report texts -----------------------------------------------------------------------------

def test_plural_and_formats():
    assert report.plural(1, "меш|меша|мешей") == "1 меш"
    assert report.plural(3, "меш|меша|мешей") == "3 меша"
    assert report.plural(11, "меш|меша|мешей") == "11 мешей"
    assert report.plural(23512, "т|та|тов") == "23 512 тов"
    assert report.plural(21, "т|тов") == "21 тов" and report.plural(1, "т|тов") == "1 т"
    assert report.fmt_len(87.4999) == "87.5"
    assert report.fmt_len(1234.5) == "1 234"
    assert report.fmt_scale(0.001) == "0.001" and report.fmt_scale(10 * 10) == "100"


@pytest.mark.parametrize("text,value", [("10", 10), ("25,4", 25.4), ("×10", 10), ("1e-3", 0.001),
                                        (" 2 ", 2), ("0", None), ("-1", None), ("abc", None),
                                        ("inf", None), ("", None)])
def test_parse_scale(text, value):
    assert report.parse_scale(text) == value


def test_size_in_units_scale_rotation():
    scene = convert.load(CANISTER)
    d = display.build(scene)
    mm = report.size_in(d, scene.unit_m, "mm")
    assert mm == pytest.approx([87.5, 87.44, 36.34], abs=0.01)
    assert report.size_in(d, scene.unit_m, "cm", 10) == pytest.approx(mm)
    from mesh2mesh import geom
    turned = report.size_in(d, scene.unit_m, "mm", rotation=geom.quarter_turn("x", 1))
    assert turned == pytest.approx([mm[0], mm[2], mm[1]])
    assert report.size_line(None, "mm") == "В файле нет геометрии"


def test_file_summary():
    scene = convert.load(CANISTER)
    lines = report.file_summary(scene, CANISTER, True, display.build(scene))
    assert lines[0] == "3MF: единицы из файла — мм."
    assert lines[1] == "2 меша, 2 экземпляра, 23 512 треугольников, 2 материала."
    assert "единиц в файле нет" in report.file_summary(scene, "x.stl", False,
                                                       display.build(scene))[0]


def test_result_summary_levels(tmp_path):
    reader = ["3MF: метаданные"]
    res = convert.Result(str(tmp_path / "x.stl"), warnings=reader + ["STL: материалы отброшены"])
    lines = report.result_summary(res, reader)
    assert lines[0][0] == report.OK and "проверка чтением пройдена" in lines[0][1].lower()
    assert (report.WARN, "STL: материалы отброшены") in lines
    assert lines[-1] == (report.WARN, "При чтении исходника: 3MF: метаданные")
    res.mismatches = ["треугольников 12 -> 10"]
    res.warnings = []
    lines = report.result_summary(res)
    assert lines[0][0] == report.ERROR
    assert (report.ERROR, "Расхождение: треугольников 12 -> 10") in lines
    assert lines[-1] == (report.OK, "Всё прочитанное перенесено без потерь.")


def test_losses_summary():
    assert report.losses_summary([], ".3mf")[0][0] == report.OK
    lines = report.losses_summary(["a", "b"], ".stl")
    assert [lv for lv, _ in lines] == [report.WARN] * 3 and lines[0][1].startswith("STL")


def test_out_path_helpers():
    assert report.default_out_path(r"C:\a\model.3mf", ".glb") == r"C:\a\model.glb"
    assert report.default_out_path(r"C:\a\model.3mf", ".glb", r"D:\out") == r"D:\out\model.glb"
    assert report.fix_out_path(r"C:\a\x.obj", ".stl") == (r"C:\a\x.obj", ".obj")
    assert report.fix_out_path(r"C:\a\x.v2", ".stl") == (r"C:\a\x.v2.stl", ".stl")
    assert report.fix_out_path(r"C:\a\x", ".stl") == (r"C:\a\x.stl", ".stl")


def test_formats_offered_are_all_writers():
    assert set(report.FORMATS) == set(formats.WRITERS)
    assert set(report.UNIT_NAMES) == set(convert.UNITS)


# --- view: panning ----------------------------------------------------------------------------

@pytest.mark.parametrize("az,el", [(-60, 22), (0, 0), (90, -30), (135, 89.9), (10, 90)])
def test_pan_follows_cursor_in_screen_plane(az, el):
    """The old look-at point moves on screen exactly by the drag (grab-style), any camera."""
    right, up, _ = display.camera_axes(az, el)
    assert np.all(np.isfinite(up)) and abs(np.linalg.norm(up) - 1) < 1e-9
    for dx, dy in ((7, 0), (0, -5), (3, 4)):
        d = display.pan_offset(az, el, dx, dy, 0.01)
        old = -d   # old look-at point relative to the new one
        assert np.allclose([old @ right / 0.01, -(old @ up) / 0.01], [dx, dy])


def test_pan_reaches_z_from_side_and_floor_stays_flat():
    d = display.pan_offset(-90, 0, 0, -10, 0.01)          # side view, drag up
    assert np.allclose(d, [0, 0, -0.1])                   # look-at goes down: model goes up
    f = display.pan_offset(-60, 22, 5, -10, 0.01, floor=True)
    assert f[2] == 0 and np.linalg.norm(f) > 0
