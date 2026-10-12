"""Iteration 2: 3MF writer, PLY, glTF/GLB, every format pair, the loss matrix.
Run: .venv/Scripts/python -m pytest tests -q"""

import base64
import json
import os
import struct
import zipfile

import numpy as np
import pytest

from mesh2mesh import convert, formats, geom
from mesh2mesh.formats import gltf, ply, threemf
from mesh2mesh.model import Material, Mesh, Node, Scene

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANISTER = os.path.join(ROOT, "samples", "Mata_Canister_CAP_Scaled.3mf")
EXTS = sorted(formats.WRITERS)

L = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]          # counter-clockwise from +z
CUBE_V = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
CUBE_T = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
          (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]


def l_prism(name="L"):
    """Closed L-shaped prism: two hexagons + six quads, flat normals, UV, gradient colours,
    material 0 on the caps and 1 on the sides. Volume 3."""
    pos = np.array([(x, y, z) for z in (0, 1) for x, y in L], dtype=np.float64)
    faces = [list(range(5, -1, -1)), list(range(6, 12))]
    faces += [[i, (i + 1) % 6, (i + 1) % 6 + 6, i + 6] for i in range(6)]
    corners = np.array([c for f in faces for c in f])
    sizes = np.array([len(f) for f in faces])
    nrm = []
    for f in faces:
        n = geom.newell_normal(pos[f])
        nrm += [n / np.linalg.norm(n)] * len(f)
    p = pos[corners]
    return Mesh(name, pos, sizes, corners, face_material=np.array([0, 0] + [1] * 6),
                uv=p[:, :2] / 2 + p[:, 2:] * 0.25, normals=np.array(nrm),
                colors=np.c_[p[:, 0] / 2, p[:, 1] / 2, p[:, 2], np.ones(len(p))])


def rich_scene():
    """Hierarchy, instancing, n-gons, every attribute; volume 3 + 3 + 8 = 14 mm^3."""
    rot = np.eye(4)
    rot[:3, :3] = geom.quarter_turn("z", 1)
    rot[:3, 3] = (0, 5, 0)
    big = np.diag([2.0, 2, 2, 1])
    root = Node("Assembly", np.eye(4), children=[
        Node("Part A", mesh=0), Node("Part B", rot, mesh=0),
        Node("Group", children=[Node("Part C", big, mesh=1)])])
    root.matrix[:3, 3] = (10, 0, 0)
    return Scene(unit_m=1e-3, nodes=[root],
                 meshes=[l_prism(), Mesh.from_triangles("cube", CUBE_V, CUBE_T)],
                 materials=[Material("Red", (1, 0, 0, 1)), Material("Blue paint", (0, 0, 1, 0.5))])


def write_read(scene, tmp_path, ext, stem="x", **opts):
    p = str(tmp_path / f"{stem}{ext}")
    unit = formats.FIXED_UNITS.get(ext, "mm")
    w = formats.WRITERS[ext](scene, p, scale=scene.unit_m / convert.UNITS[unit],
                             unit_m=convert.UNITS[unit], **opts)
    return w, convert.load(p, unit), p


def corner_rows(scene, decimals=4, attrs=("uv", "normals", "colors")):
    """Set of (world position, attributes...) per corner: order- and split-independent."""
    rows = set()
    for b in geom.bake(scene):
        m = b.mesh
        cols = [b.positions[m.corners] * (scene.unit_m * 1000)]
        for a in attrs:
            v = getattr(m, a)
            if v is not None:
                cols.append(geom.apply_normals(b.matrix, v) if a == "normals" else v)
        rows |= set(map(tuple, np.round(np.concatenate(cols, 1), decimals).tolist()))
    return rows


# --- every pair -----------------------------------------------------------------------------

@pytest.mark.parametrize("src", EXTS)
@pytest.mark.parametrize("dst", EXTS)
def test_every_pair_reads_back(tmp_path, src, dst):
    _, _, sp = write_read(rich_scene(), tmp_path, src, stem="src")
    res = convert.convert(sp, str(tmp_path / f"dst{dst}"), units="mm")
    assert res.ok, res.mismatches
    back = convert.load(str(tmp_path / f"dst{dst}"), "mm")
    assert geom.stats(back).volume == pytest.approx(14e-9, rel=1e-5)


@pytest.mark.parametrize("ext", EXTS)
def test_empty_mesh_is_never_dropped_silently(tmp_path, ext):
    s = Scene()
    s.meshes = [Mesh.from_triangles("tri", [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]]),
                Mesh("empty", np.zeros((0, 3)), np.zeros(0, np.int64), np.zeros(0, np.int64))]
    s.nodes = [Node("solid", mesh=0), Node("hollow", mesh=1)]
    s.unit_m = 1e-3
    w, back, _ = write_read(s, tmp_path, ext)
    assert any("пустые меши" in line for line in w), w
    assert sum(back.meshes[n.mesh].n_faces for n, _ in back.instances()) == 1


@pytest.mark.parametrize("ext", [".stl", ".ply"])
def test_text_variants(tmp_path, ext):
    w, back, _ = write_read(rich_scene(), tmp_path, ext, binary=False)
    assert not geom.compare(geom.stats(rich_scene()), geom.stats(back), float32=True)


@pytest.mark.skipif(not os.path.exists(CANISTER), reason="sample missing")
@pytest.mark.parametrize("dst", EXTS)
def test_canister_to_every_format(tmp_path, dst):
    res = convert.convert(CANISTER, str(tmp_path / f"c{dst}"))
    assert res.ok, res.mismatches
    back = convert.load(str(tmp_path / f"c{dst}"))
    if dst in (".3mf", ".gltf", ".glb", ".obj"):
        assert sorted(m.name for m in back.materials) == ["Default", "Green"]


# --- loss matrix ----------------------------------------------------------------------------

# rich scene -> writer: fragments that must (+) / must not (-) appear in the warnings
LOSSES = {
    ".fbx": (set(), {"иерархия", "UV", "нормали", "многоугольники", "материал", "цвет",
                     "shear"}),
    ".obj": ({"иерархия"}, {"UV", "нормали", "многоугольники", "материал"}),
    ".stl": ({"многоугольники", "материалы", "цвета вершин", "UV", "нормали", "иерархия",
              "слиты"}, set()),
    ".3mf": ({"многоугольники", "UV", "нормали"}, {"иерархия", "материал", "цвет"}),
    ".ply": ({"слиты", "иерархия", "материалов нет", "UV есть не у всех",
              "нормали есть не у всех"}, {"многоугольники"}),
    ".gltf": ({"многоугольники", "белым"}, {"иерархия", "UV", "нормали"}),
    ".glb": ({"многоугольники", "белым"}, {"иерархия", "UV", "нормали"}),
}


@pytest.mark.parametrize("ext", EXTS)
def test_loss_matrix(tmp_path, ext):
    w, _, _ = write_read(rich_scene(), tmp_path, ext)
    text = " | ".join(w)
    must, must_not = LOSSES[ext]
    for frag in must:
        assert frag in text, f"{ext}: no warning about «{frag}»: {text}"
    for frag in must_not:
        assert frag not in text, f"{ext}: unexpected «{frag}»: {text}"


def test_loss_matrix_covers_every_writer():
    assert set(LOSSES) == set(formats.WRITERS)


# --- what each format keeps -----------------------------------------------------------------

def test_obj_bare_faces_after_material_stay_bare(tmp_path):
    # usemtl persists across `o`: the material-less cube after "Blue paint" must not inherit it
    _, b, _ = write_read(rich_scene(), tmp_path, ".obj")
    cube = [m for m in b.meshes if m.name == "Part C"][0]
    assert {b.materials[i].name for i in cube.face_material} == {"no_material"}

@pytest.mark.parametrize("block", [
    b"f 1 2 3\nf 4//5 6//7 8//9 10//11",
    b"f 1/2 3/4 5/6\nf 7/8/9 10/11/12 13/14/15 16/17/18",
    b"f -1/-2/-3 007/0/-0 123456789/9876543210/123456789012345678",
    b"f 1/2/ 3/4/ 5/6/",                        # trailing slashes: token path
    b"f 1// 2// 3//",
    b"f /1 2 3",                                # empty v: both decline
    b"f 1/2/3/4 5 6",                           # four slots: both decline
    b"f  1 2 3\nf\t4 5 6",                      # spacing after f: token path counts
    b"f 1 2\nf \nf 3 4 5",                      # short lines: sizes 2, 0, 3
    b"f 1-2 3 4", b"f --1 2 3", b"f 1 2 -",     # odd minus
    b"f 1 2 3 1234567890123456789",             # 19 digits: token path (int64 check)
])
def test_obj_face_bytes_matches_tokens(block):
    from mesh2mesh.formats import obj
    n = block.count(b"\n") + 1
    fast, ref = obj._face_bytes(block, n), obj._face_tokens(block, n)
    if fast is None:
        return                                  # the token path decides
    assert ref is not None
    assert np.array_equal(fast[0], ref[0])
    keep = np.repeat(fast[0] >= 3, fast[0])     # corners of lines the reader keeps
    for x, y in zip(fast[1:], ref[1:]):
        assert np.array_equal(x[keep], y[keep])


def test_3mf_keeps_hierarchy_instances_names_materials_colours(tmp_path):
    s = rich_scene()
    w, b, p = write_read(s, tmp_path, ".3mf")
    # an instance in 3MF is a component without a name of its own: "Part B" shows the object's
    assert [n.name for n, *_ in b.walk()] == ["Assembly", "Part A", "Part A", "Group", "Part C"]
    assert any("имя первого" in x for x in w)
    assert len(b.meshes) == 2 and sum(1 for _ in b.instances()) == 3
    with zipfile.ZipFile(p) as z:
        model = z.read("3D/3dmodel.model").decode()
    assert 'unit="millimeter"' in model and "m:colorgroup" in model
    # gradient colours (8-bit) and positions per corner survive
    assert corner_rows(b, 2, ("colors",)) == corner_rows(s, 2, ("colors",))


def test_3mf_materials_without_colours_use_basematerials(tmp_path):
    s = rich_scene()
    s.meshes[0].colors = None
    _, b, p = write_read(s, tmp_path, ".3mf")
    assert sorted(m.name for m in b.materials) == ["Blue paint", "Red"]
    assert [m.color for m in b.materials if m.name == "Blue paint"][0][3] == pytest.approx(0.5, abs=1e-2)
    with zipfile.ZipFile(p) as z:
        model = z.read("3D/3dmodel.model").decode()
    assert "colorgroup" not in model
    # every object whose triangles carry a property also has an object-level pid, and every
    # such triangle its own pid (lib3mf ignores p1 without it: knowledge/3mf-textures.md)
    import xml.etree.ElementTree as ET
    for o in ET.fromstring(model).iter(f"{{{threemf.CORE}}}object"):
        tris = o.findall(".//{%s}triangle" % threemf.CORE)
        if any(t.get("p1") is not None for t in tris):
            assert o.get("pid") is not None
            assert all(t.get("pid") is not None for t in tris)


def test_3mf_units_inch(tmp_path):
    s = rich_scene()
    p = str(tmp_path / "i.3mf")
    threemf.write(s, p, scale=1e-3 / 0.0254, unit_m=0.0254)
    b = threemf.read(p)
    assert b.unit_m == 0.0254
    assert not geom.compare(geom.stats(s), geom.stats(b))


def test_3mf_mirror_is_baked(tmp_path):
    s = rich_scene()
    s.nodes[0].children[0].matrix = np.diag([-1.0, 1, 1, 1])
    w, b, _ = write_read(s, tmp_path, ".3mf")
    assert any("зеркальные" in x for x in w)
    assert all(np.allclose(n.matrix, np.eye(4)) for n, *_ in b.walk())
    assert geom.stats(b).volume == pytest.approx(14e-9, rel=1e-6)


@pytest.mark.parametrize("ext", [".gltf", ".glb"])
def test_gltf_keeps_everything_but_ngons(tmp_path, ext):
    s = rich_scene()
    _, b, _ = write_read(s, tmp_path, ext)
    assert b.unit_m == 1.0
    assert [n.name for n, *_ in b.walk()] == ["Assembly", "Part A", "Part B", "Group", "Part C"]
    assert len(b.meshes) == 2 and sum(1 for _ in b.instances()) == 3
    assert {m.name for m in b.materials} == {"Red", "Blue paint"}
    m = b.meshes[0]
    assert m.uv is not None and m.normals is not None and m.colors is not None
    # positions, UV, normals, colours per corner (sRGB <-> linear and float32 round trip)
    assert corner_rows(b, 3) == corner_rows(s, 3)


def test_gltf_material_colour_srgb_round_trip(tmp_path):
    s = rich_scene()
    s.meshes[0].colors = None
    _, b, p = write_read(s, tmp_path, ".gltf")
    doc = json.load(open(p, encoding="utf-8"))
    red = [m for m in doc["materials"] if m["name"] == "Blue paint"][0]
    assert red["pbrMetallicRoughness"]["baseColorFactor"] == [0, 0, 1, 0.5]
    assert red["alphaMode"] == "BLEND"
    grey = Scene(1e-3, [Node("g", mesh=0)], [Mesh.from_triangles("g", CUBE_V, CUBE_T,
                                                               face_material=np.zeros(12, int))],
                 [Material("grey", (0.5, 0.5, 0.5, 1))])
    _, g, p = write_read(grey, tmp_path, ".gltf", stem="g")
    f = json.load(open(p, encoding="utf-8"))["materials"][0]["pbrMetallicRoughness"]
    assert f["baseColorFactor"][0] == pytest.approx(0.21404, abs=1e-5)    # linear of sRGB 0.5
    assert g.materials[0].color[0] == pytest.approx(0.5)


def test_gltf_mirror_matrix_kept(tmp_path):
    s = rich_scene()
    s.nodes[0].children[0].matrix = np.diag([-1.0, 1, 1, 1])
    w, b, _ = write_read(s, tmp_path, ".glb")
    assert not w or all("зеркал" not in x for x in w)
    assert geom.stats(b).volume == pytest.approx(14e-9, rel=1e-5)


def test_ply_keeps_ngons_uv_normals_colours(tmp_path):
    s = Scene(1e-3, [Node("L", mesh=0)], [l_prism()], [Material("Red", (1, 0, 0, 1)),
                                                      Material("Blue", (0, 0, 1, 1))])
    for binary in (True, False):
        _, b, _ = write_read(s, tmp_path, ".ply", stem=f"p{binary}", binary=binary)
        m = b.meshes[0]
        assert sorted(m.face_sizes.tolist()) == [4] * 6 + [6, 6]
        assert corner_rows(b, 2) == corner_rows(s, 2)


def test_ply_material_colours_become_vertex_colours(tmp_path):
    s = rich_scene()
    s.meshes[0].colors = None
    w, b, _ = write_read(s, tmp_path, ".ply")
    cols = {tuple(c) for c in np.round(b.meshes[0].colors, 3).tolist()}
    assert cols == {(1, 0, 0, 1), (0, 0, 1, 0.502), (1, 1, 1, 1)}    # Red, Blue, cube = white


# --- readers: details other programs produce ------------------------------------------------

def test_ply_reader_ascii_face_colours_and_extras(tmp_path):
    p = tmp_path / "a.ply"
    p.write_text("ply\nformat ascii 1.0\ncomment x\nelement vertex 4\nproperty float x\n"
                 "property float y\nproperty float z\nproperty float quality\n"
                 "element face 2\nproperty list uchar int vertex_indices\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 "element edge 1\nproperty int vertex1\nproperty int vertex2\nend_header\n"
                 "0 0 0 1\n1 0 0 1\n1 1 0 1\n0 1 0 1\n3 0 1 2 255 0 0\n3 0 2 3 0 255 0\n0 1\n")
    s = ply.read(str(p))
    m = s.meshes[0]
    assert m.n_faces == 2 and tuple(m.colors[3]) == (0, 1, 0, 1)
    text = " ".join(s.warnings)
    assert "quality" in text and "edge" in text


def test_ply_reader_big_endian_mixed_faces(tmp_path):
    head = ("ply\nformat binary_big_endian 1.0\nelement vertex 5\nproperty double x\n"
            "property double y\nproperty double z\nelement face 2\n"
            "property list uchar uint vertex_indices\nend_header\n").encode()
    v = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0.5, 2, 0)], ">f8").tobytes()
    f = bytes([4]) + np.array([0, 1, 2, 3], ">u4").tobytes() + bytes([3]) + \
        np.array([3, 2, 4], ">u4").tobytes()
    p = tmp_path / "b.ply"
    p.write_bytes(head + v + f)
    m = ply.read(str(p)).meshes[0]
    assert m.face_sizes.tolist() == [4, 3] and m.positions[4].tolist() == [0.5, 2, 0]


@pytest.mark.parametrize("fast", [True, False])
@pytest.mark.parametrize("fmt", ["binary_little_endian", "ascii"])
def test_ply_reader_broken_files_clear_errors(tmp_path, monkeypatch, fmt, fast):
    """Negative list count / truncated data -> a 'PLY: …' error, the same with and without
    the fast paths (np.frombuffer(count=-1) used to read the rest of the file silently)."""
    monkeypatch.setattr(ply, "FAST", fast)
    head = (f"ply\nformat {fmt} 1.0\nelement vertex 3\nproperty float x\nproperty float y\n"
            "property float z\nelement face 2\nproperty list char int vertex_indices\n"
            "end_header\n").encode()
    if fmt == "ascii":
        v = b"0 0 0\n1 0 0\n0 1 0\n"
        neg, short = v + b"-1 0 1 2\n3 0 1 2\n", v + b"3 0 1 2\n3 0\n"
    else:
        v = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0)], "<f4").tobytes()
        tri = np.array([3], "<i1").tobytes() + np.array([0, 1, 2], "<i4").tobytes()
        neg = v + np.array([-1], "<i1").tobytes() + np.array([0, 1, 2], "<i4").tobytes() + tri
        short = v + tri + tri[:6]
    for body, words in ((neg, "отрицательная длина"), (short, "повреждены или обрезаны")):
        p = tmp_path / "x.ply"
        p.write_bytes(head + body)
        with pytest.raises(ValueError, match=words):
            ply.read(str(p))


def _gltf_doc(blob, **extra):
    doc = {"asset": {"version": "2.0"}, "buffers": [{
        "byteLength": len(blob),
        "uri": "data:application/octet-stream;base64," + base64.b64encode(blob).decode()}]}
    doc.update(extra)
    return doc


def test_gltf_reader_strip_trs_normalized_colours_sparse(tmp_path):
    # quad as a 4-vertex strip in glTF axes (y up): x 0..1, z 0..-1 -> scene y 0..1
    pos = np.array([(0, 0, 0), (1, 0, 0), (0, 0, -1), (1, 0, -1)], "<f4")
    col = np.array([(255, 0, 0, 255)] * 4, "<u1")
    sp_idx = np.array([3], "<u2")
    sp_val = np.array([(1, 0, -2)], "<f4")
    blob = pos.tobytes() + col.tobytes() + sp_idx.tobytes() + b"\0\0" + sp_val.tobytes()
    doc = _gltf_doc(
        blob,
        bufferViews=[{"buffer": 0, "byteOffset": 0, "byteLength": 48},
                     {"buffer": 0, "byteOffset": 48, "byteLength": 16},
                     {"buffer": 0, "byteOffset": 64, "byteLength": 2},
                     {"buffer": 0, "byteOffset": 68, "byteLength": 12}],
        accessors=[{"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
                    "sparse": {"count": 1,
                               "indices": {"bufferView": 2, "componentType": 5123},
                               "values": {"bufferView": 3}}},
                   {"bufferView": 1, "componentType": 5121, "normalized": True, "count": 4,
                    "type": "VEC4"}],
        meshes=[{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "mode": 5}]}],
        nodes=[{"name": "q", "mesh": 0, "translation": [0, 3, 0],
                "rotation": [0, 0.7071067811865476, 0, 0.7071067811865476]}],
        scenes=[{"nodes": [0]}], scene=0)
    p = tmp_path / "s.gltf"
    p.write_text(json.dumps(doc))
    s = gltf.read(str(p))
    m = s.meshes[0]
    assert m.n_faces == 2 and np.allclose(m.colors, [1, 0, 0, 1])
    st = geom.stats(s)
    assert st.area == pytest.approx(1.5)                 # sparse moved one corner: trapezoid
    assert st.bbox_min[2] == pytest.approx(3) and st.bbox_max[2] == pytest.approx(3)   # y up -> z
    # both strip triangles face the same way (+y in glTF -> -?): consistent winding
    t = geom.bake(s)[0].tri_points
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    assert np.sign(n[0, 2]) == np.sign(n[1, 2])


def test_gltf_reader_refuses_draco(tmp_path):
    p = tmp_path / "d.gltf"
    p.write_text(json.dumps({"asset": {"version": "2.0"},
                             "extensionsRequired": ["KHR_draco_mesh_compression"]}))
    with pytest.raises(ValueError, match="Draco"):
        gltf.read(str(p))


def test_glb_container_layout(tmp_path):
    _, _, p = write_read(rich_scene(), tmp_path, ".glb")
    data = open(p, "rb").read()
    magic, ver, length = struct.unpack_from("<4sII", data)
    assert (magic, ver, length) == (b"glTF", 2, len(data))
    jlen, jtype = struct.unpack_from("<II", data, 12)
    assert jtype == gltf.CHUNK_JSON and jlen % 4 == 0
    blen, btype = struct.unpack_from("<II", data, 20 + jlen)
    assert btype == gltf.CHUNK_BIN and blen % 4 == 0 and 28 + jlen + blen == len(data)
    doc = json.loads(data[20:20 + jlen])
    for a in doc["accessors"]:
        v = doc["bufferViews"][a["bufferView"]]
        assert v["byteOffset"] % 4 == 0
    assert all("min" in doc["accessors"][p["attributes"]["POSITION"]]
               for m in doc["meshes"] for p in m["primitives"])


def test_gltf_units_fixed_to_metres(tmp_path):
    _, _, sp = write_read(rich_scene(), tmp_path, ".3mf", stem="src")
    res = convert.convert(sp, str(tmp_path / "o.glb"), units="mm")
    assert res.ok and any("всегда m" in w for w in res.warnings)
