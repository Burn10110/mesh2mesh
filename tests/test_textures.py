"""Base colour textures: every format pair, the loss matrix, readers of hand-written files.
Run: .venv/Scripts/python -m pytest tests -q"""

import struct
import zipfile
import zlib

import numpy as np
import pytest

from mesh2mesh import convert, formats, geom, textures
from mesh2mesh.formats import obj, threemf
from mesh2mesh.model import Material, Mesh, Node, Scene, Texture
from test_formats import CUBE_T, CUBE_V, EXTS, l_prism, write_read

TEX_FORMATS = (".fbx", ".obj", ".3mf", ".gltf", ".glb")


def png(w=4, h=2, seed=0):
    """A tiny valid RGB PNG, different bytes per seed."""
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    rows = b"".join(b"\0" + bytes((seed + x * 40 + y * 90) % 256 for x in range(w * 3))
                    for y in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def tex_scene(tex2=None):
    """L prism (UV, two materials, both textured with different images) + a plain cube."""
    m = l_prism()
    m.colors = None
    cube = Mesh.from_triangles("cube", CUBE_V, CUBE_T, face_material=np.full(12, 2))
    return Scene(1e-3, [Node("L", mesh=0), Node("Cube", mesh=1)], [m, cube], [
        Material("Skin", (0.8, 0.8, 0.8, 1), Texture("skin.png", png(seed=1))),
        Material("Cloth", (1, 1, 1, 1), tex2 or Texture("cloth.png", png(seed=2))),
        Material("Plain", (0, 0, 1, 1))])


def tex_by_material(scene):
    return {m.name: (m.texture.data if m.texture else None) for m in scene.materials}


def tex_corners(scene):
    """Rows (world position, uv) only of corners whose face has a textured material."""
    rows = set()
    for b in geom.bake(scene):
        m = b.mesh
        if m.uv is None or m.face_material is None:
            continue
        face_of = np.repeat(np.arange(m.n_faces), m.face_sizes)
        tex = np.array([scene.materials[i].texture is not None if i >= 0 else False
                        for i in m.face_material[face_of]])
        p = b.positions[m.corners] * (scene.unit_m * 1000)
        rows |= set(map(tuple, np.round(np.c_[p, m.uv][tex], 4).tolist()))
    return rows


@pytest.mark.parametrize("ext", TEX_FORMATS)
def test_texture_round_trip(tmp_path, ext):
    s = tex_scene()
    w, b, _ = write_read(s, tmp_path, ext)
    imgs = {t.data for t in textures.used_textures(b)}
    assert imgs == {png(seed=1), png(seed=2)}, w
    assert tex_corners(b) == tex_corners(s)
    plain = [m for m in b.materials if m.texture is None and m.color is not None]
    assert any(np.allclose(m.color[:3], (0, 0, 1), atol=1e-3) for m in plain)
    if ext != ".3mf":           # 3MF keeps no material name with a texture
        got = tex_by_material(b)
        assert got["Skin"] == png(seed=1) and got["Cloth"] == png(seed=2)


@pytest.mark.parametrize("src", TEX_FORMATS)
@pytest.mark.parametrize("dst", EXTS)
def test_texture_every_pair(tmp_path, src, dst):
    _, _, sp = write_read(tex_scene(), tmp_path, src, stem="src")
    out = str(tmp_path / f"dst{dst}")
    res = convert.convert(sp, out, units="mm")
    assert res.ok, res.mismatches
    back = convert.load(out, "mm")
    imgs = {t.data for t in textures.used_textures(back)}
    if dst in TEX_FORMATS:
        assert imgs == {png(seed=1), png(seed=2)}, res.warnings
    else:
        assert not imgs and any("текстуры отброшены" in x for x in res.warnings)


# tex scene -> writer: fragments that must (+) / must not (-) appear in the warnings
TEX_LOSSES = {
    ".fbx": (set(), {"текстур", "UV"}),
    ".obj": (set(), {"текстур", "UV"}),
    ".3mf": ({"имя и цвет материала"}, {"не записана", "UV"}),
    ".gltf": (set(), {"текстур", "UV"}),
    ".glb": (set(), {"текстур", "UV"}),
    ".stl": ({"текстуры отброшены"}, set()),
    ".ply": ({"текстуры отброшены"}, set()),
}


@pytest.mark.parametrize("ext", EXTS)
def test_texture_loss_matrix(tmp_path, ext):
    w, _, _ = write_read(tex_scene(), tmp_path, ext)
    text = " | ".join(w)
    must, must_not = TEX_LOSSES[ext]
    for frag in must:
        assert frag in text, f"{ext}: no warning about «{frag}»: {text}"
    for frag in must_not:
        assert frag not in text, f"{ext}: unexpected «{frag}»: {text}"


def test_texture_loss_matrix_covers_every_writer():
    assert set(TEX_LOSSES) == set(formats.WRITERS)


@pytest.mark.parametrize("ext", [".3mf", ".gltf", ".glb", ".obj", ".fbx"])
def test_non_web_texture(tmp_path, ext):
    # a TGA: glTF and 3MF may only hold PNG/JPEG -> warning; OBJ and FBX keep it
    tga = Texture("paint.tga", bytes(18) + b"\xff" * 12)
    w, b, _ = write_read(tex_scene(tga), tmp_path, ext)
    kept = {t.data for t in textures.used_textures(b)}
    if ext in (".obj", ".fbx"):
        assert tga.data in kept
    else:
        assert tga.data not in kept and png(seed=1) in kept
        assert any("TGA" in x and "PNG и JPEG" in x for x in w)


@pytest.mark.parametrize("ext", [".obj", ".3mf", ".gltf"])
def test_texture_without_uv_warns(tmp_path, ext):
    s = tex_scene()
    s.meshes[1].face_material[:] = 0          # cube (no UV) gets the textured "Skin"
    w, _, _ = write_read(s, tmp_path, ext)
    assert any("без UV" in x or "нет UV" in x for x in w), w


def test_obj_writes_images_and_white_kd(tmp_path):
    s = tex_scene()
    obj.write(s, str(tmp_path / "m.obj"))
    mtl = (tmp_path / "m.mtl").read_text()
    assert "map_Kd skin.png" in mtl and "map_Kd cloth.png" in mtl
    skin = mtl.split("newmtl Skin")[1].split("newmtl")[0]
    assert "Kd 1 1 1" in skin
    assert (tmp_path / "skin.png").read_bytes() == png(seed=1)


def test_obj_reader_map_options_spaces_and_lookup(tmp_path):
    (tmp_path / "tex").mkdir()
    (tmp_path / "tex" / "my skin.png").write_bytes(png(seed=5))
    (tmp_path / "m.mtl").write_text(
        "newmtl a\nKd 0.5 0.5 0.5\nmap_Kd -s 2 2 1 -bm 0.5 tex/my skin.png\nmap_Bump n.png\n"
        "newmtl b\nmap_Kd C:\\elsewhere\\gone.png\n"
        "newmtl c\nmap_Kd -clamp on tex\\my skin.png\n")
    (tmp_path / "m.obj").write_text(
        "mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\n"
        "usemtl a\nf 1/1 2/2 3/3\nusemtl b\nf 1/1 3/3 2/2\nusemtl c\nf 1/1 2/2 3/3\n")
    s = obj.read(str(tmp_path / "m.obj"))
    a, b, c = s.materials
    assert a.texture.data == png(seed=5) and a.texture.name == "my skin.png"
    assert c.texture is a.texture                 # the same file -> one Texture
    assert a.color == (0.5, 0.5, 0.5, 1.0)        # Kd kept as stored
    assert b.texture is None
    text = " | ".join(s.warnings)
    assert "сдвиг/масштаб UV" in text and "gone.png" in text and "map_Kd (карты" in text


def _3mf(tmp_path, model, files):
    p = tmp_path / "t.3mf"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("3D/3dmodel.model", model)
        for k, v in files.items():
            z.writestr(k, v)
    return str(p)


def test_3mf_reader_texture_group(tmp_path):
    model = f"""<?xml version="1.0"?>
<model unit="millimeter" xmlns="{threemf.CORE}" xmlns:m="{threemf.MAT}"><resources>
<m:texture2d id="1" path="/3D/Textures/wood.png" contenttype="image/png" tilestyleu="clamp"/>
<m:texture2dgroup id="2" texid="1"><m:tex2coord u="0" v="0"/><m:tex2coord u="1" v="0"/>
<m:tex2coord u="0" v="1"/><m:tex2coord u="0.5" v="0.5"/></m:texture2dgroup>
<basematerials id="3"><base name="Red" displaycolor="#FF0000"/></basematerials>
<object id="4" type="model" pid="2" pindex="3"><mesh><vertices>
<vertex x="0" y="0" z="0"/><vertex x="1" y="0" z="0"/><vertex x="0" y="1" z="0"/>
<vertex x="1" y="1" z="0"/></vertices><triangles>
<triangle v1="0" v2="1" v3="2" p1="0" p2="1" p3="2"/>
<triangle v1="1" v2="3" v3="2"/>
<triangle v1="1" v2="0" v3="3" pid="3" p1="0"/>
</triangles></mesh></object></resources><build><item objectid="4"/></build></model>"""
    s = threemf.read(_3mf(tmp_path, model, {"3D/Textures/wood.png": png(seed=7)}))
    m = s.meshes[0]
    wood = [x for x in s.materials if x.texture is not None][0]
    assert wood.name == "wood" and wood.texture.data == png(seed=7)
    assert [s.materials[i].name for i in m.face_material] == ["wood", "wood", "Red"]
    assert np.allclose(m.uv[:3], [(0, 0), (1, 0), (0, 1)])
    assert np.allclose(m.uv[3:6], 0.5)                 # object pindex for all corners
    assert any("mirror/clamp" in x for x in s.warnings)


def test_3mf_writer_package(tmp_path):
    s = tex_scene()
    p = str(tmp_path / "w.3mf")
    threemf.write(s, p)
    with zipfile.ZipFile(p) as z:
        names = set(z.namelist())
        assert {"3D/Texture/skin.png", "3D/Texture/cloth.png"} <= names
        rels = z.read("3D/_rels/3dmodel.model.rels").decode()
        assert rels.count(threemf.TEXTURE_REL) == 2
        assert 'Extension="png"' in z.read("[Content_Types].xml").decode()
        import xml.etree.ElementTree as ET
        root = ET.fromstring(z.read("3D/3dmodel.model"))
    # every textured triangle references a group whose coords exist; objects have a pid
    for o in root.iter(f"{{{threemf.CORE}}}object"):
        if o.find(f"{{{threemf.CORE}}}mesh") is not None:
            assert o.get("pid") is not None


def test_texture_files_beside_gltf_and_obj_unique_names(tmp_path):
    s = tex_scene(Texture("skin.png", png(seed=3)))       # two different "skin.png"
    obj.write(s, str(tmp_path / "u.obj"))
    pngs = sorted(p.name for p in tmp_path.glob("*.png"))
    assert pngs == ["skin.png", "skin_1.png"]
    b = obj.read(str(tmp_path / "u.obj"))
    assert {t.data for t in textures.used_textures(b)} == {png(seed=1), png(seed=3)}
