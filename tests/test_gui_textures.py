"""Textures in the 3D view without GL: Display arrays, draw groups, image decoding (QImage, no
QApplication needed), report lines."""

import struct

import numpy as np
from PySide6.QtCore import QBuffer, QByteArray, QIODevice
from PySide6.QtGui import QColor, QImage

from mesh2mesh import geom
from mesh2mesh.model import Material, Texture
from mesh2mesh_gui import display, report, teximage

from test_textures import tex_scene


def image_bytes(w, h, fill, fmt="PNG", top_left=None, alpha=None):
    img = QImage(w, h, QImage.Format_ARGB32)
    img.fill(QColor(*fill))
    if top_left is not None:
        img.setPixelColor(0, 0, QColor(*top_left))
    if alpha is not None:
        for y in range(h):
            for x in range(w):
                c = img.pixelColor(x, y)
                c.setAlpha(alpha(x, y))
                img.setPixelColor(x, y, c)
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.WriteOnly)
    assert img.save(buf, fmt)
    return bytes(ba)


# --- Display ---------------------------------------------------------------------------------

def test_display_texture_per_triangle():
    scene = tex_scene()
    d = display.build(scene)
    assert [t.name for t in d.textures] == ["skin.png", "cloth.png"]
    start = 0
    for b in geom.bake(scene):
        part = slice(start, start + len(b.tris))
        start += len(b.tris)
        mat = b.mesh.face_material[b.tri_face]
        expect = np.where(mat == 0, 0, np.where(mat == 1, 1, -1))
        assert d.tri_texture[part].tolist() == expect.tolist()
        if b.mesh.uv is not None:
            assert np.allclose(d.corner_uv[part], b.mesh.uv[b.tris])
    assert (d.tri_texture == -1).any() and d.no_uv_textured == 0


def test_display_texture_list_compact_and_shared():
    scene = tex_scene()
    shared = scene.materials[0].texture
    scene.materials[1].texture = shared                       # one image, two materials
    scene.materials.append(Material("Unused", None, Texture("unused.png", b"x")))
    d = display.build(scene)
    assert [t.name for t in d.textures] == ["skin.png"]
    assert set(d.tri_texture.tolist()) == {-1, 0}


def test_display_textured_material_without_uv():
    scene = tex_scene()
    for m in scene.meshes:
        m.uv = None
    d = display.build(scene)
    assert d.corner_uv is None and not d.textures
    assert (d.tri_texture == -1).all() and d.no_uv_textured > 0
    assert any("нет UV" in s for s in report.texture_lines(d))


def test_display_without_textures():
    scene = tex_scene()
    for m in scene.materials:
        m.texture = None
    d = display.build(scene)
    assert d.corner_uv is None and not d.textures and report.texture_lines(d) == []


def test_texture_groups_contiguous():
    tt = np.array([1, -1, 0, 1, 0, -1, 1])
    order, groups = display.texture_groups(tt)
    assert sorted(order.tolist()) == list(range(len(tt)))
    assert groups == [(0, 2, -1), (2, 2, 0), (4, 3, 1)]
    for first, count, i in groups:
        assert (tt[order[first:first + count]] == i).all()
    assert order[:2].tolist() == [1, 5]                        # stable inside a group
    assert display.texture_groups(np.zeros(0, np.int64))[1] == []


# --- decoding --------------------------------------------------------------------------------

def test_decode_bottom_row_first():
    """UV v is up (model.py), GL row 0 is v = 0: the image's top row must come last."""
    im = teximage.decode(Texture("a.png", image_bytes(3, 2, (0, 0, 255), top_left=(255, 0, 0))))
    assert im.pixels.shape == (2, 3, 4) and im.pixels.dtype == np.uint8
    assert im.pixels[-1, 0].tolist() == [255, 0, 0, 255]
    assert im.pixels[0, 0].tolist() == [0, 0, 255, 255]
    assert im.pixels.flags["C_CONTIGUOUS"]


def tga(w, h, rgb):
    """Uncompressed 24-bit TGA (Qt reads TGA but cannot write it)."""
    head = struct.pack("<BBBHHBHHHHBB", 0, 0, 2, 0, 0, 0, 0, 0, w, h, 24, 0)
    return head + bytes(rgb[::-1]) * (w * h)


def test_decode_jpeg_tga():
    for name, data in (("a.jpg", image_bytes(4, 4, (10, 200, 30), "JPEG")),
                       ("a.tga", tga(4, 4, (10, 200, 30)))):
        im = teximage.decode(Texture(name, data))
        assert im.pixels is not None, name
        assert np.abs(im.pixels[1, 1, :3].astype(int) - (10, 200, 30)).max() < 8


def tga_file(px, kind=2, bits=32, top=False, rle=False, palette=None, footer=False):
    """Encode (h, w, 4) RGBA rows top first as TGA. kind 2 truecolor, 3 gray (uses R, and A
    at 16 bits), 1 palette (px = (h, w) indices, palette (k, 3) RGB)."""
    h, w = px.shape[:2]
    rows = px if top else px[::-1]
    if kind == 1:
        raw = rows.astype(np.uint8).reshape(-1, 1)
    elif kind == 3:
        raw = rows[..., [0, 3]] if bits == 16 else rows[..., :1]
    elif bits == 16:
        x = ((rows[..., 0].astype(np.uint16) >> 3) << 10 | (rows[..., 1].astype(np.uint16) >> 3) << 5
             | rows[..., 2].astype(np.uint16) >> 3)
        raw = x.astype("<u2").view(np.uint8).reshape(h, w, 2)
    else:
        raw = rows[..., [2, 1, 0, 3]][..., :bits // 8]
    raw = np.ascontiguousarray(raw).reshape(h * w, -1)
    body = b""
    if rle:   # alternate run and raw packets so both kinds are exercised
        i = 0
        while i < len(raw):
            j = i + 1
            while j < len(raw) and j - i < 128 and (raw[j] == raw[i]).all():
                j += 1
            if j - i > 1:
                body += bytes([0x80 | (j - i - 1)]) + raw[i].tobytes()
            else:
                j = min(i + 3, len(raw))
                body += bytes([j - i - 1]) + raw[i:j].tobytes()
            i = j
    else:
        body = raw.tobytes()
    cmap = (1, 0, len(palette), 24) if palette is not None else (0, 0, 0, 0)
    head = struct.pack("<BBBHHBHHHHBB", 0, cmap[0], kind | (8 if rle else 0), cmap[1], cmap[2],
                       cmap[3], 0, 0, w, h, bits, (0x20 if top else 0) | (8 if bits == 32 else 0))
    pal = b"" if palette is None else np.asarray(palette, np.uint8)[:, ::-1].tobytes()
    foot = struct.pack("<II", 0, 0) + b"TRUEVISION-XFILE.\0" if footer else b""
    return head + pal + body + foot


def rand_rgba(h=5, w=7, seed=0):
    px = np.random.default_rng(seed).integers(0, 256, (h, w, 4), dtype=np.uint8)
    px[1, :4] = px[1, 0]                                       # a run for RLE
    return px


def test_tga_vs_qt():
    """Qt reads uncompressed TGA 2.0: an independent check of ours (bottom row first)."""
    px = rand_rgba()
    for bits in (24, 32):
        for top in (False, True):
            data = tga_file(px, bits=bits, top=top, footer=True)
            img = QImage.fromData(QByteArray(data), "TGA").convertToFormat(QImage.Format_RGBA8888)
            assert not img.isNull()
            qt = np.array([[img.pixelColor(x, y).getRgb() for x in range(7)] for y in range(5)])
            ours = teximage.read_tga(data)
            assert ours[::-1].tolist() == qt.tolist(), (bits, top)


def test_tga_variants():
    px = rand_rgba(seed=3)
    want = px[::-1]
    for top in (False, True):
        for rle in (False, True):
            got = teximage.read_tga(tga_file(px, top=top, rle=rle))
            assert got.tolist() == want.tolist(), (top, rle)
            got = teximage.read_tga(tga_file(px, bits=24, top=top, rle=rle))
            assert got[..., :3].tolist() == want[..., :3].tolist() and (got[..., 3] == 255).all()
    gray = teximage.read_tga(tga_file(px, kind=3, bits=8, rle=True))
    assert (gray[..., 0] == want[..., 0]).all() and (gray[..., 2] == want[..., 0]).all()
    ga = teximage.read_tga(tga_file(px, kind=3, bits=16))
    assert (ga[..., 3] == want[..., 3]).all()
    c16 = teximage.read_tga(tga_file(px, bits=16))
    assert np.abs(c16[..., :3].astype(int) - want[..., :3]).max() <= 8
    pal = np.array([(255, 0, 0), (0, 255, 0), (0, 0, 255)])
    idx = np.random.default_rng(1).integers(0, 3, (5, 7))
    got = teximage.read_tga(tga_file(idx, kind=1, bits=8, palette=pal, rle=True))
    assert got[..., :3].tolist() == pal[idx[::-1]].tolist()


def test_tga_broken():
    px = rand_rgba()
    good = tga_file(px, rle=True)
    for bad in (good[:10], good[:-5], tga_file(px)[:-5]):
        im = teximage.decode(Texture("a.tga", bad))
        assert im.pixels is None and "TGA" in im.reason
    assert teximage.decode(Texture("a.tga", good)).pixels is not None


def test_decode_alpha_kept_or_ignored():
    half = teximage.decode(Texture("a.png", image_bytes(
        4, 1, (9, 9, 9), alpha=lambda x, y: 0 if x < 2 else 255)))
    assert half.pixels[0, :, 3].tolist() == [0, 0, 255, 255]
    empty = teximage.decode(Texture("b.png", image_bytes(4, 1, (9, 9, 9), alpha=lambda x, y: 0)))
    assert (empty.pixels[..., 3] == 255).all()   # nothing would show: alpha is not coverage


def test_decode_scales_down():
    im = teximage.decode(Texture("a.png", image_bytes(8, 4, (1, 2, 3))), max_size=4)
    assert im.pixels.shape == (2, 4, 4)


def test_decode_unreadable():
    dds = teximage.decode(Texture("a.dds", b"DDS " + bytes(124)))
    assert dds.pixels is None and "DDS" in dds.reason
    junk = teximage.decode(Texture("a.bin", b"not an image"))
    assert junk.pixels is None and junk.reason


# --- report ----------------------------------------------------------------------------------

def test_texture_lines():
    d = display.build(tex_scene())
    images = teximage.decode_all(d.textures)
    assert report.texture_lines(d, images) == ["2 текстуры — видны в режиме «Материал»."]
    images[1] = teximage.decode(Texture("cloth.dds", b"DDS " + bytes(124)))
    lines = report.texture_lines(d, images)
    assert len(lines) == 2 and "cloth.dds (формат DDS" in lines[1]
    assert "переносятся" in lines[1]
    summary = report.file_summary(tex_scene(), "x.fbx", True, d, images)
    assert summary[-2:] == lines
