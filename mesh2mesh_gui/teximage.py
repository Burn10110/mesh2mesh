"""Texture images for the 3D view: file bytes -> RGBA pixels ready for glTexImage2D. QImage and
numpy only (no GL, safe off the UI thread), so decoding runs in the loading job."""

import struct
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QImage

from mesh2mesh import textures

MAX_SIZE = 4096   # larger maps are scaled down: GPU limits and memory; the view needs no more


@dataclass
class Image:
    name: str
    pixels: np.ndarray | None   # (h, w, 4) uint8 RGBA, bottom row first (UV v up); None = failed
    reason: str = ""            # why it is not shown (pixels None)


def decode(tex, max_size=MAX_SIZE):
    """model.Texture -> Image. TGA by our own reader (Qt reads only TGA 2.0 without RLE); the
    rest by Qt: PNG, JPEG, BMP, TIFF, WebP, GIF; not DDS or PSD."""
    kind = textures.kind(tex.data, tex.name)
    if kind == "tga":
        try:
            px = read_tga(tex.data)
        except ValueError as e:
            return Image(tex.name, None, str(e))
        if max(px.shape[:2]) > max_size:
            k = -(-max(px.shape[:2]) // max_size)
            px = np.ascontiguousarray(px[::k, ::k])
    else:
        img = QImage.fromData(QByteArray(tex.data))
        if img.isNull():
            return Image(tex.name, None,
                         f"формат {kind.upper()} окно не показывает" if kind
                         else "картинка не читается")
        if max(img.width(), img.height()) > max_size:
            img = img.scaled(max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        img = img.convertToFormat(QImage.Format_RGBA8888)
        w, h, bpl = img.width(), img.height(), img.bytesPerLine()
        rows = np.frombuffer(img.constBits(), np.uint8, count=h * bpl).reshape(h, bpl)
        # always a copy: a view would point into the QImage buffer (read-only, freed with img)
        px = np.array(rows[::-1, :w * 4].reshape(h, w, 4), order="C", copy=True)
    if not (px[..., 3] >= 128).any():
        px[..., 3] = 255   # alpha holds something else (or is empty): the view clips at 0.5
    return Image(tex.name, px)


def decode_all(texs):
    return [decode(t) for t in texs]


# --- TGA -------------------------------------------------------------------------------------

def read_tga(data):
    """TGA bytes -> (h, w, 4) uint8 RGBA, bottom row first. Types 1/2/3 and their RLE variants
    9/10/11; 8/15/16/24/32 bits; either origin. ValueError(reason in Russian) otherwise."""
    if len(data) < 18:
        raise ValueError("TGA повреждён")
    (id_len, cmap_type, kind, cmap_first, cmap_len, cmap_bits, _, _, w, h, bits,
     desc) = struct.unpack("<BBBHHBHHHHBB", data[:18])
    base = kind & ~8
    if base not in (1, 2, 3) or w == 0 or h == 0:
        raise ValueError(f"вид TGA {kind} окно не показывает")
    pos = 18 + id_len
    palette = None
    if cmap_type == 1:
        size = cmap_len * ((cmap_bits + 7) // 8)
        palette = _tga_pixels(np.frombuffer(data, np.uint8, size, pos), cmap_bits, False)
        pos += size
    if base == 1 and palette is None:
        raise ValueError("TGA с палитрой без палитры")
    bpp = (bits + 7) // 8
    n = w * h
    if kind & 8:
        raw = _tga_rle(data, pos, n, bpp)
    else:
        raw = np.frombuffer(data, np.uint8, min(n * bpp, max(len(data) - pos, 0)), pos)
        if len(raw) < n * bpp:
            raise ValueError("TGA повреждён (обрезан)")
    if base == 1:
        idx = raw.view("<u2") if bpp == 2 else raw
        idx = np.clip(idx.astype(np.int64) - cmap_first, 0, len(palette) - 1)
        px = palette[idx]
    else:
        px = _tga_pixels(raw, bits, base == 3)
    px = px.reshape(h, w, 4)
    if desc & 0x20:            # top-left origin: rows top first -> ours bottom first
        px = px[::-1]
    if desc & 0x10:            # right-to-left
        px = px[:, ::-1]
    return np.ascontiguousarray(px)


def _tga_rle(data, pos, n, bpp):
    out = bytearray()
    need = n * bpp
    end = len(data)
    while len(out) < need:
        if pos >= end:
            raise ValueError("TGA повреждён (обрезан)")
        head = data[pos]
        count = (head & 0x7F) + 1
        pos += 1
        if head & 0x80:
            out += data[pos:pos + bpp] * count
            pos += bpp
        else:
            out += data[pos:pos + count * bpp]
            pos += count * bpp
    return np.frombuffer(bytes(out[:need]), np.uint8)


def _tga_pixels(raw, bits, gray):
    """Packed TGA pixels -> (n, 4) uint8 RGBA."""
    if gray:
        if bits == 8:
            v = raw
            return np.stack([v, v, v, np.full_like(v, 255)], 1)
        if bits == 16:
            p = raw.reshape(-1, 2)
            return np.stack([p[:, 0], p[:, 0], p[:, 0], p[:, 1]], 1)
    elif bits in (15, 16):
        x = raw.view("<u2").astype(np.uint32)
        c = [((x >> s) & 31) * 255 // 31 for s in (10, 5, 0)]
        return np.stack(c + [np.full_like(x, 255)], 1).astype(np.uint8)
    elif bits == 24:
        p = raw.reshape(-1, 3)
        return np.stack([p[:, 2], p[:, 1], p[:, 0], np.full(len(p), 255, np.uint8)], 1)
    elif bits == 32:
        p = raw.reshape(-1, 4)
        return p[:, [2, 1, 0, 3]]
    raise ValueError(f"TGA {bits} бит окно не показывает")
