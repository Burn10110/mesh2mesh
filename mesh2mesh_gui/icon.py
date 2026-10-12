"""The program icon, drawn in code (no data files to package): a dark tile covered by a wavy
quad mesh (vertex dots, edges shading green -> grey -> pink along the diagonal: one format ->
another), with a flat pink ring of two arrows (conversion) in the middle.
The same drawing gives the window icon and the exe's .ico (packaging/build.py)."""

import math
import struct

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QPointF, QRectF, Qt
from PySide6.QtGui import (QColor, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap,
                           QPolygonF)

SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)

BACKGROUND = QColor("#24142f")
FACES = [QColor(c) for c in ("#2b1838", "#3b2549", "#46324f",       # dark violet
                             "#5c7259", "#627d5c", "#56715a",       # grey-green
                             "#55555f")]                            # grey
FACE_WEIGHTS = (18, 16, 10, 8, 7, 6, 8)
EDGE_STOPS = [QColor(c) for c in ("#6fd05a", "#a9b89c", "#b9a3b8", "#d071c9")]  # top-left -> bottom-right
ARROWS = QColor("#d75c9f")

# Per size class (256 x 256 design box): mesh lattice step vectors, waviness, edge width and
# vertex dot radius in screen px, ring centre radius, band width and arrowhead length/half-width.
# Small icons keep only what reads: a coarse mesh, thin lines, no dots, a thicker ring.
LEVELS = (
    (64, dict(a=(40, -17), b=(37, 21), wave=7, edge=1.0, dot=1.6, dim=0.0, r=74, width=20, head=(34, 30))),
    (32, dict(a=(56, -24), b=(52, 30), wave=8, edge=1.0, dot=1.1, dim=0.25, r=76, width=30, head=(42, 36))),
    (0, dict(a=(86, -37), b=(80, 46), wave=0, edge=1.0, dot=0, dim=0.5, r=76, width=44, head=(52, 46))),
)


def _level(size):
    return next(k for limit, k in LEVELS if size >= limit)


def _lerp(stops, t):
    t = min(max(t, 0.0), 1.0) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2)
    a, b, f = stops[i], stops[i + 1], t - i
    return QColor.fromRgbF(*(x + (y - x) * f for x, y in zip(a.getRgbF()[:3], b.getRgbF()[:3])))


def _edge_color(pt):
    return _lerp(EDGE_STOPS, (pt.x() + pt.y()) / 512)


def _face_color(i, j):
    """Fixed pseudo-random pick (no RNG state: the icon is identical on every run)."""
    h = (i * 73856093 ^ j * 19349663) & 0xFFFF
    pick = h % sum(FACE_WEIGHTS)
    for c, w in zip(FACES, FACE_WEIGHTS):
        if pick < w:
            return c
        pick -= w
    return FACES[0]


def _mesh(k):
    """Lattice points covering the tile, shifted by smooth waves -> {(i, j): QPointF}."""
    ax, ay = k["a"]
    bx, by = k["b"]
    w = k["wave"]
    pts = {}
    n = 12
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x = 128 + i * ax + j * bx + w * math.sin(0.9 * i + 1.7 * j + 0.4)
            y = 128 + i * ay + j * by + w * math.cos(1.3 * i - 0.8 * j + 1.1)
            if -90 < x < 346 and -90 < y < 346:
                pts[i, j] = QPointF(x, y)
    return pts


def _arrow_arc(p, cx, cy, r, start, end, width, head):
    """Clockwise arc from `start` to `end` degrees (math angles, y up) ending in an arrowhead
    (length, half-width) whose tip is at `end`, as one filled outline."""
    def pt(t, rr):
        t = math.radians(t)
        return QPointF(cx + rr * math.cos(t), cy - rr * math.sin(t))

    length, half = head
    body_end = end + math.degrees(length / r)
    path = QPainterPath(pt(start, r + width / 2))
    steps = 32
    for i in range(steps + 1):
        path.lineTo(pt(start + (body_end - start) * i / steps, r + width / 2))
    path.lineTo(pt(body_end, r + half))
    path.lineTo(pt(end, r))                       # the tip
    path.lineTo(pt(body_end, r - half))
    for i in range(steps + 1):
        path.lineTo(pt(body_end + (start - body_end) * i / steps, r - width / 2))
    path.closeSubpath()
    p.drawPath(path)


def render(size):
    """QImage size x size: rounded dark tile, mesh, arrow ring."""
    k = _level(size)
    img = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    p.scale(size / 256, size / 256)
    px = 256 / size                                   # one screen pixel in design units

    tile = QPainterPath()
    corner = 40 if size >= 32 else 48
    tile.addRoundedRect(QRectF(0, 0, 256, 256), corner, corner)
    p.fillPath(tile, BACKGROUND)
    p.setClipPath(tile)

    pts = _mesh(k)
    p.setPen(Qt.NoPen)
    for (i, j), a in pts.items():
        quad = [pts.get(q) for q in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))]
        if all(quad):
            p.setBrush(_lerp([_face_color(i, j), BACKGROUND], k["dim"]))
            p.drawPolygon(QPolygonF(quad))
    edge_w = max(k["edge"] * px, 5.0)
    for (i, j), a in pts.items():
        for b in (pts.get((i + 1, j)), pts.get((i, j + 1))):
            if b is not None:
                p.setPen(QPen(_edge_color((a + b) / 2), edge_w, Qt.SolidLine, Qt.RoundCap))
                p.drawLine(a, b)
    if k["dot"]:
        p.setPen(Qt.NoPen)
        rad = max(k["dot"] * px, 6.5)
        for a in pts.values():
            p.setBrush(_edge_color(a))
            p.drawEllipse(a, rad, rad)

    p.setPen(Qt.NoPen)
    p.setBrush(ARROWS)
    r, width = k["r"], k["width"]
    _arrow_arc(p, 128, 128, r, 202, 24, width, k["head"])      # top: from lower left to the right
    _arrow_arc(p, 128, 128, r, 8, -150, width, k["head"])      # bottom: from the right back left
    p.end()
    return img


def app_icon():
    icon = QIcon()
    for s in SIZES:
        icon.addPixmap(QPixmap.fromImage(render(s)))
    return icon


def write_ico(path, sizes=SIZES):
    """Windows .ico with PNG-compressed entries (Vista+), one per size."""
    blobs = []
    for s in sizes:
        ba = QByteArray()
        buf = QBuffer(ba)
        buf.open(QIODevice.WriteOnly)
        render(s).save(buf, "PNG")
        blobs.append((s, bytes(ba)))
    out = struct.pack("<HHH", 0, 1, len(blobs))
    offset = 6 + 16 * len(blobs)
    for s, data in blobs:
        out += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    with open(path, "wb") as f:
        f.write(out + b"".join(d for _, d in blobs))
