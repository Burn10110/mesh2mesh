"""3MF reader and writer (core spec + materials extension: base materials, colours, textures).

Writer: components for hierarchy/instancing (baked flat if any transform mirrors), basematerials,
a colorgroup per mesh for corner colours, a texture2dgroup per mesh and texture (PNG/JPEG in
/3D/Texture/, D21), exact decimal coordinates; no metadata/thumbnail.

- Root model found through the OPC start-part relationship (fallback 3D/3dmodel.model).
- Objects with <components> become nodes with children; a mesh object referenced several times
  becomes one Mesh shared by several nodes (instancing kept in the Scene).
- Transforms are row-vector 3x4 (p' = p @ M) -> transposed into the column-vector convention.
- Production extension: components with p:path pointing into another .model part are followed.
- Property per triangle: pid + p1 -> a Scene material (base material by name, colour-group colour
  as 'color_RRGGBBAA'); different p2/p3 -> per-corner colours as well.
- texture2dgroup: p1/p2/p3 -> UV per corner; one white material per texture2d with the image.
- Speed: plain <vertices>/<triangles> bodies are parsed on bytes before ElementTree sees the
  part (it gets a stub element instead); anything unusual leaves the body to ElementTree.
"""

import os
import posixpath
import re
import secrets
import time
import zipfile
import zlib
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .. import geom, textures
from ..model import NO_MATERIAL, Material, Mesh, Node, Scene, Texture

CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
MAT = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"
PROD = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
RELS = "http://schemas.openxmlformats.org/package/2006/relationships"
START_PART = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"
NS = {"c": CORE, "m": MAT}
UNIT_M = {"micron": 1e-6, "millimeter": 1e-3, "centimeter": 1e-2, "meter": 1.0,
          "inch": 0.0254, "foot": 0.3048}
TEXTURE_REL = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dtexture"
# property groups we do not map: tag -> what the user loses
UNSUPPORTED_GROUPS = {f"{{{MAT}}}compositematerials": "смешанные материалы",
                      f"{{{MAT}}}multiproperties": "составные свойства"}


def _matrix(s):
    """3MF transform attribute -> 4x4 column-vector matrix."""
    m = np.eye(4)
    if s:
        v = [float(x) for x in s.split()]
        if len(v) != 12:
            raise ValueError(f"3MF: transform должен содержать 12 чисел: {s!r}")
        m[:3, :3] = np.array(v[:9]).reshape(3, 3).T
        m[:3, 3] = v[9:12]
    return m


def _rgba(s):
    """'#RRGGBB[AA]' -> (r, g, b, a) floats 0..1."""
    h = s.strip().lstrip("#")
    if len(h) == 6:
        h += "FF"
    if len(h) != 8:
        raise ValueError(f"3MF: неверный цвет {s!r}")
    return tuple(int(h[i:i + 2], 16) / 255 for i in range(0, 8, 2))


def _hex(rgba):
    return "".join(f"{round(c * 255):02X}" for c in rgba)


def _color_keys(colors):
    """(n, k) colours 0..1 -> uint64 keys whose fixed-width hex is _hex() of each row: round()
    and np.round both go half to even, and numeric order of the keys is the order of the hex
    strings."""
    c = np.clip(colors, 0, 1) * 255
    if np.isnan(c).any():
        raise ValueError("3MF: цвет вершины — не число (NaN)")
    key = np.zeros(len(c), dtype=np.uint64)
    for j in range(c.shape[1]):
        key = (key << np.uint64(8)) | np.round(c[:, j]).astype(np.uint64)
    return key


CHUNK = 1 << 16     # rows per block-formatted piece of text


def _block(tmpl, rows):
    """Format every row of a 2-D array with tmpl (%r = repr of a float, %d = str of an int):
    the text a per-row f-string would give, but the loop runs in C."""
    n, out = len(rows), []
    for s in range(0, n, CHUNK):
        part = rows[s:s + CHUNK]
        out.append((tmpl * len(part)) % tuple(part.ravel().tolist()))
    return "".join(out)


def _block_kinds(tmpls, kind, rows):
    """Like _block, with tmpls[kind[i]] for row i; row i uses the first fields its template
    needs. Runs of one kind are formatted as blocks; when kinds alternate a lot, each kind is
    formatted once and its lines are interleaved back into row order."""
    starts = np.flatnonzero(np.diff(kind)) + 1
    if len(starts) < max(64, len(kind) // 256):
        edges = [0, *starts.tolist(), len(kind)]
        return "".join(_block(tmpls[kind[a]], rows[a:b, :tmpls[kind[a]].count("%")])
                       for a, b in zip(edges[:-1], edges[1:]))
    lines = np.empty(len(kind), dtype=object)
    for k, tmpl in enumerate(tmpls):
        sel = np.flatnonzero(kind == k)
        if len(sel):
            lines[sel] = _block(tmpl, rows[sel, :tmpl.count("%")]).splitlines(keepends=True)
    return "".join(lines.tolist())


@dataclass
class TexGroup:
    """<m:texture2dgroup>: the texture2d id and its tex2coord list (n, 2)."""
    texid: str
    uv: np.ndarray


def _root_part(z):
    names = set(z.namelist())
    if "_rels/.rels" in names:
        for r in ET.fromstring(z.read("_rels/.rels")).iter(f"{{{RELS}}}Relationship"):
            if r.get("Type") == START_PART:
                return r.get("Target").lstrip("/")
    if "3D/3dmodel.model" in names:
        return "3D/3dmodel.model"
    raise ValueError("3MF: не найдена основная модель (3D/3dmodel.model)")


# --- fast path: <vertices>/<triangles> bodies on bytes (knowledge/perf.md) -------------------
# A body is taken only when it is `<vertex|triangle attr="value" …/>` elements and whitespace,
# double quotes, ASCII; the numbers then mean what ElementTree + float()/int() would give.
# Anything else (and any doubt) -> the body stays in the document for ElementTree.

FAST = True                     # tools/check_3mf_fast.py switches it off for the reference
_FAST_ATTR = "_m2m_fast"
_WS = rb"[ \t\r\n]"
_NAME = rb"([A-Za-z_][A-Za-z0-9_.\-]*(?::[A-Za-z_][A-Za-z0-9_.\-]*)?)"
_EQ = _WS + rb"*=" + _WS + rb"*"
_BLANK = re.compile(_WS + rb"*")
_CLOSE = re.compile(_WS + rb"*/>" + _WS + rb"*")
_NEXT_ATTR = re.compile(_WS + rb"+" + _NAME + _EQ)
_FIRST = {t: re.compile(_WS + rb"*<" + t + _WS + rb"+" + _NAME + _EQ) for t in (b"vertex", b"triangle")}
_NEXT_ELEM = {t: re.compile(_WS + rb"*/>" + _WS + rb"*<" + t + _WS + rb"+" + _NAME + _EQ)
              for t in (b"vertex", b"triangle")}
_KNOWN = {b"vertex": {b"x", b"y", b"z"}, b"triangle": {b"v1", b"v2", b"v3", b"pid", b"p1", b"p2", b"p3"}}
_MAX_PIECE = 64                 # longest markup run between two values we classify
_LOW = np.array([(1 << 8 * k) - 1 for k in range(9)], np.uint64)    # [k]: low k bytes
_MIX = np.uint64(0x9E3779B97F4A7C15)
_ZEROS8 = np.uint64(0x3030303030303030)
_HI4 = np.uint64(0xF0F0F0F0F0F0F0F0)
_LO4 = np.uint64(0x0F0F0F0F0F0F0F0F)
_SIX = np.uint64(0x0606060606060606)
_POW10 = 10 ** np.arange(18, dtype=np.int64)


@dataclass
class _Tris:
    """<triangles> as arrays: v (t, 3); pid (t,) index into pid_names, -1 = no attribute;
    p (t, 3) p1/p2/p3 where has (t, 3)."""
    v: np.ndarray
    pid: np.ndarray
    pid_names: list
    p: np.ndarray
    has: np.ndarray


def _words(data):
    """Little-endian uint64 at every byte offset of data (a view, no copy)."""
    return np.ndarray((max(len(data) - 7, 0),), "<u8", data, 0, (1,))


def _classes(key, same=None, limit=64):
    """Equal keys -> (class per item, first item of each class), or None past `limit` classes
    or when same(members, first) rejects a class (a hash collision)."""
    codes = np.full(len(key), -1, np.int16)
    reps, r = [], 0
    while r < len(key) and codes[r] < 0:
        if len(reps) == limit:
            return None
        m = key == key[r]
        if same is not None and not same(np.flatnonzero(m), r):
            return None
        codes[m] = len(reps)
        reps.append(r)
        r = int(np.argmax(codes < 0))
    return codes, reps


def _piece_classes(W, ps, sl):
    """Markup runs (start, length <= _MAX_PIECE, no NUL) -> (class per run, first run of each
    class) by their bytes, or None. Up to 8 bytes the masked word is the run itself; longer
    runs: a hash of their words, each class checked byte for byte."""
    w0 = W[ps] & _LOW[np.minimum(sl, 8)]
    long = np.flatnonzero(sl > 8)
    if not len(long):
        return _classes(w0)
    short = np.flatnonzero(sl <= 8)
    cs = _classes(w0[short]) if len(short) else (np.zeros(0, np.int16), [])
    lp, ll = ps[long], sl[long]
    K = int(ll.max() + 7) // 8
    words = [w0[long]] + [_piece_word(W, lp, ll, j) for j in range(1, K)]
    key = ll.astype(np.uint64)
    for w in words:
        key = key * _MIX + w

    def same(members, r):
        return bool(np.all(ll[members] == ll[r])) and all(np.all(w[members] == w[r]) for w in words)

    cl = _classes(key, same)
    if cs is None or cl is None or len(cs[1]) + len(cl[1]) > 64:
        return None
    codes = np.empty(len(sl), np.int16)
    codes[short] = cs[0]
    codes[long] = cl[0] + len(cs[1])
    return codes, [int(short[r]) for r in cs[1]] + [int(long[r]) for r in cl[1]]


def _uints(A, W, end, length):
    """Decimal digit runs of 1..18 bytes ending at `end` -> int64, or None if any is something
    else. The 8 bytes up to the end as one word, bytes before the number set to '0', checked
    to be digits, summed pairwise (SWAR, as obj._decimals); longer runs: digit matrix."""
    if not len(end):
        return np.zeros(0, np.int64)
    if length.min() < 1 or length.max() > 18 or end.min() < 8:
        return None
    x = W[end - 8]
    lead = _LOW[8 - np.minimum(length, 8)]
    x = (x & ~lead) | (_ZEROS8 & lead)
    if np.any(((x & _HI4) != _ZEROS8) | ((((x & _LO4) + _SIX) & _HI4) != 0)):
        return None
    x -= _ZEROS8                                # byte k = digit k (first digit in the low byte)
    x = (x * np.uint64(10) + (x >> np.uint64(8))) & np.uint64(0x00FF00FF00FF00FF)
    x = (x * np.uint64(100) + (x >> np.uint64(16))) & np.uint64(0x0000FFFF0000FFFF)
    x = (x * np.uint64(10000) + (x >> np.uint64(32))) & np.uint64(0xFFFFFFFF)
    val = x.view(np.int64)
    for L in np.unique(length[length > 8]).tolist():
        m = np.flatnonzero(length == L)
        dig = A[end[m, None] - L + np.arange(L)].astype(np.int64) - 0x30
        if np.any((dig < 0) | (dig > 9)):
            return None
        val[m] = dig @ _POW10[L - 1::-1]
    return val


def _piece_word(W, start, length, j):
    return W[start + 8 * j] & _LOW[np.clip(length - 8 * j, 0, 8)]


def _attrs(data, A, W, b0, b1, tag):
    """Body data[b0:b1] of <vertices>/<triangles> -> (element count, {name: (element of each
    value, value number)}, value starts, value ends), or None unless it is exactly
    `<tag name="value" …/>` elements: well-formed for XML, no duplicate attribute, no NUL.
    Values of known attributes are the caller's to check ('&', '<' fail number parsing);
    those of other attributes are checked here."""
    if data.find(b"\0", b0, b1) >= 0:           # word keys and numpy S strings need none
        return None
    q = np.flatnonzero(A[b0:b1] == 0x22) + b0
    if len(q) % 2:
        return None
    if not len(q):
        return (0, {}, q, q) if _BLANK.fullmatch(data, b0, b1) else None
    vs, ve = q[0::2] + 1, q[1::2]
    first = _FIRST[tag].fullmatch(data, b0, q[0])
    if first is None or _CLOSE.fullmatch(data, ve[-1] + 1, b1) is None:
        return None
    # markup between two values, e.g. ` v2=` or `/>\n<triangle v1=`: classified by its bytes
    ps, pe = ve[:-1] + 1, q[2::2]
    sl = pe - ps
    names, new = [first.group(1)], [True]
    if len(sl):
        if sl.max() > _MAX_PIECE or ps.max() + _MAX_PIECE > len(data) - 8:
            return None
        cl = _piece_classes(W, ps, sl)
        if cl is None:
            return None
        codes, reps = cl
        rep_name, rep_new = [], []
        for r in reps:
            s, e = int(ps[r]), int(pe[r])
            m = _NEXT_ATTR.fullmatch(data, s, e)
            rep_new.append(m is None)
            if m is None and (m := _NEXT_ELEM[tag].fullmatch(data, s, e)) is None:
                return None
            rep_name.append(m.group(1))
    else:
        codes, rep_name, rep_new = np.zeros(0, np.int16), [], []
    uniq = list(dict.fromkeys(names + rep_name))
    name_of = np.r_[uniq.index(names[0]), np.array([uniq.index(n) for n in rep_name], np.int64)[codes]]
    is_new = np.r_[True, np.array(rep_new, bool)[codes]]
    elem = np.cumsum(is_new) - 1
    n_el = int(elem[-1]) + 1
    out = {}
    for k, name in enumerate(uniq):
        if name not in _KNOWN[tag]:
            prefix = name.partition(b":")[0] if b":" in name else None
            if name == b"xmlns" or prefix == b"xmlns" or (
                    prefix not in (None, b"xml") and b"xmlns:" + prefix + b"=" not in data):
                return None
            # values we skip still have to be what XML allows: printable ASCII without '&',
            # one '<' per element (none inside a value)
            a = A[b0:b1] - np.uint8(0x20)
            if (np.count_nonzero(a >= 0x5F) != sum(data.count(c, b0, b1) for c in (b"\t", b"\n", b"\r"))
                    or data.find(b"&", b0, b1) >= 0 or data.count(b"<", b0, b1) != n_el):
                return None
        idx = np.flatnonzero(name_of == k)
        e = elem[idx]
        if np.any(e[1:] == e[:-1]):
            return None                         # duplicate attribute: XML error
        out[name] = (e, idx)
    return n_el, out, vs, ve


def _fast_vertices(data, A, W, b0, b1):
    t = _attrs(data, A, W, b0, b1, b"vertex")
    if t is None:
        return None
    n, attrs, vs, ve = t
    if any(a not in attrs or len(attrs[a][0]) != n for a in (b"x", b"y", b"z")):
        return None if n else np.zeros((0, 3))
    vals = data[b0:b1].split(b'"')[1::2]
    if len(vals) != len(vs):
        return None
    if len(attrs) > 3:                          # other attributes: their values are not numbers
        idx = np.concatenate([attrs[a][1] for a in (b"x", b"y", b"z")])
        vals = [vals[i] for i in idx.tolist()]
        cols = np.arange(3 * n).reshape(3, n)
    else:
        cols = [attrs[a][1] for a in (b"x", b"y", b"z")]
    try:
        f = np.array(vals, dtype=np.float64)    # float() of each value, as the ET path
    except (ValueError, TypeError):
        return None
    pos = np.empty((n, 3))
    for k in range(3):
        pos[:, k] = f[cols[k]]
    return pos


def _fast_triangles(data, A, W, b0, b1):
    t = _attrs(data, A, W, b0, b1, b"triangle")
    if t is None:
        return None
    n, attrs, vs, ve = t
    v = np.zeros((n, 3), np.int64)
    p = np.zeros((n, 3), np.int64)
    has = np.zeros((n, 3), bool)
    for k, a in enumerate((b"v1", b"v2", b"v3")):
        if a not in attrs or len(attrs[a][0]) != n:
            return None
        i = attrs[a][1]
        x = _uints(A, W, ve[i], ve[i] - vs[i])
        if x is None:
            return None
        v[:, k] = x
    for k, a in enumerate((b"p1", b"p2", b"p3")):
        if a in attrs:
            e, i = attrs[a]
            x = _uints(A, W, ve[i], ve[i] - vs[i])
            if x is None:
                return None
            p[e, k] = x
            has[e, k] = True
    pid, names = np.full(n, -1, np.int64), []
    if b"pid" in attrs:
        e, i = attrs[b"pid"]
        L = ve[i] - vs[i]
        if L.max() > 8:
            return None
        # the value's bytes as the high bytes of the word that ends with it: exact key
        cl = _classes(W[ve[i] - 8] & ~_LOW[8 - L])
        if cl is None:
            return None
        codes, reps = cl
        for r in reps:
            s = data[int(vs[i[r]]):int(ve[i[r]])]
            if any(c in s for c in (b"\t", b"\n", b"\r", b"&", b"<")) or not s.isascii():
                return None                     # XML would normalise / decode / reject it
            names.append(s.decode("ascii"))
        pid[e] = codes
    return _Tris(v, pid, names, p, has)


def _stub_bodies(data, fast):
    """data with each plain <vertices>/<triangles> body parsed into `fast` and replaced by
    `<vertices _m2m_fast="key"/>`; None if nothing was replaced or the document has markup
    a byte search could mistake (comments, CDATA, DOCTYPE, processing instructions)."""
    if b"<!" in data or data.count(b"<?") > 1 or data.find(b"<?") not in (-1, 0, 3) or len(data) < 64:
        return None
    A, W = np.frombuffer(data, np.uint8), _words(data)
    token = secrets.token_hex(8)
    hits = []
    for tag in (b"vertices", b"triangles"):
        i = data.find(b"<%s>" % tag)
        while i >= 0:
            hits.append((i, tag))
            i = data.find(b"<%s>" % tag, i + 1)
    out, copied, scanned = [], 0, 0
    for start, tag in sorted(hits):
        if start < scanned:
            continue                            # inside a body left to ElementTree
        b0 = start + len(tag) + 2
        b1 = data.find(b"</%s>" % tag, b0)
        if b1 < 0:
            break
        scanned = b1 + len(tag) + 3
        parse = _fast_vertices if tag == b"vertices" else _fast_triangles
        parsed = parse(data, A, W, b0, b1)
        if parsed is not None:
            key = f"{token}{len(fast)}"
            fast[key] = parsed
            out += [data[copied:start], b'<%s %s="%s"/>' % (tag, _FAST_ATTR.encode(), key.encode())]
            copied = scanned
    if not fast:
        return None
    out.append(data[copied:])
    return b"".join(out)


def _parse_part(data):
    """Part bytes -> (root element, {stub key: parsed body})."""
    fast = {}
    stub = _stub_bodies(data, fast) if FAST else None
    if stub is not None:
        try:
            return ET.fromstring(stub), fast
        except ET.ParseError:
            pass                                # ElementTree reports it on the original
    return ET.fromstring(data), {}


def _tris_et(els):
    """<triangle> elements -> _Tris (the reference for _fast_triangles)."""
    v = np.array([[int(t.get("v1")), int(t.get("v2")), int(t.get("v3"))] for t in els],
                 dtype=np.int64).reshape(-1, 3)
    pids = [t.get("pid") for t in els]
    names = list(dict.fromkeys(x for x in pids if x is not None))
    code = {s: i for i, s in enumerate(names)}
    pid = np.array([code[x] if x is not None else -1 for x in pids], dtype=np.int64)
    p = np.zeros((len(els), 3), np.int64)
    has = np.zeros((len(els), 3), bool)
    for k, a in enumerate(("p1", "p2", "p3")):
        vals = [t.get(a) for t in els]
        has[:, k] = [x is not None for x in vals]
        try:
            p[has[:, k], k] = [int(x) for x in vals if x is not None]
        except (ValueError, OverflowError):
            raise ValueError(f"3MF: неверный индекс свойства в {a} у треугольника") from None
    return _Tris(v, pid, names, p, has)


class _Reader:
    def __init__(self, z):
        self.z = z
        self.scene = Scene()
        self.parts = {}         # part path -> (objects by id, property groups by id)
        self.mesh_of = {}       # (part, object id) -> mesh index
        self.mat_of = {}        # (part, pid, index) -> material index
        self.mat_by_key = {}    # ('base', name, rgba) / ('color', rgba) -> material index
        self.tex2d = {}         # (part, texture2d id) -> element
        self.tex_mat = {}       # (part, texture2d id) -> material index
        self.fast = {}          # stub key -> parsed <vertices> (n, 3) / <triangles> _Tris

    def part(self, path):
        if path not in self.parts:
            try:
                data = self.z.read(path)
            except KeyError:
                raise ValueError(f"3MF: нет части {path}") from None
            root, fast = _parse_part(data)
            del data
            self.fast.update(fast)
            res = root.find("c:resources", NS)
            objs, groups = {}, {}
            for el in (res if res is not None else []):
                if el.tag == f"{{{CORE}}}object":
                    objs[el.get("id")] = el
                elif el.tag == f"{{{CORE}}}basematerials":
                    groups[el.get("id")] = [("base", b.get("name") or f"material_{el.get('id')}_{i}",
                                             _rgba(b.get("displaycolor", "#FFFFFF")))
                                            for i, b in enumerate(el.findall("c:base", NS))]
                elif el.tag == f"{{{MAT}}}colorgroup":
                    groups[el.get("id")] = [("color", None, _rgba(c.get("color")))
                                            for c in el.findall("m:color", NS)]
                elif el.tag == f"{{{MAT}}}texture2d":
                    self.tex2d[(path, el.get("id"))] = el
                    if {el.get("tilestyleu", "wrap"), el.get("tilestylev", "wrap")} != {"wrap"}:
                        self.scene.warn("3MF: режим края текстуры (mirror/clamp) не переносится "
                                        "— будет повтор")
                elif el.tag == f"{{{MAT}}}texture2dgroup":
                    uv = np.array([[float(c.get("u")), float(c.get("v"))]
                                   for c in el.findall("m:tex2coord", NS)]).reshape(-1, 2)
                    groups[el.get("id")] = TexGroup(el.get("texid"), uv)
                elif el.tag in UNSUPPORTED_GROUPS:
                    groups[el.get("id")] = None
                    self.scene.warn(f"3MF: {UNSUPPORTED_GROUPS[el.tag]} не поддерживаются — отброшены")
            self.parts[path] = (root, objs, groups)
        return self.parts[path]

    def material(self, path, pid, index):
        key = (path, pid, index)
        if key not in self.mat_of:
            groups = self.part(path)[2]
            if pid not in groups:
                raise ValueError(f"3MF: ссылка на несуществующую группу свойств pid={pid}")
            g = groups[pid]
            if g is None:
                self.mat_of[key] = -1
            elif isinstance(g, TexGroup):
                self.mat_of[key] = self.texture_material(path, g.texid)
            else:
                if not 0 <= index < len(g):
                    raise ValueError(f"3MF: индекс свойства {index} вне группы pid={pid}")
                kind, name, rgba = g[index]
                if kind == "color":
                    # an unnamed colour equal to a named base material is that material
                    base = next((b for gg in groups.values() if isinstance(gg, list) for b in gg
                                 if b[0] == "base" and b[2] == rgba), None)
                    if base is not None:
                        kind, name, rgba = base
                mkey = (kind, name, rgba)
                if mkey not in self.mat_by_key:
                    self.mat_by_key[mkey] = len(self.scene.materials)
                    self.scene.materials.append(
                        Material(name if kind == "base" else f"color_{_hex(rgba)}", rgba))
                self.mat_of[key] = self.mat_by_key[mkey]
        return self.mat_of[key]

    def texture_material(self, path, texid):
        """One material per texture2d (white, the image as its base colour map)."""
        key = (path, texid)
        if key not in self.tex_mat:
            el = self.tex2d.get(key)
            if el is None:
                raise ValueError(f"3MF: ссылка на несуществующую текстуру texid={texid}")
            ref = el.get("path", "")
            name = posixpath.basename(ref) or f"texture_{texid}"
            part = ref.lstrip("/") if ref.startswith("/") else \
                posixpath.join(posixpath.dirname(path), ref)
            try:
                tex = Texture(name, self.z.read(posixpath.normpath(part)))
            except KeyError:
                self.scene.warn(f"3MF: нет файла текстуры {ref} — текстура отброшена")
                tex = None
            if tex is not None and textures.kind(tex.data, name) not in textures.WEB_KINDS:
                self.scene.warn(f"3MF: текстура {ref} не PNG/JPEG (нарушение формата) — "
                                "перенесена как есть")
            self.tex_mat[key] = len(self.scene.materials)
            self.scene.materials.append(
                Material(posixpath.splitext(name)[0], (1.0, 1.0, 1.0, 1.0), tex))
        return self.tex_mat[key]

    def node(self, path, oid, matrix, depth=0):
        if depth > 64:
            raise ValueError("3MF: слишком глубокая (циклическая?) вложенность компонентов")
        objs = self.part(path)[1]
        if oid not in objs:
            raise ValueError(f"3MF: объект {oid} не найден в {path}")
        o = objs[oid]
        node = Node(o.get("name") or f"object_{oid}", matrix)
        if o.get("type") not in (None, "model"):
            self.scene.warn(f"3MF: тип объекта «{o.get('type')}» перенесён как обычная деталь")
        mesh = o.find("c:mesh", NS)
        if mesh is not None:
            key = (path, oid)
            if key not in self.mesh_of:
                self.mesh_of[key] = len(self.scene.meshes)
                self.scene.meshes.append(self.mesh(path, o, mesh))
            node.mesh = self.mesh_of[key]
        comps = o.find("c:components", NS)
        if comps is not None:
            for c in comps.findall("c:component", NS):
                cpath = c.get(f"{{{PROD}}}path")
                cpath = posixpath.normpath(cpath.lstrip("/")) if cpath else path
                node.children.append(self.node(cpath, c.get("objectid"),
                                               _matrix(c.get("transform")), depth + 1))
        return node

    def mesh(self, path, o, mesh):
        oid = o.get("id")
        verts = mesh.find("c:vertices", NS)
        tris_el = mesh.find("c:triangles", NS)
        if verts is not None and verts.get(_FAST_ATTR) in self.fast:
            pos = self.fast.pop(verts.get(_FAST_ATTR))
        else:
            pos = np.array([[float(v.get("x")), float(v.get("y")), float(v.get("z"))]
                            for v in (verts if verts is not None else [])],
                           dtype=np.float64).reshape(-1, 3)
        if tris_el is not None and tris_el.get(_FAST_ATTR) in self.fast:
            tt = self.fast.pop(tris_el.get(_FAST_ATTR))
        else:
            tt = _tris_et(list(tris_el) if tris_el is not None else [])
        tris = tt.v
        if len(tris) and (tris.min() < 0 or tris.max() >= len(pos)):
            raise ValueError(f"3MF: объект {oid}: индекс вершины вне диапазона")
        face_mat, corner_col, corner_uv = self.properties(path, o, tt)
        return Mesh.from_triangles(o.get("name") or f"mesh_{oid}", pos, tris,
                                   face_material=face_mat if np.any(face_mat >= 0) else None,
                                   colors=corner_col, uv=corner_uv)

    def properties(self, path, o, tt):
        """Triangle properties -> (face material (t,), corner colours (3t, 4) | None, corner
        UV (3t, 2) | None). Same as a loop over triangles calling material() for p1 (and p1,
        p2, p3 of colour gradients), then checking UV indices: materials appear and errors
        rise in that order; material() runs once per distinct (group, index)."""
        t = len(tt.v)
        face_mat = np.full(t, -1, dtype=np.int64)
        opid, oidx = o.get("pid"), int(o.get("pindex", 0))
        names = list(tt.pid_names)
        code = tt.pid.copy()
        own = code >= 0
        if opid is not None:
            code[~own] = len(names)
            names.append(opid)
        tri = np.flatnonzero(code >= 0)
        if not len(tri):
            return face_mat, None, None
        groups = self.part(path)[2]
        p1 = np.where(tt.has[:, 0], tt.p[:, 0], np.where(own, 0, oidx))
        pk = np.where(tt.has, tt.p, p1[:, None])        # index per corner (corner 0 = p1)
        group = [groups.get(n) for n in names]
        is_list = np.array([isinstance(g, list) and n in groups for n, g in zip(names, group)])
        is_tex = np.array([isinstance(g, TexGroup) for g in group])
        size = np.array([len(g) if isinstance(g, list) else 0 for g in group])
        # calls of material(): key = position in the loop (4 per triangle: p1, then p1..p3
        # of a gradient) -> (group, index); index 0 where the group ignores it
        c = code[tri]
        grad = tri[~is_tex[c] & (np.any(tt.has[tri, 1:] & (pk[tri, 1:] != p1[tri, None]), 1))]
        keys = np.concatenate([4 * tri, 4 * grad + 2, 4 * grad + 3])
        cpid = np.concatenate([c, code[grad], code[grad]])
        cidx = np.concatenate([p1[tri], pk[grad, 1], pk[grad, 2]])
        n_g = len(grad)
        # UV indices of texture triangles: the first bad one in loop order
        uv_err = None
        for g in np.flatnonzero(is_tex).tolist():
            rows = tri[c == g]
            bad = (pk[rows] < 0) | (pk[rows] >= len(group[g].uv))
            if np.any(bad):
                r, k = divmod(int(np.argmax(bad.ravel())), 3)
                err = (4 * int(rows[r]) + 1 + k,
                       f"3MF: индекс UV {int(pk[rows[r], k])} вне группы pid={names[g]}")
                uv_err = min(uv_err, err) if uv_err else err
        order = np.argsort(keys, kind="stable") if n_g else np.arange(len(keys))
        keys_s, cpid_s, cidx_s = keys[order], cpid[order], cidx[order]
        # distinct calls: in-range list indices apart, the rest one bucket per group (the
        # first of a bucket decides: the same result, or the error the loop would raise)
        width = int(size.max()) + 1
        lst = is_list[cpid_s]
        ok = lst & (cidx_s >= 0) & (cidx_s < size[cpid_s])
        comb = cpid_s * width + np.where(ok, cidx_s, np.where(lst, size[cpid_s], 0))
        _, first, inv = np.unique(comb, return_index=True, return_inverse=True)
        lut = np.empty(len(first), np.int64)
        for u in np.argsort(first).tolist():
            i = int(first[u])
            if uv_err and keys_s[i] > uv_err[0]:
                raise ValueError(uv_err[1])
            lut[u] = self.material(path, names[cpid_s[i]], int(cidx_s[i]))
        if uv_err:
            raise ValueError(uv_err[1])
        m = np.empty(len(keys), np.int64)
        m[order] = lut[inv.ravel()]
        face_mat[tri] = m[:len(tri)]
        corner_col = corner_uv = None
        if n_g:
            colors = np.array([mat.color for mat in self.scene.materials], np.float64).reshape(-1, 4)
            mk = np.stack([face_mat[grad], m[len(tri):len(tri) + n_g], m[len(tri) + n_g:]], 1)
            corner_col = np.full((t, 3, 4), np.nan)
            sub = corner_col[grad]
            sub[mk >= 0] = colors[mk[mk >= 0]]
            corner_col[grad] = sub
            # triangles without a gradient: their face colour on every corner
            face_col = np.ones((t, 4))
            fm = face_mat >= 0
            face_col[fm] = colors[face_mat[fm]]
            fill = np.isnan(corner_col[:, 0, 0])
            corner_col[fill] = face_col[fill][:, None, :]
            corner_col = np.nan_to_num(corner_col, nan=1.0).reshape(-1, 4)
        if np.any(is_tex[c]):
            corner_uv = np.zeros((t, 3, 2))
            for g in np.flatnonzero(is_tex).tolist():
                rows = tri[c == g]
                corner_uv[rows] = group[g].uv[pk[rows]]
            corner_uv = corner_uv.reshape(-1, 2)
        return face_mat, corner_col, corner_uv


def read(path):
    with zipfile.ZipFile(path) as z:
        r = _Reader(z)
        root_path = _root_part(z)
        root = r.part(root_path)[0]
        unit = root.get("unit", "millimeter")
        if unit not in UNIT_M:
            raise ValueError(f"3MF: неизвестная единица {unit!r}")
        r.scene.unit_m = UNIT_M[unit]
        build = root.find("c:build", NS)
        for it in (build if build is not None else []):
            ipath = it.get(f"{{{PROD}}}path")
            ipath = posixpath.normpath(ipath.lstrip("/")) if ipath else root_path
            r.scene.nodes.append(r.node(ipath, it.get("objectid"), _matrix(it.get("transform"))))
        if root.find("c:metadata", NS) is not None or any(n.startswith("Metadata/") for n in z.namelist()):
            r.scene.warn("3MF: метаданные и миниатюра не переносятся")
    if not r.scene.nodes:
        r.scene.warn("3MF: в <build> нет ни одной детали")
    return r.scene


# --- writer ----------------------------------------------------------------------------------

GREY =(0.8, 0.8, 0.8, 1.0)
CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
    '</Types>\n')
ROOT_RELS = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    f'<Relationships xmlns="{RELS}">'
    f'<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="{START_PART}"/></Relationships>\n')


def _num(x):
    """Shortest exact decimal (round-trips the float64)."""
    return repr(float(x))


def _tf_attr(m, scale):
    """4x4 column-vector matrix -> ' transform="…"' (empty for identity)."""
    if np.array_equal(m, np.eye(4)):
        return ""
    v = [*m[:3, :3].T.ravel(), *(m[:3, 3] * scale)]
    return f' transform="{" ".join(_num(x) for x in v)}"'


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


class _Writer:
    def __init__(self, scene, scale):
        self.scene, self.scale = scene, scale
        self.ids = 0
        self.groups, self.objects, self.items = [], [], []
        self.base_id, self.base_of = None, {}     # material index (-1 = none) -> base index
        self.mesh_obj = {}                        # mesh index -> (object id, name)
        self.textures, self.tex_parts, self.tex_id = [], [], {}   # xml, (part, bytes), id(tex)
        self.tex_names = textures.file_names(textures.used_textures(scene), ascii_only=True)
        self.losses = []

    def loss(self, msg):
        if msg not in self.losses:
            self.losses.append(msg)

    def new_id(self):
        self.ids += 1
        return self.ids

    def base(self, mi):
        if self.base_id is None:
            self.base_id = self.new_id()
        if mi not in self.base_of:
            self.base_of[mi] = len(self.base_of)
        return self.base_of[mi]

    def texture2d(self, tex):
        """Texture -> <m:texture2d> id, or None (with a loss line) unless PNG/JPEG."""
        if id(tex) not in self.tex_id:
            k = textures.kind(tex.data, tex.name)
            if k not in textures.WEB_KINDS:
                self.loss(f"3MF: текстура «{tex.name}» в формате {(k or 'неизвестном').upper()} "
                          "не записана — 3MF допускает только PNG и JPEG")
                self.tex_id[id(tex)] = None
            else:
                tid = self.tex_id[id(tex)] = self.new_id()
                part = f"/3D/Texture/{self.tex_names[id(tex)]}"
                self.tex_parts.append((part, tex.data))
                self.textures.append(f' <m:texture2d id="{tid}" path="{_esc(part)}" '
                                     f'contenttype="{textures.KINDS[k][1]}"/>\n')
        return self.tex_id[id(tex)]

    def tex_of_face(self, mesh, fm):
        """(t,) texture2d id per triangle, -1 = drawn without a texture."""
        out = np.full(len(fm), -1, dtype=np.int64)
        for mi in set(fm.tolist()) - {-1}:
            tex = self.scene.materials[mi].texture
            if tex is None:
                continue
            if mesh.uv is None:
                self.loss("3MF: текстура у меша без UV не записана")
                continue
            tid = self.texture2d(tex)
            if tid is not None:
                out[fm == mi] = tid
                self.loss("3MF: у граней с текстурой имя и цвет материала не хранятся "
                          "(в 3MF это просто картинка)")
        return out

    def mesh_object(self, mesh, name, positions, tris, tri_face):
        """One <object> with a <mesh>; tris index mesh.corners (geom.triangulate / bake).
        Per triangle: a texture group (textured material), else the colour group (corner
        colours), else a base material; once any triangle has a property, all have one."""
        oid = self.new_id()
        v = mesh.corners[tris]
        fm = (mesh.face_material[tri_face] if mesh.face_material is not None
              else np.full(len(tris), -1, dtype=np.int64))
        # per triangle: v1 v2 v3, pid, p1 p2 p3 (pid -1 = no property)
        rows = np.full((len(tris), 7), -1, dtype=np.int64)
        rows[:, :3] = v
        tex = self.tex_of_face(mesh, fm)
        plain = np.flatnonzero(tex < 0)
        if mesh.uv is not None and len(plain):
            self.loss("3MF: UV граней без текстуры отброшены (3MF хранит UV только вместе "
                      "с текстурой)")
        for tid in sorted(set(tex.tolist()) - {-1}):
            sel = np.flatnonzero(tex == tid)
            uv = mesh.uv[tris[sel]].reshape(-1, 2)
            palette, idx = geom.unique_rows(uv, return_inverse=True)
            gid = self.new_id()
            self.groups.append(f' <m:texture2dgroup id="{gid}" texid="{tid}">'
                               + _block('<m:tex2coord u="%r" v="%r"/>',
                                        np.asarray(palette, dtype=np.float64))
                               + "</m:texture2dgroup>\n")
            rows[sel, 3] = gid
            rows[sel, 4:] = idx.reshape(-1, 3)
        if mesh.colors is not None and not geom.colors_redundant(mesh, self.scene.materials):
            if len(plain) < len(tris):
                self.loss("3MF: цвета вершин на гранях с текстурой отброшены")
            if len(plain):
                cg = self.new_id()
                keys = _color_keys(mesh.colors)
                palette, idx = np.unique(keys[tris[plain]], return_inverse=True)
                width = 2 * mesh.colors.shape[1]
                self.groups.append(f' <m:colorgroup id="{cg}">'
                                   + _block(f'<m:color color="#%0{width}X"/>',
                                            palette.astype(np.int64)[:, None])
                                   + "</m:colorgroup>\n")
                rows[plain, 3] = cg
                rows[plain, 4:] = idx.reshape(-1, 3)
        elif (np.any(fm >= 0) or len(plain) < len(tris)) and len(plain):
            # the spec wants an object-level property once triangles carry any, so faces
            # without a material get an explicit NO_MATERIAL base instead of inheriting one;
            # base indices are given in order of first use
            mats, first, inv = np.unique(fm[plain], return_index=True, return_inverse=True)
            for mi in mats[np.argsort(first, kind="stable")].tolist():
                self.base(mi)
            rows[plain, 3] = self.base_id
            rows[plain, 4:] = np.array([self.base_of[mi] for mi in mats.tolist()])[inv][:, None]
        obj_pid, has = "", rows[:, 3] >= 0
        if has.any():
            if not has.all():
                raise AssertionError("3MF: свойства есть не у всех треугольников")
            obj_pid = f' pid="{rows[0, 3]}" pindex="{rows[0, 4]}"'
            # pid on every triangle, even when it equals the object's: lib3mf 2.5 (and so
            # programs built on it) ignores p1/p2/p3 of a triangle without its own pid
            same = (rows[:, 4] == rows[:, 5]) & (rows[:, 5] == rows[:, 6])
            tri_text = _block_kinds(
                ['<triangle v1="%d" v2="%d" v3="%d" pid="%d" p1="%d"/>\n',
                 '<triangle v1="%d" v2="%d" v3="%d" pid="%d" p1="%d" p2="%d" p3="%d"/>\n'],
                (~same).astype(np.int8), rows)
        else:
            tri_text = _block('<triangle v1="%d" v2="%d" v3="%d"/>\n', rows[:, :3])
        p = np.asarray(positions, dtype=np.float64) * self.scale
        out = [f' <object id="{oid}" name="{_esc(name)}" type="model"{obj_pid}><mesh><vertices>\n',
               _block('<vertex x="%r" y="%r" z="%r"/>\n', p),
               "</vertices><triangles>\n", tri_text, "</triangles></mesh></object>\n"]
        self.objects.append("".join(out))
        return oid

    def node(self, node):
        """-> object id that represents the node (None if it holds nothing)."""
        comps = []
        if node.mesh is not None:
            mesh = self.scene.meshes[node.mesh]
            if mesh.n_faces == 0:
                self.loss("3MF: пустые меши (без граней) отброшены")
            elif node.mesh in self.mesh_obj:
                oid, name = self.mesh_obj[node.mesh]
                if name != (node.name or mesh.name):
                    self.loss("3MF: экземпляры одного меша носят имя первого "
                              "(у инстансов в 3MF нет своих имён)")
                comps.append(oid)
            else:
                name = node.name or mesh.name
                tris, src = geom.triangulate(mesh)
                oid = self.mesh_object(mesh, name, mesh.positions, tris, src)
                self.mesh_obj[node.mesh] = (oid, name)
                comps.append(oid)
        if not node.children:
            if node.mesh is None:
                self.loss("3MF: пустые узлы (без меша и детей) отброшены")
            return comps[0] if comps else None
        parts = [f'<component objectid="{c}"/>' for c in comps]
        for ch in node.children:
            cid = self.node(ch)
            if cid is not None:
                parts.append(f'<component objectid="{cid}"{_tf_attr(ch.matrix, self.scale)}/>')
        if not parts:
            return None
        oid = self.new_id()
        self.objects.append(f' <object id="{oid}" name="{_esc(node.name)}" type="model">'
                            f'<components>{"".join(parts)}</components></object>\n')
        return oid


ZLEVEL = 3              # about twice as fast as the default 6 for ~4 % more bytes (D24)
ZCHUNK = 4 << 20        # deflate piece; fixed, so the bytes do not depend on the CPU count
ZWINDOW = 32 << 10


def _deflate_piece(data, start, end):
    """Raw deflate of data[start:end] primed with the 32 KB before it (as pigz does): pieces
    ending in a sync flush concatenate into one ordinary deflate stream."""
    prime = {"zdict": data[max(0, start - ZWINDOW):start]} if start else {}
    c = zlib.compressobj(ZLEVEL, zlib.DEFLATED, -15, **prime)
    last = end >= len(data)
    return c.compress(data[start:end]) + c.flush(zlib.Z_FINISH if last else zlib.Z_SYNC_FLUSH)


class _Precompressed:
    """Stands in for the zip member's compressor: hands out pieces deflated beforehand."""
    def __init__(self, pieces):
        self.pieces = pieces

    def compress(self, _data):
        return next(self.pieces)

    def flush(self):
        return b""


def _write_big(z, name, data):
    """writestr() for a large member, deflated in threads (zlib releases the GIL)."""
    if len(data) <= 2 * ZCHUNK:
        z.writestr(name, data)
        return
    zi = zipfile.ZipInfo(name, time.localtime(time.time())[:6])
    zi.compress_type, zi.compress_level = zipfile.ZIP_DEFLATED, ZLEVEL
    zi.external_attr = 0o600 << 16
    view = memoryview(data)
    starts = range(0, len(data), ZCHUNK)
    with ThreadPoolExecutor(max(1, min(8, os.cpu_count() or 1))) as pool:
        pieces = pool.map(lambda s: _deflate_piece(view, s, s + ZCHUNK), starts)
        with z.open(zi, "w", force_zip64=len(data) > zipfile.ZIP64_LIMIT) as f:
            if not hasattr(f, "_compressor"):       # zipfile internals changed: plain path
                f.write(data)
                return
            f._compressor = _Precompressed(iter(pieces))
            for s in starts:
                f.write(view[s:s + ZCHUNK])


def losses(scene):
    w = []
    if any(not m.is_triangles for m in scene.meshes):
        w.append("3MF: многоугольники разбиты на треугольники")
    if any(m.normals is not None for m in scene.meshes):
        w.append("3MF: нормали отброшены (формат их не хранит)")
    return w


def write(scene, path, scale=1.0, unit_m=1e-3):
    """scale: file units -> output units; unit_m: metres per output unit (one of UNIT_M).
    Hierarchy and instancing -> components; materials -> basematerials; corner colours that are
    more than the face material -> a colorgroup per mesh. Returns warning lines."""
    unit = next((n for n, v in UNIT_M.items() if np.isclose(v, unit_m, rtol=1e-9)), None)
    if unit is None:
        raise ValueError(f"3MF: единица {unit_m} м не выражается в 3MF")
    wr = _Writer(scene, scale)
    w = losses(scene)
    if any(geom.is_mirror(n.matrix) for n, *_ in scene.walk()):
        # consumers disagree on the winding of mirrored components: bake them into vertices
        w.append("3MF: есть зеркальные трансформы — иерархия и инстансы запечены в вершины")
        for b in geom.bake(scene):
            if b.mesh.n_faces:
                wr.items.append((wr.mesh_object(b.mesh, b.name, b.positions, b.tris, b.tri_face),
                                 np.eye(4)))
    else:
        for root in scene.nodes:
            oid = wr.node(root)
            if oid is not None:
                wr.items.append((oid, root.matrix))
    w += wr.losses
    if wr.base_id is not None:
        mats = [(NO_MATERIAL, GREY) if mi < 0 else
                (scene.materials[mi].name, scene.materials[mi].color or GREY)
                for mi in sorted(wr.base_of, key=wr.base_of.get)]
        wr.groups.insert(0, f' <basematerials id="{wr.base_id}">'
                         + "".join(f'<base name="{_esc(n)}" displaycolor="#{_hex(c)}"/>'
                                   for n, c in mats) + "</basematerials>\n")
    if not wr.items:
        w.append("3MF: нечего записать — в сцене нет граней")
    model = "".join([
        '<?xml version="1.0" encoding="UTF-8"?>\n',
        f'<model unit="{unit}" xml:lang="en-US" xmlns="{CORE}" xmlns:m="{MAT}">\n<resources>\n',
        *wr.textures, *wr.groups, *wr.objects, "</resources>\n<build>\n",
        *(f' <item objectid="{oid}"{_tf_attr(m, scale)}/>\n' for oid, m in wr.items),
        "</build>\n</model>\n"])
    types = CONTENT_TYPES
    for ext in sorted({posixpath.splitext(p)[1][1:].lower() for p, _ in wr.tex_parts}):
        mime = next(m for e, m in textures.KINDS.values() if e == "." + ext)
        types = types.replace("</Types>", f'<Default Extension="{ext}" ContentType="{mime}"/>'
                                          "</Types>")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=ZLEVEL) as z:
        z.writestr("[Content_Types].xml", types)
        z.writestr("_rels/.rels", ROOT_RELS)
        _write_big(z, "3D/3dmodel.model", model.encode("utf-8"))
        if wr.tex_parts:
            z.writestr("3D/_rels/3dmodel.model.rels", (
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                f'<Relationships xmlns="{RELS}">'
                + "".join(f'<Relationship Target="{_esc(p)}" Id="tex{i}" Type="{TEXTURE_REL}"/>'
                          for i, (p, _) in enumerate(wr.tex_parts))
                + "</Relationships>\n"))
            for p, data in wr.tex_parts:
                # images are compressed already
                z.writestr(p.lstrip("/"), data, zipfile.ZIP_STORED)
    return w
