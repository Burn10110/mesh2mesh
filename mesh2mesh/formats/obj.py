"""Wavefront OBJ + MTL. No units; Y up, -Z forward (the de-facto convention of Blender, Maya,
3ds Max exporters): file (x, y, z) <-> scene (x, -z, y).

Reader: v (with optional unofficial r g b), vt, vn, f with v/vt/vn and negative indices,
o / g (each starts a new object), usemtl, mtllib (Kd, d, map_Kd). l / p / curves are dropped
with a warning. Writer: one `o` per mesh instance (transforms baked), v/vt/vn, usemtl, a .mtl
beside and the base colour images beside it (map_Kd; Kd white under a texture, D21).
"""

import os
import re

import numpy as np

from .. import geom, textures
from ..model import NO_MATERIAL, Material, Mesh, Node, Scene

# scene (Z up) -> OBJ (Y up): (x, y, z) -> (x, z, -y); the inverse for reading
TO_OBJ = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
FROM_OBJ = TO_OBJ.T


# map statement options -> how many arguments they take (None = 1..3 numbers)
MAP_OPTIONS = {"-blendu": 1, "-blendv": 1, "-cc": 1, "-clamp": 1, "-mm": 2, "-o": None,
               "-s": None, "-t": None, "-texres": 1, "-bm": 1, "-imfchan": 1, "-type": 1,
               "-boost": 1}
OTHER_MAPS = ("map_ka", "map_ks", "map_ns", "map_d", "map_bump", "bump", "disp", "decal",
              "refl", "norm", "map_pr", "map_pm", "map_ps", "map_ke", "map_rma", "map_orm")


def _map_args(tokens):
    """Tokens after 'map_Kd' -> (file reference, {option: values}); the file name may hold
    spaces, so it is everything after the options."""
    opts, i = {}, 0
    while i < len(tokens) and tokens[i].lower() in MAP_OPTIONS:
        o, n = tokens[i].lower(), MAP_OPTIONS[tokens[i].lower()]
        i += 1
        vals = []
        while i < len(tokens) and (len(vals) < n if n else len(vals) < 3):
            if n is None:
                try:
                    float(tokens[i])
                except ValueError:
                    break
            vals.append(tokens[i])
            i += 1
        opts[o] = vals
    return " ".join(tokens[i:]), opts


def _read_mtl(path, scene, warned, tex_cache):
    """-> {material name: [rgba list, Texture or None]}"""
    mats, name = {}, None
    folder = os.path.dirname(path)
    try:
        fh = open(path, encoding="utf-8", errors="replace")
    except OSError:
        scene.warn(f"OBJ: библиотека материалов не найдена: {os.path.basename(path)}")
        return mats
    with fh:
        for line in fh:
            t = line.split()
            if not t:
                continue
            key = t[0].lower()
            if key == "newmtl":
                name = line.split(None, 1)[1].strip() if len(t) > 1 else "material"
                mats[name] = [[1.0, 1.0, 1.0, 1.0], None]
            elif name is None:
                continue
            elif key == "kd" and len(t) >= 4:
                mats[name][0][:3] = [float(x) for x in t[1:4]]
            elif key == "d" and len(t) >= 2:
                mats[name][0][3] = float(t[1])
            elif key == "tr" and len(t) >= 2:
                mats[name][0][3] = 1.0 - float(t[1])
            elif key == "map_kd" and len(t) >= 2:
                ref, opts = _map_args(line.strip().split()[1:])
                if any(o in opts and any(float(v) != d for v in opts[o])
                       for o, d in (("-o", 0.0), ("-s", 1.0))):
                    scene.warn(f"OBJ: сдвиг/масштаб UV текстуры «{ref}» не переносится")
                key = ref.replace("\\", "/").lower()
                if key not in tex_cache:
                    tex = textures.load_file(ref, folder)
                    if tex is None:
                        scene.warn(f"OBJ: нет файла текстуры {ref} — текстура отброшена")
                    else:   # one image under several spellings -> one Texture
                        tex = next((t for t in tex_cache.values()
                                    if t is not None and t.data == tex.data), tex)
                    tex_cache[key] = tex
                mats[name][1] = tex_cache[key]
            elif key in OTHER_MAPS and "map" not in warned:
                warned.add("map")
                scene.warn("OBJ: переносится только текстура цвета map_Kd (карты нормалей, "
                           "блеска, прозрачности — нет)")
    return mats


class _Obj:
    """Faces collected for one object; indices are global (0-based) until finalised."""

    def __init__(self, name):
        self.name = name
        self.faces = 0          # faces so far (both readers keep it for `o`/`g`)
        self.sizes, self.v, self.vt, self.vn, self.mat = [], [], [], [], []


class _State:
    """Everything but v/vt/vn/f: objects, materials, warnings. Shared by both readers, so the
    directives behave (and warn) the same."""

    def __init__(self, path):
        self.path = path
        self.scene = Scene(unit_m=None)
        self.objs = [_Obj(os.path.splitext(os.path.basename(path))[0])]
        self.mtl, self.cur_mat, self.warned, self.tex_cache = {}, -1, set(), {}
        self.mat_index = {}

    def directive(self, k, t, line):
        scene = self.scene
        if k in ("o", "g"):
            name = line.split(None, 1)[1].strip() if len(t) > 1 else k
            if self.objs[-1].faces:
                self.objs.append(_Obj(name))
            else:
                self.objs[-1].name = name
        elif k == "usemtl":
            name = line.split(None, 1)[1].strip() if len(t) > 1 else "material"
            if name not in self.mat_index:
                self.mat_index[name] = len(scene.materials)
                rgba, tex = self.mtl.get(name, (None, None))
                if rgba is None:
                    scene.warn(f"OBJ: материал «{name}» не описан в .mtl — без цвета")
                scene.materials.append(Material(name, tuple(rgba) if rgba else None, tex))
            self.cur_mat = self.mat_index[name]
        elif k == "mtllib":
            for lib in line.split(None, 1)[1].strip().split():
                self.mtl.update(_read_mtl(os.path.join(os.path.dirname(self.path), lib), scene,
                                          self.warned, self.tex_cache))
        elif k in ("l", "p", "curv", "curv2", "surf", "cstype"):
            scene.warn("OBJ: линии, точки и кривые/поверхности отброшены")
        elif k in ("s", "vp"):
            pass
        elif k not in self.warned:
            self.warned.add(k)
            scene.warn(f"OBJ: неизвестная директива «{k}» пропущена")


def read(path):
    with open(path, "rb") as fh:
        data = fh.read()
    scene = _read_fast(path, data)
    return scene if scene is not None else _read_lines(path)


def _read_lines(path):
    """The reference reader, line by line (any OBJ the fast path declines)."""
    st = _State(path)
    V, VC, VT, VN = [], [], [], []

    def idx(tok, n, line_no):
        i = int(tok)
        i = i - 1 if i > 0 else n + i
        if not 0 <= i < n:
            raise ValueError(f"OBJ: строка {line_no}: индекс {tok} вне диапазона")
        return i

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, 1):
            t = line.split()
            if not t or t[0].startswith("#"):
                continue
            k = t[0]
            if k == "v":
                V.append([float(x) for x in t[1:4]])
                if len(t) >= 7:
                    VC.append((len(V) - 1, [float(x) for x in t[4:7]]))
            elif k == "vt":
                VT.append([float(x) for x in t[1:3]] if len(t) >= 3 else [float(t[1]), 0.0])
            elif k == "vn":
                VN.append([float(x) for x in t[1:4]])
            elif k == "f":
                o = st.objs[-1]
                refs = [c.split("/") for c in t[1:]]
                if len(refs) < 3:
                    continue
                o.faces += 1
                o.sizes.append(len(refs))
                o.mat.append(st.cur_mat)
                for r in refs:
                    o.v.append(idx(r[0], len(V), line_no))
                    o.vt.append(idx(r[1], len(VT), line_no) if len(r) > 1 and r[1] else -1)
                    o.vn.append(idx(r[2], len(VN), line_no) if len(r) > 2 and r[2] else -1)
            else:
                st.directive(k, t, line)

    V = np.array(V, dtype=np.float64).reshape(-1, 3)
    col = None
    if VC:
        col = np.ones((len(V), 4))
        for i, c in VC:
            col[i, :3] = c
    return _finish(st, V, np.array(VT, dtype=np.float64).reshape(-1, 2),
                   np.array(VN, dtype=np.float64).reshape(-1, 3), col)


# --- fast path: whole blocks of v / vt / vn / f lines through numpy ---------------------------

# bytes a v/vt/vn/f line may hold for the fast path: printable ASCII, tab, \v, \f. Anything
# else (non-ASCII, where str.split() knows more whitespace; NUL, which numpy strings drop at
# the end; other control bytes) sends the file to the line reader.
_PLAIN = bytes(range(0x20, 0x7F)) + b"\t\n\x0b\x0c"
LINES_PER_BLOCK = 500_000
_V, _VT, _VN, _F, _EVENT, _SKIP = range(6)


class _Lines:
    """data.split(b"\\n") without a bytes object per line: lines[i] slices on demand."""

    def __init__(self, data, starts, ends):
        self.data, self.starts, self.ends = data, starts, ends

    def __getitem__(self, i):
        return self.data[self.starts[i]:self.ends[i]]

    def join(self, idx):
        """b"\\n".join(lines[i] for i in idx): consecutive lines as one slice of the data."""
        if not len(idx):
            return b""
        cut = np.flatnonzero(np.diff(idx) != 1) + 1
        first, last = np.r_[idx[0], idx[cut]], np.r_[idx[cut - 1], idx[-1]]
        d, s, e = self.data, self.starts[first].tolist(), self.ends[last].tolist()
        return b"\n".join([d[a:b] for a, b in zip(s, e)])


def _line_kinds(data):
    """Bytes with \\n line ends -> (lines, kind per line)."""
    arr = np.frombuffer(data, np.uint8)
    nl = np.flatnonzero(arr == 0x0A)
    starts = np.r_[0, nl + 1]
    length = np.r_[nl, len(arr)] - starts
    lines = _Lines(data, starts, starts + length)

    def byte(j):
        b = np.zeros(len(starts), np.uint8)
        ok = j < length
        b[ok] = arr[starts[ok] + j]
        return b

    b0, b1, b2 = byte(0), byte(1), byte(2)
    kind = np.full(len(starts), _EVENT, np.int8)
    kind[(length == 0) | (b0 == ord("#"))] = _SKIP
    v = b0 == ord("v")
    kind[v & (b1 == ord(" "))] = _V
    kind[v & (b1 == ord("t")) & (b2 == ord(" "))] = _VT
    kind[v & (b1 == ord("n")) & (b2 == ord(" "))] = _VN
    kind[(b0 == ord("f")) & (b1 == ord(" "))] = _F
    if data.translate(None, _PLAIN):            # odd bytes somewhere: which lines hold them?
        lut = np.ones(256, bool)
        lut[np.frombuffer(_PLAIN, np.uint8)] = False
        bad = np.flatnonzero(lut[arr])
        bad_lines = np.searchsorted(starts, bad, side="right") - 1
        if np.any(kind[bad_lines] <= _F):
            return lines, None
    return lines, kind


def _blocks(lines, idx):
    for s in range(0, len(idx), LINES_PER_BLOCK):
        yield lines.join(idx[s:s + LINES_PER_BLOCK])


def _numbers(lines, idx, key, cols_for, width):
    """v / vt / vn lines -> (n, m) float64 of the token columns cols_for(token count), or None
    unless every line has the same token count and cols_for accepts it. Values = float() of
    each token (numpy parses bytes through it). No lines -> (0, width)."""
    out, k = [], None
    for block in _blocks(lines, idx):
        tok = block.split()
        n = block.count(b"\n") + 1
        if k is None:
            k = len(tok) // n
        cols = cols_for(k)
        # the key once per line, at every k-th token: each line is exactly k tokens
        if cols is None or n * k != len(tok) or tok.count(key) != n or tok[::k].count(key) != n:
            return None
        try:
            out.append(np.stack([np.array(tok[c::k], dtype=np.float64) for c in cols], 1))
        except ValueError:
            return None
    return np.concatenate(out) if out else np.zeros((0, width))


# f-line bytes the byte parser takes: digits, '-', '/', whitespace, the line-start 'f'
_FACE_DELETE = bytes(range(0x30, 0x3A)) + b"-/ \t\n\x0b\x0cf"
_POW10 = 10 ** np.arange(18, dtype=np.int64)
# digit count -> mask of the bytes before the number in its 8-byte word (8+ digits: none)
_LEAD = np.array([(1 << 8 * (8 - n)) - 1 if n < 8 else 0 for n in range(19)], np.uint64)
_ZEROS8 = np.uint64(0x3030303030303030)


def _decimals(a, end, length):
    """Unsigned decimal numbers of `length` (1..18) ASCII digits ending at `end` in byte array
    `a` -> int64. Up to 8 digits: the 8 bytes up to the end as one little-endian word, bytes
    before the number set to '0', digits summed pairwise (SWAR); longer ones: digit matrix."""
    pad = np.concatenate([np.full(8, 0x30, np.uint8), a])
    words = np.ndarray((len(pad) - 7,), "<u8", pad, 0, (1,))
    x = words[end]                              # pad[end:end + 8] = a[end - 8:end]
    lead = _LEAD[length]
    x &= ~lead
    x |= _ZEROS8 & lead
    x -= _ZEROS8                                # byte k = digit k (first digit in the low byte)
    x = (x * np.uint64(10) + (x >> np.uint64(8))) & np.uint64(0x00FF00FF00FF00FF)
    x = (x * np.uint64(100) + (x >> np.uint64(16))) & np.uint64(0x0000FFFF0000FFFF)
    x = (x * np.uint64(10000) + (x >> np.uint64(32))) & np.uint64(0xFFFFFFFF)
    val = x.view(np.int64)
    for L in np.unique(length[length > 8]).tolist():
        m = np.flatnonzero(length == L)
        dig = a[end[m, None] - L + np.arange(L)].astype(np.int64) - 0x30
        val[m] = dig @ _POW10[L - 1::-1]
    return val


def _face_bytes(block, n_lines):
    """One block of f lines, parsed on its bytes: (sizes, v, vt, vn, has_vt, has_vn) over all
    corners (raw OBJ indices; vt/vn = 0 where absent), or None for anything else than corners
    `a`, `a/b`, `a//c`, `a/b/c` of decimal numbers ('-' leading), up to 18 digits (the token
    path then decides). vt / vn present = a number in slot 1 / 2, as the line reader's
    split("/") sees it. Works on the number runs, not on every byte."""
    # lines start with "f " (_line_kinds): one 'f' per line = nowhere else
    if block.translate(None, _FACE_DELETE) or block.count(b"f") != n_lines:
        return None
    a = np.frombuffer(block, np.uint8)
    t = a - np.uint8(0x2D)                      # '-' 0, '/' 2, digits 3..12 ('.' gone above)
    num = ((t <= 12) & (t != 2)).view(np.int8)
    edge = np.flatnonzero(np.diff(num)) + 1     # number runs: start, end, start, … (a[0] = 'f')
    n_start, n_end = edge[0::2], edge[1::2]
    if num[-1]:
        n_end = np.r_[n_end, len(a)]
    if not len(n_start):
        return None
    neg = a[n_start] == 0x2D
    d_start = n_start + neg
    length = n_end - d_start
    if block.count(b"-") != int(neg.sum()) or length.min() < 1 or length.max() > 18:
        return None                             # '-' inside or alone, too long
    val = _decimals(a, n_end, length)
    val[neg] = -val[neg]
    # a corner starts after whitespace; inside it only 1-2 slashes lie between numbers (a
    # corner starting with '/' fails here: its gap holds whitespace); none after its last one
    c_first = a[n_start - 1] <= 0x20
    gap = np.r_[0, n_start[1:] - n_end[:-1]]
    inner = ~c_first
    if not c_first[0] or np.any(inner & ((gap > 2) | ((gap == 2) & (a[n_start - 2] != 0x2F)))):
        return None
    c_idx = np.flatnonzero(c_first)
    c_last = np.r_[c_idx[1:] - 1, len(n_start) - 1]
    if np.any(a[np.minimum(n_end[c_last], len(a) - 1)] == 0x2F):
        return None
    slot = np.cumsum(np.where(inner, gap, 0))
    slot -= np.repeat(slot[c_idx], np.diff(np.r_[c_idx, len(slot)]))
    if slot.max() > 2:
        return None
    n_c = len(c_idx)
    out, present = [val[c_idx]], [None]         # slot 0: each corner's first number
    for k in (1, 2):
        m = np.flatnonzero(slot == k)
        if len(m) == n_c:                       # at most one per corner: every corner has it
            out.append(val[m])
            present.append(np.ones(n_c, bool))
            continue
        corner = np.searchsorted(c_idx, m, side="right") - 1
        out.append(np.zeros(n_c, np.int64))
        out[k][corner] = val[m]
        present.append(np.zeros(n_c, bool))
        present[k][corner] = True
    # a line's first corner right after "f ": one per line (else the token path counts)
    line_first = np.flatnonzero(a[n_start[c_idx] - 2] == 0x66)
    if len(line_first) != n_lines:
        return None
    sizes = np.diff(np.r_[line_first, n_c])
    return sizes, out[0], out[1], out[2], present[1], present[2]


def _face_tokens(block, n_lines):
    """_face_bytes through bytes tokens and numpy strings: any OBJ number int() reads."""
    tok = np.array(block.split())
    head = np.flatnonzero(tok == b"f")
    if len(head) != n_lines:
        return None
    sizes = np.diff(np.r_[head, len(tok)]) - 1
    keep = np.repeat(sizes >= 3, sizes)         # lines the line reader drops: not inspected
    c = np.delete(tok, head)
    z = np.zeros(len(c), np.int64)
    if not np.any(keep):
        return sizes, z, z, z, z.astype(bool), z.astype(bool)
    c = np.where(keep, c, b"0")
    slashes = np.strings.count(c, b"/")
    if slashes.max() > 2 or np.any(np.strings.startswith(c, b"/")):
        return None
    ends = np.strings.endswith(c, b"/")
    has_vt = (slashes >= 1) & ~np.where(slashes == 1, ends, np.strings.find(c, b"//") >= 0)
    has_vn = (slashes == 2) & ~ends
    nums = b" ".join(c.tolist()).replace(b"/", b" ").split()
    fields = 1 + has_vt + has_vn
    if len(nums) != int(fields.sum()):
        return None
    try:
        nums = np.array(nums).astype(np.int64)
    except (ValueError, OverflowError):
        return None
    first = np.cumsum(fields) - fields
    last = len(nums) - 1
    return (sizes, nums[first],
            np.where(has_vt, nums[np.minimum(first + 1, last)], 0),
            np.where(has_vn, nums[np.minimum(first + 1 + has_vt, last)], 0), has_vt, has_vn)


def _corners(lines, idx, counts_before):
    """f lines -> (sizes, v, vt, vn) 0-based (-1 = none), lines with < 3 corners dropped (as
    the line reader does), or None if anything is unusual. counts_before: (len(idx), 3)
    v/vt/vn lines above each f line, for negative indices and the range check."""
    sizes_all, keep_all, refs, b0 = [], [], [[], [], []], 0
    for block in _blocks(lines, idx):
        n_lines = block.count(b"\n") + 1
        parsed = _face_bytes(block, n_lines) or _face_tokens(block, n_lines)
        if parsed is None:
            return None
        sizes, cv, cvt, cvn, has_vt, has_vn = parsed
        keep = sizes >= 3
        kc = np.repeat(keep, sizes)
        before = np.repeat(counts_before[b0:b0 + n_lines][keep], sizes[keep], axis=0)
        b0 += n_lines
        sizes_all.append(sizes[keep])
        keep_all.append(keep)
        if not np.any(kc):
            continue
        if not np.all(kc):
            cv, cvt, cvn, has_vt, has_vn = (x[kc] for x in (cv, cvt, cvn, has_vt, has_vn))
        cols = [cv, cvt, cvn]
        for j, (i, present) in enumerate(zip(cols, (None, has_vt, has_vn))):
            n = before[:, j]
            i = np.where(i > 0, i - 1, n + i)
            ok = (i >= 0) & (i < n)
            if present is not None:
                i = np.where(present, i, -1)
                ok |= ~present
            if not np.all(ok):
                return None     # the line reader names the line
            refs[j].append(i)
    cat = lambda a, t=np.int64: np.concatenate(a) if a else np.zeros(0, t)
    return cat(sizes_all), cat(keep_all, bool), cat(refs[0]), cat(refs[1]), cat(refs[2])


def _read_fast(path, data):
    """The line reader's result from whole-block numpy work; None when the file has anything
    the blocks do not cover (it then goes through the line reader, which also reports errors):
    mixed token counts among v (or vt, vn) lines, v/vt/vn/f lines indented or tab-separated
    after the keyword, odd bytes in them, more than v/vt/vn per corner, bad indices."""
    if b"\r" in data:                           # universal newlines, as text mode reads them
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    lines, kind = _line_kinds(data)
    if kind is None:
        return None
    at = {k: np.flatnonzero(kind == k) for k in (_V, _VT, _VN, _F, _EVENT)}
    # the line reader: v x y z [w | r g b], vt u [v [w]], vn x y z
    V = _numbers(lines, at[_V], b"v",
                 lambda k: (1, 2, 3, 4, 5, 6) if k >= 7 else (1, 2, 3) if k >= 4 else None, 3)
    VT = _numbers(lines, at[_VT], b"vt",
                  lambda k: (1, 2) if k >= 3 else (1, 1) if k == 2 else None, 2)
    VN = _numbers(lines, at[_VN], b"vn", lambda k: (1, 2, 3) if k >= 4 else None, 3)
    if V is None or VT is None or VN is None:
        return None
    if len(at[_VT]) and len(lines[at[_VT][0]].split()) == 2:
        VT[:, 1] = 0.0                  # `vt u`: v = 0
    col = None
    if V.shape[1] == 6:
        col = np.ones((len(V), 4))
        col[:, :3] = V[:, 3:]
        V = V[:, :3]
    f_lines = at[_F]
    before = np.stack([np.searchsorted(at[k], f_lines) for k in (_V, _VT, _VN)], 1)
    res = _corners(lines, f_lines, before)
    if res is None:
        return None
    sizes, keep, cv, cvt, cvn = res

    # directives in file order; faces between two of them belong to the state after the first
    st = _State(path)
    events = at[_EVENT]
    seg = np.searchsorted(events, f_lines[keep])
    per_seg = np.bincount(seg, minlength=len(events) + 1)
    seg_obj, seg_mat = [], []
    try:
        for s in range(len(events) + 1):
            st.objs[-1].faces += int(per_seg[s])
            seg_obj.append(len(st.objs) - 1)
            seg_mat.append(st.cur_mat)
            if s == len(events):
                break
            line = lines[events[s]].decode("utf-8", "replace")
            t = line.split()
            if not t or t[0].startswith("#"):
                continue
            if t[0] in ("v", "vt", "vn", "f"):
                return None             # indented or tab-separated: the line reader
            st.directive(t[0], t, line)
    except Exception:                   # noqa: BLE001 - the line reader raises it in order
        return None
    face_obj = np.array(seg_obj, np.int64)[seg]
    face_mat = np.array(seg_mat, np.int64)[seg]
    cstart = np.r_[0, np.cumsum(sizes)]
    bounds = np.searchsorted(face_obj, np.arange(len(st.objs) + 1))
    for i, o in enumerate(st.objs):
        a, b = bounds[i], bounds[i + 1]
        o.sizes, o.mat = sizes[a:b], face_mat[a:b]
        o.v, o.vt, o.vn = (x[cstart[a]:cstart[b]] for x in (cv, cvt, cvn))
    return _finish(st, V, VT, VN, col)


def _finish(st, V, VT, VN, col):
    """Per object: its own vertices (in index order), corners, UV/normals/colours -> Scene."""
    scene = st.scene
    V = V @ FROM_OBJ.T
    VN = VN @ FROM_OBJ.T
    for o in st.objs:
        if not len(o.sizes):
            continue
        v = np.array(o.v, dtype=np.int64)
        # = np.unique(v, return_inverse=True), without its sort
        present = np.zeros(len(V), bool)
        present[v] = True
        used = np.flatnonzero(present)
        corners = (np.cumsum(present) - 1)[v]
        vt, vn = np.array(o.vt), np.array(o.vn)
        mat = np.array(o.mat, dtype=np.int64)
        if np.any(vt < 0) and np.any(vt >= 0):
            scene.warn(f"OBJ: «{o.name}»: UV есть не у всех граней — недостающие = (0, 0)")
        if np.any(vn < 0) and np.any(vn >= 0):
            scene.warn(f"OBJ: «{o.name}»: нормали есть не у всех граней — нормали отброшены")
        mesh = Mesh(o.name, V[used], np.array(o.sizes, dtype=np.int64), corners.astype(np.int64),
                    face_material=mat if np.any(mat >= 0) else None,
                    uv=np.where((vt >= 0)[:, None], VT[np.maximum(vt, 0)], 0.0) if np.any(vt >= 0) else None,
                    normals=VN[vn] if len(vn) and np.all(vn >= 0) else None,
                    colors=col[v] if col is not None else None)
        scene.nodes.append(Node(o.name, mesh=len(scene.meshes)))
        scene.meshes.append(mesh)
    return scene


def _mtl_name(name):
    return re.sub(r"\s+", "_", name.strip()) or "material"


def losses(scene):
    w = []
    if not scene.is_flat():
        w.append("OBJ: иерархия и инстансы запечены в вершины (объекты плоским списком)")
    if any(not scene.meshes[n.mesh].n_faces for n, _ in scene.instances()):
        w.append("OBJ: пустые меши (без граней) отброшены")
    coloured = [m for m in scene.meshes if m.colors is not None]
    per_vertex = [_per_vertex(m) for m in coloured]
    if any(c is None for c in per_vertex):
        w.append("OBJ: цвета на углах граней (разные у одной вершины) отброшены")
    if any(c is not None for c in per_vertex):
        w.append("OBJ: цвета вершин записаны неофициальным «v x y z r g b» (Blender, MeshLab читают)")
        if any(c is not None and np.any(c[:, 3] < 1) for c in per_vertex):
            w.append("OBJ: прозрачность цветов вершин отброшена")
    if any(m.uv is None and m.face_material is not None
           and any(scene.materials[i].texture is not None for i in set(m.face_material.tolist())
                   if i >= 0) for m in scene.meshes):
        w.append("OBJ: у части объектов с текстурой нет UV — текстура на них не ляжет")
    return w


def _per_vertex(mesh):
    """Corner colours that are the same for every corner of a vertex -> (n, 4) or None."""
    if mesh.colors is None:
        return None
    out = np.full((len(mesh.positions), 4), np.nan)
    out[mesh.corners] = mesh.colors
    return out if np.allclose(out[mesh.corners], mesh.colors) else None


CHUNK = 100_000     # lines per formatted block


def _rows(fh, fmt, rows):
    """One `fmt` line per row. Formatted in blocks by the % operator (C code; the same text as
    an f-string per number) instead of a Python call per line."""
    line = fmt + "\n"
    for s in range(0, len(rows), CHUNK):
        part = rows[s:s + CHUNK]
        fh.write((line * len(part)) % tuple(part.ravel().tolist()))


def _distinct(rows):
    """-> distinct rows in order of first use, (n,) index of each row into them."""
    uniq, first, inv = geom.unique_rows(rows, return_index=True, return_inverse=True)
    order = np.argsort(first)
    rank = np.empty(len(order), np.int64)
    rank[order] = np.arange(len(order))
    return uniq[order], rank[inv]


_FACE_FMT = {}


def _faces(fh, ref, sizes, vals):
    """`f` lines: sizes (f,), vals (sum(sizes), k) ints in corner order; ref = '%d/%d/%d'…"""
    templates = _FACE_FMT.setdefault(ref, {})
    c = 0
    for s in range(0, len(sizes), CHUNK):
        part = sizes[s:s + CHUNK]
        n = int(part.sum())
        for k in np.unique(part).tolist():
            if k not in templates:
                templates[k] = "f " + " ".join([ref] * k) + "\n"
        fmt = "".join([templates[k] for k in part.tolist()])
        fh.write(fmt % tuple(vals[c:c + n].ravel().tolist()))
        c += n


def write(scene, path, scale=1.0, unit_m=None):
    """scale: file units -> output units. Writes <path> and <stem>.mtl. Returns warning lines."""
    baked = geom.bake(scene)
    mtl_path = os.path.splitext(path)[0] + ".mtl"
    used_mats = sorted({int(i) for b in baked if b.mesh.face_material is not None
                        for i in np.unique(b.mesh.face_material) if i >= 0})
    names, seen = {}, set()
    for i in used_mats:
        n = base = _mtl_name(scene.materials[i].name)
        k = 1
        while n in seen:
            n, k = f"{base}.{k:03d}", k + 1
        names[i], seen = n, seen | {n}
    off_v = off_vt = off_vn = 1
    # usemtl stays in force across `o` lines: material-less faces after a material need an
    # explicit (defined) NO_MATERIAL, or every reader paints them with the previous one
    cur, need_none = None, False
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# mesh2mesh\n")
        if used_mats:
            fh.write(f"mtllib {os.path.basename(mtl_path)}\n")
        for b in baked:
            m = b.mesh
            fh.write(f"o {b.name}\n")
            p = (b.positions * scale) @ TO_OBJ.T
            vcol = _per_vertex(m)
            if vcol is not None:
                _rows(fh, "v %.10g %.10g %.10g %.6g %.6g %.6g", np.c_[p, vcol[:, :3]])
            else:
                _rows(fh, "v %.10g %.10g %.10g", p)
            # vt / vn: each distinct value once (first-use order), faces point at it
            vt = vn = None
            n_vt = n_vn = 0
            if m.uv is not None:
                vals, vt = _distinct(m.uv)
                _rows(fh, "vt %.10g %.10g", vals)
                n_vt = len(vals)
            if m.normals is not None:
                vals, vn = _distinct(geom.apply_normals(b.matrix, m.normals) @ TO_OBJ.T)
                _rows(fh, "vn %.7g %.7g %.7g", vals)
                n_vn = len(vals)
            # faces: keep n-gons as they are (OBJ holds them); only winding flips for mirrors
            sizes = m.face_sizes
            order = geom.corner_order(m, geom.is_mirror(b.matrix))
            cols = [m.corners[order] + off_v]
            if vt is not None:
                cols.append(vt[order] + off_vt)
            if vn is not None:
                cols.append(vn[order] + off_vn)
            ref = "%d" + ("/%d" if vt is not None else "/" if vn is not None else "") + (
                "/%d" if vn is not None else "")
            vals = np.stack(cols, 1)
            fm = m.face_material if m.face_material is not None else np.full(m.n_faces, -1)
            runs = np.flatnonzero(np.r_[True, fm[1:] != fm[:-1]]) if len(fm) else []
            ends = np.r_[runs[1:], len(fm)] if len(fm) else []
            cstart = np.r_[0, np.cumsum(sizes)]
            for r0, r1 in zip(runs, ends):
                mi = int(fm[r0])
                if mi != cur:
                    if mi >= 0:
                        fh.write(f"usemtl {names[mi]}\n")
                    elif cur is not None:       # back to "no material" after a usemtl
                        fh.write(f"usemtl {NO_MATERIAL}\n")
                        need_none = True
                    cur = mi
                _faces(fh, ref, sizes[r0:r1], vals[cstart[r0]:cstart[r1]])
            off_v += len(m.positions)
            off_vt += n_vt
            off_vn += n_vn
    if used_mats:
        tex_names = textures.file_names(textures.used_textures(scene))
        folder = os.path.dirname(os.path.abspath(path))
        written = set()
        with open(mtl_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("# mesh2mesh\n")
            for i in used_mats:
                mat = scene.materials[i]
                r, g, b_, a = mat.color or (0.8, 0.8, 0.8, 1.0)
                if mat.texture is not None:
                    # readers multiply Kd by the map (the MTL spec) or replace it (Blender):
                    # white looks the same in both
                    r = g = b_ = 1.0
                fh.write(f"\nnewmtl {names[i]}\nKd {r:.6g} {g:.6g} {b_:.6g}\n")
                if a < 1:
                    fh.write(f"d {a:.6g}\n")
                if mat.texture is not None:
                    fname = tex_names[id(mat.texture)]
                    fh.write(f"map_Kd {fname}\n")
                    if fname not in written:
                        with open(os.path.join(folder, fname), "wb") as img:
                            img.write(mat.texture.data)
                        written.add(fname)
            if need_none:
                fh.write(f"\nnewmtl {NO_MATERIAL}\nKd 0.8 0.8 0.8\n")
    return losses(scene)
