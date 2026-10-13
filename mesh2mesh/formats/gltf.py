"""glTF 2.0: .gltf (JSON + .bin beside, or data: URIs) and .glb. Always metres, Y up, +Z forward:
scene (x, y, z) <-> glTF (x, z, -y), the same mapping as OBJ. Colours in glTF are linear
(baseColorFactor, COLOR_0); the Scene keeps sRGB (D9) -> converted both ways. UV: glTF has v
down from the top edge, the Scene keeps OBJ's v up -> v' = 1 - v.

Reader: the default scene's node tree (matrix or TRS), mesh reuse kept as instancing, primitives
of one mesh merged into one Mesh (material per face), modes triangles/strip/fan, any accessor
type incl. normalized integers, byteStride, sparse. Draco/meshopt compression cannot be read.
Writer: a glTF mesh per Mesh, a primitive per material, vertices split where corner attributes
differ, node matrices only when not identity. Base colour textures (PNG/JPEG): inside a .glb,
image files beside a .gltf.
"""

import base64
import json
import os
import struct
import urllib.parse

import numpy as np

from .. import geom, textures
from ..model import Material, Mesh, Node, Scene, Texture
from ..i18n import tr

C3 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)   # scene -> glTF
C4 = np.eye(4)
C4[:3, :3] = C3
COMP = {5120: "i1", 5121: "u1", 5122: "i2", 5123: "u2", 5125: "u4", 5126: "f4"}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
NORM_MAX = {"i1": 127, "u1": 255, "i2": 32767, "u2": 65535, "u4": 4294967295}
COMPRESSION = {"KHR_draco_mesh_compression": "Draco", "EXT_meshopt_compression": "meshopt"}
GLB_MAGIC, CHUNK_JSON, CHUNK_BIN = b"glTF", 0x4E4F534A, 0x004E4942


# --- reader ----------------------------------------------------------------------------------

def _load(path):
    with open(path, "rb") as fh:
        data = fh.read()
    binchunk = None
    if data[:4] == GLB_MAGIC:
        version, length = struct.unpack_from("<II", data, 4)
        if version != 2:
            raise ValueError(tr("GLB: версия контейнера {version}, поддерживается 2",
                                version=version))
        doc, off = None, 12
        while off + 8 <= min(length, len(data)):
            clen, ctype = struct.unpack_from("<II", data, off)
            chunk = data[off + 8:off + 8 + clen]
            if ctype == CHUNK_JSON and doc is None:
                doc = json.loads(chunk.decode("utf-8"))
            elif ctype == CHUNK_BIN and binchunk is None:
                binchunk = chunk
            off += 8 + clen
        if doc is None:
            raise ValueError(tr("GLB: нет JSON-блока"))
    else:
        try:
            doc = json.loads(data.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(tr("glTF: не JSON ({e})", e=e)) from None
    if not str(doc.get("asset", {}).get("version", "")).startswith("2"):
        raise ValueError(tr("glTF: поддерживается только версия 2.0"))
    buffers = []
    for b in doc.get("buffers", []):
        uri = b.get("uri")
        if uri is None:
            if binchunk is None:
                raise ValueError(tr("glTF: буфер без uri, но нет BIN-блока GLB"))
            buffers.append(binchunk)
        elif uri.startswith("data:"):
            buffers.append(base64.b64decode(uri.split(",", 1)[1]))
        else:
            p = os.path.join(os.path.dirname(os.path.abspath(path)), urllib.parse.unquote(uri))
            try:
                with open(p, "rb") as fh:
                    buffers.append(fh.read())
            except OSError:
                raise ValueError(tr("glTF: нет файла буфера {uri}", uri=uri)) from None
    return doc, buffers


def _quat(q):
    x, y, z, w = np.asarray(q, dtype=np.float64) / (np.linalg.norm(q) or 1.0)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _node_matrix(nd):
    """glTF node -> 4x4 column-vector matrix in glTF axes."""
    if "matrix" in nd:
        return np.array(nd["matrix"], dtype=np.float64).reshape(4, 4).T   # column-major
    m = np.eye(4)
    m[:3, :3] = _quat(nd.get("rotation", (0, 0, 0, 1))) @ np.diag(nd.get("scale", (1, 1, 1)))
    m[:3, 3] = nd.get("translation", (0, 0, 0))
    return m


def _strip_fan(idx, mode):
    """Index list -> (t, 3) triangles (glTF 2.0 spec, 3.7.2.1)."""
    if mode == 4:
        return idx[:len(idx) // 3 * 3].reshape(-1, 3)
    n = len(idx) - 2
    if n <= 0:
        return np.zeros((0, 3), np.int64)
    i = np.arange(n)
    if mode == 5:
        odd = i % 2
        return np.stack([idx[i], idx[i + 1 + odd], idx[i + 2 - odd]], 1)
    return np.stack([idx[i + 1], idx[i + 2], np.full(n, idx[0])], 1)


class _Reader:
    def __init__(self, path):
        self.doc, self.buffers = _load(path)
        self.folder = os.path.dirname(os.path.abspath(path))
        self.scene = Scene(unit_m=1.0)
        self.mesh_of, self.mat_of, self.tex_of = {}, {}, {}

    def warn(self, msg):
        self.scene.warn(msg)

    def raw(self, view_i, off, dt, n, k):
        bv = self.doc["bufferViews"][view_i]
        buf = self.buffers[bv["buffer"]]
        start = bv.get("byteOffset", 0) + off
        stride = bv.get("byteStride") or dt.itemsize * k
        if n and start + stride * (n - 1) + dt.itemsize * k > len(buf):
            raise ValueError(tr("glTF: accessor выходит за границы буфера"))
        return np.ndarray((n, k), dt, buffer=buf, offset=start, strides=(stride, dt.itemsize))

    def accessor(self, i):
        """-> (n, k) float64 (normalized integers mapped to 0..1 / -1..1)."""
        a = self.doc["accessors"][i]
        n, k = a["count"], NCOMP[a["type"]]
        code = COMP[a["componentType"]]
        dt = np.dtype("<" + code)
        if "bufferView" in a:
            out = self.raw(a["bufferView"], a.get("byteOffset", 0), dt, n, k).astype(np.float64)
        else:
            out = np.zeros((n, k))
        sp = a.get("sparse")
        if sp:
            si = sp["indices"]
            idx = self.raw(si["bufferView"], si.get("byteOffset", 0),
                           np.dtype("<" + COMP[si["componentType"]]), sp["count"], 1).ravel()
            sv = sp["values"]
            out[idx.astype(np.int64)] = self.raw(sv["bufferView"], sv.get("byteOffset", 0), dt,
                                                 sp["count"], k)
        if a.get("normalized") and code in NORM_MAX:
            out = np.maximum(out / NORM_MAX[code], -1.0)
        return out

    def image(self, ii):
        """images[ii] -> Texture or None (with a warning); cached: shared images stay shared."""
        if ii not in self.tex_of:
            img = self.doc.get("images", [])[ii]
            name = img.get("name") or f"image_{ii}"
            data = None
            if "bufferView" in img:
                bv = self.doc["bufferViews"][img["bufferView"]]
                off = bv.get("byteOffset", 0)
                data = bytes(self.buffers[bv["buffer"]][off:off + bv["byteLength"]])
            elif str(img.get("uri", "")).startswith("data:"):
                data = base64.b64decode(img["uri"].split(",", 1)[1])
            elif img.get("uri"):
                uri = urllib.parse.unquote(img["uri"])
                t = textures.load_file(uri, self.folder)
                name, data = (t.name, t.data) if t else (name, None)
                if t is None:
                    self.warn(tr("glTF: нет файла текстуры {uri} — текстура отброшена", uri=uri))
            k = textures.kind(data, name) if data else None
            if data and "." not in name and k:
                name += textures.KINDS[k][0]
            self.tex_of[ii] = Texture(name, data) if data else None
        return self.tex_of[ii]

    def material(self, gi):
        if gi not in self.mat_of:
            gm = self.doc.get("materials", [])[gi]
            name = gm.get("name") or f"material_{gi}"
            pbr = gm.get("pbrMetallicRoughness", {})
            f = pbr.get("baseColorFactor", (1, 1, 1, 1))
            rgba = (*geom.linear_to_srgb(f[:3]).tolist(), float(f[3]))
            tex = None
            info = pbr.get("baseColorTexture")
            if info is not None:
                src = self.doc.get("textures", [])[info["index"]]
                ii = src.get("source")
                if ii is None:
                    self.warn(tr("glTF: текстуры в форматах расширений (WebP, KTX2) не переносятся"))
                else:
                    tex = self.image(ii)
                if info.get("texCoord", 0) != 0:
                    self.warn(tr("glTF: текстура на втором UV-канале — показана по первому"))
                if "KHR_texture_transform" in info.get("extensions", {}):
                    self.warn(tr("glTF: сдвиг/масштаб UV текстуры (KHR_texture_transform) не "
                                 "переносится"))
                if tex is not None and not np.allclose(f[:3], 1, atol=1e-3):
                    self.warn(tr("glTF: «{name}»: оттенок baseColorFactor поверх текстуры не "
                                 "переносится", name=name))
            others = [k for k in list(gm) + list(pbr) if k.endswith("Texture")
                      and k != "baseColorTexture"]
            if others:
                self.warn(tr("glTF: переносится только текстура цвета (карты нормалей, "
                             "металла/шероховатости, свечения, затенения — нет)"))
            if gm.get("extensions") or "emissiveFactor" in gm:
                self.warn(tr("glTF: из материалов переносится только базовый цвет и прозрачность"))
            self.mat_of[gi] = len(self.scene.materials)
            self.scene.materials.append(Material(name, rgba, tex))
        return self.mat_of[gi]

    def mesh(self, mi):
        if mi in self.mesh_of:
            return self.mesh_of[mi]
        gm = self.doc["meshes"][mi]
        name = gm.get("name") or f"mesh_{mi}"
        parts = []
        for prim in gm.get("primitives", []):
            mode = prim.get("mode", 4)
            attrs = prim.get("attributes", {})
            if mode not in (4, 5, 6):
                self.warn(tr("glTF: точки и линии отброшены"))
                continue
            if "POSITION" not in attrs:
                continue
            if prim.get("targets"):
                self.warn(tr("glTF: морф-таргеты (blend shapes) не переносятся"))
            if any(k.startswith(("JOINTS_", "WEIGHTS_")) for k in attrs):
                self.warn(tr("glTF: веса скиннинга не переносятся"))
            if any(k.startswith("TEXCOORD_") and k != "TEXCOORD_0" for k in attrs):
                self.warn(tr("glTF: переносится только первый UV-канал"))
            pos = self.accessor(attrs["POSITION"])[:, :3]
            idx = (self.accessor(prim["indices"]).ravel().astype(np.int64) if "indices" in prim
                   else np.arange(len(pos)))
            if len(idx) and (idx.min() < 0 or idx.max() >= len(pos)):
                raise ValueError(tr("glTF: меш «{name}»: индекс вершины вне диапазона", name=name))
            tris = _strip_fan(idx, mode)
            get = lambda k: self.accessor(attrs[k]) if k in attrs else None
            col = get("COLOR_0")
            if col is not None:
                col = np.c_[geom.linear_to_srgb(col[:, :3]),
                            col[:, 3] if col.shape[1] > 3 else np.ones(len(col))]
            parts.append((pos, tris, get("TEXCOORD_0"), get("NORMAL"), col,
                          self.material(prim["material"]) if "material" in prim else -1))
        if not parts or not sum(len(p[1]) for p in parts):
            self.mesh_of[mi] = None
            return None
        offs = np.cumsum([0] + [len(p[0]) for p in parts])
        corners = np.concatenate([p[1] + o for p, o in zip(parts, offs)]).ravel()
        positions = np.concatenate([p[0] for p in parts]) @ C3

        def per_corner(i, width, fill, what):
            have = [p[i] is not None for p in parts]
            if not any(have):
                return None
            if not all(have):
                if fill is None:
                    self.warn(tr("glTF: «{name}»: {what} есть не у всех примитивов — отброшены",
                                 name=name, what=what))
                    return None
                self.warn(tr("glTF: «{name}»: {what} есть не у всех примитивов — у остальных "
                             "{fill}", name=name, what=what, fill=fill))
            return np.concatenate([p[i][:, :width] if p[i] is not None
                                   else np.tile(fill, (len(p[0]), 1)) for p in parts])[corners]

        uv = per_corner(2, 2, None, "UV")
        if uv is not None:
            uv = np.c_[uv[:, 0], 1.0 - uv[:, 1]]
        nrm = per_corner(3, 3, None, tr("нормали"))
        col = per_corner(4, 4, (1.0, 1.0, 1.0, 1.0), tr("цвета вершин"))
        fm = np.concatenate([np.full(len(p[1]), p[5], dtype=np.int64) for p in parts])
        self.mesh_of[mi] = len(self.scene.meshes)
        self.scene.meshes.append(Mesh(name, positions, np.full(len(fm), 3, dtype=np.int64),
                                      corners.astype(np.int64),
                                      face_material=fm if np.any(fm >= 0) else None, uv=uv,
                                      normals=nrm @ C3 if nrm is not None else None, colors=col))
        return self.mesh_of[mi]

    def node(self, i, stack=()):
        if i in stack:
            raise ValueError(tr("glTF: циклическая иерархия узлов"))
        nd = self.doc["nodes"][i]
        for key, what in (("skin", tr("скиннинг")), ("camera", tr("камеры")),
                          ("weights", tr("веса морфов"))):
            if key in nd:
                self.warn(tr("glTF: {what} не переносятся", what=what))
        if nd.get("extensions"):
            self.warn(tr("glTF: расширения узлов ({v}) не переносятся",
                         v=', '.join(nd['extensions'])))
        node = Node(nd.get("name") or f"node_{i}", C4.T @ _node_matrix(nd) @ C4)
        if "mesh" in nd:
            node.mesh = self.mesh(nd["mesh"])
        node.children = [self.node(c, stack + (i,)) for c in nd.get("children", [])]
        return node


def read(path):
    r = _Reader(path)
    doc, warn = r.doc, r.warn
    bad = [COMPRESSION[e] for e in doc.get("extensionsRequired", []) if e in COMPRESSION]
    if bad:
        raise ValueError(tr("glTF: сжатие {v} не поддерживается — пересохраните файл без сжатия",
                            v=', '.join(bad)))
    other = [e for e in doc.get("extensionsUsed", []) if e not in COMPRESSION
             and e != "KHR_mesh_quantization"]
    if other:
        warn(tr("glTF: расширения не переносятся: {v}", v=', '.join(other)))
    for key, what in (("animations", tr("анимация")), ("skins", tr("скиннинг")),
                      ("cameras", tr("камеры"))):
        if doc.get(key):
            warn(tr("glTF: {what} не переносятся", what=what))
    scenes = doc.get("scenes", [])
    if scenes:
        si = doc.get("scene", 0)
        roots = scenes[si].get("nodes", [])
        if len(scenes) > 1:
            warn(tr("glTF: сцен {n_scenes} — взята только сцена {si}", n_scenes=len(scenes), si=si))
    else:
        kids = {c for n in doc.get("nodes", []) for c in n.get("children", [])}
        roots = [i for i in range(len(doc.get("nodes", []))) if i not in kids]
    r.scene.nodes = [r.node(i) for i in roots]
    if not r.scene.nodes and doc.get("meshes"):
        warn(tr("glTF: в сцене нет узлов — меши без узлов не переносятся"))
    return r.scene


# --- writer ----------------------------------------------------------------------------------

class _Bin:
    def __init__(self):
        self.data = bytearray()
        self.views, self.accessors = [], []

    def add(self, arr, code, gl_type, target=None, minmax=False):
        """Append an array as its own bufferView + accessor -> accessor index."""
        arr = np.ascontiguousarray(arr, dtype="<" + code)
        self.data += b"\0" * (-len(self.data) % 4)
        view = {"buffer": 0, "byteOffset": len(self.data), "byteLength": arr.nbytes}
        if target:
            view["target"] = target
        self.data += arr.tobytes()
        self.views.append(view)
        acc = {"bufferView": len(self.views) - 1,
               "componentType": {v: k for k, v in COMP.items()}[code],
               "count": len(arr), "type": gl_type}
        if minmax:
            acc["min"], acc["max"] = arr.min(0).tolist(), arr.max(0).tolist()
        self.accessors.append(acc)
        return len(self.accessors) - 1


def _is_trs(m):
    """glTF wants node matrices decomposable into T * R * S: no shear. Rotations typed with
    6 decimals (3MF from CAD) are off by ~1e-6 in the column cosines, real shear by far more."""
    a = m[:3, :3]
    ln = np.linalg.norm(a, axis=0)
    if np.any(ln == 0):
        return False
    c = a / ln
    return bool(np.abs(c.T @ c - np.eye(3)).max() < 1e-4)


class _Writer:
    def __init__(self, scene, scale, glb):
        self.scene, self.scale, self.glb = scene, scale, glb
        self.bin = _Bin()
        self.nodes, self.meshes, self.materials = [], [], []
        self.images, self.textures, self.samplers = [], [], []
        self.mesh_of, self.mat_of, self.tex_of = {}, {}, {}
        self.files = {}                 # file name beside a .gltf -> bytes
        self.names = textures.file_names(textures.used_textures(scene))
        self.losses = []

    def loss(self, msg):
        if msg not in self.losses:
            self.losses.append(msg)

    def texture(self, tex):
        """Texture -> glTF textures index, or None (with a loss line) when glTF cannot hold it."""
        if id(tex) not in self.tex_of:
            k = textures.kind(tex.data, tex.name)
            ti = None
            if k not in textures.WEB_KINDS:
                self.loss(tr("glTF: текстура «{name}» в формате {v} не записана — glTF допускает "
                             "только PNG и JPEG", name=tex.name, v=(k or tr('неизвестном')).upper()))
            else:
                img = {"name": os.path.splitext(self.names[id(tex)])[0],
                       "mimeType": textures.KINDS[k][1]}
                if self.glb:
                    self.bin.data += b"\0" * (-len(self.bin.data) % 4)
                    self.bin.views.append({"buffer": 0, "byteOffset": len(self.bin.data),
                                           "byteLength": len(tex.data)})
                    self.bin.data += tex.data
                    img["bufferView"] = len(self.bin.views) - 1
                else:
                    fname = self.names[id(tex)]
                    self.files[fname] = tex.data
                    img["uri"] = urllib.parse.quote(fname)
                if not self.samplers:      # repeat + mipmaps, as every DCC/engine defaults
                    self.samplers.append({"magFilter": 9729, "minFilter": 9987,
                                          "wrapS": 10497, "wrapT": 10497})
                self.images.append(img)
                self.textures.append({"sampler": 0, "source": len(self.images) - 1})
                ti = len(self.textures) - 1
            self.tex_of[id(tex)] = ti
        return self.tex_of[id(tex)]

    def material(self, mi, white, has_uv):
        key = (mi, white, has_uv)
        if key not in self.mat_of:
            m = self.scene.materials[mi]
            gm, pbr = {"name": m.name}, {"metallicFactor": 0.0, "roughnessFactor": 0.5}
            ti = None
            if m.texture is not None:
                if has_uv:
                    ti = self.texture(m.texture)
                else:
                    self.loss(tr("glTF: текстура у меша без UV не записана"))
            if ti is not None:
                pbr["baseColorTexture"] = {"index": ti}
            if m.color is not None or ti is not None:
                color = m.color or (1.0, 1.0, 1.0, 1.0)
                rgb = ((1.0, 1.0, 1.0) if white or ti is not None
                       else geom.srgb_to_linear(color[:3]).tolist())
                pbr["baseColorFactor"] = [*rgb, float(color[3])]
                if color[3] < 1:
                    gm["alphaMode"] = "BLEND"
            gm["pbrMetallicRoughness"] = pbr
            self.mat_of[key] = len(self.materials)
            self.materials.append(gm)
        return self.mat_of[key]

    def mesh(self, mi):
        if mi in self.mesh_of:
            return self.mesh_of[mi]
        m = self.scene.meshes[mi]
        tris, src = geom.triangulate(m)
        if not len(tris):
            self.loss(tr("glTF: пустые меши (без граней) отброшены"))
            self.mesh_of[mi] = None
            return None
        fm = m.face_material[src] if m.face_material is not None else np.full(len(tris), -1)
        vcol = m.colors is not None and not geom.colors_redundant(m, self.scene.materials)
        if vcol and np.any(fm >= 0):
            self.loss(tr("glTF: у мешей с цветами вершин цвет материала записан белым (glTF "
                         "перемножает их)"))
        nrm = None
        if m.normals is not None:
            nrm = m.normals @ C3.T
            ln = np.linalg.norm(nrm, axis=1, keepdims=True)
            if np.any(ln < 1e-12):
                self.loss(tr("glTF: нормали нулевой длины — нормали таких мешей отброшены"))
                nrm = None
            else:
                nrm = nrm / ln
        prims = []
        for g in dict.fromkeys(fm.tolist()):        # materials in order of first use
            cs = tris[fm == g].ravel()
            cols = [np.zeros((len(cs), 0))]
            if m.uv is not None:
                cols.append(m.uv[cs])
            if nrm is not None:
                cols.append(nrm[cs])
            if vcol:
                cols.append(m.colors[cs])
            first, inv = geom.split_vertices(m.corners[cs], np.concatenate(cols, 1))
            vc = cs[first]                          # one source corner per output vertex
            attrs = {"POSITION": self.bin.add(m.positions[m.corners[vc]] * self.scale @ C3.T,
                                              "f4", "VEC3", 34962, minmax=True)}
            if nrm is not None:
                attrs["NORMAL"] = self.bin.add(nrm[vc], "f4", "VEC3", 34962)
            if m.uv is not None:
                attrs["TEXCOORD_0"] = self.bin.add(np.c_[m.uv[vc, 0], 1.0 - m.uv[vc, 1]],
                                                   "f4", "VEC2", 34962)
            if vcol:
                c = np.clip(m.colors[vc], 0, 1)
                attrs["COLOR_0"] = self.bin.add(np.c_[geom.srgb_to_linear(c[:, :3]), c[:, 3]],
                                                "f4", "VEC4", 34962)
            code = "u2" if len(first) < 65535 else "u4"
            prim = {"attributes": attrs, "indices": self.bin.add(inv, code, "SCALAR", 34963),
                    "mode": 4}
            if g >= 0:
                prim["material"] = self.material(g, vcol, m.uv is not None)
            prims.append(prim)
        self.mesh_of[mi] = len(self.meshes)
        self.meshes.append({"name": m.name, "primitives": prims})
        return self.mesh_of[mi]

    def node(self, n):
        i = len(self.nodes)
        j = {"name": n.name}
        self.nodes.append(j)
        mtx = C4 @ n.matrix @ C4.T
        mtx[:3, 3] *= self.scale
        if not np.array_equal(mtx, np.eye(4)):
            if not _is_trs(mtx):
                self.loss(tr("glTF: трансформ со сдвигом (shear) записан матрицей — спецификация "
                             "такого не допускает, программы могут исказить"))
            j["matrix"] = mtx.T.ravel().tolist()        # column-major
        if n.mesh is not None:
            gm = self.mesh(n.mesh)
            if gm is not None:
                j["mesh"] = gm
        kids = [self.node(c) for c in n.children]
        if kids:
            j["children"] = kids
        return i


def write(scene, path, scale=1.0, unit_m=1.0):
    """scale: file units -> metres (glTF is always metres). .glb -> one binary file;
    .gltf -> JSON + <stem>.bin. Returns warning lines."""
    w = []
    if any(not m.is_triangles for m in scene.meshes):
        w.append(tr("glTF: многоугольники разбиты на треугольники"))
    glb = path.lower().endswith(".glb")
    wr = _Writer(scene, scale, glb)
    roots = [wr.node(n) for n in scene.nodes]
    w += wr.losses
    doc = {"asset": {"version": "2.0", "generator": "mesh2mesh"}, "scene": 0,
           "scenes": [{"nodes": roots}], "nodes": wr.nodes}
    for key, val in (("meshes", wr.meshes), ("materials", wr.materials),
                     ("textures", wr.textures), ("images", wr.images),
                     ("samplers", wr.samplers),
                     ("accessors", wr.bin.accessors), ("bufferViews", wr.bin.views)):
        if val:
            doc[key] = val
    data = bytes(wr.bin.data)
    folder = os.path.dirname(os.path.abspath(path))
    for fname, blob in wr.files.items():
        with open(os.path.join(folder, fname), "wb") as fh:
            fh.write(blob)
    if data:
        doc["buffers"] = [{"byteLength": len(data)}]
        if not glb:
            bin_path = os.path.splitext(path)[0] + ".bin"
            doc["buffers"][0]["uri"] = urllib.parse.quote(os.path.basename(bin_path))
            with open(bin_path, "wb") as fh:
                fh.write(data)
    js = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as fh:
        if glb:
            js += b" " * (-len(js) % 4)
            body = struct.pack("<II", len(js), CHUNK_JSON) + js
            if data:
                data += b"\0" * (-len(data) % 4)
                body += struct.pack("<II", len(data), CHUNK_BIN) + data
            fh.write(GLB_MAGIC + struct.pack("<II", 2, 12 + len(body)) + body)
        else:
            fh.write(js)
    return w
