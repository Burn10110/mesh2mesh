"""Texture helpers shared by the format modules: image type by magic bytes, finding a referenced
image file, unique file names for writing."""

import os
import re

from .model import Texture

# kind -> (extension, MIME)
KINDS = {"png": (".png", "image/png"), "jpeg": (".jpg", "image/jpeg"),
         "tga": (".tga", "image/x-tga"), "bmp": (".bmp", "image/bmp"),
         "dds": (".dds", "image/vnd-ms.dds"), "tiff": (".tif", "image/tiff"),
         "psd": (".psd", "image/vnd.adobe.photoshop"), "webp": (".webp", "image/webp"),
         "gif": (".gif", "image/gif")}
WEB_KINDS = ("png", "jpeg")      # what glTF and 3MF allow


def kind(data, name=""):
    """'png' | 'jpeg' | … | None: by content, TGA (no magic) by the file name."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:2] == b"BM":
        return "bmp"
    if data[:4] == b"DDS ":
        return "dds"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if data[:4] == b"8BPS":
        return "psd"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:4] == b"GIF8":
        return "gif"
    if name.lower().endswith(".tga"):
        return "tga"
    return None


def mime(tex):
    k = kind(tex.data, tex.name)
    return KINDS[k][1] if k else "application/octet-stream"


def base_name(ref):
    """File name part of a path written on any OS ('C:\\a\\b.png', '/home/x/b.png')."""
    return re.split(r"[\\/]", str(ref).strip())[-1]


def load_file(ref, folder, extra_dirs=()):
    """Find an image referenced by a model file -> Texture or None. Tries the reference as
    given (relative to `folder` or absolute), then its file name in `folder` and `extra_dirs`."""
    ref = str(ref).strip().strip('"')
    if not ref:
        return None
    name = base_name(ref)
    tries = [os.path.join(folder, ref.replace("\\", os.sep)), ref]
    tries += [os.path.join(d, name) for d in (folder, *extra_dirs)]
    for p in tries:
        try:
            if os.path.isfile(p):
                with open(p, "rb") as fh:
                    return Texture(name, fh.read())
        except OSError:
            continue
    return None


def file_names(textures, ascii_only=False):
    """Unique, file-system-safe names for textures (dict id(tex) -> name), the extension
    matching the content. ascii_only: only [A-Za-z0-9._-] (zip part names in 3MF)."""
    out, used = {}, set()
    bad = r"[^A-Za-z0-9._-]+" if ascii_only else r'[<>:"/\\|?*\x00-\x1f]+'
    for t in textures:
        if id(t) in out:
            continue
        stem, ext = os.path.splitext(base_name(t.name) or "texture")
        stem = re.sub(bad, "_", stem).strip(" .") or "texture"
        k = kind(t.data, t.name)
        ext = KINDS[k][0] if k else (ext or ".bin")
        n, i = stem + ext, 1
        while n.lower() in used:
            n, i = f"{stem}_{i}{ext}", i + 1
        used.add(n.lower())
        out[id(t)] = n
    return out


def used_textures(scene):
    """Textures of materials that faces use, each once, in material order."""
    used = {int(i) for m in scene.meshes if m.face_material is not None
            for i in set(m.face_material.tolist()) if i >= 0}
    seen, out = set(), []
    for i, mat in enumerate(scene.materials):
        t = mat.texture
        if t is not None and i in used and id(t) not in seen:
            seen.add(id(t))
            out.append(t)
    return out
