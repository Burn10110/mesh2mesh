"""Format registry: extension -> reader / writer.

Writers: write(scene, path, scale=file->output units, unit_m=metres per output unit, **opts)
-> list of warning lines (everything the format could not hold)."""

from . import fbx, gltf, obj, ply, stl, threemf

READERS = {".fbx": fbx.read, ".3mf": threemf.read, ".obj": obj.read, ".stl": stl.read, ".ply": ply.read,
           ".gltf": gltf.read, ".glb": gltf.read}
WRITERS = {".fbx": fbx.write, ".3mf": threemf.write, ".obj": obj.write, ".stl": stl.write, ".ply": ply.write,
           ".gltf": gltf.write, ".glb": gltf.write}
FLOAT32 = {".stl", ".ply", ".gltf", ".glb"}   # coordinates stored as float32 (read-back tolerance)
FIXED_UNITS = {".gltf": "m", ".glb": "m"}     # unit fixed by the spec: ext -> convert.UNITS key
TEXT_OPTION = {".stl", ".ply"}                # writers with binary=True|False
