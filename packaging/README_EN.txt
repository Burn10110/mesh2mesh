mesh2mesh — 3D model converter, version {version}
==================================================
Author: {url}

The program converts a 3D model from one format to another:
   FBX, OBJ, STL, 3MF, glTF / GLB, PLY — any to any.
It carries over everything the target format can hold: parts and their names, hierarchy,
materials and colours, textures, UV, normals, units. Whatever the target format cannot
hold, the program WARNS you about in advance, before saving. Nothing is lost silently.
No Blender, Python or other programs are needed.


INSTALLATION
------------
1. Unpack the archive into any folder (for example, C:\Programs\mesh2mesh).
   It cannot run straight from the archive — unpack it first.
2. Run mesh2mesh.exe. A desktop shortcut is handy:
   right-click mesh2mesh.exe → "Send to" → "Desktop (create shortcut)".
3. The program needs every file in the folder — do not delete or move them one by one.
   Move only the whole folder.

If Windows says "Windows protected your PC", click "More info" → "Run anyway".
The program has no digital signature, so Windows does not recognise it.


HOW TO USE
----------
1. Open a model: the "Open model" button, or drag a file into the window
   (or right onto the mesh2mesh.exe icon).
2. Check the SIZE. The large "Size: …" line is the model's size.
   OBJ and STL have no units — choose what the numbers mean (mm, cm, m…).
   If the model is 10, 100 or 1000 times off — the "÷ 10" / "× 10" buttons,
   or type an exact scale.
3. Check the AXES. If the model lies on its side — the "X 90°", "Y 90°", "Z 90°" buttons.
   "As in file" restores the original orientation.
4. Pick the FORMAT in the "Save" box. Right below it you see what this format
   cannot hold (for example, STL keeps no colours, glTF no TIFF textures).
5. "Save as GLB" (the button names the chosen format). After writing, the program reads
   the result back and compares it with the source; "Technical report" has the details.

Model view (buttons above the view):
   Clay, Metal — shape; Material — colours and textures; X-ray — see-through;
   Wireframe — edges; Smoothing — normals from the file or automatic.
Language (English / Russian) — the list at the bottom left of the window; it switches
at once and is remembered. The first time, the language follows Windows.
Mouse: left button — orbit; wheel — zoom;
       middle button or Ctrl + left — pan in the screen plane
       (up and down includes height, Z);
       the same with Shift — pan along the floor (X and Y).

Which format for what:
   3MF  — 3D printing slicers (Bambu, Prusa, Orca, Cura): parts, mm, colours.
   STL  — any slicer, but shape only: no colours, no units.
   GLB  — web, game engines, viewers: one file with everything inside.
   FBX  — Maya, 3ds Max, Blender, Unity, Unreal.
   OBJ  — almost any 3D program; texture images go next to the file.
   PLY  — scans, MeshLab: vertex colours.


IF SOMETHING IS WRONG
---------------------
- The window does not open or the 3D view is empty/black: update the graphics driver.
  The program needs OpenGL 2.0 or newer (any graphics card of the last 15 years has it;
  it may be missing over Remote Desktop or in a virtual machine).
  Converting works without the 3D view too.
- A file did not open or the result looks wrong: send the developer
  (contact: {url})
     1) the model file itself (with its texture images, if they are separate files),
     2) the log: %LOCALAPPDATA%\mesh2mesh\mesh2mesh.log
        (paste this line into the Explorer address bar and press Enter),
     3) the version number — it is in the window title,
     4) which program you opened the result in.


FOR ADVANCED USERS: COMMAND LINE
--------------------------------
mesh2mesh-cli.exe does the same without a window, for example:
   mesh2mesh-cli.exe model.fbx -o D:\out\model.glb
   mesh2mesh-cli.exe part.stl --in-units mm -o part.3mf --scale 10 --rotate x90
   mesh2mesh-cli.exe model.3mf --info          (only show what is inside)
All options: mesh2mesh-cli.exe --help   (language: --lang en or --lang ru)
Exit code 0 — all good, 1 — the result check failed, 2 — bad input.


LICENSE
-------
mesh2mesh is free software under the MIT license (LICENSE.txt).
Third-party components (Qt for Python — LGPL-3.0, NumPy, pyqtgraph, PyOpenGL, Python)
and their license texts: the licenses folder, list in licenses\THIRD-PARTY.txt.
