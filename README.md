# mesh2mesh

[English](#english) · [Русский](#русский)

---

## English

**mesh2mesh** converts 3D models between **FBX, OBJ, STL, 3MF, glTF / GLB and PLY**, in any direction.
It is a Windows app with a window and a command line, and it needs no installation.

It carries over everything the target format can hold: parts and their names, hierarchy, materials and
colors, textures, UVs, normals and units. Anything the target format cannot hold is listed **before you
save**, so nothing is lost silently. After writing, the result is read back and compared with the source.

### Download

Get `mesh2mesh-<version>.zip` from [Releases](https://github.com/Burn10110/mesh2mesh/releases),
unzip it anywhere and run `mesh2mesh.exe`. Neither Python nor Blender is needed.

The program is not code-signed. If Windows shows "Windows protected your PC", click **More info →
Run anyway**. Full instructions are in `README_EN.txt` (and `README_RU.txt`) inside the archive.

The interface is in English and Russian: it follows the Windows language at first, and the list at
the bottom left of the window switches it at once (remembered). Command line: `--lang en|ru`.

Requirements: Windows 10/11 x64. The 3D view needs OpenGL 2.0+, but conversion works without it.

### Using the window

1. Open a model with **Open model** or by dragging a file onto the window or onto `mesh2mesh.exe`.
2. Check the size. OBJ and STL carry no units, so pick what the numbers mean (mm, cm, m…). Use **÷ 10 /
   × 10** or an exact factor if the model is off by a power of ten.
3. Fix the orientation with the **X / Y / Z 90°** buttons if the model lies on its side.
4. Choose the output format. The panel below it lists what that format will not keep.
5. Save. The result is read back and checked; **Technical report** shows the details.

View modes: Clay, Metal, Material (colors and textures), X-ray, Wireframe, Smoothing (file normals or auto).
Mouse: left button orbits, the wheel zooms, the middle button (or Ctrl + left) pans in the screen plane,
and adding Shift pans along the floor.

### Command line

```
mesh2mesh-cli.exe model.fbx -o out/model.glb
mesh2mesh-cli.exe part.stl --in-units mm -o part.3mf --scale 10 --rotate x90
mesh2mesh-cli.exe model.3mf --info
```

| Option | Meaning |
|---|---|
| `-o, --out` | output file; the format comes from the extension |
| `--info` | only print the scene tree, size and warnings |
| `--in-units` | units of an OBJ/STL source (default `mm`) |
| `--units` | units written to OBJ/STL output (default: same as the source) |
| `--scale` | multiply the model (about the origin) |
| `--rotate` | quarter turns about world axes, left to right: `x90`, `x-90,z180` |
| `--ascii` | text STL/PLY instead of binary (STL then keeps object names) |

Exit code: 0 means OK, 1 means the read-back check failed, 2 means bad input.

### Which format for what

| Format | Typical use | Keeps |
|---|---|---|
| 3MF | slicers for 3D printing (Bambu, Prusa, Orca, Cura) | parts, units (mm), colors, textures |
| STL | any slicer | shape only: no colors, no units |
| GLB / glTF | web, game engines, viewers | everything in one file (GLB) |
| FBX | Maya, 3ds Max, Blender, Unity, Unreal | hierarchy, materials, textures, normals |
| OBJ | almost any 3D app | groups, materials; textures sit next to the file |
| PLY | scans, MeshLab | vertex colors |

### Running from source

The program is written in pure Python 3.14 with numpy; the window uses PySide6, pyqtgraph and PyOpenGL.
Blender and Maya were used only as references during testing and are not dependencies.

```
python -m venv .venv
.venv/Scripts/pip install numpy PySide6 pyqtgraph PyOpenGL pytest pyinstaller
.venv/Scripts/python -m mesh2mesh in.3mf -o out.glb    # CLI
.venv/Scripts/python -m mesh2mesh_gui                   # window
.venv/Scripts/python -m pytest tests -q                 # tests
.venv/Scripts/python packaging/build.py                 # exe -> dist/
```

### Feedback

If a file fails to open or the result looks wrong, send the model (with its texture images),
the log `%LOCALAPPDATA%\mesh2mesh\mesh2mesh.log` and the version number from the window title.
Author: https://burn10110.carrd.co/

---

## Русский

**mesh2mesh** переводит 3D-модели между форматами **FBX, OBJ, STL, 3MF, glTF / GLB и PLY** в любую
сторону. Это программа для Windows с окном и командной строкой, устанавливать её не нужно.

Переносит всё, что умеет целевой формат: детали и их имена, иерархию, материалы и цвета, текстуры, UV,
нормали, единицы. Чего целевой формат не вмещает, программа показывает **до сохранения**, так что молча
ничего не теряется. После записи результат читается обратно и сверяется с исходником.

### Скачать

Скачайте `mesh2mesh-<версия>.zip` в [Releases](https://github.com/Burn10110/mesh2mesh/releases),
распакуйте в любую папку и запустите `mesh2mesh.exe`. Ни Python, ни Blender не нужны.

Программа не подписана цифровой подписью. Если Windows пишет «Система Windows защитила ваш компьютер»,
нажмите **Подробнее → Выполнить в любом случае**. Подробная инструкция лежит в `README_RU.txt` (и `README_EN.txt`) внутри архива.

Интерфейс на русском и английском: сначала — по языку Windows, переключается списком слева внизу окна
сразу и запоминается. В командной строке — `--lang ru|en`.

Требования: Windows 10/11 x64. Для 3D-вида нужен OpenGL 2.0+, конвертировать можно и без него.

### Работа в окне

1. Откройте модель кнопкой **Открыть модель** или перетащите файл в окно или на `mesh2mesh.exe`.
2. Проверьте размер. В OBJ и STL единиц нет, поэтому выберите, в чём записаны числа (мм, см, м…). Если
   модель в 10, 100 или 1000 раз не того размера, помогут **÷ 10 / × 10** или точный масштаб числом.
3. Если модель лежит на боку, поверните её кнопками **X / Y / Z 90°**.
4. Выберите формат. Под ним сразу видно, что он не вместит.
5. Сохраните. Результат читается обратно и проверяется, подробности есть в **Техническом отчёте**.

Режимы вида: Глина, Металл, Материал (цвета и текстуры), Рентген, Каркас, Сглаживание (нормали из файла
или автоматически). Мышь: левая кнопка вращает, колесо приближает, средняя кнопка (или Ctrl + левая)
сдвигает в плоскости экрана, а с Shift сдвигает по полу.

### Командная строка

```
mesh2mesh-cli.exe модель.fbx -o выход/модель.glb
mesh2mesh-cli.exe деталь.stl --in-units mm -o деталь.3mf --scale 10 --rotate x90
mesh2mesh-cli.exe модель.3mf --info
```

| Параметр | Что делает |
|---|---|
| `-o, --out` | выходной файл, формат определяется по расширению |
| `--info` | только показать дерево, габарит и предупреждения |
| `--in-units` | единицы исходника OBJ/STL (по умолчанию `mm`) |
| `--units` | единицы при записи OBJ/STL (по умолчанию как у исходника) |
| `--scale` | умножить модель (от начала координат) |
| `--rotate` | четверть-обороты вокруг мировых осей слева направо: `x90`, `x-90,z180` |
| `--ascii` | STL/PLY текстом вместо binary (STL тогда сохраняет имена объектов) |

Код выхода: 0 — всё хорошо, 1 — проверка результата не прошла, 2 — ошибка входа.

### Какой формат куда

| Формат | Для чего | Что хранит |
|---|---|---|
| 3MF | слайсеры для 3D-печати (Bambu, Prusa, Orca, Cura) | детали, единицы (мм), цвета, текстуры |
| STL | любой слайсер | только форму: без цветов и единиц |
| GLB / glTF | веб, движки, просмотрщики | всё в одном файле (GLB) |
| FBX | Maya, 3ds Max, Blender, Unity, Unreal | иерархию, материалы, текстуры, нормали |
| OBJ | почти любая 3D-программа | группы, материалы; картинки текстур лежат рядом с файлом |
| PLY | сканы, MeshLab | цвета вершин |

### Запуск из исходников

Программа написана на чистом Python 3.14 с numpy, окно использует PySide6, pyqtgraph и PyOpenGL.
Blender и Maya служили только эталонами при проверке и в зависимости не входят.

```
python -m venv .venv
.venv/Scripts/pip install numpy PySide6 pyqtgraph PyOpenGL pytest pyinstaller
.venv/Scripts/python -m mesh2mesh in.3mf -o out.glb    # командная строка
.venv/Scripts/python -m mesh2mesh_gui                   # окно
.venv/Scripts/python -m pytest tests -q                 # тесты
.venv/Scripts/python packaging/build.py                 # exe -> dist/
```

### Обратная связь

Если файл не открылся или результат выглядит не так, пришлите модель (вместе с картинками текстур),
журнал `%LOCALAPPDATA%\mesh2mesh\mesh2mesh.log` и номер версии из заголовка окна.
Автор: https://burn10110.carrd.co/
