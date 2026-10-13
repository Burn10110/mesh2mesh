"""English text: Russian source text (the key used in tr()/N_()) -> English.

Placeholders {…} must match the key exactly (tests/test_i18n.py checks this, and that every
tr()/N_() literal in the code has an entry and no entry is dead)."""

EN = {
    # mesh2mesh/__main__.py (CLI)
    "Конвертирует mesh-форматы, перечисляет всё, что целевой формат не может хранить, затем "
    "читает результат своим ридером и сверяет треугольники, габарит, площадь и объём. Код "
    "выхода 1 — проверка чтением не прошла, 2 — неверный ввод.":
        "Converts between mesh formats, lists everything the target format cannot hold, then "
        "reads the result back with its own reader and compares triangles, bounding box, area "
        "and volume. Exit code 1 if that read-back check fails, 2 on bad input.",
    "{prog} {version} · автор: {url}": "{prog} {version} · author: {url}",
    "входной файл: {exts}": "input file: {exts}",
    "выходной файл: {exts}": "output file: {exts}",
    "только показать содержимое файла": "only show what the file contains",
    "единицы координат в OBJ/STL на выходе (по умолчанию — как у исходника)":
        "coordinate units of the output OBJ/STL (default: same as the source)",
    "единицы исходника без единиц в формате (OBJ/STL), по умолчанию mm":
        "units of a source whose format has none (OBJ/STL), default mm",
    "умножить модель (от начала координат)": "scale the model (about the origin)",
    "четверть-обороты вокруг мировых осей слева направо: x90 или x-90,z180":
        "quarter turns about world axes, left to right: x90 or x-90,z180",
    "STL/PLY текстом (STL так сохраняет имена объектов), иначе binary":
        "STL/PLY as text (this way STL keeps object names), otherwise binary",
    "язык сообщений: ru, en (по умолчанию — язык Windows или M2M_LANG)":
        "message language: ru, en (default: Windows language or M2M_LANG)",
    "--scale должен быть положительным": "--scale must be positive",
    "нет файла {path}": "no such file: {path}",
    "укажите -o выходной файл или --info": "give -o output file or --info",

    # mesh2mesh/convert.py
    "{v:g} м": "{v:g} m",
    "единицы: {unit}; мешей {meshes}, экземпляров {inst}, глубина иерархии {depth}, "
    "материалов {mats}":
        "units: {unit}; meshes {meshes}, instances {inst}, hierarchy depth {depth}, "
        "materials {mats}",
    "формат {ext} не поддерживается для чтения; есть: {known}":
        "format {ext} cannot be read; supported: {known}",
    "формат {ext} не поддерживается для записи; есть: {known}":
        "format {ext} cannot be written; supported: {known}",
    "масштаб должен быть положительным": "scale must be positive",
    "формат {ext} не поддерживается для записи": "format {ext} cannot be written",
    "проверка чтением: OK": "read-back check: OK",
    "проверка чтением: РАСХОЖДЕНИЕ": "read-back check: MISMATCH",
    "в файле нет единиц — принято: {u}": "the file has no units — assumed: {u}",
    "{fmt}: единицы по спецификации всегда {fixed} — --units {units} не применён":
        "{fmt}: units are always {fixed} by the spec — --units {units} not applied",
    "(без расширения)": "(no extension)",

    # mesh2mesh/geom.py
    "объектов {objects}, треугольников {tris}": "objects {objects}, triangles {tris}",
    "габарит ({unit}): [{lo}] .. [{hi}], размер [{size}]":
        "bounding box ({unit}): [{lo}] .. [{hi}], size [{size}]",
    "треугольников {a} -> {b}": "triangles {a} -> {b}",
    "площадь {a:.6g} -> {b:.6g} м²": "area {a:.6g} -> {b:.6g} m²",
    "объём {a:.6g} -> {b:.6g} м³ (знак = ориентация граней)":
        "volume {a:.6g} -> {b:.6g} m³ (sign = face orientation)",
    "габарит {nm} {a} -> {b}": "bounding box {nm} {a} -> {b}",

    # mesh2mesh/formats/fbx.py
    "FBX: не записать значение типа {v}": "FBX: cannot write a value of type {v}",
    "FBX: неизвестный тип значения {t!r} по смещению {v}":
        "FBX: unknown value type {t!r} at offset {v}",
    "FBX: не FBX-файл (нет заголовка)": "FBX: not an FBX file (no header)",
    "FBX: файл обрезан": "FBX: file is truncated",
    "FBX (ASCII): не разобрать текст у смещения {pos}: {v!r}":
        "FBX (ASCII): cannot parse text at offset {pos}: {v!r}",
    "FBX (ASCII): не хватает '}'": "FBX (ASCII): missing '}'",
    "FBX (ASCII): лишняя '}'": "FBX (ASCII): unexpected '}'",
    "FBX (ASCII): ожидалось имя узла, а не {val!r}": "FBX (ASCII): expected a node name, not {val!r}",
    "FBX {v:.1f}: поддерживаются FBX 7.x (2011+) — пересохраните файл в новой версии":
        "FBX {v:.1f}: only FBX 7.x (2011+) is supported — re-save the file in a newer version",
    "FBX: нет раздела Objects": "FBX: no Objects section",
    "FBX: скиннинг и blend shapes не переносятся (взята исходная форма)":
        "FBX: skinning and blend shapes are not transferred (rest shape taken)",
    "FBX: анимация не переносится (взята поза из файла)":
        "FBX: animation is not transferred (the pose stored in the file is taken)",
    "FBX: циклическая иерархия объектов": "FBX: cyclic object hierarchy",
    "FBX: «{name}»: наследование масштаба InheritType {it} заменено обычным (RSrs) — форма под "
    "неравномерным масштабом родителя может отличаться":
        "FBX: “{name}”: scale inheritance InheritType {it} replaced by the usual one (RSrs) — the "
        "shape under a non-uniform parent scale may differ",
    "FBX: не меши не переносятся: {v}": "FBX: non-mesh geometry is not transferred: {v}",
    "FBX: {v} не переносятся (оставлены пустыми узлами)":
        "FBX: {v} are not transferred (kept as empty nodes)",
    "камеры": "cameras",
    "источники света": "lights",
    "FBX: «{name}»: у объекта несколько геометрий — взята первая":
        "FBX: “{name}”: the object has several geometries — the first one taken",
    "FBX: переносится только текстура цвета (карты нормалей, блеска, прозрачности и прочие — нет)":
        "FBX: only the color texture is transferred (normal, specular, opacity and other maps "
        "are not)",
    "FBX: «{name}»: многослойная текстура не переносится":
        "FBX: “{name}”: layered texture is not transferred",
    "FBX: текстура «{name}» не вшита и не найдена рядом с файлом — отброшена":
        "FBX: texture “{name}” is not embedded and not found next to the file — dropped",
    "FBX: сдвиг/поворот/масштаб UV текстуры «{name}» не переносится":
        "FBX: UV offset/rotation/scale of texture “{name}” is not transferred",
    "FBX: «{name}»: слой {name2} повреждён — отброшен":
        "FBX: “{name}”: layer {name2} is damaged — dropped",
    "FBX: «{name}»: {name2} с привязкой {mapping} не переносится":
        "FBX: “{name}”: {name2} with mapping {mapping} is not transferred",
    "FBX: «{name}»: слой {name2} короче меша — отброшен":
        "FBX: “{name}”: layer {name2} is shorter than the mesh — dropped",
    "FBX: «{name}»: слой {name2} ссылается за пределы — отброшен":
        "FBX: “{name}”: layer {name2} points out of range — dropped",
    "FBX: «{name}»: число координат вершин не кратно 3":
        "FBX: “{name}”: vertex coordinate count is not a multiple of 3",
    "FBX: «{name}»: индекс вершины вне диапазона": "FBX: “{name}”: vertex index out of range",
    "нормалей": "normal layers",
    "UV-каналов": "UV channels",
    "наборов цветов вершин": "vertex color sets",
    "FBX: «{name}»: {n_elems} {what} — переносится только первый":
        "FBX: “{name}”: {n_elems} {what} — only the first one is transferred",
    "FBX: слои не переносятся: {v}": "FBX: layers not transferred: {v}",
    "FBX: «{name}»: группы сглаживания без нормалей не переносятся":
        "FBX: “{name}”: smoothing groups without normals are not transferred",
    "FBX: «{name}»: материалы с привязкой {mapping} — взят первый на весь меш":
        "FBX: “{name}”: materials with mapping {mapping} — the first one used for the whole mesh",
    "FBX: «{name}»: рёбра и точки без граней отброшены":
        "FBX: “{name}”: edges and points without faces dropped",
    "FBX 7400: файл больше 4 ГБ не записать": "FBX 7400: cannot write a file over 4 GB",
    "FBX: трансформ со сдвигом (shear) перенесён в точки геометрии и в дочерние узлы":
        "FBX: transform with shear moved into geometry points and child nodes",
    "FBX: пустые меши (без граней) записаны пустыми узлами":
        "FBX: empty meshes (no faces) written as empty nodes",
    "FBX: неизвестное сжатие массива ({enc})": "FBX: unknown array compression ({enc})",

    # mesh2mesh/formats/gltf.py
    "GLB: версия контейнера {version}, поддерживается 2":
        "GLB: container version {version}, only 2 is supported",
    "GLB: нет JSON-блока": "GLB: no JSON chunk",
    "glTF: не JSON ({e})": "glTF: not JSON ({e})",
    "glTF: поддерживается только версия 2.0": "glTF: only version 2.0 is supported",
    "glTF: буфер без uri, но нет BIN-блока GLB": "glTF: buffer without uri, but no GLB BIN chunk",
    "glTF: нет файла буфера {uri}": "glTF: buffer file not found: {uri}",
    "glTF: accessor выходит за границы буфера": "glTF: accessor goes past the end of its buffer",
    "glTF: нет файла текстуры {uri} — текстура отброшена":
        "glTF: texture file not found: {uri} — texture dropped",
    "glTF: текстуры в форматах расширений (WebP, KTX2) не переносятся":
        "glTF: textures in extension formats (WebP, KTX2) are not transferred",
    "glTF: текстура на втором UV-канале — показана по первому":
        "glTF: texture uses the second UV channel — mapped with the first",
    "glTF: сдвиг/масштаб UV текстуры (KHR_texture_transform) не переносится":
        "glTF: texture UV offset/scale (KHR_texture_transform) is not transferred",
    "glTF: «{name}»: оттенок baseColorFactor поверх текстуры не переносится":
        "glTF: “{name}”: baseColorFactor tint over the texture is not transferred",
    "glTF: переносится только текстура цвета (карты нормалей, металла/шероховатости, свечения, "
    "затенения — нет)":
        "glTF: only the color texture is transferred (normal, metallic/roughness, emissive and "
        "occlusion maps are not)",
    "glTF: из материалов переносится только базовый цвет и прозрачность":
        "glTF: only base color and opacity of materials are transferred",
    "glTF: точки и линии отброшены": "glTF: points and lines dropped",
    "glTF: морф-таргеты (blend shapes) не переносятся":
        "glTF: morph targets (blend shapes) are not transferred",
    "glTF: веса скиннинга не переносятся": "glTF: skin weights are not transferred",
    "glTF: переносится только первый UV-канал": "glTF: only the first UV channel is transferred",
    "glTF: меш «{name}»: индекс вершины вне диапазона":
        "glTF: mesh “{name}”: vertex index out of range",
    "glTF: «{name}»: {what} есть не у всех примитивов — отброшены":
        "glTF: “{name}”: not all primitives have {what} — dropped",
    "glTF: «{name}»: {what} есть не у всех примитивов — у остальных {fill}":
        "glTF: “{name}”: not all primitives have {what} — the rest get {fill}",
    "нормали": "normals",
    "цвета вершин": "vertex colors",
    "glTF: циклическая иерархия узлов": "glTF: cyclic node hierarchy",
    "скиннинг": "skinning",
    "веса морфов": "morph weights",
    "анимация": "animation",
    "glTF: {what} не переносятся": "glTF: {what} not transferred",
    "glTF: расширения узлов ({v}) не переносятся": "glTF: node extensions ({v}) are not transferred",
    "glTF: сжатие {v} не поддерживается — пересохраните файл без сжатия":
        "glTF: compression {v} is not supported — re-save the file without compression",
    "glTF: расширения не переносятся: {v}": "glTF: extensions not transferred: {v}",
    "glTF: сцен {n_scenes} — взята только сцена {si}":
        "glTF: {n_scenes} scenes — only scene {si} taken",
    "glTF: в сцене нет узлов — меши без узлов не переносятся":
        "glTF: the scene has no nodes — meshes without nodes are not transferred",
    "glTF: текстура «{name}» в формате {v} не записана — glTF допускает только PNG и JPEG":
        "glTF: texture “{name}” in {v} format not written — glTF allows only PNG and JPEG",
    "неизвестном": "unknown",
    "glTF: текстура у меша без UV не записана": "glTF: texture of a mesh without UV not written",
    "glTF: пустые меши (без граней) отброшены": "glTF: empty meshes (no faces) dropped",
    "glTF: у мешей с цветами вершин цвет материала записан белым (glTF перемножает их)":
        "glTF: meshes with vertex colors get a white material color (glTF multiplies them)",
    "glTF: нормали нулевой длины — нормали таких мешей отброшены":
        "glTF: zero-length normals — normals of those meshes dropped",
    "glTF: трансформ со сдвигом (shear) записан матрицей — спецификация такого не допускает, "
    "программы могут исказить":
        "glTF: transform with shear written as a matrix — the spec does not allow this, "
        "programs may distort it",
    "glTF: многоугольники разбиты на треугольники": "glTF: polygons split into triangles",

    # mesh2mesh/formats/obj.py
    "OBJ: библиотека материалов не найдена: {v}": "OBJ: material library not found: {v}",
    "OBJ: сдвиг/масштаб UV текстуры «{ref}» не переносится":
        "OBJ: UV offset/scale of texture “{ref}” is not transferred",
    "OBJ: нет файла текстуры {ref} — текстура отброшена":
        "OBJ: texture file not found: {ref} — texture dropped",
    "OBJ: переносится только текстура цвета map_Kd (карты нормалей, блеска, прозрачности — нет)":
        "OBJ: only the color texture map_Kd is transferred (normal, specular, opacity maps "
        "are not)",
    "OBJ: материал «{name}» не описан в .mtl — без цвета":
        "OBJ: material “{name}” is not defined in .mtl — no color",
    "OBJ: линии, точки и кривые/поверхности отброшены":
        "OBJ: lines, points and curves/surfaces dropped",
    "OBJ: неизвестная директива «{k}» пропущена": "OBJ: unknown directive “{k}” skipped",
    "OBJ: строка {line_no}: индекс {tok} вне диапазона":
        "OBJ: line {line_no}: index {tok} out of range",
    "OBJ: «{name}»: UV есть не у всех граней — недостающие = (0, 0)":
        "OBJ: “{name}”: not all faces have UV — missing ones = (0, 0)",
    "OBJ: «{name}»: нормали есть не у всех граней — нормали отброшены":
        "OBJ: “{name}”: not all faces have normals — normals dropped",
    "OBJ: иерархия и инстансы запечены в вершины (объекты плоским списком)":
        "OBJ: hierarchy and instances baked into vertices (objects as a flat list)",
    "OBJ: пустые меши (без граней) отброшены": "OBJ: empty meshes (no faces) dropped",
    "OBJ: цвета на углах граней (разные у одной вершины) отброшены":
        "OBJ: face-corner colors (different at one vertex) dropped",
    "OBJ: цвета вершин записаны неофициальным «v x y z r g b» (Blender, MeshLab читают)":
        "OBJ: vertex colors written as unofficial “v x y z r g b” (Blender, MeshLab read it)",
    "OBJ: прозрачность цветов вершин отброшена": "OBJ: vertex color alpha dropped",
    "OBJ: у части объектов с текстурой нет UV — текстура на них не ляжет":
        "OBJ: some textured objects have no UV — the texture will not map onto them",

    # mesh2mesh/formats/ply.py
    "PLY: нет заголовка ply … end_header": "PLY: no ply … end_header header",
    "PLY: непонятное свойство: {v}": "PLY: unrecognized property: {v}",
    "PLY: неизвестный формат {fmt!r}": "PLY: unknown format {fmt!r}",
    "PLY: отрицательная длина списка {n} ({k})": "PLY: negative list length {n} ({k})",
    "PLY: данные элемента «{name}» повреждены или обрезаны ({e})":
        "PLY: data of element “{name}” is damaged or truncated ({e})",
    "PLY: элемент «{name}» ({count}) не переносится":
        "PLY: element “{name}” ({count}) is not transferred",
    "PLY: нет вершин с x y z": "PLY: no vertices with x y z",
    "PLY: свойства вершин не переносятся: {v}": "PLY: vertex properties not transferred: {v}",
    "PLY: {key} граней — не список": "PLY: face {key} is not a list",
    "PLY: нет граней — облако точек не переносится":
        "PLY: no faces — point clouds are not transferred",
    "PLY: {v} граней с < 3 вершинами отброшены": "PLY: {v} faces with < 3 vertices dropped",
    "PLY: индекс вершины вне диапазона": "PLY: vertex index out of range",
    "PLY: есть цвета и у вершин, и у граней — взяты цвета вершин":
        "PLY: both vertices and faces have colors — vertex colors taken",
    "PLY: texcoord граней не совпадает с числом вершин — UV отброшены":
        "PLY: face texcoord does not match the vertex count — UV dropped",
    "PLY: свойства граней не переносятся: {v}": "PLY: face properties not transferred: {v}",
    "PLY: объекты ({n_baked}) слиты в один, имена потеряны":
        "PLY: objects ({n_baked}) merged into one, names lost",
    "PLY: иерархия и инстансы запечены в вершины": "PLY: hierarchy and instances baked into vertices",
    "PLY: пустые меши (без граней) отброшены": "PLY: empty meshes (no faces) dropped",
    "PLY: UV есть не у всех объектов — отброшены": "PLY: not all objects have UV — dropped",
    "PLY: нормали есть не у всех объектов — отброшены": "PLY: not all objects have normals — dropped",
    "PLY: материалов нет в формате — их цвета записаны цветами вершин":
        "PLY: the format has no materials — their colors written as vertex colors",
    "PLY: текстуры отброшены (формат их не хранит)":
        "PLY: textures dropped (the format does not store them)",
    "PLY: где есть и цвета вершин, и материал — взяты цвета вершин":
        "PLY: where there are both vertex colors and a material — vertex colors taken",
    "PLY: цвета округлены до 8 бит на канал": "PLY: colors rounded to 8 bits per channel",

    # mesh2mesh/formats/stl.py
    "STL: файл слишком короткий": "STL: file is too short",
    "STL: заявлено {n} треугольников, а данных меньше":
        "STL: {n} triangles declared, but there is less data",
    "STL: цвета в attribute-байтах (VisCAM/Materialise) не поддерживаются — отброшены":
        "STL: colors in attribute bytes (VisCAM/Materialise) are not supported — dropped",
    "STL: не найдено ни одного solid … endsolid": "STL: no solid … endsolid found",
    "STL: многоугольники разбиты на треугольники": "STL: polygons split into triangles",
    "STL: материалы и цвета отброшены": "STL: materials and colors dropped",
    "STL: текстуры отброшены (формат их не хранит)":
        "STL: textures dropped (the format does not store them)",
    "STL: цвета вершин отброшены": "STL: vertex colors dropped",
    "STL: UV отброшены": "STL: UV dropped",
    "STL: нормали заменены нормалями граней": "STL: normals replaced by face normals",
    "STL: иерархия и инстансы запечены в вершины": "STL: hierarchy and instances baked into vertices",
    "STL: пустые меши (без граней) отброшены": "STL: empty meshes (no faces) dropped",
    "STL (binary): объекты ({n}) слиты в одно тело, имена потеряны (--stl-ascii сохранит их "
    "раздельно)":
        "STL (binary): objects ({n}) merged into one body, names lost (--stl-ascii keeps them "
        "separate)",
    "STL: {collapsed} треуг. во float32 имеют совпадающие вершины — Blender/слайсеры их выбросят "
    "(записаны как есть)":
        "STL: {collapsed} triangles have coinciding vertices in float32 — Blender/slicers will "
        "drop them (written as is)",

    # mesh2mesh/formats/threemf.py
    "смешанные материалы": "composite materials",
    "составные свойства": "multiproperties",
    "3MF: transform должен содержать 12 чисел: {s!r}": "3MF: transform must have 12 numbers: {s!r}",
    "3MF: неверный цвет {s!r}": "3MF: invalid color {s!r}",
    "3MF: цвет вершины — не число (NaN)": "3MF: vertex color is not a number (NaN)",
    "3MF: не найдена основная модель (3D/3dmodel.model)":
        "3MF: main model not found (3D/3dmodel.model)",
    "3MF: неверный индекс свойства в {a} у треугольника":
        "3MF: invalid property index in {a} of a triangle",
    "3MF: нет части {path}": "3MF: no part {path}",
    "3MF: режим края текстуры (mirror/clamp) не переносится — будет повтор":
        "3MF: texture edge mode (mirror/clamp) is not transferred — it will repeat",
    "3MF: {v} не поддерживаются — отброшены": "3MF: {v} are not supported — dropped",
    "3MF: ссылка на несуществующую группу свойств pid={pid}":
        "3MF: reference to a missing property group pid={pid}",
    "3MF: индекс свойства {index} вне группы pid={pid}":
        "3MF: property index {index} is outside group pid={pid}",
    "3MF: ссылка на несуществующую текстуру texid={texid}":
        "3MF: reference to a missing texture texid={texid}",
    "3MF: нет файла текстуры {ref} — текстура отброшена":
        "3MF: texture file not found: {ref} — texture dropped",
    "3MF: текстура {ref} не PNG/JPEG (нарушение формата) — перенесена как есть":
        "3MF: texture {ref} is not PNG/JPEG (violates the format) — transferred as is",
    "3MF: слишком глубокая (циклическая?) вложенность компонентов":
        "3MF: components nested too deep (cyclic?)",
    "3MF: объект {oid} не найден в {path}": "3MF: object {oid} not found in {path}",
    "3MF: тип объекта «{v}» перенесён как обычная деталь":
        "3MF: object type “{v}” transferred as a regular part",
    "3MF: объект {oid}: индекс вершины вне диапазона": "3MF: object {oid}: vertex index out of range",
    "3MF: индекс UV {v} вне группы pid={v2}": "3MF: UV index {v} is outside group pid={v2}",
    "3MF: неизвестная единица {unit!r}": "3MF: unknown unit {unit!r}",
    "3MF: метаданные и миниатюра не переносятся": "3MF: metadata and thumbnail are not transferred",
    "3MF: в <build> нет ни одной детали": "3MF: <build> has no items",
    "3MF: текстура «{name}» в формате {v} не записана — 3MF допускает только PNG и JPEG":
        "3MF: texture “{name}” in {v} format not written — 3MF allows only PNG and JPEG",
    "3MF: текстура у меша без UV не записана": "3MF: texture of a mesh without UV not written",
    "3MF: у граней с текстурой имя и цвет материала не хранятся (в 3MF это просто картинка)":
        "3MF: textured faces do not keep material name and color (in 3MF it is just an image)",
    "3MF: UV граней без текстуры отброшены (3MF хранит UV только вместе с текстурой)":
        "3MF: UV of untextured faces dropped (3MF stores UV only together with a texture)",
    "3MF: цвета вершин на гранях с текстурой отброшены":
        "3MF: vertex colors on textured faces dropped",
    "3MF: свойства есть не у всех треугольников": "3MF: not all triangles have properties",
    "3MF: пустые меши (без граней) отброшены": "3MF: empty meshes (no faces) dropped",
    "3MF: экземпляры одного меша носят имя первого (у инстансов в 3MF нет своих имён)":
        "3MF: instances of one mesh carry the first one's name (3MF instances have no names "
        "of their own)",
    "3MF: пустые узлы (без меша и детей) отброшены":
        "3MF: empty nodes (no mesh and no children) dropped",
    "3MF: многоугольники разбиты на треугольники": "3MF: polygons split into triangles",
    "3MF: нормали отброшены (формат их не хранит)":
        "3MF: normals dropped (the format does not store them)",
    "3MF: единица {unit_m} м не выражается в 3MF": "3MF: unit {unit_m} m cannot be expressed in 3MF",
    "3MF: есть зеркальные трансформы — иерархия и инстансы запечены в вершины":
        "3MF: there are mirrored transforms — hierarchy and instances baked into vertices",
    "3MF: нечего записать — в сцене нет граней": "3MF: nothing to write — the scene has no faces",

    # mesh2mesh_gui/__main__.py
    "\n\nПодробности записаны в журнал:\n{log}": "\n\nDetails were written to the log:\n{log}",
    "Ошибка программы": "Program error",
    "Что-то пошло не так: {name}: {value}{where}": "Something went wrong: {name}: {value}{where}",

    # mesh2mesh_gui/display.py
    "Из файла": "From file",
    "Нормали из файла: гладко там, где автор сгладил, острые рёбра остаются. Меши без нормалей "
    "(STL, 3MF…) — плоско":
        "Normals from the file: smooth where the author smoothed, sharp edges stay sharp. Meshes "
        "without normals (STL, 3MF…) look flat",
    "Авто {angle:g}°": "Auto {angle:g}°",
    "Нормали пересчитаны по углу между гранями (до {angle:g}° — гладко), нормали файла не "
    "используются":
        "Normals recomputed from the angle between faces (up to {angle:g}° is smooth); the "
        "file's normals are not used",
    "Плоско": "Flat",
    "Каждая грань плоская — видны все фасеты": "Every face flat — all facets visible",

    # mesh2mesh_gui/report.py
    "мкм": "µm",
    "мм": "mm",
    "см": "cm",
    "м": "m",
    "дюйм": "in",
    "фут": "ft",
    "3MF — 3D-печать, слайсеры": "3MF — 3D printing, slicers",
    "STL — 3D-печать, CAD": "STL — 3D printing, CAD",
    "OBJ — универсальный": "OBJ — universal",
    "FBX — Maya, 3ds Max, движки": "FBX — Maya, 3ds Max, game engines",
    "GLB — веб, движки (один файл)": "GLB — web, engines (single file)",
    "glTF — веб, движки (.gltf + .bin)": "glTF — web, engines (.gltf + .bin)",
    "PLY — сканы, MeshLab": "PLY — scans, MeshLab",
    "Габарит: {size} {unit}": "Size: {size} {unit}",
    "Без нормалей в файле: {n} из {total} — они плоские в «Из файла».":
        "No normals in the file: {n} of {total} — they look flat in “From file”.",
    "В файле нет геометрии": "The file has no geometry",
    "В этом файле нормали есть у всех мешей.": "In this file every mesh has normals.",
    "В этом файле нормалей нет — «Из файла» показывает плоско.":
        "This file has no normals — “From file” shows it flat.",
    "Модели ({patterns})": "Models ({patterns})",
    "{fmt}: единицы из файла — {unit}.": "{fmt}: units from the file — {unit}.",
    "{fmt}: единиц в файле нет — выберите, в чём записаны числа.":
        "{fmt}: the file has no units — choose what its numbers mean.",
    "{textures} — видны в режиме «Материал».": "{textures} — visible in “Material” mode.",
    " и ещё {n}": " and {n} more",
    "Не показаны в 3D-виде: {shown}{more} — там цвет материала; в сохранённый файл переносятся.":
        "Not shown in the 3D view: {shown}{more} — the material colour is shown instead; they "
        "are still carried into the saved file.",
    "У {tris} есть текстура, но нет UV — показаны цветом материала.":
        "{tris} have a texture but no UV — shown in the material colour.",
    "При чтении исходника: {w}": "While reading the source: {w}",
    "меша|мешей|мешей": "mesh|meshes",
    "{name} вместит всё, что прочитано из файла.": "{name} will hold everything read from the file.",
    "{name} не вместит — будет отброшено или изменено:":
        "{name} cannot hold this — it will be dropped or changed:",
    "Сохранено: {name}. Проверка чтением пройдена — геометрия совпадает с исходной.":
        "Saved: {name}. Read-back check passed — the geometry matches the source.",
    "Файл {name} записан, но проверка чтением не прошла: записанное не совпадает с исходной "
    "моделью. Не используйте его — пришлите исходный файл разработчику.":
        "File {name} was written, but the read-back check failed: what was written does not "
        "match the source model. Do not use it — please send the source file to the developer.",
    "Расхождение: {m}": "Mismatch: {m}",
    "Не вошло в файл ({n}):": "Not carried into the file ({n}):",
    "Всё прочитанное перенесено без потерь.": "Everything read was carried over without loss.",
    "меш|меша|мешей": "mesh|meshes",
    "экземпляр|экземпляра|экземпляров": "instance|instances",
    "треугольник|треугольника|треугольников": "triangle|triangles",
    "материал|материала|материалов": "material|materials",
    "текстура|текстуры|текстур": "texture|textures",
    "треугольника|треугольников|треугольников": "triangle|triangles",

    # mesh2mesh_gui/shading.py
    "Глина": "Clay",
    "Матовая глина (matcap): форма без цветов — видны вмятины и рёбра":
        "Matte clay (matcap): shape without colours — dents and edges show",
    "Металл": "Metal",
    "Красный металл (matcap): блики подчёркивают кривизну и гладкость":
        "Red metal (matcap): highlights bring out curvature and smoothness",
    "Материал": "Material",
    "Цвета материалов и цвета вершин (vertex paint) из файла":
        "Material colours and vertex colours (vertex paint) from the file",
    "Рентген": "X-ray",
    "Полупрозрачно: видны внутренние и перекрытые детали":
        "Semi-transparent: inner and hidden parts show",

    # mesh2mesh_gui/teximage.py
    "TGA {bits} бит окно не показывает": "{bits}-bit TGA is not shown by the window",
    "TGA повреждён": "TGA is damaged",
    "вид TGA {kind} окно не показывает": "TGA type {kind} is not shown by the window",
    "TGA с палитрой без палитры": "palette TGA without a palette",
    "TGA повреждён (обрезан)": "TGA is damaged (truncated)",
    "формат {fmt} окно не показывает": "the window cannot show {fmt}",
    "картинка не читается": "image cannot be read",

    # mesh2mesh_gui/viewer.py
    "ЛКМ — вращать · колесо — зум · СКМ / Ctrl+ЛКМ — сдвиг (+Shift — по полу)":
        "LMB — orbit · wheel — zoom · MMB / Ctrl+LMB — pan (+Shift — along the floor)",
    "Каркас": "Wireframe",
    "Рёбра многоугольников поверх любого режима": "Polygon edges on top of any mode",
    "Сглаживание: {mode}": "Smoothing: {mode}",
    "Z вверх": "Z up",

    # mesh2mesh_gui/window.py
    "mesh2mesh — конвертер 3D-моделей": "mesh2mesh — 3D model converter",
    "формат {ext} не поддерживается": "format {ext} is not supported",
    "1. Файл": "1. File",
    "Открыть модель  ({exts})": "Open model  ({exts})",
    "Числа в файле — это:": "Numbers in the file are:",
    "2. Содержимое": "2. Contents",
    "Щелчок по строке — объект крупно (с вложенными); «Показать всё» — вся модель. ×N — один "
    "меш, повторённый N раз (экземпляры).":
        "Click a row to zoom in on that object (with its children); “Show all” — the whole "
        "model. ×N — one mesh repeated N times (instances).",
    "3. Размер и оси": "3. Size and axes",
    "Масштаб:": "Scale:",
    "Любое число > 0: 25.4 — дюймы в мм, 0.1 — мм в см… Габарит не тот в 10/100/1000 раз — "
    "поправьте здесь":
        "Any number > 0: 25.4 — inches to mm, 0.1 — mm to cm… If the size is off by "
        "10/100/1000 times, fix it here",
    "Как в файле": "As in file",
    "Показать всё": "Show all",
    "Вся модель в кадре, без подсветки": "The whole model in view, no highlight",
    "4. Сохранить": "4. Save",
    "Формат:": "Format:",
    "Единицы:": "Units:",
    "текстом (ASCII)": "as text (ASCII)",
    "STL/PLY текстом: больше файл; STL так сохраняет имена объектов":
        "STL/PLY as text: bigger file; this way STL keeps object names",
    "Язык интерфейса": "Interface language",
    "Сохранить как {fmt}": "Save as {fmt}",
    "{axes} · клетка сетки {step} {unit}": "{axes} · grid cell {step} {unit}",
    "Открыть модель": "Open model",
    "Читаю {name}…": "Reading {name}…",
    "Рёбер больше {n} — каркас не рисуется": "More than {n} edges — wireframe is not drawn",
    "Не удалось прочитать файл\n{path}\n\n{msg}": "Could not read the file\n{path}\n\n{msg}",
    "Сохраняю и проверяю чтением…": "Saving and checking by reading back…",
    "Не удалось сохранить.\n\n{msg}": "Could not save.\n\n{msg}",
    "Объект": "Object",
    "Треугольников": "Triangles",
    "Материалы": "Materials",
    "Повернуть модель на 90° вокруг оси {axis}. Лежит на боку — крутите, пока синяя ось Z не "
    "смотрит вверх модели":
        "Rotate the model 90° about the {axis} axis. Lying on its side? Turn it until the blue "
        "Z axis points to the top of the model",
    "Файл не выбран. Можно перетащить файл в окно.":
        "No file selected. You can drag a file into the window.",
    "{fmt} по спецификации всегда в метрах.": "{fmt} is always in metres by specification.",
    "Единица в этот формат не записывается — получатель должен знать, что числа в {unit}.":
        "This format does not store the unit — the recipient must know the numbers are in "
        "{unit}.",
    "Этот формат не поддерживается.\nЕсть: {exts}.":
        "This format is not supported.\nSupported: {exts}.",
    "(без имени)": "(unnamed)",
    "Масштаб — положительное число, например 10 или 0.001":
        "Scale is a positive number, e.g. 10 or 0.001",
    "Нельзя записать поверх исходного файла — выберите другое имя.":
        "Cannot write over the source file — choose another name.",
    "Сохранено": "Saved",
    "Сохранено, но проверка чтением не прошла": "Saved, but the read-back check failed",
    "Технический отчёт": "Technical report",
    "Отчёт как в командной строке (можно скопировать).":
        "The report as on the command line (can be copied).",
    "В файле нет геометрии — сохранять нечего.": "The file has no geometry — nothing to save.",
    "Будет записано в {unit}.": "Will be written in {unit}.",
    "Показать в папке": "Show in folder",
    "Открыть модель, посмотреть содержимое и сохранить в другой формат можно как обычно — не "
    "будет только картинки. Поможет обновление драйвера видеокарты.":
        "You can still open a model, look at its contents and save it to another format — only "
        "the picture is missing. Updating the graphics driver may help.",
    "Идёт работа. Всё равно закрыть?": "Work is in progress. Close anyway?",
    "как у исходника": "same as source",
    "автор: {site}": "author: {site}",
    "Повёрнута относительно файла": "Rotated relative to the file",
    "Оси как в файле": "Axes as in the file",
    "Меш «{mesh}» повторён {n} раз (экземпляры)": "Mesh “{mesh}” repeated {n} times (instances)",
    "При чтении: {w}": "While reading: {w}",
    "Проверяю, что вместит {fmt}…": "Checking what {fmt} can hold…",
    "Пробная запись не удалась: {msg}": "Trial write failed: {msg}",
    "Видеокарта или драйвер не дают OpenGL 2.0 (так бывает в удалённом рабочем столе и "
    "виртуальных машинах).":
        "The graphics card or driver does not provide OpenGL 2.0 (this happens over Remote "
        "Desktop and in virtual machines).",
    "3D-просмотр недоступен": "3D view is not available",
}
