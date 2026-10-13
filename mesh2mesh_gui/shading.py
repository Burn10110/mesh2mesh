"""View modes of the 3D view: GLSL programs for pyqtgraph's GLMeshItem + GL state per mode.

Matcaps are procedural (a function of the eye-space normal), no textures. Lighting follows the
camera. Shaders stay GLSL ES 1.00 / desktop 1.20 compatible: pyqtgraph prepends "#version 100"
when the driver has ARB_ES2_compatibility (NVIDIA/AMD on Windows do).

Colour attribute contract: CLAY / METAL / XRAY get white for plain triangles and
display.SELECT_COLOR for highlighted ones; MATERIAL gets the model's own colours (or the
highlight). MATERIAL also samples u_tex when u_use_tex = 1 (set per texture group by the mesh
item) and the colour alpha is 1: highlighted triangles get alpha 0 = flat highlight, no map;
the texture replaces the colour RGB (the colour stays as the fallback when the image fails);
its alpha clips at 0.5 (no sorting of transparent surfaces in an opaque view)."""

from OpenGL import GL
from pyqtgraph.opengl import shaders

from mesh2mesh.i18n import N_

CLAY, METAL, MATERIAL, XRAY = "clay", "metal", "material", "xray"

# mode -> (button text, tooltip); show them through tr()
MODES = {
    CLAY: (N_("Глина"), N_("Матовая глина (matcap): форма без цветов — видны вмятины и рёбра")),
    METAL: (N_("Металл"), N_("Красный металл (matcap): блики подчёркивают кривизну и гладкость")),
    MATERIAL: (N_("Материал"), N_("Цвета материалов и цвета вершин (vertex paint) из файла")),
    XRAY: (N_("Рентген"), N_("Полупрозрачно: видны внутренние и перекрытые детали")),
}

_VERTEX = """
uniform mat4 u_mvp;
uniform mat3 u_normal;
attribute vec4 a_position;
attribute vec3 a_normal;
attribute vec4 a_color;
attribute vec2 a_texcoord;
varying vec4 v_color;
varying vec3 v_normal;
varying vec2 v_uv;
void main() {
    // normalise here, in highp: raw cross products (~1e-6 in the unit-sized view) would be
    // flushed to zero by mediump fragments; ours arrive unit-length, but keep it safe
    v_normal = normalize(u_normal * normalize(a_normal));
    v_color = a_color;
    v_uv = a_texcoord;
    gl_Position = u_mvp * a_position;
}
"""

_COMMON = """
#ifdef GL_ES
precision mediump float;
#endif
varying vec4 v_color;
varying vec3 v_normal;
varying vec2 v_uv;

// eye-space normal turned towards the camera: faces are two-sided
vec3 facing() {
    vec3 n = normalize(v_normal);
    return n.z < 0.0 ? -n : n;
}
// white = plain, anything coloured = highlighted (see the colour contract)
float selected() {
    return step(0.02, 1.0 - min(v_color.r, min(v_color.g, v_color.b)));
}
vec3 tint(vec3 m) {
    float l = dot(m, vec3(0.299, 0.587, 0.114));
    return mix(m, v_color.rgb * (0.35 + 0.9 * l), selected());
}
float phong(vec3 n, vec3 l, float power) {
    return pow(max(dot(n, normalize(l + vec3(0.0, 0.0, 1.0))), 0.0), power);
}
"""

_FRAGMENT = {
    CLAY: """
void main() {
    vec3 n = facing();
    vec3 key = normalize(vec3(-0.45, 0.6, 0.65));
    float wrap = clamp((dot(n, key) + 0.45) / 1.45, 0.0, 1.0);
    float cavity = 0.6 + 0.4 * n.z;                       // grazing faces darker
    vec3 base = vec3(0.80, 0.74, 0.68);
    vec3 col = base * (0.22 + 0.78 * wrap) * cavity;
    col += vec3(0.05, 0.06, 0.08) * max(-n.y, 0.0);       // cool bounce from below
    col += vec3(0.07) * phong(n, key, 20.0);
    gl_FragColor = vec4(tint(col), 1.0);
}
""",
    METAL: """
void main() {
    vec3 n = facing();
    vec3 r = reflect(vec3(0.0, 0.0, -1.0), n);
    float sky = smoothstep(-0.08, 0.08, r.y);             // sharp horizon in the reflection
    vec3 col = mix(vec3(0.09, 0.008, 0.008), vec3(0.62, 0.05, 0.04), sky);
    col *= 0.75 + 0.25 * r.y;
    col += vec3(0.30, 0.03, 0.02) * pow(1.0 - n.z, 2.0);  // red rim
    col += vec3(1.0, 0.86, 0.80) * phong(n, vec3(-0.4, 0.5, 0.75), 90.0);
    col += vec3(0.9, 0.55, 0.5) * 0.5 * phong(n, vec3(0.65, 0.25, 0.7), 220.0);
    gl_FragColor = vec4(tint(col), 1.0);
}
""",
    MATERIAL: """
uniform sampler2D u_tex;
uniform float u_use_tex;
void main() {
    vec3 n = facing();
    vec3 key = normalize(vec3(-0.35, 0.5, 0.8));
    float k = 0.42 + 0.58 * max(dot(n, key), 0.0);
    vec3 base = v_color.rgb;
    if (u_use_tex * v_color.a > 0.5) {
        vec4 t = texture2D(u_tex, v_uv);
        if (t.a < 0.5)
            discard;              // alpha clip: lashes, hair cards, decals
        base = t.rgb;
    }
    vec3 col = base * k + vec3(0.04) * phong(n, key, 30.0);
    gl_FragColor = vec4(col, 1.0);
}
""",
    XRAY: """
void main() {
    vec3 n = facing();
    float f = 1.0 - n.z;
    float a = 0.035 + 0.5 * f * f;                        // edges on, faces almost clear
    float s = selected();
    vec3 col = mix(vec3(0.55, 0.78, 1.0), v_color.rgb, s);
    gl_FragColor = vec4(col, mix(a, 0.12 + 1.2 * a, s));
}
""",
}

_SOLID = {GL.GL_DEPTH_TEST: True, GL.GL_BLEND: False, GL.GL_CULL_FACE: False,
          GL.GL_POLYGON_OFFSET_FILL: True, "glPolygonOffset": (1.0, 1.0)}  # edges drawn on top
# additive, no depth test (so no depth writes either): order-independent, everything shows
_XRAY = {GL.GL_DEPTH_TEST: False, GL.GL_BLEND: True, GL.GL_CULL_FACE: False,
         GL.GL_POLYGON_OFFSET_FILL: False, "glBlendFunc": (GL.GL_SRC_ALPHA, GL.GL_ONE)}

EDGE_DARK = (0.10, 0.10, 0.12, 0.55)
EDGE_XRAY = (0.70, 0.85, 1.00, 0.22)

def program(mode):
    """New pyqtgraph ShaderProgram of a mode. It compiles lazily and keeps the GL id of the
    first context it ran in, so every view keeps its own (programs are not shared between
    contexts: a second view would get 'invalid value')."""
    return shaders.ShaderProgram(
        f"mesh2mesh_{mode}",
        [shaders.VertexShader(_VERTEX), shaders.FragmentShader(_COMMON + _FRAGMENT[mode])])


def gl_options(mode):
    return _XRAY if mode == XRAY else _SOLID


def edge_color(mode):
    return EDGE_XRAY if mode == XRAY else EDGE_DARK


def uses_model_colors(mode):
    return mode == MATERIAL


def uses_textures(mode):
    return mode == MATERIAL
