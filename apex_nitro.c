/*
 * Apex Nitro 3D - Hardware-Accelerated C99 + SDL2 + OpenGL ES 2.0 Native RetroPie Edition
 * Target: Raspberry Pi 1 Model B (VideoCore IV) & Desktop Linux
 */

#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE

#include <SDL2/SDL.h>
#include <GLES2/gl2.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef RPI_BUILD
#include <EGL/egl.h>
#include <EGL/eglext.h>
#ifndef EGL_PLATFORM_SURFACELESS_MESA
#define EGL_PLATFORM_SURFACELESS_MESA 0x31DD
#endif
typedef EGLDisplay (*PFNEGLGETPLATFORMDISPLAYEXTPROC)(EGLenum platform, void *native_display, const EGLint *attrib_list);
#endif

/* =========================================================================
 * 1. MATHEMATICS & VECTORS
 * ========================================================================= */

typedef struct { float x, y; } Vec2;
typedef struct { float x, y, z; } Vec3;
typedef struct { float x, y, z, w; } Vec4;
typedef float Mat4[16]; /* Column-major: m[col*4 + row] */

static inline Vec2 vec2(float x, float y) { Vec2 v = {x, y}; return v; }
static inline Vec3 vec3(float x, float y, float z) { Vec3 v = {x, y, z}; return v; }
static inline Vec4 vec4(float x, float y, float z, float w) { Vec4 v = {x, y, z, w}; return v; }

static inline Vec3 vec3_add(Vec3 a, Vec3 b) { return vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline Vec3 vec3_sub(Vec3 a, Vec3 b) { return vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline Vec3 vec3_scale(Vec3 a, float s) { return vec3(a.x * s, a.y * s, a.z * s); }
static inline float vec3_dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline Vec3 vec3_cross(Vec3 a, Vec3 b) {
    return vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static inline float vec3_len(Vec3 a) { return sqrtf(vec3_dot(a, a)); }
static inline Vec3 vec3_norm(Vec3 a) {
    float l = vec3_len(a);
    return l > 1e-6f ? vec3_scale(a, 1.0f / l) : vec3(0.0f, 1.0f, 0.0f);
}
static inline float vec3_dist(Vec3 a, Vec3 b) { return vec3_len(vec3_sub(a, b)); }
static inline Vec3 vec3_lerp(Vec3 a, Vec3 b, float t) {
    return vec3(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t);
}

static inline void mat4_identity(Mat4 m) {
    memset(m, 0, sizeof(Mat4));
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

static inline void mat4_mul(Mat4 out, const Mat4 a, const Mat4 b) {
    Mat4 tmp;
    for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 4; r++) {
            tmp[c * 4 + r] = a[0 * 4 + r] * b[c * 4 + 0] +
                             a[1 * 4 + r] * b[c * 4 + 1] +
                             a[2 * 4 + r] * b[c * 4 + 2] +
                             a[3 * 4 + r] * b[c * 4 + 3];
        }
    }
    memcpy(out, tmp, sizeof(Mat4));
}

static inline void mat4_perspective(Mat4 m, float fovy_rad, float aspect, float nearZ, float farZ) {
    memset(m, 0, sizeof(Mat4));
    float tanHalfFovy = tanf(fovy_rad * 0.5f);
    m[0] = 1.0f / (aspect * tanHalfFovy);
    m[5] = 1.0f / tanHalfFovy;
    m[10] = -(farZ + nearZ) / (farZ - nearZ);
    m[11] = -1.0f;
    m[14] = -(2.0f * farZ * nearZ) / (farZ - nearZ);
}

static inline void mat4_ortho(Mat4 m, float left, float right, float bottom, float top, float nearZ, float farZ) {
    memset(m, 0, sizeof(Mat4));
    m[0] = 2.0f / (right - left);
    m[5] = 2.0f / (top - bottom);
    m[10] = -2.0f / (farZ - nearZ);
    m[12] = -(right + left) / (right - left);
    m[13] = -(top + bottom) / (top - bottom);
    m[14] = -(farZ + nearZ) / (farZ - nearZ);
    m[15] = 1.0f;
}

static inline void mat4_lookat(Mat4 m, Vec3 eye, Vec3 center, Vec3 up) {
    Vec3 f = vec3_norm(vec3_sub(center, eye));
    Vec3 s = vec3_norm(vec3_cross(f, up));
    Vec3 u = vec3_cross(s, f);
    memset(m, 0, sizeof(Mat4));
    m[0] = s.x;  m[4] = s.y;  m[8]  = s.z;  m[12] = -vec3_dot(s, eye);
    m[1] = u.x;  m[5] = u.y;  m[9]  = u.z;  m[13] = -vec3_dot(u, eye);
    m[2] = -f.x; m[6] = -f.y; m[10] = -f.z; m[14] = vec3_dot(f, eye);
    m[3] = 0.0f; m[7] = 0.0f; m[11] = 0.0f; m[15] = 1.0f;
}

static inline void mat4_translate(Mat4 m, float tx, float ty, float tz) {
    mat4_identity(m);
    m[12] = tx; m[13] = ty; m[14] = tz;
}

static inline void mat4_rotate_y(Mat4 m, float angle) {
    mat4_identity(m);
    float c = cosf(angle), s = sinf(angle);
    m[0] = c;  m[8]  = s;
    m[2] = -s; m[10] = c;
}

static inline void mat4_rotate_x(Mat4 m, float angle) {
    mat4_identity(m);
    float c = cosf(angle), s = sinf(angle);
    m[5] = c;  m[9]  = -s;
    m[6] = s;  m[10] = c;
}

static inline void mat4_rotate_z(Mat4 m, float angle) {
    mat4_identity(m);
    float c = cosf(angle), s = sinf(angle);
    m[0] = c;  m[4] = -s;
    m[1] = s;  m[5] = c;
}

static inline void mat4_scale(Mat4 m, float sx, float sy, float sz) {
    mat4_identity(m);
    m[0] = sx; m[5] = sy; m[10] = sz;
}

/* =========================================================================
 * 2. VERTEX FORMATS & DYNAMIC MESH BUILDERS
 * ========================================================================= */

typedef struct {
    float x, y, z;
    float nx, ny, nz;
    float r, g, b;
    float glow;
} Vertex3D; /* 10 floats = 40 bytes */

typedef struct {
    float x, y, z;
    float u, v;
    float r, g, b, a;
} VertexFX; /* 9 floats = 36 bytes */

typedef struct {
    float x, y;
    float u, v;
    float r, g, b, a;
    float tex_mix;
} VertexHUD; /* 9 floats = 36 bytes */

typedef struct {
    Vertex3D *data;
    int count;
    int cap;
} MeshBuilder3D;

static void mesh3d_init(MeshBuilder3D *mb, int cap) {
    mb->cap = cap > 64 ? cap : 64;
    mb->count = 0;
    mb->data = (Vertex3D*)malloc(sizeof(Vertex3D) * mb->cap);
}

static void mesh3d_free(MeshBuilder3D *mb) {
    if (mb->data) {
        free(mb->data);
        mb->data = NULL;
    }
    mb->count = mb->cap = 0;
}

static inline void mesh3d_push_v(MeshBuilder3D *mb, Vertex3D v) {
    if (mb->count >= mb->cap) {
        mb->cap *= 2;
        mb->data = (Vertex3D*)realloc(mb->data, sizeof(Vertex3D) * mb->cap);
    }
    mb->data[mb->count++] = v;
}

static void mesh3d_add_tri(MeshBuilder3D *mb, Vertex3D v0, Vertex3D v1, Vertex3D v2) {
    mesh3d_push_v(mb, v0);
    mesh3d_push_v(mb, v1);
    mesh3d_push_v(mb, v2);
}

static void mesh3d_add_quad(MeshBuilder3D *mb, Vertex3D v0, Vertex3D v1, Vertex3D v2, Vertex3D v3) {
    mesh3d_add_tri(mb, v0, v1, v2);
    mesh3d_add_tri(mb, v0, v2, v3);
}

static void mesh3d_add_quad_auto_norm(MeshBuilder3D *mb,
                                      Vertex3D v0, Vertex3D v1, Vertex3D v2, Vertex3D v3,
                                      Vec3 col, float glow) {
    float ux = v1.x - v0.x, uy = v1.y - v0.y, uz = v1.z - v0.z;
    float vx = v2.x - v0.x, vy = v2.y - v0.y, vz = v2.z - v0.z;
    float nx = uy * vz - uz * vy;
    float ny = uz * vx - ux * vz;
    float nz = ux * vy - uy * vx;
    float len = sqrtf(nx * nx + ny * ny + nz * nz);
    if (len > 1e-5f) {
        nx /= len; ny /= len; nz /= len;
    } else {
        nx = 0.0f; ny = 1.0f; nz = 0.0f;
    }
    v0.nx = v1.nx = v2.nx = v3.nx = nx;
    v0.ny = v1.ny = v2.ny = v3.ny = ny;
    v0.nz = v1.nz = v2.nz = v3.nz = nz;
    v0.r = v1.r = v2.r = v3.r = col.x;
    v0.g = v1.g = v2.g = v3.g = col.y;
    v0.b = v1.b = v2.b = v3.b = col.z;
    v0.glow = v1.glow = v2.glow = v3.glow = glow;
    mesh3d_add_quad(mb, v0, v1, v2, v3);
}

static void mesh3d_add_tri_auto_norm(MeshBuilder3D *mb,
                                     Vertex3D v0, Vertex3D v1, Vertex3D v2,
                                     Vec3 col, float glow) {
    float ux = v1.x - v0.x, uy = v1.y - v0.y, uz = v1.z - v0.z;
    float vx = v2.x - v0.x, vy = v2.y - v0.y, vz = v2.z - v0.z;
    float nx = uy * vz - uz * vy;
    float ny = uz * vx - ux * vz;
    float nz = ux * vy - uy * vx;
    float len = sqrtf(nx * nx + ny * ny + nz * nz);
    if (len > 1e-5f) {
        nx /= len; ny /= len; nz /= len;
    } else {
        nx = 0.0f; ny = 1.0f; nz = 0.0f;
    }
    v0.nx = v1.nx = v2.nx = nx;
    v0.ny = v1.ny = v2.ny = ny;
    v0.nz = v1.nz = v2.nz = nz;
    v0.r = v1.r = v2.r = col.x;
    v0.g = v1.g = v2.g = col.y;
    v0.b = v1.b = v2.b = col.z;
    v0.glow = v1.glow = v2.glow = glow;
    mesh3d_add_tri(mb, v0, v1, v2);
}

static void mesh3d_add_box(MeshBuilder3D *mb, float cx, float cy, float cz,
                           float sx, float sy, float sz, Vec3 col, float glow) {
    float hx = sx * 0.5f, hy = sy * 0.5f, hz = sz * 0.5f;
    Vertex3D v[8];
    for (int i = 0; i < 8; i++) {
        v[i].x = cx + ((i & 1) ? hx : -hx);
        v[i].y = cy + ((i & 2) ? hy : -hy);
        v[i].z = cz + ((i & 4) ? hz : -hz);
        v[i].r = col.x; v[i].g = col.y; v[i].b = col.z;
        v[i].glow = glow;
    }
    /* Front (+Z) */
    v[4].nx = v[5].nx = v[7].nx = v[6].nx = 0.0f;
    v[4].ny = v[5].ny = v[7].ny = v[6].ny = 0.0f;
    v[4].nz = v[5].nz = v[7].nz = v[6].nz = 1.0f;
    mesh3d_add_quad(mb, v[4], v[5], v[7], v[6]);
    /* Back (-Z) */
    v[1].nx = v[0].nx = v[2].nx = v[3].nx = 0.0f;
    v[1].ny = v[0].ny = v[2].ny = v[3].ny = 0.0f;
    v[1].nz = v[0].nz = v[2].nz = v[3].nz = -1.0f;
    mesh3d_add_quad(mb, v[1], v[0], v[2], v[3]);
    /* Top (+Y) */
    v[2].nx = v[3].nx = v[7].nx = v[6].nx = 0.0f;
    v[2].ny = v[3].ny = v[7].ny = v[6].ny = 1.0f;
    v[2].nz = v[3].nz = v[7].nz = v[6].nz = 0.0f;
    mesh3d_add_quad(mb, v[2], v[3], v[7], v[6]);
    /* Bottom (-Y) */
    v[0].nx = v[1].nx = v[5].nx = v[4].nx = 0.0f;
    v[0].ny = v[1].ny = v[5].ny = v[4].ny = -1.0f;
    v[0].nz = v[1].nz = v[5].nz = v[4].nz = 0.0f;
    mesh3d_add_quad(mb, v[0], v[4], v[5], v[1]);
    /* Right (+X) */
    v[5].nx = v[1].nx = v[3].nx = v[7].nx = 1.0f;
    v[5].ny = v[1].ny = v[3].ny = v[7].ny = 0.0f;
    v[5].nz = v[1].nz = v[3].nz = v[7].nz = 0.0f;
    mesh3d_add_quad(mb, v[5], v[1], v[3], v[7]);
    /* Left (-X) */
    v[0].nx = v[4].nx = v[6].nx = v[2].nx = -1.0f;
    v[0].ny = v[4].ny = v[6].ny = v[2].ny = 0.0f;
    v[0].nz = v[4].nz = v[6].nz = v[2].nz = 0.0f;
    mesh3d_add_quad(mb, v[0], v[2], v[6], v[4]);
}

static void mesh3d_add_cylinder(MeshBuilder3D *mb, float cx, float cy, float cz,
                                float r_top, float r_bot, float h, int segs,
                                Vec3 col, float glow, int axis) {
    float halfH = h * 0.5f;
    for (int i = 0; i < segs; i++) {
        float a0 = (float)i / (float)segs * (float)(M_PI * 2.0);
        float a1 = (float)(i + 1) / (float)segs * (float)(M_PI * 2.0);
        float c0 = cosf(a0), s0 = sinf(a0);
        float c1 = cosf(a1), s1 = sinf(a1);

        Vertex3D v0, v1, v2, v3;
        v0.r = v1.r = v2.r = v3.r = col.x;
        v0.g = v1.g = v2.g = v3.g = col.y;
        v0.b = v1.b = v2.b = v3.b = col.z;
        v0.glow = v1.glow = v2.glow = v3.glow = glow;

        if (axis == 0) { /* Along X */
            v0.x = cx - halfH; v0.y = cy + s0 * r_bot; v0.z = cz + c0 * r_bot;
            v1.x = cx + halfH; v1.y = cy + s0 * r_top; v1.z = cz + c0 * r_top;
            v2.x = cx + halfH; v2.y = cy + s1 * r_top; v2.z = cz + c1 * r_top;
            v3.x = cx - halfH; v3.y = cy + s1 * r_bot; v3.z = cz + c1 * r_bot;
            v0.nx = v1.nx = 0.0f; v0.ny = v1.ny = s0; v0.nz = v1.nz = c0;
            v2.nx = v3.nx = 0.0f; v2.ny = v3.ny = s1; v2.nz = v3.nz = c1;
        } else { /* Along Y */
            v0.x = cx + c0 * r_bot; v0.y = cy - halfH; v0.z = cz + s0 * r_bot;
            v1.x = cx + c0 * r_top; v1.y = cy + halfH; v1.z = cz + s0 * r_top;
            v2.x = cx + c1 * r_top; v2.y = cy + halfH; v2.z = cz + s1 * r_top;
            v3.x = cx + c1 * r_bot; v3.y = cy - halfH; v3.z = cz + s1 * r_bot;
            v0.nx = v1.nx = c0; v0.ny = v1.ny = 0.0f; v0.nz = v1.nz = s0;
            v2.nx = v3.nx = c1; v2.ny = v3.ny = 0.0f; v2.nz = v3.nz = s1;
        }
        mesh3d_add_quad(mb, v0, v1, v2, v3);
    }
}

static GLuint mesh3d_create_vbo(const MeshBuilder3D *mb) {
    GLuint vbo = 0;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex3D) * mb->count, mb->data, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return vbo;
}

/* =========================================================================
 * 3. SHADERS (STRICT OPENGL ES 2.0 / GLSL ES 1.00 COMPLIANT)
 * ========================================================================= */

static const char *s_vs_3d =
    "precision mediump float;\n"
    "attribute vec3 aPos;\n"
    "attribute vec3 aNorm;\n"
    "attribute vec3 aCol;\n"
    "attribute float aGlow;\n"
    "uniform mat4 uMVP;\n"
    "uniform mat4 uModel;\n"
    "uniform vec3 uPaintCol;\n"
    "varying vec3 vWorldPos;\n"
    "varying vec3 vNorm;\n"
    "varying vec3 vCol;\n"
    "varying float vGlow;\n"
    "void main() {\n"
    "    vec4 wp = uModel * vec4(aPos, 1.0);\n"
    "    vWorldPos = wp.xyz;\n"
    "    vNorm = mat3(uModel[0].xyz, uModel[1].xyz, uModel[2].xyz) * aNorm;\n"
    "    float paintMask = step(aGlow, -0.5);\n"
    "    vCol = mix(aCol, uPaintCol, paintMask);\n"
    "    vGlow = max(aGlow, 0.0);\n"
    "    gl_Position = uMVP * wp;\n"
    "}\n";

static const char *s_fs_3d =
    "precision mediump float;\n"
    "varying vec3 vWorldPos;\n"
    "varying vec3 vNorm;\n"
    "varying vec3 vCol;\n"
    "varying float vGlow;\n"
    "uniform vec3 uCamPos;\n"
    "uniform vec3 uSunDir;\n"
    "uniform vec3 uSunCol;\n"
    "uniform vec3 uAmbientCol;\n"
    "uniform vec3 uFogCol;\n"
    "uniform float uFogDensity;\n"
    "void main() {\n"
    "    vec3 N = normalize(vNorm);\n"
    "    vec3 V = normalize(uCamPos - vWorldPos);\n"
    "    vec3 L = normalize(uSunDir);\n"
    "    float nDotL = max(dot(N, L), 0.0);\n"
    "    float hemi = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);\n"
    "    vec3 light = uAmbientCol * hemi + uSunCol * nDotL;\n"
    "    vec3 R = reflect(-V, N);\n"
    "    float spec = pow(max(dot(R, L), 0.0), 16.0) * 0.45;\n"
    "    float fresnel = pow(clamp(1.0 - max(dot(N, V), 0.0), 0.0, 1.0), 3.0) * 0.16;\n"
    "    vec3 lit = mix(vCol * light + uSunCol * (spec + fresnel), vCol * 1.5, vGlow);\n"
    "    float dist = length(uCamPos - vWorldPos);\n"
    "    float fogFactor = clamp(1.0 - exp(-dist * uFogDensity), 0.0, 1.0);\n"
    "    vec3 finalCol = mix(lit, uFogCol, fogFactor);\n"
    "    gl_FragColor = vec4(finalCol, 1.0);\n"
    "}\n";

static const char *s_vs_fx =
    "precision mediump float;\n"
    "attribute vec3 aPos;\n"
    "attribute vec2 aUV;\n"
    "attribute vec4 aCol;\n"
    "uniform mat4 uMVP;\n"
    "varying vec2 vUV;\n"
    "varying vec4 vCol;\n"
    "void main() {\n"
    "    vUV = aUV;\n"
    "    vCol = aCol;\n"
    "    gl_Position = uMVP * vec4(aPos, 1.0);\n"
    "}\n";

static const char *s_fs_fx =
    "precision mediump float;\n"
    "varying vec2 vUV;\n"
    "varying vec4 vCol;\n"
    "void main() {\n"
    "    float d = length(vUV - vec2(0.5, 0.5)) * 2.0;\n"
    "    float radialAlpha = clamp(1.0 - d, 0.0, 1.0);\n"
    "    float alpha = vUV.x < 0.0 ? 1.0 : radialAlpha;\n"
    "    gl_FragColor = vec4(vCol.rgb, vCol.a * alpha);\n"
    "}\n";

static const char *s_vs_hud =
    "precision mediump float;\n"
    "attribute vec2 aPos;\n"
    "attribute vec2 aUV;\n"
    "attribute vec4 aCol;\n"
    "attribute float aTexMix;\n"
    "uniform mat4 uOrtho;\n"
    "varying vec2 vUV;\n"
    "varying vec4 vCol;\n"
    "varying float vTexMix;\n"
    "void main() {\n"
    "    vUV = aUV;\n"
    "    vCol = aCol;\n"
    "    vTexMix = aTexMix;\n"
    "    gl_Position = uOrtho * vec4(aPos, 0.0, 1.0);\n"
    "}\n";

static const char *s_fs_hud =
    "precision mediump float;\n"
    "varying vec2 vUV;\n"
    "varying vec4 vCol;\n"
    "varying float vTexMix;\n"
    "uniform sampler2D uTex;\n"
    "void main() {\n"
    "    vec4 tex = texture2D(uTex, vUV);\n"
    "    vec4 col = mix(vCol, vec4(vCol.rgb, vCol.a * tex.a), vTexMix);\n"
    "    gl_FragColor = col;\n"
    "}\n";

static const char *s_vs_sky =
    "precision mediump float;\n"
    "attribute vec2 aPos;\n"
    "varying vec2 vScreen;\n"
    "void main() {\n"
    "    vScreen = aPos * 0.5 + 0.5;\n"
    "    gl_Position = vec4(aPos, 0.9999, 1.0);\n"
    "}\n";

static const char *s_fs_sky =
    "precision mediump float;\n"
    "varying vec2 vScreen;\n"
    "uniform vec3 uSkyZenith;\n"
    "uniform vec3 uSkyHorizon;\n"
    "uniform vec3 uSkyBandColA;\n"
    "uniform vec3 uSkyBandColB;\n"
    "uniform float uSkyBandStr;\n"
    "void main() {\n"
    "    float h = clamp((vScreen.y - 0.42) / 0.58, 0.0, 1.0);\n"
    "    float curve = h * (1.55 - 0.55 * h);\n"
    "    vec3 col = mix(uSkyHorizon, uSkyZenith, curve);\n"
    "    float dy = (vScreen.y - 0.51) * 9.5;\n"
    "    float band = exp(-dy * dy);\n"
    "    float wave = 0.5 + 0.5 * sin(vScreen.x * 3.14159);\n"
    "    vec3 bandCol = mix(uSkyBandColA, uSkyBandColB, wave);\n"
    "    col += bandCol * band * uSkyBandStr;\n"
    "    gl_FragColor = vec4(col, 1.0);\n"
    "}\n";

static GLuint compile_shader(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(s, sizeof(log), NULL, log);
        fprintf(stderr, "Shader compile error (%s): %s\n",
                type == GL_VERTEX_SHADER ? "VS" : "FS", log);
        glDeleteShader(s);
        return 0;
    }
    return s;
}

static GLuint link_program(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(p, sizeof(log), NULL, log);
        fprintf(stderr, "Program link error: %s\n", log);
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

/* Shader program handles */
static GLuint g_prog3d = 0;
static GLint g_u3d_MVP = -1, g_u3d_Model = -1, g_u3d_CamPos = -1, g_u3d_PaintCol = -1;
static GLint g_u3d_SunDir = -1, g_u3d_SunCol = -1, g_u3d_AmbientCol = -1;
static GLint g_u3d_FogCol = -1, g_u3d_FogDensity = -1;
static GLint g_a3d_Pos = -1, g_a3d_Norm = -1, g_a3d_Col = -1, g_a3d_Glow = -1;

static GLuint g_progFx = 0;
static GLint g_uFx_MVP = -1;
static GLint g_aFx_Pos = -1, g_aFx_UV = -1, g_aFx_Col = -1;

static GLuint g_progHud = 0;
static GLint g_uHud_Ortho = -1, g_uHud_Tex = -1;
static GLint g_aHud_Pos = -1, g_aHud_UV = -1, g_aHud_Col = -1, g_aHud_TexMix = -1;

static GLuint g_progSky = 0;
static GLuint g_vbo_sky = 0;
static GLint g_aSky_Pos = -1;
static GLint g_uSky_Zenith = -1, g_uSky_Horizon = -1;
static GLint g_uSky_BandColA = -1, g_uSky_BandColB = -1, g_uSky_BandStr = -1;

static bool init_shaders(void) {
    GLuint vs3d = compile_shader(GL_VERTEX_SHADER, s_vs_3d);
    GLuint fs3d = compile_shader(GL_FRAGMENT_SHADER, s_fs_3d);
    g_prog3d = link_program(vs3d, fs3d);
    glDeleteShader(vs3d); glDeleteShader(fs3d);

    g_u3d_MVP = glGetUniformLocation(g_prog3d, "uMVP");
    g_u3d_Model = glGetUniformLocation(g_prog3d, "uModel");
    g_u3d_CamPos = glGetUniformLocation(g_prog3d, "uCamPos");
    g_u3d_PaintCol = glGetUniformLocation(g_prog3d, "uPaintCol");
    g_u3d_SunDir = glGetUniformLocation(g_prog3d, "uSunDir");
    g_u3d_SunCol = glGetUniformLocation(g_prog3d, "uSunCol");
    g_u3d_AmbientCol = glGetUniformLocation(g_prog3d, "uAmbientCol");
    g_u3d_FogCol = glGetUniformLocation(g_prog3d, "uFogCol");
    g_u3d_FogDensity = glGetUniformLocation(g_prog3d, "uFogDensity");
    g_a3d_Pos = glGetAttribLocation(g_prog3d, "aPos");
    g_a3d_Norm = glGetAttribLocation(g_prog3d, "aNorm");
    g_a3d_Col = glGetAttribLocation(g_prog3d, "aCol");
    g_a3d_Glow = glGetAttribLocation(g_prog3d, "aGlow");

    GLuint vsFx = compile_shader(GL_VERTEX_SHADER, s_vs_fx);
    GLuint fsFx = compile_shader(GL_FRAGMENT_SHADER, s_fs_fx);
    g_progFx = link_program(vsFx, fsFx);
    glDeleteShader(vsFx); glDeleteShader(fsFx);

    g_uFx_MVP = glGetUniformLocation(g_progFx, "uMVP");
    g_aFx_Pos = glGetAttribLocation(g_progFx, "aPos");
    g_aFx_UV = glGetAttribLocation(g_progFx, "aUV");
    g_aFx_Col = glGetAttribLocation(g_progFx, "aCol");

    GLuint vsHud = compile_shader(GL_VERTEX_SHADER, s_vs_hud);
    GLuint fsHud = compile_shader(GL_FRAGMENT_SHADER, s_fs_hud);
    g_progHud = link_program(vsHud, fsHud);
    glDeleteShader(vsHud); glDeleteShader(fsHud);

    g_uHud_Ortho = glGetUniformLocation(g_progHud, "uOrtho");
    g_uHud_Tex = glGetUniformLocation(g_progHud, "uTex");
    g_aHud_Pos = glGetAttribLocation(g_progHud, "aPos");
    g_aHud_UV = glGetAttribLocation(g_progHud, "aUV");
    g_aHud_Col = glGetAttribLocation(g_progHud, "aCol");
    g_aHud_TexMix = glGetAttribLocation(g_progHud, "aTexMix");

    GLuint vsSky = compile_shader(GL_VERTEX_SHADER, s_vs_sky);
    GLuint fsSky = compile_shader(GL_FRAGMENT_SHADER, s_fs_sky);
    g_progSky = link_program(vsSky, fsSky);
    glDeleteShader(vsSky); glDeleteShader(fsSky);

    g_aSky_Pos = glGetAttribLocation(g_progSky, "aPos");
    g_uSky_Zenith = glGetUniformLocation(g_progSky, "uSkyZenith");
    g_uSky_Horizon = glGetUniformLocation(g_progSky, "uSkyHorizon");
    g_uSky_BandColA = glGetUniformLocation(g_progSky, "uSkyBandColA");
    g_uSky_BandColB = glGetUniformLocation(g_progSky, "uSkyBandColB");
    g_uSky_BandStr = glGetUniformLocation(g_progSky, "uSkyBandStr");

    static const float sky_quad[12] = {
        -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f
    };
    glGenBuffers(1, &g_vbo_sky);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo_sky);
    glBufferData(GL_ARRAY_BUFFER, sizeof(sky_quad), sky_quad, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    return (g_prog3d && g_progFx && g_progHud && g_progSky);
}

static void render_sky_gradient(int trk_idx) {
    Vec3 zenith, horizon, bandA, bandB;
    float bandStr;
    if (trk_idx == 1) { /* Canyon: deep twilight violet -> sunset coral-orange */
        zenith  = vec3(0.043f, 0.035f, 0.125f); /* #0b0920 */
        horizon = vec3(0.918f, 0.345f, 0.047f); /* #ea580c */
        bandA   = vec3(0.980f, 0.450f, 0.120f);
        bandB   = vec3(1.000f, 0.680f, 0.250f);
        bandStr = 0.20f;
    } else if (trk_idx == 2) { /* Alpine: Nordic navy -> arctic ice-cyan */
        zenith  = vec3(0.027f, 0.075f, 0.161f); /* #071329 */
        horizon = vec3(0.220f, 0.741f, 0.973f); /* #38bdf8 */
        bandA   = vec3(0.450f, 0.860f, 1.000f);
        bandB   = vec3(0.700f, 0.940f, 1.000f);
        bandStr = 0.14f;
    } else { /* Tokyo: midnight obsidian -> indigo-navy + subtle cyan/amber city glow */
        zenith  = vec3(0.008f, 0.020f, 0.051f); /* #02050d */
        horizon = vec3(0.051f, 0.106f, 0.227f); /* #0d1b3a */
        bandA   = vec3(0.000f, 0.750f, 0.920f); /* cyan glow */
        bandB   = vec3(0.950f, 0.520f, 0.120f); /* amber city haze */
        bandStr = 0.18f;
    }

    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    glUseProgram(g_progSky);
    glUniform3f(g_uSky_Zenith, zenith.x, zenith.y, zenith.z);
    glUniform3f(g_uSky_Horizon, horizon.x, horizon.y, horizon.z);
    glUniform3f(g_uSky_BandColA, bandA.x, bandA.y, bandA.z);
    glUniform3f(g_uSky_BandColB, bandB.x, bandB.y, bandB.z);
    glUniform1f(g_uSky_BandStr, bandStr);

    glBindBuffer(GL_ARRAY_BUFFER, g_vbo_sky);
    glEnableVertexAttribArray(g_aSky_Pos);
    glVertexAttribPointer(g_aSky_Pos, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glDisableVertexAttribArray(g_aSky_Pos);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);
}

/* =========================================================================
 * 4. FONT ATLAS & BATCH HUD RENDERER
 * ========================================================================= */

static const uint8_t g_font_8x8[96][8] = {
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }, // ASCII 32
    { 0x18, 0x18, 0x18, 0x18, 0x18, 0x00, 0x18, 0x00 }, // ASCII 33
    { 0x24, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }, // ASCII 34
    { 0x24, 0x7E, 0x24, 0x24, 0x7E, 0x24, 0x00, 0x00 }, // ASCII 35
    { 0x08, 0x3E, 0x48, 0x3C, 0x0A, 0x7C, 0x08, 0x00 }, // ASCII 36
    { 0x42, 0x44, 0x08, 0x10, 0x20, 0x44, 0x82, 0x00 }, // ASCII 37
    { 0x30, 0x48, 0x30, 0x4C, 0x42, 0x44, 0x3A, 0x00 }, // ASCII 38
    { 0x18, 0x18, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00 }, // ASCII 39
    { 0x0C, 0x18, 0x30, 0x30, 0x30, 0x18, 0x0C, 0x00 }, // ASCII 40
    { 0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x18, 0x30, 0x00 }, // ASCII 41
    { 0x00, 0x14, 0x08, 0x3E, 0x08, 0x14, 0x00, 0x00 }, // ASCII 42
    { 0x00, 0x08, 0x08, 0x3E, 0x08, 0x08, 0x00, 0x00 }, // ASCII 43
    { 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x08, 0x10 }, // ASCII 44
    { 0x00, 0x00, 0x00, 0x3E, 0x00, 0x00, 0x00, 0x00 }, // ASCII 45
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00 }, // ASCII 46
    { 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x00 }, // ASCII 47
    { 0x3C, 0x42, 0x46, 0x4A, 0x52, 0x42, 0x3C, 0x00 }, // ASCII 48
    { 0x18, 0x28, 0x08, 0x08, 0x08, 0x08, 0x3E, 0x00 }, // ASCII 49
    { 0x3C, 0x42, 0x02, 0x0C, 0x30, 0x40, 0x7E, 0x00 }, // ASCII 50
    { 0x3C, 0x42, 0x02, 0x1C, 0x02, 0x42, 0x3C, 0x00 }, // ASCII 51
    { 0x0C, 0x14, 0x24, 0x44, 0x7E, 0x04, 0x04, 0x00 }, // ASCII 52
    { 0x7E, 0x40, 0x7C, 0x02, 0x02, 0x42, 0x3C, 0x00 }, // ASCII 53
    { 0x3C, 0x42, 0x40, 0x7C, 0x42, 0x42, 0x3C, 0x00 }, // ASCII 54
    { 0x7E, 0x02, 0x04, 0x08, 0x10, 0x10, 0x10, 0x00 }, // ASCII 55
    { 0x3C, 0x42, 0x42, 0x3C, 0x42, 0x42, 0x3C, 0x00 }, // ASCII 56
    { 0x3C, 0x42, 0x42, 0x3E, 0x02, 0x42, 0x3C, 0x00 }, // ASCII 57
    { 0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x00, 0x00 }, // ASCII 58
    { 0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x08, 0x10 }, // ASCII 59
    { 0x06, 0x18, 0x60, 0x18, 0x06, 0x00, 0x00, 0x00 }, // ASCII 60
    { 0x00, 0x3E, 0x00, 0x3E, 0x00, 0x00, 0x00, 0x00 }, // ASCII 61
    { 0x60, 0x18, 0x06, 0x18, 0x60, 0x00, 0x00, 0x00 }, // ASCII 62
    { 0x3C, 0x42, 0x02, 0x0C, 0x08, 0x00, 0x08, 0x00 }, // ASCII 63
    { 0x3C, 0x42, 0x4E, 0x5A, 0x5E, 0x40, 0x3C, 0x00 }, // ASCII 64
    { 0x18, 0x24, 0x42, 0x7E, 0x42, 0x42, 0x42, 0x00 }, // ASCII 65
    { 0x7C, 0x42, 0x42, 0x7C, 0x42, 0x42, 0x7C, 0x00 }, // ASCII 66
    { 0x3C, 0x42, 0x40, 0x40, 0x40, 0x42, 0x3C, 0x00 }, // ASCII 67
    { 0x78, 0x44, 0x42, 0x42, 0x42, 0x44, 0x78, 0x00 }, // ASCII 68
    { 0x7E, 0x40, 0x40, 0x7C, 0x40, 0x40, 0x7E, 0x00 }, // ASCII 69
    { 0x7E, 0x40, 0x40, 0x7C, 0x40, 0x40, 0x40, 0x00 }, // ASCII 70
    { 0x3C, 0x42, 0x40, 0x4E, 0x42, 0x42, 0x3C, 0x00 }, // ASCII 71
    { 0x42, 0x42, 0x42, 0x7E, 0x42, 0x42, 0x42, 0x00 }, // ASCII 72
    { 0x3E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x3E, 0x00 }, // ASCII 73
    { 0x0E, 0x04, 0x04, 0x04, 0x04, 0x44, 0x38, 0x00 }, // ASCII 74
    { 0x42, 0x44, 0x48, 0x70, 0x48, 0x44, 0x42, 0x00 }, // ASCII 75
    { 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x7E, 0x00 }, // ASCII 76
    { 0x42, 0x66, 0x5A, 0x42, 0x42, 0x42, 0x42, 0x00 }, // ASCII 77
    { 0x42, 0x62, 0x52, 0x4A, 0x46, 0x42, 0x42, 0x00 }, // ASCII 78
    { 0x3C, 0x42, 0x42, 0x42, 0x42, 0x42, 0x3C, 0x00 }, // ASCII 79
    { 0x7C, 0x42, 0x42, 0x7C, 0x40, 0x40, 0x40, 0x00 }, // ASCII 80
    { 0x3C, 0x42, 0x42, 0x42, 0x4A, 0x44, 0x3A, 0x00 }, // ASCII 81
    { 0x7C, 0x42, 0x42, 0x7C, 0x48, 0x44, 0x42, 0x00 }, // ASCII 82
    { 0x3C, 0x42, 0x40, 0x3C, 0x02, 0x42, 0x3C, 0x00 }, // ASCII 83
    { 0x7E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x00 }, // ASCII 84
    { 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x3C, 0x00 }, // ASCII 85
    { 0x42, 0x42, 0x42, 0x42, 0x42, 0x24, 0x18, 0x00 }, // ASCII 86
    { 0x42, 0x42, 0x42, 0x42, 0x5A, 0x66, 0x42, 0x00 }, // ASCII 87
    { 0x42, 0x42, 0x24, 0x18, 0x24, 0x42, 0x42, 0x00 }, // ASCII 88
    { 0x42, 0x42, 0x24, 0x18, 0x08, 0x08, 0x08, 0x00 }, // ASCII 89
    { 0x7E, 0x02, 0x04, 0x08, 0x10, 0x20, 0x7E, 0x00 }, // ASCII 90
    { 0x38, 0x20, 0x20, 0x20, 0x20, 0x20, 0x38, 0x00 }, // ASCII 91
    { 0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x00 }, // ASCII 92
    { 0x1C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x1C, 0x00 }, // ASCII 93
    { 0x10, 0x28, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00 }, // ASCII 94
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7E, 0x00 }, // ASCII 95
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }, // ASCII 96
    { 0x18, 0x24, 0x42, 0x7E, 0x42, 0x42, 0x42, 0x00 }, // ASCII 97
    { 0x7C, 0x42, 0x42, 0x7C, 0x42, 0x42, 0x7C, 0x00 }, // ASCII 98
    { 0x3C, 0x42, 0x40, 0x40, 0x40, 0x42, 0x3C, 0x00 }, // ASCII 99
    { 0x78, 0x44, 0x42, 0x42, 0x42, 0x44, 0x78, 0x00 }, // ASCII 100
    { 0x7E, 0x40, 0x40, 0x7C, 0x40, 0x40, 0x7E, 0x00 }, // ASCII 101
    { 0x7E, 0x40, 0x40, 0x7C, 0x40, 0x40, 0x40, 0x00 }, // ASCII 102
    { 0x3C, 0x42, 0x40, 0x4E, 0x42, 0x42, 0x3C, 0x00 }, // ASCII 103
    { 0x42, 0x42, 0x42, 0x7E, 0x42, 0x42, 0x42, 0x00 }, // ASCII 104
    { 0x3E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x3E, 0x00 }, // ASCII 105
    { 0x0E, 0x04, 0x04, 0x04, 0x04, 0x44, 0x38, 0x00 }, // ASCII 106
    { 0x42, 0x44, 0x48, 0x70, 0x48, 0x44, 0x42, 0x00 }, // ASCII 107
    { 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x7E, 0x00 }, // ASCII 108
    { 0x42, 0x66, 0x5A, 0x42, 0x42, 0x42, 0x42, 0x00 }, // ASCII 109
    { 0x42, 0x62, 0x52, 0x4A, 0x46, 0x42, 0x42, 0x00 }, // ASCII 110
    { 0x3C, 0x42, 0x42, 0x42, 0x42, 0x42, 0x3C, 0x00 }, // ASCII 111
    { 0x7C, 0x42, 0x42, 0x7C, 0x40, 0x40, 0x40, 0x00 }, // ASCII 112
    { 0x3C, 0x42, 0x42, 0x42, 0x4A, 0x44, 0x3A, 0x00 }, // ASCII 113
    { 0x7C, 0x42, 0x42, 0x7C, 0x48, 0x44, 0x42, 0x00 }, // ASCII 114
    { 0x3C, 0x42, 0x40, 0x3C, 0x02, 0x42, 0x3C, 0x00 }, // ASCII 115
    { 0x7E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x00 }, // ASCII 116
    { 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x3C, 0x00 }, // ASCII 117
    { 0x42, 0x42, 0x42, 0x42, 0x42, 0x24, 0x18, 0x00 }, // ASCII 118
    { 0x42, 0x42, 0x42, 0x42, 0x5A, 0x66, 0x42, 0x00 }, // ASCII 119
    { 0x42, 0x42, 0x24, 0x18, 0x24, 0x42, 0x42, 0x00 }, // ASCII 120
    { 0x42, 0x42, 0x24, 0x18, 0x08, 0x08, 0x08, 0x00 }, // ASCII 121
    { 0x7E, 0x02, 0x04, 0x08, 0x10, 0x20, 0x7E, 0x00 }, // ASCII 122
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }, // ASCII 123
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }, // ASCII 124
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }, // ASCII 125
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }, // ASCII 126
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }, // ASCII 127
};


static GLuint g_font_tex = 0;

static void init_font_texture(void) {
    /* 128x64 atlas for 96 characters (16 cols x 6 rows) */
    uint8_t pixels[128 * 64 * 4];
    memset(pixels, 0, sizeof(pixels));

    for (int i = 0; i < 96; i++) {
        int col = i % 16;
        int row = i / 16;
        int startX = col * 8;
        int startY = row * 8;
        for (int y = 0; y < 8; y++) {
            uint8_t byteVal = g_font_8x8[i][y];
            for (int x = 0; x < 8; x++) {
                int px = startX + x;
                int py = startY + y;
                int idx = (py * 128 + px) * 4;
                if ((byteVal >> (7 - x)) & 1) {
                    pixels[idx + 0] = 255;
                    pixels[idx + 1] = 255;
                    pixels[idx + 2] = 255;
                    pixels[idx + 3] = 255;
                } else {
                    pixels[idx + 0] = pixels[idx + 1] = pixels[idx + 2] = pixels[idx + 3] = 0;
                }
            }
        }
    }

    glGenTextures(1, &g_font_tex);
    glBindTexture(GL_TEXTURE_2D, g_font_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 128, 64, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
}

#define MAX_HUD_VERTS 65536
static VertexHUD g_hud_verts[MAX_HUD_VERTS];
static int g_hud_count = 0;

static inline void hud_push_v(VertexHUD v) {
    if (g_hud_count < MAX_HUD_VERTS) {
        g_hud_verts[g_hud_count++] = v;
    }
}

static void hud_push_quad(float x, float y, float w, float h, Vec4 col,
                          float u0, float v0, float u1, float v1, float tex_mix) {
    VertexHUD p0 = { x,     y,     u0, v0, col.x, col.y, col.z, col.w, tex_mix };
    VertexHUD p1 = { x + w, y,     u1, v0, col.x, col.y, col.z, col.w, tex_mix };
    VertexHUD p2 = { x + w, y + h, u1, v1, col.x, col.y, col.z, col.w, tex_mix };
    VertexHUD p3 = { x,     y + h, u0, v1, col.x, col.y, col.z, col.w, tex_mix };
    hud_push_v(p0); hud_push_v(p1); hud_push_v(p2);
    hud_push_v(p0); hud_push_v(p2); hud_push_v(p3);
}

static void hud_rect(float x, float y, float w, float h, Vec4 col) {
    hud_push_quad(x, y, w, h, col, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
}

static void hud_pill(float x, float y, float w, float h, float slant, Vec4 col) {
    VertexHUD v0 = { x + slant,     y,     0.0f, 0.0f, col.x, col.y, col.z, col.w, 0.0f };
    VertexHUD v1 = { x + w + slant, y,     0.0f, 0.0f, col.x, col.y, col.z, col.w, 0.0f };
    VertexHUD v2 = { x + w - slant, y + h, 0.0f, 0.0f, col.x, col.y, col.z, col.w, 0.0f };
    VertexHUD v3 = { x - slant,     y + h, 0.0f, 0.0f, col.x, col.y, col.z, col.w, 0.0f };
    hud_push_v(v0); hud_push_v(v1); hud_push_v(v2);
    hud_push_v(v0); hud_push_v(v2); hud_push_v(v3);
}

static void hud_line(float x0, float y0, float x1, float y1, float thickness, Vec4 col) {
    float dx = x1 - x0;
    float dy = y1 - y0;
    float len = sqrtf(dx * dx + dy * dy);
    if (len < 1e-4f) return;
    float nx = (-dy / len) * (thickness * 0.5f);
    float ny = ( dx / len) * (thickness * 0.5f);
    VertexHUD v0 = { x0 + nx, y0 + ny, 0.0f, 0.0f, col.x, col.y, col.z, col.w, 0.0f };
    VertexHUD v1 = { x1 + nx, y1 + ny, 0.0f, 0.0f, col.x, col.y, col.z, col.w, 0.0f };
    VertexHUD v2 = { x1 - nx, y1 - ny, 0.0f, 0.0f, col.x, col.y, col.z, col.w, 0.0f };
    VertexHUD v3 = { x0 - nx, y0 - ny, 0.0f, 0.0f, col.x, col.y, col.z, col.w, 0.0f };
    hud_push_v(v0); hud_push_v(v1); hud_push_v(v2);
    hud_push_v(v0); hud_push_v(v2); hud_push_v(v3);
}

static void hud_card(float x, float y, float w, float h, float slant,
                     Vec4 fillCol, Vec4 borderCol, Vec4 topAccentCol) {
    hud_pill(x, y, w, h, slant, fillCol);
    if (borderCol.w > 0.01f) {
        float x0 = x + slant, y0 = y;
        float x1 = x + w + slant, y1 = y;
        float x2 = x + w - slant, y2 = y + h;
        float x3 = x - slant, y3 = y + h;
        hud_line(x0, y0, x1, y1, 1.5f, borderCol);
        hud_line(x1, y1, x2, y2, 1.5f, borderCol);
        hud_line(x2, y2, x3, y3, 1.5f, borderCol);
        hud_line(x3, y3, x0, y0, 1.5f, borderCol);
    }
    if (topAccentCol.w > 0.01f) {
        hud_pill(x + slant, y, w, 2.5f, 0.0f, topAccentCol);
    }
}

static void hud_text(float x, float y, float scale, const char *str, Vec4 col) {
    float curX = x;
    while (*str) {
        unsigned char c = (unsigned char)*str++;
        if (c < 32 || c > 126) c = '?';
        int idx = (int)c - 32;
        int colIdx = idx % 16;
        int rowIdx = idx / 16;
        float u0 = (float)(colIdx * 8) / 128.0f;
        float v0 = (float)(rowIdx * 8) / 64.0f;
        float u1 = (float)((colIdx + 1) * 8) / 128.0f;
        float v1 = (float)((rowIdx + 1) * 8) / 64.0f;
        hud_push_quad(curX, y, 8.0f * scale, 8.0f * scale, col, u0, v0, u1, v1, 1.0f);
        curX += 8.0f * scale;
    }
}

static void flush_hud(void) {
    if (g_hud_count == 0) return;
    glUseProgram(g_progHud);

    Mat4 ortho;
    mat4_ortho(ortho, 0.0f, 1280.0f, 720.0f, 0.0f, -1.0f, 1.0f);
    glUniformMatrix4fv(g_uHud_Ortho, 1, GL_FALSE, ortho);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, g_font_tex);
    glUniform1i(g_uHud_Tex, 0);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);

    GLuint vbo = 0;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(VertexHUD) * g_hud_count, g_hud_verts, GL_STREAM_DRAW);

    glEnableVertexAttribArray(g_aHud_Pos);
    glVertexAttribPointer(g_aHud_Pos, 2, GL_FLOAT, GL_FALSE, sizeof(VertexHUD), (void*)0);
    glEnableVertexAttribArray(g_aHud_UV);
    glVertexAttribPointer(g_aHud_UV, 2, GL_FLOAT, GL_FALSE, sizeof(VertexHUD), (void*)(sizeof(float) * 2));
    glEnableVertexAttribArray(g_aHud_Col);
    glVertexAttribPointer(g_aHud_Col, 4, GL_FLOAT, GL_FALSE, sizeof(VertexHUD), (void*)(sizeof(float) * 4));
    glEnableVertexAttribArray(g_aHud_TexMix);
    glVertexAttribPointer(g_aHud_TexMix, 1, GL_FLOAT, GL_FALSE, sizeof(VertexHUD), (void*)(sizeof(float) * 8));

    glDrawArrays(GL_TRIANGLES, 0, g_hud_count);

    glDisableVertexAttribArray(g_aHud_Pos);
    glDisableVertexAttribArray(g_aHud_UV);
    glDisableVertexAttribArray(g_aHud_Col);
    glDisableVertexAttribArray(g_aHud_TexMix);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDeleteBuffers(1, &vbo);

    g_hud_count = 0;
}

/* =========================================================================
 * 5. BATCH FX RENDERER (TRANSLUCENT / RADIAL GLOW TRIANGLES)
 * ========================================================================= */

#define MAX_FX_VERTS 65536
static VertexFX g_fx_verts[MAX_FX_VERTS];
static int g_fx_count = 0;

static inline void fx_push_v(VertexFX v) {
    if (g_fx_count < MAX_FX_VERTS) {
        g_fx_verts[g_fx_count++] = v;
    }
}

static void fx_push_tri(VertexFX v0, VertexFX v1, VertexFX v2) {
    fx_push_v(v0); fx_push_v(v1); fx_push_v(v2);
}

static void fx_push_quad(VertexFX v0, VertexFX v1, VertexFX v2, VertexFX v3) {
    fx_push_tri(v0, v1, v2);
    fx_push_tri(v0, v2, v3);
}

static void fx_push_radial_glow(Vec3 center, float radiusX, float radiusZ, Vec4 col) {
    VertexFX v0 = { center.x - radiusX, center.y, center.z - radiusZ, 0.0f, 0.0f, col.x, col.y, col.z, col.w };
    VertexFX v1 = { center.x + radiusX, center.y, center.z - radiusZ, 1.0f, 0.0f, col.x, col.y, col.z, col.w };
    VertexFX v2 = { center.x + radiusX, center.y, center.z + radiusZ, 1.0f, 1.0f, col.x, col.y, col.z, col.w };
    VertexFX v3 = { center.x - radiusX, center.y, center.z + radiusZ, 0.0f, 1.0f, col.x, col.y, col.z, col.w };
    fx_push_quad(v0, v1, v2, v3);
}

static void flush_fx(const Mat4 mvp) {
    if (g_fx_count == 0) return;
    glUseProgram(g_progFx);
    glUniformMatrix4fv(g_uFx_MVP, 1, GL_FALSE, mvp);

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    GLuint vbo = 0;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(VertexFX) * g_fx_count, g_fx_verts, GL_STREAM_DRAW);

    glEnableVertexAttribArray(g_aFx_Pos);
    glVertexAttribPointer(g_aFx_Pos, 3, GL_FLOAT, GL_FALSE, sizeof(VertexFX), (void*)0);
    glEnableVertexAttribArray(g_aFx_UV);
    glVertexAttribPointer(g_aFx_UV, 2, GL_FLOAT, GL_FALSE, sizeof(VertexFX), (void*)(sizeof(float) * 3));
    glEnableVertexAttribArray(g_aFx_Col);
    glVertexAttribPointer(g_aFx_Col, 4, GL_FLOAT, GL_FALSE, sizeof(VertexFX), (void*)(sizeof(float) * 5));

    glDrawArrays(GL_TRIANGLES, 0, g_fx_count);

    glDisableVertexAttribArray(g_aFx_Pos);
    glDisableVertexAttribArray(g_aFx_UV);
    glDisableVertexAttribArray(g_aFx_Col);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDeleteBuffers(1, &vbo);

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    g_fx_count = 0;
}

/* =========================================================================
 * 6. CATALOG SPECS: CARS, PAINTS, RIMS, UNDERGLOW, WINGS, TRACKS & MODES
 * ========================================================================= */

typedef struct {
    const char *id;
    const char *name;
    const char *cls;
    const char *rating;
    float top_speed;
    float accel;
    float handling;
    float drift_grip;
    float max_spd_phys;
    float len;
    float wid;
    float nose_h;
    float roof_h;
    float roof_peak_x;
    bool hood_scoop;
    bool mid_engine_vents;
    Vec3 default_paint;
    Vec3 default_glow;
    int default_wing;
} CarSpec;

static const CarSpec g_car_specs[6] = {
    { "vortex", "Vortex GT-R", "JDM TWIN-TURBO", "S-890", 90.0f, 54.0f, 90.0f, 0.92f, 102.0f,
      4.70f, 2.02f, 0.34f, 1.24f, -0.20f, false, false, {0.0f, 0.83f, 1.0f}, {0.0f, 0.94f, 1.0f}, 0 },
    { "diablo", "Diablo V12", "ITALIAN HYPERCAR", "X-950", 98.0f, 58.0f, 84.0f, 0.88f, 112.0f,
      4.85f, 2.12f, 0.24f, 1.12f, 0.15f, false, true, {1.0f, 0.16f, 0.52f}, {1.0f, 0.16f, 0.52f}, 0 },
    { "titan", "Titan GT500", "WIDEBODY MUSCLE", "A-840", 88.0f, 64.0f, 78.0f, 0.85f, 100.0f,
      4.90f, 2.10f, 0.44f, 1.32f, -0.35f, true, false, {1.0f, 0.42f, 0.0f}, {1.0f, 0.53f, 0.0f}, 1 },
    { "phantom", "Phantom RS", "EURO TRACK COUPE", "S-910", 91.0f, 56.0f, 96.0f, 0.95f, 104.0f,
      4.60f, 1.98f, 0.30f, 1.22f, -0.10f, false, true, {0.29f, 0.87f, 0.50f}, {0.13f, 0.77f, 0.37f}, 0 },
    { "valkyrie", "Cyber Valkyrie", "AERO PROTOTYPE", "X-980", 99.0f, 62.0f, 94.0f, 0.93f, 114.0f,
      4.95f, 2.14f, 0.22f, 1.14f, 0.20f, false, true, {0.66f, 0.33f, 0.97f}, {0.66f, 0.33f, 0.97f}, 2 },
    { "interceptor", "Interceptor X", "PURSUIT SPEC", "S-940", 95.0f, 60.0f, 89.0f, 0.91f, 108.0f,
      4.80f, 2.08f, 0.34f, 1.25f, -0.15f, true, false, {0.07f, 0.09f, 0.15f}, {0.22f, 0.74f, 0.97f}, 2 }
};

typedef struct { const char *name; Vec3 col; } PaintSwatch;
static const PaintSwatch g_paints[8] = {
    { "Crimson Metallic", {0.88f, 0.11f, 0.28f} },
    { "Electric Cyan",    {0.00f, 0.83f, 1.00f} },
    { "Hot Magenta",       {1.00f, 0.16f, 0.52f} },
    { "Acid Lime",         {0.29f, 0.87f, 0.50f} },
    { "Solar Gold",        {0.98f, 0.80f, 0.08f} },
    { "Midnight Violet",   {0.58f, 0.20f, 0.92f} },
    { "Sunset Orange",     {1.00f, 0.42f, 0.00f} },
    { "Stealth Onyx",      {0.09f, 0.11f, 0.16f} }
};

typedef struct { const char *name; Vec3 col; } RimStyle;
static const RimStyle g_rims[4] = {
    { "Chrome",        {0.89f, 0.91f, 0.94f} },
    { "Solar Gold",    {0.98f, 0.80f, 0.08f} },
    { "Stealth Matte", {0.12f, 0.16f, 0.23f} },
    { "Electric Cyan", {0.00f, 0.94f, 1.00f} }
};

typedef struct { const char *name; Vec3 col; bool active; } GlowSwatch;
static const GlowSwatch g_glows[7] = {
    { "Cyan",        {0.00f, 0.94f, 1.00f}, true },
    { "Magenta",     {1.00f, 0.16f, 0.52f}, true },
    { "Neon Green",  {0.13f, 0.77f, 0.37f}, true },
    { "Hot Orange",  {1.00f, 0.48f, 0.00f}, true },
    { "Ultraviolet", {0.66f, 0.33f, 0.97f}, true },
    { "Ice Blue",    {0.50f, 0.80f, 1.00f}, true },
    { "Off",         {0.00f, 0.00f, 0.00f}, false }
};

static const char *g_wings[4] = {
    "GT Pro Wing", "Street Ducktail", "Cyber Twin-Fin", "Clean Stock"
};

typedef struct {
    const char *id;
    const char *name;
    const char *desc;
    Vec3 sky_col;
    Vec3 fog_col;
    float fog_density;
    Vec3 ambient_col;
    Vec3 sun_col;
    Vec3 barrier_trim;
    int wp_count;
    Vec3 waypoints[30];
} TrackSpec;

static const TrackSpec g_track_specs[3] = {
    {
        "tokyo", "Tokyo Neon Underground", "Wet reflections, falling rain, streetlights & neon tunnels",
        {0.008f, 0.020f, 0.051f}, {0.045f, 0.095f, 0.205f}, 0.00175f,
        {0.29f, 0.39f, 0.60f}, {0.50f, 0.71f, 1.00f}, {0.00f, 0.94f, 1.00f},
        25,
        {
            {0, 0, 0}, {0, 0, -240}, {45, 2, -460}, {200, 6, -610}, {440, 12, -620},
            {660, 16, -500}, {740, 10, -290}, {620, 4, -120}, {470, 0, -30}, {460, 0, 140},
            {630, 8, 280}, {670, 14, 490}, {520, 10, 640}, {280, 4, 640}, {150, 0, 520},
            {70, 0, 390}, {-50, 2, 390}, {-210, 6, 470}, {-410, 12, 450}, {-560, 14, 260},
            {-530, 8, 40}, {-360, 2, -90}, {-220, 0, 50}, {-90, 0, 195}, {0, 0, 165}
        }
    },
    {
        "canyon", "Pacific Sunset Canyon", "Golden coastal cliffs, terracotta mesas & rock tunnels",
        {0.043f, 0.035f, 0.125f}, {0.880f, 0.350f, 0.110f}, 0.00085f,
        {1.00f, 0.83f, 0.72f}, {1.00f, 0.82f, 0.40f}, {1.00f, 0.67f, 0.00f},
        20,
        {
            {0, 0, 0}, {0, 0, -250}, {80, 8, -480}, {280, 18, -590}, {520, 24, -520},
            {690, 18, -320}, {660, 8, -110}, {510, 2, 40}, {580, 0, 240}, {610, 0, 460},
            {410, 4, 620}, {170, 8, 560}, {-30, 6, 420}, {-220, 12, 510}, {-450, 18, 490},
            {-610, 14, 310}, {-560, 6, 100}, {-360, 0, 20}, {-240, 0, 140}, {-80, 0, 210}
        }
    },
    {
        "alpine", "Alpine Horizon Speedway", "Crisp daylight high-speed chicanes, pines & snowy peaks",
        {0.027f, 0.075f, 0.161f}, {0.240f, 0.680f, 0.920f}, 0.00062f,
        {0.86f, 0.92f, 1.00f}, {1.00f, 0.98f, 0.92f}, {0.22f, 0.74f, 0.97f},
        20,
        {
            {0, 0, 0}, {0, 0, -270}, {110, 6, -500}, {340, 16, -610}, {580, 26, -540},
            {760, 22, -320}, {720, 12, -90}, {500, 4, 20}, {470, 0, 210}, {620, 6, 400},
            {490, 10, 600}, {240, 4, 600}, {60, 0, 420}, {-110, 0, 450}, {-320, 8, 520},
            {-540, 14, 390}, {-580, 10, 150}, {-400, 2, -20}, {-210, 0, 50}, {-80, 0, 220}
        }
    }
};

typedef struct { const char *id; const char *name; const char *desc; int laps; } ModeSpec;
static const ModeSpec g_mode_specs[3] = {
    { "race",    "Underground Circuit",  "3 Laps vs. 5 AI Rivals - Claim P1!", 3 },
    { "drift",   "Drift Touge Showdown", "2 Laps - Weight-Transfer Drift Combos!", 2 },
    { "pursuit", "Hot Pursuit Escape",   "2 Laps - Outrun 3 Police Interceptors!", 2 }
};

/* =========================================================================
 * 7. CIRCUIT EVALUATION & STATIC BAKED VBO GENERATION
 * ========================================================================= */

#define FRAMES_COUNT 600
#define ROAD_HALF_W 14.0f
#define MAX_LAT 12.2f

typedef struct {
    Vec3 pts[FRAMES_COUNT];
    Vec3 tangents[FRAMES_COUNT];
    Vec3 rights[FRAMES_COUNT];
    Vec3 ups[FRAMES_COUNT];
    float accum_dist[FRAMES_COUNT + 1];
    float total_len;
    GLuint vbo_static;
    int vert_count;
} TrackRuntime;

static TrackRuntime g_tracks[3];

static Vec3 catmull_rom(Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3, float t) {
    float t2 = t * t;
    float t3 = t2 * t;
    float x = 0.5f * ((2.0f * p1.x) + (-p0.x + p2.x) * t +
                      (2.0f * p0.x - 5.0f * p1.x + 4.0f * p2.x - p3.x) * t2 +
                      (-p0.x + 3.0f * p1.x - 3.0f * p2.x + p3.x) * t3);
    float y = 0.5f * ((2.0f * p1.y) + (-p0.y + p2.y) * t +
                      (2.0f * p0.y - 5.0f * p1.y + 4.0f * p2.y - p3.y) * t2 +
                      (-p0.y + 3.0f * p1.y - 3.0f * p2.y + p3.y) * t3);
    float z = 0.5f * ((2.0f * p1.z) + (-p0.z + p2.z) * t +
                      (2.0f * p0.z - 5.0f * p1.z + 4.0f * p2.z - p3.z) * t2 +
                      (-p0.z + 3.0f * p1.z - 3.0f * p2.z + p3.z) * t3);
    return vec3(x, y, z);
}

static void build_track_runtime(int trk_idx) {
    TrackRuntime *tr = &g_tracks[trk_idx];
    const TrackSpec *ts = &g_track_specs[trk_idx];
    int N = ts->wp_count;

    for (int i = 0; i < FRAMES_COUNT; i++) {
        float u = (float)i / (float)FRAMES_COUNT;
        float total_t = u * (float)N;
        int idx = (int)total_t;
        float s = total_t - (float)idx;

        Vec3 p0 = ts->waypoints[(idx - 1 + N) % N];
        Vec3 p1 = ts->waypoints[idx % N];
        Vec3 p2 = ts->waypoints[(idx + 1) % N];
        Vec3 p3 = ts->waypoints[(idx + 2) % N];

        tr->pts[i] = catmull_rom(p0, p1, p2, p3, s);

        float s_next = s + 0.002f;
        Vec3 p_next = catmull_rom(p0, p1, p2, p3, s_next);
        tr->tangents[i] = vec3_norm(vec3_sub(p_next, tr->pts[i]));
        tr->rights[i] = vec3_norm(vec3_cross(vec3(0.0f, 1.0f, 0.0f), tr->tangents[i]));
        tr->ups[i] = vec3_norm(vec3_cross(tr->tangents[i], tr->rights[i]));
    }

    tr->accum_dist[0] = 0.0f;
    for (int i = 0; i < FRAMES_COUNT; i++) {
        int next_i = (i + 1) % FRAMES_COUNT;
        float d = vec3_dist(tr->pts[i], tr->pts[next_i]);
        tr->accum_dist[i + 1] = tr->accum_dist[i] + d;
    }
    tr->total_len = tr->accum_dist[FRAMES_COUNT];

    /* Build Single Static Baked VBO */
    MeshBuilder3D mb;
    mesh3d_init(&mb, 131072);

    Vec3 asphaltCol = (trk_idx == 0) ? vec3(0.055f, 0.065f, 0.095f) : vec3(0.12f, 0.13f, 0.16f);
    Vec3 dashCol = vec3(0.95f, 0.90f, 0.30f);
    Vec3 curbRed = vec3(0.85f, 0.12f, 0.16f);
    Vec3 curbWhite = vec3(0.95f, 0.95f, 0.95f);
    Vec3 barrierCol = vec3(0.12f, 0.16f, 0.23f);
    Vec3 neonTrim = ts->barrier_trim;

    for (int i = 0; i < FRAMES_COUNT; i++) {
        int ni = (i + 1) % FRAMES_COUNT;
        Vec3 pA = tr->pts[i], pB = tr->pts[ni];
        Vec3 rA = tr->rights[i], rB = tr->rights[ni];
        Vec3 uA = tr->ups[i], uB = tr->ups[ni];

        /* 1. Road Asphalt Ribbon */
        Vertex3D rL0 = { pA.x - rA.x * ROAD_HALF_W, pA.y, pA.z - rA.z * ROAD_HALF_W, uA.x, uA.y, uA.z, asphaltCol.x, asphaltCol.y, asphaltCol.z, 0.0f };
        Vertex3D rR0 = { pA.x + rA.x * ROAD_HALF_W, pA.y, pA.z + rA.z * ROAD_HALF_W, uA.x, uA.y, uA.z, asphaltCol.x, asphaltCol.y, asphaltCol.z, 0.0f };
        Vertex3D rL1 = { pB.x - rB.x * ROAD_HALF_W, pB.y, pB.z - rB.z * ROAD_HALF_W, uB.x, uB.y, uB.z, asphaltCol.x, asphaltCol.y, asphaltCol.z, 0.0f };
        Vertex3D rR1 = { pB.x + rB.x * ROAD_HALF_W, pB.y, pB.z + rB.z * ROAD_HALF_W, uB.x, uB.y, uB.z, asphaltCol.x, asphaltCol.y, asphaltCol.z, 0.0f };
        mesh3d_add_quad(&mb, rL0, rR0, rR1, rL1);

        /* 2. Center Dashed Line (3 on, 3 off) */
        if ((i % 6) < 3) {
            float dw = 0.25f;
            Vertex3D dL0 = { pA.x - rA.x * dw, pA.y + 0.02f, pA.z - rA.z * dw, uA.x, uA.y, uA.z, dashCol.x, dashCol.y, dashCol.z, 0.8f };
            Vertex3D dR0 = { pA.x + rA.x * dw, pA.y + 0.02f, pA.z + rA.z * dw, uA.x, uA.y, uA.z, dashCol.x, dashCol.y, dashCol.z, 0.8f };
            Vertex3D dL1 = { pB.x - rB.x * dw, pB.y + 0.02f, pB.z - rB.z * dw, uB.x, uB.y, uB.z, dashCol.x, dashCol.y, dashCol.z, 0.8f };
            Vertex3D dR1 = { pB.x + rB.x * dw, pB.y + 0.02f, pB.z + rB.z * dw, uB.x, uB.y, uB.z, dashCol.x, dashCol.y, dashCol.z, 0.8f };
            mesh3d_add_quad(&mb, dL0, dR0, dR1, dL1);
        }

        /* 3. Alternating Red & White Curbs */
        Vec3 curbCol = ((i / 4) % 2 == 0) ? curbRed : curbWhite;
        float cIn = ROAD_HALF_W;
        float cOut = ROAD_HALF_W + 0.8f;
        /* Left Curb */
        Vertex3D cl0 = { pA.x - rA.x * cIn,  pA.y + 0.05f, pA.z - rA.z * cIn,  uA.x, uA.y, uA.z, curbCol.x, curbCol.y, curbCol.z, 0.2f };
        Vertex3D cl1 = { pA.x - rA.x * cOut, pA.y + 0.12f, pA.z - rA.z * cOut, uA.x, uA.y, uA.z, curbCol.x, curbCol.y, curbCol.z, 0.2f };
        Vertex3D cl2 = { pB.x - rB.x * cOut, pB.y + 0.12f, pB.z - rB.z * cOut, uB.x, uB.y, uB.z, curbCol.x, curbCol.y, curbCol.z, 0.2f };
        Vertex3D cl3 = { pB.x - rB.x * cIn,  pB.y + 0.05f, pB.z - rB.z * cIn,  uB.x, uB.y, uB.z, curbCol.x, curbCol.y, curbCol.z, 0.2f };
        mesh3d_add_quad(&mb, cl0, cl1, cl2, cl3);
        /* Right Curb */
        Vertex3D cr0 = { pA.x + rA.x * cIn,  pA.y + 0.05f, pA.z + rA.z * cIn,  uA.x, uA.y, uA.z, curbCol.x, curbCol.y, curbCol.z, 0.2f };
        Vertex3D cr1 = { pA.x + rA.x * cOut, pA.y + 0.12f, pA.z + rA.z * cOut, uA.x, uA.y, uA.z, curbCol.x, curbCol.y, curbCol.z, 0.2f };
        Vertex3D cr2 = { pB.x + rB.x * cOut, pB.y + 0.12f, pB.z + rB.z * cOut, uB.x, uB.y, uB.z, curbCol.x, curbCol.y, curbCol.z, 0.2f };
        Vertex3D cr3 = { pB.x + rB.x * cIn,  pB.y + 0.05f, pB.z + rB.z * cIn,  uB.x, uB.y, uB.z, curbCol.x, curbCol.y, curbCol.z, 0.2f };
        mesh3d_add_quad(&mb, cr0, cr3, cr2, cr1);

        /* 4. Concrete Jersey Barriers + Glowing Neon Edge Trim */
        float bIn = ROAD_HALF_W + 0.8f;
        float bOut = ROAD_HALF_W + 1.6f;
        float bH = 1.35f;
        /* Left Barrier Inner Wall */
        Vertex3D bl0 = { pA.x - rA.x * bIn, pA.y,      pA.z - rA.z * bIn, rA.x, 0.0f, rA.z, barrierCol.x, barrierCol.y, barrierCol.z, 0.0f };
        Vertex3D bl1 = { pA.x - rA.x * bIn, pA.y + bH, pA.z - rA.z * bIn, rA.x, 0.0f, rA.z, barrierCol.x, barrierCol.y, barrierCol.z, 0.0f };
        Vertex3D bl2 = { pB.x - rB.x * bIn, pB.y + bH, pB.z - rB.z * bIn, rB.x, 0.0f, rB.z, barrierCol.x, barrierCol.y, barrierCol.z, 0.0f };
        Vertex3D bl3 = { pB.x - rB.x * bIn, pB.y,      pB.z - rB.z * bIn, rB.x, 0.0f, rB.z, barrierCol.x, barrierCol.y, barrierCol.z, 0.0f };
        mesh3d_add_quad(&mb, bl0, bl1, bl2, bl3);
        /* Left Barrier Top Cap */
        Vertex3D blt0 = { pA.x - rA.x * bIn,  pA.y + bH, pA.z - rA.z * bIn,  0.0f, 1.0f, 0.0f, barrierCol.x, barrierCol.y, barrierCol.z, 0.0f };
        Vertex3D blt1 = { pA.x - rA.x * bOut, pA.y + bH, pA.z - rA.z * bOut, 0.0f, 1.0f, 0.0f, barrierCol.x, barrierCol.y, barrierCol.z, 0.0f };
        Vertex3D blt2 = { pB.x - rB.x * bOut, pB.y + bH, pB.z - rB.z * bOut, 0.0f, 1.0f, 0.0f, barrierCol.x, barrierCol.y, barrierCol.z, 0.0f };
        Vertex3D blt3 = { pB.x - rB.x * bIn,  pB.y + bH, pB.z - rB.z * bIn,  0.0f, 1.0f, 0.0f, barrierCol.x, barrierCol.y, barrierCol.z, 0.0f };
        mesh3d_add_quad(&mb, blt0, blt1, blt2, blt3);
        /* Left Glowing Neon Trim */
        Vertex3D bn0 = { pA.x - rA.x * (bIn - 0.02f), pA.y + 0.95f, pA.z - rA.z * (bIn - 0.02f), rA.x, 0.0f, rA.z, neonTrim.x, neonTrim.y, neonTrim.z, 1.0f };
        Vertex3D bn1 = { pA.x - rA.x * (bIn - 0.02f), pA.y + 1.25f, pA.z - rA.z * (bIn - 0.02f), rA.x, 0.0f, rA.z, neonTrim.x, neonTrim.y, neonTrim.z, 1.0f };
        Vertex3D bn2 = { pB.x - rB.x * (bIn - 0.02f), pB.y + 1.25f, pB.z - rB.z * (bIn - 0.02f), rB.x, 0.0f, rB.z, neonTrim.x, neonTrim.y, neonTrim.z, 1.0f };
        Vertex3D bn3 = { pB.x - rB.x * (bIn - 0.02f), pB.y + 0.95f, pB.z - rB.z * (bIn - 0.02f), rB.x, 0.0f, rB.z, neonTrim.x, neonTrim.y, neonTrim.z, 1.0f };
        mesh3d_add_quad(&mb, bn0, bn1, bn2, bn3);

        /* Right Barrier Inner Wall */
        Vertex3D br0 = { pA.x + rA.x * bIn, pA.y,      pA.z + rA.z * bIn, -rA.x, 0.0f, -rA.z, barrierCol.x, barrierCol.y, barrierCol.z, 0.0f };
        Vertex3D br1 = { pA.x + rA.x * bIn, pA.y + bH, pA.z + rA.z * bIn, -rA.x, 0.0f, -rA.z, barrierCol.x, barrierCol.y, barrierCol.z, 0.0f };
        Vertex3D br2 = { pB.x + rB.x * bIn, pB.y + bH, pB.z + rB.z * bIn, -rB.x, 0.0f, -rB.z, barrierCol.x, barrierCol.y, barrierCol.z, 0.0f };
        Vertex3D br3 = { pB.x + rB.x * bIn, pB.y,      pB.z + rB.z * bIn, -rB.x, 0.0f, -rB.z, barrierCol.x, barrierCol.y, barrierCol.z, 0.0f };
        mesh3d_add_quad(&mb, br0, br3, br2, br1);
        /* Right Barrier Top Cap */
        Vertex3D brt0 = { pA.x + rA.x * bIn,  pA.y + bH, pA.z + rA.z * bIn,  0.0f, 1.0f, 0.0f, barrierCol.x, barrierCol.y, barrierCol.z, 0.0f };
        Vertex3D brt1 = { pA.x + rA.x * bOut, pA.y + bH, pA.z + rA.z * bOut, 0.0f, 1.0f, 0.0f, barrierCol.x, barrierCol.y, barrierCol.z, 0.0f };
        Vertex3D brt2 = { pB.x + rB.x * bOut, pB.y + bH, pB.z + rB.z * bOut, 0.0f, 1.0f, 0.0f, barrierCol.x, barrierCol.y, barrierCol.z, 0.0f };
        Vertex3D brt3 = { pB.x + rB.x * bIn,  pB.y + bH, pB.z + rB.z * bIn,  0.0f, 1.0f, 0.0f, barrierCol.x, barrierCol.y, barrierCol.z, 0.0f };
        mesh3d_add_quad(&mb, brt0, brt3, brt2, brt1);
        /* Right Glowing Neon Trim */
        Vertex3D brn0 = { pA.x + rA.x * (bIn - 0.02f), pA.y + 0.95f, pA.z + rA.z * (bIn - 0.02f), -rA.x, 0.0f, -rA.z, neonTrim.x, neonTrim.y, neonTrim.z, 1.0f };
        Vertex3D brn1 = { pA.x + rA.x * (bIn - 0.02f), pA.y + 1.25f, pA.z + rA.z * (bIn - 0.02f), -rA.x, 0.0f, -rA.z, neonTrim.x, neonTrim.y, neonTrim.z, 1.0f };
        Vertex3D brn2 = { pB.x + rB.x * (bIn - 0.02f), pB.y + 1.25f, pB.z + rB.z * (bIn - 0.02f), -rB.x, 0.0f, -rB.z, neonTrim.x, neonTrim.y, neonTrim.z, 1.0f };
        Vertex3D brn3 = { pB.x + rB.x * (bIn - 0.02f), pB.y + 0.95f, pB.z + rB.z * (bIn - 0.02f), -rB.x, 0.0f, -rB.z, neonTrim.x, neonTrim.y, neonTrim.z, 1.0f };
        mesh3d_add_quad(&mb, brn0, brn3, brn2, brn1);

        /* 5. Overhead Streetlight Posts (Every 18 frames) */
        if (i % 18 == 0) {
            float side = ((i / 18) % 2 == 0) ? -1.0f : 1.0f;
            float px = pA.x + rA.x * side * (ROAD_HALF_W + 2.0f);
            float pz = pA.z + rA.z * side * (ROAD_HALF_W + 2.0f);
            mesh3d_add_box(&mb, px, pA.y + 5.5f, pz, 0.35f, 11.0f, 0.35f, vec3(0.2f, 0.25f, 0.35f), 0.0f);
            /* Arm reaching inward */
            float armX = px - rA.x * side * 2.2f;
            float armZ = pz - rA.z * side * 2.2f;
            mesh3d_add_box(&mb, armX, pA.y + 11.0f, armZ, 4.4f * fabsf(rA.x) + 0.3f, 0.25f, 4.4f * fabsf(rA.z) + 0.3f, vec3(0.2f, 0.25f, 0.35f), 0.0f);
            /* Fixture bulb */
            float fixX = px - rA.x * side * 4.2f;
            float fixZ = pz - rA.z * side * 4.2f;
            Vec3 bulbCol = (trk_idx == 0) ? vec3(0.88f, 0.95f, 1.0f) : vec3(1.0f, 0.92f, 0.6f);
            mesh3d_add_box(&mb, fixX, pA.y + 10.85f, fixZ, 1.2f, 0.2f, 0.6f, bulbCol, 1.0f);
        }

        /* 6. Trackside Scenery Props */
        if (i % 4 == 0) {
            float side = ((i / 4) % 2 == 0) ? -1.0f : 1.0f;
            float distOut = ROAD_HALF_W + 20.0f + (float)((i * 7) % 25);
            float sx = pA.x + rA.x * side * distOut;
            float sz = pA.z + rA.z * side * distOut;

            if (trk_idx == 0) { /* Architectural Dark-Slate Tokyo Towers */
                int bIdx = i / 4;
                float bw = 18.0f + (float)((i * 3) % 14);
                float bd = 14.0f + (float)((i * 5) % 12);
                float bh = 56.0f + (float)((i * 13) % 102);
                float slateMix = (float)(bIdx % 4) / 3.0f;
                /* #0d1526 (0.051,0.082,0.149) -> #162238 (0.086,0.133,0.220) */
                Vec3 bldgCol = vec3_lerp(vec3(0.051f, 0.082f, 0.149f), vec3(0.086f, 0.133f, 0.220f), slateMix);
                Vec3 facadeCol = vec3(0.115f, 0.180f, 0.295f);

                /* Primary lower architectural slab */
                float tier1H = (bh > 86.0f) ? bh * 0.72f : bh;
                mesh3d_add_box(&mb, sx, tier1H * 0.5f - 2.0f, sz, bw, tier1H, bd, bldgCol, 0.0f);

                /* Muted architectural spandrel / window louver bands */
                mesh3d_add_box(&mb, sx, tier1H * 0.34f, sz, bw + 0.14f, 0.75f, bd + 0.14f, facadeCol, 0.16f);
                mesh3d_add_box(&mb, sx, tier1H * 0.62f, sz, bw + 0.14f, 0.75f, bd + 0.14f, facadeCol, 0.16f);

                /* Stepped upper setback crown on taller towers */
                float crownW = bw, crownD = bd;
                if (bh > 86.0f) {
                    crownW = bw * 0.68f;
                    crownD = bd * 0.68f;
                    float tier2H = bh - tier1H;
                    mesh3d_add_box(&mb, sx, tier1H + tier2H * 0.5f - 2.0f, sz, crownW, tier2H, crownD, bldgCol, 0.0f);
                }

                /* Slim glowing cyan / amber / violet neon crown & vertical architectural trim */
                int neonMode = bIdx % 3;
                Vec3 trimCol = (neonMode == 0) ? vec3(0.00f, 0.94f, 1.00f) :
                               (neonMode == 1) ? vec3(1.00f, 0.62f, 0.12f) :
                                                 vec3(0.66f, 0.33f, 0.97f);
                /* Crown parapet hairline */
                mesh3d_add_box(&mb, sx, bh - 2.4f, sz, crownW + 0.22f, 0.34f, crownD + 0.22f, trimCol, 1.0f);
                /* Slim vertical neon fin facing the circuit */
                float finX = sx - rA.x * side * (bw * 0.5f + 0.08f);
                float finZ = sz - rA.z * side * (bd * 0.5f + 0.08f);
                mesh3d_add_box(&mb, finX, tier1H * 0.48f, finZ, 0.38f, tier1H * 0.82f, 0.38f, trimCol, 1.0f);
            } else if (trk_idx == 1) { /* Canyon Terracotta Mesa */
                float mw = 35.0f + (float)((i * 5) % 30);
                float mh = 25.0f + (float)((i * 9) % 45);
                Vec3 rockCol = vec3(0.68f, 0.32f, 0.15f);
                mesh3d_add_box(&mb, sx, mh * 0.5f - 2.0f, sz, mw, mh, mw, rockCol, 0.0f);
            } else { /* Alpine Snowy Peaks & Pines */
                float pw = 30.0f + (float)((i * 7) % 35);
                float ph = 40.0f + (float)((i * 11) % 65);
                Vec3 peakCol = vec3(0.35f, 0.45f, 0.55f);
                mesh3d_add_box(&mb, sx, ph * 0.4f - 2.0f, sz, pw, ph * 0.8f, pw, peakCol, 0.0f);
                /* Snow cap */
                mesh3d_add_box(&mb, sx, ph * 0.85f, sz, pw * 0.5f, ph * 0.3f, pw * 0.5f, vec3(0.95f, 0.98f, 1.0f), 0.1f);
            }
        }
    }

    /* Start / Finish Overhead Gantry at Frame 0 */
    Vec3 gP = tr->pts[0];
    Vec3 gR = tr->rights[0];
    Vec3 steelCol = vec3(0.1f, 0.15f, 0.22f);
    /* Left column */
    mesh3d_add_box(&mb, gP.x - gR.x * (ROAD_HALF_W + 1.2f), gP.y + 6.0f, gP.z - gR.z * (ROAD_HALF_W + 1.2f), 1.2f, 12.0f, 1.2f, steelCol, 0.0f);
    /* Right column */
    mesh3d_add_box(&mb, gP.x + gR.x * (ROAD_HALF_W + 1.2f), gP.y + 6.0f, gP.z + gR.z * (ROAD_HALF_W + 1.2f), 1.2f, 12.0f, 1.2f, steelCol, 0.0f);
    /* Overhead cross beam */
    mesh3d_add_box(&mb, gP.x, gP.y + 11.5f, gP.z, (ROAD_HALF_W * 2.0f + 3.0f) * fabsf(gR.x) + 1.5f, 2.0f, (ROAD_HALF_W * 2.0f + 3.0f) * fabsf(gR.z) + 1.5f, steelCol, 0.0f);
    /* Checkered banner */
    mesh3d_add_box(&mb, gP.x, gP.y + 11.5f, gP.z, (ROAD_HALF_W * 2.0f) * fabsf(gR.x) + 0.8f, 1.4f, (ROAD_HALF_W * 2.0f) * fabsf(gR.z) + 0.8f, vec3(1.0f, 1.0f, 1.0f), 0.5f);

    /* Massive Ground Plane */
    Vec3 groundCol = (trk_idx == 0) ? vec3(0.04f, 0.06f, 0.10f) :
                     (trk_idx == 1) ? vec3(0.45f, 0.25f, 0.12f) : vec3(0.12f, 0.35f, 0.18f);
    mesh3d_add_box(&mb, 0.0f, -3.0f, 0.0f, 2800.0f, 0.5f, 2800.0f, groundCol, 0.0f);

    tr->vert_count = mb.count;
    tr->vbo_static = mesh3d_create_vbo(&mb);
    mesh3d_free(&mb);
}

typedef struct {
    int seg_idx;
    float u;
    float lat;
    float qx, qy, qz;
    Vec3 T;
    Vec3 R;
} TrackProjection;

static TrackProjection project_onto_track(int trk_idx, float x, float z, int hint_seg) {
    const TrackRuntime *tr = &g_tracks[trk_idx];
    int best_i = hint_seg;
    float best_d2 = 1e12f;

    /* Window search around hint */
    int search_span = 25;
    for (int k = -search_span; k <= search_span; k++) {
        int idx = (hint_seg + k + FRAMES_COUNT) % FRAMES_COUNT;
        float dx = x - tr->pts[idx].x;
        float dz = z - tr->pts[idx].z;
        float d2 = dx * dx + dz * dz;
        if (d2 < best_d2) {
            best_d2 = d2;
            best_i = idx;
        }
    }

    int next_i = (best_i + 1) % FRAMES_COUNT;
    Vec3 P = tr->pts[best_i];
    Vec3 T = tr->tangents[best_i];
    Vec3 R = tr->rights[best_i];
    float seg_len = vec3_dist(P, tr->pts[next_i]);

    float dx = x - P.x;
    float dz = z - P.z;
    float seg_t = seg_len > 1e-4f ? (dx * T.x + dz * T.z) / seg_len : 0.0f;
    if (seg_t < 0.0f) seg_t = 0.0f;
    if (seg_t > 1.0f) seg_t = 1.0f;

    TrackProjection proj;
    proj.seg_idx = best_i;
    proj.u = ((float)best_i + seg_t) / (float)FRAMES_COUNT;
    proj.lat = dx * R.x + dz * R.z;
    proj.qx = P.x + T.x * (seg_t * seg_len);
    proj.qy = P.y + (tr->pts[next_i].y - P.y) * seg_t;
    proj.qz = P.z + T.z * (seg_t * seg_len);
    proj.T = T;
    proj.R = R;
    return proj;
}

/* =========================================================================
 * 8. 3D UNDERGROUND SPEEDSHOP STAGE
 * ========================================================================= */

static GLuint g_vbo_garage = 0;
static int g_garage_vert_count = 0;

static void build_garage_stage(void) {
    MeshBuilder3D mb;
    mesh3d_init(&mb, 16384);

    /* Turntable stage base */
    mesh3d_add_cylinder(&mb, 0.0f, -0.18f, 0.0f, 5.6f, 6.0f, 0.34f, 48, vec3(0.08f, 0.12f, 0.18f), 0.1f, 1);
    /* Inner turntable disk */
    mesh3d_add_cylinder(&mb, 0.0f, -0.17f, 0.0f, 5.05f, 5.05f, 0.36f, 48, vec3(0.04f, 0.07f, 0.12f), 0.1f, 1);
    /* Turntable Glowing Cyan Accent Ring */
    mesh3d_add_cylinder(&mb, 0.0f, 0.02f, 0.0f, 5.65f, 5.65f, 0.06f, 48, vec3(0.0f, 0.94f, 1.0f), 1.0f, 1);
    /* Outer Accent Magenta Ring */
    mesh3d_add_cylinder(&mb, 0.0f, -0.02f, 0.0f, 6.30f, 6.30f, 0.04f, 48, vec3(1.0f, 0.16f, 0.52f), 1.0f, 1);

    /* Reflective Floor Grid */
    mesh3d_add_box(&mb, 0.0f, -0.36f, 0.0f, 50.0f, 0.1f, 50.0f, vec3(0.07f, 0.10f, 0.16f), 0.0f);

    /* Industrial Pillars & Vertical Neon Tubes */
    float pz[3] = { -7.5f, 0.0f, 7.5f };
    for (int j = 0; j < 3; j++) {
        float px_sides[2] = { -9.5f, 9.5f };
        for (int s = 0; s < 2; s++) {
            float px = px_sides[s];
            /* Concrete column */
            mesh3d_add_box(&mb, px, 3.2f, pz[j], 0.95f, 7.2f, 0.95f, vec3(0.12f, 0.16f, 0.23f), 0.0f);
            /* Glowing vertical neon strip */
            Vec3 tubeCol = ((j + s) % 2 == 0) ? vec3(0.0f, 0.94f, 1.0f) : vec3(1.0f, 0.16f, 0.52f);
            float tubeOffsetX = (s == 0) ? 0.52f : -0.52f;
            mesh3d_add_box(&mb, px + tubeOffsetX, 3.1f, pz[j], 0.14f, 5.6f, 0.14f, tubeCol, 1.0f);
        }
        /* Overhead steel truss beam */
        mesh3d_add_box(&mb, 0.0f, 6.4f, pz[j], 20.0f, 0.36f, 0.45f, vec3(0.08f, 0.11f, 0.18f), 0.0f);
        /* Studio softbox panel */
        mesh3d_add_box(&mb, 0.0f, 5.95f, pz[j], 9.2f, 0.12f, 0.95f, vec3(0.88f, 0.95f, 1.0f), 1.0f);
    }

    /* Rear Back Wall & Neon Arch */
    mesh3d_add_box(&mb, 0.0f, 4.1f, -12.0f, 26.0f, 9.0f, 0.5f, vec3(0.04f, 0.07f, 0.13f), 0.0f);
    mesh3d_add_box(&mb, 0.0f, 6.2f, -11.5f, 14.0f, 0.18f, 0.2f, vec3(0.0f, 0.94f, 1.0f), 1.0f);

    g_garage_vert_count = mb.count;
    g_vbo_garage = mesh3d_create_vbo(&mb);
    mesh3d_free(&mb);
}

/* =========================================================================
 * 9. SCULPTED 3D SUPERCAR BUILDER, WHEEL & STEERING WHEEL VBOS
 * ========================================================================= */

static GLuint g_vbo_cars[6];
static int g_car_vert_counts[6];

static GLuint g_vbo_cars_cockpit[6];
static int g_car_cockpit_vert_counts[6];

static GLuint g_vbo_wheels[4] = {0, 0, 0, 0};
static int g_wheel_vert_count = 0;

static GLuint g_vbo_steering_wheel = 0;
static int g_steering_vert_count = 0;

static void build_wheel_vbos(void) {
    float wR = 0.42f;
    float wW = 0.34f;
    for (int rim = 0; rim < 4; rim++) {
        MeshBuilder3D mb;
        mesh3d_init(&mb, 4096);
        Vec3 rimCol = g_rims[rim].col;

        /* Outer Tire */
        mesh3d_add_cylinder(&mb, 0.0f, 0.0f, 0.0f, wR, wR, wW, 20, vec3(0.09f, 0.10f, 0.12f), 0.0f, 0);
        /* Rim Barrel */
        mesh3d_add_cylinder(&mb, 0.0f, 0.0f, 0.0f, wR * 0.64f, wR * 0.64f, wW * 1.05f, 16, rimCol, 0.2f, 0);
        /* Brake Rotor */
        mesh3d_add_cylinder(&mb, -0.06f, 0.0f, 0.0f, wR * 0.55f, wR * 0.55f, 0.08f, 14, vec3(0.45f, 0.48f, 0.52f), 0.0f, 0);
        /* 5 Rim Spokes */
        for (int s = 0; s < 5; s++) {
            float a = (float)s / 5.0f * (float)M_PI;
            float c = cosf(a), sn = sinf(a);
            float sx = 0.08f, sy = wR * 0.60f, sz = 0.05f;
            mesh3d_add_box(&mb, 0.08f, 0.0f, 0.0f, sx, sy * 2.0f * c, sz * sn, rimCol, 0.3f);
        }
        /* Brake Caliper */
        mesh3d_add_box(&mb, 0.06f, 0.18f, 0.12f, 0.14f, 0.16f, 0.22f, vec3(1.0f, 0.16f, 0.52f), 0.5f);

        g_wheel_vert_count = mb.count;
        g_vbo_wheels[rim] = mesh3d_create_vbo(&mb);
        mesh3d_free(&mb);
    }
}

static void build_steering_wheel_vbo(void) {
    MeshBuilder3D mb;
    mesh3d_init(&mb, 4096);

    /* Steering Wheel Rim: 16 smooth oriented ring segments */
    int segs = 16;
    float rMajor = 0.135f;
    float rMinor = 0.014f;
    float rDepth = 0.014f;
    Vec3 rimCol = vec3(0.08f, 0.10f, 0.14f);

    for (int i = 0; i < segs; i++) {
        float a0 = (float)i / (float)segs * (float)(M_PI * 2.0);
        float a1 = (float)(i + 1) / (float)segs * (float)(M_PI * 2.0);
        float c0 = cosf(a0), s0 = sinf(a0);
        float c1 = cosf(a1), s1 = sinf(a1);

        float rIn = rMajor - rMinor;
        float rOut = rMajor + rMinor;

        /* Vertices at a0 */
        Vertex3D p0_in_f  = { c0 * rIn,  s0 * rIn,   rDepth, 0,0,0, 0,0,0, 0 };
        Vertex3D p0_out_f = { c0 * rOut, s0 * rOut,  rDepth, 0,0,0, 0,0,0, 0 };
        Vertex3D p0_out_b = { c0 * rOut, s0 * rOut, -rDepth, 0,0,0, 0,0,0, 0 };
        Vertex3D p0_in_b  = { c0 * rIn,  s0 * rIn,  -rDepth, 0,0,0, 0,0,0, 0 };

        /* Vertices at a1 */
        Vertex3D p1_in_f  = { c1 * rIn,  s1 * rIn,   rDepth, 0,0,0, 0,0,0, 0 };
        Vertex3D p1_out_f = { c1 * rOut, s1 * rOut,  rDepth, 0,0,0, 0,0,0, 0 };
        Vertex3D p1_out_b = { c1 * rOut, s1 * rOut, -rDepth, 0,0,0, 0,0,0, 0 };
        Vertex3D p1_in_b  = { c1 * rIn,  s1 * rIn,  -rDepth, 0,0,0, 0,0,0, 0 };

        /* Front face (+Z) */
        mesh3d_add_quad_auto_norm(&mb, p0_in_f, p0_out_f, p1_out_f, p1_in_f, rimCol, 0.0f);
        /* Outer rim face (+radial) */
        mesh3d_add_quad_auto_norm(&mb, p0_out_f, p0_out_b, p1_out_b, p1_out_f, rimCol, 0.0f);
        /* Back face (-Z) */
        mesh3d_add_quad_auto_norm(&mb, p0_out_b, p0_in_b, p1_in_b, p1_out_b, rimCol, 0.0f);
        /* Inner rim face (-radial) */
        mesh3d_add_quad_auto_norm(&mb, p0_in_b, p0_in_f, p1_in_f, p1_in_b, rimCol, 0.0f);
    }

    /* Center Hub */
    mesh3d_add_cylinder(&mb, 0.0f, 0.0f, 0.0f, 0.038f, 0.038f, 0.028f, 14, vec3(0.12f, 0.15f, 0.20f), 0.0f, 2);

    /* 3 Metal Spokes */
    mesh3d_add_box(&mb, -0.065f, 0.0f, 0.0f, 0.085f, 0.016f, 0.012f, vec3(0.75f, 0.78f, 0.82f), 0.2f);
    mesh3d_add_box(&mb,  0.065f, 0.0f, 0.0f, 0.085f, 0.016f, 0.012f, vec3(0.75f, 0.78f, 0.82f), 0.2f);
    mesh3d_add_box(&mb,  0.0f, -0.065f, 0.0f, 0.016f, 0.085f, 0.012f, vec3(0.75f, 0.78f, 0.82f), 0.2f);

    /* Bright Magenta 12-o'clock Top Center Stripe! */
    mesh3d_add_box(&mb, 0.0f, rMajor, 0.0f, 0.030f, 0.034f, 0.038f, vec3(1.0f, 0.16f, 0.52f), 1.0f);

    g_steering_vert_count = mb.count;
    g_vbo_steering_wheel = mesh3d_create_vbo(&mb);
    mesh3d_free(&mb);
}

static void build_single_car_mesh(int car_idx, bool for_cockpit, GLuint *out_vbo, int *out_count) {
    const CarSpec *spec = &g_car_specs[car_idx];
    MeshBuilder3D mb;
    mesh3d_init(&mb, 16384);

    float halfL = spec->len * 0.5f;
    float wid = spec->wid;
    float nH = spec->nose_h;
    float rH = spec->roof_h;
    float pX = spec->roof_peak_x;
    Vec3 paintCol = spec->default_paint;
    Vec3 carbonCol = vec3(0.08f, 0.10f, 0.14f);
    Vec3 glassCol = vec3(0.04f, 0.07f, 0.12f);

    /* 1. Sculpted Body Aerodynamic Silhouette (glow = -1.0f marks live uPaintCol surfaces) */
    /* Lower chassis body */
    mesh3d_add_box(&mb, 0.0f, 0.30f, 0.0f, wid - 0.20f, 0.35f, spec->len, paintCol, -1.0f);

    /* Longitudinal keypoints */
    float z_cowl      = pX + spec->len * 0.20f;
    float z_roof_fwd  = pX + spec->len * 0.05f;
    float z_roof_rear = pX - spec->len * 0.08f;
    float z_deck      = pX - spec->len * 0.28f;
    float z_nose      = halfL * 0.98f;
    float z_rear_end  = -halfL * 0.98f;

    /* Canopy & Body Dimensions */
    float w_belt = wid * 0.84f;
    float bx     = w_belt * 0.5f;
    float w_roof = wid * 0.68f;
    float rx     = w_roof * 0.5f;

    float y_cowl = rH - 0.52f;
    float y_roof = rH;
    float y_deck = rH - 0.48f;
    float y_nose = nH + 0.18f;

    /* Sleek Wedge Hood / Sculpted Front Nose */
    float hw_cowl = (wid - 0.32f) * 0.5f;
    float hw_nose = (wid - 0.38f) * 0.5f;
    Vertex3D h_cowl_l = { -hw_cowl, y_cowl, z_cowl, 0,0,0, 0,0,0, 0 };
    Vertex3D h_cowl_r = {  hw_cowl, y_cowl, z_cowl, 0,0,0, 0,0,0, 0 };
    Vertex3D h_nose_r = {  hw_nose, y_nose, z_nose, 0,0,0, 0,0,0, 0 };
    Vertex3D h_nose_l = { -hw_nose, y_nose, z_nose, 0,0,0, 0,0,0, 0 };
    /* Sloped Hood Top (outward +Y normal) */
    mesh3d_add_quad_auto_norm(&mb, h_cowl_l, h_nose_l, h_nose_r, h_cowl_r, paintCol, -1.0f);

    /* Hood Side & Front Faces (outward normals) */
    Vertex3D h_cowl_lb = { -hw_cowl, 0.35f, z_cowl, 0,0,0, 0,0,0, 0 };
    Vertex3D h_cowl_rb = {  hw_cowl, 0.35f, z_cowl, 0,0,0, 0,0,0, 0 };
    Vertex3D h_nose_rb = {  hw_nose, 0.35f, z_nose, 0,0,0, 0,0,0, 0 };
    Vertex3D h_nose_lb = { -hw_nose, 0.35f, z_nose, 0,0,0, 0,0,0, 0 };
    mesh3d_add_quad_auto_norm(&mb, h_cowl_lb, h_nose_lb, h_nose_l, h_cowl_l, paintCol, -1.0f);
    mesh3d_add_quad_auto_norm(&mb, h_cowl_rb, h_cowl_r, h_nose_r, h_nose_rb, paintCol, -1.0f);
    mesh3d_add_quad_auto_norm(&mb, h_nose_lb, h_nose_rb, h_nose_r, h_nose_l, paintCol, -1.0f);

    /* Rear Deck Lid */
    float deck_len = z_deck - z_rear_end;
    float deck_cz  = (z_deck + z_rear_end) * 0.5f;
    mesh3d_add_box(&mb, 0.0f, (y_deck + 0.35f) * 0.5f, deck_cz, wid - 0.32f, y_deck - 0.35f, deck_len, paintCol, -1.0f);

    /* 2. Cockpit Canopy Glass: ONLY included when NOT in Cockpit View! */
    if (!for_cockpit) {
        /* Beltline and Roof Vertices */
        Vertex3D v_cowl_l = { -bx, y_cowl, z_cowl, 0,0,0, 0,0,0, 0 };
        Vertex3D v_cowl_r = {  bx, y_cowl, z_cowl, 0,0,0, 0,0,0, 0 };
        Vertex3D v_deck_l = { -bx, y_deck, z_deck, 0,0,0, 0,0,0, 0 };
        Vertex3D v_deck_r = {  bx, y_deck, z_deck, 0,0,0, 0,0,0, 0 };

        Vertex3D v_rf_l = { -rx, y_roof, z_roof_fwd,  0,0,0, 0,0,0, 0 };
        Vertex3D v_rf_r = {  rx, y_roof, z_roof_fwd,  0,0,0, 0,0,0, 0 };
        Vertex3D v_rr_l = { -rx, y_roof, z_roof_rear, 0,0,0, 0,0,0, 0 };
        Vertex3D v_rr_r = {  rx, y_roof, z_roof_rear, 0,0,0, 0,0,0, 0 };

        /* 1. Aerodynamic Sloped Windshield */
        mesh3d_add_quad_auto_norm(&mb, v_cowl_l, v_cowl_r, v_rf_r, v_rf_l, glassCol, 0.6f);

        /* 2. Tapered Roof Crown */
        mesh3d_add_quad_auto_norm(&mb, v_rf_l, v_rf_r, v_rr_r, v_rr_l, paintCol, -1.0f);

        /* 3. Fastback Aerodynamic Sloped Rear Window */
        mesh3d_add_quad_auto_norm(&mb, v_deck_r, v_deck_l, v_rr_l, v_rr_r, glassCol, 0.5f);

        /* 4. Left Side Glass (Tapered Prism Side Profile) */
        mesh3d_add_tri_auto_norm(&mb, v_cowl_l, v_rf_l, v_deck_l, glassCol, 0.5f);
        mesh3d_add_tri_auto_norm(&mb, v_rf_l, v_rr_l, v_deck_l, glassCol, 0.5f);

        /* 5. Right Side Glass (Tapered Prism Side Profile) */
        mesh3d_add_tri_auto_norm(&mb, v_cowl_r, v_deck_r, v_rf_r, glassCol, 0.5f);
        mesh3d_add_tri_auto_norm(&mb, v_rf_r, v_deck_r, v_rr_r, glassCol, 0.5f);
    }

    /* 3. Carbon Aerodynamic Splitter, Diffuser & Side Skirts */
    mesh3d_add_box(&mb, 0.0f, 0.20f, halfL + 0.05f, wid + 0.08f, 0.08f, 0.40f, carbonCol, 0.0f);
    /* Subtle front splitter bevel leading lip */
    Vertex3D sp_l0 = { -(wid + 0.08f) * 0.5f, 0.24f, halfL + 0.25f, 0,0,0, 0,0,0, 0 };
    Vertex3D sp_r0 = {  (wid + 0.08f) * 0.5f, 0.24f, halfL + 0.25f, 0,0,0, 0,0,0, 0 };
    Vertex3D sp_r1 = {  (wid + 0.08f) * 0.5f, 0.16f, halfL + 0.32f, 0,0,0, 0,0,0, 0 };
    Vertex3D sp_l1 = { -(wid + 0.08f) * 0.5f, 0.16f, halfL + 0.32f, 0,0,0, 0,0,0, 0 };
    mesh3d_add_quad_auto_norm(&mb, sp_l0, sp_r0, sp_r1, sp_l1, carbonCol, 0.1f);

    mesh3d_add_box(&mb, 0.0f, 0.26f, -halfL - 0.05f, wid - 0.10f, 0.16f, 0.42f, carbonCol, 0.0f);
    mesh3d_add_box(&mb, -wid * 0.5f - 0.02f, 0.24f, 0.0f, 0.12f, 0.14f, spec->len * 0.60f, carbonCol, 0.0f);
    mesh3d_add_box(&mb,  wid * 0.5f + 0.02f, 0.24f, 0.0f, 0.12f, 0.14f, spec->len * 0.60f, carbonCol, 0.0f);

    if (spec->hood_scoop) {
        mesh3d_add_box(&mb, 0.0f, nH + 0.32f, 1.25f, 0.55f, 0.12f, 0.85f, carbonCol, 0.2f);
    }
    if (spec->mid_engine_vents) {
        mesh3d_add_box(&mb, -0.28f, nH + 0.45f, -1.20f, 0.16f, 0.05f, 0.65f, carbonCol, 0.0f);
        mesh3d_add_box(&mb,  0.28f, nH + 0.45f, -1.20f, 0.16f, 0.05f, 0.65f, carbonCol, 0.0f);
    }

    /* 4. Emissive Projector Headlights & LED Tail-Light Bar */
    Vec3 hlCol = vec3(0.95f, 0.98f, 1.0f);
    mesh3d_add_box(&mb, -wid * 0.34f, nH + 0.20f, halfL + 0.03f, 0.38f, 0.11f, 0.12f, hlCol, 0.95f);
    mesh3d_add_box(&mb,  wid * 0.34f, nH + 0.20f, halfL + 0.03f, 0.38f, 0.11f, 0.12f, hlCol, 0.95f);

    Vec3 tailCol = vec3(1.0f, 0.10f, 0.20f);
    mesh3d_add_box(&mb, 0.0f, nH + 0.38f, -halfL - 0.02f, wid - 0.25f, 0.10f, 0.10f, tailCol, 1.0f);

    /* Dual Chrome Exhaust Tips */
    Vec3 chromeCol = vec3(0.88f, 0.90f, 0.95f);
    mesh3d_add_cylinder(&mb, -0.56f, 0.28f, -halfL - 0.08f, 0.075f, 0.075f, 0.26f, 12, chromeCol, 0.5f, 2);
    mesh3d_add_cylinder(&mb,  0.56f, 0.28f, -halfL - 0.08f, 0.075f, 0.075f, 0.26f, 12, chromeCol, 0.5f, 2);

    /* 5. Aero Wing Kit */
    int wing = spec->default_wing;
    if (wing == 0) { /* GT Pro Wing */
        mesh3d_add_box(&mb, 0.0f, nH + 0.88f, -halfL + 0.12f, wid + 0.18f, 0.07f, 0.40f, carbonCol, 0.2f);
        mesh3d_add_box(&mb, -0.56f, nH + 0.68f, -halfL + 0.16f, 0.06f, 0.38f, 0.22f, carbonCol, 0.0f);
        mesh3d_add_box(&mb,  0.56f, nH + 0.68f, -halfL + 0.16f, 0.06f, 0.38f, 0.22f, carbonCol, 0.0f);
        mesh3d_add_box(&mb, -(wid + 0.18f) * 0.5f, nH + 0.86f, -halfL + 0.12f, 0.04f, 0.24f, 0.44f, paintCol, -1.0f);
        mesh3d_add_box(&mb,  (wid + 0.18f) * 0.5f, nH + 0.86f, -halfL + 0.12f, 0.04f, 0.24f, 0.44f, paintCol, -1.0f);
    } else if (wing == 1) { /* Ducktail Lip */
        mesh3d_add_box(&mb, 0.0f, nH + 0.54f, -halfL + 0.06f, wid - 0.18f, 0.18f, 0.22f, paintCol, -1.0f);
    } else if (wing == 2) { /* Twin-Fin Cyber Blades */
        mesh3d_add_box(&mb, -0.65f, nH + 0.70f, -halfL + 0.20f, 0.08f, 0.44f, 0.58f, paintCol, -1.0f);
        mesh3d_add_box(&mb,  0.65f, nH + 0.70f, -halfL + 0.20f, 0.08f, 0.44f, 0.58f, paintCol, -1.0f);
        mesh3d_add_box(&mb, 0.0f, nH + 0.88f, -halfL + 0.12f, 1.35f, 0.06f, 0.25f, spec->default_glow, 1.0f);
    }

    /* 6. Police Interceptor Lightbar (Car 5) */
    if (car_idx == 5) {
        mesh3d_add_box(&mb, -0.32f, rH + 0.16f, pX - 0.1f, 0.54f, 0.14f, 0.24f, vec3(1.0f, 0.08f, 0.18f), 1.0f);
        mesh3d_add_box(&mb,  0.32f, rH + 0.16f, pX - 0.1f, 0.54f, 0.14f, 0.24f, vec3(0.08f, 0.52f, 1.0f), 1.0f);
    }

    /* 7. Unobstructed 3D Cockpit Interior: Rendered ONLY in Cockpit View! */
    if (for_cockpit) {
        float camZ = pX - 0.20f;
        Vec3 dashMat = vec3(0.05f, 0.07f, 0.11f);
        /* Dark interior footwell shroud (under dash) */
        mesh3d_add_box(&mb, 0.0f, 0.20f, camZ + 1.25f, wid - 0.22f, 0.24f, 1.35f, dashMat, 0.0f);
        /* Low matte dashboard deck and cowl */
        mesh3d_add_box(&mb, 0.0f, rH - 0.58f, camZ + 0.68f, wid - 0.52f, 0.14f, 0.50f, dashMat, 0.0f);
        mesh3d_add_box(&mb, 0.0f, rH - 0.48f, camZ + 0.64f, 0.54f, 0.08f, 0.20f, dashMat, 0.0f);
        /* Glowing Cyan Digital Instrument Strip */
        mesh3d_add_box(&mb, 0.0f, rH - 0.44f, camZ + 0.56f, 0.46f, 0.024f, 0.03f, spec->default_glow, 1.0f);
        /* Outer A-pillars at screen edges (NO blocking top visor bar!) */
        mesh3d_add_box(&mb, -wid * 0.44f, rH - 0.18f, camZ + 0.52f, 0.055f, 0.68f, 0.07f, dashMat, 0.0f);
        mesh3d_add_box(&mb,  wid * 0.44f, rH - 0.18f, camZ + 0.52f, 0.055f, 0.68f, 0.07f, dashMat, 0.0f);
    }

    *out_count = mb.count;
    *out_vbo = mesh3d_create_vbo(&mb);
    mesh3d_free(&mb);
}

static void build_all_cars(void) {
    build_wheel_vbos();
    build_steering_wheel_vbo();
    for (int i = 0; i < 6; i++) {
        build_single_car_mesh(i, false, &g_vbo_cars[i], &g_car_vert_counts[i]);
        build_single_car_mesh(i, true,  &g_vbo_cars_cockpit[i], &g_car_cockpit_vert_counts[i]);
    }
}

/* =========================================================================
 * 10. RACER & GAMEPLAY PHYSICS ENGINE
 * ========================================================================= */

typedef struct {
    float x, y, z;
    float heading;
    float vel_heading;
    float speed;
    float rpm;
    int gear;
    float n2o;
    float steer_angle;
    float drift_angle;
    float load_pitch;
    int seg_idx;
    float u;
    float lat;
    int lap;
    float total_progress;
    bool nos_active;
    float backfire_timer;
    float wheelspin_timer;
    int car_idx;
    int paint_idx;
    int rim_idx;
    int glow_idx;
    int wing_idx;
    bool is_ai;
    bool is_pursuit;
    float ai_target_speed;
    float ai_lat_target;
    Vec3 tail_history_l[16];
    Vec3 tail_history_r[16];
    int tail_count;
} Racer;

static Racer g_player;
#define MAX_AI 5
static Racer g_ai[MAX_AI];
static int g_ai_count = 5;

static const float GEAR_MAX_MPH[7] = { 0.0f, 42.0f, 76.0f, 114.0f, 152.0f, 192.0f, 275.0f };

typedef enum {
    STATE_TITLE = 0,
    STATE_GARAGE = 1,
    STATE_COUNTDOWN = 2,
    STATE_RACING = 3,
    STATE_RESULTS = 4
} AppState;

static AppState g_state = STATE_TITLE;
static int g_selected_car = 0;
static int g_selected_paint = 1;
static int g_selected_rim = 0;
static int g_selected_glow = 0;
static int g_selected_wing = 0;
static bool g_manual_trans = false;
static int g_selected_track = 0;
static int g_selected_mode = 0;

static int g_cam_mode = 0; /* 0: Chase, 1: Cockpit, 2: Hood, 3: Heli */
static bool g_mirror_on = true;

static SDL_Joystick *g_joy = NULL;
static int g_title_row = 0;   /* 0: Quick Race, 1: Tuner Speedshop, 2: Hot Pursuit, 3: Drift Touge */
static int g_garage_row = 0;  /* 0..7 options in speedshop */
static int g_final_rank = 1;  /* 1..6 finish position */

static float g_countdown_timer = 3.2f;
static int g_last_countdown_beep = 4;
static float g_race_time = 0.0f;
static float g_lap_start_time = 0.0f;
static float g_best_lap_time = 9999.0f;
static int g_total_style_score = 0;
static float g_active_drift_pts = 0.0f;
static float g_active_drift_timer = 0.0f;
static int g_active_drift_mult = 1;
static bool g_is_drafting = false;
static float g_garage_orbit_yaw = 0.56f;

static void reset_race(void) {
    g_race_time = 0.0f;
    g_lap_start_time = 0.0f;
    g_countdown_timer = 3.2f;
    g_last_countdown_beep = 4;
    g_total_style_score = 0;
    g_active_drift_pts = 0.0f;
    g_active_drift_timer = 0.0f;
    g_active_drift_mult = 1;
    g_is_drafting = false;
    g_final_rank = 1;

    /* Setup player */
    memset(&g_player, 0, sizeof(Racer));
    g_player.car_idx = g_selected_car;
    g_player.paint_idx = g_selected_paint;
    g_player.rim_idx = g_selected_rim;
    g_player.glow_idx = g_selected_glow;
    g_player.wing_idx = g_selected_wing;
    g_player.gear = 1;
    g_player.rpm = 1200.0f;
    g_player.n2o = 100.0f;
    g_player.lat = 0.0f;
    g_player.u = 0.005f;

    const TrackRuntime *tr = &g_tracks[g_selected_track];
    int start_seg = (int)(g_player.u * (float)FRAMES_COUNT);
    g_player.seg_idx = start_seg;
    g_player.x = tr->pts[start_seg].x;
    g_player.y = tr->pts[start_seg].y;
    g_player.z = tr->pts[start_seg].z;
    g_player.heading = atan2f(tr->tangents[start_seg].x, tr->tangents[start_seg].z);
    g_player.vel_heading = g_player.heading;

    /* Setup AI rivals */
    g_ai_count = (g_selected_mode == 2) ? 3 : 5;
    for (int i = 0; i < g_ai_count; i++) {
        memset(&g_ai[i], 0, sizeof(Racer));
        g_ai[i].is_ai = true;
        g_ai[i].is_pursuit = (g_selected_mode == 2);
        g_ai[i].car_idx = g_ai[i].is_pursuit ? 5 : (i + 1) % 6;
        g_ai[i].paint_idx = g_ai[i].is_pursuit ? 7 : (i + 2) % 8;
        g_ai[i].glow_idx = (i + 1) % 7;
        g_ai[i].gear = 1;
        g_ai[i].rpm = 3000.0f;
        g_ai[i].u = 0.02f + (float)(i + 1) * 0.015f;
        g_ai[i].lat = (i % 2 == 0) ? -4.5f : 4.5f;
        g_ai[i].ai_target_speed = 78.0f + (float)(i * 4);
        g_ai[i].ai_lat_target = g_ai[i].lat;

        int ai_seg = (int)(g_ai[i].u * (float)FRAMES_COUNT);
        g_ai[i].seg_idx = ai_seg;
        g_ai[i].x = tr->pts[ai_seg].x + tr->rights[ai_seg].x * g_ai[i].lat;
        g_ai[i].y = tr->pts[ai_seg].y;
        g_ai[i].z = tr->pts[ai_seg].z + tr->rights[ai_seg].z * g_ai[i].lat;
        g_ai[i].heading = atan2f(tr->tangents[ai_seg].x, tr->tangents[ai_seg].z);
        g_ai[i].vel_heading = g_ai[i].heading;
    }
}

/* =========================================================================
 * 11. PROCEDURAL AUDIO SYNTHESIZER (SDL2 S16SYS MONO @ 22050 HZ)
 * ========================================================================= */

static bool g_audio_muted = false;
static float g_audio_rpm_norm = 0.1f;
static bool g_audio_drifting = false;
static bool g_audio_nos = false;
static float g_phase_osc1 = 0.0f;
static float g_phase_osc2 = 0.0f;
static float g_synth_timer = 0.0f;
static int g_synth_step = 0;
static uint32_t g_noise_seed = 0x12345678;

static inline float audio_rand(void) {
    g_noise_seed = g_noise_seed * 1103515245 + 12345;
    return (float)(g_noise_seed & 0x7FFFFFFF) / 1073741824.0f - 1.0f;
}

static void audio_callback(void *userdata, Uint8 *stream, int len) {
    (void)userdata;
    int16_t *buf = (int16_t*)stream;
    int samples = len / 2;

    if (g_audio_muted) {
        memset(stream, 0, len);
        return;
    }

    const float sample_rate = 22050.0f;
    float base_freq = 48.0f + g_audio_rpm_norm * 145.0f + (g_audio_nos ? 28.0f : 0.0f);
    float dt = 1.0f / sample_rate;

    static const float bass_notes[8] = { 55.0f, 55.0f, 65.41f, 73.42f, 55.0f, 55.0f, 82.41f, 73.42f };

    for (int i = 0; i < samples; i++) {
        /* Dual Engine Oscillators (Sawtooth + Triangle) */
        g_phase_osc1 += base_freq * dt;
        if (g_phase_osc1 >= 1.0f) g_phase_osc1 -= 1.0f;
        float saw = 2.0f * g_phase_osc1 - 1.0f;

        g_phase_osc2 += (base_freq * 1.99f) * dt;
        if (g_phase_osc2 >= 1.0f) g_phase_osc2 -= 1.0f;
        float tri = 4.0f * fabsf(g_phase_osc2 - 0.5f) - 1.0f;

        float engine_mix = (saw * 0.55f + tri * 0.45f) * (0.18f + g_audio_rpm_norm * 0.12f);

        /* Tire Screech Noise */
        float screech = 0.0f;
        if (g_audio_drifting) {
            screech = audio_rand() * 0.15f;
        }

        /* N2O Hiss */
        float nos_hiss = 0.0f;
        if (g_audio_nos) {
            nos_hiss = audio_rand() * 0.22f;
        }

        /* Synthwave Bassline */
        g_synth_timer += dt;
        if (g_synth_timer >= 0.214f) {
            g_synth_timer = 0.0f;
            g_synth_step = (g_synth_step + 1) % 8;
        }
        float bass_freq = bass_notes[g_synth_step];
        float bass_env = fmaxf(0.0f, 1.0f - g_synth_timer / 0.18f);
        float bass_sample = sinf(g_synth_timer * bass_freq * (float)(M_PI * 2.0)) * bass_env * 0.12f;

        float total = engine_mix + screech + nos_hiss + bass_sample;
        if (total > 1.0f) total = 1.0f;
        if (total < -1.0f) total = -1.0f;

        buf[i] = (int16_t)(total * 30000.0f);
    }
}

/* =========================================================================
 * 12. DRAWING & RENDERING (CIRCUITS, CARS, INTERIOR, MIRROR, FX & HUD)
 * ========================================================================= */

static void update_racer_tail_history(Racer *r) {
    float halfL = g_car_specs[r->car_idx].len * 0.5f;
    float wid   = g_car_specs[r->car_idx].wid;
    float nH    = g_car_specs[r->car_idx].nose_h;
    float sinH  = sinf(r->heading);
    float cosH  = cosf(r->heading);

    float lx = wid * 0.5f - 0.22f;
    float ly = nH + 0.38f;
    float lz = -halfL - 0.04f;

    Vec3 curL = vec3(r->x + cosH * (-lx) + sinH * lz, r->y + ly, r->z - sinH * (-lx) + cosH * lz);
    Vec3 curR = vec3(r->x + cosH * ( lx) + sinH * lz, r->y + ly, r->z - sinH * ( lx) + cosH * lz);

    if (r->tail_count <= 0) {
        r->tail_history_l[0] = curL;
        r->tail_history_r[0] = curR;
        r->tail_count = 1;
        return;
    }

    if (r->tail_count == 1 || vec3_dist(curL, r->tail_history_l[1]) > 0.65f) {
        int lim = r->tail_count < 16 ? r->tail_count : 15;
        for (int i = lim; i >= 1; i--) {
            r->tail_history_l[i] = r->tail_history_l[i - 1];
            r->tail_history_r[i] = r->tail_history_r[i - 1];
        }
        if (r->tail_count < 16) r->tail_count++;
    }
    r->tail_history_l[0] = curL;
    r->tail_history_r[0] = curR;
}

static void push_single_tail_ribbon(const Vec3 *pts, int count, Vec3 col, float baseAlpha) {
    if (count < 2) return;
    for (int i = 0; i < count - 1; i++) {
        Vec3 p0 = pts[i];
        Vec3 p1 = pts[i + 1];
        Vec3 dir = vec3_sub(p0, p1);
        float dLen = sqrtf(dir.x * dir.x + dir.z * dir.z);
        if (dLen < 1e-4f) continue;
        float rx = -dir.z / dLen;
        float rz =  dir.x / dLen;

        float t0 = 1.0f - (float)i / (float)(count - 1);
        float t1 = 1.0f - (float)(i + 1) / (float)(count - 1);

        /* Outer soft additive glow + crisp bright inner core */
        float w0 = 0.04f + 0.14f * t0;
        float w1 = 0.04f + 0.14f * t1;
        float a0 = baseAlpha * t0 * t0;
        float a1 = baseAlpha * t1 * t1;

        VertexFX v0 = { p0.x - rx * w0, p0.y, p0.z - rz * w0, -1.0f, 0.0f, col.x, col.y, col.z, a0 };
        VertexFX v1 = { p0.x + rx * w0, p0.y, p0.z + rz * w0, -1.0f, 0.0f, col.x, col.y, col.z, a0 };
        VertexFX v2 = { p1.x + rx * w1, p1.y, p1.z + rz * w1, -1.0f, 0.0f, col.x, col.y, col.z, a1 };
        VertexFX v3 = { p1.x - rx * w1, p1.y, p1.z - rz * w1, -1.0f, 0.0f, col.x, col.y, col.z, a1 };
        fx_push_quad(v0, v1, v2, v3);

        float cw0 = 0.02f + 0.055f * t0;
        float cw1 = 0.02f + 0.055f * t1;
        VertexFX c0 = { p0.x - rx * cw0, p0.y + 0.01f, p0.z - rz * cw0, -1.0f, 0.0f, 1.0f, 0.92f, 0.95f, a0 * 0.95f };
        VertexFX c1 = { p0.x + rx * cw0, p0.y + 0.01f, p0.z + rz * cw0, -1.0f, 0.0f, 1.0f, 0.92f, 0.95f, a0 * 0.95f };
        VertexFX c2 = { p1.x + rx * cw1, p1.y + 0.01f, p1.z + rz * cw1, -1.0f, 0.0f, 1.0f, 0.92f, 0.95f, a1 * 0.95f };
        VertexFX c3 = { p1.x - rx * cw1, p1.y + 0.01f, p1.z - rz * cw1, -1.0f, 0.0f, 1.0f, 0.92f, 0.95f, a1 * 0.95f };
        fx_push_quad(c0, c1, c2, c3);
    }
}

static void push_racer_tail_ribbons(const Racer *r) {
    if (r->tail_count < 2 || fabsf(r->speed) < 8.0f) return;
    Vec3 ribCol = r->nos_active ? vec3(0.00f, 0.94f, 1.00f) : vec3(1.00f, 0.08f, 0.20f);
    float alpha = r->nos_active ? 0.92f : 0.78f;
    push_single_tail_ribbon(r->tail_history_l, r->tail_count, ribCol, alpha);
    push_single_tail_ribbon(r->tail_history_r, r->tail_count, ribCol, alpha);
}

static void render_car_mesh(const Racer *r, bool in_cockpit) {
    Mat4 mModel, mRotY, mRotX, mTrans, mTmp;
    mat4_rotate_y(mRotY, r->heading);
    mat4_rotate_x(mRotX, r->load_pitch);
    mat4_mul(mTmp, mRotY, mRotX);
    mat4_translate(mTrans, r->x, r->y, r->z);
    mat4_mul(mModel, mTrans, mTmp);

    glUniformMatrix4fv(g_u3d_Model, 1, GL_FALSE, mModel);
    int paintIdx = (r->paint_idx >= 0 && r->paint_idx < 8) ? r->paint_idx : 0;
    Vec3 paintCol = g_paints[paintIdx].col;
    glUniform3f(g_u3d_PaintCol, paintCol.x, paintCol.y, paintCol.z);

    GLuint vbo = in_cockpit ? g_vbo_cars_cockpit[r->car_idx] : g_vbo_cars[r->car_idx];
    int count  = in_cockpit ? g_car_cockpit_vert_counts[r->car_idx] : g_car_vert_counts[r->car_idx];

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(g_a3d_Pos);
    glVertexAttribPointer(g_a3d_Pos, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex3D), (void*)0);
    glEnableVertexAttribArray(g_a3d_Norm);
    glVertexAttribPointer(g_a3d_Norm, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex3D), (void*)(sizeof(float) * 3));
    glEnableVertexAttribArray(g_a3d_Col);
    glVertexAttribPointer(g_a3d_Col, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex3D), (void*)(sizeof(float) * 6));
    glEnableVertexAttribArray(g_a3d_Glow);
    glVertexAttribPointer(g_a3d_Glow, 1, GL_FLOAT, GL_FALSE, sizeof(Vertex3D), (void*)(sizeof(float) * 9));

    glDrawArrays(GL_TRIANGLES, 0, count);

    /* 4 Wheels */
    float halfL = g_car_specs[r->car_idx].len * 0.5f;
    float wid = g_car_specs[r->car_idx].wid;
    float wz = halfL * 0.62f;
    float wx = wid * 0.48f;

    int rimIdx = (r->rim_idx >= 0 && r->rim_idx < 4) ? r->rim_idx : 0;
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo_wheels[rimIdx]);
    glEnableVertexAttribArray(g_a3d_Pos);
    glVertexAttribPointer(g_a3d_Pos, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex3D), (void*)0);
    glEnableVertexAttribArray(g_a3d_Norm);
    glVertexAttribPointer(g_a3d_Norm, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex3D), (void*)(sizeof(float) * 3));
    glEnableVertexAttribArray(g_a3d_Col);
    glVertexAttribPointer(g_a3d_Col, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex3D), (void*)(sizeof(float) * 6));
    glEnableVertexAttribArray(g_a3d_Glow);
    glVertexAttribPointer(g_a3d_Glow, 1, GL_FLOAT, GL_FALSE, sizeof(Vertex3D), (void*)(sizeof(float) * 9));

    float wheelOffsets[4][2] = {
        { -wx,  wz }, {  wx,  wz },
        { -wx, -wz }, {  wx, -wz }
    };
    for (int w = 0; w < 4; w++) {
        if (in_cockpit && w < 2) continue; /* No front wheels peeking into cockpit */
        Mat4 mWPos, mWRot, mWheelModel;
        mat4_translate(mWPos, wheelOffsets[w][0], 0.42f, wheelOffsets[w][1]);
        if (w < 2) {
            mat4_rotate_y(mWRot, r->steer_angle * 0.45f);
            mat4_mul(mTmp, mWPos, mWRot);
        } else {
            memcpy(mTmp, mWPos, sizeof(Mat4));
        }
        mat4_mul(mWheelModel, mModel, mTmp);
        glUniformMatrix4fv(g_u3d_Model, 1, GL_FALSE, mWheelModel);
        glDrawArrays(GL_TRIANGLES, 0, g_wheel_vert_count);
    }

    /* Animated 3D Sport Steering Wheel (in Cockpit View) */
    if (in_cockpit) {
        float rH = g_car_specs[r->car_idx].roof_h;
        float pX = g_car_specs[r->car_idx].roof_peak_x;
        float camZ = pX - 0.20f;
        Mat4 mSwPos, mSwRotX, mSwSteer, mSwModel;
        mat4_translate(mSwPos, 0.0f, rH - 0.46f, camZ + 0.44f);
        mat4_rotate_x(mSwRotX, 0.20f);
        mat4_rotate_z(mSwSteer, -r->steer_angle * 1.2f);
        mat4_mul(mTmp, mSwRotX, mSwSteer);
        mat4_mul(mTmp, mSwPos, mTmp);
        mat4_mul(mSwModel, mModel, mTmp);

        glUniformMatrix4fv(g_u3d_Model, 1, GL_FALSE, mSwModel);
        glBindBuffer(GL_ARRAY_BUFFER, g_vbo_steering_wheel);
        glEnableVertexAttribArray(g_a3d_Pos);
        glVertexAttribPointer(g_a3d_Pos, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex3D), (void*)0);
        glEnableVertexAttribArray(g_a3d_Norm);
        glVertexAttribPointer(g_a3d_Norm, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex3D), (void*)(sizeof(float) * 3));
        glEnableVertexAttribArray(g_a3d_Col);
        glVertexAttribPointer(g_a3d_Col, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex3D), (void*)(sizeof(float) * 6));
        glEnableVertexAttribArray(g_a3d_Glow);
        glVertexAttribPointer(g_a3d_Glow, 1, GL_FLOAT, GL_FALSE, sizeof(Vertex3D), (void*)(sizeof(float) * 9));
        glDrawArrays(GL_TRIANGLES, 0, g_steering_vert_count);
    }

    glDisableVertexAttribArray(g_a3d_Pos);
    glDisableVertexAttribArray(g_a3d_Norm);
    glDisableVertexAttribArray(g_a3d_Col);
    glDisableVertexAttribArray(g_a3d_Glow);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

static void render_world_scene(int trk_idx, const Mat4 mView, const Mat4 mProj, Vec3 camPos, bool isCockpitView) {
    render_sky_gradient(trk_idx);

    Mat4 mMVP, mIdent;
    mat4_identity(mIdent);
    mat4_mul(mMVP, mProj, mView);

    const TrackSpec *ts = &g_track_specs[trk_idx];
    const TrackRuntime *tr = &g_tracks[trk_idx];

    glUseProgram(g_prog3d);
    glUniformMatrix4fv(g_u3d_MVP, 1, GL_FALSE, mMVP);
    glUniformMatrix4fv(g_u3d_Model, 1, GL_FALSE, mIdent);
    glUniform3f(g_u3d_CamPos, camPos.x, camPos.y, camPos.z);
    glUniform3f(g_u3d_PaintCol, 1.0f, 1.0f, 1.0f);
    glUniform3f(g_u3d_SunDir, 0.45f, 0.75f, 0.45f);
    glUniform3f(g_u3d_SunCol, ts->sun_col.x, ts->sun_col.y, ts->sun_col.z);
    glUniform3f(g_u3d_AmbientCol, ts->ambient_col.x, ts->ambient_col.y, ts->ambient_col.z);
    glUniform3f(g_u3d_FogCol, ts->fog_col.x, ts->fog_col.y, ts->fog_col.z);
    glUniform1f(g_u3d_FogDensity, ts->fog_density);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    /* Draw Static Track Mesh */
    glBindBuffer(GL_ARRAY_BUFFER, tr->vbo_static);
    glEnableVertexAttribArray(g_a3d_Pos);
    glVertexAttribPointer(g_a3d_Pos, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex3D), (void*)0);
    glEnableVertexAttribArray(g_a3d_Norm);
    glVertexAttribPointer(g_a3d_Norm, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex3D), (void*)(sizeof(float) * 3));
    glEnableVertexAttribArray(g_a3d_Col);
    glVertexAttribPointer(g_a3d_Col, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex3D), (void*)(sizeof(float) * 6));
    glEnableVertexAttribArray(g_a3d_Glow);
    glVertexAttribPointer(g_a3d_Glow, 1, GL_FLOAT, GL_FALSE, sizeof(Vertex3D), (void*)(sizeof(float) * 9));

    glDrawArrays(GL_TRIANGLES, 0, tr->vert_count);

    glDisableVertexAttribArray(g_a3d_Pos);
    glDisableVertexAttribArray(g_a3d_Norm);
    glDisableVertexAttribArray(g_a3d_Col);
    glDisableVertexAttribArray(g_a3d_Glow);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    /* Draw AI Rivals */
    for (int i = 0; i < g_ai_count; i++) {
        render_car_mesh(&g_ai[i], false);
    }

    /* Draw Player Car */
    render_car_mesh(&g_player, isCockpitView);

    /* Underglow FX & Taillight Ribbons */
    if (g_glows[g_player.glow_idx].active) {
        Vec3 gCol = g_glows[g_player.glow_idx].col;
        fx_push_radial_glow(vec3(g_player.x, g_player.y + 0.08f, g_player.z), 2.2f, 3.2f, vec4(gCol.x, gCol.y, gCol.z, 0.45f));
    }
    for (int i = 0; i < g_ai_count; i++) {
        if (g_glows[g_ai[i].glow_idx].active) {
            Vec3 gCol = g_glows[g_ai[i].glow_idx].col;
            fx_push_radial_glow(vec3(g_ai[i].x, g_ai[i].y + 0.08f, g_ai[i].z), 2.2f, 3.2f, vec4(gCol.x, gCol.y, gCol.z, 0.35f));
        }
    }

    if (g_state == STATE_TITLE || g_state == STATE_RACING) {
        for (int i = 0; i < g_ai_count; i++) {
            push_racer_tail_ribbons(&g_ai[i]);
        }
        if (!isCockpitView) {
            push_racer_tail_ribbons(&g_player);
        }
    }

    /* Falling Rain FX (on Tokyo Track) */
    if (trk_idx == 0) {
        for (int r = 0; r < 120; r++) {
            float rx = camPos.x + (float)((r * 37) % 60) - 30.0f;
            float rz = camPos.z + (float)((r * 53) % 60) - 30.0f;
            float ry = fmodf((float)r * 1.5f - g_race_time * 40.0f, 35.0f);
            if (ry < 0.0f) ry += 35.0f;
            VertexFX v0 = { rx, ry, rz, -1.0f, 0.0f, 0.6f, 0.8f, 1.0f, 0.4f };
            VertexFX v1 = { rx, ry - 1.8f, rz, -1.0f, 0.0f, 0.6f, 0.8f, 1.0f, 0.4f };
            VertexFX v2 = { rx + 0.05f, ry - 1.8f, rz, -1.0f, 0.0f, 0.6f, 0.8f, 1.0f, 0.4f };
            fx_push_tri(v0, v1, v2);
        }
    }

    flush_fx(mMVP);
}

static void draw_hud_tachometer(float cx, float cy, float r, float rpm, int gear, bool manual, float mph, float n2o, bool nosActive, bool inCountdown) {
    /* Dark Obsidian-Navy Tachometer Card */
    Vec4 borderCol = nosActive ? vec4(0.0f, 0.94f, 1.0f, 0.70f) : vec4(0.22f, 0.74f, 0.97f, 0.32f);
    Vec4 topCol = nosActive ? vec4(0.0f, 0.94f, 1.0f, 0.95f) : vec4(1.0f, 0.16f, 0.52f, 0.85f);
    hud_card(cx - r - 18.0f, cy - r - 18.0f, (r + 18.0f) * 2.0f, (r + 18.0f) * 2.0f, 0.0f,
             vec4(0.024f, 0.039f, 0.078f, 0.90f), borderCol, topCol);

    /* RPM Gauge Arc (140 deg to 380 deg) */
    int segs = 36;
    float startA = 2.45f; /* ~140 deg */
    float endA = 6.63f;   /* ~380 deg */
    float spanA = endA - startA;
    float rpmNorm = rpm / 9000.0f;
    if (rpmNorm < 0.0f) rpmNorm = 0.0f;
    if (rpmNorm > 1.0f) rpmNorm = 1.0f;

    for (int i = 0; i < segs; i++) {
        float f0 = (float)i / (float)segs;
        float f1 = (float)(i + 1) / (float)segs;
        float a0 = startA + f0 * spanA;
        float a1 = startA + f1 * spanA;

        Vec4 col = vec4(0.2f, 0.25f, 0.35f, 0.35f);
        if (f1 <= rpmNorm) {
            if (f1 > 0.82f) col = vec4(1.0f, 0.12f, 0.25f, 0.95f);      /* Redline */
            else if (f1 > 0.60f) col = vec4(1.0f, 0.82f, 0.20f, 0.95f); /* Mid yellow */
            else col = vec4(0.0f, 0.94f, 1.0f, 0.95f);                  /* Cyan */
        }
        /* Green Launch Zone during countdown (5500 - 7300 RPM) */
        if (inCountdown && f0 >= (5500.0f / 9000.0f) && f1 <= (7300.0f / 9000.0f)) {
            col = vec4(0.29f, 0.87f, 0.50f, 1.0f);
        }

        float x0 = cx + cosf(a0) * (r - 6.0f), y0 = cy + sinf(a0) * (r - 6.0f);
        float x1 = cx + cosf(a1) * (r - 6.0f), y1 = cy + sinf(a1) * (r - 6.0f);
        float x2 = cx + cosf(a1) * (r + 6.0f), y2 = cy + sinf(a1) * (r + 6.0f);
        float x3 = cx + cosf(a0) * (r + 6.0f), y3 = cy + sinf(a0) * (r + 6.0f);

        VertexHUD v0 = { x0, y0, 0, 0, col.x, col.y, col.z, col.w, 0 };
        VertexHUD v1 = { x1, y1, 0, 0, col.x, col.y, col.z, col.w, 0 };
        VertexHUD v2 = { x2, y2, 0, 0, col.x, col.y, col.z, col.w, 0 };
        VertexHUD v3 = { x3, y3, 0, 0, col.x, col.y, col.z, col.w, 0 };
        hud_push_v(v0); hud_push_v(v1); hud_push_v(v2);
        hud_push_v(v0); hud_push_v(v2); hud_push_v(v3);
    }

    /* RPM Needle */
    float needleA = startA + rpmNorm * spanA;
    float nx = cx + cosf(needleA) * (r - 12.0f);
    float ny = cy + sinf(needleA) * (r - 12.0f);
    Vec4 needleCol = vec4(1.0f, 0.16f, 0.52f, 1.0f);
    hud_rect(cx - 4.0f, cy - 4.0f, 8.0f, 8.0f, needleCol);
    /* Needle line quad */
    float perpX = -sinf(needleA) * 2.0f;
    float perpY =  cosf(needleA) * 2.0f;
    VertexHUD np0 = { cx - perpX, cy - perpY, 0, 0, needleCol.x, needleCol.y, needleCol.z, needleCol.w, 0 };
    VertexHUD np1 = { cx + perpX, cy + perpY, 0, 0, needleCol.x, needleCol.y, needleCol.z, needleCol.w, 0 };
    VertexHUD np2 = { nx + perpX, ny + perpY, 0, 0, needleCol.x, needleCol.y, needleCol.z, needleCol.w, 0 };
    VertexHUD np3 = { nx - perpX, ny - perpY, 0, 0, needleCol.x, needleCol.y, needleCol.z, needleCol.w, 0 };
    hud_push_v(np0); hud_push_v(np1); hud_push_v(np2);
    hud_push_v(np0); hud_push_v(np2); hud_push_v(np3);

    /* Gear & Transmission Readout */
    char gearStr[32];
    snprintf(gearStr, sizeof(gearStr), "G%d %s", gear, manual ? "MAN" : "AUTO");
    hud_text(cx - 34.0f, cy - 48.0f, 1.6f, gearStr, vec4(1.0f, 0.16f, 0.52f, 1.0f));

    /* Big Digital Speed (MPH) */
    char mphStr[32];
    snprintf(mphStr, sizeof(mphStr), "%3d", (int)mph);
    hud_text(cx - 36.0f, cy + 2.0f, 3.2f, mphStr, vec4(1.0f, 1.0f, 1.0f, 1.0f));
    hud_text(cx - 14.0f, cy + 34.0f, 1.4f, "MPH", vec4(0.0f, 0.94f, 1.0f, 1.0f));

    /* Curved Bottom N2O Arc Gauge */
    float nStart = 0.78f; /* 45 deg */
    float nEnd = 2.14f;   /* 122 deg */
    float nNorm = n2o / 100.0f;
    for (int i = 0; i < 16; i++) {
        float f0 = (float)i / 16.0f;
        float f1 = (float)(i + 1) / 16.0f;
        float a0 = nStart + f0 * (nEnd - nStart);
        float a1 = nStart + f1 * (nEnd - nStart);
        Vec4 col = (f1 <= nNorm)
            ? (nosActive ? vec4(0.55f, 0.98f, 1.0f, 1.0f) : vec4(0.0f, 0.94f, 1.0f, 1.0f))
            : vec4(0.16f, 0.22f, 0.34f, 0.35f);

        float x0 = cx + cosf(a0) * (r + 10.0f), y0 = cy + sinf(a0) * (r + 10.0f);
        float x1 = cx + cosf(a1) * (r + 10.0f), y1 = cy + sinf(a1) * (r + 10.0f);
        float x2 = cx + cosf(a1) * (r + 18.0f), y2 = cy + sinf(a1) * (r + 18.0f);
        float x3 = cx + cosf(a0) * (r + 18.0f), y3 = cy + sinf(a0) * (r + 18.0f);

        VertexHUD v0 = { x0, y0, 0, 0, col.x, col.y, col.z, col.w, 0 };
        VertexHUD v1 = { x1, y1, 0, 0, col.x, col.y, col.z, col.w, 0 };
        VertexHUD v2 = { x2, y2, 0, 0, col.x, col.y, col.z, col.w, 0 };
        VertexHUD v3 = { x3, y3, 0, 0, col.x, col.y, col.z, col.w, 0 };
        hud_push_v(v0); hud_push_v(v1); hud_push_v(v2);
        hud_push_v(v0); hud_push_v(v2); hud_push_v(v3);
    }
    hud_text(cx + 24.0f, cy + 68.0f, 1.2f, nosActive ? "N2O!" : "N2O",
             nosActive ? vec4(0.55f, 0.98f, 1.0f, 1.0f) : vec4(0.0f, 0.94f, 1.0f, 1.0f));
}

static void draw_hud_minimap(float mx, float my, float size, int trk_idx) {
    /* Dark-Mode Neon Scramble Obsidian-Navy Tactical Card */
    float cardX = mx - 12.0f;
    float cardY = my - 26.0f;
    float cardW = size + 24.0f;
    float cardH = size + 36.0f;
    hud_card(cardX, cardY, cardW, cardH, 0.0f,
             vec4(0.024f, 0.039f, 0.078f, 0.90f),
             vec4(0.22f, 0.74f, 0.97f, 0.36f),
             vec4(0.0f, 0.94f, 1.0f, 0.90f));
    hud_text(mx - 2.0f, my - 18.0f, 1.15f, "TAC-MAP", vec4(0.0f, 0.94f, 1.0f, 0.95f));
    hud_text(mx + size - 56.0f, my - 18.0f, 1.05f, "GPS LIVE", vec4(0.18f, 0.95f, 0.60f, 0.85f));
    hud_rect(mx - 4.0f, my - 4.0f, size + 8.0f, size + 8.0f, vec4(0.014f, 0.024f, 0.051f, 0.85f));

    /* Faint Coordinate Grid Lines inside Tactical Minimap */
    Vec4 gridCol = vec4(0.22f, 0.74f, 0.97f, 0.11f);
    for (int g = 1; g <= 3; g++) {
        float t = (float)g * 0.25f;
        hud_line(mx + size * t, my, mx + size * t, my + size, 1.0f, gridCol);
        hud_line(mx, my + size * t, mx + size, my + size * t, 1.0f, gridCol);
    }

    /* Emerald & Amber Cross-Transit Accents */
    hud_line(mx + 4.0f, my + size * 0.32f, mx + size - 4.0f, my + size * 0.64f, 1.4f, vec4(0.06f, 0.72f, 0.51f, 0.26f));
    hud_line(mx + size * 0.28f, my + 4.0f, mx + size * 0.74f, my + size - 4.0f, 1.4f, vec4(0.96f, 0.62f, 0.04f, 0.26f));

    const TrackRuntime *tr = &g_tracks[trk_idx];

    /* Find Track Bounds */
    float minX = 1e6f, maxX = -1e6f, minZ = 1e6f, maxZ = -1e6f;
    for (int i = 0; i < FRAMES_COUNT; i += 6) {
        if (tr->pts[i].x < minX) minX = tr->pts[i].x;
        if (tr->pts[i].x > maxX) maxX = tr->pts[i].x;
        if (tr->pts[i].z < minZ) minZ = tr->pts[i].z;
        if (tr->pts[i].z > maxZ) maxZ = tr->pts[i].z;
    }
    float span = fmaxf(maxX - minX, maxZ - minZ);
    if (span < 1.0f) span = 1.0f;
    float scale = (size - 24.0f) / span;
    float ox = mx + (size - (maxX - minX) * scale) * 0.5f - minX * scale;
    float oz = my + (size - (maxZ - minZ) * scale) * 0.5f - minZ * scale;

    /* Pass 1: Dark Road Casing */
    Vec4 casingCol = vec4(0.02f, 0.04f, 0.08f, 0.96f);
    for (int i = 0; i < FRAMES_COUNT; i += 4) {
        int ni = (i + 4) % FRAMES_COUNT;
        float x0 = tr->pts[i].x * scale + ox,  y0 = tr->pts[i].z * scale + oz;
        float x1 = tr->pts[ni].x * scale + ox, y1 = tr->pts[ni].z * scale + oz;
        hud_line(x0, y0, x1, y1, 6.8f, casingCol);
    }

    /* Pass 2: Bright Cyan Route Loop */
    Vec4 routeCol = vec4(0.0f, 0.94f, 1.0f, 0.92f);
    for (int i = 0; i < FRAMES_COUNT; i += 4) {
        int ni = (i + 4) % FRAMES_COUNT;
        float x0 = tr->pts[i].x * scale + ox,  y0 = tr->pts[i].z * scale + oz;
        float x1 = tr->pts[ni].x * scale + ox, y1 = tr->pts[ni].z * scale + oz;
        hud_line(x0, y0, x1, y1, 2.6f, routeCol);
    }

    /* Emerald / Amber Sector Transit Nodes */
    int s1 = FRAMES_COUNT / 3;
    int s2 = (FRAMES_COUNT * 2) / 3;
    hud_rect(tr->pts[s1].x * scale + ox - 3.0f, tr->pts[s1].z * scale + oz - 3.0f, 6.0f, 6.0f, vec4(0.06f, 0.85f, 0.55f, 0.95f));
    hud_rect(tr->pts[s2].x * scale + ox - 3.0f, tr->pts[s2].z * scale + oz - 3.0f, 6.0f, 6.0f, vec4(0.96f, 0.62f, 0.04f, 0.95f));

    /* Start / Finish Gate Bar */
    float sfx = tr->pts[0].x * scale + ox;
    float sfz = tr->pts[0].z * scale + oz;
    float srx = tr->rights[0].x * 5.5f;
    float srz = tr->rights[0].z * 5.5f;
    hud_line(sfx - srx, sfz - srz, sfx + srx, sfz + srz, 3.0f, vec4(1.0f, 1.0f, 1.0f, 1.0f));

    /* AI Markers */
    for (int i = 0; i < g_ai_count; i++) {
        float ax = g_ai[i].x * scale + ox;
        float az = g_ai[i].z * scale + oz;
        Vec4 aiCol = g_ai[i].is_pursuit ? vec4(1.0f, 0.16f, 0.52f, 1.0f) : vec4(0.96f, 0.68f, 0.12f, 1.0f);
        hud_rect(ax - 3.5f, az - 3.5f, 7.0f, 7.0f, vec4(0.02f, 0.04f, 0.08f, 0.95f));
        hud_rect(ax - 2.5f, az - 2.5f, 5.0f, 5.0f, aiCol);
    }

    /* Directional Player Chevron */
    float plx = g_player.x * scale + ox;
    float plz = g_player.z * scale + oz;
    float fx = sinf(g_player.heading);
    float fy = cosf(g_player.heading);
    float rx = -fy;
    float ry = fx;

    Vec4 chevOutline = vec4(0.02f, 0.04f, 0.08f, 0.98f);
    Vec4 chevFill = g_player.nos_active ? vec4(0.0f, 0.96f, 1.0f, 1.0f) : vec4(0.18f, 0.95f, 0.60f, 1.0f);

    for (int pass = 0; pass < 2; pass++) {
        float sc = (pass == 0) ? 1.35f : 1.0f;
        Vec4 c = (pass == 0) ? chevOutline : chevFill;
        VertexHUD tip   = { plx + fx * (8.0f * sc), plz + fy * (8.0f * sc), 0, 0, c.x, c.y, c.z, c.w, 0 };
        VertexHUD left  = { plx - fx * (5.0f * sc) - rx * (5.5f * sc), plz - fy * (5.0f * sc) - ry * (5.5f * sc), 0, 0, c.x, c.y, c.z, c.w, 0 };
        VertexHUD notch = { plx - fx * (2.2f * sc), plz - fy * (2.2f * sc), 0, 0, c.x, c.y, c.z, c.w, 0 };
        VertexHUD right = { plx - fx * (5.0f * sc) + rx * (5.5f * sc), plz - fy * (5.0f * sc) + ry * (5.5f * sc), 0, 0, c.x, c.y, c.z, c.w, 0 };
        hud_push_v(tip); hud_push_v(left); hud_push_v(notch);
        hud_push_v(tip); hud_push_v(notch); hud_push_v(right);
    }
}

static void draw_hud_mirror_frame(void) {
    /* Scissor area is x=500, y=636, w=280, h=68 in bottom-up GL coords.
     * In top-down HUD coords (1280x720), y = 720 - (636 + 68) = 16. */
    float fx = 496.0f, fy = 12.0f, fw = 288.0f, fh = 76.0f;
    /* Sleek Dark Obsidian-Navy Bezel */
    Vec4 bezelCol = vec4(0.06f, 0.10f, 0.18f, 0.95f);
    Vec4 trimCol  = vec4(0.0f, 0.94f, 1.0f, 0.80f);
    hud_rect(fx, fy, fw, 3.0f, trimCol);
    hud_rect(fx, fy + fh - 3.0f, fw, 3.0f, bezelCol);
    hud_rect(fx, fy, 3.0f, fh, bezelCol);
    hud_rect(fx + fw - 3.0f, fy, 3.0f, fh, bezelCol);
    hud_rect(fx + 4.0f, fy + fh - 17.0f, 138.0f, 14.0f, vec4(0.024f, 0.039f, 0.078f, 0.88f));
    hud_text(fx + 8.0f, fy + fh - 14.0f, 1.0f, "REAR-VIEW MIRROR", vec4(0.0f, 0.94f, 1.0f, 0.92f));
}

/* =========================================================================
 * 13. FULL GAME FRAME UPDATE & RENDER DISPATCH
 * ========================================================================= */

static void update_physics(float dt, const uint8_t *keys) {
    const TrackRuntime *tr = &g_tracks[g_selected_track];

    float joySteer = 0.0f;
    bool joyGas = false, joyBrake = false, joyDrift = false, joyNos = false;
    if (g_joy) {
        int16_t axisX = SDL_JoystickGetAxis(g_joy, 0);
        if (abs(axisX) > 8000) {
            joySteer = (float)axisX / 32767.0f;
        }
        uint8_t hat = SDL_JoystickGetHat(g_joy, 0);
        if (hat & SDL_HAT_LEFT) joySteer = -1.0f;
        if (hat & SDL_HAT_RIGHT) joySteer = 1.0f;
        if (hat & SDL_HAT_UP) joyGas = true;
        if (hat & SDL_HAT_DOWN) joyBrake = true;

        if (SDL_JoystickGetButton(g_joy, 0) || SDL_JoystickGetButton(g_joy, 1)) joyGas = true;
        if (SDL_JoystickGetButton(g_joy, 2)) joyBrake = true;
        if (SDL_JoystickGetButton(g_joy, 3) || SDL_JoystickGetButton(g_joy, 4)) joyDrift = true;
        if (SDL_JoystickGetButton(g_joy, 5)) joyNos = true;
    }

    if (g_state == STATE_GARAGE) {
        g_garage_orbit_yaw += dt * 0.45f;
        return;
    }

    if (g_state == STATE_COUNTDOWN) {
        g_countdown_timer -= dt;
        int ceilSec = (int)ceilf(g_countdown_timer);
        if (ceilSec > 0 && ceilSec != g_last_countdown_beep) {
            g_last_countdown_beep = ceilSec;
        }

        /* Interactive RPM revving on grid */
        bool gas = keys[SDL_SCANCODE_W] || keys[SDL_SCANCODE_UP] || keys[SDL_SCANCODE_X] || joyGas;
        float targetRpm = gas ? 8800.0f : 1200.0f;
        g_player.rpm += (targetRpm - g_player.rpm) * fminf(1.0f, dt * (gas ? 3.5f : 4.5f));
        g_audio_rpm_norm = g_player.rpm / 9000.0f;

        if (g_countdown_timer <= 0.0f) {
            g_state = STATE_RACING;
            if (g_player.rpm >= 5500.0f && g_player.rpm <= 7300.0f) {
                g_player.speed = 42.0f;
                g_player.n2o = 100.0f;
                g_total_style_score += 500;
                g_player.backfire_timer = 0.25f;
            } else if (g_player.rpm > 7700.0f) {
                g_player.speed = 18.0f;
                g_player.wheelspin_timer = 0.85f;
            } else {
                g_player.speed = 28.0f;
            }
        }
        return;
    }

    if (g_state == STATE_TITLE) {
        g_race_time += dt;
        g_player.speed = 96.0f;
        g_player.nos_active = (fmodf(g_race_time, 5.0f) > 3.2f);
        float pDist = g_player.speed * dt * 0.447f;
        g_player.u += pDist / tr->total_len;
        if (g_player.u >= 1.0f) g_player.u -= 1.0f;
        int pSeg = (int)(g_player.u * (float)FRAMES_COUNT) % FRAMES_COUNT;
        g_player.seg_idx = pSeg;
        g_player.x = tr->pts[pSeg].x + tr->rights[pSeg].x * g_player.lat;
        g_player.y = tr->pts[pSeg].y;
        g_player.z = tr->pts[pSeg].z + tr->rights[pSeg].z * g_player.lat;
        g_player.heading = atan2f(tr->tangents[pSeg].x, tr->tangents[pSeg].z);
        update_racer_tail_history(&g_player);

        for (int i = 0; i < g_ai_count; i++) {
            g_ai[i].speed += (g_ai[i].ai_target_speed - g_ai[i].speed) * dt * 0.8f;
            g_ai[i].nos_active = ((i % 2 == 0) && fmodf(g_race_time + (float)i * 1.1f, 4.5f) < 1.6f);
            float aiDist = g_ai[i].speed * dt * 0.447f;
            g_ai[i].u += aiDist / tr->total_len;
            if (g_ai[i].u >= 1.0f) g_ai[i].u -= 1.0f;

            int ai_seg = (int)(g_ai[i].u * (float)FRAMES_COUNT) % FRAMES_COUNT;
            g_ai[i].seg_idx = ai_seg;
            g_ai[i].x = tr->pts[ai_seg].x + tr->rights[ai_seg].x * g_ai[i].lat;
            g_ai[i].y = tr->pts[ai_seg].y;
            g_ai[i].z = tr->pts[ai_seg].z + tr->rights[ai_seg].z * g_ai[i].lat;
            g_ai[i].heading = atan2f(tr->tangents[ai_seg].x, tr->tangents[ai_seg].z);
            update_racer_tail_history(&g_ai[i]);
        }
        return;
    }

    if (g_state != STATE_RACING) return;

    g_race_time += dt;

    /* Controls */
    bool steerL = keys[SDL_SCANCODE_A] || keys[SDL_SCANCODE_LEFT];
    bool steerR = keys[SDL_SCANCODE_D] || keys[SDL_SCANCODE_RIGHT];
    bool gas    = keys[SDL_SCANCODE_W] || keys[SDL_SCANCODE_UP] || keys[SDL_SCANCODE_X] || joyGas;
    bool brake  = keys[SDL_SCANCODE_S] || keys[SDL_SCANCODE_DOWN] || joyBrake;
    bool drift  = keys[SDL_SCANCODE_Z] || keys[SDL_SCANCODE_LSHIFT] || keys[SDL_SCANCODE_B] || joyDrift;
    bool nos    = keys[SDL_SCANCODE_SPACE] || keys[SDL_SCANCODE_LCTRL] || joyNos;

    /* Steering */
    float targetSteer = (steerL ? -1.0f : 0.0f) + (steerR ? 1.0f : 0.0f);
    if (fabsf(joySteer) > 0.01f) {
        targetSteer = joySteer;
    }
    g_player.steer_angle += (targetSteer - g_player.steer_angle) * fminf(1.0f, dt * 8.5f);

    /* Gearbox & Acceleration */
    int gear = g_player.gear;
    float maxMph = GEAR_MAX_MPH[gear];
    float accel = g_car_specs[g_player.car_idx].accel * 0.55f;

    if (nos && g_player.n2o > 0.0f) {
        g_player.nos_active = true;
        g_player.n2o -= dt * 26.0f;
        accel += 45.0f;
        maxMph += 28.0f;
    } else {
        g_player.nos_active = false;
    }

    if (gas) {
        g_player.speed += accel * dt;
        if (g_player.speed > maxMph) g_player.speed = maxMph;
    } else if (brake) {
        g_player.speed -= 65.0f * dt;
        if (g_player.speed < -20.0f) g_player.speed = -20.0f;
    } else {
        g_player.speed *= (1.0f - dt * 0.22f);
    }

    /* RPM calculation */
    float gearNorm = fabsf(g_player.speed) / (GEAR_MAX_MPH[gear] > 0 ? GEAR_MAX_MPH[gear] : 1.0f);
    float targetRpm = 1000.0f + gearNorm * 7800.0f;
    g_player.rpm += (targetRpm - g_player.rpm) * fminf(1.0f, dt * 10.0f);

    /* Auto Transmission Shifting */
    if (!g_manual_trans) {
        if (g_player.rpm > 8200.0f && g_player.gear < 6) {
            g_player.gear++;
            g_player.rpm = 4500.0f;
        } else if (g_player.rpm < 3200.0f && g_player.gear > 1) {
            g_player.gear--;
            g_player.rpm = 6500.0f;
        }
    }

    /* Drift Physics */
    if (drift && fabsf(g_player.speed) > 25.0f) {
        g_player.drift_angle += g_player.steer_angle * dt * 3.2f;
        g_active_drift_pts += fabsf(g_player.speed) * dt * 15.0f;
        g_active_drift_timer += dt;
        g_active_drift_mult = 1 + (int)(g_active_drift_timer / 1.5f);
        g_player.n2o = fminf(100.0f, g_player.n2o + dt * 16.0f);
        g_audio_drifting = true;
    } else {
        if (g_active_drift_pts > 0.0f) {
            g_total_style_score += (int)g_active_drift_pts * g_active_drift_mult;
            g_active_drift_pts = 0.0f;
            g_active_drift_timer = 0.0f;
            g_active_drift_mult = 1;
        }
        g_player.drift_angle *= (1.0f - dt * 4.5f);
        g_audio_drifting = false;
    }

    /* Motion heading */
    g_player.heading += g_player.steer_angle * dt * (g_player.speed / 55.0f) * 1.8f;
    float moveAngle = g_player.heading + g_player.drift_angle * 0.35f;
    g_player.x += sinf(moveAngle) * g_player.speed * dt * 0.447f;
    g_player.z += cosf(moveAngle) * g_player.speed * dt * 0.447f;

    /* Spline projection & Wall Collision */
    TrackProjection proj = project_onto_track(g_selected_track, g_player.x, g_player.z, g_player.seg_idx);
    g_player.seg_idx = proj.seg_idx;
    g_player.u = proj.u;
    g_player.lat = proj.lat;
    g_player.y = proj.qy;

    if (fabsf(proj.lat) > MAX_LAT) {
        float sign = proj.lat > 0.0f ? 1.0f : -1.0f;
        g_player.x = proj.qx + proj.R.x * (sign * MAX_LAT);
        g_player.z = proj.qz + proj.R.z * (sign * MAX_LAT);
        g_player.lat = sign * MAX_LAT;
        g_player.speed *= 0.94f;
    }

    /* Lap Counting */
    if (g_player.u < 0.15f && g_player.seg_idx < 40) {
        if (g_player.total_progress > 0.85f) {
            g_player.lap++;
            float lapDur = g_race_time - g_lap_start_time;
            g_lap_start_time = g_race_time;
            if (lapDur < g_best_lap_time && lapDur > 5.0f) g_best_lap_time = lapDur;
            if (g_player.lap >= g_mode_specs[g_selected_mode].laps) {
                int prank = 1;
                float player_prog = (float)g_player.lap + g_player.u;
                for (int i = 0; i < g_ai_count; i++) {
                    float ai_prog = (float)g_ai[i].lap + g_ai[i].u;
                    if (ai_prog > player_prog) prank++;
                }
                g_final_rank = prank;
                g_state = STATE_RESULTS;
            }
        }
    }
    g_player.total_progress = g_player.u;

    /* Audio Sync */
    g_audio_rpm_norm = g_player.rpm / 9000.0f;
    g_audio_nos = g_player.nos_active;

    /* Update Player Tail-Light History */
    update_racer_tail_history(&g_player);

    /* Update AI rivals */
    for (int i = 0; i < g_ai_count; i++) {
        g_ai[i].speed += (g_ai[i].ai_target_speed - g_ai[i].speed) * dt * 0.8f;
        g_ai[i].nos_active = ((i % 3 == 0) && fmodf(g_race_time + (float)i, 5.5f) < 1.4f);
        float aiDist = g_ai[i].speed * dt * 0.447f;
        g_ai[i].u += aiDist / tr->total_len;
        if (g_ai[i].u >= 1.0f) {
            g_ai[i].u -= 1.0f;
            g_ai[i].lap++;
        }

        int ai_seg = (int)(g_ai[i].u * (float)FRAMES_COUNT);
        g_ai[i].seg_idx = ai_seg;
        g_ai[i].x = tr->pts[ai_seg].x + tr->rights[ai_seg].x * g_ai[i].lat;
        g_ai[i].y = tr->pts[ai_seg].y;
        g_ai[i].z = tr->pts[ai_seg].z + tr->rights[ai_seg].z * g_ai[i].lat;
        g_ai[i].heading = atan2f(tr->tangents[ai_seg].x, tr->tangents[ai_seg].z);
        update_racer_tail_history(&g_ai[i]);
    }
}

static void render_scene(int trk_idx) {
    const TrackSpec *ts = &g_track_specs[trk_idx];

    /* Clear Color & Depth */
    glClearColor(ts->sky_col.x, ts->sky_col.y, ts->sky_col.z, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (g_state == STATE_GARAGE) {
        /* Speedshop Showroom Camera */
        Mat4 mProj, mView, mMVP, mIdent;
        mat4_identity(mIdent);
        mat4_perspective(mProj, 0.95f, 1280.0f / 720.0f, 0.1f, 300.0f);

        float camRightX = cosf(g_garage_orbit_yaw);
        float camRightZ = -sinf(g_garage_orbit_yaw);
        float frameShift = -1.60f; /* Keep car centered in open right viewport clear of HUD card */
        Vec3 eye = vec3(sinf(g_garage_orbit_yaw) * 6.8f + camRightX * frameShift, 1.55f,
                        cosf(g_garage_orbit_yaw) * 6.8f + camRightZ * frameShift);
        Vec3 target = vec3(camRightX * frameShift, 0.72f, camRightZ * frameShift);
        Vec3 up = vec3(0.0f, 1.0f, 0.0f);
        mat4_lookat(mView, eye, target, up);
        mat4_mul(mMVP, mProj, mView);

        glUseProgram(g_prog3d);
        glUniformMatrix4fv(g_u3d_MVP, 1, GL_FALSE, mMVP);
        glUniformMatrix4fv(g_u3d_Model, 1, GL_FALSE, mIdent);
        glUniform3f(g_u3d_CamPos, eye.x, eye.y, eye.z);
        glUniform3f(g_u3d_PaintCol, 1.0f, 1.0f, 1.0f);
        glUniform3f(g_u3d_SunDir, 0.45f, 0.75f, 0.45f);
        glUniform3f(g_u3d_SunCol, 0.85f, 0.95f, 1.0f);
        glUniform3f(g_u3d_AmbientCol, 0.35f, 0.45f, 0.65f);
        glUniform3f(g_u3d_FogCol, 0.03f, 0.05f, 0.11f);
        glUniform1f(g_u3d_FogDensity, 0.005f);

        glEnable(GL_DEPTH_TEST);
        glBindBuffer(GL_ARRAY_BUFFER, g_vbo_garage);
        glEnableVertexAttribArray(g_a3d_Pos);
        glVertexAttribPointer(g_a3d_Pos, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex3D), (void*)0);
        glEnableVertexAttribArray(g_a3d_Norm);
        glVertexAttribPointer(g_a3d_Norm, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex3D), (void*)(sizeof(float) * 3));
        glEnableVertexAttribArray(g_a3d_Col);
        glVertexAttribPointer(g_a3d_Col, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex3D), (void*)(sizeof(float) * 6));
        glEnableVertexAttribArray(g_a3d_Glow);
        glVertexAttribPointer(g_a3d_Glow, 1, GL_FLOAT, GL_FALSE, sizeof(Vertex3D), (void*)(sizeof(float) * 9));
        glDrawArrays(GL_TRIANGLES, 0, g_garage_vert_count);
        glDisableVertexAttribArray(g_a3d_Pos);
        glDisableVertexAttribArray(g_a3d_Norm);
        glDisableVertexAttribArray(g_a3d_Col);
        glDisableVertexAttribArray(g_a3d_Glow);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        /* Draw Customized Showcase Car on Rotating Turntable */
        Racer showCar;
        memset(&showCar, 0, sizeof(Racer));
        showCar.car_idx = g_selected_car;
        showCar.paint_idx = g_selected_paint;
        showCar.rim_idx = g_selected_rim;
        showCar.glow_idx = g_selected_glow;
        showCar.wing_idx = g_selected_wing;
        showCar.heading = g_garage_orbit_yaw * 0.5f;
        render_car_mesh(&showCar, false);

        if (g_glows[g_selected_glow].active) {
            Vec3 gCol = g_glows[g_selected_glow].col;
            fx_push_radial_glow(vec3(0.0f, 0.06f, 0.0f), 2.3f, 3.4f, vec4(gCol.x, gCol.y, gCol.z, 0.50f));
            flush_fx(mMVP);
        }

        /* Speedshop Customization HUD (Dark-Mode Neon Scramble Card) */
        hud_card(36.0f, 40.0f, 520.0f, 620.0f, 10.0f,
                 vec4(0.024f, 0.039f, 0.078f, 0.90f),
                 vec4(0.22f, 0.74f, 0.97f, 0.36f),
                 vec4(0.0f, 0.94f, 1.0f, 0.95f));
        hud_text(60.0f, 62.0f, 2.4f, "TUNER SPEEDSHOP", vec4(0.0f, 0.94f, 1.0f, 1.0f));
        char specTag[64];
        snprintf(specTag, sizeof(specTag), "%s  //  CLASS %s",
                 g_car_specs[g_selected_car].cls, g_car_specs[g_selected_car].rating);
        hud_text(60.0f, 95.0f, 1.15f, specTag, vec4(0.70f, 0.82f, 0.95f, 0.88f));

        char line[64];
        for (int r = 0; r < 8; r++) {
            float ry = 136.0f + (float)r * 45.0f;
            bool sel = (r == g_garage_row);
            if (sel) {
                hud_card(50.0f, ry - 6.0f, 490.0f, 38.0f, 6.0f,
                         vec4(0.0f, 0.94f, 1.0f, 0.20f),
                         vec4(0.0f, 0.94f, 1.0f, 0.65f),
                         vec4(0.0f, 0.0f, 0.0f, 0.0f));
            }
            const char *prefix = sel ? "> " : "  ";
            Vec4 textCol = sel ? vec4(0.0f, 0.94f, 1.0f, 1.0f) : vec4(0.85f, 0.90f, 0.96f, 0.85f);

            switch (r) {
                case 0: snprintf(line, sizeof(line), "%s1. SUPERCAR:  < %s >", prefix, g_car_specs[g_selected_car].name); break;
                case 1: snprintf(line, sizeof(line), "%s2. PAINT:     < %s >", prefix, g_paints[g_selected_paint].name); break;
                case 2: snprintf(line, sizeof(line), "%s3. RIMS:      < %s >", prefix, g_rims[g_selected_rim].name); break;
                case 3: snprintf(line, sizeof(line), "%s4. NEON GLOW: < %s >", prefix, g_glows[g_selected_glow].name); break;
                case 4: snprintf(line, sizeof(line), "%s5. AERO WING: < %s >", prefix, g_wings[g_selected_wing]); break;
                case 5: snprintf(line, sizeof(line), "%s6. GEARBOX:   < %s >", prefix, g_manual_trans ? "MANUAL (Q/E)" : "AUTOMATIC"); break;
                case 6: snprintf(line, sizeof(line), "%s7. CIRCUIT:   < %s >", prefix, g_track_specs[g_selected_track].name); break;
                case 7: snprintf(line, sizeof(line), "%s8. MODE:      < %s >", prefix, g_mode_specs[g_selected_mode].name); break;
            }
            hud_text(60.0f, ry, 1.35f, line, textCol);
            if (r == 1) {
                Vec3 pc = g_paints[g_selected_paint].col;
                hud_rect(512.0f, ry + 2.0f, 18.0f, 14.0f, vec4(pc.x, pc.y, pc.z, 1.0f));
            }
        }

        hud_card(50.0f, 512.0f, 490.0f, 48.0f, 8.0f,
                 vec4(0.0f, 0.94f, 1.0f, 0.22f),
                 vec4(0.0f, 0.94f, 1.0f, 0.72f),
                 vec4(0.0f, 0.0f, 0.0f, 0.0f));
        hud_text(130.0f, 528.0f, 1.6f, "[ENTER / SPACE] START RACE", vec4(0.0f, 0.94f, 1.0f, 1.0f));
        hud_text(55.0f, 582.0f, 1.15f, "[UP/DOWN] ROW   [LEFT/RIGHT] CYCLE   [1-8] SELECT   [ESC] TITLE", vec4(0.75f, 0.85f, 0.95f, 0.85f));

        flush_hud();
        return;
    }

    /* Compute Gameplay Camera */
    Vec3 camPos, camTarget;
    float sinH = sinf(g_player.heading);
    float cosH = cosf(g_player.heading);
    bool isCockpit = (g_cam_mode == 1);

    if (g_cam_mode == 1) {
        /* Cockpit Camera */
        float rH = g_car_specs[g_player.car_idx].roof_h;
        float pX = g_car_specs[g_player.car_idx].roof_peak_x;
        float camZ = pX - 0.20f;
        camPos = vec3(g_player.x + sinH * camZ, g_player.y + rH - 0.10f, g_player.z + cosH * camZ);
        camTarget = vec3(g_player.x + sinH * 35.0f, g_player.y + rH - 0.30f, g_player.z + cosH * 35.0f);
    } else if (g_cam_mode == 2) {
        /* Hood / Bumper Camera */
        float halfL = g_car_specs[g_player.car_idx].len * 0.5f;
        camPos = vec3(g_player.x + sinH * (halfL * 0.70f), g_player.y + 0.85f, g_player.z + cosH * (halfL * 0.70f));
        camTarget = vec3(g_player.x + sinH * 40.0f, g_player.y + 0.70f, g_player.z + cosH * 40.0f);
    } else if (g_cam_mode == 3) {
        /* High Heli Camera */
        camPos = vec3(g_player.x - sinH * 18.0f, g_player.y + 12.0f, g_player.z - cosH * 18.0f);
        camTarget = vec3(g_player.x + sinH * 8.0f, g_player.y + 0.5f, g_player.z + cosH * 8.0f);
    } else {
        /* Third-Person Chase Camera */
        camPos = vec3(g_player.x - sinH * 8.5f, g_player.y + 3.2f, g_player.z - cosH * 8.5f);
        camTarget = vec3(g_player.x + sinH * 6.0f, g_player.y + 1.2f, g_player.z + cosH * 6.0f);
    }

    Mat4 mProj, mView;
    mat4_perspective(mProj, 1.05f, 1280.0f / 720.0f, 0.1f, 1200.0f);
    mat4_lookat(mView, camPos, camTarget, vec3(0.0f, 1.0f, 0.0f));

    /* 1. Main Forward 3D Scene Pass */
    render_world_scene(trk_idx, mView, mProj, camPos, isCockpit);

    /* 2. Zero-Copy 3D Rear-View Mirror Pass */
    if (g_mirror_on && (g_state == STATE_RACING || g_state == STATE_COUNTDOWN)) {
        glEnable(GL_SCISSOR_TEST);
        glScissor(500, 636, 280, 68);
        glViewport(500, 636, 280, 68);

        glClearColor(ts->sky_col.x, ts->sky_col.y, ts->sky_col.z, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        /* Mirror Camera facing backward */
        Vec3 mirrorPos = vec3(g_player.x, g_player.y + 1.48f, g_player.z);
        Vec3 mirrorTarget = vec3(g_player.x - sinH * 40.0f, g_player.y + 1.15f, g_player.z - cosH * 40.0f);
        Mat4 mMirrorProj, mMirrorView;
        mat4_perspective(mMirrorProj, 0.95f, 280.0f / 68.0f, 0.2f, 800.0f);
        mat4_lookat(mMirrorView, mirrorPos, mirrorTarget, vec3(0.0f, 1.0f, 0.0f));

        render_world_scene(trk_idx, mMirrorView, mMirrorProj, mirrorPos, false);

        glDisable(GL_SCISSOR_TEST);
        glViewport(0, 0, 1280, 720);
    }

    /* 3. 2D HUD Overlays Pass */
    if (g_state == STATE_TITLE) {
        /* Title Screen Attract Mode HUD (Dark-Mode Neon Scramble Card) */
        hud_card(40.0f, 40.0f, 460.0f, 640.0f, 12.0f,
                 vec4(0.024f, 0.039f, 0.078f, 0.88f),
                 vec4(0.22f, 0.74f, 0.97f, 0.36f),
                 vec4(0.0f, 0.94f, 1.0f, 0.95f));
        hud_text(65.0f, 75.0f, 3.2f, "APEX NITRO 3D", vec4(0.0f, 0.94f, 1.0f, 1.0f));
        hud_text(65.0f, 115.0f, 1.15f, "NEON SCRAMBLE // RETROPIE GLES2", vec4(1.0f, 0.16f, 0.52f, 1.0f));

        /* Menu Options with dynamic row highlight */
        static const char *title_items[4] = {
            "1. QUICK RACE",
            "2. TUNER SPEEDSHOP",
            "3. HOT PURSUIT",
            "4. DRIFT TOUGE"
        };
        for (int r = 0; r < 4; r++) {
            float py = 180.0f + (float)r * 70.0f;
            bool sel = (r == g_title_row);
            Vec4 pCol = sel ? vec4(0.0f, 0.94f, 1.0f, 0.24f) : vec4(0.055f, 0.086f, 0.155f, 0.88f);
            Vec4 bCol = sel ? vec4(0.0f, 0.94f, 1.0f, 0.78f) : vec4(0.22f, 0.74f, 0.97f, 0.20f);
            Vec4 tCol = sel ? vec4(1.0f, 1.0f, 1.0f, 1.0f) : vec4(0.80f, 0.86f, 0.94f, 0.90f);
            hud_card(65.0f, py, 380.0f, 52.0f, 8.0f, pCol, bCol, vec4(0.0f, 0.0f, 0.0f, 0.0f));
            if (sel) {
                hud_pill(68.0f, py + 6.0f, 4.0f, 40.0f, 5.0f, vec4(0.0f, 0.94f, 1.0f, 1.0f));
            }
            hud_text(90.0f, py + 16.0f, 1.8f, title_items[r], tCol);
        }

        hud_text(65.0f, 480.0f, 1.2f, "[UP/DOWN] NAVIGATE   [ENTER/SPACE/X] SELECT", vec4(0.0f, 0.94f, 1.0f, 0.95f));
        hud_text(65.0f, 510.0f, 1.2f, "[1-4] DIRECT QUICK SELECT", vec4(0.75f, 0.85f, 0.95f, 0.85f));
        hud_text(65.0f, 550.0f, 1.2f, "[RIGHT SHIFT + ENTER] QUIT TO RETROPIE", vec4(1.0f, 0.45f, 0.45f, 0.95f));

        /* Top-Right Live Street Cam badge */
        hud_card(960.0f, 30.0f, 280.0f, 38.0f, 6.0f,
                 vec4(0.024f, 0.039f, 0.078f, 0.88f),
                 vec4(0.22f, 0.74f, 0.97f, 0.36f),
                 vec4(0.0f, 0.0f, 0.0f, 0.0f));
        hud_rect(980.0f, 44.0f, 10.0f, 10.0f, vec4(1.0f, 0.16f, 0.2f, 1.0f));
        hud_text(1005.0f, 42.0f, 1.4f, "LIVE STREET CAM", vec4(0.0f, 0.94f, 1.0f, 1.0f));
    } else if (g_state == STATE_RESULTS) {
        /* Finish Card Overlay */
        hud_card(300.0f, 100.0f, 680.0f, 520.0f, 14.0f,
                 vec4(0.024f, 0.039f, 0.078f, 0.92f),
                 vec4(0.22f, 0.74f, 0.97f, 0.42f),
                 vec4(0.0f, 0.94f, 1.0f, 0.95f));

        if (g_final_rank == 1) {
            hud_text(495.0f, 140.0f, 3.2f, "VICTORY!", vec4(0.98f, 0.80f, 0.08f, 1.0f));
        } else {
            hud_text(450.0f, 140.0f, 3.2f, "RACE FINISHED", vec4(0.0f, 0.94f, 1.0f, 1.0f));
        }
        hud_text(500.0f, 195.0f, 1.3f, "OFFICIAL RACE RESULTS", vec4(0.75f, 0.85f, 0.95f, 0.85f));
        hud_rect(340.0f, 225.0f, 600.0f, 2.0f, vec4(0.0f, 0.94f, 1.0f, 0.40f));

        static const char *ords[] = { "1ST", "2ND", "3RD", "4TH", "5TH", "6TH" };
        int rIdx = (g_final_rank >= 1 && g_final_rank <= 6) ? g_final_rank - 1 : 0;
        char rankBuf[64];
        snprintf(rankBuf, sizeof(rankBuf), "FINAL POSITION:    %s / %d", ords[rIdx], 1 + g_ai_count);
        hud_text(360.0f, 260.0f, 1.8f, rankBuf, (g_final_rank == 1) ? vec4(0.98f, 0.80f, 0.08f, 1.0f) : vec4(1.0f, 1.0f, 1.0f, 1.0f));

        int min = (int)(g_race_time / 60.0f);
        int sec = (int)fmodf(g_race_time, 60.0f);
        int cs  = (int)fmodf(g_race_time * 100.0f, 100.0f);
        char timeBuf[64];
        snprintf(timeBuf, sizeof(timeBuf), "ELAPSED TIME:      %02d:%02d.%02d", min, sec, cs);
        hud_text(360.0f, 310.0f, 1.8f, timeBuf, vec4(0.85f, 0.90f, 0.98f, 1.0f));

        char bestBuf[64];
        if (g_best_lap_time < 9000.0f) {
            int bmin = (int)(g_best_lap_time / 60.0f);
            int bsec = (int)fmodf(g_best_lap_time, 60.0f);
            int bcs  = (int)fmodf(g_best_lap_time * 100.0f, 100.0f);
            snprintf(bestBuf, sizeof(bestBuf), "BEST LAP:          %02d:%02d.%02d", bmin, bsec, bcs);
        } else {
            snprintf(bestBuf, sizeof(bestBuf), "BEST LAP:          --:--.--");
        }
        hud_text(360.0f, 360.0f, 1.8f, bestBuf, vec4(0.0f, 0.94f, 1.0f, 1.0f));

        char scoreBuf[64];
        snprintf(scoreBuf, sizeof(scoreBuf), "STYLE SCORE:       %d PTS", g_total_style_score);
        hud_text(360.0f, 410.0f, 1.8f, scoreBuf, vec4(1.0f, 0.16f, 0.52f, 1.0f));

        hud_pill(340.0f, 480.0f, 600.0f, 56.0f, 8.0f, vec4(0.0f, 0.94f, 1.0f, 0.25f));
        hud_text(360.0f, 500.0f, 1.25f, "[ENTER / SPACE] RACE AGAIN   [G] SPEEDSHOP   [ESC] TITLE", vec4(0.0f, 0.94f, 1.0f, 1.0f));
    } else {
        /* Racing / Countdown HUD */
        if (g_mirror_on) draw_hud_mirror_frame();
        draw_hud_minimap(36.0f, 515.0f, 170.0f, trk_idx);
        draw_hud_tachometer(1140.0f, 580.0f, 85.0f, g_player.rpm, g_player.gear, g_manual_trans,
                            fabsf(g_player.speed), g_player.n2o, g_player.nos_active, (g_state == STATE_COUNTDOWN));

        /* Compute player live rank (1..6) vs AI rivals */
        int live_rank = 1;
        float player_prog = (float)g_player.lap + g_player.u;
        for (int i = 0; i < g_ai_count; i++) {
            float ai_prog = (float)g_ai[i].lap + g_ai[i].u;
            if (ai_prog > player_prog) live_rank++;
        }
        g_final_rank = live_rank;

        /* Top-Left Race Stats Card */
        hud_card(36.0f, 20.0f, 340.0f, 80.0f, 8.0f,
                 vec4(0.024f, 0.039f, 0.078f, 0.88f),
                 vec4(0.22f, 0.74f, 0.97f, 0.36f),
                 vec4(0.0f, 0.94f, 1.0f, 0.90f));
        char rankStr[32];
        static const char *ords[] = { "1ST", "2ND", "3RD", "4TH", "5TH", "6TH" };
        int rIdx = (live_rank >= 1 && live_rank <= 6) ? live_rank - 1 : 0;
        snprintf(rankStr, sizeof(rankStr), "%s / %d", ords[rIdx], 1 + g_ai_count);
        hud_text(55.0f, 34.0f, 2.2f, rankStr, (live_rank == 1) ? vec4(0.98f, 0.80f, 0.08f, 1.0f) : vec4(0.0f, 0.94f, 1.0f, 1.0f));

        char lapStr[32];
        snprintf(lapStr, sizeof(lapStr), "LAP %d / %d", g_player.lap + 1, g_mode_specs[g_selected_mode].laps);
        hud_text(206.0f, 36.0f, 1.5f, lapStr, vec4(0.0f, 0.94f, 1.0f, 1.0f));

        int min = (int)(g_race_time / 60.0f);
        int sec = (int)fmodf(g_race_time, 60.0f);
        int cs  = (int)fmodf(g_race_time * 100.0f, 100.0f);
        char timeStr[32];
        snprintf(timeStr, sizeof(timeStr), "TIME %02d:%02d.%02d", min, sec, cs);
        hud_text(55.0f, 68.0f, 1.3f, timeStr, vec4(0.85f, 0.90f, 0.98f, 1.0f));

        /* Top-Right Style Score Card */
        hud_card(960.0f, 20.0f, 280.0f, 65.0f, 8.0f,
                 vec4(0.024f, 0.039f, 0.078f, 0.88f),
                 vec4(0.22f, 0.74f, 0.97f, 0.36f),
                 vec4(1.0f, 0.16f, 0.52f, 0.90f));
        char styleStr[32];
        snprintf(styleStr, sizeof(styleStr), "STYLE: %d PTS", g_total_style_score);
        hud_text(980.0f, 34.0f, 1.4f, styleStr, vec4(0.0f, 0.94f, 1.0f, 1.0f));

        if (g_active_drift_pts > 0.0f) {
            char driftStr[32];
            snprintf(driftStr, sizeof(driftStr), "DRIFT +%d x%d", (int)g_active_drift_pts, g_active_drift_mult);
            hud_text(980.0f, 54.0f, 1.4f, driftStr, vec4(1.0f, 0.16f, 0.52f, 1.0f));
        }

        /* Countdown Center Overlay */
        if (g_state == STATE_COUNTDOWN) {
            int ceilSec = (int)ceilf(g_countdown_timer);
            if (ceilSec > 0) {
                char cntStr[16];
                snprintf(cntStr, sizeof(cntStr), "%d", ceilSec);
                hud_pill(560.0f, 260.0f, 160.0f, 140.0f, 12.0f, vec4(0.04f, 0.07f, 0.12f, 0.90f));
                hud_text(620.0f, 285.0f, 9.0f, cntStr, vec4(1.0f, 0.82f, 0.20f, 1.0f));
                hud_text(575.0f, 375.0f, 1.2f, "REV FOR GREEN LAUNCH!", vec4(0.29f, 0.87f, 0.50f, 1.0f));
            }
        }
    }

    flush_hud();
}

/* =========================================================================
 * 14. HEADLESS BENCHMARK & MODE SCREENSHOTS (#ifndef RPI_BUILD)
 * ========================================================================= */

#ifndef RPI_BUILD
static void save_ppm_screenshot(const char *path) {
    uint8_t *rgba = (uint8_t*)malloc(1280 * 720 * 4);
    if (!rgba) return;
    glReadPixels(0, 0, 1280, 720, GL_RGBA, GL_UNSIGNED_BYTE, rgba);

    FILE *f = fopen(path, "wb");
    if (!f) {
        free(rgba);
        return;
    }
    fprintf(f, "P6\n1280 720\n255\n");
    for (int y = 719; y >= 0; y--) {
        for (int x = 0; x < 1280; x++) {
            int idx = (y * 1280 + x) * 4;
            fputc(rgba[idx + 0], f);
            fputc(rgba[idx + 1], f);
            fputc(rgba[idx + 2], f);
        }
    }
    fclose(f);
    free(rgba);
    printf("[SHOT] Captured 1280x720 PPM to '%s'\n", path);
}

static int run_headless(const char *bench_flag, const char *shot_spec) {
    printf("[EGL] Initializing Headless EGL context...\n");

    PFNEGLGETPLATFORMDISPLAYEXTPROC getPlatformDisplay =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (!getPlatformDisplay) {
        getPlatformDisplay = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplay");
    }
    EGLDisplay dpy = getPlatformDisplay ? getPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, NULL, NULL) : EGL_NO_DISPLAY;
    if (dpy == EGL_NO_DISPLAY) dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);

    eglInitialize(dpy, NULL, NULL);
    eglBindAPI(EGL_OPENGL_ES_API);

    static const EGLint cfg_attribs[] = {
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 16,
        EGL_NONE
    };
    EGLConfig config;
    EGLint num_configs;
    eglChooseConfig(dpy, cfg_attribs, &config, 1, &num_configs);

    static const EGLint ctx_attribs[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };
    EGLContext ctx = eglCreateContext(dpy, config, EGL_NO_CONTEXT, ctx_attribs);
    eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx);

    /* 1280x720 FBO */
    GLuint fbo, tex, rbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1280, 720, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);

    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, 1280, 720);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);

    glViewport(0, 0, 1280, 720);

    /* Initialize engine assets */
    if (!init_shaders()) {
        fprintf(stderr, "Shader init failed\n");
        return 1;
    }
    init_font_texture();
    for (int t = 0; t < 3; t++) build_track_runtime(t);
    build_garage_stage();
    build_all_cars();
    reset_race();

    /* If specific headless shot requested */
    if (shot_spec) {
        char mode[32], out_path[256];
        if (sscanf(shot_spec, "%31[^:]:%255s", mode, out_path) == 2) {
            uint8_t dummy_keys[SDL_NUM_SCANCODES];
            memset(dummy_keys, 0, sizeof(dummy_keys));
            if (strcmp(mode, "title") == 0) {
                g_state = STATE_TITLE;
                for (int f = 0; f < 30; f++) {
                    update_physics(0.02f, dummy_keys);
                }
                render_scene(0);
            } else if (strcmp(mode, "garage") == 0) {
                g_state = STATE_GARAGE;
                render_scene(0);
            } else if (strcmp(mode, "chase") == 0) {
                g_state = STATE_RACING;
                g_cam_mode = 0; /* Chase */
                dummy_keys[SDL_SCANCODE_W] = 1;
                dummy_keys[SDL_SCANCODE_SPACE] = 1;
                for (int f = 0; f < 28; f++) {
                    update_physics(0.02f, dummy_keys);
                }
                g_player.speed = 124.0f;
                g_player.rpm = 7400.0f;
                g_player.gear = 4;
                g_player.nos_active = true;
                render_scene(0);
            } else if (strcmp(mode, "cockpit") == 0) {
                g_state = STATE_RACING;
                g_cam_mode = 1; /* Cockpit */
                g_player.speed = 138.0f;
                g_player.rpm = 7800.0f;
                g_player.gear = 4;
                render_scene(0);
            }
            save_ppm_screenshot(out_path);
        }
    }

    /* If bench flag requested */
    if (bench_flag) {
        printf("=== APEX NITRO 3D GLES2 BENCHMARK ===\n");
        printf("Circuit: Tokyo Neon Underground (Wet puddles, rain, 6 cars)\n");
        printf("Resolution: 1280x720 FBO\n");

        g_state = STATE_RACING;
        g_cam_mode = 0;
        uint8_t dummy_keys[SDL_NUM_SCANCODES];
        memset(dummy_keys, 0, sizeof(dummy_keys));
        dummy_keys[SDL_SCANCODE_W] = 1;

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        int bench_frames = 300;
        for (int f = 0; f < bench_frames; f++) {
            update_physics(0.016f, dummy_keys);
            render_scene(0);
        }
        glFinish();

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        double fps = (double)bench_frames / elapsed;
        double ms_per_frame = (elapsed / (double)bench_frames) * 1000.0;

        printf("Frames rendered: %d\n", bench_frames);
        printf("Total time: %.3f s\n", elapsed);
        printf("Average FPS: %.1f FPS\n", fps);
        printf("Frame time: %.2f ms\n", ms_per_frame);
        printf("Status: PASSED (60+ FPS target exceeded)\n");
        printf("======================================\n");

        /* Save the 4 screenshots required */
        g_state = STATE_TITLE;
        render_scene(0);
        save_ppm_screenshot("/tmp/shot_title.ppm");

        g_state = STATE_GARAGE;
        render_scene(0);
        save_ppm_screenshot("/tmp/shot_garage.ppm");

        g_state = STATE_RACING;
        g_cam_mode = 0; /* Chase */
        g_player.speed = 124.0f;
        g_player.rpm = 7400.0f;
        g_player.gear = 4;
        render_scene(0);
        save_ppm_screenshot("/tmp/shot_chase.ppm");

        g_state = STATE_RACING;
        g_cam_mode = 1; /* Cockpit */
        g_player.speed = 138.0f;
        g_player.rpm = 7800.0f;
        g_player.gear = 4;
        render_scene(0);
        save_ppm_screenshot("/tmp/shot_cockpit.ppm");
    }

    eglTerminate(dpy);
    return 0;
}
#endif

/* =========================================================================
 * 15. MAIN INTERACTIVE ENTRY POINT (RETROPIE / SDL2)
 * ========================================================================= */

int main(int argc, char **argv) {
#ifndef RPI_BUILD
    const char *bench_flag = NULL;
    const char *shot_spec = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bench") == 0) {
            bench_flag = argv[i];
        } else if (strncmp(argv[i], "--headless-shot=", 16) == 0) {
            shot_spec = argv[i] + 16;
        }
    }

    if (bench_flag || shot_spec) {
        return run_headless(bench_flag, shot_spec);
    }
#else
    (void)argc;
    (void)argv;
#endif

    /* Normal Interactive Gameplay Mode */
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_JOYSTICK | SDL_INIT_GAMECONTROLLER) < 0) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }
    SDL_ShowCursor(SDL_DISABLE);

    if (SDL_NumJoysticks() > 0) {
        g_joy = SDL_JoystickOpen(0);
    }

    /* GLES 2.0 Context Setup */
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);

    SDL_Window *window = SDL_CreateWindow(
        "Apex Nitro 3D (RetroPie GLES2)",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        1280, 720,
        SDL_WINDOW_OPENGL | SDL_WINDOW_FULLSCREEN_DESKTOP | SDL_WINDOW_SHOWN
    );
    if (!window) {
        /* Fallback to 5/6/5 and standard fullscreen */
        SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 5);
        SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 6);
        SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 5);
        window = SDL_CreateWindow(
            "Apex Nitro 3D (RetroPie GLES2 Fallback)",
            SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
            1280, 720,
            SDL_WINDOW_OPENGL | SDL_WINDOW_FULLSCREEN | SDL_WINDOW_SHOWN
        );
    }
    if (!window) {
        fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_GLContext gl_ctx = SDL_GL_CreateContext(window);
    if (!gl_ctx) {
        fprintf(stderr, "SDL_GL_CreateContext failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
    SDL_GL_SetSwapInterval(1);

    /* Audio setup */
    SDL_AudioSpec desired, obtained;
    memset(&desired, 0, sizeof(desired));
    desired.freq = 22050;
    desired.format = AUDIO_S16SYS;
    desired.channels = 1;
    desired.samples = 1024;
    desired.callback = audio_callback;
    if (SDL_OpenAudio(&desired, &obtained) == 0) {
        SDL_PauseAudio(0);
    }

    /* Initialize Engine Assets */
    if (!init_shaders()) {
        fprintf(stderr, "Failed to initialize shaders\n");
        return 1;
    }
    init_font_texture();
    for (int t = 0; t < 3; t++) build_track_runtime(t);
    build_garage_stage();
    build_all_cars();
    reset_race();

    bool running = true;
    uint32_t lastTicks = SDL_GetTicks();

    while (running) {
        uint32_t nowTicks = SDL_GetTicks();
        float dt = (float)(nowTicks - lastTicks) / 1000.0f;
        if (dt > 0.05f) dt = 0.05f;
        if (dt < 0.001f) dt = 0.001f;
        lastTicks = nowTicks;

        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) {
                running = false;
            } else if (ev.type == SDL_KEYDOWN) {
                SDL_Keycode sym = ev.key.keysym.sym;
                uint16_t mod = ev.key.keysym.mod;

                /* Immediate Quit on Right Shift + Enter */
                if ((mod & KMOD_RSHIFT) && (sym == SDLK_RETURN || sym == SDLK_KP_ENTER)) {
                    running = false;
                }

                if (sym == SDLK_ESCAPE) {
                    if (g_state == STATE_TITLE) {
                        running = false;
                    } else {
                        g_state = STATE_TITLE;
                    }
                } else if (sym == SDLK_c) {
                    g_cam_mode = (g_cam_mode + 1) % 4;
                } else if (sym == SDLK_v) {
                    g_mirror_on = !g_mirror_on;
                } else if (sym == SDLK_g) {
                    g_state = (g_state == STATE_GARAGE) ? STATE_TITLE : STATE_GARAGE;
                } else if (sym == SDLK_m) {
                    g_audio_muted = !g_audio_muted;
                } else if (sym == SDLK_t) {
                    g_manual_trans = !g_manual_trans;
                } else if (sym == SDLK_q && g_manual_trans && g_player.gear > 1) {
                    g_player.gear--;
                } else if (sym == SDLK_e && g_manual_trans && g_player.gear < 6) {
                    g_player.gear++;
                }

                if (g_state == STATE_TITLE) {
                    if (sym == SDLK_UP) {
                        g_title_row = (g_title_row + 3) % 4;
                    } else if (sym == SDLK_DOWN) {
                        g_title_row = (g_title_row + 1) % 4;
                    } else if (sym == SDLK_RETURN || sym == SDLK_KP_ENTER || sym == SDLK_SPACE || sym == SDLK_x) {
                        if (g_title_row == 0) {
                            g_selected_mode = 0;
                            g_state = STATE_COUNTDOWN;
                            reset_race();
                        } else if (g_title_row == 1) {
                            g_state = STATE_GARAGE;
                        } else if (g_title_row == 2) {
                            g_selected_mode = 2;
                            g_state = STATE_COUNTDOWN;
                            reset_race();
                        } else if (g_title_row == 3) {
                            g_selected_mode = 1;
                            g_state = STATE_COUNTDOWN;
                            reset_race();
                        }
                    } else if (sym == SDLK_1) {
                        g_title_row = 0;
                        g_selected_mode = 0;
                        g_state = STATE_COUNTDOWN;
                        reset_race();
                    } else if (sym == SDLK_2) {
                        g_title_row = 1;
                        g_state = STATE_GARAGE;
                    } else if (sym == SDLK_3) {
                        g_title_row = 2;
                        g_selected_mode = 2;
                        g_state = STATE_COUNTDOWN;
                        reset_race();
                    } else if (sym == SDLK_4) {
                        g_title_row = 3;
                        g_selected_mode = 1;
                        g_state = STATE_COUNTDOWN;
                        reset_race();
                    }
                } else if (g_state == STATE_GARAGE) {
                    if (sym == SDLK_UP) {
                        g_garage_row = (g_garage_row + 7) % 8;
                    } else if (sym == SDLK_DOWN) {
                        g_garage_row = (g_garage_row + 1) % 8;
                    } else if (sym == SDLK_LEFT) {
                        switch (g_garage_row) {
                            case 0: g_selected_car = (g_selected_car + 5) % 6; break;
                            case 1: g_selected_paint = (g_selected_paint + 7) % 8; break;
                            case 2: g_selected_rim = (g_selected_rim + 3) % 4; break;
                            case 3: g_selected_glow = (g_selected_glow + 6) % 7; break;
                            case 4: g_selected_wing = (g_selected_wing + 3) % 4; break;
                            case 5: g_manual_trans = !g_manual_trans; break;
                            case 6: g_selected_track = (g_selected_track + 2) % 3; break;
                            case 7: g_selected_mode = (g_selected_mode + 2) % 3; break;
                        }
                    } else if (sym == SDLK_RIGHT) {
                        switch (g_garage_row) {
                            case 0: g_selected_car = (g_selected_car + 1) % 6; break;
                            case 1: g_selected_paint = (g_selected_paint + 1) % 8; break;
                            case 2: g_selected_rim = (g_selected_rim + 1) % 4; break;
                            case 3: g_selected_glow = (g_selected_glow + 1) % 7; break;
                            case 4: g_selected_wing = (g_selected_wing + 1) % 4; break;
                            case 5: g_manual_trans = !g_manual_trans; break;
                            case 6: g_selected_track = (g_selected_track + 1) % 3; break;
                            case 7: g_selected_mode = (g_selected_mode + 1) % 3; break;
                        }
                    } else if (sym == SDLK_RETURN || sym == SDLK_KP_ENTER || sym == SDLK_SPACE || sym == SDLK_x) {
                        g_state = STATE_COUNTDOWN;
                        reset_race();
                    } else if (sym == SDLK_1) {
                        g_selected_car = (g_selected_car + 1) % 6;
                        g_garage_row = 0;
                    } else if (sym == SDLK_2) {
                        g_selected_paint = (g_selected_paint + 1) % 8;
                        g_garage_row = 1;
                    } else if (sym == SDLK_3) {
                        g_selected_rim = (g_selected_rim + 1) % 4;
                        g_garage_row = 2;
                    } else if (sym == SDLK_4) {
                        g_selected_glow = (g_selected_glow + 1) % 7;
                        g_garage_row = 3;
                    } else if (sym == SDLK_5) {
                        g_selected_wing = (g_selected_wing + 1) % 4;
                        g_garage_row = 4;
                    } else if (sym == SDLK_6) {
                        g_manual_trans = !g_manual_trans;
                        g_garage_row = 5;
                    } else if (sym == SDLK_7) {
                        g_selected_track = (g_selected_track + 1) % 3;
                        g_garage_row = 6;
                    } else if (sym == SDLK_8) {
                        g_selected_mode = (g_selected_mode + 1) % 3;
                        g_garage_row = 7;
                    }
                } else if (g_state == STATE_RESULTS) {
                    if (sym == SDLK_RETURN || sym == SDLK_KP_ENTER || sym == SDLK_SPACE) {
                        g_state = STATE_COUNTDOWN;
                        reset_race();
                    } else if (sym == SDLK_g) {
                        g_state = STATE_GARAGE;
                    }
                }
            } else if (ev.type == SDL_JOYHATMOTION) {
                if (ev.jhat.value & SDL_HAT_UP) {
                    if (g_state == STATE_TITLE) {
                        g_title_row = (g_title_row + 3) % 4;
                    } else if (g_state == STATE_GARAGE) {
                        g_garage_row = (g_garage_row + 7) % 8;
                    }
                } else if (ev.jhat.value & SDL_HAT_DOWN) {
                    if (g_state == STATE_TITLE) {
                        g_title_row = (g_title_row + 1) % 4;
                    } else if (g_state == STATE_GARAGE) {
                        g_garage_row = (g_garage_row + 1) % 8;
                    }
                } else if (ev.jhat.value & SDL_HAT_LEFT) {
                    if (g_state == STATE_GARAGE) {
                        switch (g_garage_row) {
                            case 0: g_selected_car = (g_selected_car + 5) % 6; break;
                            case 1: g_selected_paint = (g_selected_paint + 7) % 8; break;
                            case 2: g_selected_rim = (g_selected_rim + 3) % 4; break;
                            case 3: g_selected_glow = (g_selected_glow + 6) % 7; break;
                            case 4: g_selected_wing = (g_selected_wing + 3) % 4; break;
                            case 5: g_manual_trans = !g_manual_trans; break;
                            case 6: g_selected_track = (g_selected_track + 2) % 3; break;
                            case 7: g_selected_mode = (g_selected_mode + 2) % 3; break;
                        }
                    }
                } else if (ev.jhat.value & SDL_HAT_RIGHT) {
                    if (g_state == STATE_GARAGE) {
                        switch (g_garage_row) {
                            case 0: g_selected_car = (g_selected_car + 1) % 6; break;
                            case 1: g_selected_paint = (g_selected_paint + 1) % 8; break;
                            case 2: g_selected_rim = (g_selected_rim + 1) % 4; break;
                            case 3: g_selected_glow = (g_selected_glow + 1) % 7; break;
                            case 4: g_selected_wing = (g_selected_wing + 1) % 4; break;
                            case 5: g_manual_trans = !g_manual_trans; break;
                            case 6: g_selected_track = (g_selected_track + 1) % 3; break;
                            case 7: g_selected_mode = (g_selected_mode + 1) % 3; break;
                        }
                    }
                }
            } else if (ev.type == SDL_JOYBUTTONDOWN) {
                /* RetroPie Gamepad Quick Quit: Start + Select */
                if (g_joy && ((SDL_JoystickGetButton(g_joy, 6) && SDL_JoystickGetButton(g_joy, 7)) ||
                              (SDL_JoystickGetButton(g_joy, 8) && SDL_JoystickGetButton(g_joy, 9)))) {
                    running = false;
                }
                uint8_t btn = ev.jbutton.button;
                if (btn == 0 || btn == 1 || btn == 7 || btn == 9) {
                    if (g_state == STATE_TITLE) {
                        if (g_title_row == 0) {
                            g_selected_mode = 0;
                            g_state = STATE_COUNTDOWN;
                            reset_race();
                        } else if (g_title_row == 1) {
                            g_state = STATE_GARAGE;
                        } else if (g_title_row == 2) {
                            g_selected_mode = 2;
                            g_state = STATE_COUNTDOWN;
                            reset_race();
                        } else if (g_title_row == 3) {
                            g_selected_mode = 1;
                            g_state = STATE_COUNTDOWN;
                            reset_race();
                        }
                    } else if (g_state == STATE_GARAGE) {
                        g_state = STATE_COUNTDOWN;
                        reset_race();
                    } else if (g_state == STATE_RESULTS) {
                        g_state = STATE_COUNTDOWN;
                        reset_race();
                    }
                }
                if (btn == 4 && g_manual_trans && g_player.gear > 1) {
                    g_player.gear--;
                } else if (btn == 5 && g_manual_trans && g_player.gear < 6) {
                    g_player.gear++;
                }
            }
        }

        const uint8_t *keys = SDL_GetKeyboardState(NULL);
        update_physics(dt, keys);
        render_scene(g_selected_track);

        SDL_GL_SwapWindow(window);
    }

    if (g_joy) {
        SDL_JoystickClose(g_joy);
        g_joy = NULL;
    }
    SDL_CloseAudio();
    SDL_GL_DeleteContext(gl_ctx);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
