// Forward vertex shader: per-tet face rasterization setup.
//
// Each tet is drawn as 4 triangles (12 vertices) via instanced draw.
// The vertex shader computes ray-plane intersection parameters for the fragment shader.

struct Uniforms {
    vp_col0: vec4<f32>,
    vp_col1: vec4<f32>,
    vp_col2: vec4<f32>,
    vp_col3: vec4<f32>,
    c2w_col0: vec4<f32>,
    c2w_col1: vec4<f32>,
    c2w_col2: vec4<f32>,
    intrinsics: vec4<f32>,
    cam_pos_pad: vec4<f32>,
    screen_width: f32,
    screen_height: f32,
    tet_count: u32,
    step: u32,
    tile_size_u: u32,
    ray_mode: u32,
    min_t: f32,
    _pad1c: u32,
    _pad2: vec4<u32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) tet_density: f32,
    @location(1) @interpolate(flat) base_color: vec3<f32>,
    @location(2) plane_numerators: vec4<f32>,
    @location(3) plane_denominators: vec4<f32>,
    @location(4) ray_dir: vec3<f32>,
    @location(5) dc_dt: f32,
    @location(6) @interpolate(flat) face_n0: vec3<f32>,
    @location(7) @interpolate(flat) face_n1: vec3<f32>,
    @location(8) @interpolate(flat) face_n2: vec3<f32>,
    @location(9) @interpolate(flat) face_n3: vec3<f32>,
};

@group(0) @binding(0) var<storage, read> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> vertices: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> colors: array<f32>;
@group(0) @binding(4) var<storage, read> densities: array<f32>;
@group(0) @binding(5) var<storage, read> color_grads: array<f32>;
@group(0) @binding(6) var<storage, read> sorted_indices: array<u32>;

// Face (a, b, c, opposite_vertex) -- opposite used to flip normal inward
const TET_FACES: array<vec4<u32>, 4> = array<vec4<u32>, 4>(
    vec4<u32>(0u, 2u, 1u, 3u),
    vec4<u32>(1u, 2u, 3u, 0u),
    vec4<u32>(0u, 3u, 2u, 1u),
    vec4<u32>(3u, 0u, 1u, 2u),
);

// 12 entries: face index for each of the 12 vertices (4 faces x 3 verts)
const FACE_VERTEX_MAP: array<vec2<u32>, 12> = array<vec2<u32>, 12>(
    vec2<u32>(0u, 0u), vec2<u32>(0u, 1u), vec2<u32>(0u, 2u), // face 0
    vec2<u32>(1u, 0u), vec2<u32>(1u, 1u), vec2<u32>(1u, 2u), // face 1
    vec2<u32>(2u, 0u), vec2<u32>(2u, 1u), vec2<u32>(2u, 2u), // face 2
    vec2<u32>(3u, 0u), vec2<u32>(3u, 1u), vec2<u32>(3u, 2u), // face 3
);

fn load_f32x3(buf_base: u32) -> vec3<f32> {
    return vec3<f32>(vertices[buf_base], vertices[buf_base + 1u], vertices[buf_base + 2u]);
}

fn load_color(idx: u32) -> vec3<f32> {
    return vec3<f32>(colors[idx * 3u], colors[idx * 3u + 1u], colors[idx * 3u + 2u]);
}

fn load_grad(idx: u32) -> vec3<f32> {
    return vec3<f32>(color_grads[idx * 3u], color_grads[idx * 3u + 1u], color_grads[idx * 3u + 2u]);
}

@vertex
fn main(@builtin(instance_index) instance_idx: u32, @builtin(vertex_index) vert_idx: u32) -> VertexOutput {
    var out: VertexOutput;

    let tet_id = sorted_indices[instance_idx];

    // Load 4 vertex indices
    let i0 = indices[tet_id * 4u];
    let i1 = indices[tet_id * 4u + 1u];
    let i2 = indices[tet_id * 4u + 2u];
    let i3 = indices[tet_id * 4u + 3u];

    // Load vertex positions
    var verts: array<vec3<f32>, 4>;
    verts[0] = load_f32x3(i0 * 3u);
    verts[1] = load_f32x3(i1 * 3u);
    verts[2] = load_f32x3(i2 * 3u);
    verts[3] = load_f32x3(i3 * 3u);

    // Which face and which vertex within that face?
    let fv = FACE_VERTEX_MAP[vert_idx];
    let face = TET_FACES[fv.x];
    let face_vert_idx = face[fv.y];
    let world_pos = verts[face_vert_idx];

    let cam = uniforms.cam_pos_pad.xyz;
    // IMPORTANT: Do NOT normalize here. Perspective interpolation of unnormalized
    // (world_pos - cam) gives exact (p - cam) at each fragment, so the fragment
    // shader's division by d = length(ray_dir) yields the true normalized direction.
    // Normalizing here would introduce interpolation error (weighted sum of unit
    // vectors != unit vector in the correct direction).
    let ray_dir = world_pos - cam;

    // Density
    out.tet_density = densities[tet_id];

    // Color gradient -> dc/dt along ray
    let grad = load_grad(tet_id);
    out.dc_dt = dot(grad, ray_dir);

    // Base color at ray origin
    let color = load_color(tet_id);
    let offset = dot(grad, cam - verts[0]);
    out.base_color = color + vec3<f32>(offset);

    // Ray-plane intersection parameters for all 4 faces
    // Also store inward-pointing face normals for fragment shader
    var numerators = vec4<f32>(0.0);
    var denominators = vec4<f32>(0.0);
    var face_normals: array<vec3<f32>, 4>;

    for (var i = 0u; i < 4u; i++) {
        let f = TET_FACES[i];
        let va = verts[f[0]];
        let vb = verts[f[1]];
        let vc = verts[f[2]];
        var n = cross(vc - va, vb - va);
        // Flip normal to point inward (toward opposite vertex)
        let v_opp = verts[f[3]];
        if (dot(n, v_opp - va) < 0.0) {
            n = -n;
        }
        face_normals[i] = n;

        numerators[i] = dot(n, va - cam);
        denominators[i] = dot(n, ray_dir);
    }

    out.plane_numerators = numerators;
    out.plane_denominators = denominators;
    out.ray_dir = ray_dir;
    out.face_n0 = face_normals[0];
    out.face_n1 = face_normals[1];
    out.face_n2 = face_normals[2];
    out.face_n3 = face_normals[3];

    // Project to clip space
    let vp = mat4x4<f32>(uniforms.vp_col0, uniforms.vp_col1, uniforms.vp_col2, uniforms.vp_col3);
    out.position = vp * vec4<f32>(world_pos, 1.0);

    // Cull back-facing triangles: degenerate to zero-area triangle.
    // numerator > 0 means camera is on the outward side (front-facing).
    if numerators[fv.x] <= 0.0 {
        out.position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    return out;
}
