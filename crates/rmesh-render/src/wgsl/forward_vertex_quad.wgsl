// Lightweight quad vertex shader: reads precomputed world positions + per-tet data,
// projects to clip space, and computes interpolated ray_dir / plane_denominators / dc_dt.
//
// Output matches forward_fragment.wgsl exactly — same fragment shader for both paths.

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
@group(0) @binding(1) var<storage, read> precomputed: array<vec4<f32>>;

@vertex
fn main(@builtin(instance_index) instance_idx: u32, @builtin(vertex_index) vert_idx: u32) -> VertexOutput {
    var out: VertexOutput;

    let base = instance_idx * 10u;

    // World-space position (hull-ordered, near-plane-clamped by prepass)
    let world_pos = precomputed[base + vert_idx].xyz;

    // Project to clip space
    let vp = mat4x4<f32>(uniforms.vp_col0, uniforms.vp_col1, uniforms.vp_col2, uniforms.vp_col3);
    out.position = vp * vec4<f32>(world_pos, 1.0);

    let cam = uniforms.cam_pos_pad.xyz;
    let ray_dir = world_pos - cam;
    out.ray_dir = ray_dir;

    // Face normals + numerators (flat, same for all 4 vertices)
    let fn0 = precomputed[base + 4u];
    let fn1 = precomputed[base + 5u];
    let fn2 = precomputed[base + 6u];
    let fn3 = precomputed[base + 7u];

    out.face_n0 = fn0.xyz;
    out.face_n1 = fn1.xyz;
    out.face_n2 = fn2.xyz;
    out.face_n3 = fn3.xyz;

    // Plane numerators (flat — same for all verts of this tet)
    out.plane_numerators = vec4<f32>(fn0.w, fn1.w, fn2.w, fn3.w);

    // Plane denominators = dot(normal, ray_dir) — interpolated per-vertex
    out.plane_denominators = vec4<f32>(
        dot(fn0.xyz, ray_dir),
        dot(fn1.xyz, ray_dir),
        dot(fn2.xyz, ray_dir),
        dot(fn3.xyz, ray_dir),
    );

    // Material (flat)
    let mat = precomputed[base + 8u];
    out.base_color = mat.xyz;
    out.tet_density = mat.w;

    // dc/dt = dot(grad, ray_dir) — interpolated
    let grad = precomputed[base + 9u].xyz;
    out.dc_dt = dot(grad, ray_dir);

    return out;
}
