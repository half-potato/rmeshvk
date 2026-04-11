// Vertex shader for compute-based interval shading.
//
// Uses draw_indexed_indirect with a static fan index buffer.
// vertex_index = tet*5 + local_vert (from index buffer).
// Reads 2 vec4s per vertex from storage + 1 vec4 per tet flat data.
// Output struct matches interval_fragment.wgsl's FragmentInput exactly.

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
    sh_degree: u32,
    near_plane: f32,
    far_plane: f32,
    _pad1: vec2<u32>,
};

@group(0) @binding(0) var<storage, read> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> verts: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> tet_data: array<vec4<f32>>;

struct IntervalVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) depths: vec2<f32>,
    @location(1) color_offsets: vec2<f32>,
    @location(2) @interpolate(flat) density: f32,
    @location(3) @interpolate(flat) base_color: vec3<f32>,
    @location(4) @interpolate(flat) tet_id: u32,
};

@vertex
fn main(@builtin(vertex_index) vid: u32) -> IntervalVertexOutput {
    // vid = tet*5 + local_vert, from the static index buffer.
    // Derive tet via integer division by 5 (compiles to multiply-shift).
    let tet = vid / 5u;
    let slot = vid;

    // Read 2 vec4s per vertex
    let pos_depth = verts[slot * 2u + 0u];   // (ndc_x, ndc_y, z_front, z_back)
    let offsets = verts[slot * 2u + 1u];      // (off_front, off_back, 0, 0)

    // Read per-tet flat data (2 vec4s per tet: [color+density, tet_id+pad])
    let td = tet_data[tet * 2u];              // (base_r, base_g, base_b, density)
    let td1 = tet_data[tet * 2u + 1u];        // (bitcast tet_id, 0, 0, 0)

    var out: IntervalVertexOutput;
    out.position = vec4<f32>(pos_depth.xy, 0.0, 1.0);
    out.depths = pos_depth.zw;
    out.color_offsets = offsets.xy;
    out.density = td.w;
    out.base_color = td.xyz;
    out.tet_id = bitcast<u32>(td1.x);
    return out;
}
