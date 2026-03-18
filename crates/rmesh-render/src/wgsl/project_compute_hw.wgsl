// Lean projection compute shader for HW rasterization path only.
// Frustum culling + depth key generation. No tile counting or color copy.

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

struct DrawIndirectArgs {
    vertex_count: u32,
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0) var<storage, read> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> vertices: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> circumdata: array<f32>;
@group(0) @binding(4) var<storage, read_write> sort_keys: array<u32>;
@group(0) @binding(5) var<storage, read_write> sort_values: array<u32>;
@group(0) @binding(6) var<storage, read_write> indirect_args: DrawIndirectArgs;
@group(0) @binding(7) var<storage, read_write> colors: array<f32>;
@group(0) @binding(8) var<storage, read> base_colors: array<f32>;

fn project_to_ndc(pos: vec3<f32>, vp: mat4x4<f32>) -> vec4<f32> {
    let clip = vp * vec4<f32>(pos, 1.0);
    let inv_w = 1.0 / (clip.w + 1e-6);
    return vec4<f32>(clip.xyz * inv_w, clip.w);
}

fn load_vertex(idx: u32) -> vec3<f32> {
    let i = idx * 3u;
    return vec3<f32>(vertices[i], vertices[i + 1u], vertices[i + 2u]);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let tet_id = global_id.x + global_id.y * num_workgroups.x * 64u;

    if (tet_id >= uniforms.tet_count) {
        if (tet_id < arrayLength(&sort_keys)) {
            sort_keys[tet_id] = 0xFFFFFFFFu;
            sort_values[tet_id] = tet_id;
        }
        return;
    }

    // Copy SH-evaluated base colors to colors buffer (vertex shader reads colors)
    colors[tet_id * 3u] = base_colors[tet_id * 3u];
    colors[tet_id * 3u + 1u] = base_colors[tet_id * 3u + 1u];
    colors[tet_id * 3u + 2u] = base_colors[tet_id * 3u + 2u];

    // --- 1. Load geometry ---
    let i0 = indices[tet_id * 4u];
    let i1 = indices[tet_id * 4u + 1u];
    let i2 = indices[tet_id * 4u + 2u];
    let i3 = indices[tet_id * 4u + 3u];

    let v0 = load_vertex(i0);
    let v1 = load_vertex(i1);
    let v2 = load_vertex(i2);
    let v3 = load_vertex(i3);

    // --- 2. Frustum culling ---
    let vp = mat4x4<f32>(uniforms.vp_col0, uniforms.vp_col1, uniforms.vp_col2, uniforms.vp_col3);
    let p0 = project_to_ndc(v0, vp);
    let p1 = project_to_ndc(v1, vp);
    let p2 = project_to_ndc(v2, vp);
    let p3 = project_to_ndc(v3, vp);

    let all_behind = (p0.w <= 0.0) && (p1.w <= 0.0) && (p2.w <= 0.0) && (p3.w <= 0.0);
    let any_behind = (p0.w <= 0.0) || (p1.w <= 0.0) || (p2.w <= 0.0) || (p3.w <= 0.0);

    var visible = true;
    if (all_behind) {
        visible = false;
    } else if (!any_behind) {
        let min_x = min(min(p0.x, p1.x), min(p2.x, p3.x));
        let max_x = max(max(p0.x, p1.x), max(p2.x, p3.x));
        let min_y = min(min(p0.y, p1.y), min(p2.y, p3.y));
        let max_y = max(max(p0.y, p1.y), max(p2.y, p3.y));
        let min_z = min(min(p0.z, p1.z), min(p2.z, p3.z));

        if (max_x < -1.0 || min_x > 1.0 || max_y < -1.0 || min_y > 1.0 || min_z > 1.0) {
            visible = false;
        }
        let ext_x = (max_x - min_x) * uniforms.screen_width;
        let ext_y = (max_y - min_y) * uniforms.screen_height;
        if (ext_x * ext_y < 1.0) {
            visible = false;
        }
    }

    if (!visible) {
        sort_keys[tet_id] = 0xFFFFFFFFu;
        sort_values[tet_id] = tet_id;
        return;
    }

    // --- 3. Depth key from circumsphere ---
    let cx = circumdata[tet_id * 4u];
    let cy = circumdata[tet_id * 4u + 1u];
    let cz = circumdata[tet_id * 4u + 2u];
    let r2 = circumdata[tet_id * 4u + 3u];

    let cam = uniforms.cam_pos_pad.xyz;
    let diff = vec3<f32>(cx, cy, cz) - cam;
    let depth_raw = dot(diff, diff) - r2;
    let depth = clamp(depth_raw, -1e20, 1e20);
    let depth_bits = bitcast<u32>(depth);
    sort_keys[tet_id] = ~depth_bits;
    sort_values[tet_id] = tet_id;

    atomicAdd(&indirect_args.instance_count, 1u);
}
