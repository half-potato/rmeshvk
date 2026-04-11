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
    sh_degree: u32,
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
@group(0) @binding(9) var<storage, read> color_grads: array<f32>;
@group(0) @binding(10) var<storage, read> sh_coeffs: array<u32>;

// --- SH constants ---
const C0: f32 = 0.28209479177387814;
const C1: f32 = 0.4886025119029199;
const C2_0: f32 = 1.0925484305920792;
const C2_1: f32 = -1.0925484305920792;
const C2_2: f32 = 0.31539156525252005;
const C2_3: f32 = -1.0925484305920792;
const C2_4: f32 = 0.5462742152960396;
const C3_0: f32 = -0.5900435899266435;
const C3_1: f32 = 2.890611442640554;
const C3_2: f32 = -0.4570457994644658;
const C3_3: f32 = 0.3731763325901154;
const C3_4: f32 = -0.4570457994644658;
const C3_5: f32 = 1.445305721320277;
const C3_6: f32 = -0.5900435899266435;

fn softplus(x: f32) -> f32 {
    if (x * 10.0 > 20.0) { return x; }
    return 0.1 * log(1.0 + exp(10.0 * x));
}

fn read_sh_f16(base: u32, i: u32) -> f32 {
    let word = sh_coeffs[base + i / 2u];
    let unpacked = unpack2x16float(word);
    return select(unpacked.x, unpacked.y, (i & 1u) != 0u);
}

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
    // Use algebraic identity: power(P) = (P-v0)·(P+v0-2C) to avoid catastrophic
    // cancellation when circumcenter is far from camera (sliver tets).
    let cx = circumdata[tet_id * 4u];
    let cy = circumdata[tet_id * 4u + 1u];
    let cz = circumdata[tet_id * 4u + 2u];

    let cam = uniforms.cam_pos_pad.xyz;
    let center = vec3<f32>(cx, cy, cz);
    let depth_raw = dot(cam - v0, cam + v0 - 2.0 * center);
    let depth = clamp(depth_raw, -1e20, 1e20);
    let depth_bits = bitcast<u32>(depth);
    sort_keys[tet_id] = ~depth_bits;
    sort_values[tet_id] = tet_id;

    atomicAdd(&indirect_args.instance_count, 1u);

    // --- 4. Color evaluation (only for visible tets) ---
    // sh_degree >= 0 all need SH eval (degree 0 = DC only). Else branch is training-only.
    if (arrayLength(&sh_coeffs) > 1u) {
        let centroid = (v0 + v1 + v2 + v3) * 0.25;
        let cam = uniforms.cam_pos_pad.xyz;
        let raw_dir = centroid - cam;
        let len = length(raw_dir);
        var dir = vec3<f32>(0.0);
        if (len > 0.0) { dir = raw_dir / len; }
        let x = dir.x; let y = dir.y; let z = dir.z;

        let nc = (uniforms.sh_degree + 1u) * (uniforms.sh_degree + 1u);
        let stride = nc * 3u;
        let sh_base = tet_id * ((stride + 1u) / 2u);

        var rgb = vec3<f32>(0.0);
        for (var c = 0u; c < 3u; c++) {
            let ch_offset = c * nc;
            var val = C0 * read_sh_f16(sh_base, ch_offset);

            if (uniforms.sh_degree >= 1u) {
                val -= C1 * y * read_sh_f16(sh_base, ch_offset + 1u);
                val += C1 * z * read_sh_f16(sh_base, ch_offset + 2u);
                val -= C1 * x * read_sh_f16(sh_base, ch_offset + 3u);
            }
            if (uniforms.sh_degree >= 2u) {
                let xx = x * x; let yy = y * y; let zz = z * z;
                let xy = x * y; let yz = y * z; let xz = x * z;
                val += C2_0 * xy * read_sh_f16(sh_base, ch_offset + 4u);
                val += C2_1 * yz * read_sh_f16(sh_base, ch_offset + 5u);
                val += C2_2 * (2.0 * zz - xx - yy) * read_sh_f16(sh_base, ch_offset + 6u);
                val += C2_3 * xz * read_sh_f16(sh_base, ch_offset + 7u);
                val += C2_4 * (xx - yy) * read_sh_f16(sh_base, ch_offset + 8u);
            }
            if (uniforms.sh_degree >= 3u) {
                let xx = x * x; let yy = y * y; let zz = z * z;
                val += C3_0 * y * (3.0 * xx - yy) * read_sh_f16(sh_base, ch_offset + 9u);
                val += C3_1 * x * y * z * read_sh_f16(sh_base, ch_offset + 10u);
                val += C3_2 * y * (4.0 * zz - xx - yy) * read_sh_f16(sh_base, ch_offset + 11u);
                val += C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * read_sh_f16(sh_base, ch_offset + 12u);
                val += C3_4 * x * (4.0 * zz - xx - yy) * read_sh_f16(sh_base, ch_offset + 13u);
                val += C3_5 * z * (xx - yy) * read_sh_f16(sh_base, ch_offset + 14u);
                val += C3_6 * x * (xx - 3.0 * yy) * read_sh_f16(sh_base, ch_offset + 15u);
            }
            rgb[c] = val + 0.5;
        }

        let grad = vec3<f32>(color_grads[tet_id * 3u], color_grads[tet_id * 3u + 1u], color_grads[tet_id * 3u + 2u]);
        let offset = dot(grad, v0 - centroid);
        rgb += vec3<f32>(offset);

        colors[tet_id * 3u]     = softplus(rgb.x);
        colors[tet_id * 3u + 1u] = softplus(rgb.y);
        colors[tet_id * 3u + 2u] = softplus(rgb.z);
    } else {
        colors[tet_id * 3u] = base_colors[tet_id * 3u];
        colors[tet_id * 3u + 1u] = base_colors[tet_id * 3u + 1u];
        colors[tet_id * 3u + 2u] = base_colors[tet_id * 3u + 2u];
    }
}
