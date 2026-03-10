// Forward compute shader: per-tet SH evaluation, frustum culling, depth key generation.
//
// Ported from forward_compute.rs (rust-gpu).
// Key differences from webrm: direct SH coefficients (no PCA), safe math.

struct Uniforms {
    vp_col0: vec4<f32>,
    vp_col1: vec4<f32>,
    vp_col2: vec4<f32>,
    vp_col3: vec4<f32>,
    inv_vp_col0: vec4<f32>,
    inv_vp_col1: vec4<f32>,
    inv_vp_col2: vec4<f32>,
    inv_vp_col3: vec4<f32>,
    cam_pos_pad: vec4<f32>,
    screen_width: f32,
    screen_height: f32,
    tet_count: u32,
    sh_degree: u32,
    step: u32,
    _pad1: vec3<u32>,
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
@group(0) @binding(3) var<storage, read> sh_coeffs: array<f32>;
@group(0) @binding(4) var<storage, read> densities: array<f32>;
@group(0) @binding(5) var<storage, read> color_grads: array<f32>;
@group(0) @binding(6) var<storage, read> circumdata: array<f32>;
@group(0) @binding(7) var<storage, read_write> colors: array<f32>;
@group(0) @binding(8) var<storage, read_write> sort_keys: array<u32>;
@group(0) @binding(9) var<storage, read_write> sort_values: array<u32>;
@group(0) @binding(10) var<storage, read_write> indirect_args: DrawIndirectArgs;
// Compact outputs for tiled path (indexed by vis_idx from atomicAdd)
@group(0) @binding(11) var<storage, read_write> tiles_touched: array<u32>;
@group(0) @binding(12) var<storage, read_write> compact_tet_ids: array<u32>;

// --- SH Constants ---
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

fn project_to_ndc(pos: vec3<f32>, vp: mat4x4<f32>) -> vec4<f32> {
    let clip = vp * vec4<f32>(pos, 1.0);
    let inv_w = 1.0 / (clip.w + 1e-6);
    return vec4<f32>(clip.xyz * inv_w, clip.w);
}

fn softplus(x: f32) -> f32 {
    if (x > 8.0) {
        return x;
    }
    return 0.1 * log(1.0 + exp(10.0 * x));
}

fn load_vertex(idx: u32) -> vec3<f32> {
    let i = idx * 3u;
    return vec3<f32>(vertices[i], vertices[i + 1u], vertices[i + 2u]);
}

fn eval_sh(dir: vec3<f32>, sh_degree: u32, base: u32) -> f32 {
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;

    var val = C0 * sh_coeffs[base];

    if (sh_degree >= 1u) {
        val += -C1 * y * sh_coeffs[base + 1u];
        val += C1 * z * sh_coeffs[base + 2u];
        val += -C1 * x * sh_coeffs[base + 3u];
    }

    if (sh_degree >= 2u) {
        let xx = x * x;
        let yy = y * y;
        let zz = z * z;

        val += C2_0 * x * y * sh_coeffs[base + 4u];
        val += C2_1 * y * z * sh_coeffs[base + 5u];
        val += C2_2 * (2.0 * zz - xx - yy) * sh_coeffs[base + 6u];
        val += C2_3 * x * z * sh_coeffs[base + 7u];
        val += C2_4 * (xx - yy) * sh_coeffs[base + 8u];

        if (sh_degree >= 3u) {
            val += C3_0 * y * (3.0 * xx - yy) * sh_coeffs[base + 9u];
            val += C3_1 * x * y * z * sh_coeffs[base + 10u];
            val += C3_2 * y * (4.0 * zz - xx - yy) * sh_coeffs[base + 11u];
            val += C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coeffs[base + 12u];
            val += C3_4 * x * (4.0 * zz - xx - yy) * sh_coeffs[base + 13u];
            val += C3_5 * z * (xx - yy) * sh_coeffs[base + 14u];
            val += C3_6 * x * (xx - 3.0 * yy) * sh_coeffs[base + 15u];
        }
    }

    return val;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let tet_id = global_id.x + global_id.y * num_workgroups.x * 64u;
    // Padding threads (beyond tet_count): initialize sort buffers for bitonic sort.
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

    // If any vertex is behind the camera (clip.w <= 0), NDC coords are unreliable.
    // Mark as visible and let the hardware rasterizer handle near-plane clipping.
    let any_behind = (p0.w <= 0.0) || (p1.w <= 0.0) || (p2.w <= 0.0) || (p3.w <= 0.0);

    var visible = true;
    if (!any_behind) {
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
    let depth = dot(diff, diff) - r2;
    let depth_bits = bitcast<u32>(depth);
    sort_keys[tet_id] = ~depth_bits; // Invert for back-to-front sort
    sort_values[tet_id] = tet_id;

    // Atomic increment instance count, get compact vis_idx
    let vis_idx = atomicAdd(&indirect_args.instance_count, 1u);

    // Write compact tet id list for tiled path
    compact_tet_ids[vis_idx] = tet_id;

    // Compute tiles_touched for prefix scan (conservative scanline)
    let W = uniforms.screen_width;
    let H = uniforms.screen_height;
    let tile_size = 16.0; // must match tile_size used in tile_gen

    if (any_behind) {
        // Cull entirely — behind-camera tets would overflow pair buffers
        tiles_touched[vis_idx] = 0u;
    } else {
        // NDC to pixel coords for all 4 vertices
        var proj: array<vec2<f32>, 4>;
        proj[0] = vec2<f32>((p0.x + 1.0) * 0.5 * W, (1.0 - p0.y) * 0.5 * H);
        proj[1] = vec2<f32>((p1.x + 1.0) * 0.5 * W, (1.0 - p1.y) * 0.5 * H);
        proj[2] = vec2<f32>((p2.x + 1.0) * 0.5 * W, (1.0 - p2.y) * 0.5 * H);
        proj[3] = vec2<f32>((p3.x + 1.0) * 0.5 * W, (1.0 - p3.y) * 0.5 * H);

        // AABB for tile bounds
        let pix_min_x = max(min(min(proj[0].x, proj[1].x), min(proj[2].x, proj[3].x)), 0.0);
        let pix_max_x = min(max(max(proj[0].x, proj[1].x), max(proj[2].x, proj[3].x)), W - 1.0);
        let pix_min_y = max(min(min(proj[0].y, proj[1].y), min(proj[2].y, proj[3].y)), 0.0);
        let pix_max_y = min(max(max(proj[0].y, proj[1].y), max(proj[2].y, proj[3].y)), H - 1.0);

        if (pix_min_x > pix_max_x || pix_min_y > pix_max_y) {
            tiles_touched[vis_idx] = 0u;
        } else {
            let T = tile_size;
            let tiles_x_total = u32(ceil(W / T));
            let tile_min_x = u32(pix_min_x / T);
            let tile_max_x = min(u32(pix_max_x / T), tiles_x_total - 1u);
            let tile_min_y = u32(pix_min_y / T);
            let tile_max_y = min(u32(pix_max_y / T), u32(ceil(H / T)) - 1u);

            // Conservative scanline tile counting (Variant B)
            var tile_count = 0u;
            for (var ty = tile_min_y; ty <= tile_max_y; ty++) {
                var xl = 1e10f;
                var xr = -1e10f;

                // Edge intersections at both tile-row boundaries
                let ei_arr = array<u32, 6>(0u, 0u, 0u, 1u, 1u, 2u);
                let ej_arr = array<u32, 6>(1u, 2u, 3u, 2u, 3u, 3u);
                for (var e = 0u; e < 6u; e++) {
                    let ei = ei_arr[e];
                    let ej = ej_arr[e];
                    let yi = proj[ei].y;
                    let yj = proj[ej].y;
                    let xi = proj[ei].x;
                    let xj = proj[ej].x;
                    for (var b = 0u; b < 2u; b++) {
                        let ytest = f32(ty + b) * T;
                        if ((yi <= ytest && yj > ytest) || (yj <= ytest && yi > ytest)) {
                            let t = (ytest - yi) / (yj - yi);
                            let x = xi + t * (xj - xi);
                            xl = min(xl, x);
                            xr = max(xr, x);
                        }
                    }
                }

                // Vertices within tile-row band
                for (var v = 0u; v < 4u; v++) {
                    let vy = proj[v].y;
                    if (vy >= f32(ty) * T && vy <= f32(ty + 1u) * T) {
                        xl = min(xl, proj[v].x);
                        xr = max(xr, proj[v].x);
                    }
                }

                if (xl <= xr) {
                    let tx0 = clamp(u32(max(floor(xl / T), 0.0)), tile_min_x, tile_max_x);
                    let tx1 = clamp(u32(max(floor(xr / T), 0.0)), tile_min_x, tile_max_x);
                    tile_count += tx1 - tx0 + 1u;
                }
            }
            tiles_touched[vis_idx] = tile_count;
        }
    }

    // --- 4. SH evaluation ---
    let centroid = (v0 + v1 + v2 + v3) * 0.25;
    let dir = normalize(centroid - cam);

    let num_coeffs = (uniforms.sh_degree + 1u) * (uniforms.sh_degree + 1u);
    let sh_stride = num_coeffs * 3u;
    let sh_base = tet_id * sh_stride;
    let nc = num_coeffs;

    var result_color = vec3<f32>(0.0);
    result_color.x = eval_sh(dir, uniforms.sh_degree, sh_base);
    result_color.y = eval_sh(dir, uniforms.sh_degree, sh_base + nc);
    result_color.z = eval_sh(dir, uniforms.sh_degree, sh_base + 2u * nc);

    // Add 0.5 bias (SH2RGB)
    result_color += vec3<f32>(0.5);

    // --- 5. Apply color gradient (linear field) ---
    let gx = color_grads[tet_id * 3u];
    let gy = color_grads[tet_id * 3u + 1u];
    let gz = color_grads[tet_id * 3u + 2u];
    let grad = vec3<f32>(gx, gy, gz);
    let offset = dot(grad, v0 - centroid);
    let input_val = result_color + vec3<f32>(offset);

    // Softplus activation
    let sp = vec3<f32>(softplus(input_val.x), softplus(input_val.y), softplus(input_val.z));

    colors[tet_id * 3u] = sp.x;
    colors[tet_id * 3u + 1u] = sp.y;
    colors[tet_id * 3u + 2u] = sp.z;
}
