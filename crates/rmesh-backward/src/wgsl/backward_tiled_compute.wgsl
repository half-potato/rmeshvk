// Compute-based backward tiled renderer: 1 warp (32 threads) per 16×16 tile.
//
// Same scanline-fill thread model as rasterize_compute.wgsl.
// Back-to-front state undoing in shared memory, gradient reduction via warp shuffle.
//
// One workgroup per tile. Dispatch: (num_tiles, 1, 1).

// enable subgroups;

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

struct TileUniforms {
    screen_width: u32,
    screen_height: u32,
    tile_size: u32,
    tiles_x: u32,
    tiles_y: u32,
    num_tiles: u32,
    visible_tet_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
};

// Group 0: read-only scene data
@group(0) @binding(0) var<storage, read> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> dl_d_image: array<f32>;
@group(0) @binding(2) var<storage, read> rendered_image: array<f32>;
@group(0) @binding(3) var<storage, read> vertices: array<f32>;
@group(0) @binding(4) var<storage, read> indices: array<u32>;
@group(0) @binding(5) var<storage, read> densities: array<f32>;
@group(0) @binding(6) var<storage, read> color_grads_buf: array<f32>;
@group(0) @binding(7) var<storage, read> colors_buf: array<f32>;
@group(0) @binding(8) var<storage, read> tile_sort_values: array<u32>;

// Group 1: read-write gradient outputs + tile metadata
@group(1) @binding(0) var<storage, read_write> d_vertices: array<atomic<f32>>;
@group(1) @binding(1) var<storage, read_write> d_densities: array<atomic<f32>>;
@group(1) @binding(2) var<storage, read_write> d_color_grads: array<atomic<f32>>;
@group(1) @binding(3) var<storage, read> tile_ranges: array<u32>;
@group(1) @binding(4) var<storage, read> tile_uniforms: TileUniforms;
@group(1) @binding(5) var<storage, read_write> d_base_colors: array<atomic<f32>>;

// --- Safe math utilities (subset of safe_math.wgsl) ---
const TINY_VAL: f32 = 1.0754944e-20;
const MIN_VAL: f32 = -1e+20;
const MAX_VAL: f32 = 1e+20;
const LOG_MAX_VAL: f32 = 46.0517;  // log(1e+20), precomputed

fn safe_clip_f32(v: f32, minv: f32, maxv: f32) -> f32 {
    return max(min(v, maxv), minv);
}

fn safe_clip_v3f(v: vec3<f32>, minv: f32, maxv: f32) -> vec3<f32> {
    return vec3<f32>(
        max(min(v.x, maxv), minv),
        max(min(v.y, maxv), minv),
        max(min(v.z, maxv), minv)
    );
}

fn safe_div_f32(a: f32, b: f32) -> f32 {
    if (abs(b) < TINY_VAL) {
        return clamp(a / TINY_VAL, MIN_VAL, MAX_VAL);
    } else {
        return clamp(a / b, MIN_VAL, MAX_VAL);
    }
}

fn safe_exp_f32(v: f32) -> f32 {
    return exp(clamp(v, -88.0, LOG_MAX_VAL));
}
// --- End safe math utilities ---

// Face (a, b, c, opposite_vertex) — opposite used to flip normal inward
const FACES: array<vec4<u32>, 4> = array<vec4<u32>, 4>(
    vec4<u32>(0u, 2u, 1u, 3u),
    vec4<u32>(1u, 2u, 3u, 0u),
    vec4<u32>(0u, 3u, 2u, 1u),
    vec4<u32>(3u, 0u, 1u, 2u),
);

// Workgroup shared memory
var<workgroup> sm_state: array<vec4<f32>, 256>;   // .xyz = color, .w = log_t
var<workgroup> sm_d_log_t: array<f32, 256>;       // running d(log_t) per pixel
var<workgroup> sm_xl: array<i32, 16>;
var<workgroup> sm_xr: array<i32, 16>;
var<workgroup> sm_prefix: array<u32, 17>;

fn phi(x: f32) -> f32 {
    if (abs(x) < 0.02) { return 1.0 + x * (-0.5 + x * (1.0/6.0 + x * (-1.0/24.0))); }
    return safe_div_f32(1.0 - exp(-x), x);
}

fn dphi_dx(x: f32) -> f32 {
    if (abs(x) < 0.02) { return -0.5 + x * (1.0/3.0 + x * (-1.0/8.0 + x * (1.0/30.0))); }
    let ex = exp(-x);
    return safe_div_f32(ex * (x + 1.0) - 1.0, x * x);
}

fn load_f32x3_v(idx: u32) -> vec3<f32> {
    return vec3<f32>(vertices[idx * 3u], vertices[idx * 3u + 1u], vertices[idx * 3u + 2u]);
}

// Native float atomicAdd (requires VK_EXT_shader_atomic_float / SHADER_FLOAT32_ATOMIC)
fn global_cas_add_density(buf_idx: u32, val: f32) {
    atomicAdd(&d_densities[buf_idx], val);
}

fn global_cas_add_vertex(buf_idx: u32, val: f32) {
    atomicAdd(&d_vertices[buf_idx], val);
}

fn global_cas_add_color_grad(buf_idx: u32, val: f32) {
    atomicAdd(&d_color_grads[buf_idx], val);
}

fn global_cas_add_base_color(buf_idx: u32, val: f32) {
    atomicAdd(&d_base_colors[buf_idx], val);
}

// Full warp reduction: sum across 32 lanes via butterfly XOR pattern.
fn warp_reduce(val: f32) -> f32 {
    var v = val;
    v += subgroupShuffleXor(v, 16u);
    v += subgroupShuffleXor(v, 8u);
    v += subgroupShuffleXor(v, 4u);
    v += subgroupShuffleXor(v, 2u);
    v += subgroupShuffleXor(v, 1u);
    return v;
}

@compute @workgroup_size(32)
fn main(
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let tile_id = wg_id.x;
    if (tile_id >= tile_uniforms.num_tiles) {
        return;
    }

    let tile_x = tile_id % tile_uniforms.tiles_x;
    let tile_y = tile_id / tile_uniforms.tiles_x;
    let w = tile_uniforms.screen_width;
    let h = tile_uniforms.screen_height;
    let W = f32(w);
    let H = f32(h);
    let TS = tile_uniforms.tile_size;
    let tile_ox = f32(tile_x * TS);
    let tile_oy = f32(tile_y * TS);

    let range_start = tile_ranges[tile_id * 2u];
    let range_end = tile_ranges[tile_id * 2u + 1u];

    let cam = uniforms.cam_pos_pad.xyz;
    let vp = mat4x4<f32>(uniforms.vp_col0, uniforms.vp_col1, uniforms.vp_col2, uniforms.vp_col3);
    let c2w = mat3x3<f32>(uniforms.c2w_col0.xyz, uniforms.c2w_col1.xyz, uniforms.c2w_col2.xyz);
    let fx = uniforms.intrinsics.x;
    let fy = uniforms.intrinsics.y;
    let cx_cam = uniforms.intrinsics.z;
    let cy_cam = uniforms.intrinsics.w;

    // Initialize state from rendered_image (final forward state) and dl_d_image
    for (var i = lane; i < TS * TS; i += 32u) {
        let row = i / TS;
        let col = i % TS;
        let px = tile_x * TS + col;
        let py = tile_y * TS + row;
        if (px < w && py < h) {
            let idx = py * w + px;
            let a = rendered_image[idx * 4u + 3u];
            sm_state[i] = vec4<f32>(
                rendered_image[idx * 4u],
                rendered_image[idx * 4u + 1u],
                rendered_image[idx * 4u + 2u],
                log(max(1.0 - a, 1e-20)));
            sm_d_log_t[i] = -dl_d_image[idx * 4u + 3u] * (1.0 - a);
        } else {
            sm_state[i] = vec4<f32>(0.0);
            sm_d_log_t[i] = 0.0;
        }
    }
    workgroupBarrier();

    // Process tets back-to-front (reverse of forward order, for state undoing)
    for (var cursor = range_start; cursor < range_end; cursor += 1u) {
        let tet_id = tile_sort_values[cursor];

        // Load tet geometry
        let vi0 = indices[tet_id * 4u];
        let vi1 = indices[tet_id * 4u + 1u];
        let vi2 = indices[tet_id * 4u + 2u];
        let vi3 = indices[tet_id * 4u + 3u];
        let v0 = load_f32x3_v(vi0);
        let v1 = load_f32x3_v(vi1);
        let v2 = load_f32x3_v(vi2);
        let v3 = load_f32x3_v(vi3);

        var verts: array<vec3<f32>, 4>;
        verts[0] = v0; verts[1] = v1; verts[2] = v2; verts[3] = v3;
        var vidx: array<u32, 4>;
        vidx[0] = vi0; vidx[1] = vi1; vidx[2] = vi2; vidx[3] = vi3;

        // Project to clip space
        let c0 = vp * vec4<f32>(v0, 1.0);
        let c1 = vp * vec4<f32>(v1, 1.0);
        let c2 = vp * vec4<f32>(v2, 1.0);
        let c3 = vp * vec4<f32>(v3, 1.0);
        let any_behind = (c0.w <= 0.0) || (c1.w <= 0.0) || (c2.w <= 0.0) || (c3.w <= 0.0);

        var proj: array<vec2<f32>, 4>;
        if (!any_behind) {
            proj[0] = vec2<f32>((c0.x / c0.w + 1.0) * 0.5 * W - tile_ox, (1.0 - c0.y / c0.w) * 0.5 * H - tile_oy);
            proj[1] = vec2<f32>((c1.x / c1.w + 1.0) * 0.5 * W - tile_ox, (1.0 - c1.y / c1.w) * 0.5 * H - tile_oy);
            proj[2] = vec2<f32>((c2.x / c2.w + 1.0) * 0.5 * W - tile_ox, (1.0 - c2.y / c2.w) * 0.5 * H - tile_oy);
            proj[3] = vec2<f32>((c3.x / c3.w + 1.0) * 0.5 * W - tile_ox, (1.0 - c3.y / c3.w) * 0.5 * H - tile_oy);
        }

        // Scanline fill: threads 0..TS compute row ranges
        if (lane < TS) {
            var xl_f: f32 = 1e10;
            var xr_f: f32 = -1e10;
            if (!any_behind) {
                let yc = f32(lane) + 0.5;
                // Edge (0, 1)
                { let yi = proj[0].y; let yj = proj[1].y;
                  if ((yi <= yc && yj > yc) || (yj <= yc && yi > yc)) {
                      let t = (yc - yi) / (yj - yi);
                      let x = proj[0].x + t * (proj[1].x - proj[0].x);
                      xl_f = min(xl_f, x); xr_f = max(xr_f, x); } }
                // Edge (0, 2)
                { let yi = proj[0].y; let yj = proj[2].y;
                  if ((yi <= yc && yj > yc) || (yj <= yc && yi > yc)) {
                      let t = (yc - yi) / (yj - yi);
                      let x = proj[0].x + t * (proj[2].x - proj[0].x);
                      xl_f = min(xl_f, x); xr_f = max(xr_f, x); } }
                // Edge (0, 3)
                { let yi = proj[0].y; let yj = proj[3].y;
                  if ((yi <= yc && yj > yc) || (yj <= yc && yi > yc)) {
                      let t = (yc - yi) / (yj - yi);
                      let x = proj[0].x + t * (proj[3].x - proj[0].x);
                      xl_f = min(xl_f, x); xr_f = max(xr_f, x); } }
                // Edge (1, 2)
                { let yi = proj[1].y; let yj = proj[2].y;
                  if ((yi <= yc && yj > yc) || (yj <= yc && yi > yc)) {
                      let t = (yc - yi) / (yj - yi);
                      let x = proj[1].x + t * (proj[2].x - proj[1].x);
                      xl_f = min(xl_f, x); xr_f = max(xr_f, x); } }
                // Edge (1, 3)
                { let yi = proj[1].y; let yj = proj[3].y;
                  if ((yi <= yc && yj > yc) || (yj <= yc && yi > yc)) {
                      let t = (yc - yi) / (yj - yi);
                      let x = proj[1].x + t * (proj[3].x - proj[1].x);
                      xl_f = min(xl_f, x); xr_f = max(xr_f, x); } }
                // Edge (2, 3)
                { let yi = proj[2].y; let yj = proj[3].y;
                  if ((yi <= yc && yj > yc) || (yj <= yc && yi > yc)) {
                      let t = (yc - yi) / (yj - yi);
                      let x = proj[2].x + t * (proj[3].x - proj[2].x);
                      xl_f = min(xl_f, x); xr_f = max(xr_f, x); } }
            }
            if (xl_f <= xr_f) {
                let eps = 0.001;
                let xl_i = max(i32(ceil(xl_f - 0.5 - eps)), 0);
                let xr_i = min(i32(floor(xr_f - 0.5 + eps)), i32(TS) - 1);
                if (xl_i <= xr_i) {
                    sm_xl[lane] = xl_i;
                    sm_xr[lane] = xr_i;
                } else {
                    sm_xl[lane] = 0;
                    sm_xr[lane] = -1;
                }
            } else {
                sm_xl[lane] = 0;
                sm_xr[lane] = -1;
            }
        }
        workgroupBarrier();

        if (lane == 0u) {
            sm_prefix[0] = 0u;
            for (var r = 0u; r < TS; r++) {
                let row_w = u32(max(sm_xr[r] - sm_xl[r] + 1, 0));
                sm_prefix[r + 1u] = sm_prefix[r] + row_w;
            }
        }
        workgroupBarrier();

        let total = sm_prefix[TS];
        if (total == 0u) {
            continue;
        }

        // Load tet attributes
        let density_raw = densities[tet_id];
        let colors_tet = vec3<f32>(colors_buf[tet_id * 3u], colors_buf[tet_id * 3u + 1u], colors_buf[tet_id * 3u + 2u]);
        let grad_vec = vec3<f32>(color_grads_buf[tet_id * 3u], color_grads_buf[tet_id * 3u + 1u], color_grads_buf[tet_id * 3u + 2u]);

        // Per-thread gradient accumulators (sum across all pixels this thread processes)
        // NOTE: explicit zero-init required — WGSL var inside a loop may retain
        // values from the previous iteration in some runtimes (naga/wgpu).
        var d_density_accum: f32 = 0.0;
        var d_vert_accum: array<vec3<f32>, 4>;
        d_vert_accum[0] = vec3<f32>(0.0);
        d_vert_accum[1] = vec3<f32>(0.0);
        d_vert_accum[2] = vec3<f32>(0.0);
        d_vert_accum[3] = vec3<f32>(0.0);
        var d_grad_accum = vec3<f32>(0.0);
        var d_base_colors_accum = vec3<f32>(0.0);

        // Process covered pixels
        for (var idx = lane; idx < total; idx += 32u) {
            var row = 0u;
            for (var r = 0u; r < TS; r++) {
                if (idx < sm_prefix[r + 1u]) {
                    row = r;
                    break;
                }
            }
            let col = u32(sm_xl[row]) + (idx - sm_prefix[row]);
            let pixel_local = row * TS + col;
            let px = tile_x * TS + col;
            let py = tile_y * TS + row;

            if (px >= w || py >= h) {
                continue;
            }

            let pixel_idx = py * w + px;

            // Compute ray direction from camera intrinsics (matches camera.slang get_ray):
            let dir_cam = normalize(vec3<f32>(
                (f32(px) + 0.5 - cx_cam) / fx,
                (f32(py) + 0.5 - cy_cam) / fy,
                1.0
            ));
            let ray_dir = normalize(c2w * dir_cam);

            // Ray-tet intersection
            var t_min_val = -3.402823e38;
            var t_max_val = 3.402823e38;
            var valid = true;
            var min_face = 0u;
            var max_face = 0u;

            for (var fi = 0u; fi < 4u; fi++) {
                let f = FACES[fi];
                let va = verts[f[0]]; let vb = verts[f[1]]; let vc = verts[f[2]];
                var n = cross(vc - va, vb - va);
                // Flip normal to point inward (toward opposite vertex)
                let v_opp = verts[f[3]];
                if (dot(n, v_opp - va) < 0.0) {
                    n = -n;
                }
                let num = dot(n, va - cam);
                let den = dot(n, ray_dir);
                if (abs(den) < 1e-20) {
                    if (num > 0.0) { valid = false; }
                    continue;
                }
                let t = num / den;
                if (den > 0.0) {
                    if (t > t_min_val) { t_min_val = t; min_face = fi; }
                } else {
                    if (t < t_max_val) { t_max_val = t; max_face = fi; }
                }
            }

            if (!valid || t_min_val >= t_max_val) {
                continue;
            }

            // Forward quantities
            let base_offset = dot(grad_vec, cam - verts[0]);
            let base_color = colors_tet + vec3<f32>(base_offset);
            let dc_dt = dot(grad_vec, ray_dir);

            let c_start_raw = base_color + vec3<f32>(dc_dt * t_min_val);
            let c_end_raw = base_color + vec3<f32>(dc_dt * t_max_val);
            let c_start = max(c_start_raw, vec3<f32>(0.0));
            let c_end = max(c_end_raw, vec3<f32>(0.0));

            let dist = t_max_val - t_min_val;
            let od = clamp(density_raw * dist, 1e-8, 88.0);
            let alpha_t = safe_exp_f32(-od);
            let phi_val = phi(od);
            let w0 = phi_val - alpha_t;
            let w1 = 1.0 - phi_val;
            let c_premul = safe_clip_v3f(c_end * w0 + c_start * w1, MIN_VAL, MAX_VAL);

            // Read running state from shared memory
            let state = sm_state[pixel_local];
            let color = state.xyz;
            let log_t = state.w;
            let d_log_t = sm_d_log_t[pixel_local];

            // Undo this tet's contribution (recover state before this tet in forward order)
            // Clamp prev_log_t to <= 0: transmittance can never exceed 1.0.
            // The backward initializes log_t from log(1-alpha) which saturates at ~-46
            // for alpha≈1, but the true accumulated log_t can be -6000+. Undoing a
            // high-od tet would push prev_log_t far above 0, causing gradient explosion.
            let prev_log_t = min(log_t + od, 0.0);
            let t_prev = safe_exp_f32(prev_log_t);
            let prev_color = color - c_premul * t_prev;

            // Update shared memory state (always, even if we skip gradient)
            sm_state[pixel_local] = vec4<f32>(prev_color, prev_log_t);

            // Early termination: if transmittance before this tet was negligible,
            // it contributed ~0 to the image in forward, so its gradient is ~0.
            // Skip expensive gradient computation.
            if (t_prev < 1e-6) {
                sm_d_log_t[pixel_local] = d_log_t;
                continue;
            }

            // Load per-pixel upstream color gradient (constant per pixel)
            let d_color = vec3<f32>(
                dl_d_image[pixel_idx * 4u],
                dl_d_image[pixel_idx * 4u + 1u],
                dl_d_image[pixel_idx * 4u + 2u],
            );

            // === Backward computation (matching non-tiled backward_compute.wgsl) ===
            let d_c_premul = safe_clip_v3f(d_color * t_prev, MIN_VAL, MAX_VAL);
            let d_od_state = -d_log_t;
            let d_old_log_t = safe_clip_f32(d_log_t + dot(d_color, c_premul) * t_prev, MIN_VAL, MAX_VAL);

            sm_d_log_t[pixel_local] = d_old_log_t;

            let dphi_val = dphi_dx(od);
            let dw0_dod = dphi_val + alpha_t;
            let dw1_dod = -dphi_val;

            let d_c_end_integral = d_c_premul * w0;
            let d_c_start_integral = d_c_premul * w1;
            let d_od_integral = dot(d_c_premul, c_end * dw0_dod + c_start * dw1_dod);

            let d_od = safe_clip_f32(d_od_state + d_od_integral, MIN_VAL, MAX_VAL);
            // Respect the clamp(od_raw, 1e-8, 88.0) derivative: when od_raw is
            // outside [1e-8, 88], the clamp output is constant so d(od)/d(od_raw)=0.
            let od_raw = density_raw * dist;
            let od_active = (od_raw >= 1e-8) && (od_raw <= 88.0);
            let d_od_raw = select(0.0, d_od, od_active);
            let d_density_local = safe_clip_f32(d_od_raw * dist, MIN_VAL, MAX_VAL);
            let d_dist = safe_clip_f32(d_od_raw * density_raw, MIN_VAL, MAX_VAL);
            var d_t_min = -d_dist;
            var d_t_max = d_dist;

            // Color chain through ReLU
            let d_c_start_raw = vec3<f32>(
                select(0.0, d_c_start_integral.x, c_start_raw.x > 0.0),
                select(0.0, d_c_start_integral.y, c_start_raw.y > 0.0),
                select(0.0, d_c_start_integral.z, c_start_raw.z > 0.0),
            );
            let d_c_end_raw = vec3<f32>(
                select(0.0, d_c_end_integral.x, c_end_raw.x > 0.0),
                select(0.0, d_c_end_integral.y, c_end_raw.y > 0.0),
                select(0.0, d_c_end_integral.z, c_end_raw.z > 0.0),
            );

            let d_base_color = d_c_start_raw + d_c_end_raw;
            let d_dc_dt_val = (d_c_start_raw.x + d_c_start_raw.y + d_c_start_raw.z) * t_min_val
                + (d_c_end_raw.x + d_c_end_raw.y + d_c_end_raw.z) * t_max_val;
            d_t_min += (d_c_start_raw.x + d_c_start_raw.y + d_c_start_raw.z) * dc_dt;
            d_t_max += (d_c_end_raw.x + d_c_end_raw.y + d_c_end_raw.z) * dc_dt;

            let d_base_offset_scalar = d_base_color.x + d_base_color.y + d_base_color.z;
            var d_grad_local = (cam - verts[0]) * d_base_offset_scalar;
            let d_v0_from_base = -grad_vec * d_base_offset_scalar;
            d_grad_local += ray_dir * d_dc_dt_val;

            // Gradient w.r.t. base_colors
            d_base_colors_accum += safe_clip_v3f(d_base_color, MIN_VAL, MAX_VAL);

            // Intersection gradients
            // NOTE: explicit zero-init required — WGSL var inside a loop may not
            // be re-zero-initialized on each iteration in some runtimes.
            var d_vert_local: array<vec3<f32>, 4>;
            d_vert_local[0] = vec3<f32>(0.0);
            d_vert_local[1] = vec3<f32>(0.0);
            d_vert_local[2] = vec3<f32>(0.0);
            d_vert_local[3] = vec3<f32>(0.0);
            {
                let f = FACES[min_face];
                let va = verts[f[0]]; let vb = verts[f[1]]; let vc = verts[f[2]];
                let n = cross(vc - va, vb - va);
                let den = dot(n, ray_dir);
                let inv_den = safe_div_f32(1.0, den);
                let hit = cam + ray_dir * t_min_val;
                let dt_dva = (cross(va - hit, vb - vc) + n) * inv_den;
                let dt_dvb = cross(va - hit, vc - va) * inv_den;
                let dt_dvc = cross(va - hit, va - vb) * inv_den;
                d_vert_local[f[0]] += safe_clip_v3f(dt_dva * d_t_min, MIN_VAL, MAX_VAL);
                d_vert_local[f[1]] += safe_clip_v3f(dt_dvb * d_t_min, MIN_VAL, MAX_VAL);
                d_vert_local[f[2]] += safe_clip_v3f(dt_dvc * d_t_min, MIN_VAL, MAX_VAL);
            }
            {
                let f = FACES[max_face];
                let va = verts[f[0]]; let vb = verts[f[1]]; let vc = verts[f[2]];
                let n = cross(vc - va, vb - va);
                let den = dot(n, ray_dir);
                let inv_den = safe_div_f32(1.0, den);
                let hit = cam + ray_dir * t_max_val;
                let dt_dva = (cross(va - hit, vb - vc) + n) * inv_den;
                let dt_dvb = cross(va - hit, vc - va) * inv_den;
                let dt_dvc = cross(va - hit, va - vb) * inv_den;
                d_vert_local[f[0]] += safe_clip_v3f(dt_dva * d_t_max, MIN_VAL, MAX_VAL);
                d_vert_local[f[1]] += safe_clip_v3f(dt_dvb * d_t_max, MIN_VAL, MAX_VAL);
                d_vert_local[f[2]] += safe_clip_v3f(dt_dvc * d_t_max, MIN_VAL, MAX_VAL);
            }

            d_vert_local[0] += d_v0_from_base;

            // Accumulate per-thread (clamp contributions to prevent overflow)
            d_density_accum += safe_clip_f32(d_density_local, MIN_VAL, MAX_VAL);
            d_vert_accum[0] += safe_clip_v3f(d_vert_local[0], MIN_VAL, MAX_VAL);
            d_vert_accum[1] += safe_clip_v3f(d_vert_local[1], MIN_VAL, MAX_VAL);
            d_vert_accum[2] += safe_clip_v3f(d_vert_local[2], MIN_VAL, MAX_VAL);
            d_vert_accum[3] += safe_clip_v3f(d_vert_local[3], MIN_VAL, MAX_VAL);
            d_grad_accum += safe_clip_v3f(d_grad_local, MIN_VAL, MAX_VAL);
        }

        // ===== Warp reduce + flush =====
        let reduced_d_density = warp_reduce(d_density_accum);

        var reduced_d_vert: array<vec3<f32>, 4>;
        for (var vi = 0u; vi < 4u; vi++) {
            reduced_d_vert[vi].x = warp_reduce(d_vert_accum[vi].x);
            reduced_d_vert[vi].y = warp_reduce(d_vert_accum[vi].y);
            reduced_d_vert[vi].z = warp_reduce(d_vert_accum[vi].z);
        }
        let reduced_d_grad = vec3<f32>(
            warp_reduce(d_grad_accum.x),
            warp_reduce(d_grad_accum.y),
            warp_reduce(d_grad_accum.z),
        );
        let reduced_d_base = vec3<f32>(
            warp_reduce(d_base_colors_accum.x),
            warp_reduce(d_base_colors_accum.y),
            warp_reduce(d_base_colors_accum.z),
        );

        // Lane 0 writes to global memory
        if (lane == 0u) {
            global_cas_add_density(tet_id, reduced_d_density);

            for (var vi = 0u; vi < 4u; vi++) {
                let gi = vidx[vi] * 3u;
                global_cas_add_vertex(gi, reduced_d_vert[vi].x);
                global_cas_add_vertex(gi + 1u, reduced_d_vert[vi].y);
                global_cas_add_vertex(gi + 2u, reduced_d_vert[vi].z);
            }

            global_cas_add_color_grad(tet_id * 3u, reduced_d_grad.x);
            global_cas_add_color_grad(tet_id * 3u + 1u, reduced_d_grad.y);
            global_cas_add_color_grad(tet_id * 3u + 2u, reduced_d_grad.z);

            global_cas_add_base_color(tet_id * 3u, reduced_d_base.x);
            global_cas_add_base_color(tet_id * 3u + 1u, reduced_d_base.y);
            global_cas_add_base_color(tet_id * 3u + 2u, reduced_d_base.z);
        }
        workgroupBarrier();
    }
}
