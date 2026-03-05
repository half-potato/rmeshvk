// Forward-order tiled backward compute shader with shared-memory gradient accumulation.
//
// Phase 1: Iterates nearest -> furthest (forward compositing order).
//   Uses closed-form d_log_t = d_log_t_final + dot(d_color, color_final - color_after),
//   eliminating sequential d_log_t state tracking and the "undo" step.
//
// Phase 2: Per-batch shared-memory gradient accumulators.
//   Each batch of BATCH_SIZE tets accumulates gradients in workgroup memory,
//   then flushes to global with one atomic add per entry per tile.
//   Reduces global atomic operations by ~Nx (N = avg pixel hits per tet per tile).
//
// One workgroup per tile (16x16 pixels). Each thread = one pixel.
// Sort order: range_start = furthest, range_end = nearest.
// Forward compositing: nearest first = range_end-1 down to range_start.

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

struct TileUniforms {
    screen_width: u32,
    screen_height: u32,
    tile_size: u32,
    tiles_x: u32,
    tiles_y: u32,
    num_tiles: u32,
    visible_tet_count: u32,
    max_pairs: u32,
    max_pairs_pow2: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

// Group 0: read-only scene data
@group(0) @binding(0) var<storage, read> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> dl_d_image: array<f32>;
@group(0) @binding(2) var<storage, read> rendered_image: array<f32>;
@group(0) @binding(3) var<storage, read> vertices: array<f32>;
@group(0) @binding(4) var<storage, read> indices: array<u32>;
@group(0) @binding(5) var<storage, read> sh_coeffs: array<f32>;
@group(0) @binding(6) var<storage, read> densities: array<f32>;
@group(0) @binding(7) var<storage, read> color_grads_buf: array<f32>;
@group(0) @binding(8) var<storage, read> circumdata: array<f32>;
@group(0) @binding(9) var<storage, read> colors_buf: array<f32>;
@group(0) @binding(10) var<storage, read> tile_sort_values: array<u32>;

// Group 1: read-write gradient outputs + tile metadata
@group(1) @binding(0) var<storage, read_write> d_sh_coeffs: array<atomic<u32>>;
@group(1) @binding(1) var<storage, read_write> d_vertices: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read_write> d_densities: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read_write> d_color_grads: array<atomic<u32>>;
@group(1) @binding(4) var<storage, read> tile_ranges: array<u32>;
@group(1) @binding(5) var<storage, read> tile_uniforms: TileUniforms;
@group(1) @binding(6) var<storage, read_write> debug_image: array<f32>;

const TINY_VAL: f32 = 1e-20;
const BATCH_SIZE: u32 = 32u;
const MAX_GRAD_PER_TET: u32 = 64u;
const GRAD_BUF_SIZE: u32 = 2048u; // BATCH_SIZE * MAX_GRAD_PER_TET
const WG_SIZE: u32 = 256u;

// Gradient layout per tet (within MAX_GRAD_PER_TET block):
//   [0]:       density
//   [1..12]:   vertex gradients (4 verts x 3 axes)
//   [13..15]:  color gradient gradients (3 axes)
//   [16..63]:  SH coefficient gradients (up to 3 channels x 16 basis)

// SH Constants
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

// Face winding
const FACES: array<vec3<u32>, 4> = array<vec3<u32>, 4>(
    vec3<u32>(0u, 2u, 1u),
    vec3<u32>(1u, 2u, 3u),
    vec3<u32>(0u, 3u, 2u),
    vec3<u32>(3u, 0u, 1u),
);

// Shared memory for cooperative tet loading
struct TetData {
    tet_id: u32,
    density_raw: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    grad_x: f32,
    grad_y: f32,
    grad_z: f32,
};

var<workgroup> shared_tets: array<TetData, 32>;
var<workgroup> wg_grads: array<atomic<u32>, 2048>;

fn phi(x: f32) -> f32 {
    if (abs(x) < 1e-6) { return 1.0 - x * 0.5; }
    return (1.0 - exp(-x)) / x;
}

fn dphi_dx(x: f32) -> f32 {
    if (abs(x) < 1e-6) { return -0.5 + x / 3.0; }
    return (exp(-x) * (1.0 + x) - 1.0) / (x * x);
}

fn softplus(x: f32) -> f32 {
    if (x > 8.0) { return x; }
    return 0.1 * log(1.0 + exp(10.0 * x));
}

fn dsoftplus(x: f32) -> f32 {
    if (x > 8.0) { return 1.0; }
    let e = exp(10.0 * x);
    return e / (1.0 + e);
}

fn load_f32x3_v(idx: u32) -> vec3<f32> {
    return vec3<f32>(vertices[idx * 3u], vertices[idx * 3u + 1u], vertices[idx * 3u + 2u]);
}

fn eval_sh(dir: vec3<f32>, sh_degree: u32, base: u32) -> f32 {
    let x = dir.x; let y = dir.y; let z = dir.z;
    var val = C0 * sh_coeffs[base];
    if (sh_degree >= 1u) {
        val += -C1 * y * sh_coeffs[base + 1u];
        val += C1 * z * sh_coeffs[base + 2u];
        val += -C1 * x * sh_coeffs[base + 3u];
    }
    if (sh_degree >= 2u) {
        let xx = x * x; let yy = y * y; let zz = z * z;
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

// Atomic f32 add on workgroup memory
fn wg_add_f32(idx: u32, val: f32) {
    var ob = atomicLoad(&wg_grads[idx]);
    loop {
        let nv = bitcast<u32>(bitcast<f32>(ob) + val);
        let r = atomicCompareExchangeWeak(&wg_grads[idx], ob, nv);
        if (r.exchanged) { break; }
        ob = r.old_value;
    }
}

// Scatter SH gradients to workgroup shared memory
fn scatter_sh_grads_wg(dir: vec3<f32>, sh_degree: u32, d_sh_result: vec3<f32>, bi: u32, nc: u32) {
    let x = dir.x; let y = dir.y; let z = dir.z;
    var basis: array<f32, 16>;
    basis[0] = C0;
    var n_basis = 1u;
    if (sh_degree >= 1u) {
        basis[1] = -C1 * y;
        basis[2] = C1 * z;
        basis[3] = -C1 * x;
        n_basis = 4u;
    }
    if (sh_degree >= 2u) {
        let xx = x * x; let yy = y * y; let zz = z * z;
        basis[4] = C2_0 * x * y;
        basis[5] = C2_1 * y * z;
        basis[6] = C2_2 * (2.0 * zz - xx - yy);
        basis[7] = C2_3 * x * z;
        basis[8] = C2_4 * (xx - yy);
        n_basis = 9u;
        if (sh_degree >= 3u) {
            basis[9] = C3_0 * y * (3.0 * xx - yy);
            basis[10] = C3_1 * x * y * z;
            basis[11] = C3_2 * y * (4.0 * zz - xx - yy);
            basis[12] = C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
            basis[13] = C3_4 * x * (4.0 * zz - xx - yy);
            basis[14] = C3_5 * z * (xx - yy);
            basis[15] = C3_6 * x * (xx - 3.0 * yy);
            n_basis = 16u;
        }
    }
    let d_channels = array<f32, 3>(d_sh_result.x, d_sh_result.y, d_sh_result.z);
    let base_offset = bi * MAX_GRAD_PER_TET + 16u;
    for (var c = 0u; c < 3u; c++) {
        for (var k = 0u; k < n_basis; k++) {
            let add_val = d_channels[c] * basis[k];
            wg_add_f32(base_offset + c * nc + k, add_val);
        }
    }
}

// Atomic f32 add on global storage buffer (generic CAS pattern)
fn global_cas_add(buf_idx: u32, val: f32, buf_type: u32) {
    // buf_type: 0=d_densities, 1=d_vertices, 2=d_color_grads, 3=d_sh_coeffs
    if (buf_type == 0u) {
        var ob = atomicLoad(&d_densities[buf_idx]);
        loop {
            let nv = bitcast<u32>(bitcast<f32>(ob) + val);
            let r = atomicCompareExchangeWeak(&d_densities[buf_idx], ob, nv);
            if (r.exchanged) { break; }
            ob = r.old_value;
        }
    } else if (buf_type == 1u) {
        var ob = atomicLoad(&d_vertices[buf_idx]);
        loop {
            let nv = bitcast<u32>(bitcast<f32>(ob) + val);
            let r = atomicCompareExchangeWeak(&d_vertices[buf_idx], ob, nv);
            if (r.exchanged) { break; }
            ob = r.old_value;
        }
    } else if (buf_type == 2u) {
        var ob = atomicLoad(&d_color_grads[buf_idx]);
        loop {
            let nv = bitcast<u32>(bitcast<f32>(ob) + val);
            let r = atomicCompareExchangeWeak(&d_color_grads[buf_idx], ob, nv);
            if (r.exchanged) { break; }
            ob = r.old_value;
        }
    } else {
        var ob = atomicLoad(&d_sh_coeffs[buf_idx]);
        loop {
            let nv = bitcast<u32>(bitcast<f32>(ob) + val);
            let r = atomicCompareExchangeWeak(&d_sh_coeffs[buf_idx], ob, nv);
            if (r.exchanged) { break; }
            ob = r.old_value;
        }
    }
}

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let tile_x = wg_id.x;
    let tile_y = wg_id.y;

    // Bounds check: tile outside grid
    if (tile_x >= tile_uniforms.tiles_x || tile_y >= tile_uniforms.tiles_y) {
        return;
    }

    let tile_id = tile_y * tile_uniforms.tiles_x + tile_x;

    // Load tile range
    let range_start = tile_ranges[tile_id * 2u];
    let range_end = tile_ranges[tile_id * 2u + 1u];

    // Pixel coordinates
    let px = tile_x * tile_uniforms.tile_size + (local_idx % tile_uniforms.tile_size);
    let py = tile_y * tile_uniforms.tile_size + (local_idx / tile_uniforms.tile_size);
    let w = tile_uniforms.screen_width;
    let h = tile_uniforms.screen_height;

    let valid_pixel = (px < w) && (py < h);

    // Load per-pixel constants (invariant across all tets)
    let pixel_idx = py * w + px;

    var d_color = vec3<f32>(0.0);
    var d_log_t_final: f32 = 0.0;
    var color_final = vec3<f32>(0.0);

    if (valid_pixel) {
        d_color = vec3<f32>(
            dl_d_image[pixel_idx * 4u],
            dl_d_image[pixel_idx * 4u + 1u],
            dl_d_image[pixel_idx * 4u + 2u],
        );
        // image_alpha = 1 - T_final (opacity). So T_final = 1 - alpha_final.
        // d_log_t_final = dL/d_log_t = dL/d_alpha * d_alpha/d_T * dT/d_log_t
        //               = dl_d_alpha * (-1) * T_final
        let alpha_final_val = rendered_image[pixel_idx * 4u + 3u];
        d_log_t_final = -dl_d_image[pixel_idx * 4u + 3u] * (1.0 - alpha_final_val);
        color_final = vec3<f32>(
            rendered_image[pixel_idx * 4u],
            rendered_image[pixel_idx * 4u + 1u],
            rendered_image[pixel_idx * 4u + 2u],
        );
    }

    // Forward state: starts at initial (no compositing yet)
    var color_accum = vec3<f32>(0.0);
    var log_t: f32 = 0.0;

    // Compute ray from pixel coordinates via inverse VP
    // wgpu maps ndc_y to screen as: pixel_y = (1 - ndc_y) * 0.5 * H
    // So: ndc_y = 1 - (2 * (pixel_y + 0.5) / H)
    let ndc_x = (2.0 * (f32(px) + 0.5) / f32(w)) - 1.0;
    let ndc_y = 1.0 - (2.0 * (f32(py) + 0.5) / f32(h));

    let inv_vp = mat4x4<f32>(uniforms.inv_vp_col0, uniforms.inv_vp_col1, uniforms.inv_vp_col2, uniforms.inv_vp_col3);
    let near_clip = inv_vp * vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    let far_clip = inv_vp * vec4<f32>(ndc_x, ndc_y, 1.0, 1.0);
    let near_world = near_clip.xyz / near_clip.w;
    let far_world = far_clip.xyz / far_clip.w;

    let cam = uniforms.cam_pos_pad.xyz;
    let ray_dir = normalize(far_world - near_world);

    let num_coeffs = (uniforms.sh_degree + 1u) * (uniforms.sh_degree + 1u);
    let sh_stride = num_coeffs * 3u;
    let nc = num_coeffs;

    // Process batches: nearest -> furthest (forward compositing order)
    // Sort order: range_start = furthest, range_end-1 = nearest
    // We iterate from the end backward, taking batches from the nearest side first.
    var cursor = range_end;

    while (cursor > range_start) {
        let remaining = cursor - range_start;
        let batch_count = min(remaining, BATCH_SIZE);
        let batch_begin = cursor - batch_count;

        // === Zero shared gradients + cooperative load ===
        // Zero wg_grads: 2048 entries, 256 threads -> 8 per thread
        for (var z = local_idx; z < GRAD_BUF_SIZE; z += WG_SIZE) {
            atomicStore(&wg_grads[z], 0u);
        }
        // Cooperative load: each of first batch_count threads loads one tet
        if (local_idx < batch_count) {
            let sort_idx = batch_begin + local_idx;
            let tet_id = tile_sort_values[sort_idx];

            shared_tets[local_idx].tet_id = tet_id;
            shared_tets[local_idx].density_raw = densities[tet_id];
            shared_tets[local_idx].color_r = colors_buf[tet_id * 3u];
            shared_tets[local_idx].color_g = colors_buf[tet_id * 3u + 1u];
            shared_tets[local_idx].color_b = colors_buf[tet_id * 3u + 2u];
            shared_tets[local_idx].grad_x = color_grads_buf[tet_id * 3u];
            shared_tets[local_idx].grad_y = color_grads_buf[tet_id * 3u + 1u];
            shared_tets[local_idx].grad_z = color_grads_buf[tet_id * 3u + 2u];
        }

        workgroupBarrier();

        // === Per-pixel processing (nearest to furthest within batch) ===
        // shared_tets[0] = furthest in batch, shared_tets[batch_count-1] = nearest
        // Iterate from batch_count-1 down to 0 for front-to-back compositing order
        if (valid_pixel) {
            for (var bi_rev = 0u; bi_rev < batch_count; bi_rev++) {
                let bi = batch_count - 1u - bi_rev;

                let tet_id = shared_tets[bi].tet_id;
                let density_raw = shared_tets[bi].density_raw;
                let colors_tet = vec3<f32>(shared_tets[bi].color_r, shared_tets[bi].color_g, shared_tets[bi].color_b);
                let grad = vec3<f32>(shared_tets[bi].grad_x, shared_tets[bi].grad_y, shared_tets[bi].grad_z);

                // Load tet geometry
                let ti0 = indices[tet_id * 4u];
                let ti1 = indices[tet_id * 4u + 1u];
                let ti2 = indices[tet_id * 4u + 2u];
                let ti3 = indices[tet_id * 4u + 3u];

                var verts: array<vec3<f32>, 4>;
                verts[0] = load_f32x3_v(ti0);
                verts[1] = load_f32x3_v(ti1);
                verts[2] = load_f32x3_v(ti2);
                verts[3] = load_f32x3_v(ti3);

                // Ray-tet intersection
                var t_min_val = -3.402823e38;
                var t_max_val = 3.402823e38;
                var min_face = 0u;
                var max_face = 0u;
                var valid = true;

                for (var fi = 0u; fi < 4u; fi++) {
                    let f = FACES[fi];
                    let va = verts[f[0]];
                    let vb = verts[f[1]];
                    let vc = verts[f[2]];
                    let n = cross(vc - va, vb - va);

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

                if (!valid || t_min_val >= t_max_val) { continue; }

                // Forward replay: colors & integral
                let base_offset = dot(grad, cam - verts[0]);
                let base_color = colors_tet + vec3<f32>(base_offset);
                let dc_dt = dot(grad, ray_dir);

                let c_start_raw = base_color + vec3<f32>(dc_dt * t_min_val);
                let c_end_raw = base_color + vec3<f32>(dc_dt * t_max_val);
                let c_start = max(c_start_raw, vec3<f32>(0.0));
                let c_end = max(c_end_raw, vec3<f32>(0.0));

                let dist = t_max_val - t_min_val;
                let od = max(density_raw * dist, 1e-8);

                let alpha_t = exp(-od);
                let phi_val = phi(od);
                let w0 = phi_val - alpha_t;
                let w1 = 1.0 - phi_val;
                let c_premul = c_end * w0 + c_start * w1;

                // Forward state: transmittance at this tet's position
                let T_j = exp(log_t);

                // Forward state update
                let color_after = color_accum + c_premul * T_j;
                let log_t_after = log_t - od;

                // Closed-form d_log_t: no sequential state tracking needed
                let d_log_t_j = d_log_t_final + dot(d_color, color_final - color_after);

                // Gradient computation (same math as original, using closed-form d_log_t_j)
                let d_c_premul = d_color * T_j;
                let d_od_state = -d_log_t_j;

                // Backward through compute_integral
                let dphi_val = dphi_dx(od);
                let dw0_dod = dphi_val + alpha_t;
                let dw1_dod = -dphi_val;

                let d_c_end_integral = d_c_premul * w0;
                let d_c_start_integral = d_c_premul * w1;
                let d_od_integral = dot(d_c_premul, c_end * dw0_dod + c_start * dw1_dod);

                let d_od = d_od_state + d_od_integral;

                // Backward: od and dist
                let d_density_local = d_od * dist;
                let d_dist = d_od * density_raw;
                var d_t_min = -d_dist;
                var d_t_max = d_dist;

                // Backward: color chain (through ReLU clamp)
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
                let d_dc_dt = (d_c_start_raw.x + d_c_start_raw.y + d_c_start_raw.z) * t_min_val
                    + (d_c_end_raw.x + d_c_end_raw.y + d_c_end_raw.z) * t_max_val;
                d_t_min += (d_c_start_raw.x + d_c_start_raw.y + d_c_start_raw.z) * dc_dt;
                d_t_max += (d_c_end_raw.x + d_c_end_raw.y + d_c_end_raw.z) * dc_dt;

                // base_color = colors + grad.(cam - v0)
                let d_base_offset_scalar = d_base_color.x + d_base_color.y + d_base_color.z;
                var d_grad = (cam - verts[0]) * d_base_offset_scalar;
                let d_v0_from_base = -grad * d_base_offset_scalar;

                // dc_dt = grad . ray_dir
                d_grad += ray_dir * d_dc_dt;

                // Backward: softplus
                let centroid = (verts[0] + verts[1] + verts[2] + verts[3]) * 0.25;
                let sh_dir = normalize(centroid - cam);

                let sh_base = tet_id * sh_stride;
                var sh_result = vec3<f32>(0.0);
                sh_result.x = eval_sh(sh_dir, uniforms.sh_degree, sh_base);
                sh_result.y = eval_sh(sh_dir, uniforms.sh_degree, sh_base + nc);
                sh_result.z = eval_sh(sh_dir, uniforms.sh_degree, sh_base + 2u * nc);

                let offset_val = dot(grad, verts[0] - centroid);
                let sp_input = sh_result + vec3<f32>(0.5 + offset_val);

                let d_sp_input = vec3<f32>(
                    d_base_color.x * dsoftplus(sp_input.x),
                    d_base_color.y * dsoftplus(sp_input.y),
                    d_base_color.z * dsoftplus(sp_input.z),
                );

                // sp_input = sh_result + 0.5 + offset
                let d_sh_result = d_sp_input;
                let d_offset_scalar = d_sp_input.x + d_sp_input.y + d_sp_input.z;

                d_grad += (verts[0] - centroid) * d_offset_scalar;
                let d_v0_from_offset = grad * d_offset_scalar * 0.75;
                let d_vother_from_offset = -grad * d_offset_scalar * 0.25;

                // SH gradients -> shared memory
                scatter_sh_grads_wg(sh_dir, uniforms.sh_degree, d_sh_result, bi, nc);

                // Intersection gradients
                var d_vert_i: array<vec3<f32>, 4>;

                // t_min face
                {
                    let f = FACES[min_face];
                    let va = verts[f[0]]; let vb = verts[f[1]]; let vc = verts[f[2]];
                    let n = cross(vc - va, vb - va);
                    let den = dot(n, ray_dir);
                    let hit = cam + ray_dir * t_min_val;

                    let dt_dva = (cross(va - hit, vb - vc) + n) * (1.0 / den);
                    let dt_dvb = cross(va - hit, vc - va) * (1.0 / den);
                    let dt_dvc = cross(va - hit, va - vb) * (1.0 / den);

                    d_vert_i[f[0]] += dt_dva * d_t_min;
                    d_vert_i[f[1]] += dt_dvb * d_t_min;
                    d_vert_i[f[2]] += dt_dvc * d_t_min;
                }

                // t_max face
                {
                    let f = FACES[max_face];
                    let va = verts[f[0]]; let vb = verts[f[1]]; let vc = verts[f[2]];
                    let n = cross(vc - va, vb - va);
                    let den = dot(n, ray_dir);
                    let hit = cam + ray_dir * t_max_val;

                    let dt_dva = (cross(va - hit, vb - vc) + n) * (1.0 / den);
                    let dt_dvb = cross(va - hit, vc - va) * (1.0 / den);
                    let dt_dvc = cross(va - hit, va - vb) * (1.0 / den);

                    d_vert_i[f[0]] += dt_dva * d_t_max;
                    d_vert_i[f[1]] += dt_dvb * d_t_max;
                    d_vert_i[f[2]] += dt_dvc * d_t_max;
                }

                // Combine vertex gradients
                d_vert_i[0] += d_v0_from_base + d_v0_from_offset;
                d_vert_i[1] += d_vother_from_offset;
                d_vert_i[2] += d_vother_from_offset;
                d_vert_i[3] += d_vother_from_offset;

                // Scatter all gradients to shared memory
                let grad_base = bi * MAX_GRAD_PER_TET;

                // Density
                wg_add_f32(grad_base, d_density_local);

                // Vertices (4 verts x 3 axes)
                for (var vi = 0u; vi < 4u; vi++) {
                    let dv = d_vert_i[vi];
                    wg_add_f32(grad_base + 1u + vi * 3u, dv.x);
                    wg_add_f32(grad_base + 1u + vi * 3u + 1u, dv.y);
                    wg_add_f32(grad_base + 1u + vi * 3u + 2u, dv.z);
                }

                // Color gradients (3 axes)
                wg_add_f32(grad_base + 13u, d_grad.x);
                wg_add_f32(grad_base + 14u, d_grad.y);
                wg_add_f32(grad_base + 15u, d_grad.z);

                // Update forward state
                color_accum = color_after;
                log_t = log_t_after;
            } // end per-tet loop
        } // end valid_pixel

        workgroupBarrier();

        // === Flush shared gradients to global buffers ===
        // Distribute flush work across all 256 threads
        let num_entries_per_tet = 16u + sh_stride;
        let total_flush_entries = batch_count * num_entries_per_tet;

        for (var flush_idx = local_idx; flush_idx < total_flush_entries; flush_idx += WG_SIZE) {
            let bi = flush_idx / num_entries_per_tet;
            let entry = flush_idx % num_entries_per_tet;
            let tet_id_flush = shared_tets[bi].tet_id;

            // Map entry index to wg_grads offset
            let wg_offset = bi * MAX_GRAD_PER_TET + entry;
            let val = bitcast<f32>(atomicLoad(&wg_grads[wg_offset]));

            // Skip zero entries (common for missed ray-tet intersections)
            if (val == 0.0) { continue; }

            if (entry == 0u) {
                // Density gradient
                global_cas_add(tet_id_flush, val, 0u);
            } else if (entry < 13u) {
                // Vertex gradients: entry 1..12 -> vi = (entry-1)/3, ax = (entry-1)%3
                let vi = (entry - 1u) / 3u;
                let ax = (entry - 1u) % 3u;
                let vidx = indices[tet_id_flush * 4u + vi];
                let gi = vidx * 3u + ax;
                global_cas_add(gi, val, 1u);
            } else if (entry < 16u) {
                // Color gradient gradients: entry 13..15 -> c = entry - 13
                let c = entry - 13u;
                let gidx = tet_id_flush * 3u + c;
                global_cas_add(gidx, val, 2u);
            } else {
                // SH coefficient gradients: entry 16.. -> sh_offset = entry - 16
                let sh_offset = entry - 16u;
                let sh_global_idx = tet_id_flush * sh_stride + sh_offset;
                global_cas_add(sh_global_idx, val, 3u);
            }
        }

        workgroupBarrier();
        cursor = batch_begin;
    } // end batch loop

    // === Debug: write composited image from forward replay ===
    if (valid_pixel) {
        let T_final = exp(log_t);
        debug_image[pixel_idx * 4u] = color_accum.x;
        debug_image[pixel_idx * 4u + 1u] = color_accum.y;
        debug_image[pixel_idx * 4u + 2u] = color_accum.z;
        debug_image[pixel_idx * 4u + 3u] = 1.0 - T_final; // opacity
    }
}
