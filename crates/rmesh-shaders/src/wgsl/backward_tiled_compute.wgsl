// Warp-per-tile backward compute shader with subgroup shuffle gradient reduction.
//
// Thread model (32 threads per 4x4 tile):
//   Threads 0-15:  each owns one pixel, evaluates tet A
//   Threads 16-31: each evaluates the same pixel (lane-16) for tet B
//   Load 2 tets per iteration, evaluate simultaneously, shuffle to combine.
//
// Gradient reduction via half-warp subgroupShuffleXor (offsets 8,4,2,1).
// No shared memory needed — all communication through warp shuffles.
//
// One workgroup per tile. Dispatch: (num_tiles, 1, 1).

// enable subgroups;

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

// Atomic f32 add on global storage buffer (CAS pattern)
fn global_cas_add_density(buf_idx: u32, val: f32) {
    var ob = atomicLoad(&d_densities[buf_idx]);
    loop {
        let nv = bitcast<u32>(bitcast<f32>(ob) + val);
        let r = atomicCompareExchangeWeak(&d_densities[buf_idx], ob, nv);
        if (r.exchanged) { break; }
        ob = r.old_value;
    }
}

fn global_cas_add_vertex(buf_idx: u32, val: f32) {
    var ob = atomicLoad(&d_vertices[buf_idx]);
    loop {
        let nv = bitcast<u32>(bitcast<f32>(ob) + val);
        let r = atomicCompareExchangeWeak(&d_vertices[buf_idx], ob, nv);
        if (r.exchanged) { break; }
        ob = r.old_value;
    }
}

fn global_cas_add_color_grad(buf_idx: u32, val: f32) {
    var ob = atomicLoad(&d_color_grads[buf_idx]);
    loop {
        let nv = bitcast<u32>(bitcast<f32>(ob) + val);
        let r = atomicCompareExchangeWeak(&d_color_grads[buf_idx], ob, nv);
        if (r.exchanged) { break; }
        ob = r.old_value;
    }
}

fn global_cas_add_sh(buf_idx: u32, val: f32) {
    var ob = atomicLoad(&d_sh_coeffs[buf_idx]);
    loop {
        let nv = bitcast<u32>(bitcast<f32>(ob) + val);
        let r = atomicCompareExchangeWeak(&d_sh_coeffs[buf_idx], ob, nv);
        if (r.exchanged) { break; }
        ob = r.old_value;
    }
}

// Half-warp reduction: sum across 16 lanes using subgroupShuffleXor.
// XOR offsets 8,4,2,1 stay within each 16-lane half-warp.
fn half_warp_reduce(val: f32) -> f32 {
    var v = val;
    v += subgroupShuffleXor(v, 8u);
    v += subgroupShuffleXor(v, 4u);
    v += subgroupShuffleXor(v, 2u);
    v += subgroupShuffleXor(v, 1u);
    return v;
}

// Compute per-pixel gradients for one tet, given forward quantities.
// Returns gradient structure suitable for half-warp reduction.
struct PerPixelResult {
    // Ray-tet intersection
    t_min_val: f32,
    t_max_val: f32,
    min_face: u32,
    max_face: u32,
    intersection_valid: bool,
    // Forward quantities
    od: f32,
    c_premul: vec3<f32>,
    c_start_raw: vec3<f32>,
    c_end_raw: vec3<f32>,
    c_start: vec3<f32>,
    c_end: vec3<f32>,
    dist: f32,
    dc_dt: f32,
    base_color: vec3<f32>,
    // Geometry
    verts: array<vec3<f32>, 4>,
    vidx: array<u32, 4>,
    density_raw: f32,
    grad_vec: vec3<f32>,
    tet_id: u32,
};

fn intersect_and_eval(
    tet_id: u32,
    cam: vec3<f32>,
    ray_dir: vec3<f32>,
    valid_pixel: bool,
    has_tet: bool,
) -> PerPixelResult {
    var r: PerPixelResult;
    r.t_min_val = -3.402823e38;
    r.t_max_val = 3.402823e38;
    r.min_face = 0u;
    r.max_face = 0u;
    r.intersection_valid = false;
    r.tet_id = tet_id;

    // Always load tet data when has_tet is true, even if pixel is invalid.
    // The flush lane needs valid vidx/tet_id even when its pixel misses the tet.
    if (!has_tet) {
        return r;
    }

    r.vidx[0] = indices[tet_id * 4u];
    r.vidx[1] = indices[tet_id * 4u + 1u];
    r.vidx[2] = indices[tet_id * 4u + 2u];
    r.vidx[3] = indices[tet_id * 4u + 3u];

    r.verts[0] = load_f32x3_v(r.vidx[0]);
    r.verts[1] = load_f32x3_v(r.vidx[1]);
    r.verts[2] = load_f32x3_v(r.vidx[2]);
    r.verts[3] = load_f32x3_v(r.vidx[3]);

    r.density_raw = densities[tet_id];
    let colors_tet = vec3<f32>(colors_buf[tet_id * 3u], colors_buf[tet_id * 3u + 1u], colors_buf[tet_id * 3u + 2u]);
    r.grad_vec = vec3<f32>(color_grads_buf[tet_id * 3u], color_grads_buf[tet_id * 3u + 1u], color_grads_buf[tet_id * 3u + 2u]);

    if (!valid_pixel) {
        return r;
    }

    // Ray-tet intersection
    var valid = true;
    for (var fi = 0u; fi < 4u; fi++) {
        let f = FACES[fi];
        let va = r.verts[f[0]];
        let vb = r.verts[f[1]];
        let vc = r.verts[f[2]];
        let n = cross(vc - va, vb - va);
        let num = dot(n, va - cam);
        let den = dot(n, ray_dir);

        if (abs(den) < 1e-20) {
            if (num > 0.0) { valid = false; }
            continue;
        }

        let t = num / den;
        if (den > 0.0) {
            if (t > r.t_min_val) { r.t_min_val = t; r.min_face = fi; }
        } else {
            if (t < r.t_max_val) { r.t_max_val = t; r.max_face = fi; }
        }
    }

    if (!valid || r.t_min_val >= r.t_max_val) {
        return r;
    }
    r.intersection_valid = true;

    // Forward quantities
    let base_offset = dot(r.grad_vec, cam - r.verts[0]);
    r.base_color = colors_tet + vec3<f32>(base_offset);
    r.dc_dt = dot(r.grad_vec, ray_dir);

    r.c_start_raw = r.base_color + vec3<f32>(r.dc_dt * r.t_min_val);
    r.c_end_raw = r.base_color + vec3<f32>(r.dc_dt * r.t_max_val);
    r.c_start = max(r.c_start_raw, vec3<f32>(0.0));
    r.c_end = max(r.c_end_raw, vec3<f32>(0.0));

    r.dist = r.t_max_val - r.t_min_val;
    r.od = max(r.density_raw * r.dist, 1e-8);

    let alpha_t = exp(-r.od);
    let phi_val = phi(r.od);
    let w0 = phi_val - alpha_t;
    let w1 = 1.0 - phi_val;
    r.c_premul = r.c_end * w0 + r.c_start * w1;

    return r;
}

// Compute per-pixel gradients for one tet, reduce across half-warp, and flush to global.
// `r` contains the per-pixel intersection/forward data.
// `is_active` indicates whether this lane has valid gradient data (false lanes contribute 0).
// `color_after` and `log_t_before` are forward replay state BEFORE this tet is composited.
//
// IMPORTANT: All 16 lower-half lanes MUST call this function unconditionally so that
// half_warp_reduce (subgroupShuffleXor) has uniform control flow across the half-warp.
// Inactive lanes contribute 0 to the reduction.
fn compute_and_flush_grads(
    r: PerPixelResult,
    d_color: vec3<f32>,
    d_log_t_final: f32,
    color_final: vec3<f32>,
    color_after: vec3<f32>,
    cam: vec3<f32>,
    ray_dir: vec3<f32>,
    log_t_before: f32,
    lane: u32,
    sh_stride: u32,
    nc: u32,
    flush_lane: u32,
    is_active: bool,
) {
    var d_density_local: f32 = 0.0;
    var d_vert_local: array<vec3<f32>, 4>;
    var d_grad_local = vec3<f32>(0.0);
    var d_sh_result = vec3<f32>(0.0);
    var sh_dir = vec3<f32>(0.0, 0.0, 1.0);

    if (is_active) {
        let T_j = exp(log_t_before);

        // Closed-form d_log_t
        let d_log_t_j = d_log_t_final + dot(d_color, color_final - color_after);

        let d_c_premul = d_color * T_j;
        let d_od_state = -d_log_t_j;

        let dphi_val = dphi_dx(r.od);
        let dw0_dod = dphi_val + exp(-r.od);
        let dw1_dod = -dphi_val;

        let d_c_end_integral = d_c_premul * (phi(r.od) - exp(-r.od));
        let d_c_start_integral = d_c_premul * (1.0 - phi(r.od));
        let d_od_integral = dot(d_c_premul, r.c_end * dw0_dod + r.c_start * dw1_dod);

        let d_od = d_od_state + d_od_integral;

        d_density_local = d_od * r.dist;
        let d_dist = d_od * r.density_raw;
        var d_t_min = -d_dist;
        var d_t_max = d_dist;

        // Color chain through ReLU
        let d_c_start_raw = vec3<f32>(
            select(0.0, d_c_start_integral.x, r.c_start_raw.x > 0.0),
            select(0.0, d_c_start_integral.y, r.c_start_raw.y > 0.0),
            select(0.0, d_c_start_integral.z, r.c_start_raw.z > 0.0),
        );
        let d_c_end_raw = vec3<f32>(
            select(0.0, d_c_end_integral.x, r.c_end_raw.x > 0.0),
            select(0.0, d_c_end_integral.y, r.c_end_raw.y > 0.0),
            select(0.0, d_c_end_integral.z, r.c_end_raw.z > 0.0),
        );

        let d_base_color = d_c_start_raw + d_c_end_raw;
        let d_dc_dt_val = (d_c_start_raw.x + d_c_start_raw.y + d_c_start_raw.z) * r.t_min_val
            + (d_c_end_raw.x + d_c_end_raw.y + d_c_end_raw.z) * r.t_max_val;
        d_t_min += (d_c_start_raw.x + d_c_start_raw.y + d_c_start_raw.z) * r.dc_dt;
        d_t_max += (d_c_end_raw.x + d_c_end_raw.y + d_c_end_raw.z) * r.dc_dt;

        let d_base_offset_scalar = d_base_color.x + d_base_color.y + d_base_color.z;
        d_grad_local = (cam - r.verts[0]) * d_base_offset_scalar;
        let d_v0_from_base = -r.grad_vec * d_base_offset_scalar;

        d_grad_local += ray_dir * d_dc_dt_val;

        // Softplus backward
        let centroid = (r.verts[0] + r.verts[1] + r.verts[2] + r.verts[3]) * 0.25;
        sh_dir = normalize(centroid - cam);

        let sh_base = r.tet_id * sh_stride;
        var sh_result = vec3<f32>(0.0);
        sh_result.x = eval_sh(sh_dir, uniforms.sh_degree, sh_base);
        sh_result.y = eval_sh(sh_dir, uniforms.sh_degree, sh_base + nc);
        sh_result.z = eval_sh(sh_dir, uniforms.sh_degree, sh_base + 2u * nc);

        let offset_val = dot(r.grad_vec, r.verts[0] - centroid);
        let sp_input = sh_result + vec3<f32>(0.5 + offset_val);

        let d_sp_input = vec3<f32>(
            d_base_color.x * dsoftplus(sp_input.x),
            d_base_color.y * dsoftplus(sp_input.y),
            d_base_color.z * dsoftplus(sp_input.z),
        );

        d_sh_result = d_sp_input;
        let d_offset_scalar = d_sp_input.x + d_sp_input.y + d_sp_input.z;

        d_grad_local += (r.verts[0] - centroid) * d_offset_scalar;
        let d_v0_from_offset = r.grad_vec * d_offset_scalar * 0.75;
        let d_vother_from_offset = -r.grad_vec * d_offset_scalar * 0.25;

        // Intersection gradients
        {
            let f = FACES[r.min_face];
            let va = r.verts[f[0]]; let vb = r.verts[f[1]]; let vc = r.verts[f[2]];
            let n = cross(vc - va, vb - va);
            let den = dot(n, ray_dir);
            let hit = cam + ray_dir * r.t_min_val;
            let dt_dva = (cross(va - hit, vb - vc) + n) * (1.0 / den);
            let dt_dvb = cross(va - hit, vc - va) * (1.0 / den);
            let dt_dvc = cross(va - hit, va - vb) * (1.0 / den);
            d_vert_local[f[0]] += dt_dva * d_t_min;
            d_vert_local[f[1]] += dt_dvb * d_t_min;
            d_vert_local[f[2]] += dt_dvc * d_t_min;
        }
        {
            let f = FACES[r.max_face];
            let va = r.verts[f[0]]; let vb = r.verts[f[1]]; let vc = r.verts[f[2]];
            let n = cross(vc - va, vb - va);
            let den = dot(n, ray_dir);
            let hit = cam + ray_dir * r.t_max_val;
            let dt_dva = (cross(va - hit, vb - vc) + n) * (1.0 / den);
            let dt_dvb = cross(va - hit, vc - va) * (1.0 / den);
            let dt_dvc = cross(va - hit, va - vb) * (1.0 / den);
            d_vert_local[f[0]] += dt_dva * d_t_max;
            d_vert_local[f[1]] += dt_dvb * d_t_max;
            d_vert_local[f[2]] += dt_dvc * d_t_max;
        }

        d_vert_local[0] += d_v0_from_base + d_v0_from_offset;
        d_vert_local[1] += d_vother_from_offset;
        d_vert_local[2] += d_vother_from_offset;
        d_vert_local[3] += d_vother_from_offset;
    }

    // ===== SH gradient reduction + flush =====
    // All 16 lower-half lanes participate (inactive lanes contribute 0).
    {
        let x = sh_dir.x; let y = sh_dir.y; let z = sh_dir.z;
        var basis: array<f32, 16>;
        basis[0] = C0;
        var n_basis = 1u;
        if (uniforms.sh_degree >= 1u) {
            basis[1] = -C1 * y;
            basis[2] = C1 * z;
            basis[3] = -C1 * x;
            n_basis = 4u;
        }
        if (uniforms.sh_degree >= 2u) {
            let xx = x * x; let yy = y * y; let zz = z * z;
            basis[4] = C2_0 * x * y;
            basis[5] = C2_1 * y * z;
            basis[6] = C2_2 * (2.0 * zz - xx - yy);
            basis[7] = C2_3 * x * z;
            basis[8] = C2_4 * (xx - yy);
            n_basis = 9u;
            if (uniforms.sh_degree >= 3u) {
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
        for (var c = 0u; c < 3u; c++) {
            for (var k = 0u; k < n_basis; k++) {
                let sh_val = d_channels[c] * basis[k];
                let reduced = half_warp_reduce(sh_val);
                if (lane == flush_lane) {
                    let sh_global_idx = r.tet_id * sh_stride + c * nc + k;
                    global_cas_add_sh(sh_global_idx, reduced);
                }
            }
        }
    }

    // ===== Reduce and flush scalar/vector gradients =====
    // All 16 lower-half lanes participate (inactive lanes contribute 0).
    let reduced_d_density = half_warp_reduce(d_density_local);

    var reduced_d_vert: array<vec3<f32>, 4>;
    for (var vi = 0u; vi < 4u; vi++) {
        reduced_d_vert[vi].x = half_warp_reduce(d_vert_local[vi].x);
        reduced_d_vert[vi].y = half_warp_reduce(d_vert_local[vi].y);
        reduced_d_vert[vi].z = half_warp_reduce(d_vert_local[vi].z);
    }
    let reduced_d_grad = vec3<f32>(
        half_warp_reduce(d_grad_local.x),
        half_warp_reduce(d_grad_local.y),
        half_warp_reduce(d_grad_local.z),
    );

    // Flush reduced gradients to global memory.
    // No is_active check — the reduced values correctly sum over all pixels
    // (inactive lanes contributed 0). The flush lane always has valid
    // r.tet_id and r.vidx since intersect_and_eval loads them when has_tet=true.
    if (lane == flush_lane) {
        global_cas_add_density(r.tet_id, reduced_d_density);

        for (var vi = 0u; vi < 4u; vi++) {
            let gi = r.vidx[vi] * 3u;
            global_cas_add_vertex(gi, reduced_d_vert[vi].x);
            global_cas_add_vertex(gi + 1u, reduced_d_vert[vi].y);
            global_cas_add_vertex(gi + 2u, reduced_d_vert[vi].z);
        }

        global_cas_add_color_grad(r.tet_id * 3u, reduced_d_grad.x);
        global_cas_add_color_grad(r.tet_id * 3u + 1u, reduced_d_grad.y);
        global_cas_add_color_grad(r.tet_id * 3u + 2u, reduced_d_grad.z);
    }
}

@compute @workgroup_size(32)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let tile_id = wg_id.x;
    if (tile_id >= tile_uniforms.num_tiles) {
        return;
    }

    let tile_x = tile_id % tile_uniforms.tiles_x;
    let tile_y = tile_id / tile_uniforms.tiles_x;

    let range_start = tile_ranges[tile_id * 2u];
    let range_end = tile_ranges[tile_id * 2u + 1u];

    let pixel_lane = lane % 16u;
    let is_second_half = lane >= 16u;

    let px = tile_x * tile_uniforms.tile_size + (pixel_lane % tile_uniforms.tile_size);
    let py = tile_y * tile_uniforms.tile_size + (pixel_lane / tile_uniforms.tile_size);
    let w = tile_uniforms.screen_width;
    let h = tile_uniforms.screen_height;
    let valid_pixel = (px < w) && (py < h);
    let pixel_idx = py * w + px;

    // Load per-pixel constants
    var d_color = vec3<f32>(0.0);
    var d_log_t_final: f32 = 0.0;
    var color_final = vec3<f32>(0.0);

    if (valid_pixel) {
        d_color = vec3<f32>(
            dl_d_image[pixel_idx * 4u],
            dl_d_image[pixel_idx * 4u + 1u],
            dl_d_image[pixel_idx * 4u + 2u],
        );
        let alpha_final_val = rendered_image[pixel_idx * 4u + 3u];
        d_log_t_final = -dl_d_image[pixel_idx * 4u + 3u] * (1.0 - alpha_final_val);
        color_final = vec3<f32>(
            rendered_image[pixel_idx * 4u],
            rendered_image[pixel_idx * 4u + 1u],
            rendered_image[pixel_idx * 4u + 2u],
        );
    }

    // Compute ray
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

    // Forward state (threads 0-15 only)
    var color_accum = vec3<f32>(0.0);
    var log_t: f32 = 0.0;

    // Process tets: nearest first (range_end-1 down to range_start)
    // Process 1 tet at a time to keep the code manageable and correct.
    // The dual-tet optimization can be added later once single-tet path is verified.
    var cursor = range_end;

    while (cursor > range_start) {
        let tet_id = tile_sort_values[cursor - 1u];

        // Broadcast tet ID to all threads
        let my_tet_id = subgroupShuffle(tet_id, 0u);

        // Every thread evaluates the same tet for their own pixel
        let r = intersect_and_eval(my_tet_id, cam, ray_dir, valid_pixel, true);

        // Threads 0-15: forward replay + gradient computation
        // ALL lower-half lanes must call compute_and_flush_grads so that
        // half_warp_reduce has uniform control flow across the half-warp.
        if (!is_second_half) {
            let is_active = valid_pixel && r.intersection_valid;
            let T_j = exp(log_t);
            let color_after = color_accum + r.c_premul * T_j;

            // Compute and reduce+flush gradients (inactive lanes contribute 0)
            compute_and_flush_grads(
                r, d_color, d_log_t_final, color_final, color_after,
                cam, ray_dir, log_t, lane, sh_stride, nc, 0u, is_active
            );

            // Update forward state (only for is_active lanes)
            if (is_active) {
                color_accum = color_after;
                log_t -= r.od;
            }
        }

        cursor -= 1u;
    }

    // Debug: write composited image from forward replay
    if (!is_second_half && valid_pixel) {
        let T_final = exp(log_t);
        debug_image[pixel_idx * 4u] = color_accum.x;
        debug_image[pixel_idx * 4u + 1u] = color_accum.y;
        debug_image[pixel_idx * 4u + 2u] = color_accum.z;
        debug_image[pixel_idx * 4u + 3u] = 1.0 - T_final;
    }
}
