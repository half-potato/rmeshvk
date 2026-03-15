// Backward compute shader: gradient computation for all trainable parameters.
//
// Per-pixel backward pass: iterates all sorted tets (furthest→nearest),
// forward-replays intersection + integral, undoes pixel state, computes
// gradients via chain rule, and atomically scatters to global gradient buffers.

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
    step: u32,
    _pad1: vec4<u32>,
    _pad2: vec4<u32>,
};

// Group 0: read-only scene data + sorted indices
@group(0) @binding(0) var<storage, read> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> dl_d_image: array<f32>;
@group(0) @binding(2) var<storage, read> rendered_image: array<f32>;
@group(0) @binding(3) var<storage, read> vertices: array<f32>;
@group(0) @binding(4) var<storage, read> indices: array<u32>;
@group(0) @binding(5) var<storage, read> densities: array<f32>;
@group(0) @binding(6) var<storage, read> color_grads_buf: array<f32>;
@group(0) @binding(7) var<storage, read> circumdata: array<f32>;
@group(0) @binding(8) var<storage, read> colors_buf: array<f32>;
@group(0) @binding(9) var<storage, read> sorted_indices: array<u32>;

// Group 1: read-write gradient outputs
@group(1) @binding(0) var<storage, read_write> d_vertices: array<atomic<f32>>;
@group(1) @binding(1) var<storage, read_write> d_densities: array<atomic<f32>>;
@group(1) @binding(2) var<storage, read_write> d_color_grads: array<atomic<f32>>;

const TINY_VAL: f32 = 1e-20;

// Face (a, b, c, opposite_vertex) — opposite used to flip normal inward
const FACES: array<vec4<u32>, 4> = array<vec4<u32>, 4>(
    vec4<u32>(0u, 2u, 1u, 3u),
    vec4<u32>(1u, 2u, 3u, 0u),
    vec4<u32>(0u, 3u, 2u, 1u),
    vec4<u32>(3u, 0u, 1u, 2u),
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

fn load_f32x3_c(idx: u32) -> vec3<f32> {
    return vec3<f32>(colors_buf[idx * 3u], colors_buf[idx * 3u + 1u], colors_buf[idx * 3u + 2u]);
}

fn load_f32x3_g(idx: u32) -> vec3<f32> {
    return vec3<f32>(color_grads_buf[idx * 3u], color_grads_buf[idx * 3u + 1u], color_grads_buf[idx * 3u + 2u]);
}

// Atomic float add via compare-and-swap — inlined at call sites
// (naga does not allow ptr<storage, atomic<u32>, read_write> as function arguments)

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let px = global_id.x;
    let py = global_id.y;
    let w = u32(uniforms.screen_width);
    let h = u32(uniforms.screen_height);

    if (px >= w || py >= h) { return; }

    let pixel_idx = py * w + px;

    // 1. Load final pixel state
    let final_r = rendered_image[pixel_idx * 4u];
    let final_g = rendered_image[pixel_idx * 4u + 1u];
    let final_b = rendered_image[pixel_idx * 4u + 2u];
    let final_a = rendered_image[pixel_idx * 4u + 3u];

    var color = vec3<f32>(final_r, final_g, final_b);
    let transmittance = max(1.0 - final_a, TINY_VAL);
    var log_t = log(transmittance);

    // 2. Load loss gradient
    var d_color = vec3<f32>(
        dl_d_image[pixel_idx * 4u],
        dl_d_image[pixel_idx * 4u + 1u],
        dl_d_image[pixel_idx * 4u + 2u],
    );
    var d_log_t = -dl_d_image[pixel_idx * 4u + 3u] * (1.0 - final_a);

    // 3. Compute ray from pixel coordinates via inverse VP
    let ndc_x = (2.0 * (f32(px) + 0.5) / uniforms.screen_width) - 1.0;
    // wgpu framebuffer y=0 is top, but NDC y=+1 is top → flip
    let ndc_y = 1.0 - (2.0 * (f32(py) + 0.5) / uniforms.screen_height);

    let inv_vp = mat4x4<f32>(uniforms.inv_vp_col0, uniforms.inv_vp_col1, uniforms.inv_vp_col2, uniforms.inv_vp_col3);
    let near_clip = inv_vp * vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    let far_clip = inv_vp * vec4<f32>(ndc_x, ndc_y, 1.0, 1.0);
    let near_world = near_clip.xyz / near_clip.w;
    let far_world = far_clip.xyz / far_clip.w;

    let cam = uniforms.cam_pos_pad.xyz;
    let ray_dir = normalize(far_world - near_world);

    let tet_count = uniforms.tet_count;

    // 4. Iterate tets: furthest → nearest
    for (var tet_offset = 0u; tet_offset < tet_count; tet_offset++) {
        let tet_id = sorted_indices[tet_offset];

        // 4a. Load tet geometry
        let ti0 = indices[tet_id * 4u];
        let ti1 = indices[tet_id * 4u + 1u];
        let ti2 = indices[tet_id * 4u + 2u];
        let ti3 = indices[tet_id * 4u + 3u];

        var verts: array<vec3<f32>, 4>;
        verts[0] = load_f32x3_v(ti0);
        verts[1] = load_f32x3_v(ti1);
        verts[2] = load_f32x3_v(ti2);
        verts[3] = load_f32x3_v(ti3);

        let density_raw = densities[tet_id];
        let colors_tet = load_f32x3_c(tet_id);
        let grad = load_f32x3_g(tet_id);

        // 4b. Forward replay: ray-tet intersection
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

        if (!valid || t_min_val >= t_max_val) { continue; }

        // 4c. Forward replay: colors & integral
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

        // 4d. Undo pixel state
        let prev_log_t = log_t + od;
        let t_prev = exp(prev_log_t);
        let prev_color = color - c_premul * t_prev;

        // 4e. Backward: pixel state
        let d_c_premul = d_color * t_prev;
        let d_od_state = -d_log_t;
        let d_old_color = d_color;
        let d_old_log_t = d_log_t + dot(d_color, c_premul) * t_prev;

        // 4f. Backward: compute_integral
        let dphi_val = dphi_dx(od);
        let dw0_dod = dphi_val + alpha_t;
        let dw1_dod = -dphi_val;

        let d_c_end_integral = d_c_premul * w0;
        let d_c_start_integral = d_c_premul * w1;
        let d_od_integral = dot(d_c_premul, c_end * dw0_dod + c_start * dw1_dod);

        let d_od = d_od_state + d_od_integral;

        // 4g. Backward: od and dist
        let d_density_local = d_od * dist;
        let d_dist = d_od * density_raw;
        var d_t_min = -d_dist;
        var d_t_max = d_dist;

        // 4h. Backward: color chain (max clamp → base_color → dc_dt)
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

        // 4h-cont. base_color = colors + grad.(cam - v0)
        let d_base_offset_scalar = d_base_color.x + d_base_color.y + d_base_color.z;
        var d_grad = (cam - verts[0]) * d_base_offset_scalar;
        let d_v0_from_base = -grad * d_base_offset_scalar;

        // dc_dt = grad . ray_dir
        d_grad += ray_dir * d_dc_dt;

        // 4i. Intersection gradients (dt/d(vertices))
        // NOTE: explicit zero-init required — WGSL var inside a loop may not
        // be re-zero-initialized on each iteration in some runtimes (naga/wgpu).
        var d_vert_i: array<vec3<f32>, 4>;
        d_vert_i[0] = vec3<f32>(0.0);
        d_vert_i[1] = vec3<f32>(0.0);
        d_vert_i[2] = vec3<f32>(0.0);
        d_vert_i[3] = vec3<f32>(0.0);

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

        // 4j. Combine vertex gradients and scatter
        d_vert_i[0] += d_v0_from_base;

        let vert_indices = array<u32, 4>(ti0, ti1, ti2, ti3);
        for (var vi = 0u; vi < 4u; vi++) {
            let vidx = vert_indices[vi];
            let dv = d_vert_i[vi];
            for (var ax = 0u; ax < 3u; ax++) {
                let comp = select(select(dv.z, dv.y, ax == 1u), dv.x, ax == 0u);
                let gi = vidx * 3u + ax;
                atomicAdd(&d_vertices[gi], comp);
            }
        }

        atomicAdd(&d_densities[tet_id], d_density_local);

        let dg = d_grad;
        for (var gi2 = 0u; gi2 < 3u; gi2++) {
            let gc = select(select(dg.z, dg.y, gi2 == 1u), dg.x, gi2 == 0u);
            let gidx = tet_id * 3u + gi2;
            atomicAdd(&d_color_grads[gidx], gc);
        }

        // 4n. Update pixel state for next iteration
        color = prev_color;
        log_t = prev_log_t;
        d_color = d_old_color;
        d_log_t = d_old_log_t;
    }
}
