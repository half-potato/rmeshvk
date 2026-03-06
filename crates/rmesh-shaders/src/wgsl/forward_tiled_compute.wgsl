// Compute-based forward tiled renderer with warp-per-tile (32 threads per 4x4 tile).
//
// Thread model:
//   Threads 0-15:  each owns one pixel in the 4x4 tile, evaluates tet A
//   Threads 16-31: each evaluates the same pixel (lane - 16) for tet B
//   Load 2 tets per iteration, evaluate simultaneously, shuffle to combine.
//   Forward state (color_accum, log_t) lives in threads 0-15 only.
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

@group(0) @binding(0) var<storage, read> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> vertices: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> colors_buf: array<f32>;
@group(0) @binding(4) var<storage, read> densities: array<f32>;
@group(0) @binding(5) var<storage, read> color_grads_buf: array<f32>;
@group(0) @binding(6) var<storage, read> tile_sort_values: array<u32>;
@group(0) @binding(7) var<storage, read> tile_ranges: array<u32>;
@group(0) @binding(8) var<storage, read> tile_uniforms: TileUniforms;
@group(0) @binding(9) var<storage, read_write> rendered_image: array<f32>;

// Face winding (inward normals)
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

fn load_f32x3_v(idx: u32) -> vec3<f32> {
    return vec3<f32>(vertices[idx * 3u], vertices[idx * 3u + 1u], vertices[idx * 3u + 2u]);
}

// Evaluate a single tet for a given ray. Returns (c_premul, od) or od < 0 if miss.
fn eval_tet(
    tet_id: u32,
    cam: vec3<f32>,
    ray_dir: vec3<f32>,
) -> vec4<f32> {
    // Load tet geometry
    let vi0 = indices[tet_id * 4u];
    let vi1 = indices[tet_id * 4u + 1u];
    let vi2 = indices[tet_id * 4u + 2u];
    let vi3 = indices[tet_id * 4u + 3u];

    let v0 = load_f32x3_v(vi0);
    let v1 = load_f32x3_v(vi1);
    let v2 = load_f32x3_v(vi2);
    let v3 = load_f32x3_v(vi3);

    var verts = array<vec3<f32>, 4>(v0, v1, v2, v3);

    // Ray-tet intersection
    var t_min_val = -3.402823e38;
    var t_max_val = 3.402823e38;
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
            if (t > t_min_val) { t_min_val = t; }
        } else {
            if (t < t_max_val) { t_max_val = t; }
        }
    }

    if (!valid || t_min_val >= t_max_val) {
        return vec4<f32>(0.0, 0.0, 0.0, -1.0); // miss
    }

    // Volume integral
    let density_raw = densities[tet_id];
    let colors_tet = vec3<f32>(colors_buf[tet_id * 3u], colors_buf[tet_id * 3u + 1u], colors_buf[tet_id * 3u + 2u]);
    let grad = vec3<f32>(color_grads_buf[tet_id * 3u], color_grads_buf[tet_id * 3u + 1u], color_grads_buf[tet_id * 3u + 2u]);

    let base_offset = dot(grad, cam - verts[0]);
    let base_color = colors_tet + vec3<f32>(base_offset);
    let dc_dt = dot(grad, ray_dir);

    let c_start = max(base_color + vec3<f32>(dc_dt * t_min_val), vec3<f32>(0.0));
    let c_end = max(base_color + vec3<f32>(dc_dt * t_max_val), vec3<f32>(0.0));

    let dist = t_max_val - t_min_val;
    let od = max(density_raw * dist, 1e-8);

    let alpha_t = exp(-od);
    let phi_val = phi(od);
    let w0 = phi_val - alpha_t;
    let w1 = 1.0 - phi_val;
    let c_premul = c_end * w0 + c_start * w1;

    return vec4<f32>(c_premul, od);
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

    // Compute tile coordinates
    let tile_x = tile_id % tile_uniforms.tiles_x;
    let tile_y = tile_id / tile_uniforms.tiles_x;

    // Load tile range
    let range_start = tile_ranges[tile_id * 2u];
    let range_end = tile_ranges[tile_id * 2u + 1u];

    // Pixel index within tile: threads 0-15 each own a pixel, threads 16-31 mirror
    let pixel_lane = lane % 16u;
    let is_second_half = lane >= 16u;

    let px = tile_x * tile_uniforms.tile_size + (pixel_lane % tile_uniforms.tile_size);
    let py = tile_y * tile_uniforms.tile_size + (pixel_lane / tile_uniforms.tile_size);
    let w = tile_uniforms.screen_width;
    let h = tile_uniforms.screen_height;
    let valid_pixel = (px < w) && (py < h);

    // Compute ray from pixel coordinates via inverse VP
    let ndc_x = (2.0 * (f32(px) + 0.5) / f32(w)) - 1.0;
    let ndc_y = 1.0 - (2.0 * (f32(py) + 0.5) / f32(h));

    let inv_vp = mat4x4<f32>(uniforms.inv_vp_col0, uniforms.inv_vp_col1, uniforms.inv_vp_col2, uniforms.inv_vp_col3);
    let near_clip = inv_vp * vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    let far_clip = inv_vp * vec4<f32>(ndc_x, ndc_y, 1.0, 1.0);
    let near_world = near_clip.xyz / near_clip.w;
    let far_world = far_clip.xyz / far_clip.w;

    let cam = uniforms.cam_pos_pad.xyz;
    let ray_dir = normalize(far_world - near_world);

    // Forward state (only meaningful for threads 0-15)
    var color_accum = vec3<f32>(0.0);
    var log_t: f32 = 0.0;

    // Process tets: nearest first (range_end-1 down to range_start), 2 at a time
    var cursor = range_end;

    while (cursor > range_start) {
        let remaining = cursor - range_start;

        // Load up to 2 tets
        let has_tet_a = remaining >= 1u;
        let has_tet_b = remaining >= 2u;

        // Thread 0 loads tet A, thread 16 loads tet B
        var tet_id_a = 0u;
        var tet_id_b = 0u;
        if (has_tet_a) {
            tet_id_a = tile_sort_values[cursor - 1u];
        }
        if (has_tet_b) {
            tet_id_b = tile_sort_values[cursor - 2u];
        }

        // Broadcast tet IDs within each half-warp
        let my_tet_id = select(
            subgroupShuffle(tet_id_a, 0u),  // first half uses tet A
            subgroupShuffle(tet_id_b, 16u), // second half uses tet B
            is_second_half
        );

        // Each thread evaluates their assigned tet
        var my_result = vec4<f32>(0.0, 0.0, 0.0, -1.0);
        let has_my_tet = select(has_tet_a, has_tet_b, is_second_half);
        if (valid_pixel && has_my_tet) {
            my_result = eval_tet(my_tet_id, cam, ray_dir);
        }

        // Threads 0-15: read tet B results from threads 16-31
        let result_b = vec4<f32>(
            subgroupShuffle(my_result.x, pixel_lane + 16u),
            subgroupShuffle(my_result.y, pixel_lane + 16u),
            subgroupShuffle(my_result.z, pixel_lane + 16u),
            subgroupShuffle(my_result.w, pixel_lane + 16u),
        );

        // Threads 0-15 composite: tet A first, then tet B
        if (!is_second_half && valid_pixel) {
            // Composite tet A
            if (has_tet_a && my_result.w > 0.0) {
                let T_j = exp(log_t);
                color_accum += my_result.xyz * T_j;
                log_t -= my_result.w;
            }
            // Composite tet B
            if (has_tet_b && result_b.w > 0.0) {
                let T_j = exp(log_t);
                color_accum += result_b.xyz * T_j;
                log_t -= result_b.w;
            }
        }

        cursor -= select(1u, 2u, has_tet_b);
    }

    // Write output (threads 0-15 only)
    if (!is_second_half && valid_pixel) {
        let pixel_idx = py * w + px;
        let T_final = exp(log_t);
        rendered_image[pixel_idx * 4u] = color_accum.x;
        rendered_image[pixel_idx * 4u + 1u] = color_accum.y;
        rendered_image[pixel_idx * 4u + 2u] = color_accum.z;
        rendered_image[pixel_idx * 4u + 3u] = 1.0 - T_final;
    }
}
