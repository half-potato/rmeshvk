// Compute-based forward tiled renderer: 1 warp (32 threads) per 16×16 tile.
//
// Uses workgroup shared memory for per-pixel state. For each tet assigned to
// this tile, a scanline fill determines which of the 256 pixels are covered,
// and only those pixels are processed (ray-tet intersection + compositing).
//
// Thread model:
//   32 threads per workgroup (1 warp).
//   Threads 0-15 compute scanline row ranges.
//   Thread 0 builds prefix sum.
//   All 32 threads process covered pixels via linear index distribution.
//
// One workgroup per tile. Dispatch: (num_tiles, 1, 1).

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

// Face (a, b, c, opposite_vertex) — opposite used to flip normal inward
const FACES: array<vec4<u32>, 4> = array<vec4<u32>, 4>(
    vec4<u32>(0u, 2u, 1u, 3u),
    vec4<u32>(1u, 2u, 3u, 0u),
    vec4<u32>(0u, 3u, 2u, 1u),
    vec4<u32>(3u, 0u, 1u, 2u),
);

// Workgroup shared memory
var<workgroup> sm_color: array<vec4<f32>, 256>;  // .xyz = color_accum, .w = log_t
var<workgroup> sm_xl: array<i32, 16>;             // scanline left x per row
var<workgroup> sm_xr: array<i32, 16>;             // scanline right x per row
var<workgroup> sm_prefix: array<u32, 17>;          // prefix sum of row widths

fn phi(x: f32) -> f32 {
    if (abs(x) < 1e-6) { return 1.0 - x * 0.5; }
    return (1.0 - exp(-x)) / x;
}

fn softplus(x: f32) -> f32 {
    if (x > 8.0) { return x; }
    return 0.1 * log(1.0 + exp(10.0 * x));
}

fn load_f32x3_v(idx: u32) -> vec3<f32> {
    return vec3<f32>(vertices[idx * 3u], vertices[idx * 3u + 1u], vertices[idx * 3u + 2u]);
}

// Scanline edge test: if edge (proj[ei], proj[ej]) crosses y = yc, compute x intersection
// and update xl/xr. Inlined as a macro-like pattern to avoid function overhead.

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
    let tile_ox = f32(tile_x * 16u);
    let tile_oy = f32(tile_y * 16u);

    let range_start = tile_ranges[tile_id * 2u];
    let range_end = tile_ranges[tile_id * 2u + 1u];

    let cam = uniforms.cam_pos_pad.xyz;
    let vp = mat4x4<f32>(uniforms.vp_col0, uniforms.vp_col1, uniforms.vp_col2, uniforms.vp_col3);
    let inv_vp = mat4x4<f32>(uniforms.inv_vp_col0, uniforms.inv_vp_col1, uniforms.inv_vp_col2, uniforms.inv_vp_col3);

    // Initialize per-pixel state (32 threads × 8 pixels each)
    for (var i = lane; i < 256u; i += 32u) {
        sm_color[i] = vec4<f32>(0.0);
    }
    workgroupBarrier();

    // Process tets front-to-back (nearest first)
    var cursor = range_end;
    while (cursor > range_start) {
        cursor -= 1u;
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

        // Project to clip space
        let c0 = vp * vec4<f32>(v0, 1.0);
        let c1 = vp * vec4<f32>(v1, 1.0);
        let c2 = vp * vec4<f32>(v2, 1.0);
        let c3 = vp * vec4<f32>(v3, 1.0);

        let any_behind = (c0.w <= 0.0) || (c1.w <= 0.0) || (c2.w <= 0.0) || (c3.w <= 0.0);

        // Project to tile-local pixel coords
        var proj: array<vec2<f32>, 4>;
        if (!any_behind) {
            proj[0] = vec2<f32>((c0.x / c0.w + 1.0) * 0.5 * W - tile_ox, (1.0 - c0.y / c0.w) * 0.5 * H - tile_oy);
            proj[1] = vec2<f32>((c1.x / c1.w + 1.0) * 0.5 * W - tile_ox, (1.0 - c1.y / c1.w) * 0.5 * H - tile_oy);
            proj[2] = vec2<f32>((c2.x / c2.w + 1.0) * 0.5 * W - tile_ox, (1.0 - c2.y / c2.w) * 0.5 * H - tile_oy);
            proj[3] = vec2<f32>((c3.x / c3.w + 1.0) * 0.5 * W - tile_ox, (1.0 - c3.y / c3.w) * 0.5 * H - tile_oy);
        }

        // Scanline fill (Variant A): threads 0-15 each compute one row
        if (lane < 16u) {
            var xl_f: f32 = 1e10;
            var xr_f: f32 = -1e10;

            if (!any_behind) {
                let yc = f32(lane) + 0.5;

                // Edge (0, 1)
                {
                    let yi = proj[0].y; let yj = proj[1].y;
                    if ((yi <= yc && yj > yc) || (yj <= yc && yi > yc)) {
                        let t = (yc - yi) / (yj - yi);
                        let x = proj[0].x + t * (proj[1].x - proj[0].x);
                        xl_f = min(xl_f, x); xr_f = max(xr_f, x);
                    }
                }
                // Edge (0, 2)
                {
                    let yi = proj[0].y; let yj = proj[2].y;
                    if ((yi <= yc && yj > yc) || (yj <= yc && yi > yc)) {
                        let t = (yc - yi) / (yj - yi);
                        let x = proj[0].x + t * (proj[2].x - proj[0].x);
                        xl_f = min(xl_f, x); xr_f = max(xr_f, x);
                    }
                }
                // Edge (0, 3)
                {
                    let yi = proj[0].y; let yj = proj[3].y;
                    if ((yi <= yc && yj > yc) || (yj <= yc && yi > yc)) {
                        let t = (yc - yi) / (yj - yi);
                        let x = proj[0].x + t * (proj[3].x - proj[0].x);
                        xl_f = min(xl_f, x); xr_f = max(xr_f, x);
                    }
                }
                // Edge (1, 2)
                {
                    let yi = proj[1].y; let yj = proj[2].y;
                    if ((yi <= yc && yj > yc) || (yj <= yc && yi > yc)) {
                        let t = (yc - yi) / (yj - yi);
                        let x = proj[1].x + t * (proj[2].x - proj[1].x);
                        xl_f = min(xl_f, x); xr_f = max(xr_f, x);
                    }
                }
                // Edge (1, 3)
                {
                    let yi = proj[1].y; let yj = proj[3].y;
                    if ((yi <= yc && yj > yc) || (yj <= yc && yi > yc)) {
                        let t = (yc - yi) / (yj - yi);
                        let x = proj[1].x + t * (proj[3].x - proj[1].x);
                        xl_f = min(xl_f, x); xr_f = max(xr_f, x);
                    }
                }
                // Edge (2, 3)
                {
                    let yi = proj[2].y; let yj = proj[3].y;
                    if ((yi <= yc && yj > yc) || (yj <= yc && yi > yc)) {
                        let t = (yc - yi) / (yj - yi);
                        let x = proj[2].x + t * (proj[3].x - proj[2].x);
                        xl_f = min(xl_f, x); xr_f = max(xr_f, x);
                    }
                }

                // Include vertices within this pixel row. Without this,
                // the top/bottom apex of the projected tet drops an entire
                // row because the half-open edge test misses all edges
                // emanating from a vertex exactly at yc.
                for (var v = 0u; v < 4u; v++) {
                    let vy = proj[v].y;
                    if (vy >= yc - 0.5 && vy < yc + 0.5) {
                        xl_f = min(xl_f, proj[v].x);
                        xr_f = max(xr_f, proj[v].x);
                    }
                }
            }

            // Convert float range to integer pixel range within [0, 15]
            if (xl_f <= xr_f) {
                let xl_i = max(i32(ceil(xl_f - 0.5)), 0);
                let xr_i = min(i32(floor(xr_f - 0.5)), 15);
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

        // Prefix sum of row widths (thread 0)
        if (lane == 0u) {
            sm_prefix[0] = 0u;
            for (var r = 0u; r < 16u; r++) {
                let row_w = u32(max(sm_xr[r] - sm_xl[r] + 1, 0));
                sm_prefix[r + 1u] = sm_prefix[r] + row_w;
            }
        }
        workgroupBarrier();

        let total = sm_prefix[16u];
        if (total == 0u) {
            // No pixels covered — skip to next tet (barrier already done)
            continue;
        }

        // Load tet attributes
        let density_raw = densities[tet_id];
        let colors_tet = vec3<f32>(colors_buf[tet_id * 3u], colors_buf[tet_id * 3u + 1u], colors_buf[tet_id * 3u + 2u]);
        let grad_vec = vec3<f32>(color_grads_buf[tet_id * 3u], color_grads_buf[tet_id * 3u + 1u], color_grads_buf[tet_id * 3u + 2u]);
        var verts: array<vec3<f32>, 4>;
        verts[0] = v0; verts[1] = v1; verts[2] = v2; verts[3] = v3;

        // Process covered pixels (each thread handles indices: lane, lane+32, ...)
        for (var idx = lane; idx < total; idx += 32u) {
            // Map linear index to (row, col) via prefix sum
            var row = 0u;
            for (var r = 0u; r < 16u; r++) {
                if (idx < sm_prefix[r + 1u]) {
                    row = r;
                    break;
                }
            }
            let col = u32(sm_xl[row]) + (idx - sm_prefix[row]);
            let pixel_local = row * 16u + col;

            let px = tile_x * 16u + col;
            let py = tile_y * 16u + row;

            if (px >= w || py >= h) {
                continue;
            }

            // Compute ray from pixel center via inverse VP
            let ndc_x = (2.0 * (f32(px) + 0.5) / W) - 1.0;
            let ndc_y = 1.0 - (2.0 * (f32(py) + 0.5) / H);
            let near_clip = inv_vp * vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
            let far_clip = inv_vp * vec4<f32>(ndc_x, ndc_y, 1.0, 1.0);
            let near_world = near_clip.xyz / near_clip.w;
            let far_world = far_clip.xyz / far_clip.w;
            let ray_dir = normalize(far_world - near_world);

            // Ray-tet intersection
            var t_min_val = -3.402823e38;
            var t_max_val = 3.402823e38;
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
                    if (t > t_min_val) { t_min_val = t; }
                } else {
                    if (t < t_max_val) { t_max_val = t; }
                }
            }

            if (valid && t_min_val < t_max_val) {
                // Volume integral
                let base_offset = dot(grad_vec, cam - verts[0]);
                let base_color = colors_tet + vec3<f32>(base_offset);
                let dc_dt = dot(grad_vec, ray_dir);

                let c_start = max(base_color + vec3<f32>(dc_dt * t_min_val), vec3<f32>(0.0));
                let c_end = max(base_color + vec3<f32>(dc_dt * t_max_val), vec3<f32>(0.0));

                let dist = t_max_val - t_min_val;
                let od = max(density_raw * dist, 1e-8);

                let alpha_t = exp(-od);
                let phi_val = phi(od);
                let w0 = phi_val - alpha_t;
                let w1 = 1.0 - phi_val;
                let c_premul = c_end * w0 + c_start * w1;

                // Composite into shared memory
                let state = sm_color[pixel_local];
                let T_j = exp(state.w);
                sm_color[pixel_local] = vec4<f32>(
                    state.xyz + c_premul * T_j,
                    state.w - od,
                );
            }
        }
        workgroupBarrier();
    }

    // Write output (32 threads × 8 pixels each)
    for (var i = lane; i < 256u; i += 32u) {
        let row = i / 16u;
        let col = i % 16u;
        let px = tile_x * 16u + col;
        let py = tile_y * 16u + row;
        if (px < w && py < h) {
            let pixel_idx = py * w + px;
            let state = sm_color[i];
            let T_final = exp(state.w);
            rendered_image[pixel_idx * 4u] = state.x;
            rendered_image[pixel_idx * 4u + 1u] = state.y;
            rendered_image[pixel_idx * 4u + 2u] = state.z;
            rendered_image[pixel_idx * 4u + 3u] = 1.0 - T_final;
        }
    }
}
