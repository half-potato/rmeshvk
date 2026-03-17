// Rasterize compute shader: 1 warp (32 threads) per tile.
//
// Uses workgroup shared memory for per-pixel state. For each tet assigned to
// this tile, a scanline fill determines which of the pixels are covered,
// and only those pixels are processed (ray-tet intersection + compositing).
//
// Thread model:
//   32 threads per workgroup (1 warp).
//   Threads 0..TS compute scanline row ranges.
//   Thread 0 builds prefix sum.
//   All 32 threads process covered pixels via linear index distribution.
//
// One workgroup per tile. Dispatch: (num_tiles, 1, 1).

// naga_oil selective imports (plain `#import module` doesn't work in 0.21).
// Struct definitions are inlined — naga_oil 0.21 mangles struct member names.
#import rmesh::math::{MAX_VAL, safe_clip_v3f, safe_exp_f32, phi}
#import rmesh::intersect::FACES

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

const AUX_DIM: u32 = /*AUX_DIM*/0u;
const AUX_STRIDE: u32 = 8u + AUX_DIM;
const SM_AUX_SIZE: u32 = /*SM_AUX_SIZE*/1u;

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

@group(1) @binding(0) var<storage, read_write> aux_image: array<f32>;
@group(1) @binding(1) var<storage, read> aux_data: array<f32>;
@group(1) @binding(2) var<storage, read_write> debug_image: array<u32>;

// Workgroup shared memory
var<workgroup> sm_color: array<vec4<f32>, 256>;  // .xyz = color_accum, .w = log_t
var<workgroup> sm_nd: array<vec4<f32>, 256>;     // .xyz = normal_accum, .w = depth_accum
var<workgroup> sm_ent: array<vec4<f32>, 256>;    // entropy (4 components)
var<workgroup> sm_aux: array<f32, SM_AUX_SIZE>;  // variable aux [256 * AUX_DIM]
var<workgroup> sm_stats: array<vec4<u32>, 256>;  // [ray_miss, ghost, occluded, useful]
var<workgroup> sm_xl: array<i32, 16>;             // scanline left x per row
var<workgroup> sm_xr: array<i32, 16>;             // scanline right x per row
var<workgroup> sm_prefix: array<u32, 17>;          // prefix sum of row widths

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

    // Initialize per-pixel state (32 threads x pixels each)
    for (var i = lane; i < TS * TS; i += 32u) {
        sm_color[i] = vec4<f32>(0.0);
        sm_nd[i] = vec4<f32>(0.0);
        sm_ent[i] = vec4<f32>(0.0);
        sm_stats[i] = vec4<u32>(0u);
    }
    // Initialize aux shared memory
    for (var i = lane; i < TS * TS * AUX_DIM; i += 32u) {
        sm_aux[i] = 0.0;
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

        // Scanline fill (Variant A): threads 0..TS each compute one row
        if (lane < TS) {
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

            // Convert float range to integer pixel range within [0, 15].
            // Use a small epsilon to conservatively include pixels whose centers
            // fall exactly on the silhouette boundary.  Without this, floating-point
            // rounding in the scanline intersection can exclude boundary pixels that
            // the 3D ray-tet intersection correctly handles.
            if (xl_f <= xr_f) {
                let eps = 0.001;
                var xl_i = max(i32(ceil(xl_f - 0.5 - eps)), 0);
                var xr_i = min(i32(floor(xr_f - 0.5 + eps)), i32(TS) - 1);

                // Trim saturated pixels from row edges.
                // Pixels with T_j < 1e-6 can't contribute — exclude from work distribution.
                while (xl_i <= xr_i && sm_color[lane * TS + u32(xl_i)].w < -13.8) {
                    xl_i++;
                }
                while (xr_i >= xl_i && sm_color[lane * TS + u32(xr_i)].w < -13.8) {
                    xr_i--;
                }

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
            for (var r = 0u; r < TS; r++) {
                let row_w = u32(max(sm_xr[r] - sm_xl[r] + 1, 0));
                sm_prefix[r + 1u] = sm_prefix[r] + row_w;
            }
        }
        workgroupBarrier();

        // Load tet attributes (per-tet, not per-pixel -- hoisted above pixel loop)
        let density_raw = densities[tet_id];
        let colors_tet = vec3<f32>(colors_buf[tet_id * 3u], colors_buf[tet_id * 3u + 1u], colors_buf[tet_id * 3u + 2u]);
        let grad_vec = vec3<f32>(color_grads_buf[tet_id * 3u], color_grads_buf[tet_id * 3u + 1u], color_grads_buf[tet_id * 3u + 2u]);
        var verts: array<vec3<f32>, 4>;
        verts[0] = v0; verts[1] = v1; verts[2] = v2; verts[3] = v3;

        let total = sm_prefix[TS];
        if (total == 0u) {
            // No pixels covered -- skip to next tet (barrier already done)
            continue;
        }

        // Precompute face normals (inward-pointing) for entry normal lookup
        var face_normals: array<vec3<f32>, 4>;
        for (var fi = 0u; fi < 4u; fi++) {
            let f = FACES[fi];
            let va = verts[f[0]];
            let vb = verts[f[1]];
            let vc = verts[f[2]];
            var n = cross(vc - va, vb - va);
            let v_opp = verts[f[3]];
            if (dot(n, v_opp - va) < 0.0) {
                n = -n;
            }
            face_normals[fi] = n;
        }

        // Process covered pixels (each thread handles indices: lane, lane+32, ...)
        for (var idx = lane; idx < total; idx += 32u) {
            // Map linear index to (row, col) via prefix sum
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

            // Skip saturated pixels before expensive ray intersection.
            // log_t < -13.8 means T_j = exp(log_t) < 1e-6.
            if (sm_color[pixel_local].w < -13.8) {
                sm_stats[pixel_local].z += 1u; // occluded
                continue;
            }

            // Compute ray direction from camera intrinsics (matches camera.slang get_ray):
            let dir_cam = normalize(vec3<f32>(
                (f32(px) + 0.5 - cx_cam) / fx,
                (f32(py) + 0.5 - cy_cam) / fy,
                1.0
            ));
            let ray_dir = normalize(c2w * dir_cam);

            // Apply min_t ray origin offset (matches Slang camera.min_t)
            let ray_o = cam + ray_dir * uniforms.min_t;

            // Ray-tet intersection
            var t_min_val = -3.402823e38;
            var t_max_val = 3.402823e38;
            var entry_face_idx = 4u;
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
                let num = dot(n, va - ray_o);
                let den = dot(n, ray_dir);

                if (abs(den) < 1e-20) {
                    if (num > 0.0) { valid = false; }
                    continue;
                }

                let t = num / den;
                if (den > 0.0) {
                    if (t > t_min_val) {
                        t_min_val = t;
                        entry_face_idx = fi;
                    }
                } else {
                    if (t < t_max_val) { t_max_val = t; }
                }
            }

            if (!(valid && t_min_val < t_max_val)) {
                // Ray miss: scanline covers pixel but ray doesn't hit tet.
                // Classify: degenerate slab (thin tet, t_min ≈ t_max) vs true miss.
                if (valid && (t_min_val - t_max_val) < 0.01) {
                    // Degenerate: valid slab but t_min >= t_max by a tiny margin
                    sm_stats[pixel_local].z += 1u; // reuse occluded slot
                } else {
                    sm_stats[pixel_local].x += 1u;
                }
            } else {
                // Volume integral
                let base_offset = dot(grad_vec, ray_o - verts[0]);
                let base_color = colors_tet + vec3<f32>(base_offset);
                let dc_dt = dot(grad_vec, ray_dir);

                let c_start = max(base_color + vec3<f32>(dc_dt * t_min_val), vec3<f32>(0.0));
                let c_end = max(base_color + vec3<f32>(dc_dt * t_max_val), vec3<f32>(0.0));

                let dist = t_max_val - t_min_val;
                let od = clamp(density_raw * dist, 1e-8, 88.0);

                let alpha_t = safe_exp_f32(-od);
                let phi_val = phi(od);
                let w0 = phi_val - alpha_t;
                let w1 = 1.0 - phi_val;
                let c_premul = safe_clip_v3f(c_end * w0 + c_start * w1, 0.0, MAX_VAL);
                let alpha = 1.0 - alpha_t;

                // Composite color into shared memory
                let state = sm_color[pixel_local];
                let T_j = safe_exp_f32(state.w);

                if (od < 0.01) {
                    // Ghost: valid intersection but near-zero optical depth
                    sm_stats[pixel_local].y += 1u;
                } else if (T_j < 1e-6) {
                    // Occluded: decent density but pixel already saturated
                    sm_stats[pixel_local].z += 1u;
                } else {
                    // Useful: contributes meaningful color
                    sm_stats[pixel_local].w += 1u;
                }

                // Early termination: pixel fully opaque, skip all writes
                if (T_j >= 1e-6) {
                    sm_color[pixel_local] = vec4<f32>(
                        state.xyz + c_premul * T_j,
                        state.w - od,
                    );

                    // Composite aux: normals + depth
                    let nd_state = sm_nd[pixel_local];
                    var entry_normal = vec3<f32>(0.0);
                    if (entry_face_idx < 4u) {
                        entry_normal = normalize(-face_normals[entry_face_idx]);
                    }
                    let depth_premul = w0 * t_max_val + w1 * t_min_val;
                    sm_nd[pixel_local] = vec4<f32>(
                        nd_state.xyz + T_j * alpha * entry_normal,
                        nd_state.w + T_j * depth_premul,
                    );

                    // Composite aux: entropy
                    let ent_state = sm_ent[pixel_local];
                    let wt = alpha * T_j;
                    let c_mid = (t_min_val + t_max_val) * 0.5;
                    let wc = wt * c_mid;
                    sm_ent[pixel_local] = vec4<f32>(
                        ent_state.x + wt,
                        ent_state.y + select(0.0, wt * log(wt), wt > 1e-20),
                        ent_state.z + wt * c_mid,
                        ent_state.w + select(0.0, wc * log(wc), wc > 1e-20),
                    );

                    // Composite aux: variable aux (no-op when AUX_DIM=0)
                    for (var ai = 0u; ai < AUX_DIM; ai++) {
                        let sm_idx = pixel_local * AUX_DIM + ai;
                        sm_aux[sm_idx] += T_j * alpha * aux_data[tet_id * AUX_DIM + ai];
                    }
                }
            }
        }
        workgroupBarrier();
    }

    // Write output (32 threads x pixels each)
    for (var i = lane; i < TS * TS; i += 32u) {
        let row = i / TS;
        let col = i % TS;
        let px = tile_x * TS + col;
        let py = tile_y * TS + row;
        if (px < w && py < h) {
            let pixel_idx = py * w + px;

            // RGBA
            let state = sm_color[i];
            let T_final = safe_exp_f32(state.w);
            rendered_image[pixel_idx * 4u] = state.x;
            rendered_image[pixel_idx * 4u + 1u] = state.y;
            rendered_image[pixel_idx * 4u + 2u] = state.z;
            rendered_image[pixel_idx * 4u + 3u] = 1.0 - T_final;

            // Aux: [normal.xyz, depth, entropy.4, aux_data...]
            let aux_base = pixel_idx * AUX_STRIDE;
            let nd = sm_nd[i];
            aux_image[aux_base + 0u] = nd.x;
            aux_image[aux_base + 1u] = nd.y;
            aux_image[aux_base + 2u] = nd.z;
            aux_image[aux_base + 3u] = nd.w;
            let ent = sm_ent[i];
            aux_image[aux_base + 4u] = ent.x;
            aux_image[aux_base + 5u] = ent.y;
            aux_image[aux_base + 6u] = ent.z;
            aux_image[aux_base + 7u] = ent.w;
            for (var ai = 0u; ai < AUX_DIM; ai++) {
                aux_image[aux_base + 8u + ai] = sm_aux[i * AUX_DIM + ai];
            }

            // Debug stats: [ray_miss, ghost, occluded, useful] per pixel
            let stats = sm_stats[i];
            let dbg_base = pixel_idx * 4u;
            debug_image[dbg_base + 0u] = stats.x;
            debug_image[dbg_base + 1u] = stats.y;
            debug_image[dbg_base + 2u] = stats.z;
            debug_image[dbg_base + 3u] = stats.w;
        }
    }
}
