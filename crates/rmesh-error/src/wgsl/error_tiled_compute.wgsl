// Error tiled compute shader: per-tet error statistics accumulation.
//
// Same tiled traversal and scanline fill as rasterize_compute.wgsl, but
// accumulates per-tet error statistics via atomic<f32>/atomic<i32> instead
// of compositing colors. Used during densification.
//
// Thread model: 32 threads per workgroup (1 warp), 1 workgroup per tile.
// Dispatch: (num_tiles, 1, 1).

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

// Group 0: 9 read-only scene/tile bindings
@group(0) @binding(0) var<storage, read> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> vertices: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> colors_buf: array<f32>;
@group(0) @binding(4) var<storage, read> densities: array<f32>;
@group(0) @binding(5) var<storage, read> color_grads_buf: array<f32>;
@group(0) @binding(6) var<storage, read> tile_sort_values: array<u32>;
@group(0) @binding(7) var<storage, read> tile_ranges: array<u32>;
@group(0) @binding(8) var<storage, read> tile_uniforms: TileUniforms;

// Group 1: error inputs (read) + error outputs (atomic rw)
@group(1) @binding(0) var<storage, read> pixel_err: array<f32>;
@group(1) @binding(1) var<storage, read> ssim_err: array<f32>;
@group(1) @binding(2) var<storage, read_write> tet_err: array<atomic<f32>>;
@group(1) @binding(3) var<storage, read_write> tet_count_buf: array<atomic<i32>>;

// Face (a, b, c, opposite_vertex) -- opposite used to flip normal inward
const FACES: array<vec4<u32>, 4> = array<vec4<u32>, 4>(
    vec4<u32>(0u, 2u, 1u, 3u),
    vec4<u32>(1u, 2u, 3u, 0u),
    vec4<u32>(0u, 3u, 2u, 1u),
    vec4<u32>(3u, 0u, 1u, 2u),
);

// Workgroup shared memory
var<workgroup> sm_log_t: array<f32, 256>;    // per-pixel log transmittance
var<workgroup> sm_xl: array<i32, 16>;
var<workgroup> sm_xr: array<i32, 16>;
var<workgroup> sm_prefix: array<u32, 17>;

fn load_f32x3_v(idx: u32) -> vec3<f32> {
    return vec3<f32>(vertices[idx * 3u], vertices[idx * 3u + 1u], vertices[idx * 3u + 2u]);
}

fn safe_exp_f32(v: f32) -> f32 {
    return exp(clamp(v, -88.0, 46.0517));
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

    // Initialize per-pixel log transmittance to 0 (T = 1)
    for (var i = lane; i < TS * TS; i += 32u) {
        sm_log_t[i] = 0.0;
    }
    workgroupBarrier();

    // Process tets front-to-back (same order as forward rasterize)
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

        // Scanline fill: threads 0..TS each compute one row
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

                // Include vertices within this pixel row
                for (var v = 0u; v < 4u; v++) {
                    let vy = proj[v].y;
                    if (vy >= yc - 0.5 && vy < yc + 0.5) {
                        xl_f = min(xl_f, proj[v].x);
                        xr_f = max(xr_f, proj[v].x);
                    }
                }
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

        // Prefix sum of row widths (thread 0)
        if (lane == 0u) {
            sm_prefix[0] = 0u;
            for (var r = 0u; r < TS; r++) {
                let row_w = u32(max(sm_xr[r] - sm_xl[r] + 1, 0));
                sm_prefix[r + 1u] = sm_prefix[r] + row_w;
            }
        }
        workgroupBarrier();

        var verts: array<vec3<f32>, 4>;
        verts[0] = v0; verts[1] = v1; verts[2] = v2; verts[3] = v3;

        let total = sm_prefix[TS];
        if (total == 0u) {
            continue;
        }

        let density_raw = densities[tet_id];
        let centroid = (v0 + v1 + v2 + v3) * 0.25;

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

            // Early termination: pixel nearly opaque
            if (sm_log_t[pixel_local] < -5.5) {
                continue;
            }

            let pixel_idx = py * w + px;

            // Compute ray from camera intrinsics (matches camera.slang get_ray):
            let dir_cam = normalize(vec3<f32>(
                (f32(px) + 0.5 - cx_cam) / fx,
                (f32(py) + 0.5 - cy_cam) / fy,
                1.0
            ));
            let ray_dir = normalize(c2w * dir_cam);

            let ray_o = cam + ray_dir * uniforms.min_t;

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
                    if (t > t_min_val) { t_min_val = t; }
                } else {
                    if (t < t_max_val) { t_max_val = t; }
                }
            }

            if (valid && t_min_val < t_max_val) {
                let dist = t_max_val - t_min_val;
                let od = clamp(density_raw * dist, 1e-8, 88.0);
                let trans = safe_exp_f32(sm_log_t[pixel_local]);

                if (trans >= 1e-6) {
                    let T_val = trans * od;

                    let entry_pt = ray_o + ray_dir * t_min_val;
                    let exit_pt = ray_o + ray_dir * t_max_val;

                    let pe = pixel_err[pixel_idx];
                    let se = ssim_err[pixel_idx];

                    let base = tet_id * 16u;
                    atomicAdd(&tet_err[base + 0u], T_val);              // total weight
                    atomicAdd(&tet_err[base + 1u], T_val * pe);         // weighted L1
                    atomicAdd(&tet_err[base + 2u], T_val * pe * pe);    // weighted L1^2
                    atomicAdd(&tet_err[base + 5u], T_val * se);         // weighted SSIM
                    atomicAdd(&tet_err[base + 6u], T_val * entry_pt.x); // weighted entry.x
                    atomicAdd(&tet_err[base + 7u], T_val * entry_pt.y); // weighted entry.y
                    atomicAdd(&tet_err[base + 8u], T_val * entry_pt.z); // weighted entry.z
                    atomicAdd(&tet_err[base + 9u], T_val * exit_pt.x);  // weighted exit.x
                    atomicAdd(&tet_err[base + 10u], T_val * exit_pt.y); // weighted exit.y
                    atomicAdd(&tet_err[base + 11u], T_val * exit_pt.z); // weighted exit.z
                    atomicAdd(&tet_err[base + 12u], T_val);             // weight sum
                    atomicAdd(&tet_err[base + 13u], T_val * centroid.x);// weighted centroid.x
                    atomicAdd(&tet_err[base + 14u], T_val * centroid.y);// weighted centroid.y
                    atomicAdd(&tet_err[base + 15u], T_val * centroid.z);// weighted centroid.z

                    let count_base = tet_id * 2u;
                    atomicAdd(&tet_count_buf[count_base], 1i);
                    atomicMax(&tet_count_buf[count_base + 1u], i32(65535.0 * T_val));

                    sm_log_t[pixel_local] -= od;
                }
            }
        }
        workgroupBarrier();
    }
}
