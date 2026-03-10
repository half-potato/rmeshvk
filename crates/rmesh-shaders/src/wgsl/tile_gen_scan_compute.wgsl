// Tile-tet pair generation using prefix-scan offsets (no atomics).
//
// Conservative scanline fill (Variant B) for tile-tet overlap testing.
// Replaces the AABB + hull_overlaps_tile approach with exact convex hull
// scanline fill at tile granularity.
//
// Key encoding: 17-bit tile_id (supports 131K tiles), 15-bit depth.
// Dispatched indirectly for visible_tet_count threads.

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

@group(0) @binding(0) var<storage, read> tile_uniforms: TileUniforms;
@group(0) @binding(1) var<storage, read> uniforms: Uniforms;
@group(0) @binding(2) var<storage, read> vertices: array<f32>;
@group(0) @binding(3) var<storage, read> indices: array<u32>;
@group(0) @binding(4) var<storage, read> compact_tet_ids: array<u32>;
@group(0) @binding(5) var<storage, read> circumdata: array<f32>;
@group(0) @binding(6) var<storage, read_write> tile_sort_keys: array<u32>;
@group(0) @binding(7) var<storage, read_write> tile_sort_values: array<u32>;
@group(0) @binding(8) var<storage, read> pair_offsets: array<u32>;
@group(0) @binding(9) var<storage, read> tiles_touched: array<u32>;
@group(0) @binding(10) var<storage, read> visible_count: array<u32>;
@group(0) @binding(11) var<storage, read_write> num_keys_out: array<u32>;

fn load_vertex(idx: u32) -> vec3<f32> {
    let i = idx * 3u;
    return vec3<f32>(vertices[i], vertices[i + 1u], vertices[i + 2u]);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let vis_idx = gid.x + gid.y * nwg.x * 64u;
    let n = visible_count[0];
    if (vis_idx >= n) {
        return;
    }

    let tet_id = compact_tet_ids[vis_idx];
    // pair_offsets contains INCLUSIVE prefix sum; convert to exclusive offset
    let max_write = tiles_touched[vis_idx];
    let write_start = pair_offsets[vis_idx] - max_write;

    // Last visible tet writes total_pairs for radix sort
    if (vis_idx == n - 1u) {
        num_keys_out[0] = pair_offsets[vis_idx];
    }

    // Early return for behind-camera tets (tiles_touched=0 from forward_compute)
    if (max_write == 0u) {
        return;
    }

    // Load tet vertices
    let i0 = indices[tet_id * 4u];
    let i1 = indices[tet_id * 4u + 1u];
    let i2 = indices[tet_id * 4u + 2u];
    let i3 = indices[tet_id * 4u + 3u];

    let v0 = load_vertex(i0);
    let v1 = load_vertex(i1);
    let v2 = load_vertex(i2);
    let v3 = load_vertex(i3);

    // Project to clip space
    let vp = mat4x4<f32>(uniforms.vp_col0, uniforms.vp_col1, uniforms.vp_col2, uniforms.vp_col3);
    let c0 = vp * vec4<f32>(v0, 1.0);
    let c1 = vp * vec4<f32>(v1, 1.0);
    let c2 = vp * vec4<f32>(v2, 1.0);
    let c3 = vp * vec4<f32>(v3, 1.0);

    let W = f32(tile_uniforms.screen_width);
    let H = f32(tile_uniforms.screen_height);
    let ts = f32(tile_uniforms.tile_size);

    // NDC to pixel coords (epsilon matches forward_compute.wgsl project_to_ndc)
    let n0 = c0.xyz / (c0.w + 1e-6);
    let n1 = c1.xyz / (c1.w + 1e-6);
    let n2 = c2.xyz / (c2.w + 1e-6);
    let n3 = c3.xyz / (c3.w + 1e-6);

    var proj: array<vec2<f32>, 4>;
    proj[0] = vec2<f32>((n0.x + 1.0) * 0.5 * W, (1.0 - n0.y) * 0.5 * H);
    proj[1] = vec2<f32>((n1.x + 1.0) * 0.5 * W, (1.0 - n1.y) * 0.5 * H);
    proj[2] = vec2<f32>((n2.x + 1.0) * 0.5 * W, (1.0 - n2.y) * 0.5 * H);
    proj[3] = vec2<f32>((n3.x + 1.0) * 0.5 * W, (1.0 - n3.y) * 0.5 * H);

    // AABB for tile bounds
    let pix_min_x = max(min(min(proj[0].x, proj[1].x), min(proj[2].x, proj[3].x)), 0.0);
    let pix_max_x = min(max(max(proj[0].x, proj[1].x), max(proj[2].x, proj[3].x)), W - 1.0);
    let pix_min_y = max(min(min(proj[0].y, proj[1].y), min(proj[2].y, proj[3].y)), 0.0);
    let pix_max_y = min(max(max(proj[0].y, proj[1].y), max(proj[2].y, proj[3].y)), H - 1.0);

    if (pix_min_x > pix_max_x || pix_min_y > pix_max_y) {
        return;
    }

    let tiles_x_total = u32(ceil(W / ts));
    let tile_min_x = u32(pix_min_x / ts);
    let tile_max_x = min(u32(pix_max_x / ts), tiles_x_total - 1u);
    let tile_min_y = u32(pix_min_y / ts);
    let tile_max_y = min(u32(pix_max_y / ts), u32(ceil(H / ts)) - 1u);

    // Depth key from circumsphere
    let cx = circumdata[tet_id * 4u];
    let cy = circumdata[tet_id * 4u + 1u];
    let cz = circumdata[tet_id * 4u + 2u];
    let r2 = circumdata[tet_id * 4u + 3u];
    let cam = uniforms.cam_pos_pad.xyz;
    let diff = vec3<f32>(cx, cy, cz) - cam;
    let depth = dot(diff, diff) - r2;
    let inv_depth = ~bitcast<u32>(depth);
    let depth_bits = inv_depth >> 17u;

    // Conservative scanline tile enumeration (Variant B) — same algorithm as forward_compute
    let T = ts;
    let ei_arr = array<u32, 6>(0u, 0u, 0u, 1u, 1u, 2u);
    let ej_arr = array<u32, 6>(1u, 2u, 3u, 2u, 3u, 3u);

    var local_count = 0u;
    for (var ty = tile_min_y; ty <= tile_max_y; ty++) {
        var xl = 1e10f;
        var xr = -1e10f;

        // Edge intersections at both tile-row boundaries
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

            for (var tx = tx0; tx <= tx1; tx++) {
                let tile_id = ty * tile_uniforms.tiles_x + tx;
                let key = (tile_id << 15u) | (depth_bits & 0x7FFFu);

                if (local_count < max_write) {
                    let write_idx = write_start + local_count;
                    if (write_idx < arrayLength(&tile_sort_keys)) {
                        tile_sort_keys[write_idx] = key;
                        tile_sort_values[write_idx] = tet_id;
                    }
                    local_count++;
                }
            }
        }
    }
    // Remaining pre-allocated slots keep sentinel values from tile_fill
}
