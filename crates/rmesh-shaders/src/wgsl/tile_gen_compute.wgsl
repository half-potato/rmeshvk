// Tile-tet pair generation: per visible tet, project to screen, compute tile AABB,
// emit (tile_id << 17 | depth) key for each overlapping tile.
// Dispatched for visible_tet_count threads.

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
@group(0) @binding(4) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(5) var<storage, read> circumdata: array<f32>;
@group(0) @binding(6) var<storage, read_write> tile_sort_keys: array<u32>;
@group(0) @binding(7) var<storage, read_write> tile_sort_values: array<u32>;
@group(0) @binding(8) var<storage, read_write> tile_pair_count: array<atomic<u32>>;

fn load_vertex(idx: u32) -> vec3<f32> {
    let i = idx * 3u;
    return vec3<f32>(vertices[i], vertices[i + 1u], vertices[i + 2u]);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let vis_idx = gid.x + gid.y * nwg.x * 64u;
    if (vis_idx >= tile_uniforms.visible_tet_count) {
        return;
    }

    let tet_id = sorted_indices[vis_idx];

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

    let any_behind = (c0.w <= 0.0) || (c1.w <= 0.0) || (c2.w <= 0.0) || (c3.w <= 0.0);

    let W = f32(tile_uniforms.screen_width);
    let H = f32(tile_uniforms.screen_height);

    var tile_min_x: u32;
    var tile_min_y: u32;
    var tile_max_x: u32;
    var tile_max_y: u32;

    if (any_behind) {
        // Conservative: cover all tiles
        tile_min_x = 0u;
        tile_min_y = 0u;
        tile_max_x = tile_uniforms.tiles_x - 1u;
        tile_max_y = tile_uniforms.tiles_y - 1u;
    } else {
        // NDC coords
        let n0 = c0.xyz / c0.w;
        let n1 = c1.xyz / c1.w;
        let n2 = c2.xyz / c2.w;
        let n3 = c3.xyz / c3.w;

        let ndc_min_x = min(min(n0.x, n1.x), min(n2.x, n3.x));
        let ndc_max_x = max(max(n0.x, n1.x), max(n2.x, n3.x));
        let ndc_min_y = min(min(n0.y, n1.y), min(n2.y, n3.y));
        let ndc_max_y = max(max(n0.y, n1.y), max(n2.y, n3.y));

        // NDC to pixel: wgpu has Y-UP in NDC (+1 at top, -1 at bottom)
        // pix_x = (ndc_x + 1) * 0.5 * W
        // pix_y = (1 - ndc_y) * 0.5 * H  (flip Y: ndc_y=+1 → pix_y=0)
        let pix_left = (ndc_min_x + 1.0) * 0.5 * W;
        let pix_right = (ndc_max_x + 1.0) * 0.5 * W;
        let pix_top = (1.0 - ndc_max_y) * 0.5 * H;
        let pix_bottom = (1.0 - ndc_min_y) * 0.5 * H;

        let ts = f32(tile_uniforms.tile_size);

        // Clamp to screen bounds
        let clamped_left = max(pix_left, 0.0);
        let clamped_right = min(pix_right, W - 1.0);
        let clamped_top = max(pix_top, 0.0);
        let clamped_bottom = min(pix_bottom, H - 1.0);

        if (clamped_left > clamped_right || clamped_top > clamped_bottom) {
            return; // Off-screen
        }

        tile_min_x = u32(clamped_left / ts);
        tile_max_x = min(u32(clamped_right / ts), tile_uniforms.tiles_x - 1u);
        tile_min_y = u32(clamped_top / ts);
        tile_max_y = min(u32(clamped_bottom / ts), tile_uniforms.tiles_y - 1u);
    }

    // Depth key from circumsphere (same as forward_compute)
    let cx = circumdata[tet_id * 4u];
    let cy = circumdata[tet_id * 4u + 1u];
    let cz = circumdata[tet_id * 4u + 2u];
    let r2 = circumdata[tet_id * 4u + 3u];
    let cam = uniforms.cam_pos_pad.xyz;
    let diff = vec3<f32>(cx, cy, cz) - cam;
    let depth = dot(diff, diff) - r2;
    let inv_depth = ~bitcast<u32>(depth);
    let depth_bits = inv_depth >> 15u; // 17 bits of depth

    // Emit tile-tet pairs
    for (var ty = tile_min_y; ty <= tile_max_y; ty++) {
        for (var tx = tile_min_x; tx <= tile_max_x; tx++) {
            let tile_id = ty * tile_uniforms.tiles_x + tx;
            let key = (tile_id << 17u) | (depth_bits & 0x1FFFFu);

            let pair_idx = atomicAdd(&tile_pair_count[0], 1u);
            if (pair_idx < tile_uniforms.max_pairs) {
                tile_sort_keys[pair_idx] = key;
                tile_sort_values[pair_idx] = tet_id;
            }
        }
    }
}
