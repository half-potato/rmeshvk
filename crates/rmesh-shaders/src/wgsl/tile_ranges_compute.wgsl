// Find per-tile [start, end) ranges in the sorted tile-tet pair array.
// Dispatched for actual_pair_count threads (clamped to max_pairs).

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

@group(0) @binding(0) var<storage, read> tile_sort_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> tile_ranges: array<u32>;
@group(0) @binding(2) var<storage, read> tile_uniforms: TileUniforms;
@group(0) @binding(3) var<storage, read> tile_pair_count: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 256u;
    // Use min(actual pair count, max_pairs) as the effective count
    let actual_count = min(tile_pair_count[0], tile_uniforms.max_pairs);

    if (idx >= actual_count) {
        return;
    }

    let key = tile_sort_keys[idx];
    // Sentinel keys (0xFFFFFFFF) are sorted to end — skip them
    if (key == 0xFFFFFFFFu) {
        return;
    }

    let tile_id = key >> 17u;

    // Check if this is the start of a new tile
    if (idx == 0u) {
        tile_ranges[tile_id * 2u] = idx;
    } else {
        let prev_key = tile_sort_keys[idx - 1u];
        let prev_tile = prev_key >> 17u;
        if (tile_id != prev_tile) {
            // Start of new tile
            tile_ranges[tile_id * 2u] = idx;
            // End of previous tile
            if (prev_key != 0xFFFFFFFFu) {
                tile_ranges[prev_tile * 2u + 1u] = idx;
            }
        }
    }

    // If this is the last valid entry, set the end
    if (idx == actual_count - 1u) {
        tile_ranges[tile_id * 2u + 1u] = idx + 1u;
    }
}
