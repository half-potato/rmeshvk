// Find per-tile [start, end) ranges in the sorted tile-tet pair array.
// Dispatched for tile_pair_count[0] threads.

@group(0) @binding(0) var<storage, read> tile_sort_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> tile_ranges: array<u32>;
@group(0) @binding(2) var<storage, read> tile_uniforms: array<u32>; // unused but kept for bind group compat
@group(0) @binding(3) var<storage, read> tile_pair_count: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 256u;
    let actual_count = tile_pair_count[0];

    if (idx >= actual_count) {
        return;
    }

    let key = tile_sort_keys[idx];
    // Sentinel keys (0xFFFFFFFF) are sorted to end — skip them
    if (key == 0xFFFFFFFFu) {
        return;
    }

    let tile_id = key >> 15u;

    // Check if this is the start of a new tile
    if (idx == 0u) {
        tile_ranges[tile_id * 2u] = idx;
    } else {
        let prev_key = tile_sort_keys[idx - 1u];
        let prev_tile = prev_key >> 15u;
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
