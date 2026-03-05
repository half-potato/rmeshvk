// Fill tile sort buffers with sentinel values (0xFFFFFFFF).
// Dispatched for max_pairs_pow2 threads.

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

@group(0) @binding(0) var<storage, read> tile_uniforms: TileUniforms;
@group(0) @binding(1) var<storage, read_write> tile_sort_keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> tile_sort_values: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 256u;
    if (idx >= tile_uniforms.max_pairs_pow2) {
        return;
    }
    tile_sort_keys[idx] = 0xFFFFFFFFu;
    tile_sort_values[idx] = 0u;
}
