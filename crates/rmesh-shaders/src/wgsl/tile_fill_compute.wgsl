// Fill tile sort buffers with sentinel values (0xFFFFFFFF).
// Dispatched for arrayLength(&tile_sort_keys) threads.

@group(0) @binding(0) var<storage, read> tile_uniforms: array<u32>; // unused but kept for bind group compat
@group(0) @binding(1) var<storage, read_write> tile_sort_keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> tile_sort_values: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 256u;
    if (idx >= arrayLength(&tile_sort_keys)) {
        return;
    }
    tile_sort_keys[idx] = 0xFFFFFFFFu;
    tile_sort_values[idx] = 0u;
}
