// Workgroup-level exclusive prefix sum (Blelloch scan).
//
// Input: tiles_touched[0..N]
// Output: pair_offsets[0..N] (exclusive scan), block_sums[0..num_blocks]
//
// Each workgroup processes 256 elements.
// Dispatched as (ceil(N/256), 1, 1) via indirect dispatch.

@group(0) @binding(0) var<storage, read> tiles_touched: array<u32>;
@group(0) @binding(1) var<storage, read_write> pair_offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(3) var<storage, read> visible_count: array<u32>;

var<workgroup> shared_data: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let n = visible_count[0];
    let global_idx = wg_id.x * 256u + lid;

    // Load input (0 for out-of-bounds)
    var val = 0u;
    if (global_idx < n) {
        val = tiles_touched[global_idx];
    }
    shared_data[lid] = val;
    workgroupBarrier();

    // Up-sweep (reduce)
    for (var stride = 1u; stride < 256u; stride *= 2u) {
        let idx = (lid + 1u) * stride * 2u - 1u;
        if (idx < 256u) {
            shared_data[idx] += shared_data[idx - stride];
        }
        workgroupBarrier();
    }

    // Store block sum and clear last element
    if (lid == 0u) {
        block_sums[wg_id.x] = shared_data[255];
        shared_data[255] = 0u;
    }
    workgroupBarrier();

    // Down-sweep
    for (var stride = 128u; stride >= 1u; stride /= 2u) {
        let idx = (lid + 1u) * stride * 2u - 1u;
        if (idx < 256u) {
            let tmp = shared_data[idx - stride];
            shared_data[idx - stride] = shared_data[idx];
            shared_data[idx] += tmp;
        }
        workgroupBarrier();
    }

    // Write output
    if (global_idx < n) {
        pair_offsets[global_idx] = shared_data[lid];
    }
}
