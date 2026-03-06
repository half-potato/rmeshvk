// Propagate scanned block sums into per-element prefix offsets.
//
// After prefix_scan.wgsl produces per-block pair_offsets and block_sums,
// and after block_sums is scanned (via another prefix_scan dispatch),
// this shader adds the scanned block sum to each element.
//
// Also: the last active thread writes total_pairs to num_keys_buf
// (for radix sort to read directly, no CPU readback needed).
//
// Dispatched as (ceil(N/256), 1, 1) via indirect dispatch.

@group(0) @binding(0) var<storage, read_write> pair_offsets: array<u32>;
@group(0) @binding(1) var<storage, read> block_sums_scanned: array<u32>;
@group(0) @binding(2) var<storage, read> visible_count: array<u32>;
@group(0) @binding(3) var<storage, read_write> num_keys_buf: array<u32>;
// Total pairs = last block_sum_scanned + last block_sum_original
// We store original block sums here for the final computation
@group(0) @binding(4) var<storage, read> block_sums_original: array<u32>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let n = visible_count[0];
    let global_idx = wg_id.x * 256u + lid;

    // Add scanned block sum to each element
    if (global_idx < n) {
        pair_offsets[global_idx] += block_sums_scanned[wg_id.x];
    }

    // Last thread of last active workgroup writes total_pairs
    let num_blocks = (n + 255u) / 256u;
    if (wg_id.x == num_blocks - 1u && lid == 0u) {
        let total = block_sums_scanned[num_blocks - 1u] + block_sums_original[num_blocks - 1u];
        num_keys_buf[0] = total;
    }
}
