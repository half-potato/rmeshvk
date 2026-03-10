// Large block-level prefix scan: single workgroup scans up to 2048 block sums.
//
// 256 threads x 8 elements per thread = 2048 max blocks = 524,288 max visible tets.
// Uses the same bind group layout as prefix_scan.wgsl.
//
// Algorithm: sequential scan of K elements per thread -> parallel Blelloch scan
// of 256 partial sums -> write back final exclusive prefix sums.
//
// Dispatched as (1, 1, 1).

@group(0) @binding(0) var<storage, read> block_sums_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> block_sums_out: array<u32>;
@group(0) @binding(2) var<storage, read_write> scratch: array<u32>;  // unused, layout compat
@group(0) @binding(3) var<storage, read> visible_count: array<u32>;

const EPT: u32 = 8u;  // elements per thread

var<workgroup> shared_data: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lid: u32) {
    // Number of blocks = ceil(visible_count / 256)
    let n = (visible_count[0] + 255u) / 256u;

    // Phase 1: Each thread sequentially scans EPT elements, building exclusive prefix sums
    var local_prefix: array<u32, 8>;  // EPT
    var thread_total = 0u;

    for (var k = 0u; k < EPT; k++) {
        let idx = lid * EPT + k;
        var val = 0u;
        if (idx < n) {
            val = block_sums_in[idx];
        }
        local_prefix[k] = thread_total;
        thread_total += val;
    }
    shared_data[lid] = thread_total;
    workgroupBarrier();

    // Phase 2: Blelloch scan on shared_data[0..255]
    // Up-sweep (reduce)
    for (var stride = 1u; stride < 256u; stride *= 2u) {
        let idx = (lid + 1u) * stride * 2u - 1u;
        if (idx < 256u) {
            shared_data[idx] += shared_data[idx - stride];
        }
        workgroupBarrier();
    }

    if (lid == 0u) {
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

    // Phase 3: Write back final exclusive prefix sums
    let base_prefix = shared_data[lid];
    for (var k = 0u; k < EPT; k++) {
        let idx = lid * EPT + k;
        if (idx < n) {
            block_sums_out[idx] = base_prefix + local_prefix[k];
        }
    }
}
