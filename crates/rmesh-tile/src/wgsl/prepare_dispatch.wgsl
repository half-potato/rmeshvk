// Prepare indirect dispatch args from visible_tet_count.
//
// Reads indirect_args.instance_count (= visible_tet_count set by project_compute)
// and writes dispatch args for RTS prefix scan and tile gen.
// Also computes RTS info struct {size, vec_size, thread_blocks}.
// Eliminates CPU readback of visible_tet_count.
//
// Dispatched as (1, 1, 1).

struct DrawIndirectArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0) var<storage, read> indirect_args: DrawIndirectArgs;
// dispatch_scan: (thread_blocks, 1, 1) for RTS reduce/downsweep
@group(0) @binding(1) var<storage, read_write> dispatch_scan: array<u32>;
// dispatch_tile_gen: (ceil(visible_count/64), 1, 1) — split into 2D if > 65535
@group(0) @binding(2) var<storage, read_write> dispatch_tile_gen: array<u32>;
// visible_tet_count output (for shaders that need it as a standalone value)
@group(0) @binding(3) var<storage, read_write> visible_count_out: array<u32>;
// RTS info: {size, vec_size, thread_blocks}
@group(0) @binding(4) var<storage, read_write> rts_info: array<u32>;

@compute @workgroup_size(1)
fn main() {
    let count = indirect_args.instance_count;
    visible_count_out[0] = count;

    // RTS prefix scan params
    let vec_size = (count + 3u) / 4u;
    let thread_blocks = (vec_size + 1023u) / 1024u;

    rts_info[0] = count;         // size (number of u32 elements)
    rts_info[1] = vec_size;      // number of vec4 elements
    rts_info[2] = thread_blocks; // number of workgroups for reduce/downsweep

    // Scan dispatch: (thread_blocks, 1, 1) for RTS reduce/downsweep indirect dispatch
    dispatch_scan[0] = thread_blocks;
    dispatch_scan[1] = 1u;
    dispatch_scan[2] = 1u;

    // Tile gen dispatch: workgroup_size=64, split into 2D if needed
    let gen_wgs = (count + 63u) / 64u;
    if (gen_wgs <= 65535u) {
        dispatch_tile_gen[0] = gen_wgs;
        dispatch_tile_gen[1] = 1u;
    } else {
        dispatch_tile_gen[0] = 65535u;
        dispatch_tile_gen[1] = (gen_wgs + 65534u) / 65535u;
    }
    dispatch_tile_gen[2] = 1u;
}
