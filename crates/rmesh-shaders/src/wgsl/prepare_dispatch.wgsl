// Prepare indirect dispatch args from visible_tet_count.
//
// Reads indirect_args.instance_count (= visible_tet_count set by forward_compute)
// and writes dispatch args for prefix scan and tile gen.
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
// dispatch_scan: (ceil(visible_count/256), 1, 1)
@group(0) @binding(1) var<storage, read_write> dispatch_scan: array<u32>;
// dispatch_tile_gen: (ceil(visible_count/64), 1, 1) — split into 2D if > 65535
@group(0) @binding(2) var<storage, read_write> dispatch_tile_gen: array<u32>;
// visible_tet_count output (for shaders that need it as a standalone value)
@group(0) @binding(3) var<storage, read_write> visible_count_out: array<u32>;

@compute @workgroup_size(1)
fn main() {
    let count = indirect_args.instance_count;
    visible_count_out[0] = count;

    // Prefix scan dispatch: workgroup_size=256
    let scan_wgs = (count + 255u) / 256u;
    dispatch_scan[0] = scan_wgs;
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
