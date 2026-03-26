// Converts DrawIndirectArgs (instance_count = visible tet count) to
// a combined buffer with compute dispatch args + draw-indexed-indirect args.
//
// Output layout (8 x u32 = 32 bytes):
//   [0..2]: dispatch_workgroups(X, Y, Z) for interval_compute
//   [3..7]: draw_indexed_indirect(index_count, instance_count, first_index, base_vertex, first_instance)

struct DrawIndirectArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0) var<storage, read> indirect_args: DrawIndirectArgs;
@group(0) @binding(1) var<storage, read_write> combined_args: array<u32, 8>;

override WG_SIZE: u32 = 64u;

@compute @workgroup_size(1)
fn main() {
    let count = indirect_args.instance_count;
    // Compute dispatch args
    combined_args[0] = (count + WG_SIZE - 1u) / WG_SIZE;
    combined_args[1] = 1u;
    combined_args[2] = 1u;
    // Draw-indexed-indirect args (static fan index buffer, 12 indices per tet)
    combined_args[3] = count * 12u;  // index_count
    combined_args[4] = 1u;           // instance_count
    combined_args[5] = 0u;           // first_index
    combined_args[6] = 0u;           // base_vertex
    combined_args[7] = 0u;           // first_instance
}
