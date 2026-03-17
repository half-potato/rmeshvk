// Converts DrawIndirectArgs (instance_count = visible tet count) to
// mesh shader dispatch args: ceil(visible_count / 32) workgroups.

struct DrawIndirectArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0) var<storage, read> indirect_args: DrawIndirectArgs;
@group(0) @binding(1) var<storage, read_write> mesh_indirect: array<u32, 3>;

const TETS_PER_GROUP: u32 = 8u;

@compute @workgroup_size(1)
fn main() {
    let count = indirect_args.instance_count;
    mesh_indirect[0] = (count + TETS_PER_GROUP - 1u) / TETS_PER_GROUP;
    mesh_indirect[1] = 1u;
    mesh_indirect[2] = 1u;
}
