// Compute velocity divergence ∇·u for the pressure solve.

struct FluidUniforms {
    dt: f32,
    viscosity: f32,
    tet_count: u32,
    jacobi_iter: u32,
    gravity: vec4<f32>,
    source_pos: vec4<f32>,
    source_strength: f32,
    buoyancy: f32,
    density_scale: f32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> uniforms: FluidUniforms;
@group(0) @binding(1) var<storage, read> tet_neighbors: array<i32>;
@group(0) @binding(2) var<storage, read> face_geo: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> tet_volumes: array<f32>;

@group(1) @binding(0) var<storage, read> velocity: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> divergence: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.tet_count) {
        return;
    }

    let vel_i = velocity[tid].xyz;
    var div = 0.0;

    for (var f = 0u; f < 4u; f++) {
        let neighbor = tet_neighbors[tid * 4u + f];
        if (neighbor < 0) {
            continue; // boundary: no-flow, zero flux
        }

        let n_area = face_geo[tid * 4u + f].xyz; // area-weighted outward normal
        let vel_j = velocity[u32(neighbor)].xyz;

        // Face velocity (average of cell and neighbor)
        let vel_face = 0.5 * (vel_i + vel_j);

        // Flux contribution to divergence
        div += dot(vel_face, n_area);
    }

    divergence[tid] = div / max(tet_volumes[tid], 1e-10);
}
