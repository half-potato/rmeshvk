// Jacobi iteration for pressure Poisson equation.
// Solves ∇²p = ∇·u via iterative relaxation.
// Ping-pong between pressure (read) and pressure_tmp (write) is handled by the Rust side.

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

@group(1) @binding(0) var<storage, read> pressure: array<f32>;
@group(1) @binding(1) var<storage, read_write> pressure_tmp: array<f32>;
@group(1) @binding(2) var<storage, read> divergence: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.tet_count) {
        return;
    }

    var sum_coeff = 0.0;
    var sum_p = 0.0;

    for (var f = 0u; f < 4u; f++) {
        let neighbor = tet_neighbors[tid * 4u + f];
        if (neighbor < 0) {
            continue; // Neumann BC: dp/dn = 0
        }

        let coeff = face_geo[tid * 4u + f].w; // A_f / d_ij
        sum_coeff += coeff;
        sum_p += coeff * pressure[u32(neighbor)];
    }

    if (sum_coeff > 0.0) {
        pressure_tmp[tid] = (sum_p - divergence[tid] * tet_volumes[tid]) / sum_coeff;
    } else {
        pressure_tmp[tid] = 0.0; // isolated tet
    }
}
