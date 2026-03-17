// Jacobi iteration for viscous diffusion.
// Solves (I - ν*dt*∇²)v = v_old via iterative relaxation.
// Ping-pong between velocity (read) and velocity_tmp (write) is handled by the Rust side.

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
@group(1) @binding(1) var<storage, read_write> velocity_tmp: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.tet_count) {
        return;
    }

    let vol = tet_volumes[tid];
    let vel_old = velocity[tid].xyz;

    var sum_coeff = 0.0;
    var sum_vel = vec3<f32>(0.0);

    for (var f = 0u; f < 4u; f++) {
        let neighbor = tet_neighbors[tid * 4u + f];
        if (neighbor < 0) {
            continue;
        }

        let coeff = face_geo[tid * 4u + f].w; // A_f / d_ij
        sum_coeff += coeff;
        sum_vel += coeff * velocity[u32(neighbor)].xyz;
    }

    let alpha = uniforms.viscosity * uniforms.dt / max(vol, 1e-10);
    let denom = 1.0 + alpha * sum_coeff;
    let new_vel = (vel_old + alpha * sum_vel) / denom;

    velocity_tmp[tid] = vec4<f32>(new_vel, 0.0);
}
