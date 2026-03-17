// Pressure gradient subtraction to enforce incompressibility.
// v = v - ∇p

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

@group(1) @binding(0) var<storage, read_write> velocity: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> pressure: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.tet_count) {
        return;
    }

    let p_i = pressure[tid];
    var grad_p = vec3<f32>(0.0);

    for (var f = 0u; f < 4u; f++) {
        let neighbor = tet_neighbors[tid * 4u + f];
        if (neighbor < 0) {
            continue;
        }

        let n_area = face_geo[tid * 4u + f].xyz; // area-weighted outward normal
        let p_j = pressure[u32(neighbor)];

        // Face pressure (average)
        let p_face = 0.5 * (p_i + p_j);

        // Discrete gradient: (1/V) * Σ p_face * n*A_f
        grad_p += p_face * n_area;
    }

    let vol = max(tet_volumes[tid], 1e-10);
    var new_vel = velocity[tid].xyz - grad_p / vol;

    // Clamp velocity to prevent numerical explosion
    let speed = length(new_vel);
    if (speed > 50.0) {
        new_vel = new_vel * (50.0 / speed);
    }

    velocity[tid] = vec4<f32>(new_vel, 0.0);
}
