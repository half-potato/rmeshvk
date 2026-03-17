// Upwind advection of velocity and density fields.

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
@group(0) @binding(4) var<storage, read> tet_centers: array<vec4<f32>>;

@group(1) @binding(0) var<storage, read> velocity: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> density: array<f32>;
@group(1) @binding(2) var<storage, read_write> velocity_tmp: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> density_tmp: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.tet_count) {
        return;
    }

    let vol = tet_volumes[tid];
    let vel_i = velocity[tid].xyz;
    let den_i = density[tid];

    var d_vel = vec3<f32>(0.0);
    var d_den = 0.0;

    for (var f = 0u; f < 4u; f++) {
        let neighbor = tet_neighbors[tid * 4u + f];
        if (neighbor < 0) {
            continue; // boundary: zero flux
        }

        let fg = face_geo[tid * 4u + f]; // (nx*Af, ny*Af, nz*Af, Af/d_ij)
        let n_area = fg.xyz; // area-weighted outward normal

        let vel_j = velocity[u32(neighbor)].xyz;
        let den_j = density[u32(neighbor)];

        // Face velocity (average)
        let vel_face = 0.5 * (vel_i + vel_j);

        // Flux through face: dot(u_face, n*A_f)
        let flux = dot(vel_face, n_area);

        // Upwind: if flux > 0, flow goes outward (use tet i values)
        //         if flux < 0, flow comes inward (use neighbor j values)
        var vel_source: vec3<f32>;
        var den_source: f32;
        if (flux > 0.0) {
            vel_source = vel_i;
            den_source = den_i;
        } else {
            vel_source = vel_j;
            den_source = den_j;
        }

        // Accumulate: outward flux removes from cell, inward flux adds
        d_vel -= flux * vel_source;
        d_den -= flux * den_source;
    }

    let dt_over_vol = uniforms.dt / max(vol, 1e-10);
    var new_vel = vel_i + dt_over_vol * d_vel;

    // Clamp velocity to prevent numerical explosion
    let speed = length(new_vel);
    if (speed > 50.0) {
        new_vel = new_vel * (50.0 / speed);
    }

    velocity_tmp[tid] = vec4<f32>(new_vel, 0.0);
    density_tmp[tid] = max(den_i + dt_over_vol * d_den, 0.0);
}
