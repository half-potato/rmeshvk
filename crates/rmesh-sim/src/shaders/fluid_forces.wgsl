// Apply external forces: gravity, buoyancy, and smoke source injection.

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
@group(0) @binding(4) var<storage, read> tet_centers: array<vec4<f32>>;

@group(1) @binding(0) var<storage, read_write> velocity: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> density: array<f32>;

const MAX_VELOCITY: f32 = 50.0;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.tet_count) {
        return;
    }

    var vel = velocity[tid].xyz;
    var den = density[tid];
    let dt = uniforms.dt;

    // Only apply body forces where there is density (smoke).
    // Applying gravity everywhere creates huge boundary divergence
    // that the pressure solver can't handle.
    let grav_len = length(uniforms.gravity.xyz);

    if (den > 1e-6) {
        // Gravity (proportional to density — heavier smoke sinks)
        vel += dt * uniforms.gravity.xyz * den;

        // Buoyancy: push density upward (opposite to gravity direction)
        if (grav_len > 0.0) {
            let up_dir = -uniforms.gravity.xyz / grav_len;
            vel += dt * uniforms.buoyancy * den * up_dir;
        }
    }

    // Smoke source injection
    let center = tet_centers[tid].xyz;
    let dist = length(center - uniforms.source_pos.xyz);
    let radius = uniforms.source_pos.w;
    if (dist < radius) {
        den += dt * uniforms.source_strength;
        // Add upward push at source
        if (grav_len > 0.0) {
            let up_dir = -uniforms.gravity.xyz / grav_len;
            vel += dt * uniforms.source_strength * up_dir;
        }
    }

    // Clamp velocity to prevent numerical explosion
    let speed = length(vel);
    if (speed > MAX_VELOCITY) {
        vel = vel * (MAX_VELOCITY / speed);
    }

    velocity[tid] = vec4<f32>(vel, 0.0);
    density[tid] = max(den, 0.0);
}
