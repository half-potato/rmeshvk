// Map simulation state (density, velocity) to rendering buffers (opacity, color).

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

@group(1) @binding(0) var<storage, read> density: array<f32>;
@group(1) @binding(1) var<storage, read> velocity: array<vec4<f32>>;

@group(2) @binding(0) var<storage, read_write> out_densities: array<f32>;
@group(2) @binding(1) var<storage, read_write> out_colors: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.tet_count) {
        return;
    }

    let fluid_den = density[tid];

    // Only modify tets that have smoke — leave scene untouched otherwise
    if (fluid_den > 1e-6) {
        // Add fluid density on top of existing scene density
        out_densities[tid] = out_densities[tid] + fluid_den * uniforms.density_scale;

        // Velocity magnitude → heat colormap (white-ish to orange)
        let speed = length(velocity[tid].xyz);
        let t = clamp(speed * 5.0, 0.0, 1.0);
        let color = mix(vec3<f32>(0.8, 0.8, 0.9), vec3<f32>(1.0, 0.3, 0.0), t);

        out_colors[tid * 3u + 0u] = color.x;
        out_colors[tid * 3u + 1u] = color.y;
        out_colors[tid * 3u + 2u] = color.z;
    }
}
