// DSM primitive rendering: projects primitives from light viewpoint,
// outputs Fourier coefficients for full opaque occlusion (alpha=1) at the
// primitive's NDC depth. 3 MRT outputs matching dsm_fourier_fragment.wgsl.

struct PrimitiveUniforms {
    vp_col0: vec4<f32>,
    vp_col1: vec4<f32>,
    vp_col2: vec4<f32>,
    vp_col3: vec4<f32>,
    model_col0: vec4<f32>,
    model_col1: vec4<f32>,
    model_col2: vec4<f32>,
    model_col3: vec4<f32>,
    color: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: PrimitiveUniforms;

const TWO_PI: f32 = 6.28318530717958647692;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
}

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
) -> VsOut {
    let model = mat4x4<f32>(u.model_col0, u.model_col1, u.model_col2, u.model_col3);
    let vp = mat4x4<f32>(u.vp_col0, u.vp_col1, u.vp_col2, u.vp_col3);
    var out: VsOut;
    out.pos = vp * model * vec4<f32>(position, 1.0);
    return out;
}

struct FourierOutput {
    @location(0) rt0: vec4<f32>,
    @location(1) rt1: vec4<f32>,
    @location(2) rt2: vec4<f32>,
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> FourierOutput {
    // Convert NDC depth to linear view-space Z, normalize to [0,1]
    let near = u.color.x;
    let far = u.color.y;
    let z_linear = near * far / (far - frag_coord.z * (far - near));
    let z = clamp((z_linear - near) / (far - near), 0.0, 1.0);

    // Opaque surface: depth and depth² with alpha=1
    var out: FourierOutput;
    out.rt0 = vec4<f32>(z, z, z, 1.0);
    out.rt1 = vec4<f32>(z * z, z * z, z * z, 1.0);
    out.rt2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    return out;
}
