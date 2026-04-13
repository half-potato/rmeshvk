// MRT primitive rendering: same vertex shader as primitive.wgsl,
// but fragment outputs to 4 MRT targets matching the volume interval pipeline.
//
// MRT outputs (all Rgba16Float):
//   location(0): plaster.rgb, alpha=1  (flat-lit color as plaster stand-in)
//   location(1): roughness, env_f0, env_f1, alpha=1
//   location(2): world_normal.xyz, alpha=1
//   location(3): albedo.rgb, alpha=1

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

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
}

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
) -> VsOut {
    let model = mat4x4<f32>(u.model_col0, u.model_col1, u.model_col2, u.model_col3);
    let vp = mat4x4<f32>(u.vp_col0, u.vp_col1, u.vp_col2, u.vp_col3);

    let world_pos = model * vec4<f32>(position, 1.0);

    let model3 = mat3x3<f32>(model[0].xyz, model[1].xyz, model[2].xyz);
    let wn = normalize(model3 * normal);

    var out: VsOut;
    out.pos = vp * world_pos;
    out.world_normal = wn;
    return out;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) aux0: vec4<f32>,
    @location(2) normals: vec4<f32>,
    @location(3) albedo: vec4<f32>,
}

@fragment
fn fs_main(in: VsOut) -> FragmentOutput {
    let n = normalize(in.world_normal);

    // Plaster stand-in: flat-lit color (deferred will re-light using the normal + albedo)
    let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let lit = 0.3 + 0.7 * max(dot(n, light_dir), 0.0);
    let plaster = u.color.rgb * lit;

    var out: FragmentOutput;
    out.color = vec4<f32>(plaster, 1.0);
    out.aux0 = vec4<f32>(0.5, 0.0, 0.0, 1.0);  // roughness=0.5, env_f0=0, env_f1=0
    out.normals = vec4<f32>(n, 1.0);              // world normal (deferred normalizes)
    out.albedo = vec4<f32>(u.color.rgb, 1.0);     // base albedo = primitive color
    return out;
}
