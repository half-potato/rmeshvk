// MRT primitive rendering: same vertex shader as primitive.wgsl,
// but fragment outputs to 4 MRT targets matching the volume interval pipeline.
//
// MRT outputs (all Rgba16Float):
//   location(0): albedo.rgb, alpha=1
//   location(1): roughness, env_f0, env_f1, alpha=1
//   location(2): world_normal.xyz, alpha=1
//   location(3): view_depth, 0, 0, alpha=1

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
    @location(1) view_depth: f32,
}

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
) -> VsOut {
    let model = mat4x4<f32>(u.model_col0, u.model_col1, u.model_col2, u.model_col3);
    let vp = mat4x4<f32>(u.vp_col0, u.vp_col1, u.vp_col2, u.vp_col3);

    let world_pos = model * vec4<f32>(position, 1.0);
    let clip = vp * world_pos;

    let model3 = mat3x3<f32>(model[0].xyz, model[1].xyz, model[2].xyz);
    let wn = normalize(model3 * normal);

    var out: VsOut;
    out.pos = clip;
    out.world_normal = wn;
    out.view_depth = clip.w;  // linear view-space Z
    return out;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) aux0: vec4<f32>,
    @location(2) normals: vec4<f32>,
    @location(3) expected_depth: vec4<f32>,
}

@fragment
fn fs_main(in: VsOut) -> FragmentOutput {
    let n = normalize(in.world_normal);

    var out: FragmentOutput;
    out.color = vec4<f32>(u.color.rgb, 1.0);      // albedo (was plaster)
    out.aux0 = vec4<f32>(0.5, 0.0, 0.0, 1.0);    // roughness=0.5, env_f0=0, env_f1=0
    out.normals = vec4<f32>(-n, 1.0);              // negated to match volume normal convention
    out.expected_depth = vec4<f32>(in.view_depth, 0.0, 0.0, 1.0); // linear view-space depth
    return out;
}
