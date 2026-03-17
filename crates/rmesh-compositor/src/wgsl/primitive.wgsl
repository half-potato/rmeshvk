// Primitive rendering: projects unit-mesh vertices through model + VP matrices,
// applies flat directional lighting in the fragment shader.

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

    // Transform normal by upper-left 3x3 of model matrix
    let model3 = mat3x3<f32>(model[0].xyz, model[1].xyz, model[2].xyz);
    let wn = normalize(model3 * normal);

    var out: VsOut;
    out.pos = vp * world_pos;
    out.world_normal = wn;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let lit = 0.3 + 0.7 * max(dot(n, light_dir), 0.0);
    return vec4<f32>(u.color.rgb * lit, 1.0);
}
