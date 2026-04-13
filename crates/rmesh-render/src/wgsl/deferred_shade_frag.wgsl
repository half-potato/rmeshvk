// Deferred PBR shading — fullscreen triangle render pass.
//
// Reads MRT textures from the forward rasterization pass:
//   color_tex:   plaster.rgb * a, a
//   aux0_tex:    roughness * a, env_f0 * a, env_f1 * a, a
//   normals_tex: oct_normal.xy * a, pack(env_f2,env_f3) * a, a
//   albedo_tex:  albedo.rgb * a, a
// Plus hw depth buffer for world-position reconstruction.
//
// Phase A: Lambertian diffuse only (no neural specular, no shadows).

struct DeferredUniforms {
    inv_vp: mat4x4f,
    cam_pos: vec3f,
    num_lights: u32,
    width: u32,
    height: u32,
    ambient: f32,
    debug_mode: u32,
    near_plane: f32,
    far_plane: f32,
    _pad: vec2f,
}

struct Light {
    position: vec3f,
    light_type: u32,    // 0=point, 1=spot, 2=directional
    color: vec3f,
    intensity: f32,
    direction: vec3f,
    inner_angle: f32,
    outer_angle: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> uniforms: DeferredUniforms;
@group(0) @binding(1) var color_tex: texture_2d<f32>;     // plaster RGBA
@group(0) @binding(2) var aux0_tex: texture_2d<f32>;      // roughness, env_f0, env_f1, alpha
@group(0) @binding(3) var normals_tex: texture_2d<f32>;   // raw field gradient.xyz * alpha, alpha
@group(0) @binding(4) var albedo_tex: texture_2d<f32>;    // albedo.rgb, alpha
@group(0) @binding(5) var<storage, read> lights: array<Light>;
@group(0) @binding(6) var hw_depth_tex: texture_depth_2d; // hardware depth buffer
@group(0) @binding(7) var<storage, read> tone_curve: array<f32>; // [y_knots..., slope, intercept, intercept_bias]

// Debug mode constants
const DBG_FINAL:       u32 = 0u;
const DBG_RAW_ALBEDO:  u32 = 1u;
const DBG_TRUE_ALBEDO: u32 = 2u;
const DBG_NORMALS:     u32 = 3u;
const DBG_ROUGHNESS:   u32 = 4u;
const DBG_ENV_FEATURE: u32 = 5u;
const DBG_DEPTH:       u32 = 6u;
const DBG_SPECULAR:    u32 = 7u;
const DBG_DIFFUSE:     u32 = 8u;
const DBG_SHADOW:      u32 = 9u;
const DBG_RETRO:       u32 = 10u;
const DBG_LAMBDA:      u32 = 11u;
const DBG_PLASTER:     u32 = 12u;
const DBG_ALPHA:       u32 = 13u;

/// Evaluate the monotonic piecewise-linear tone curve loaded from the .rmesh file.
/// Layout: [y_knots..., slope, intercept, intercept_bias], n_knots = len - 3.
fn eval_tone_curve(x_in: f32) -> f32 {
    let total_len = arrayLength(&tone_curve);
    let n_knots = total_len - 3u;
    let x = clamp(x_in, 0.0, 1.0);

    // Piecewise linear interpolation over uniform x knots in [0, 1]
    let t = x * f32(n_knots - 1u);
    let bin = min(u32(t), n_knots - 2u);
    let alpha = t - f32(bin);
    let y = tone_curve[bin] + alpha * (tone_curve[bin + 1u] - tone_curve[bin]);

    let slope = tone_curve[total_len - 3u];
    let intercept = tone_curve[total_len - 2u];
    let intercept_bias = tone_curve[total_len - 1u];

    return max(y + slope * x + intercept + intercept_bias, 1e-3);
}

struct VsOut {
    @builtin(position) pos: vec4f,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    var out: VsOut;
    out.pos = vec4f(x, y, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
    let coords = vec2u(u32(frag_coord.x), u32(frag_coord.y));

    // Load MRT textures
    let color_raw = textureLoad(color_tex, coords, 0);
    let aux0_raw = textureLoad(aux0_tex, coords, 0);
    let normals_raw = textureLoad(normals_tex, coords, 0);
    let albedo_raw = textureLoad(albedo_tex, coords, 0);
    let hw_depth = textureLoad(hw_depth_tex, coords, 0);

    // Alpha from any target (all have alpha in .a via premul blend)
    let alpha = color_raw.a;

    if (alpha < 0.01) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }

    // Un-premultiply all channels
    let inv_alpha = 1.0 / max(alpha, 1e-6);
    let plaster = color_raw.rgb * inv_alpha;
    let roughness = aux0_raw.r * inv_alpha;
    let env_f0 = aux0_raw.g * inv_alpha;
    let env_f1 = aux0_raw.b * inv_alpha;

    // Un-premultiply and normalize raw field gradient to get surface normal.
    // Gradient was composited unnormalized across all tets — normalize here.
    let raw_gradient = normals_raw.rgb * inv_alpha;
    let normal = normalize(raw_gradient);

    // env_f2/f3 not stored in MRT (gradient uses all 3 channels). Set to 0 for now.
    let env_feat = vec4f(env_f0, env_f1, 0.0, 0.0);

    let raw_albedo = albedo_raw.rgb * inv_alpha;

    // Reconstruct depth from hardware depth buffer (NDC → linear view-space Z)
    let near = uniforms.near_plane;
    let far = uniforms.far_plane;
    let ndc_depth = hw_depth;
    let depth = near * far / (far - ndc_depth * (far - near));

    // Recover true albedo by applying calibrated tone curve to plaster luminance
    let plaster_lum = max(0.2126 * plaster.r + 0.7152 * plaster.g + 0.0722 * plaster.b, 1e-3);
    let calibrated_lighting = eval_tone_curve(plaster_lum);
    let albedo = raw_albedo / calibrated_lighting;

    // Reconstruct world position from hw depth + inverse VP
    let ndc_x = (frag_coord.x + 0.5) / f32(uniforms.width) * 2.0 - 1.0;
    let ndc_y = 1.0 - (frag_coord.y + 0.5) / f32(uniforms.height) * 2.0;
    let clip_pos = vec4f(ndc_x, ndc_y, ndc_depth, 1.0);
    let world_h = uniforms.inv_vp * clip_pos;
    let world_pos = world_h.xyz / world_h.w;

    // Accumulate lighting
    var total_contribution = vec3f(0.0);

    for (var li = 0u; li < uniforms.num_lights; li++) {
        let light = lights[li];
        var to_light: vec3f;
        var atten: f32;

        if (light.light_type == 2u) {
            to_light = normalize(-light.direction);
            atten = 1.0;
        } else {
            let to_light_raw = light.position - world_pos;
            let dist = max(length(to_light_raw), 1e-6);
            to_light = to_light_raw / dist;
            atten = 1.0 / (dist * dist);
        }

        if (light.light_type == 1u) {
            let l_fwd = normalize(light.direction);
            let cos_a = dot(-to_light, l_fwd);
            let inner_cos = cos(light.inner_angle);
            let outer_cos = cos(light.outer_angle);
            let spot = clamp((cos_a - outer_cos) / (inner_cos - outer_cos + 1e-8), 0.0, 1.0);
            atten *= spot;
        }

        let NdotL = max(dot(normal, to_light), 0.0);
        let l_color = light.color * light.intensity;
        total_contribution += albedo * NdotL * atten * l_color;
    }

    var final_color = uniforms.ambient * albedo + total_contribution;

    // Debug mode overrides
    let dm = uniforms.debug_mode;
    if (dm == DBG_RAW_ALBEDO)       { final_color = raw_albedo; }
    else if (dm == DBG_TRUE_ALBEDO) { final_color = albedo; }
    else if (dm == DBG_NORMALS)     { final_color = normal * 0.5 + 0.5; }
    else if (dm == DBG_ROUGHNESS)   { final_color = vec3f(roughness); }
    else if (dm == DBG_ENV_FEATURE) { final_color = env_feat.xyz * 0.5 + 0.5; }
    else if (dm == DBG_DEPTH)       { final_color = vec3f(depth * 0.1); }
    else if (dm == DBG_SPECULAR)    { final_color = vec3f(0.0); }
    else if (dm == DBG_DIFFUSE)     { final_color = total_contribution; }
    else if (dm == DBG_SHADOW)      { final_color = vec3f(0.0); }
    else if (dm == DBG_RETRO)       { final_color = vec3f(0.0); }
    else if (dm == DBG_LAMBDA)      { final_color = vec3f(0.0); }
    else if (dm == DBG_PLASTER)     { final_color = plaster; }
    else if (dm == DBG_ALPHA)       { final_color = vec3f(alpha); }

    return vec4f(max(final_color, vec3f(0.0)), alpha);
}
