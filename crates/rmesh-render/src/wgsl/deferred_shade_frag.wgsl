// Deferred PBR shading — fullscreen triangle render pass.
//
// Reads MRT textures from the forward rasterization pass:
//   color_tex:   albedo.rgb * a, a
//   aux0_tex:    roughness * a, metallic * a, f0_dielectric * a, a
//   normals_tex: gradient.xyz * a, a
//   depth_tex:   expected_depth * a, retro * a, expected_z2 * a, a
// Plus hw depth buffer for world-position reconstruction and a precomputed
// AO factor (R8Unorm) from the GTAO pass.
// Group 1: cached Fourier DSM shadow atlas for transmittance lookup.
//
// Material model: GGX D + Smith G (height-correlated) + Schlick Fresnel
// metallic workflow, plus a back-Lambertian retro lobe for retroreflective
// surfaces. Output is linear color (Rgba16Float); a downstream blit applies
// linear → sRGB encoding, so this shader does NOT gamma-correct.

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
    dsm_enabled: u32,
    exposure: f32,
    sky_color: vec3f,
    ao_strength: f32,
    ground_color: vec3f,
    ssgi_strength: f32,
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

struct ShadowLight {
    vp0: mat4x4f,
    vp1: mat4x4f,
    vp2: mat4x4f,
    vp3: mat4x4f,
    vp4: mat4x4f,
    vp5: mat4x4f,
    face_offset: u32,
    face_count: u32,
    near: f32,
    far: f32,
    light_type: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// Group 0: MRT textures + lights + AO
@group(0) @binding(0) var<uniform> uniforms: DeferredUniforms;
@group(0) @binding(1) var color_tex: texture_2d<f32>;     // albedo RGBA (premul alpha)
@group(0) @binding(2) var aux0_tex: texture_2d<f32>;      // roughness, metallic, f0_dielectric, alpha
@group(0) @binding(3) var normals_tex: texture_2d<f32>;   // raw field gradient.xyz * alpha, alpha
@group(0) @binding(4) var depth_tex: texture_2d<f32>;     // expected depth, retro, 0, alpha
@group(0) @binding(5) var<storage, read> lights: array<Light>;
@group(0) @binding(6) var hw_depth_tex: texture_depth_2d; // hardware depth buffer
@group(0) @binding(7) var ao_tex: texture_2d<f32>;        // R8Unorm GTAO output
@group(0) @binding(8) var ssgi_tex: texture_2d<f32>;      // Rgba16Float SSGI radiance
@group(0) @binding(9) var lit_history_tex: texture_2d<f32>; // Rgba16Float, prev frame's deferred output

// Group 1: DSM shadow cubemap
@group(1) @binding(0) var dsm_rt0: texture_cube<f32>;
@group(1) @binding(1) var dsm_rt1: texture_cube<f32>;
@group(1) @binding(2) var dsm_rt2: texture_cube<f32>;
@group(1) @binding(3) var<storage, read> shadow_meta: array<ShadowLight>;
@group(1) @binding(4) var dsm_sampler: sampler;

// Debug mode constants
const DBG_FINAL:       u32 = 0u;
const DBG_RAW_ALBEDO:  u32 = 1u;
const DBG_TRUE_ALBEDO: u32 = 2u;
const DBG_NORMALS:     u32 = 3u;
const DBG_ROUGHNESS:   u32 = 4u;
const DBG_PBR:         u32 = 5u;  // metallic / f0_dielectric / retro packed RGB
const DBG_DEPTH:       u32 = 6u;
const DBG_SPECULAR:    u32 = 7u;
const DBG_DIFFUSE:     u32 = 8u;
const DBG_SHADOW:      u32 = 9u;
const DBG_RETRO:       u32 = 10u;
const DBG_LAMBDA:      u32 = 11u;
const DBG_PLASTER:     u32 = 12u;
const DBG_ALPHA:       u32 = 13u;
const DBG_PRIMITIVES:  u32 = 14u;
const DBG_DSM:         u32 = 15u;
const DBG_AO:          u32 = 16u;
const DBG_SSGI:        u32 = 17u;
const DBG_LIT_HISTORY: u32 = 18u;

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;

// ---------------------------------------------------------------------------
// PBR helpers
// ---------------------------------------------------------------------------

fn fresnel_schlick(cos_theta: f32, F0: vec3f) -> vec3f {
    let c = clamp(1.0 - cos_theta, 0.0, 1.0);
    let c5 = (c * c) * (c * c) * c;
    return F0 + (vec3f(1.0) - F0) * c5;
}

fn d_ggx(n_dot_h: f32, a: f32) -> f32 {
    let a2 = a * a;
    let n_h = clamp(n_dot_h, 0.0, 1.0);
    let denom = n_h * n_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + 1e-7);
}

fn v_smith_ggx_correlated(n_dot_v: f32, n_dot_l: f32, a: f32) -> f32 {
    let a2 = a * a;
    let gv = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    let gl = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    return 0.5 / (gv + gl + 1e-5);
}

fn aces_narkowicz(x: vec3f) -> vec3f {
    return clamp(
        (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
        vec3f(0.0), vec3f(1.0),
    );
}

// ---------------------------------------------------------------------------
// DSM shadow helpers
// ---------------------------------------------------------------------------

fn get_shadow_vp(sm: ShadowLight, face: u32) -> mat4x4f {
    switch face {
        case 0u: { return sm.vp0; }
        case 1u: { return sm.vp1; }
        case 2u: { return sm.vp2; }
        case 3u: { return sm.vp3; }
        case 4u: { return sm.vp4; }
        default: { return sm.vp5; }
    }
}

/// Select cubemap face from a direction vector (world-space, light → surface).
fn select_cubemap_face(dir: vec3f) -> u32 {
    let a = abs(dir);
    if a.x >= a.y && a.x >= a.z {
        return select(1u, 0u, dir.x > 0.0);
    } else if a.y >= a.z {
        return select(3u, 2u, dir.y > 0.0);
    } else {
        return select(5u, 4u, dir.z > 0.0);
    }
}

/// Evaluate transmittance T(world_pos) using variance shadow map (Chebyshev bound).
fn evaluate_transmittance(world_pos: vec3f, li: u32, NdotL: f32) -> f32 {
    let sm = shadow_meta[li];

    let dir = world_pos - lights[li].position;
    let dist = length(dir);
    if dist < 1e-6 { return 1.0; }

    let face = select_cubemap_face(dir);
    let vp = get_shadow_vp(sm, face);
    let clip = vp * vec4f(world_pos, 1.0);
    if clip.w <= 0.0 { return 1.0; }

    let c0 = textureSample(dsm_rt0, dsm_sampler, dir);
    let c1 = textureSample(dsm_rt1, dsm_sampler, dir);
    let shadow_alpha = c0.a;
    if shadow_alpha < 0.01 { return 1.0; }

    let inv_alpha = 1.0 / shadow_alpha;
    let mean = c0.r * inv_alpha;        // E[z]
    let mean_sq = c1.r * inv_alpha;     // E[z²]

    let z = (clip.w - sm.near) / (sm.far - sm.near);
    if z <= mean { return 1.0; }

    let variance = max(mean_sq - mean * mean, 3e-5);
    let d = z - mean;
    let p_max = variance / (variance + d * d);

    let T_total = 1.0 - shadow_alpha;
    return max(p_max, T_total);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

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

// Two outputs: the display target (where debug-mode visualizations land) and
// the lit_history target (the true lit color, never overridden by debug
// modes). Splitting them keeps the SSGI feedback chain valid even when the
// user is staring at a debug view.
struct DeferredOut {
    @location(0) display: vec4f,
    @location(1) lit:     vec4f,
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4f) -> DeferredOut {
    let coords = vec2u(u32(frag_coord.x), u32(frag_coord.y));

    // Load MRT textures
    let color_raw = textureLoad(color_tex, coords, 0);      // albedo (premul)
    let aux0_raw = textureLoad(aux0_tex, coords, 0);
    let normals_raw = textureLoad(normals_tex, coords, 0);
    let depth_raw = textureLoad(depth_tex, coords, 0);       // expected depth + retro
    let hw_depth = textureLoad(hw_depth_tex, coords, 0);
    let ao = textureLoad(ao_tex, coords, 0).r;
    let ssgi_indirect = textureLoad(ssgi_tex, coords, 0).rgb;

    let alpha = color_raw.a;
    if (alpha < 0.01) {
        return DeferredOut(vec4f(0.0, 0.0, 0.0, 0.0), vec4f(0.0, 0.0, 0.0, 0.0));
    }

    // Un-premultiply
    let inv_alpha = 1.0 / max(alpha, 1e-6);
    let albedo        = color_raw.rgb * inv_alpha;
    let roughness     = aux0_raw.r * inv_alpha;
    let metallic      = aux0_raw.g * inv_alpha;
    let f0_dielectric = aux0_raw.b * inv_alpha;

    // Un-premultiply and normalize raw field gradient to get surface normal.
    // Transform from colmap (Y-down, Z-forward) to wgpu (Y-up, Z-backward).
    let raw_gradient = normals_raw.rgb * inv_alpha;
    let normal = -normalize(vec3f(raw_gradient.x, raw_gradient.y, raw_gradient.z));

    // Volume's expected termination depth + retro
    let z_expected = depth_raw.r * inv_alpha;
    let retro      = depth_raw.g * inv_alpha;

    let near = uniforms.near_plane;
    let far  = uniforms.far_plane;

    // Use hw depth where an opaque primitive wrote to it (hw_depth < 1.0),
    // otherwise use the volume's expected termination depth.
    var z_final = max(z_expected, near);
    if hw_depth < 0.999 {
        let z_hw = near * far / (far - hw_depth * (far - near));
        z_final = min(z_final, z_hw);
    }

    // Reconstruct world position from depth + inverse VP
    let ndc_z = (far * (z_final - near)) / (z_final * (far - near));
    let ndc_x = frag_coord.x / f32(uniforms.width) * 2.0 - 1.0;
    let ndc_y = 1.0 - frag_coord.y / f32(uniforms.height) * 2.0;
    let clip_pos = vec4f(ndc_x, ndc_y, ndc_z, 1.0);
    let world_h = uniforms.inv_vp * clip_pos;
    let world_pos = world_h.xyz / world_h.w;

    // Per-pixel constants for the BRDF (parametric metallic workflow).
    let V = normalize(uniforms.cam_pos - world_pos);
    let N = normal;
    let NoV = max(dot(N, V), 1e-4);
    // Match pbr_bsdf.PBRBsdf.forward: roughness clamped to [min_roughness, 1].
    let r_clamped = clamp(roughness, 0.08, 1.0);
    let alpha_g = r_clamped * r_clamped;
    let F0 = mix(vec3f(f0_dielectric), albedo, metallic);

    // Energy-consistent diffuse scale — matches `diffuse_scaling(metallic, F0, NdotV)`
    // from pbr_bsdf.py: view-angle Fresnel, evaluated once per pixel.
    let F_view = fresnel_schlick(NoV, F0);
    let kd = (vec3f(1.0) - F_view) * (1.0 - metallic);
    // Note: no 1/π — matches pbr_renderer.render_relit (parametric branch).
    let diff_base = kd * albedo;

    // Accumulate per-light direct lighting
    var total_diffuse  = vec3f(0.0);
    var total_specular = vec3f(0.0);

    for (var li = 0u; li < uniforms.num_lights; li = li + 1u) {
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

        let L = to_light;
        let NoL = max(dot(N, L), 0.0);
        if (NoL <= 0.0) { continue; }

        let H = normalize(L + V);
        let NoH = max(dot(N, H), 0.0);
        let LoH = max(dot(L, H), 0.0);

        // Specular: full GGX D × Smith V × Schlick F at the half-angle.
        let F = fresnel_schlick(LoH, F0);
        let D = d_ggx(NoH, alpha_g);
        let Vis = v_smith_ggx_correlated(NoV, NoL, alpha_g);
        let spec = D * Vis * F;

        var T = 1.0;
        if (uniforms.dsm_enabled != 0u) {
            T = evaluate_transmittance(world_pos, li, 1.0);
        }

        // Plain Lambertian NoL — retro is computed/exposed via DBG_RETRO but
        // not applied to lighting (matches pbr_renderer.render_relit).
        let l_color = light.color * light.intensity;
        let energy = T * atten;
        total_diffuse  += diff_base * NoL * energy * l_color;
        total_specular += spec      * NoL * energy * l_color;
    }
    // Hemispherical ambient × AO. Energy split mirrors the per-light path:
    // diffuse ambient is gated by `kd` (so metals get no ambient diffuse),
    // and a cheap ambient specular (`F_view·hemi`) gives metals a tinted
    // reflection from the sky/ground hemisphere when no env map is around.
    let hemi = mix(uniforms.ground_color, uniforms.sky_color, normal.y * 0.5 + 0.5);
    let ao_factor = mix(1.0, ao, uniforms.ao_strength);
    // Diffuse ambient: lerp from constant hemi-stand-in toward SSGI's actual
    // gathered indirect radiance based on `ssgi_strength`. SSGI rays already
    // multi-bounce via prior-frame feedback; here we still need `kd · albedo`
    // to apply the receiving surface's BRDF (the receiver hadn't been mixed
    // in yet — the SSGI buffer holds incoming radiance L_in).
    let ambient_diffuse_const = kd * albedo * hemi;
    let ambient_diffuse_ssgi  = kd * albedo * ssgi_indirect;
    let ambient_diffuse = mix(ambient_diffuse_const, ambient_diffuse_ssgi, uniforms.ssgi_strength);
    // Specular ambient: same shape as the diffuse path, so metals (which have
    // kd≈0 and rely entirely on F_view·env) actually reflect colored SSGI
    // radiance from neighbors instead of just the constant sky/ground gradient.
    // Approximation: SSGI is a hemisphere-averaged radiance, not the reflection
    // direction — strictly correct only for rough materials. Mirrors want SSR.
    let ambient_specular_const = F_view * hemi;
    let ambient_specular_ssgi  = F_view * ssgi_indirect;
    let ambient_specular = mix(ambient_specular_const, ambient_specular_ssgi, uniforms.ssgi_strength);
    let ambient_term = uniforms.ambient * (ambient_diffuse + ambient_specular) * ao_factor;

    var lit = total_diffuse + total_specular + ambient_term;
    var final_color = aces_narkowicz(lit * uniforms.exposure);

    // Snapshot the true lit value for the lit_history feedback — this is what
    // SSGI samples next frame. Must be taken BEFORE any debug-mode override
    // below; otherwise the SSGI feedback chain gets poisoned with debug viz.
    let lit_history_out = vec4f(max(final_color, vec3f(0.0)), alpha);

    // Debug mode overrides — bypass tonemap for visualizations
    let dm = uniforms.debug_mode;
    if (dm == DBG_RAW_ALBEDO)       { final_color = albedo; }
    else if (dm == DBG_TRUE_ALBEDO) { final_color = albedo; }
    else if (dm == DBG_NORMALS)     { final_color = normal * 0.5 + 0.5; }
    else if (dm == DBG_ROUGHNESS)   { final_color = vec3f(roughness); }
    else if (dm == DBG_PBR)         { final_color = vec3f(roughness, metallic, f0_dielectric); }
    else if (dm == DBG_DEPTH)       { final_color = vec3f(z_expected * 0.1); }
    else if (dm == DBG_SPECULAR)    { final_color = aces_narkowicz(total_specular * uniforms.exposure); }
    else if (dm == DBG_DIFFUSE)     { final_color = aces_narkowicz((total_diffuse + ambient_term) * uniforms.exposure); }
    else if (dm == DBG_SHADOW) {
        var T_total = 0.0;
        if uniforms.dsm_enabled != 0u {
            for (var li = 0u; li < uniforms.num_lights; li = li + 1u) {
                T_total += evaluate_transmittance(world_pos, li, 1.0);
            }
            final_color = vec3f(T_total / max(f32(uniforms.num_lights), 1.0));
        } else {
            final_color = vec3f(1.0);
        }
    }
    else if (dm == DBG_RETRO)       { final_color = vec3f(retro); }
    else if (dm == DBG_LAMBDA) {
        // Reused: UV coords in light-space (first light) for debugging shadow projection.
        if uniforms.dsm_enabled != 0u && uniforms.num_lights > 0u {
            let sm = shadow_meta[0u];
            let to_surface = world_pos - lights[0u].position;
            var face = 0u;
            if sm.light_type == 0u {
                face = select_cubemap_face(to_surface);
            }
            let vp = get_shadow_vp(sm, face);
            let clip = vp * vec4f(world_pos, 1.0);
            if clip.w > 0.0 {
                let ndc_xy = clip.xy / clip.w;
                let uv = vec2f(ndc_xy.x * 0.5 + 0.5, 0.5 - ndc_xy.y * 0.5);
                final_color = vec3f(uv.x, uv.y, 0.0);
            } else {
                final_color = vec3f(1.0, 0.0, 1.0);
            }
        } else {
            final_color = vec3f(0.5);
        }
    }
    else if (dm == DBG_PLASTER)     { final_color = albedo; }
    else if (dm == DBG_ALPHA)       { final_color = vec3f(alpha); }
    else if (dm == DBG_PRIMITIVES)  { /* final_color already correct — volume pass skipped on CPU */ }
    else if (dm == DBG_AO)          { final_color = vec3f(ao); }
    else if (dm == DBG_SSGI)        { final_color = ssgi_indirect; }
    else if (dm == DBG_LIT_HISTORY) { final_color = textureLoad(lit_history_tex, coords, 0).rgb; }

    let display_out = vec4f(max(final_color, vec3f(0.0)), alpha);
    return DeferredOut(display_out, lit_history_out);
}
