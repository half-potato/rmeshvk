// Deferred PBR shading — fullscreen triangle render pass.
//
// Reads MRT textures from the forward rasterization pass:
//   color_tex:   albedo.rgb * a, a
//   aux0_tex:    roughness * a, metallic * a, f0_dielectric * a, a
//   normals_tex: gradient.xyz * a, a
//   depth_tex:   expected_depth * a, 0, expected_z2 * a, a
// Plus hw depth buffer for world-position reconstruction and a precomputed
// AO factor (R8Unorm) from the GTAO pass.
// Group 1: cached Fourier DSM shadow atlas for transmittance lookup.
//
// Material model: GGX D + Smith G (height-correlated) + Schlick Fresnel
// metallic workflow. Output is linear color (Rgba16Float); a downstream blit
// applies linear → sRGB encoding, so this shader does NOT gamma-correct.

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
@group(0) @binding(4) var depth_tex: texture_2d<f32>;     // expected depth, 0, expected_z2, alpha
@group(0) @binding(5) var<storage, read> lights: array<Light>;
@group(0) @binding(6) var hw_depth_tex: texture_depth_2d; // hardware depth buffer
@group(0) @binding(7) var ao_tex: texture_2d<f32>;        // R8Unorm GTAO output
@group(0) @binding(8) var ssgi_tex: texture_2d<f32>;      // Rgba16Float SSGI radiance
@group(0) @binding(9) var lit_history_tex: texture_2d<f32>; // Rgba16Float, prev frame's deferred output
@group(0) @binding(10) var ssr_tex: texture_2d<f32>;      // Rgba16Float SSR radiance (reflection direction)

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
const DBG_PBR:         u32 = 5u;  // roughness / metallic / f0_dielectric packed RGB
const DBG_DEPTH:       u32 = 6u;
const DBG_SPECULAR:    u32 = 7u;
const DBG_DIFFUSE:     u32 = 8u;
const DBG_SHADOW:      u32 = 9u;
const DBG_LAMBDA:      u32 = 10u;
const DBG_PLASTER:     u32 = 11u;
const DBG_ALPHA:       u32 = 12u;
const DBG_PRIMITIVES:  u32 = 13u;
const DBG_DSM:         u32 = 14u;
const DBG_AO:          u32 = 15u;
const DBG_SSGI:        u32 = 16u;
const DBG_LIT_HISTORY: u32 = 17u;
const DBG_SSR:         u32 = 18u;
const DBG_DEPTH_STD:   u32 = 19u;

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
    let a2 = a;// * a;
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

// Lazarov 2013 / Karis fit — closed-form replacement for the 2D BRDF LUT.
// Returns the energy-conserving Fresnel factor for ambient/IBL specular as
// `F0 * scale + F90 * bias`, where scale and bias account for the GGX lobe
// shape over (NoV, roughness). Use in place of `F_view` for ambient specular.
fn env_brdf_approx(F0: vec3f, roughness: f32, n_dot_v: f32) -> vec3f {
    let c0 = vec4f(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4f( 1.0,  0.0425,  1.040, -0.040);
    let r = c0 * roughness + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
    let env = vec2f(-1.04, 1.04) * a004 + r.zw;
    return F0 * env.x + vec3f(env.y);
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

/// Evaluate transmittance T(world_pos) by fitting the α-weighted light-side
/// termination distribution as Exp(λ=1/σ_l) anchored at z₀ = μ_l − σ_l (mean
/// and variance of this exponential match the stored moments). Closed-form CDF:
///   T(z) = (1 − α_total) + α_total · exp(−(z − z₀)/σ_l),  z ≥ z₀
///   T(z) = 1,                                              z < z₀
/// Symmetric with the view-side exponential model used for quadrature; replaces
/// the distribution-free Chebyshev/Cantelli bound with the actual exponential
/// transmittance. The (1 − α_total) floor is built in naturally.
///
/// Receiver-side bias: a lit surface sits inside its own absorber distribution
/// (μ_l ≈ surface depth, σ_l ≈ surface thickness), so without bias z ≈ μ_l and
/// decay → e⁻¹ even when no occluders precede the receiver. Shift z toward the
/// light by σ_l so the receiver lands at z₀ and reads T = 1. Not modulated by
/// NdotL: the Lambertian factor is already applied by the caller, so doing it
/// here too produces an NdotL² falloff that's visible in volumetric regions
/// where the field gradient is a poor surface-normal proxy.
fn evaluate_transmittance(world_pos: vec3f, li: u32) -> f32 {
    let sm = shadow_meta[li];

    let dir = world_pos - lights[li].position;
    let dist = length(dir);
    // if dist < 1e-6 { return 1.0; }

    let face = select_cubemap_face(dir);
    let vp = get_shadow_vp(sm, face);
    let clip = vp * vec4f(world_pos, 1.0);
    // if clip.w <= 0.0 { return 1.0; }

    let c0 = textureSample(dsm_rt0, dsm_sampler, dir);
    let c1 = textureSample(dsm_rt1, dsm_sampler, dir);
    let shadow_alpha = c0.a;
    if shadow_alpha < 0.01 { return 1.0; }

    let inv_alpha = 1.0 / shadow_alpha;
    let mean = c0.r * inv_alpha;        // E[z]
    let mean_sq = c1.r * inv_alpha;     // E[z²]

    let z = (clip.w - sm.near) / (sm.far - sm.near);

    let variance = max(mean_sq - mean * mean, 3e-5);
    let sigma = sqrt(variance);
    let z0 = mean - sigma;

    let z_biased = z - sigma;
    if z_biased <= z0 { return 1.0; }
    let decay = exp(-(z_biased - z0) / sigma);
    return (1.0 - shadow_alpha) + shadow_alpha * decay;
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
    let depth_raw = textureLoad(depth_tex, coords, 0);       // expected depth + E[z²]
    let hw_depth = textureLoad(hw_depth_tex, coords, 0);
    let ao = textureLoad(ao_tex, coords, 0).r;
    let ssgi_indirect = textureLoad(ssgi_tex, coords, 0).rgb;
    let ssr_radiance  = textureLoad(ssr_tex, coords, 0).rgb;

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

    // Volume's expected termination depth. Slot 3 is filtered ("thick tets
    // only") at the producer, so its α lags color's α — normalize by slot 3's
    // own α, not color's. When slot 3 has no thick-tet coverage at this pixel
    // (depth_alpha < 0.01), the volume contributes no usable depth and we
    // fall back to hw_depth (or far) for world-pos reconstruction.
    let depth_alpha = depth_raw.a;
    let inv_da = 1.0 / max(depth_alpha, 1e-6);
    let z_expected = depth_raw.r * inv_da;

    // Shadow-integration samples along the view ray. Model the alpha-weighted
    // depth PDF as Exp(λ) anchored at z_expected − σ with scale s = σ, so the
    // mean still matches z_expected. 2-point Gauss–Laguerre quadrature places
    // samples at z₀ + s·{2−√2, 2+√2} with weights {(2+√2)/4, (2−√2)/4} ≈
    // {0.854, 0.146}. The asymmetric weights cap floater damage at ~15% even
    // when the far sample lands in empty space.
    // Only meaningful when slot 3 has thick-tet coverage; otherwise fall back to
    // single-sample at world_pos (which came from hw_depth).
    let z_var = max(depth_raw.b - depth_raw.r * depth_raw.r, 0.0);
    // Clamp σ_v to a fraction of z_expected. At silhouette edges σ_v inflates
    // because the alpha-weighted depth distribution captures the volume's
    // thinness at the edge (not real depth spread), which would otherwise push
    // the far sample z_b = z_expected + 2.414·σ_v deep into empty space behind
    // the volume — where the DSM returns T ≈ 1 and the 0.146-weighted sample
    // brightens the edge. Interior pixels have σ_v ≪ z_expected so the clamp
    // is inactive there.
    let z_sigma = min(sqrt(z_var), 0.0 * z_expected);

    let near = uniforms.near_plane;
    let far  = uniforms.far_plane;

    // Pick the closer of volume-depth and hw-depth, but only consider the
    // volume-depth when slot 3 actually has thick-tet coverage.
    var z_final: f32;
    if depth_alpha < 0.01 {
        if hw_depth < 0.999 {
            z_final = near * far / (far - hw_depth * (far - near));
        } else {
            z_final = far;
        }
    } else {
        z_final = max(z_expected, near);
        if hw_depth < 0.999 {
            let z_hw = near * far / (far - hw_depth * (far - near));
            z_final = min(z_final, z_hw);
        }
    }

    // Reconstruct world position from depth + inverse VP
    let ndc_z = (far * (z_final - near)) / (z_final * (far - near));
    let ndc_x = frag_coord.x / f32(uniforms.width) * 2.0 - 1.0;
    let ndc_y = 1.0 - frag_coord.y / f32(uniforms.height) * 2.0;
    let clip_pos = vec4f(ndc_x, ndc_y, ndc_z, 1.0);
    let world_h = uniforms.inv_vp * clip_pos;
    let world_pos = world_h.xyz / world_h.w;

    let use_two_sample = depth_alpha >= 0.01 && z_sigma > 1e-4;
    let z_shadow_a = max(z_expected - 0.41421356 * z_sigma, near);
    let z_shadow_b = min(z_expected + 2.41421356 * z_sigma, far);
    let ndc_z_a = (far * (z_shadow_a - near)) / (z_shadow_a * (far - near));
    let ndc_z_b = (far * (z_shadow_b - near)) / (z_shadow_b * (far - near));
    let world_ha = uniforms.inv_vp * vec4f(ndc_x, ndc_y, ndc_z_a, 1.0);
    let world_hb = uniforms.inv_vp * vec4f(ndc_x, ndc_y, ndc_z_b, 1.0);
    let world_pos_a = world_ha.xyz / world_ha.w;
    let world_pos_b = world_hb.xyz / world_hb.w;

    // Per-pixel constants for the BRDF (parametric metallic workflow).
    let V = normalize(uniforms.cam_pos - world_pos);
    let N = normal;
    let NoV = max(dot(N, V), 1e-4);
    // Match pbr_bsdf.PBRBsdf.forward: roughness clamped to [min_roughness, 1].
    let r_clamped = clamp(roughness, 0.08, 1.0);
    let alpha_g = r_clamped * r_clamped;
    // let F0 = mix(vec3f(f0_dielectric), albedo, metallic);
    let F0 = (1.0 - metallic) * f0_dielectric + metallic * albedo;

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
        // return (1.0 - metallic) * f0_d + metallic * albedo
        // F0 = compute_F0(metallic, f0_diel, albedo_sel)
        // spec_color = brdf_model(half_v, view_dir, norm_sel, rough_sel, F0)
        let spec = D * Vis * F * NoL;

        var T = 1.0;
        if (uniforms.dsm_enabled != 0u) {
            if (use_two_sample) {
                let Ta = evaluate_transmittance(world_pos_a, li);
                let Tb = evaluate_transmittance(world_pos_b, li);
                T = 0.85355339 * Ta + 0.14644661 * Tb;
            } else {
                T = evaluate_transmittance(world_pos, li);
            }
        }

        // Plain Lambertian NoL.
        let l_color = light.color * light.intensity;
        let energy = T * atten;
        total_diffuse  += diff_base * NoL * energy * l_color;
        total_specular += spec      * energy * l_color;
    }
    // Hemispherical ambient × AO.
    //
    // The user-set sky/ground hemi is treated as a *far environment probe* —
    // it only contributes via Fresnel-tinted specular reflection (env_F · hemi).
    // It does NOT drive a diffuse irradiance term, because adding `kd · albedo
    // · hemi` paints the receiver's albedo into what should read as a clean
    // tinted reflection (visually indistinguishable from Lambertian ambient,
    // and wrong on metals tinted by a colored hemi).
    //
    // Diffuse ambient comes from SSGI only — that's the actual gathered
    // indirect radiance from on-screen bounces. With SSGI off, diffuse ambient
    // is zero and the receiver relies on direct lights for its base color.
    let hemi = mix(uniforms.ground_color, uniforms.sky_color, normal.y * 0.5 + 0.5);
    let ao_factor = mix(1.0, ao, uniforms.ao_strength);

    let ambient_diffuse = kd * albedo * ssgi_indirect * uniforms.ssgi_strength;

    // Indirect specular radiance: smooth surfaces use the SSR ray (sharp,
    // along reflect(V, N)); rough surfaces fall back to SSGI's hemisphere
    // average. Smoothness² mix is the standard PBR specular envelope.
    //
    // NoV fade kills the SSR contribution at silhouettes, where reflect(V,N)
    // skims along the surface (screen-space ray collapses, march false-hits
    // the silhouette pixel itself). SSGI takes over there — same fallback as
    // the off-screen / no-hit case inside ssr_compute.wgsl.
    let smoothness = 1.0 - r_clamped;
    let smoothness_sq = smoothness * smoothness;
    let ssr_grazing_fade = smoothstep(0.05, 0.35, NoV);
    let ssr_effective = mix(ssgi_indirect, ssr_radiance, ssr_grazing_fade);
    let indirect_spec_radiance = mix(ssgi_indirect, ssr_effective, smoothness_sq);

    // Lazarov env-BRDF fit: replaces F_view for the ambient/IBL specular so
    // the Fresnel factor accounts for roughness (lobe broadening + bias),
    // not just NoV. Without this, a r=1 surface and a r=0.05 surface would
    // get identical specular intensity.
    //
    // No constant `env_F · hemi` term. For metals, env_F ≈ F0 = albedo, so
    // env_F · hemi reduces to albedo · hemi — visually indistinguishable from
    // a Lambertian albedo·hemi multiply, which paints the metal's base color
    // across the whole surface and reads as a diffuse leak. Real environment
    // reflection requires an actual environment signal; here that comes from
    // SSR rays whose miss path falls back to SSGI's hemi gradient — so the
    // user's sky/ground colors still tint metals, but only through rays that
    // actually look toward those directions.
    let env_F = env_brdf_approx(F0, r_clamped, NoV);
    let ambient_specular = env_F * indirect_spec_radiance * uniforms.ssgi_strength;

    let ambient_term = uniforms.ambient * (ambient_diffuse) * ao_factor;

    var lit = total_diffuse + total_specular + ambient_term;
    var final_color = aces_narkowicz(lit * uniforms.exposure);

    // Snapshot the true lit value for the lit_history feedback — this is what
    // SSGI samples next frame. Must be taken BEFORE any debug-mode override
    // below; otherwise the SSGI feedback chain gets poisoned with debug viz.
    let lit_history_out = vec4f(max(final_color, vec3f(0.0)), alpha);

    // Debug mode overrides — bypass tonemap for visualizations
    let dm = uniforms.debug_mode;
    if (dm == DBG_RAW_ALBEDO)       { final_color = uniforms.exposure * albedo; }
    else if (dm == DBG_TRUE_ALBEDO) { final_color = uniforms.exposure * albedo; }
    else if (dm == DBG_NORMALS)     { final_color = normal * 0.5 + 0.5; }
    else if (dm == DBG_ROUGHNESS)   { final_color = vec3f(roughness); }
    else if (dm == DBG_PBR)         { final_color = vec3f(roughness, metallic, f0_dielectric); }
    else if (dm == DBG_DEPTH)       { final_color = vec3f(uniforms.exposure * z_expected * 0.1); }
    else if (dm == DBG_SPECULAR)    { final_color = aces_narkowicz(total_specular * uniforms.exposure); }
    else if (dm == DBG_DIFFUSE)     { final_color = aces_narkowicz((total_diffuse + ambient_term) * uniforms.exposure); }
    else if (dm == DBG_SHADOW) {
        var T_total = 0.0;
        if uniforms.dsm_enabled != 0u {
            for (var li = 0u; li < uniforms.num_lights; li = li + 1u) {
                T_total += evaluate_transmittance(world_pos, li);
            }
            final_color = uniforms.exposure * vec3f(T_total / max(f32(uniforms.num_lights), 1.0));
        } else {
            final_color = uniforms.exposure * vec3f(1.0);
        }
    }
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
    else if (dm == DBG_SSR)         { final_color = ssr_radiance; }
    else if (dm == DBG_DEPTH_STD) {
        // σ = √(E[z²]/α − (E[z]/α)²) — same quantity SSR/GTAO use to weight
        // neighbor confidence. Raw view-Z units; reads as grayscale where
        // bright = high uncertainty. Normalize by slot 3's own α (matches
        // pixel_std in gtao.wgsl / ssr_compute.wgsl, which see only the
        // post-filter "thick tets" population).
        let m1 = depth_raw.r;// * inv_da;
        let m2 = depth_raw.b;// * inv_da;
        final_color = vec3f(sqrt(max(m2 - m1 * m1, 0.0)));
    }

    let display_out = vec4f(max(final_color, vec3f(0.0)), alpha);
    return DeferredOut(display_out, lit_history_out);
}
