// Specular SSR — fullscreen fragment pass.
//
// Per pixel: ray-march the *reflection direction* `reflect(view, N)` through
// the Hi-Z chain, sample the previous frame's deferred output ("lit history")
// at the hit, and output that radiance directly. The deferred shader mixes
// this with the SSGI hemi-average by smoothness² so smooth surfaces get
// sharp reflections and rough ones fall back to the hemisphere integral.
//
// Single ray per pixel (no Monte Carlo over a lobe) — temporal accumulation
// downstream integrates out the small per-pixel start jitter. Off-screen and
// no-hit rays return the 2-color hemi gradient, same as SSGI.
//
// Reflection-origin lobe: the volume's per-pixel termination is a distribution,
// not a delta. We treat it as Normal(E[z], σ) with σ = √(E[z²]/α − (E[z]/α)²)
// from the volume MRT, draw one Gaussian sample per frame, and use the jittered
// z to reconstruct the ray origin. Sharp surfaces (σ ≈ 0) collapse to the
// deterministic origin; fuzzy volumes get a reflection lobe spread along the
// view ray, which temporal accumulation integrates out.

const MAX_STEPS: u32 = 24u;
const Z_SKY_HALF: f32 = 0.5e20;

struct SsrUniforms {
    inv_proj: mat4x4f,
    proj: mat4x4f,
    view: mat4x4f,
    inv_view: mat4x4f,
    width: u32,
    height: u32,
    near: f32,
    far: f32,
    max_mip: u32,
    frame: u32,
    radius_world: f32,
    thickness: f32,
    sky_color: vec3f,
    _pad0: f32,
    ground_color: vec3f,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> u: SsrUniforms;
@group(0) @binding(1) var hiz_tex: texture_2d<f32>;
@group(0) @binding(2) var normals_tex: texture_2d<f32>;
@group(0) @binding(3) var lit_history_tex: texture_2d<f32>;
// SSGI's hemi-averaged radiance — used as the soft miss fallback for
// SSR rays that go off-screen or find no occluder. Pulls misses into the
// same brightness range as hits, killing the binary "hit white / miss
// dim-hemi" patches that single-ray SSR produces.
@group(0) @binding(4) var ssgi_tex: texture_2d<f32>;
// Volume's expected-termination depth MRT: (E[z]·α, 0, E[z²]·α, α). Used
// to derive σ for the per-frame Gaussian jitter of the reflection origin.
@group(0) @binding(5) var volume_depth_tex: texture_2d<f32>;

struct VsOut { @builtin(position) pos: vec4f }

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    var out: VsOut;
    out.pos = vec4f(x, y, 0.0, 1.0);
    return out;
}

fn ign(p: vec2f) -> f32 {
    return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}

// Per-pixel σ from the volume's accumulated E[z], E[z²]. 0 if no volume coverage
// (so primitive-only pixels naturally collapse to the deterministic ray origin).
fn pixel_std(coords: vec2u) -> f32 {
    let v = textureLoad(volume_depth_tex, coords, 0);
    if (v.a < 0.01) { return 0.0; }
    let inv_a = 1.0 / v.a;
    let m1 = v.r * inv_a;
    let m2 = v.b * inv_a;
    return sqrt(max(m2 - m1 * m1, 0.0));
}

// Box-Muller from two IGN draws — one Gaussian sample per pixel per frame.
fn gauss(p: vec2f, frame: u32) -> f32 {
    let u1 = ign(p + vec2f(23.0, f32(frame) * 13.0));
    let u2 = ign(p + vec2f(41.0, f32(frame) * 17.0));
    let r  = sqrt(-2.0 * log(max(u1, 1e-7)));
    return r * cos(6.28318530717958647692 * u2);
}

fn linear_z_to_ndc(z_view: f32) -> f32 {
    return (u.far * (z_view - u.near)) / (z_view * (u.far - u.near));
}

fn view_pos_from_linear_z(uv: vec2f, z_view: f32) -> vec3f {
    let ndc = vec3f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, linear_z_to_ndc(max(z_view, u.near)));
    let h = u.inv_proj * vec4f(ndc, 1.0);
    return h.xyz / h.w;
}

fn project_view(p: vec3f) -> vec3f {
    let clip = u.proj * vec4f(p, 1.0);
    let ndc = clip.xyz / clip.w;
    let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return vec3f(uv, -p.z);
}

fn hiz_load(uv: vec2f, mip: u32) -> f32 {
    let m = min(mip, u.max_mip);
    let mip_dim = max(vec2u(u.width, u.height) >> vec2u(m, m), vec2u(1u, 1u));
    let mip_dim_f = vec2f(mip_dim);
    let coords = vec2u(clamp(uv * mip_dim_f, vec2f(0.0), mip_dim_f - vec2f(1.0)));
    return textureLoad(hiz_tex, coords, m).r;
}

struct RayResult { color: vec3f, hit: f32 }

// Hi-Z screen-space ray-march along the reflection direction. Returns
// `hit = 1.0` and the lit_history radiance at the hit pixel; on miss leaves
// `color = 0` and `hit = 0` so the caller can substitute its own fallback.
fn march_ray(p_origin_view: vec3f, d_view: vec3f, t_max: f32, jitter: f32) -> RayResult {
    var r: RayResult;
    r.hit = 0.0;
    r.color = vec3f(0.0);
    let p_end = p_origin_view + t_max * d_view;
    let s0 = project_view(p_origin_view);
    let s1 = project_view(p_end);
    let uv0 = s0.xy;
    let uv1 = s1.xy;
    let z0  = s0.z;
    let z1  = s1.z;
    let inv_z0 = 1.0 / max(z0, u.near);
    let inv_z1 = 1.0 / max(z1, u.near);

    let dim = vec2f(f32(u.width), f32(u.height));
    let pix_len = length((uv1 - uv0) * dim);

    var prev_t: f32 = 0.0;
    for (var s: u32 = 1u; s <= MAX_STEPS; s = s + 1u) {
        let t = (f32(s) - 1.0 + jitter) / f32(MAX_STEPS - 1u);
        let uv_t = mix(uv0, uv1, t);
        if (uv_t.x < 0.0 || uv_t.x > 1.0 || uv_t.y < 0.0 || uv_t.y > 1.0) {
            // Off-screen — caller substitutes fallback.
            return r;
        }
        let z_ray = 1.0 / mix(inv_z0, inv_z1, t);
        let stride = max(pix_len * (t - prev_t), 1.0);
        let mip = u32(clamp(floor(log2(stride)), 0.0, f32(u.max_mip)));
        let z_hiz = hiz_load(uv_t, mip);
        let thick = max(u.thickness, z_ray * 0.05);
        if (z_hiz < Z_SKY_HALF && z_ray > z_hiz && z_ray < z_hiz + thick) {
            // 6-step bisection refinement at mip 0.
            var lo = prev_t;
            var hi = t;
            for (var k: u32 = 0u; k < 6u; k = k + 1u) {
                let mid = 0.5 * (lo + hi);
                let uv_m = mix(uv0, uv1, mid);
                let z_m  = 1.0 / mix(inv_z0, inv_z1, mid);
                let z_h  = hiz_load(uv_m, 0u);
                let thick_m = max(u.thickness, z_m * 0.05);
                if (z_h < Z_SKY_HALF && z_m > z_h && z_m < z_h + thick_m) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            let uv_hit = mix(uv0, uv1, hi);
            let dim_i = vec2i(textureDimensions(lit_history_tex));
            let c = vec2i(clamp(uv_hit * vec2f(dim_i), vec2f(0.0), vec2f(dim_i - vec2i(1, 1))));
            r.color = textureLoad(lit_history_tex, c, 0).rgb;
            r.hit = 1.0;
            return r;
        }
        prev_t = t;
    }

    // No hit within radius — caller substitutes fallback.
    return r;
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
    let coords = vec2u(u32(frag_coord.x), u32(frag_coord.y));
    let dim = vec2f(f32(u.width), f32(u.height));
    let uv = vec2f(frag_coord.xy) / dim;

    // Center linear Z from Hi-Z mip 0.
    let z_view = hiz_load(uv, 0u);
    if (z_view >= Z_SKY_HALF) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }

    // Sample reflection origin from Normal(z_view, σ): treat the volume's
    // termination depth as a distribution rather than a delta, so fuzzy
    // pixels reflect from a range of points along the view ray. σ is in
    // linear view-Z (same units as z_view), so the jitter is added directly.
    let sigma_z = pixel_std(coords);
    let z_jitter = sigma_z * gauss(frag_coord.xy, u.frame);
    let z_sampled = max(z_view + z_jitter, u.near);

    let p_view = view_pos_from_linear_z(uv, z_sampled);

    let n_raw = textureLoad(normals_tex, coords, 0);
    if (n_raw.a < 0.01) { return vec4f(0.0, 0.0, 0.0, 0.0); }
    let n_world = -normalize(n_raw.rgb / n_raw.a);
    let n_view = normalize((u.view * vec4f(n_world, 0.0)).xyz);

    // Reflection direction: incident view ray = from camera (origin in view
    // space) to surface = normalize(p_view). Reflect about the surface normal.
    let view_ray = normalize(p_view);
    let dir_v = reflect(view_ray, n_view);

    // Per-pixel jitter on the starting step so temporal accumulation can
    // integrate out the resulting striping along grazing reflections.
    let step_jit = ign(frag_coord.xy + vec2f(11.0, f32(u.frame) * 7.0));

    // Self-hit bias: same scale as SSGI's so it works across scene scales.
    let bias = abs(p_view.z) * 5e-4;
    let p_o = p_view + n_view * bias;

    let res = march_ray(p_o, dir_v, u.radius_world, step_jit);

    // Miss fallback: SSGI's hemi-averaged radiance at this pixel. Vastly
    // smoother than a 2-color sky/ground gradient and brings miss pixels
    // into the same radiance range as hits, killing the binary "hit-bright /
    // miss-dim" patches that plague single-ray SSR.
    let fallback = textureLoad(ssgi_tex, vec2i(coords), 0).rgb;
    let final_color = mix(fallback, res.color, res.hit);
    return vec4f(final_color, 1.0);
}
