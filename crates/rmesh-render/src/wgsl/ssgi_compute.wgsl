// Diffuse SSGI — fullscreen fragment pass.
//
// Per pixel: ray-march N cosine-weighted hemisphere directions through the
// Hi-Z chain, sample the previous frame's deferred output ("lit history") at
// each ray hit, and average. Output is the indirect-diffuse radiance L_in;
// the deferred shader multiplies it by `kd · albedo` to complete the bounce.
//
// Cosine importance sampling: pdf(ω) = (n·ω)/π, so the Monte Carlo estimate
// of (1/π) ∫ L_in(ω)(n·ω)dω becomes simply (1/N) Σ L_i.
//
// Off-screen / no-hit rays fall back to the existing 2-color hemisphere
// (sky_color/ground_color), matching the previous ambient look beyond
// SSGI's reach.

const NUM_RAYS: u32 = 4u;
const MAX_STEPS: u32 = 24u;
const Z_SKY_HALF: f32 = 0.5e20;
const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;

struct SsgiUniforms {
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

@group(0) @binding(0) var<uniform> u: SsgiUniforms;
@group(0) @binding(1) var hiz_tex: texture_2d<f32>;
@group(0) @binding(2) var normals_tex: texture_2d<f32>;
@group(0) @binding(3) var lit_history_tex: texture_2d<f32>;

struct VsOut { @builtin(position) pos: vec4f }

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    var out: VsOut;
    out.pos = vec4f(x, y, 0.0, 1.0);
    return out;
}

// Jimenez interleaved gradient noise — cheap blue-ish noise from screen pos.
fn ign(p: vec2f) -> f32 {
    return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}

// Van der Corput sequence (radical inverse base 2).
fn radical_inverse_vdc(b: u32) -> f32 {
    var bits: u32 = b;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10;
}

fn hammersley(i: u32, n: u32) -> vec2f {
    return vec2f(f32(i) / f32(n), radical_inverse_vdc(i));
}

// Cosine-weighted hemisphere sample in tangent space (z = up).
fn cosine_sample_hemisphere(uv: vec2f) -> vec3f {
    let phi = TWO_PI * uv.x;
    let cos_t = sqrt(uv.y);
    let sin_t = sqrt(max(1.0 - uv.y, 0.0));
    return vec3f(sin_t * cos(phi), sin_t * sin(phi), cos_t);
}

// Build an orthonormal basis around N (Frisvad / Pixar branchless).
fn make_tbn(n: vec3f) -> mat3x3f {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let t = vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let bv = vec3f(b, s + n.y * n.y * a, -n.y);
    return mat3x3f(t, bv, n);
}

fn linear_z_to_ndc(z_view: f32) -> f32 {
    return (u.far * (z_view - u.near)) / (z_view * (u.far - u.near));
}

fn view_pos_from_linear_z(uv: vec2f, z_view: f32) -> vec3f {
    let ndc = vec3f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, linear_z_to_ndc(max(z_view, u.near)));
    let h = u.inv_proj * vec4f(ndc, 1.0);
    return h.xyz / h.w;
}

// Project a view-space point to (uv, linear_z). Linear z = -p_view.z under
// the renderer's right-handed -Z-forward convention.
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

// 2-color hemisphere fallback for off-screen / missed rays.
fn hemi_fallback(world_dir: vec3f) -> vec3f {
    return mix(u.ground_color, u.sky_color, world_dir.y * 0.5 + 0.5);
}

// Hi-Z screen-space ray march. Returns (hit_uv, hit, miss_dir_world).
struct RayResult {
    color: vec3f,
}

fn march_ray(p_origin_view: vec3f, d_view: vec3f, t_max: f32, jitter: f32) -> RayResult {
    var r: RayResult;
    let p_end = p_origin_view + t_max * d_view;
    let s0 = project_view(p_origin_view);   // (uv0, z0)
    let s1 = project_view(p_end);           // (uv1, z1)
    let uv0 = s0.xy;
    let uv1 = s1.xy;
    let z0  = s0.z;
    let z1  = s1.z;
    // #8 Perspective-correct z along the ray: interpolate 1/z, not z.
    let inv_z0 = 1.0 / max(z0, u.near);
    let inv_z1 = 1.0 / max(z1, u.near);

    // Average pixel stride across the ray; used to pick the starting Hi-Z mip.
    let dim = vec2f(f32(u.width), f32(u.height));
    let pix_len = length((uv1 - uv0) * dim);

    var prev_t: f32 = 0.0;
    for (var s: u32 = 1u; s <= MAX_STEPS; s = s + 1u) {
        let t = (f32(s) - 1.0 + jitter) / f32(MAX_STEPS - 1u);
        let uv_t = mix(uv0, uv1, t);
        if (uv_t.x < 0.0 || uv_t.x > 1.0 || uv_t.y < 0.0 || uv_t.y > 1.0) {
            // Off-screen → fall back to hemi.
            let world_dir = (u.inv_view * vec4f(d_view, 0.0)).xyz;
            r.color = hemi_fallback(normalize(world_dir));
            return r;
        }
        let z_ray = 1.0 / mix(inv_z0, inv_z1, t);
        // Pick mip from this step's pixel stride along the ray.
        let stride = max(pix_len * (t - prev_t), 1.0);
        let mip = u32(clamp(floor(log2(stride)), 0.0, f32(u.max_mip)));
        let z_hiz = hiz_load(uv_t, mip);
        // #7 Depth-relative thickness: floor at u.thickness for near-camera
        // rays, scale with depth for distant ones.
        let thick = max(u.thickness, z_ray * 0.05);
        if (z_hiz < Z_SKY_HALF && z_ray > z_hiz && z_ray < z_hiz + thick) {
            // #4 Coarse hit found in cell at `mip`. Refine to mip 0 within
            // the current segment [prev_t, t] via 6-step bisection so the
            // sample point isn't off by up to a mip-cell width.
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
            return r;
        }
        prev_t = t;
    }

    // Reached max steps without a hit → fall back to hemi.
    let world_dir = (u.inv_view * vec4f(d_view, 0.0)).xyz;
    r.color = hemi_fallback(normalize(world_dir));
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

    let p_view = view_pos_from_linear_z(uv, z_view);

    // World normal → view-space (matches deferred shader's flip).
    let n_raw = textureLoad(normals_tex, coords, 0);
    if (n_raw.a < 0.01) { return vec4f(0.0, 0.0, 0.0, 0.0); }
    let n_world = -normalize(n_raw.rgb / n_raw.a);
    let n_view = normalize((u.view * vec4f(n_world, 0.0)).xyz);

    let tbn = make_tbn(n_view);

    // Per-pixel rotation jitter so neighboring pixels sample different sets.
    let rot = ign(frag_coord.xy + vec2f(f32(u.frame), 0.0));
    let cos_r = cos(rot * TWO_PI);
    let sin_r = sin(rot * TWO_PI);
    let step_jit = ign(frag_coord.xy + vec2f(13.0, f32(u.frame) * 7.0));

    var sum = vec3f(0.0);
    for (var i: u32 = 0u; i < NUM_RAYS; i = i + 1u) {
        var hxy = hammersley(i, NUM_RAYS);
        // Rotate the (azimuthal) Hammersley.x by per-pixel angle.
        hxy.x = fract(hxy.x + rot);
        let dir_t = cosine_sample_hemisphere(hxy);          // tangent space
        let dir_v = normalize(tbn * dir_t);                 // view space

        // #9 Self-hit bias scales with view-space depth so it stays
        // appropriate at any scene scale.
        let bias = abs(p_view.z) * 5e-4;
        let p_o = p_view + n_view * bias;
        let res = march_ray(p_o, dir_v, u.radius_world, step_jit);
        sum = sum + res.color;
    }
    let avg = sum / f32(NUM_RAYS);
    return vec4f(avg, 1.0);
}
