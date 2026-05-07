// Temporal accumulation — shared shader for SSGI and AO.
//
// Given current-frame radiance/AO + a reprojected history sample + Hi-Z
// (for world-pos reconstruction), produces a temporally-blended output:
//
//   out = mix(clamp(history(prev_uv), aabb_3x3_of_current), current, alpha)
//
// The history is reprojected via prev_vp (camera-only motion vectors,
// computed inline — works because the volume is static). A 3×3 min/max of
// the current frame's neighborhood acts as a hard clamp on the history
// sample, suppressing ghosting on disocclusion or fast camera motion.
//
// The same shader is used for both SSGI (Rgba16Float) and AO (R8Unorm) —
// only the render-target format on the pipeline differs. textureLoad
// always returns vec4<f32>; for R8Unorm the .gb channels are unused but
// harmless. Writing vec4<f32> to R8Unorm keeps only .r.

const Z_SKY_HALF: f32 = 0.5e20;

struct TemporalUniforms {
    inv_vp:  mat4x4f,
    prev_vp: mat4x4f,
    width: u32,
    height: u32,
    near: f32,
    far: f32,
    max_mip: u32,
    alpha: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> u: TemporalUniforms;
@group(0) @binding(1) var current_tex: texture_2d<f32>;
@group(0) @binding(2) var history_tex: texture_2d<f32>;
@group(0) @binding(3) var hiz_tex:     texture_2d<f32>;
@group(0) @binding(4) var hist_sampler: sampler;

struct VsOut { @builtin(position) pos: vec4f }

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    var out: VsOut;
    out.pos = vec4f(x, y, 0.0, 1.0);
    return out;
}

fn linear_z_to_ndc(z_view: f32) -> f32 {
    return (u.far * (z_view - u.near)) / (z_view * (u.far - u.near));
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
    let coords = vec2i(i32(frag_coord.x), i32(frag_coord.y));
    let dim = vec2i(i32(u.width), i32(u.height));
    let dim_f = vec2f(dim);
    let uv = vec2f(frag_coord.xy) / dim_f;

    let current = textureLoad(current_tex, coords, 0);

    // Reconstruct world position from this pixel's Hi-Z mip 0 depth.
    let z_view = textureLoad(hiz_tex, coords, 0).r;
    if (z_view >= Z_SKY_HALF) {
        // Sky pixel — no surface; just pass current through. History is
        // meaningless here.
        return current;
    }
    let ndc_z = linear_z_to_ndc(max(z_view, u.near));
    let ndc = vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, ndc_z, 1.0);
    let world_h = u.inv_vp * ndc;
    let world = world_h.xyz / world_h.w;

    // Project to previous frame's clip space, then to screen UV.
    let prev_clip = u.prev_vp * vec4f(world, 1.0);
    var hist_valid = prev_clip.w > 0.0;
    var prev_uv = vec2f(0.0);
    if (hist_valid) {
        let prev_ndc = prev_clip.xy / prev_clip.w;
        prev_uv = vec2f(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);
        hist_valid = prev_uv.x >= 0.0 && prev_uv.x <= 1.0
                  && prev_uv.y >= 0.0 && prev_uv.y <= 1.0;
    }

    if (!hist_valid) {
        return current;
    }

    // 3×3 neighborhood AABB of the current frame for history clamping.
    var lo = vec4f(1e30);
    var hi = vec4f(-1e30);
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let p = clamp(coords + vec2i(dx, dy), vec2i(0, 0), dim - vec2i(1, 1));
            let v = textureLoad(current_tex, p, 0);
            lo = min(lo, v);
            hi = max(hi, v);
        }
    }

    // Bilinear-sample history at prev UV.
    let history = textureSampleLevel(history_tex, hist_sampler, prev_uv, 0.0);

    // Cold-history bypass: if the sampled history is essentially zero (e.g.
    // after init / resize where we cleared the texture, or in regions newly
    // disoccluded), the AABB clamp would pin the output to the local minimum
    // of `current` and mute strong signal (first-bounce point-light energy
    // never propagates through). Output current directly in that case so the
    // signal can seed the next frame's history.
    let hist_max = max(max(max(history.r, history.g), history.b), history.a);
    if (hist_max < 1e-4) {
        return current;
    }

    // History is real → 3×3 AABB clamp suppresses ghosting on disocclusion.
    let clamped = clamp(history, lo, hi);

    // Disocclusion-aware α: when the clamp had to drag history a long way
    // to fit inside current's local AABB, the pixel just changed —
    // disocclusion (camera revealed new geometry) or sudden lighting change
    // (point light turned on, surface property changed). Trust history less
    // by biasing α toward 1 (full current) for those pixels. Pixels whose
    // history sits naturally inside the AABB get the user's α (smooth
    // temporal). Avoids the multi-frame "fade in" of new shadows / bounces.
    let aabb_size = length(hi - lo) + 1e-4;
    let clamp_distance = length(history - clamped);
    let confidence = 1.0 - smoothstep(0.0, aabb_size * 0.5, clamp_distance);
    let effective_alpha = mix(1.0, u.alpha, confidence);
    return mix(clamped, current, effective_alpha);
}
