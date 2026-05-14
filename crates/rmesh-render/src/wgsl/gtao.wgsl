// Ground-Truth Ambient Occlusion (Jiménez 2016) over a Hi-Z chain.
//
// Fullscreen fragment pass. Reads the Hi-Z mip pyramid (linear view-space Z,
// fused from hw depth and the volume's expected-termination depth) and the
// volume normals MRT; writes a single-channel R8Unorm AO factor in [0, 1]
// (1 = no occlusion, 0 = fully occluded).
//
// Canonical GTAO horizon-search:
//   - center linear Z from Hi-Z mip 0; reconstruct view-space pos
//   - transform world-space gradient → view-space normal
//   - per slice (azimuth φ), project the view normal into the slice plane,
//     measure signed normal angle `n` from view direction
//   - sample 2*N steps along the slice (forward + backward) and track
//     signed horizon angles h1, h2 (clamped to the visible cone)
//   - per-sample occlusion is attenuated by a smooth `1 - (d/r)^falloff_pow`
//     falloff (canonical GTAO) instead of a hard depth cutoff
//   - per-slice visibility uses the closed-form integral
//       v_i = 0.5 * (sin(h1 - n) + sin(h2 + n))   ∈ [-1, +1]
//   - neighbor occluder contribution is weighted by the neighbor's depth
//     std (read from the volume MRT) — fuzzy neighbors aren't sharp occluders
//   - final AO is faded toward 1 by the center pixel's own depth std

const PI: f32 = 3.14159265358979323846;
const HALF_PI: f32 = 1.57079632679489661923;
const Z_SKY: f32 = 1.0e20;
// Canonical GTAO distance falloff exponent. Larger = sharper near-radius cutoff.
const FALLOFF_POW: f32 = 2.0;

struct GtaoUniforms {
    inv_proj: mat4x4f,    // clip → view-space pos
    view: mat4x4f,        // world → view (for transforming world normals)
    width: u32,
    height: u32,
    radius_world: f32,
    thickness: f32,       // unused; retained for buffer-layout compatibility
    proj_scale: f32,      // = proj[1][1] * height * 0.5
    near: f32,
    far: f32,
    max_mip: u32,
}

@group(0) @binding(0) var<uniform> g: GtaoUniforms;
// Hi-Z: full mip chain over fused linear view-space Z. Z_SKY for true sky.
@group(0) @binding(1) var hiz_tex: texture_2d<f32>;
@group(0) @binding(2) var normals_tex: texture_2d<f32>;
// Volume's expected-termination depth MRT: (depth*α, 0, depth²*α, α).
// Used to compute per-pixel σ for neighbor weighting + center-pixel AO fade.
@group(0) @binding(3) var volume_depth_tex: texture_2d<f32>;

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

fn ign(p: vec2f) -> f32 {
    // Jimenez interleaved gradient noise — cheap blue-ish noise from screen pos.
    return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}

// Convert linear view-space Z to NDC z in [0, 1] for inv_proj reconstruction.
fn linear_z_to_ndc(z_view: f32) -> f32 {
    return (g.far * (z_view - g.near)) / (z_view * (g.far - g.near));
}

fn view_pos_from_linear_z(uv: vec2f, z_view: f32) -> vec3f {
    let ndc = vec3f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, linear_z_to_ndc(max(z_view, g.near)));
    let h = g.inv_proj * vec4f(ndc, 1.0);
    return h.xyz / h.w;
}

// Sample Hi-Z at the given mip. UV is in [0, 1] full-res space; we map to the
// mip's own coordinate grid.
fn hiz_load(uv: vec2f, mip: u32) -> f32 {
    let m = min(mip, g.max_mip);
    let mip_dim = max(vec2u(g.width, g.height) >> vec2u(m, m), vec2u(1u, 1u));
    let mip_dim_f = vec2f(mip_dim);
    let coords = vec2u(clamp(uv * mip_dim_f, vec2f(0.0), mip_dim_f - vec2f(1.0)));
    return textureLoad(hiz_tex, coords, m).r;
}

// Per-pixel σ from the volume's accumulated E[z], E[z²]. 0 if no volume coverage.
fn pixel_std(coords: vec2u) -> f32 {
    let v = textureLoad(volume_depth_tex, coords, 0);
    if (v.a < 0.01) { return 0.0; }
    let inv_a = 1.0 / v.a;
    let m1 = v.r * inv_a;
    let m2 = v.b * inv_a;
    return sqrt(max(m2 - m1 * m1, 0.0));
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
    let coords = vec2u(u32(frag_coord.x), u32(frag_coord.y));
    let dim = vec2f(f32(g.width), f32(g.height));
    let uv = vec2f(frag_coord.xy) / dim;

    // Center linear Z from Hi-Z mip 0.
    let z_center = hiz_load(uv, 0u);
    if (z_center >= Z_SKY * 0.5) { return vec4f(1.0); }   // sky → no occlusion

    let p = view_pos_from_linear_z(uv, z_center);
    // View direction: from fragment toward camera (camera at origin in view space).
    let view_dir = normalize(-p);

    // World normal → view-space (matches deferred shader's flip).
    let n_raw = textureLoad(normals_tex, coords, 0);
    if (n_raw.a < 0.01) { return vec4f(1.0); }
    let n_world = -normalize(n_raw.rgb / n_raw.a);
    let n_view = normalize((g.view * vec4f(n_world, 0.0)).xyz);

    // Perspective-correct screen-space radius (clamped to keep work bounded).
    let pixel_radius = clamp(
        g.radius_world * g.proj_scale / max(abs(p.z), 0.01),
        4.0, 200.0,
    );

    let phi_rand = ign(frag_coord.xy);
    let off_rand = ign(frag_coord.xy + vec2f(31.0, 17.0));

    let NUM_DIRS: u32 = 4u;
    let NUM_STEPS: u32 = 6u;

    var sum_visibility = 0.0;

    for (var di: u32 = 0u; di < NUM_DIRS; di = di + 1u) {
        let phi = phi_rand * PI + (f32(di) / f32(NUM_DIRS)) * PI;
        let dir2d = vec2f(cos(phi), sin(phi));

        // Build the view-space slice frame. Take a tiny screen-space probe to
        // get the in-slice tangent (the +dir direction in view space).
        let uv_probe = uv + dir2d / dim;
        let p_probe = view_pos_from_linear_z(uv_probe, z_center);
        let slice_dir_raw = p_probe - p;
        if (dot(slice_dir_raw, slice_dir_raw) < 1e-12) {
            sum_visibility = sum_visibility + 1.0;
            continue;
        }
        let slice_dir = normalize(slice_dir_raw);

        // Out-of-slice axis (perpendicular to view_dir and slice_dir).
        let slice_axis = cross(view_dir, slice_dir);
        let slice_axis_len = length(slice_axis);
        if (slice_axis_len < 1e-6) {
            sum_visibility = sum_visibility + 1.0;
            continue;
        }
        let slice_axis_n = slice_axis / slice_axis_len;

        // Project the view normal into the slice plane.
        let n_proj_vec = n_view - slice_axis_n * dot(n_view, slice_axis_n);
        let n_proj_len = length(n_proj_vec);
        if (n_proj_len < 1e-4) {
            sum_visibility = sum_visibility + 1.0;
            continue;
        }
        let n_proj = n_proj_vec / n_proj_len;

        // Signed angle of projected normal from view_dir, in slice plane,
        // positive toward +slice_dir.
        let cos_n = clamp(dot(n_proj, view_dir), -1.0, 1.0);
        let sgn_n = sign(dot(n_proj, slice_dir));
        let n = sgn_n * acos(cos_n);

        // Signed horizon angles, measured from view_dir in the slice plane.
        // h1 on +slice_dir side, h2 on −slice_dir side (h2 is also signed
        // such that "positive" means it's the magnitude of the angle from
        // view_dir going toward −slice_dir).
        // Initialize at the visible-cone bounds (no occluder yet).
        var h1 = n + HALF_PI;
        var h2 = HALF_PI - n;

        for (var s: u32 = 1u; s <= NUM_STEPS; s = s + 1u) {
            let t = (f32(s) - 1.0 + off_rand) / f32(NUM_STEPS - 1u);
            let stride_pixels = max(t * pixel_radius, 1.0);
            let mip = u32(clamp(floor(log2(stride_pixels)), 0.0, f32(g.max_mip)));
            let off = dir2d * t * pixel_radius / dim;

            // ----- +slice_dir side -----
            let uv_p = uv + off;
            if (uv_p.x >= 0.0 && uv_p.x <= 1.0 && uv_p.y >= 0.0 && uv_p.y <= 1.0) {
                let z_p = hiz_load(uv_p, mip);
                if (z_p < Z_SKY * 0.5) {
                    let p_p = view_pos_from_linear_z(uv_p, z_p);
                    let delta = p_p - p;
                    let len = length(delta);
                    let d_over_r = len / max(g.radius_world, 1e-6);
                    if (len > 1e-4 && d_over_r < 1.0) {
                        let sample_dir = delta / len;
                        // 2D coords in slice plane: x along view_dir, y along +slice_dir.
                        let sx = dot(sample_dir, view_dir);
                        let sy = dot(sample_dir, slice_dir);
                        // Signed horizon angle of this sample from view_dir
                        // (positive = above view_dir on +slice_dir side).
                        let theta = atan2(sy, sx);

                        // Smooth distance falloff (canonical GTAO) +
                        // neighbor-confidence weight from volume σ.
                        let falloff = 1.0 - pow(d_over_r, FALLOFF_POW);
                        let c_p = vec2u(clamp(uv_p * dim, vec2f(0.0), dim - vec2f(1.0)));
                        let sig_p = pixel_std(c_p);
                        let neighbor_conf = 1.0 - smoothstep(0.0, g.radius_world * 0.5, sig_p);
                        let w = clamp(falloff * neighbor_conf, 0.0, 1.0);

                        // Occluders shrink the visible cone: pull h1 down toward theta
                        // by weight w, but only if the sample actually raises occlusion
                        // (theta < h1 means the sample sits inside the current cone).
                        if (theta < h1) {
                            h1 = mix(h1, theta, w);
                        }
                    }
                }
            }

            // ----- −slice_dir side -----
            let uv_n = uv - off;
            if (uv_n.x >= 0.0 && uv_n.x <= 1.0 && uv_n.y >= 0.0 && uv_n.y <= 1.0) {
                let z_n = hiz_load(uv_n, mip);
                if (z_n < Z_SKY * 0.5) {
                    let p_n = view_pos_from_linear_z(uv_n, z_n);
                    let delta = p_n - p;
                    let len = length(delta);
                    let d_over_r = len / max(g.radius_world, 1e-6);
                    if (len > 1e-4 && d_over_r < 1.0) {
                        let sample_dir = delta / len;
                        // Mirror y so the same logic applies to the −slice_dir side.
                        let sx = dot(sample_dir, view_dir);
                        let sy = -dot(sample_dir, slice_dir);
                        let theta = atan2(sy, sx);

                        let falloff = 1.0 - pow(d_over_r, FALLOFF_POW);
                        let c_n = vec2u(clamp(uv_n * dim, vec2f(0.0), dim - vec2f(1.0)));
                        let sig_n = pixel_std(c_n);
                        let neighbor_conf = 1.0 - smoothstep(0.0, g.radius_world * 0.5, sig_n);
                        let w = clamp(falloff * neighbor_conf, 0.0, 1.0);

                        if (theta < h2) {
                            h2 = mix(h2, theta, w);
                        }
                    }
                }
            }
        }

        // Clamp horizons to the visible cone (visibility half-spaces around the
        // projected normal): h1 - n ∈ [-π/2, π/2], h2 + n ∈ [-π/2, π/2].
        h1 = clamp(h1, n - HALF_PI, n + HALF_PI);
        h2 = clamp(h2, -HALF_PI - n, HALF_PI - n);

        // Per-slice visibility (closed-form GTAO integral) ∈ [-1, +1].
        //   +1 = no occluders, -1 = fully occluded.
        let v_slice = 0.5 * (sin(h1 - n) + sin(h2 + n));
        sum_visibility = sum_visibility + v_slice;
    }

    let v_avg = sum_visibility / f32(NUM_DIRS);
    // Map signed visibility ∈ [-1, +1] → AO factor ∈ [0, 1].
    let ao = clamp(0.5 * (v_avg + 1.0), 0.0, 1.0);
    return vec4f(ao);
}
