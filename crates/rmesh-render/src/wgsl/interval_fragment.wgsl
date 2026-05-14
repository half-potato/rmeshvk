// Interval shading fragment shader.
//
// Linearizes interpolated NDC depths to view-space Z, computes ray segment
// distance, and evaluates the volume rendering integral.
//
// MRT outputs (all Rgba16Float, .a = alpha for correct hardware blend):
//   location(0): albedo.rgb * a, a
//   location(1): roughness * a, metallic * a, f0_dielectric * a, a
//   location(2): field_gradient.xyz * a, a  — raw, normalized only in deferred
//   location(3): expected_depth * a, 0, expected_z2 * a, a

struct Uniforms {
    vp_col0: vec4<f32>,
    vp_col1: vec4<f32>,
    vp_col2: vec4<f32>,
    vp_col3: vec4<f32>,
    c2w_col0: vec4<f32>,
    c2w_col1: vec4<f32>,
    c2w_col2: vec4<f32>,
    intrinsics: vec4<f32>,
    cam_pos_pad: vec4<f32>,
    screen_width: f32,
    screen_height: f32,
    tet_count: u32,
    step: u32,
    tile_size_u: u32,
    ray_mode: u32,
    min_t: f32,
    sh_degree: u32,
    near_plane: f32,
    far_plane: f32,
    _pad1: vec2<u32>,
};

@group(0) @binding(0) var<storage, read> uniforms: Uniforms;
@group(0) @binding(3) var<storage, read> aux_data: array<f32>;        // [M * AUX_DIM]
@group(0) @binding(4) var<storage, read> vertex_normals: array<f32>;  // [V * 3]
@group(0) @binding(5) var<storage, read> tet_indices: array<u32>;     // [M * 4]

const AUX_DIM: u32 = 6u;

struct FragmentInput {
    @location(0) depths: vec2<f32>,                  // (z_front_ndc, z_back_ndc)
    @location(1) color_offsets: vec2<f32>,            // (offset_front, offset_back)
    @location(2) @interpolate(flat) density: f32,
    @location(3) @interpolate(flat) base_color: vec3<f32>,
    @location(4) @interpolate(flat) tet_id: u32,
    @location(5) field_gradient: vec3<f32>,          // interpolated raw gradient, normalize in pixel
};

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) aux0: vec4<f32>,
    @location(2) normals: vec4<f32>,
    @location(3) expected_depth: vec4<f32>,
};

// phi(x) = (1 - exp(-x)) / x
// Taylor with 4 terms for |x| < 0.02 avoids catastrophic cancellation.
fn phi(x: f32) -> f32 {
    if (abs(x) < 0.02) {
        return 1.0 + x * (-0.5 + x * (1.0 / 6.0 + x * (-1.0 / 24.0)));
    }
    return (1.0 - exp(-x)) / x;
}

// Volume rendering integral for a ray segment through a tet.
fn compute_integral(c0: vec3<f32>, c1: vec3<f32>, optical_depth: f32) -> vec4<f32> {
    let alpha = exp(-optical_depth);
    let phi_val = phi(optical_depth);
    let w0 = phi_val - alpha;
    let w1 = 1.0 - phi_val;
    let c = c0 * w0 + c1 * w1;
    return vec4<f32>(c, 1.0 - alpha);
}

@fragment
fn main(@builtin(position) frag_coord: vec4<f32>, in: FragmentInput) -> FragmentOutput {
    var out: FragmentOutput;

    let near = uniforms.near_plane;
    let far = uniforms.far_plane;

    // Linearize interpolated NDC depths -> view-space Z (positive = into screen)
    let range = far - near;
    let z_front_clamped = clamp(in.depths.x, 0.0, 1.0);
    let z_back_clamped = clamp(in.depths.y, 0.0, 1.0);
    let z_f = near * far / (far - z_front_clamped * range);
    let z_b = near * far / (far - z_back_clamped * range);

    // Reconstruct ray direction scale factor from pixel position
    let fx = uniforms.intrinsics.x;
    let fy = uniforms.intrinsics.y;
    let cx = uniforms.intrinsics.z;
    let cy = uniforms.intrinsics.w;

    let x_cam = (frag_coord.x - cx) / fx;
    let y_cam = (frag_coord.y - cy) / fy;
    let ray_scale = length(vec3<f32>(x_cam, y_cam, 1.0));

    // Distance through tet = |z_b - z_f| * ray_scale
    let dist = abs(z_b - z_f) * ray_scale;

    // Colors at entry/exit from interpolated offsets
    let c_front = max(in.base_color + vec3<f32>(in.color_offsets.x), vec3<f32>(0.0));
    let c_back = max(in.base_color + vec3<f32>(in.color_offsets.y), vec3<f32>(0.0));

    // Volume rendering integral
    let od = clamp(in.density * dist, 0.0, 88.0);
    let alpha_t = exp(-od);
    let alpha = 1.0 - alpha_t;

    // Per-tet aux channels lookup (parametric BRDF layout, AUX_DIM=6).
    let aux_base = in.tet_id * AUX_DIM;
    let roughness     = aux_data[aux_base + 0u];
    let metallic      = aux_data[aux_base + 1u];
    let f0_dielectric = aux_data[aux_base + 2u];
    let alb_r         = aux_data[aux_base + 3u];
    let alb_g         = aux_data[aux_base + 4u];
    let alb_b         = aux_data[aux_base + 5u];

    // Slot 0: albedo (flat per tet, premultiplied alpha)
    let albedo = vec3f(alb_r, alb_g, alb_b);
    out.color = vec4f(albedo * alpha, alpha);

    // Slot 1: PBR material (roughness, metallic, f0_dielectric)
    out.aux0 = vec4f(roughness * alpha, metallic * alpha, f0_dielectric * alpha, alpha);

    // Slot 2: normals (raw gradient, normalized in deferred)
    out.normals = vec4f(in.field_gradient * alpha, alpha);

    // Slot 3: expected termination depth + E[z²] (for std). .g is unused
    // (formerly retro * alpha; left at 0). .b holds the second-moment
    // contribution, composited via the same premul-alpha blend as .r.
    //
    // Exact within-segment α·E[z²] for the exponential termination distribution
    // on [z_f, z_b] (PDF f(z) = σ_eff · exp(−σ_eff(z−z_f))):
    //   α·E[z²] = w0·z_b² + w1·z_f² − L² · correction(od)
    // where correction(od) = w0 + α_t − 2·w0/od ≥ 0 measures how much the
    // 2-point upper bound at the extremes overshoots the true exponential.
    // For small od the direct form has catastrophic cancellation; Taylor:
    //   correction(od) ≈ od/6 − od²/12 + od³/40
    //
    // Thin-tet filter: emit (0,0,0,0) when fragment α is below threshold so
    // premul-alpha blending becomes a pass-through (dst·(1-0) = dst). Slot 3
    // therefore tracks "thick tets only" — its α lags color's α at pixels
    // covered by thin haze, and consumers (pixel_std, the deferred shader's
    // depth normalization) must divide by depth_raw.a, NOT color's alpha.
    if (alpha > 0.002) {
        let phi_val = phi(od);
        let w0 = phi_val - alpha_t;   // weight for back (z_b)
        let w1 = 1.0 - phi_val;       // weight for front (z_f)
        let depth_premul = w0 * z_b + w1 * z_f;

        var correction: f32;
        if (abs(od) < 0.02) {
            correction = od * ((1.0/6.0) - od * ((1.0/12.0) - od * (1.0/40.0)));
        } else {
            correction = w0 + alpha_t - 2.0 * w0 / od;
        }
        let L = z_b - z_f;
        let depth2_premul = w0 * z_b * z_b + w1 * z_f * z_f - L * L * correction;

        out.expected_depth = vec4f(depth_premul, 0.0, depth2_premul, alpha);
    } else {
        out.expected_depth = vec4f(0.0, 0.0, 0.0, 0.0);
    }

    return out;
}
