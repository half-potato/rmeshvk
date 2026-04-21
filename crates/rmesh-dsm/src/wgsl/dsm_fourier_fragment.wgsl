// Fourier deep shadow map fragment shader.
//
// Outputs Fourier coefficients of the opacity density into 3 MRT targets,
// accumulated via additive blending (ONE, ONE). See TMSM.md for math.
//
// MRT layout (N=4, 9 coefficients):
//   RT0: (a0, a1, b1, a2)
//   RT1: (b2, a3, b3, a4)
//   RT2: (b4, 0, 0, 0)

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

struct FragmentInput {
    @location(0) depths: vec2<f32>,
    @location(1) color_offsets: vec2<f32>,
    @location(2) @interpolate(flat) density: f32,
    @location(3) @interpolate(flat) base_color: vec3<f32>,
    @location(4) @interpolate(flat) tet_id: u32,
};

struct FourierOutput {
    @location(0) rt0: vec4<f32>,
    @location(1) rt1: vec4<f32>,
    @location(2) rt2: vec4<f32>,
};

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;

// phi(x) = (1 - exp(-x)) / x
// Taylor with 4 terms for |x| < 0.02 avoids catastrophic cancellation.
fn phi(x: f32) -> f32 {
    if (abs(x) < 0.02) {
        return 1.0 + x * (-0.5 + x * (1.0 / 6.0 + x * (-1.0 / 24.0)));
    }
    return (1.0 - exp(-x)) / x;
}

@fragment
fn main(@builtin(position) frag_coord: vec4<f32>, in: FragmentInput) -> FourierOutput {
    let near = uniforms.near_plane;
    let far = uniforms.far_plane;
    let range = far - near;

    // Linearize NDC depths to view-space Z
    let z_f = near * far / (far - clamp(in.depths.x, 0.0, 1.0) * range);
    let z_b = near * far / (far - clamp(in.depths.y, 0.0, 1.0) * range);

    // Ray direction scale factor
    let fx = uniforms.intrinsics.x;
    let fy = uniforms.intrinsics.y;
    let x_cam = (frag_coord.x - uniforms.intrinsics.z) / fx;
    let y_cam = (frag_coord.y - uniforms.intrinsics.w) / fy;
    let ray_scale = length(vec3<f32>(x_cam, y_cam, 1.0));

    // Opacity of this interval
    let dist = abs(z_b - z_f) * ray_scale;
    let od = clamp(in.density * dist, 0.0, 88.0);
    let alpha = 1.0 - exp(-od);

    // Normalized depths [0,1]
    let za = clamp((z_f - near) / range, 0.0, 1.0);
    let zb = clamp((z_b - near) / range, 0.0, 1.0);

    // Volume rendering weights (same as forward_fragment.wgsl / interval_fragment.wgsl):
    //   w0 = phi(od) - exp(-od)   → weight for back depth (zb)
    //   w1 = 1 - phi(od)          → weight for front depth (za)
    // These are the exact Beer-Lambert quadrature weights.
    // w0 + w1 = alpha = 1 - exp(-od)
    let alpha_t = exp(-od);
    let phi_val = phi(od);
    let w0 = phi_val - alpha_t;  // back weight
    let w1 = 1.0 - phi_val;     // front weight

    // Power moments using the same quadrature:
    //   m_k = w0 * zb^k + w1 * za^k
    // This is already premultiplied by alpha (since w0 + w1 = alpha).
    let za2 = za * za; let zb2 = zb * zb;
    let za3 = za2 * za; let zb3 = zb2 * zb;
    let za4 = za3 * za; let zb4 = zb3 * zb;

    let mk0 = w0 + w1;                    // = alpha
    let mk1 = w0 * zb + w1 * za;          // = expected depth (premul)
    let mk2 = w0 * zb2 + w1 * za2;
    let mk3 = w0 * zb3 + w1 * za3;
    let mk4 = w0 * zb4 + w1 * za4;

    var out: FourierOutput;
    out.rt0 = vec4<f32>(mk0, mk1, mk2, mk3);
    out.rt1 = vec4<f32>(mk4, 0.0, 0.0, 0.0);
    out.rt2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    return out;
}
