// Forward fragment shader: volume rendering integral with MRT output.
//
// Computes ray-tet intersection, evaluates the volume integral with linear color
// interpolation, and outputs premultiplied alpha for back-to-front blending.
//
// MRT outputs:
//   location(0): premultiplied color (RGBA)
//   location(1): auxiliary float4 (t_enter, t_exit, optical_depth, dist)
//   location(2): normals (premultiplied alpha)
//   location(3): depth (premultiplied alpha)

struct FragmentInput {
    @location(0) @interpolate(flat) tet_density: f32,
    @location(1) @interpolate(flat) base_color: vec3<f32>,
    @location(2) plane_numerators: vec4<f32>,
    @location(3) plane_denominators: vec4<f32>,
    @location(4) ray_dir: vec3<f32>,
    @location(5) dc_dt: f32,
    @location(6) @interpolate(flat) face_n0: vec3<f32>,
    @location(7) @interpolate(flat) face_n1: vec3<f32>,
    @location(8) @interpolate(flat) face_n2: vec3<f32>,
    @location(9) @interpolate(flat) face_n3: vec3<f32>,
};

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) aux0: vec4<f32>,
    @location(2) normals: vec4<f32>,
    @location(3) depth: vec4<f32>,
};

// --- Safe math utilities (subset of safe_math.wgsl) ---
const MIN_VAL: f32 = -1e+20;
const MAX_VAL: f32 = 1e+20;

fn safe_clip_v4f(v: vec4<f32>, minv: f32, maxv: f32) -> vec4<f32> {
    return vec4<f32>(
        max(min(v.x, maxv), minv),
        max(min(v.y, maxv), minv),
        max(min(v.z, maxv), minv),
        max(min(v.w, maxv), minv)
    );
}
// --- End safe math utilities ---

// phi(x) = (1 - exp(-x)) / x
// Taylor with 4 terms for |x| < 0.02 avoids catastrophic cancellation.
fn phi(x: f32) -> f32 {
    if (abs(x) < 0.02) {
        return 1.0 + x * (-0.5 + x * (1.0/6.0 + x * (-1.0/24.0)));
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
fn main(in: FragmentInput) -> FragmentOutput {
    var out: FragmentOutput;

    let base_color = in.base_color;
    let d = length(in.ray_dir);
    let plane_denom = in.plane_denominators / d;
    let dc_dt = in.dc_dt / d;
    let all_t = in.plane_numerators / plane_denom;

    let neg_inf = vec4<f32>(-3.402823e38);
    let pos_inf = vec4<f32>(3.402823e38);

    // Classify planes: denom > 0 -> entering, denom < 0 -> exiting
    let t_enter = vec4<f32>(
        select(neg_inf.x, all_t.x, plane_denom.x > 0.0),
        select(neg_inf.y, all_t.y, plane_denom.y > 0.0),
        select(neg_inf.z, all_t.z, plane_denom.z > 0.0),
        select(neg_inf.w, all_t.w, plane_denom.w > 0.0),
    );
    let t_exit = vec4<f32>(
        select(pos_inf.x, all_t.x, plane_denom.x < 0.0),
        select(pos_inf.y, all_t.y, plane_denom.y < 0.0),
        select(pos_inf.z, all_t.z, plane_denom.z < 0.0),
        select(pos_inf.w, all_t.w, plane_denom.w < 0.0),
    );

    let t_min = max(max(t_enter.x, t_enter.y), max(t_enter.z, t_enter.w));
    let t_max = min(min(t_exit.x, t_exit.y), min(t_exit.z, t_exit.w));

    let dist = max(t_max - t_min, 0.0);
    let od = clamp(in.tet_density * dist, 0.0, 88.0);

    let c_start = max(base_color + vec3<f32>(dc_dt * t_min), vec3<f32>(0.0));
    let c_end = max(base_color + vec3<f32>(dc_dt * t_max), vec3<f32>(0.0));

    // Note: (c_end, c_start) -- exit color first, matching webrm convention
    out.color = safe_clip_v4f(compute_integral(c_end, c_start, od), MIN_VAL, MAX_VAL);
    out.aux0 = vec4<f32>(t_min, t_max, od, dist);

    // Determine entry face (which t_enter component is max)
    var entry_face = 0u;
    var max_t_enter = t_enter.x;
    if (t_enter.y > max_t_enter) { max_t_enter = t_enter.y; entry_face = 1u; }
    if (t_enter.z > max_t_enter) { max_t_enter = t_enter.z; entry_face = 2u; }
    if (t_enter.w > max_t_enter) { max_t_enter = t_enter.w; entry_face = 3u; }

    // Entry normal (outward = negated inward face normal)
    var entry_normal_raw: vec3<f32>;
    switch entry_face {
        case 0u: { entry_normal_raw = in.face_n0; }
        case 1u: { entry_normal_raw = in.face_n1; }
        case 2u: { entry_normal_raw = in.face_n2; }
        default: { entry_normal_raw = in.face_n3; }
    }
    let entry_normal = normalize(-entry_normal_raw);

    // Alpha and depth from volume integral
    let alpha_t = exp(-od);
    let alpha = 1.0 - alpha_t;
    let phi_val = phi(od);
    let w0 = phi_val - alpha_t;
    let w1 = 1.0 - phi_val;
    let depth_premul = w0 * t_max + w1 * t_min;

    // MRT target 2: normals (premultiplied alpha blend)
    out.normals = vec4<f32>(alpha * entry_normal, alpha);

    // MRT target 3: depth (premultiplied alpha blend)
    // depth_premul is already weighted by (w0+w1)=alpha, so it's premultiplied
    out.depth = vec4<f32>(depth_premul, 0.0, 0.0, alpha);

    return out;
}
