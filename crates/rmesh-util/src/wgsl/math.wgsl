#define_import_path rmesh::math

// Safe math constants
const TINY_VAL: f32 = 1.0754944e-20;
const MIN_VAL: f32 = -1e+20;
const MAX_VAL: f32 = 1e+20;

fn safe_clip_f32(v: f32, minv: f32, maxv: f32) -> f32 {
    return max(min(v, maxv), minv);
}

fn safe_div_f32(a: f32, b: f32) -> f32 {
    if (abs(b) < TINY_VAL) {
        return safe_clip_f32(a / TINY_VAL, MIN_VAL, MAX_VAL);
    } else {
        return safe_clip_f32(a / b, MIN_VAL, MAX_VAL);
    }
}

fn softplus(x: f32) -> f32 {
    if (x * 10.0 > 20.0) {
        return x;
    }
    return 0.1 * log(1.0 + exp(10.0 * x));
}

fn dsoftplus(x: f32) -> f32 {
    if (x * 10.0 > 20.0) {
        return 1.0;
    }
    let e = exp(10.0 * x);
    return safe_div_f32(e, 1.0 + e);
}

fn phi(x: f32) -> f32 {
    // Taylor: 1 - x/2 + x^2/6 - x^3/24.  Threshold 0.02: next term x^4/120
    // gives relative error < 1e-9, well within f32 precision.
    // Old threshold 1e-6 caused catastrophic cancellation in (1-exp(-x))/x
    // for x in [1e-6, 0.02] (4-8 digits lost).
    if (abs(x) < 0.02) { return 1.0 + x * (-0.5 + x * (1.0/6.0 + x * (-1.0/24.0))); }
    return safe_div_f32(1.0 - exp(-x), x);
}

fn dphi_dx(x: f32) -> f32 {
    // Taylor: -1/2 + x/3 - x^2/8 + x^3/30.  Same threshold rationale.
    // Old threshold 1e-6 caused catastrophic cancellation in
    // (exp(-x)*(x+1) - 1) / x^2 for x in [1e-6, 0.02].
    if (abs(x) < 0.02) { return -0.5 + x * (1.0/3.0 + x * (-1.0/8.0 + x * (1.0/30.0))); }
    let ex = exp(-x);
    return safe_div_f32(ex * (x + 1.0) - 1.0, x * x);
}

fn safe_clip_v3f(v: vec3<f32>, minv: f32, maxv: f32) -> vec3<f32> {
    return vec3<f32>(
        max(min(v.x, maxv), minv),
        max(min(v.y, maxv), minv),
        max(min(v.z, maxv), minv)
    );
}

const LOG_MAX_VAL: f32 = 46.0517;  // log(1e+20)

fn safe_exp_f32(v: f32) -> f32 {
    return exp(clamp(v, -88.0, LOG_MAX_VAL));
}

fn project_to_ndc(pos: vec3<f32>, vp: mat4x4<f32>) -> vec4<f32> {
    let clip = vp * vec4<f32>(pos, 1.0);
    let inv_w = safe_div_f32(1.0, clip.w);
    return vec4<f32>(clip.xyz * inv_w, clip.w);
}
