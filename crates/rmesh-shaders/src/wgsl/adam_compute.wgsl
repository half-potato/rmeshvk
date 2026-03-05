// Adam optimizer compute shader.
//
// One dispatch per parameter group (different learning rates for
// SH coefficients, vertices, densities, color gradients).
// Each thread updates one parameter element.

struct AdamUniforms {
    param_count: u32,
    step: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    _pad: vec2<u32>,
};

@group(0) @binding(0) var<storage, read> uniforms: AdamUniforms;
@group(0) @binding(1) var<storage, read_write> params: array<f32>;
@group(0) @binding(2) var<storage, read> grads: array<f32>;
@group(0) @binding(3) var<storage, read_write> m: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = global_id.x + global_id.y * nwg.x * 256u;
    if (idx >= uniforms.param_count) {
        return;
    }

    let grad = grads[idx];
    let step_f = f32(uniforms.step);

    // Update biased first moment
    let m_new = uniforms.beta1 * m[idx] + (1.0 - uniforms.beta1) * grad;
    m[idx] = m_new;

    // Update biased second moment
    let v_new = uniforms.beta2 * v[idx] + (1.0 - uniforms.beta2) * grad * grad;
    v[idx] = v_new;

    // Bias correction: pow(a, n) = exp(n * ln(a))
    let beta1_pow = exp(step_f * log(uniforms.beta1));
    let beta2_pow = exp(step_f * log(uniforms.beta2));
    let m_hat = m_new / (1.0 - beta1_pow);
    let v_hat = v_new / (1.0 - beta2_pow);

    // Parameter update
    params[idx] -= uniforms.lr * m_hat / (sqrt(v_hat) + uniforms.epsilon);
}
