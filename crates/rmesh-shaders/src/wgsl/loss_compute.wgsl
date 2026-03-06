// Loss computation shader: per-pixel L1/L2 loss and gradient computation.
//
// Dispatched over pixels. Reads the rendered image and ground truth,
// computes loss and per-pixel gradient dL/d(pixel).

struct LossUniforms {
    width: u32,
    height: u32,
    loss_type: u32,
    lambda_ssim: f32,
};

@group(0) @binding(0) var<storage, read> uniforms: LossUniforms;
@group(0) @binding(1) var<storage, read> rendered: array<f32>;
@group(0) @binding(2) var<storage, read> ground_truth: array<f32>;
@group(0) @binding(3) var<storage, read_write> dl_d_image: array<f32>;
@group(0) @binding(4) var<storage, read_write> loss_value: array<atomic<u32>>;

fn sign_f32(x: f32) -> f32 {
    if (x > 0.0) { return 1.0; }
    if (x < 0.0) { return -1.0; }
    return 0.0;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= uniforms.width || y >= uniforms.height) {
        return;
    }

    let pixel_idx = y * uniforms.width + x;
    let n_pixels = f32(uniforms.width * uniforms.height);

    // Read rendered pixel (premultiplied RGBA)
    let r_idx = pixel_idx * 4u;
    let rendered_r = rendered[r_idx];
    let rendered_g = rendered[r_idx + 1u];
    let rendered_b = rendered[r_idx + 2u];

    // Read ground truth (RGB)
    let gt_idx = pixel_idx * 3u;
    let gt_r = ground_truth[gt_idx];
    let gt_g = ground_truth[gt_idx + 1u];
    let gt_b = ground_truth[gt_idx + 2u];

    let diff_r = rendered_r - gt_r;
    let diff_g = rendered_g - gt_g;
    let diff_b = rendered_b - gt_b;

    var grad_r: f32;
    var grad_g: f32;
    var grad_b: f32;
    var pixel_loss: f32;

    if (uniforms.loss_type == 0u) {
        // L1 loss
        grad_r = sign_f32(diff_r) / n_pixels;
        grad_g = sign_f32(diff_g) / n_pixels;
        grad_b = sign_f32(diff_b) / n_pixels;
        pixel_loss = (abs(diff_r) + abs(diff_g) + abs(diff_b)) / n_pixels;
    } else {
        // L2 loss
        grad_r = 2.0 * diff_r / n_pixels;
        grad_g = 2.0 * diff_g / n_pixels;
        grad_b = 2.0 * diff_b / n_pixels;
        pixel_loss = (diff_r * diff_r + diff_g * diff_g + diff_b * diff_b) / n_pixels;
    }

    dl_d_image[r_idx] = grad_r;
    dl_d_image[r_idx + 1u] = grad_g;
    dl_d_image[r_idx + 2u] = grad_b;
    dl_d_image[r_idx + 3u] = 0.0;

    // Atomic float accumulation via CAS loop
    var old_bits = atomicLoad(&loss_value[0]);
    loop {
        let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + pixel_loss);
        let result = atomicCompareExchangeWeak(&loss_value[0], old_bits, new_bits);
        if (result.exchanged) { break; }
        old_bits = result.old_value;
    }
}
