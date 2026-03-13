//! Per-kernel unit tests for the forward rendering pipeline.
//!
//! Tests each shader/kernel in isolation:
//!   - forward_compute.wgsl: color eval, cull, depth key generation
//!   - radix sort (5-pass): tet depth sorting
//!   - forward_tiled_compute.wgsl: subgroup-based tiled forward rendering
//!   - tex_to_buffer.wgsl: texture → storage buffer conversion
//!
//! Each test creates a GPU device with SUBGROUP feature, dispatches a single
//! kernel, reads back the result, and validates correctness.
//!
//! Run: `cargo test -p rmesh-render --test kernel_tests -- --nocapture`

mod common;

use common::*;
use glam::Vec3;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rmesh_util::test_util::{create_test_device_default as create_test_device, read_buffer};

const SEED: u64 = 42424242;
const W: u32 = 16;
const H: u32 = 16;

fn setup_camera(eye: Vec3, target: Vec3) -> (glam::Mat4, glam::Mat4) {
    let aspect = W as f32 / H as f32;
    let proj = perspective_matrix(std::f32::consts::FRAC_PI_2, aspect, 0.01, 100.0);
    let view = look_at(eye, target, Vec3::new(0.0, 0.0, 1.0));
    let vp = proj * view;
    let inv_vp = vp.inverse();
    (vp, inv_vp)
}

// ---------------------------------------------------------------------------
// Test: forward_compute.wgsl (color eval + cull + depth keys)
// ---------------------------------------------------------------------------

/// Dispatches only the forward compute kernel (no sort, no render).
/// Verifies that the compute pass:
///   - Sets instance_count > 0 in indirect args (tet is visible)
///   - Does not crash or produce GPU errors
#[test]
fn test_forward_compute_kernel() {
    let (device, queue) = match create_test_device() {
        Some(dq) => dq,
        None => {
            eprintln!("Skipping test_forward_compute_kernel (no GPU)");
            return;
        }
    };

    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = random_single_tet_scene(&mut rng, 0.3);

    let buffers = rmesh_render::SceneBuffers::upload(&device, &queue, &scene);
    let zero_base_colors = vec![0.5f32; scene.tet_count as usize * 3];
    let material = rmesh_render::MaterialBuffers::upload(&device, &zero_base_colors, &scene.color_grads, scene.tet_count);
    let pipelines = rmesh_render::ForwardPipelines::new(
        &device,
        wgpu::TextureFormat::Rgba16Float,
        wgpu::TextureFormat::Rgba32Float,
    );
    let compute_bg = rmesh_render::create_compute_bind_group(&device, &pipelines, &buffers, &material);

    // Camera looking at tet from outside
    let verts = load_tet_verts(&scene, 0);
    let centroid = (verts[0] + verts[1] + verts[2] + verts[3]) * 0.25;
    let eye = centroid + Vec3::new(2.0, 0.0, 0.0);
    let (vp, inv_vp) = setup_camera(eye, centroid);

    let uniforms =
        rmesh_render::make_uniforms(vp, inv_vp, eye, W as f32, H as f32, scene.tet_count, 0);
    queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

    // Reset indirect args
    let reset_cmd = rmesh_render::DrawIndirectCommand {
        vertex_count: 12,
        instance_count: 0,
        first_vertex: 0,
        first_instance: 0,
    };
    queue.write_buffer(&buffers.indirect_args, 0, bytemuck::bytes_of(&reset_cmd));

    // Dispatch compute only
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("forward_compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.compute_pipeline);
        pass.set_bind_group(0, &compute_bg, &[]);
        let n_pow2 = scene.tet_count.next_power_of_two();
        pass.dispatch_workgroups((n_pow2 + 63) / 64, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    // Read back indirect args to verify instance_count > 0
    let args: Vec<u32> = read_buffer(&device, &queue, &buffers.indirect_args, 4);
    let instance_count = args[1];
    eprintln!("forward_compute: instance_count = {instance_count}");
    assert!(
        instance_count > 0,
        "Tet should be visible but instance_count=0"
    );
}

// ---------------------------------------------------------------------------
// Test: forward_tiled_compute.wgsl (subgroup shader compilation)
// ---------------------------------------------------------------------------

/// Validates that the forward tiled compute pipeline (which requires
/// `enable subgroups;` in WGSL) can be compiled successfully.
///
/// This is the primary test for verifying that wgpu's SUBGROUP feature
/// is properly hooked up. If subgroups aren't supported, pipeline
/// creation will fail.
#[test]
fn test_forward_tiled_pipeline_creation() {
    let (device, _queue) = match create_test_device() {
        Some(dq) => dq,
        None => {
            eprintln!("Skipping test_forward_tiled_pipeline_creation (no GPU)");
            return;
        }
    };

    // This will panic if the shader fails to compile (e.g., subgroups not supported)
    let pipeline = rmesh_render::ForwardTiledPipeline::new(&device, W, H);

    // Verify the output buffer was created with correct size
    assert_eq!(pipeline.width, W);
    assert_eq!(pipeline.height, H);
    eprintln!("forward_tiled_compute.wgsl compiled successfully with subgroup support");
}

// ---------------------------------------------------------------------------
// Test: full forward pass (compute + sort + render)
// ---------------------------------------------------------------------------

/// End-to-end forward pass: compute → hardware rasterize.
/// Verifies the legacy (non-tiled) pipeline produces output matching CPU.
#[test]
fn test_legacy_forward_e2e() {
    // Use larger resolution (64×64) so the tet covers enough pixels to be visible.
    // 16×16 can miss pixel centers for small tets at distance.
    const E2E_W: u32 = 64;
    const E2E_H: u32 = 64;

    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = random_single_tet_scene(&mut rng, 0.3);

    let verts = load_tet_verts(&scene, 0);
    let centroid = (verts[0] + verts[1] + verts[2] + verts[3]) * 0.25;
    let eye = centroid + Vec3::new(2.0, 0.0, 0.0);

    let aspect = E2E_W as f32 / E2E_H as f32;
    let proj = perspective_matrix(std::f32::consts::FRAC_PI_2, aspect, 0.01, 100.0);
    let view = look_at(eye, centroid, Vec3::new(0.0, 0.0, 1.0));
    let vp = proj * view;
    let inv_vp = vp.inverse();

    let cpu_image = cpu_render_scene(&scene, eye, vp, inv_vp, E2E_W, E2E_H);
    let total_alpha: f32 = cpu_image.iter().map(|p| p[3]).sum();
    assert!(total_alpha > 0.01, "CPU image is all-zero");

    if let Some(gpu_image) = gpu_render_scene(&scene, eye, vp, inv_vp, E2E_W, E2E_H) {
        let (max_diff, mean_diff, _) = compare_images(&cpu_image, &gpu_image);
        eprintln!("legacy_forward_e2e: max_diff={max_diff:.4}, mean_diff={mean_diff:.6}");
        assert!(mean_diff < 0.1, "mean_diff {mean_diff} >= 0.1");
    } else {
        eprintln!("Skipping GPU comparison (no adapter)");
    }
}
