mod common;

use common::*;
use glam::Vec3;

const W: u32 = 64;
const H: u32 = 64;

/// Build a 2-tet scene with known aux data for MRT testing.
fn make_mrt_test_scene() -> (SceneData, MrtAuxData) {
    // Two tets sharing a face, centered at origin, radius ~1
    let vertices = vec![
        0.0, 0.0, 1.0,   // 0: top
        1.0, 0.0, -0.5,  // 1: front-right
       -1.0, 0.0, -0.5,  // 2: front-left
        0.0, 1.0, -0.5,  // 3: back-up
        0.0, -1.0, -0.5, // 4: back-down
    ];
    let indices = vec![
        0, 1, 2, 3,  // tet 0
        0, 1, 2, 4,  // tet 1
    ];
    let densities = vec![2.0, 3.0];
    let color_grads = vec![0.1, 0.0, 0.0, 0.0, 0.1, 0.0]; // per-tet [3]

    let scene = build_test_scene(vertices, indices, densities, color_grads);
    let n = scene.tet_count as usize;
    let v = scene.vertex_count as usize;

    let mut aux_flat = vec![0.0f32; n * 8];
    for t in 0..n {
        let base = t * 8;
        aux_flat[base]     = 0.3 + 0.2 * t as f32; // roughness
        aux_flat[base + 1] = 0.1 + 0.1 * t as f32; // env_f0
        aux_flat[base + 2] = 0.25;                  // env_f1
        aux_flat[base + 3] = 0.4;                   // env_f2
        aux_flat[base + 4] = 0.6;                   // env_f3
        aux_flat[base + 5] = 0.8;                   // albedo_r
        aux_flat[base + 6] = 0.5;                   // albedo_g
        aux_flat[base + 7] = 0.3;                   // albedo_b
    }

    // Vertex normals pointing +Z
    let mut vertex_normals = vec![0.0f32; v * 3];
    for vi in 0..v {
        vertex_normals[vi * 3 + 2] = 1.0;
    }

    (scene, MrtAuxData { aux_flat, vertex_normals })
}

fn setup_camera(eye: Vec3, target: Vec3) -> (glam::Mat4, glam::Mat3, [f32; 4]) {
    let aspect = W as f32 / H as f32;
    let fov = std::f32::consts::FRAC_PI_2;
    let proj = perspective_matrix(fov, aspect, 0.01, 100.0);
    let view = look_at(eye, target, Vec3::new(0.0, 0.0, 1.0));
    let vp = proj * view;
    let (c2w, intrinsics) = test_camera_c2w_intrinsics(eye, target, fov, W as f32, H as f32);
    (vp, c2w, intrinsics)
}

#[test]
fn test_mrt_compositing() {
    let (scene, aux) = make_mrt_test_scene();

    let cam_pos = Vec3::new(0.0, -3.0, 0.0);
    let target = Vec3::ZERO;
    let (vp, c2w, intrinsics) = setup_camera(cam_pos, target);

    // CPU reference
    let cpu = cpu_render_scene_with_mrt(&scene, &aux, cam_pos, vp, c2w, intrinsics, W, H);

    let cpu_alpha_sum: f32 = cpu.color.iter().map(|p| p[3]).sum();
    println!("CPU: color alpha sum = {cpu_alpha_sum:.4}");
    assert!(cpu_alpha_sum > 0.0, "CPU color is all zero — scene not visible");

    // GPU
    let gpu = match gpu_compute_interval_render_scene_with_mrt(
        &scene, &aux, cam_pos, vp, c2w, intrinsics, W, H,
    ) {
        Some(g) => g,
        None => {
            println!("No GPU adapter — skipping GPU comparison");
            return;
        }
    };

    let targets = [
        ("color",   &cpu.color,   &gpu.color),
        ("aux0",    &cpu.aux0,    &gpu.aux0),
        ("normals", &cpu.normals, &gpu.normals),
        ("albedo",  &cpu.albedo,  &gpu.albedo),
    ];

    let mut any_zero = false;
    for (name, cpu_img, gpu_img) in &targets {
        let (max_d, mean_d, _) = compare_images(cpu_img, gpu_img);
        let gpu_sum: f32 = gpu_img.iter().flat_map(|p| p.iter()).sum();
        let cpu_sum: f32 = cpu_img.iter().flat_map(|p| p.iter()).sum();
        println!("{name:>8}: max_diff={max_d:.6}, mean_diff={mean_d:.6}, cpu_sum={cpu_sum:.2}, gpu_sum={gpu_sum:.2}");
        if gpu_sum == 0.0 && cpu_sum > 0.0 {
            println!("  *** GPU {name} is ALL ZERO");
            any_zero = true;
        }
    }

    // Color must match closely
    let (max_d, mean_d, _) = compare_images(&cpu.color, &gpu.color);
    assert!(max_d < 0.05, "Color max diff too large: {max_d}");
    assert!(mean_d < 0.01, "Color mean diff too large: {mean_d}");

    // MRT must be non-zero
    assert!(!any_zero, "One or more GPU MRT targets are all zero");

    // Compare aux channels
    for (name, cpu_img, gpu_img) in &targets {
        let (max_d, mean_d, _) = compare_images(cpu_img, gpu_img);
        assert!(max_d < 0.15, "{name} max diff too large: {max_d}");
        assert!(mean_d < 0.03, "{name} mean diff too large: {mean_d}");
    }

    println!("All MRT channels match CPU reference within tolerance.");
}
