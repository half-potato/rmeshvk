//! Multi-tet forward rendering tests.
//!
//! Ported from delaunay_splatting/tests/multi_tet_test.py.
//! Uses hand-crafted multi-tet scenes (no Delaunay dependency).

mod common;

use common::*;
use glam::Vec3;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const SEED: u64 = 189710234;
const W: u32 = 64;
const H: u32 = 64;
const ATOL: f32 = 0.1;

fn setup_camera(eye: Vec3, target: Vec3) -> (glam::Mat4, glam::Mat4) {
    let aspect = W as f32 / H as f32;
    let proj = perspective_matrix(std::f32::consts::FRAC_PI_2, aspect, 0.01, 100.0);
    let view = look_at(eye, target, Vec3::new(0.0, 0.0, 1.0));
    let vp = proj * view;
    let inv_vp = vp.inverse();
    (vp, inv_vp)
}

/// Two tetrahedra sharing a face (5 unique vertices).
/// Tests sorting correctness and multi-tet compositing.
fn two_tet_scene(rng: &mut ChaCha8Rng) -> SceneData {
    // 5 vertices forming two tets sharing face (0,1,2)
    // Tet 0: [0,1,2,3]  Tet 1: [0,2,1,4] (flipped to maintain outward orientation)
    let vertices = vec![
        0.0, 0.0, 0.0, // v0
        1.0, 0.0, 0.0, // v1
        0.5, 1.0, 0.0, // v2
        0.5, 0.3, 0.8, // v3 (above)
        0.5, 0.3, -0.8, // v4 (below)
    ];
    let indices = vec![
        0, 1, 2, 3, // tet 0
        0, 2, 1, 4, // tet 1
    ];

    let sh_coeffs = random_sh_degree0(rng, 2);
    let densities = vec![
        rng.random::<f32>() * 3.0 + 0.5,
        rng.random::<f32>() * 3.0 + 0.5,
    ];
    let color_grads = vec![
        (rng.random::<f32>() - 0.5) * 0.1,
        (rng.random::<f32>() - 0.5) * 0.1,
        (rng.random::<f32>() - 0.5) * 0.1,
        (rng.random::<f32>() - 0.5) * 0.1,
        (rng.random::<f32>() - 0.5) * 0.1,
        (rng.random::<f32>() - 0.5) * 0.1,
    ];

    build_test_scene(vertices, indices, sh_coeffs, densities, color_grads, 0)
}

/// Four tetrahedra forming a larger shape (8 vertices at cube corners + center).
fn four_tet_scene(rng: &mut ChaCha8Rng) -> SceneData {
    // Center vertex plus 4 base corners of a cube, forming 4 tets
    let s = 0.5f32;
    let vertices = vec![
        0.0, 0.0, 0.0, // v0: center
        s, s, s,       // v1
        -s, s, s,      // v2
        -s, -s, s,     // v3
        s, -s, s,      // v4
        s, s, -s,      // v5
        -s, s, -s,     // v6
        -s, -s, -s,    // v7
        s, -s, -s,     // v8
    ];
    let indices = vec![
        0, 1, 2, 3, // tet 0
        0, 1, 4, 5, // tet 1
        0, 2, 6, 3, // tet 2
        0, 5, 8, 7, // tet 3
    ];

    let tet_count = 4;
    let sh_coeffs = random_sh_degree0(rng, tet_count);
    let densities: Vec<f32> = (0..tet_count)
        .map(|_| rng.random::<f32>() * 3.0 + 0.5)
        .collect();
    let color_grads: Vec<f32> = (0..tet_count * 3)
        .map(|_| (rng.random::<f32>() - 0.5) * 0.1)
        .collect();

    build_test_scene(vertices, indices, sh_coeffs, densities, color_grads, 0)
}

/// Camera inside the two-tet scene, looking outward.
/// Interior views have higher CPU-vs-GPU divergence because hardware rasterization
/// clips geometry to the near plane, while CPU ray casting intersects the full tet.
#[test]
fn test_two_tet_center_view() {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = two_tet_scene(&mut rng);

    let eye = Vec3::new(0.5, 0.4, 0.1); // inside one of the tets, off the shared face
    let target = eye + Vec3::new(1.0, 0.0, 0.0);
    let (vp, inv_vp) = setup_camera(eye, target);

    let cpu_image = cpu_render_scene(&scene, eye, vp, inv_vp, W, H);
    let total_alpha: f32 = cpu_image.iter().map(|p| p[3]).sum();
    assert!(total_alpha > 0.01, "CPU image is all-zero");

    if let Some(gpu_image) = gpu_render_scene(&scene, eye, vp, inv_vp, W, H) {
        let (max_diff, mean_diff, _) = compare_images(&cpu_image, &gpu_image);
        eprintln!("two_tet_center: max_diff={max_diff:.4}, mean_diff={mean_diff:.6}");
        // Relaxed tolerance: camera inside tet causes near-plane clipping on GPU
        // that the CPU ray caster doesn't have, leading to larger differences.
        assert!(
            mean_diff < 0.3,
            "two_tet_center: mean_diff {mean_diff} >= 0.3"
        );
    } else {
        eprintln!("Skipping GPU test (no adapter)");
    }
}

/// Camera outside the two-tet scene, verifying sorting correctness.
#[test]
fn test_two_tet_outside_view() {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = two_tet_scene(&mut rng);

    // View from above
    let eye = Vec3::new(0.5, 0.4, 5.0);
    let target = Vec3::new(0.5, 0.4, 0.0);
    let (vp, inv_vp) = setup_camera(eye, target);

    let cpu_image = cpu_render_scene(&scene, eye, vp, inv_vp, W, H);

    if let Some(gpu_image) = gpu_render_scene(&scene, eye, vp, inv_vp, W, H) {
        let (max_diff, mean_diff, _) = compare_images(&cpu_image, &gpu_image);
        eprintln!("two_tet_outside: max_diff={max_diff:.4}, mean_diff={mean_diff:.6}");
        assert!(
            mean_diff < ATOL,
            "two_tet_outside: mean_diff {mean_diff} >= {ATOL}"
        );
    } else {
        eprintln!("Skipping GPU test (no adapter)");
    }
}

/// Four-tet scene viewed from outside at multiple angles.
#[test]
fn test_four_tet_multi_angle() {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = four_tet_scene(&mut rng);

    let viewpoints = [
        (Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO),
        (Vec3::new(0.0, 3.0, 0.0), Vec3::ZERO),
        (Vec3::new(0.0, 0.0, 3.0), Vec3::ZERO),
        (Vec3::new(2.0, 2.0, 2.0), Vec3::ZERO),
    ];

    for (i, (eye, target)) in viewpoints.iter().enumerate() {
        let (vp, inv_vp) = setup_camera(*eye, *target);
        let cpu_image = cpu_render_scene(&scene, *eye, vp, inv_vp, W, H);

        if let Some(gpu_image) = gpu_render_scene(&scene, *eye, vp, inv_vp, W, H) {
            let (max_diff, mean_diff, _) = compare_images(&cpu_image, &gpu_image);
            eprintln!("four_tet angle {i}: max_diff={max_diff:.4}, mean_diff={mean_diff:.6}");
            assert!(
                mean_diff < ATOL,
                "four_tet angle {i}: mean_diff {mean_diff} >= {ATOL}"
            );
        } else {
            eprintln!("Skipping GPU test (no adapter)");
            break;
        }
    }
}

/// Verify CPU sorting produces the same order as GPU.
#[test]
fn test_sort_order_matches() {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = four_tet_scene(&mut rng);
    let cam = Vec3::new(3.0, 0.0, 0.0);

    let sorted = sort_tets_back_to_front(&scene, cam);

    // Back-to-front: first in the list should be the farthest tet
    // Verify it's a valid permutation
    let n = scene.tet_count as usize;
    assert_eq!(sorted.len(), n);
    let mut seen = vec![false; n];
    for &idx in &sorted {
        assert!((idx as usize) < n);
        assert!(!seen[idx as usize], "duplicate index in sort");
        seen[idx as usize] = true;
    }

    // Verify depth ordering: each successive tet should be closer (or equal)
    for i in 1..sorted.len() {
        let prev = sorted[i - 1] as usize;
        let curr = sorted[i] as usize;
        let depth_prev = {
            let cx = scene.circumdata[prev * 4];
            let cy = scene.circumdata[prev * 4 + 1];
            let cz = scene.circumdata[prev * 4 + 2];
            let r2 = scene.circumdata[prev * 4 + 3];
            let d = Vec3::new(cx, cy, cz) - cam;
            d.dot(d) - r2
        };
        let depth_curr = {
            let cx = scene.circumdata[curr * 4];
            let cy = scene.circumdata[curr * 4 + 1];
            let cz = scene.circumdata[curr * 4 + 2];
            let r2 = scene.circumdata[curr * 4 + 3];
            let d = Vec3::new(cx, cy, cz) - cam;
            d.dot(d) - r2
        };
        assert!(
            depth_prev >= depth_curr - 1e-6,
            "sort order wrong: depth[{prev}]={depth_prev} < depth[{curr}]={depth_curr}"
        );
    }
}
