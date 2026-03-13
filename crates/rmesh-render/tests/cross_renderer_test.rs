//! Cross-renderer forward pass comparison tests.
//!
//! Compares all 4 renderers (CPU reference, GPU hw raster, GPU raytrace,
//! GPU tiled compute) against each other to ensure they produce matching output.
//!
//! These tests are the primary way to catch rendering discrepancies between
//! the different GPU pipelines and the CPU reference implementation.
//!
//! Run: `cargo test -p rmesh-render --test cross_renderer_test -- --nocapture`

mod common;

use common::*;
use glam::Vec3;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const SEED: u64 = 189710234;
const W: u32 = 64;
const H: u32 = 64;

fn setup_camera(eye: Vec3, target: Vec3) -> (glam::Mat4, glam::Mat4) {
    let aspect = W as f32 / H as f32;
    let proj = perspective_matrix(std::f32::consts::FRAC_PI_2, aspect, 0.01, 100.0);
    let view = look_at(eye, target, Vec3::new(0.0, 0.0, 1.0));
    let vp = proj * view;
    let inv_vp = vp.inverse();
    (vp, inv_vp)
}

/// Helper: render a scene with all available renderers and return results.
/// Returns (cpu, gpu_hw_raster, gpu_raytrace, gpu_tiled).
/// GPU results are None if no adapter available.
fn render_all(
    scene: &rmesh_data::SceneData,
    eye: Vec3,
    target: Vec3,
) -> (
    Vec<[f32; 4]>,
    Option<Vec<[f32; 4]>>,
    Option<Vec<[f32; 4]>>,
    Option<Vec<[f32; 4]>>,
) {
    let (vp, inv_vp) = setup_camera(eye, target);

    let cpu = cpu_render_scene(scene, eye, vp, inv_vp, W, H);
    let hw_raster = gpu_render_scene(scene, eye, vp, inv_vp, W, H);
    let raytrace = gpu_raytrace_scene(scene, eye, vp, inv_vp, W, H);
    let tiled = gpu_tiled_render_scene(scene, eye, vp, inv_vp, W, H);

    (cpu, hw_raster, raytrace, tiled)
}

/// Helper: print per-pixel details for the first N differing pixels.
fn print_pixel_diffs(
    name_a: &str,
    name_b: &str,
    a: &[[f32; 4]],
    b: &[[f32; 4]],
    threshold: f32,
    max_print: usize,
) {
    let mut count = 0;
    for (i, (pa, pb)) in a.iter().zip(b.iter()).enumerate() {
        let max_ch_diff = (0..4)
            .map(|ch| (pa[ch] - pb[ch]).abs())
            .fold(0.0f32, f32::max);
        if max_ch_diff > threshold {
            let px = i as u32 % W;
            let py = i as u32 / W;
            if count < max_print {
                eprintln!(
                    "  pixel ({px},{py}): {name_a}={:.4?} {name_b}={:.4?} diff={max_ch_diff:.6}",
                    pa, pb
                );
            }
            count += 1;
        }
    }
    if count > max_print {
        eprintln!("  ... and {} more pixels above threshold", count - max_print);
    }
}

// ===========================================================================
// Single-tet cross-renderer tests
// ===========================================================================

/// Single tet, camera outside looking in: compare all 4 renderers.
#[test]
fn test_single_tet_all_renderers_outside() {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = random_single_tet_scene(&mut rng, 0.3);

    let verts = load_tet_verts(&scene, 0);
    let centroid = (verts[0] + verts[1] + verts[2] + verts[3]) * 0.25;
    let eye = centroid + Vec3::new(2.0, 0.0, 0.0);

    let (cpu, hw_raster, raytrace, tiled) = render_all(&scene, eye, centroid);

    // CPU should have non-zero content
    let total_alpha: f32 = cpu.iter().map(|p| p[3]).sum();
    assert!(total_alpha > 0.1, "CPU image is all-zero");

    // Compare CPU vs each GPU renderer
    if let Some(ref hw) = hw_raster {
        let (max_diff, mean_diff, _) = compare_images(&cpu, hw);
        eprintln!("CPU vs HW raster: max={max_diff:.6}, mean={mean_diff:.8}");
        if mean_diff > 0.01 {
            print_pixel_diffs("cpu", "hw", &cpu, hw, 0.01, 5);
        }
        // HW raster uses f16 texture, so tolerance is higher
        assert!(
            mean_diff < 0.05,
            "CPU vs HW raster: mean_diff {mean_diff} >= 0.05"
        );
    }

    if let Some(ref rt) = raytrace {
        let (max_diff, mean_diff, _) = compare_images(&cpu, rt);
        eprintln!("CPU vs raytrace:  max={max_diff:.6}, mean={mean_diff:.8}");
        if mean_diff > 0.005 {
            print_pixel_diffs("cpu", "rt", &cpu, rt, 0.005, 5);
        }
        assert!(
            mean_diff < 0.01,
            "CPU vs raytrace: mean_diff {mean_diff} >= 0.01"
        );
    }

    if let Some(ref ti) = tiled {
        let (max_diff, mean_diff, _) = compare_images(&cpu, ti);
        eprintln!("CPU vs tiled:     max={max_diff:.6}, mean={mean_diff:.8}");
        if mean_diff > 0.005 {
            print_pixel_diffs("cpu", "tiled", &cpu, ti, 0.005, 5);
        }
        assert!(
            mean_diff < 0.01,
            "CPU vs tiled: mean_diff {mean_diff} >= 0.01"
        );
    }

    // GPU vs GPU comparisons (should be tighter than CPU vs GPU)
    if let (Some(ref rt), Some(ref ti)) = (&raytrace, &tiled) {
        let (max_diff, mean_diff, _) = compare_images(rt, ti);
        eprintln!("raytrace vs tiled: max={max_diff:.6}, mean={mean_diff:.8}");
        if mean_diff > 0.001 {
            print_pixel_diffs("rt", "tiled", rt, ti, 0.001, 5);
        }
        assert!(
            mean_diff < 0.005,
            "raytrace vs tiled: mean_diff {mean_diff} >= 0.005"
        );
    }

    if let (Some(ref hw), Some(ref ti)) = (&hw_raster, &tiled) {
        let (max_diff, mean_diff, _) = compare_images(hw, ti);
        eprintln!("HW raster vs tiled: max={max_diff:.6}, mean={mean_diff:.8}");
        // HW raster is f16, tiled is f32, so differences expected
        assert!(
            mean_diff < 0.05,
            "HW raster vs tiled: mean_diff {mean_diff} >= 0.05"
        );
    }
}

/// Known deterministic tet scene: all renderers should match tightly.
/// Uses zero color gradients for maximum predictability.
#[test]
fn test_known_tet_all_renderers() {
    let scene = known_single_tet_scene();

    let eye = Vec3::new(3.0, 0.0, 0.0);
    let target = Vec3::ZERO;

    let (cpu, _hw_raster, raytrace, tiled) = render_all(&scene, eye, target);

    let total_alpha: f32 = cpu.iter().map(|p| p[3]).sum();
    assert!(total_alpha > 0.1, "CPU image is all-zero for known tet");

    eprintln!("=== Known tet scene (zero color_grads, density=3.0) ===");

    if let Some(ref ti) = tiled {
        let (max_diff, mean_diff, _) = compare_images(&cpu, ti);
        eprintln!("CPU vs tiled:     max={max_diff:.6}, mean={mean_diff:.8}");
        if mean_diff > 0.005 {
            print_pixel_diffs("cpu", "tiled", &cpu, ti, 0.005, 10);
        }
        assert!(
            mean_diff < 0.01,
            "CPU vs tiled (known tet): mean_diff {mean_diff} >= 0.01"
        );
    }

    if let Some(ref rt) = raytrace {
        let (max_diff, mean_diff, _) = compare_images(&cpu, rt);
        eprintln!("CPU vs raytrace:  max={max_diff:.6}, mean={mean_diff:.8}");
        assert!(
            mean_diff < 0.01,
            "CPU vs raytrace (known tet): mean_diff {mean_diff} >= 0.01"
        );
    }

    if let (Some(ref rt), Some(ref ti)) = (&raytrace, &tiled) {
        let (max_diff, mean_diff, _) = compare_images(rt, ti);
        eprintln!("raytrace vs tiled: max={max_diff:.6}, mean={mean_diff:.8}");
        assert!(
            mean_diff < 0.005,
            "raytrace vs tiled (known tet): mean_diff {mean_diff} >= 0.005"
        );
    }
}

/// Single tet, multiple camera angles: compare all renderers at each angle.
#[test]
fn test_single_tet_multi_angle_cross_renderer() {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = random_single_tet_scene(&mut rng, 0.3);

    let verts = load_tet_verts(&scene, 0);
    let centroid = (verts[0] + verts[1] + verts[2] + verts[3]) * 0.25;

    let viewpoints = [
        (centroid + Vec3::new(2.0, 0.0, 0.0), "along_x"),
        (centroid + Vec3::new(0.0, 2.0, 0.0), "along_y"),
        (centroid + Vec3::new(0.0, 0.0, 2.0), "along_z"),
        (centroid + Vec3::new(1.5, 1.5, 1.5), "diagonal"),
    ];

    for (eye, label) in &viewpoints {
        let (vp, inv_vp) = setup_camera(*eye, centroid);

        let cpu = cpu_render_scene(&scene, *eye, vp, inv_vp, W, H);
        let total_alpha: f32 = cpu.iter().map(|p| p[3]).sum();
        if total_alpha < 0.01 {
            eprintln!("{label}: tet not visible, skipping");
            continue;
        }

        if let Some(tiled) = gpu_tiled_render_scene(&scene, *eye, vp, inv_vp, W, H) {
            let (max_diff, mean_diff, _) = compare_images(&cpu, &tiled);
            eprintln!("{label}: CPU vs tiled: max={max_diff:.6}, mean={mean_diff:.8}");
            assert!(
                mean_diff < 0.01,
                "{label}: CPU vs tiled: mean_diff {mean_diff} >= 0.01"
            );
        } else {
            eprintln!("Skipping GPU test (no adapter)");
            return;
        }

        if let Some(raytrace) = gpu_raytrace_scene(&scene, *eye, vp, inv_vp, W, H) {
            let (max_diff, mean_diff, _) = compare_images(&cpu, &raytrace);
            eprintln!("{label}: CPU vs raytrace: max={max_diff:.6}, mean={mean_diff:.8}");
            assert!(
                mean_diff < 0.01,
                "{label}: CPU vs raytrace: mean_diff {mean_diff} >= 0.01"
            );
        }
    }
}

// ===========================================================================
// Multi-tet cross-renderer tests
// ===========================================================================

/// Two tets sharing a face — tests sorting + compositing across renderers.
#[test]
fn test_two_tet_cross_renderer() {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);

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

    let scene = build_test_scene(vertices, indices, densities, color_grads);

    // Camera outside, looking at center.
    // Avoid looking along Z axis since up=(0,0,1) creates degenerate look_at.
    let eye = Vec3::new(3.0, 0.4, 1.0);
    let target = Vec3::new(0.5, 0.4, 0.0);
    let (vp, inv_vp) = setup_camera(eye, target);

    let cpu = cpu_render_scene(&scene, eye, vp, inv_vp, W, H);
    let total_alpha: f32 = cpu.iter().map(|p| p[3]).sum();
    assert!(total_alpha > 0.01, "CPU image is all-zero");

    eprintln!("=== Two-tet cross-renderer test ===");

    if let Some(tiled) = gpu_tiled_render_scene(&scene, eye, vp, inv_vp, W, H) {
        let (max_diff, mean_diff, _) = compare_images(&cpu, &tiled);
        eprintln!("CPU vs tiled:     max={max_diff:.6}, mean={mean_diff:.8}");
        if mean_diff > 0.005 {
            print_pixel_diffs("cpu", "tiled", &cpu, &tiled, 0.01, 5);
        }
        assert!(
            mean_diff < 0.01,
            "Two-tet CPU vs tiled: mean_diff {mean_diff} >= 0.01"
        );
    } else {
        eprintln!("Skipping GPU test (no adapter)");
        return;
    }

    if let Some(raytrace) = gpu_raytrace_scene(&scene, eye, vp, inv_vp, W, H) {
        let (max_diff, mean_diff, _) = compare_images(&cpu, &raytrace);
        eprintln!("CPU vs raytrace:  max={max_diff:.6}, mean={mean_diff:.8}");
        assert!(
            mean_diff < 0.01,
            "Two-tet CPU vs raytrace: mean_diff {mean_diff} >= 0.01"
        );
    }
}

/// Four tets forming a larger shape — full multi-tet compositing.
#[test]
fn test_four_tet_cross_renderer() {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);

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
    let densities: Vec<f32> = (0..tet_count)
        .map(|_| rng.random::<f32>() * 3.0 + 0.5)
        .collect();
    let color_grads: Vec<f32> = (0..tet_count * 3)
        .map(|_| (rng.random::<f32>() - 0.5) * 0.1)
        .collect();

    let scene = build_test_scene(vertices, indices, densities, color_grads);

    let viewpoints = [
        (Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO, "from_x"),
        (Vec3::new(2.0, 2.0, 2.0), Vec3::ZERO, "diagonal"),
    ];

    for (eye, target, label) in &viewpoints {
        let (vp, inv_vp) = setup_camera(*eye, *target);

        let cpu = cpu_render_scene(&scene, *eye, vp, inv_vp, W, H);
        let total_alpha: f32 = cpu.iter().map(|p| p[3]).sum();
        if total_alpha < 0.01 {
            eprintln!("  {label}: not visible, skipping");
            continue;
        }

        if let Some(tiled) = gpu_tiled_render_scene(&scene, *eye, vp, inv_vp, W, H) {
            let (max_diff, mean_diff, _) = compare_images(&cpu, &tiled);
            eprintln!("  {label}: CPU vs tiled: max={max_diff:.6}, mean={mean_diff:.8}");
            assert!(
                mean_diff < 0.01,
                "Four-tet {label}: CPU vs tiled: mean_diff {mean_diff} >= 0.01"
            );
        } else {
            eprintln!("Skipping GPU test (no adapter)");
            return;
        }

        if let Some(raytrace) = gpu_raytrace_scene(&scene, *eye, vp, inv_vp, W, H) {
            let (max_diff, mean_diff, _) = compare_images(&cpu, &raytrace);
            eprintln!("  {label}: CPU vs raytrace: max={max_diff:.6}, mean={mean_diff:.8}");
            assert!(
                mean_diff < 0.01,
                "Four-tet {label}: CPU vs raytrace: mean_diff {mean_diff} >= 0.01"
            );
        }
    }
}

// ===========================================================================
// Edge case tests
// ===========================================================================

/// Tet at image boundary: only partially in view.
/// Tests tile assignment when tet spans the frustum edge.
#[test]
fn test_tet_at_boundary_cross_renderer() {
    let scene = known_single_tet_scene();

    // Camera aimed so tet is at the edge of the frame
    let eye = Vec3::new(3.0, 1.5, 0.0);
    let target = Vec3::new(0.0, 1.5, 0.0); // looking past the tet
    let (vp, inv_vp) = setup_camera(eye, target);

    let cpu = cpu_render_scene(&scene, eye, vp, inv_vp, W, H);

    if let Some(tiled) = gpu_tiled_render_scene(&scene, eye, vp, inv_vp, W, H) {
        let (max_diff, mean_diff, _) = compare_images(&cpu, &tiled);
        eprintln!("boundary: CPU vs tiled: max={max_diff:.6}, mean={mean_diff:.8}");
        // Partial coverage can cause edge differences in tiling
        assert!(
            mean_diff < 0.02,
            "boundary: CPU vs tiled: mean_diff {mean_diff} >= 0.02"
        );
    } else {
        eprintln!("Skipping GPU test (no adapter)");
    }
}

/// Tet behind camera: should produce all-zero image in all renderers.
#[test]
fn test_tet_behind_camera_cross_renderer() {
    let scene = known_single_tet_scene();

    // Camera looking away from the tet
    let eye = Vec3::new(3.0, 0.0, 0.0);
    let target = Vec3::new(10.0, 0.0, 0.0); // looking away
    let (vp, inv_vp) = setup_camera(eye, target);

    let cpu = cpu_render_scene(&scene, eye, vp, inv_vp, W, H);
    let cpu_alpha: f32 = cpu.iter().map(|p| p[3]).sum();
    assert!(
        cpu_alpha < 0.01,
        "CPU should show nothing when tet is behind camera, alpha={cpu_alpha}"
    );

    if let Some(tiled) = gpu_tiled_render_scene(&scene, eye, vp, inv_vp, W, H) {
        let tiled_alpha: f32 = tiled.iter().map(|p| p[3]).sum();
        assert!(
            tiled_alpha < 0.01,
            "Tiled should show nothing when tet is behind camera, alpha={tiled_alpha}"
        );
    }

    if let Some(raytrace) = gpu_raytrace_scene(&scene, eye, vp, inv_vp, W, H) {
        let rt_alpha: f32 = raytrace.iter().map(|p| p[3]).sum();
        assert!(
            rt_alpha < 0.01,
            "Raytrace should show nothing when tet is behind camera, alpha={rt_alpha}"
        );
    }
}

/// Far away tet: very small projected area.
/// Tests sub-pixel tet handling across renderers.
#[test]
fn test_distant_tet_cross_renderer() {
    let scene = known_single_tet_scene();

    let eye = Vec3::new(50.0, 0.0, 0.0);
    let target = Vec3::ZERO;
    let (vp, inv_vp) = setup_camera(eye, target);

    let cpu = cpu_render_scene(&scene, eye, vp, inv_vp, W, H);

    // At distance 50, the tet (radius ~1) subtends a very small angle.
    // It should either hit 0-1 pixels consistently across renderers.
    if let Some(tiled) = gpu_tiled_render_scene(&scene, eye, vp, inv_vp, W, H) {
        let (_, mean_diff, _) = compare_images(&cpu, &tiled);
        eprintln!("distant: CPU vs tiled: mean={mean_diff:.8}");
        // Both should be near-zero (tet is tiny or culled)
        assert!(
            mean_diff < 0.01,
            "distant: CPU vs tiled: mean_diff {mean_diff} >= 0.01"
        );
    }
}

// ===========================================================================
// Single-pixel validation
// ===========================================================================

/// Validate a specific center pixel's value across all renderers.
/// Uses known geometry with zero color gradient for analytical predictability.
#[test]
fn test_center_pixel_cross_renderer() {
    let scene = known_single_tet_scene();

    let eye = Vec3::new(3.0, 0.0, 0.0);
    let target = Vec3::ZERO;
    let (vp, inv_vp) = setup_camera(eye, target);

    let cpu = cpu_render_scene(&scene, eye, vp, inv_vp, W, H);

    let center_idx = ((H / 2) * W + (W / 2)) as usize;
    let cpu_pixel = cpu[center_idx];

    eprintln!("Center pixel CPU: RGBA={:.6?}", cpu_pixel);

    // The center pixel should have non-trivial alpha (tet is at center)
    assert!(
        cpu_pixel[3] > 0.01,
        "Center pixel should have nonzero alpha, got {}",
        cpu_pixel[3]
    );

    // Premultiplied color should be non-negative
    for ch in 0..3 {
        assert!(
            cpu_pixel[ch] >= -1e-6,
            "Center pixel channel {ch} is negative: {}",
            cpu_pixel[ch]
        );
    }

    // Compare with GPU renderers at this specific pixel
    if let Some(tiled) = gpu_tiled_render_scene(&scene, eye, vp, inv_vp, W, H) {
        let tiled_pixel = tiled[center_idx];
        eprintln!("Center pixel tiled: RGBA={:.6?}", tiled_pixel);
        for ch in 0..4 {
            let diff = (cpu_pixel[ch] - tiled_pixel[ch]).abs();
            assert!(
                diff < 0.01,
                "Center pixel ch{ch}: CPU={:.6} tiled={:.6} diff={diff:.6}",
                cpu_pixel[ch], tiled_pixel[ch]
            );
        }
    }

    if let Some(raytrace) = gpu_raytrace_scene(&scene, eye, vp, inv_vp, W, H) {
        let rt_pixel = raytrace[center_idx];
        eprintln!("Center pixel raytrace: RGBA={:.6?}", rt_pixel);
        for ch in 0..4 {
            let diff = (cpu_pixel[ch] - rt_pixel[ch]).abs();
            assert!(
                diff < 0.01,
                "Center pixel ch{ch}: CPU={:.6} raytrace={:.6} diff={diff:.6}",
                cpu_pixel[ch], rt_pixel[ch]
            );
        }
    }

    if let Some(hw) = gpu_render_scene(&scene, eye, vp, inv_vp, W, H) {
        let hw_pixel = hw[center_idx];
        eprintln!("Center pixel HW raster: RGBA={:.6?}", hw_pixel);
        for ch in 0..4 {
            let diff = (cpu_pixel[ch] - hw_pixel[ch]).abs();
            // Wider tolerance for f16
            assert!(
                diff < 0.02,
                "Center pixel ch{ch}: CPU={:.6} HW={:.6} diff={diff:.6}",
                cpu_pixel[ch], hw_pixel[ch]
            );
        }
    }
}

/// Scan across all pixels: find the pixel with maximum divergence between renderers.
/// Useful for debugging — identifies the worst-case discrepancy location.
#[test]
fn test_worst_pixel_divergence() {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = random_single_tet_scene(&mut rng, 0.3);

    let verts = load_tet_verts(&scene, 0);
    let centroid = (verts[0] + verts[1] + verts[2] + verts[3]) * 0.25;
    let eye = centroid + Vec3::new(2.0, 0.0, 0.0);
    let (vp, inv_vp) = setup_camera(eye, centroid);

    let cpu = cpu_render_scene(&scene, eye, vp, inv_vp, W, H);

    if let Some(tiled) = gpu_tiled_render_scene(&scene, eye, vp, inv_vp, W, H) {
        let mut worst_diff = 0.0f32;
        let mut worst_idx = 0;

        for (i, (c, t)) in cpu.iter().zip(tiled.iter()).enumerate() {
            let max_ch = (0..4)
                .map(|ch| (c[ch] - t[ch]).abs())
                .fold(0.0f32, f32::max);
            if max_ch > worst_diff {
                worst_diff = max_ch;
                worst_idx = i;
            }
        }

        let px = worst_idx as u32 % W;
        let py = worst_idx as u32 / W;
        eprintln!(
            "Worst pixel divergence (CPU vs tiled): ({px},{py}) diff={worst_diff:.6}"
        );
        eprintln!(
            "  CPU:   {:?}",
            cpu[worst_idx]
        );
        eprintln!(
            "  Tiled: {:?}",
            tiled[worst_idx]
        );

        // Log overall stats
        let (max_diff, mean_diff, _) = compare_images(&cpu, &tiled);
        eprintln!("Overall: max={max_diff:.6}, mean={mean_diff:.8}");
    } else {
        eprintln!("Skipping GPU test (no adapter)");
    }
}
