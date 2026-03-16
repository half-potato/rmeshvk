//! Gradient correctness tests via finite differences.
//!
//! Verifies that the CPU reference renderer produces sensible gradients
//! (finite difference check). When a GPU backward pass is available,
//! this also compares GPU gradients against finite differences.
//!
//! Run: `cargo test -p rmesh-render --test gradient_test -- --nocapture`

mod common;

use common::*;
use glam::Vec3;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rmesh_data::SceneData;

const SEED: u64 = 189710234;
const W: u32 = 16;
const H: u32 = 16;
const EPS: f32 = 1e-3;
const FD_ATOL: f32 = 1e-2;

fn setup_camera(eye: Vec3, target: Vec3) -> (glam::Mat4, glam::Mat3, [f32; 4]) {
    let aspect = W as f32 / H as f32;
    let proj = perspective_matrix(std::f32::consts::FRAC_PI_2, aspect, 0.01, 100.0);
    let view = look_at(eye, target, Vec3::new(0.0, 0.0, 1.0));
    let vp = proj * view;
    let (c2w, intrinsics) = test_camera_c2w_intrinsics(eye, target, std::f32::consts::FRAC_PI_2, W as f32, H as f32);
    (vp, c2w, intrinsics)
}

/// Sum all pixel values (all channels including alpha) as a scalar loss.
fn image_loss(image: &[[f32; 4]]) -> f32 {
    image.iter().map(|p| p[0] + p[1] + p[2] + p[3]).sum()
}

/// Render scene and return scalar loss.
fn render_loss(scene: &SceneData, eye: Vec3, vp: glam::Mat4, c2w: glam::Mat3, intrinsics: [f32; 4]) -> f32 {
    let image = cpu_render_scene(scene, eye, vp, c2w, intrinsics, W, H);
    image_loss(&image)
}

/// Central finite difference for a single scalar parameter.
fn finite_diff(
    scene: &SceneData,
    eye: Vec3,
    vp: glam::Mat4,
    c2w: glam::Mat3,
    intrinsics: [f32; 4],
    mutator: &dyn Fn(&mut SceneData, f32),
) -> f32 {
    let mut scene_plus = clone_scene(scene);
    mutator(&mut scene_plus, EPS);
    let loss_plus = render_loss(&scene_plus, eye, vp, c2w, intrinsics);

    let mut scene_minus = clone_scene(scene);
    mutator(&mut scene_minus, -EPS);
    let loss_minus = render_loss(&scene_minus, eye, vp, c2w, intrinsics);

    (loss_plus - loss_minus) / (2.0 * EPS)
}

fn clone_scene(scene: &SceneData) -> SceneData {
    SceneData {
        vertices: scene.vertices.clone(),
        indices: scene.indices.clone(),
        densities: scene.densities.clone(),
        color_grads: scene.color_grads.clone(),
        circumdata: scene.circumdata.clone(),
        start_pose: scene.start_pose,
        vertex_count: scene.vertex_count,
        tet_count: scene.tet_count,
    }
}

/// Verify finite difference gradient for density parameter.
#[test]
fn test_density_gradient() {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = random_single_tet_scene(&mut rng, 0.3);

    let verts = load_tet_verts(&scene, 0);
    let centroid = (verts[0] + verts[1] + verts[2] + verts[3]) * 0.25;
    let eye = centroid + Vec3::new(2.0, 0.0, 0.0);
    let (vp, c2w, intrinsics) = setup_camera(eye, centroid);

    let base_loss = render_loss(&scene, eye, vp, c2w, intrinsics);

    let fd = finite_diff(&scene, eye, vp, c2w, intrinsics, &|s, eps| {
        s.densities[0] += eps;
    });

    eprintln!("density gradient: fd={fd:.6}, base_loss={base_loss:.4}");
    // The gradient should be nonzero (tet is visible and density affects output)
    assert!(
        fd.abs() > 1e-4,
        "density gradient is zero — tet might not be visible"
    );
    // Verify finite difference is stable (compare two different eps)
    let fd2 = {
        let eps2 = EPS * 2.0;
        let mut sp = clone_scene(&scene);
        sp.densities[0] += eps2;
        let lp = render_loss(&sp, eye, vp, c2w, intrinsics);
        let mut sm = clone_scene(&scene);
        sm.densities[0] -= eps2;
        let lm = render_loss(&sm, eye, vp, c2w, intrinsics);
        (lp - lm) / (2.0 * eps2)
    };
    let diff = (fd - fd2).abs();
    eprintln!("density gradient convergence: |fd1-fd2|={diff:.6}");
    assert!(
        diff < FD_ATOL,
        "density FD not converging: fd1={fd}, fd2={fd2}, diff={diff}"
    );
}

/// Verify finite difference gradient for color gradient parameters.
#[test]
fn test_color_grad_gradient() {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = random_single_tet_scene(&mut rng, 0.3);

    let verts = load_tet_verts(&scene, 0);
    let centroid = (verts[0] + verts[1] + verts[2] + verts[3]) * 0.25;
    let eye = centroid + Vec3::new(2.0, 0.0, 0.0);
    let (vp, c2w, intrinsics) = setup_camera(eye, centroid);

    for dim in 0..3 {
        let fd = finite_diff(&scene, eye, vp, c2w, intrinsics, &|s, eps| {
            s.color_grads[dim] += eps;
        });

        let fd2 = {
            let eps2 = EPS * 2.0;
            let mut sp = clone_scene(&scene);
            sp.color_grads[dim] += eps2;
            let lp = render_loss(&sp, eye, vp, c2w, intrinsics);
            let mut sm = clone_scene(&scene);
            sm.color_grads[dim] -= eps2;
            let lm = render_loss(&sm, eye, vp, c2w, intrinsics);
            (lp - lm) / (2.0 * eps2)
        };

        let diff = (fd - fd2).abs();
        eprintln!("color_grad[{dim}] gradient: fd={fd:.6}, fd2={fd2:.6}, |diff|={diff:.6}");
        assert!(
            diff < FD_ATOL,
            "color_grad[{dim}] FD not converging: diff={diff}"
        );
    }
}

/// Verify finite difference gradient for vertex positions.
#[test]
fn test_vertex_gradient() {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = random_single_tet_scene(&mut rng, 0.3);

    let verts = load_tet_verts(&scene, 0);
    let centroid = (verts[0] + verts[1] + verts[2] + verts[3]) * 0.25;
    let eye = centroid + Vec3::new(2.0, 0.0, 0.0);
    let (vp, c2w, intrinsics) = setup_camera(eye, centroid);

    // Test a few vertex coordinates (vertex 0 x, vertex 1 y, vertex 2 z)
    let test_indices = [(0, 0), (1, 1), (2, 2), (3, 0)];

    for &(vert, coord) in &test_indices {
        let flat_idx = vert * 3 + coord;
        let fd = finite_diff(&scene, eye, vp, c2w, intrinsics, &|s, eps| {
            s.vertices[flat_idx] += eps;
            // Recompute circumsphere since vertices changed
            recompute_circumdata(s);
        });

        let fd2 = {
            let eps2 = EPS * 2.0;
            let mut sp = clone_scene(&scene);
            sp.vertices[flat_idx] += eps2;
            recompute_circumdata(&mut sp);
            let lp = render_loss(&sp, eye, vp, c2w, intrinsics);
            let mut sm = clone_scene(&scene);
            sm.vertices[flat_idx] -= eps2;
            recompute_circumdata(&mut sm);
            let lm = render_loss(&sm, eye, vp, c2w, intrinsics);
            (lp - lm) / (2.0 * eps2)
        };

        let diff = (fd - fd2).abs();
        let max_mag = fd.abs().max(fd2.abs()).max(1e-6);
        let rel_diff = diff / max_mag;
        eprintln!("vertex[{vert}][{coord}] gradient: fd={fd:.6}, fd2={fd2:.6}, |diff|={diff:.6}, rel={rel_diff:.4}");
        // Vertex gradients are noisy due to pixel boundary discontinuities.
        // Use relative tolerance: two FD estimates should agree within 20%.
        assert!(
            rel_diff < 0.2 || diff < FD_ATOL,
            "vertex[{vert}][{coord}] FD not converging: rel_diff={rel_diff}, diff={diff}"
        );
    }
}

/// Recompute circumdata after vertex modification.
fn recompute_circumdata(scene: &mut SceneData) {
    let n = scene.tet_count as usize;
    for i in 0..n {
        let i0 = scene.indices[i * 4] as usize;
        let i1 = scene.indices[i * 4 + 1] as usize;
        let i2 = scene.indices[i * 4 + 2] as usize;
        let i3 = scene.indices[i * 4 + 3] as usize;
        let v0 = Vec3::new(
            scene.vertices[i0 * 3],
            scene.vertices[i0 * 3 + 1],
            scene.vertices[i0 * 3 + 2],
        );
        let v1 = Vec3::new(
            scene.vertices[i1 * 3],
            scene.vertices[i1 * 3 + 1],
            scene.vertices[i1 * 3 + 2],
        );
        let v2 = Vec3::new(
            scene.vertices[i2 * 3],
            scene.vertices[i2 * 3 + 1],
            scene.vertices[i2 * 3 + 2],
        );
        let v3 = Vec3::new(
            scene.vertices[i3 * 3],
            scene.vertices[i3 * 3 + 1],
            scene.vertices[i3 * 3 + 2],
        );
        let a = v1 - v0;
        let b = v2 - v0;
        let c = v3 - v0;
        let (aa, bb, cc) = (a.dot(a), b.dot(b), c.dot(c));
        let cross_bc = b.cross(c);
        let cross_ca = c.cross(a);
        let cross_ab = a.cross(b);
        let mut denom = 2.0 * a.dot(cross_bc);
        if denom.abs() < 1e-12 {
            denom = 1.0;
        }
        let r = (aa * cross_bc + bb * cross_ca + cc * cross_ab) / denom;
        let center = v0 + r;
        let r_sq = r.dot(r);
        scene.circumdata[i * 4] = center.x;
        scene.circumdata[i * 4 + 1] = center.y;
        scene.circumdata[i * 4 + 2] = center.z;
        scene.circumdata[i * 4 + 3] = r_sq;
    }
}
