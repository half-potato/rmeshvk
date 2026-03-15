//! Shared camera, projection, and ray-tet intersection utilities.
//!
//! Analogous to `camera.slang`. All functions match the WGSL shader conventions exactly.

use glam::{Mat4, Vec3, Vec4};

/// Tet face winding: (a, b, c, opposite_vertex).
/// Matches WGSL FACES constant in forward_tiled, backward_tiled, etc.
pub const TET_FACES: [[usize; 4]; 4] = [[0, 2, 1, 3], [1, 2, 3, 0], [0, 3, 2, 1], [3, 0, 1, 2]];

/// Perspective projection matrix. wgpu depth [0,1], LH clip space.
pub fn perspective_matrix(fov_y_rad: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fov_y_rad / 2.0).tan();
    Mat4::from_cols(
        Vec4::new(f / aspect, 0.0, 0.0, 0.0),
        Vec4::new(0.0, f, 0.0, 0.0),
        Vec4::new(0.0, 0.0, far / (far - near), 1.0),
        Vec4::new(0.0, 0.0, -(far * near) / (far - near), 0.0),
    )
}

/// Look-at view matrix. Z-up world, forward = +Z in view space.
pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
    let f = (target - eye).normalize();
    let r = f.cross(up).normalize();
    let u = r.cross(f);

    Mat4::from_cols(
        Vec4::new(r.x, u.x, f.x, 0.0),
        Vec4::new(r.y, u.y, f.y, 0.0),
        Vec4::new(r.z, u.z, f.z, 0.0),
        Vec4::new(-r.dot(eye), -u.dot(eye), -f.dot(eye), 1.0),
    )
}

/// World → NDC. Returns (ndc_xyz, clip_w). Matches project_compute.wgsl `project_to_ndc`.
pub fn project_to_ndc(pos: Vec3, vp: Mat4) -> (Vec3, f32) {
    let clip = vp * Vec4::new(pos.x, pos.y, pos.z, 1.0);
    let inv_w = 1.0 / (clip.w + 1e-6);
    (Vec3::new(clip.x * inv_w, clip.y * inv_w, clip.z * inv_w), clip.w)
}

/// NDC → pixel. Matches WGSL: px = (ndc_x+1)*0.5*W, py = (1-ndc_y)*0.5*H.
pub fn ndc_to_pixel(ndc_x: f32, ndc_y: f32, w: f32, h: f32) -> (f32, f32) {
    let px = (ndc_x + 1.0) * 0.5 * w;
    let py = (1.0 - ndc_y) * 0.5 * h;
    (px, py)
}

/// Pixel center → NDC. Matches rasterize_compute.wgsl ray construction.
pub fn pixel_to_ndc(px: f32, py: f32, w: f32, h: f32) -> (f32, f32) {
    let ndc_x = (2.0 * px + 1.0) / w - 1.0;
    let ndc_y = 1.0 - (2.0 * py + 1.0) / h;
    (ndc_x, ndc_y)
}

/// Pixel → world-space ray (origin, direction). Matches rasterize_compute.wgsl.
pub fn pixel_ray(inv_vp: Mat4, cam_pos: Vec3, px: f32, py: f32, w: f32, h: f32) -> (Vec3, Vec3) {
    let (ndc_x, ndc_y) = pixel_to_ndc(px, py, w, h);

    let clip_near = Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
    let world_h = inv_vp * clip_near;
    let world_pos = world_h.truncate() / world_h.w;

    let dir = (world_pos - cam_pos).normalize();
    (cam_pos, dir)
}

/// Ray-tet intersection. Returns `Some((t_enter, t_exit))` or `None`.
///
/// `verts` indexed per scene winding (matching `TET_FACES`).
/// The ray direction should be normalized for correct t values.
pub fn ray_tet_intersect(origin: Vec3, dir: Vec3, verts: &[Vec3; 4]) -> Option<(f32, f32)> {
    let d = dir.length();
    if d < 1e-12 {
        return None;
    }

    let mut t_min = f32::NEG_INFINITY;
    let mut t_max = f32::INFINITY;

    for face in &TET_FACES {
        let va = verts[face[0]];
        let vb = verts[face[1]];
        let vc = verts[face[2]];
        let v_opp = verts[face[3]];
        let mut n = (vc - va).cross(vb - va);
        // Flip normal to point inward (toward opposite vertex)
        if n.dot(v_opp - va) < 0.0 {
            n = -n;
        }
        let num = n.dot(va - origin);
        let den = n.dot(dir) / d;

        if den.abs() < 1e-12 {
            continue; // parallel to face
        }

        let t = num / den;
        if den > 0.0 {
            t_min = t_min.max(t); // entering
        } else {
            t_max = t_max.min(t); // exiting
        }
    }

    if t_max > t_min && t_max > 0.0 {
        Some((t_min, t_max))
    } else {
        None
    }
}

/// Softplus activation (matches WGSL). `softplus(x) = 0.1 * ln(1 + exp(10x))`.
pub fn softplus(x: f32) -> f32 {
    if x > 8.0 {
        x
    } else {
        0.1 * (1.0 + (10.0 * x).exp()).ln()
    }
}

/// Sigmoid activation `phi(x) = (1 - exp(-x)) / x` (matches WGSL).
///
/// Note: this is the volume rendering integral helper, not the logistic sigmoid.
/// For `|x| < 1e-6`, returns the Taylor approximation `1 - x/2`.
pub fn phi(x: f32) -> f32 {
    if x.abs() < 1e-6 {
        1.0 - x * 0.5
    } else {
        (1.0 - (-x).exp()) / x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn test_vp_roundtrip() {
        let eye = Vec3::new(0.0, -2.0, 0.5);
        let target = Vec3::ZERO;
        let proj = perspective_matrix(FRAC_PI_2, 1.0, 0.01, 100.0);
        let view = look_at(eye, target, Vec3::Z);
        let vp = proj * view;
        let inv_vp = vp.inverse();
        let roundtrip = vp * inv_vp;
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (roundtrip.col(i)[j] - expected).abs() < 1e-4,
                    "vp * inv_vp [{i}][{j}] = {}, expected {expected}",
                    roundtrip.col(i)[j]
                );
            }
        }
    }

    #[test]
    fn test_ndc_pixel_roundtrip() {
        let (w, h) = (64.0, 64.0);
        for px in [0.0, 15.5, 31.0, 63.0] {
            for py in [0.0, 15.5, 31.0, 63.0] {
                let (ndc_x, ndc_y) = pixel_to_ndc(px, py, w, h);
                let (px2, py2) = ndc_to_pixel(ndc_x, ndc_y, w, h);
                assert!(
                    (px2 - (px + 0.5)).abs() < 1e-5,
                    "px roundtrip: {px} -> {ndc_x} -> {px2}"
                );
                assert!(
                    (py2 - (py + 0.5)).abs() < 1e-5,
                    "py roundtrip: {py} -> {ndc_y} -> {py2}"
                );
            }
        }
    }

    #[test]
    fn test_project_to_ndc_center() {
        let eye = Vec3::new(0.0, -3.0, 0.0);
        let target = Vec3::ZERO;
        let proj = perspective_matrix(FRAC_PI_2, 1.0, 0.01, 100.0);
        let view = look_at(eye, target, Vec3::Z);
        let vp = proj * view;

        // Target is at screen center
        let (ndc, clip_w) = project_to_ndc(target, vp);
        assert!(clip_w > 0.0, "clip_w should be positive");
        assert!(ndc.x.abs() < 0.1, "ndc_x should be near 0, got {}", ndc.x);
        assert!(ndc.y.abs() < 0.1, "ndc_y should be near 0, got {}", ndc.y);
        assert!(ndc.z >= 0.0 && ndc.z <= 1.0, "ndc_z should be in [0,1], got {}", ndc.z);
    }

    #[test]
    fn test_ray_tet_intersect_hit() {
        // Regular tet centered at origin, positive orientation (det > 0)
        let v0 = Vec3::new(1.0, 1.0, 1.0);
        let v1 = Vec3::new(-1.0, -1.0, 1.0);
        let v2 = Vec3::new(-1.0, 1.0, -1.0);
        let v3 = Vec3::new(1.0, -1.0, -1.0);
        // Verify positive orientation
        let det = (v1 - v0).dot((v2 - v0).cross(v3 - v0));
        assert!(det > 0.0, "tet must have positive orientation, det={det}");
        let verts = [v0, v1, v2, v3];

        // Ray from outside towards origin
        let origin = Vec3::new(0.0, -5.0, 0.0);
        let dir = Vec3::Y; // towards origin

        let result = ray_tet_intersect(origin, dir, &verts);
        assert!(result.is_some(), "ray should hit tet");
        let (t_enter, t_exit) = result.unwrap();
        assert!(t_enter < t_exit, "t_enter ({t_enter}) should be < t_exit ({t_exit})");
        assert!(t_enter > 0.0, "t_enter should be positive");
    }

    #[test]
    fn test_ray_tet_intersect_miss() {
        let verts = [
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
        ];

        // Ray parallel, far away
        let origin = Vec3::new(10.0, 10.0, 10.0);
        let dir = Vec3::Y;

        let result = ray_tet_intersect(origin, dir, &verts);
        assert!(result.is_none(), "ray should miss tet");
    }

    #[test]
    fn test_softplus_values() {
        assert!((softplus(0.0) - 0.1 * 2.0_f32.ln()).abs() < 1e-6);
        assert!((softplus(10.0) - 10.0).abs() < 1e-6); // linear for large x
        assert!(softplus(-10.0) < 0.01); // near zero for large negative x
    }

    #[test]
    fn test_phi_values() {
        // phi(0) ≈ 1
        assert!((phi(0.0) - 1.0).abs() < 1e-5);
        // phi(1) = (1 - e^-1) / 1 ≈ 0.6321
        assert!((phi(1.0) - 0.6321206).abs() < 1e-5);
        // phi(large) → 1/x
        assert!((phi(100.0) - 0.01).abs() < 1e-3);
    }
}
