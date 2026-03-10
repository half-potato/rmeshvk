"""Test torch.autograd gradient correctness via finite differences.

Creates a single-tet scene, runs forward_tiled + backward through the
RMeshForward autograd function, and compares analytical gradients against
numerical (finite-difference) gradients.
"""

import numpy as np
import torch
from rmesh_wgpu import RMeshRenderer
from rmesh_wgpu.autograd import RMeshForward


def compute_circumsphere(v0, v1, v2, v3):
    """Compute circumsphere center and radius^2 for a tetrahedron."""
    a = v1 - v0
    b = v2 - v0
    c = v3 - v0
    aa = np.dot(a, a)
    bb = np.dot(b, b)
    cc = np.dot(c, c)
    cross_bc = np.cross(b, c)
    cross_ca = np.cross(c, a)
    cross_ab = np.cross(a, b)
    denom = 2.0 * np.dot(a, cross_bc)
    if abs(denom) < 1e-12:
        denom = 1.0
    r = (aa * cross_bc + bb * cross_ca + cc * cross_ab) / denom
    center = v0 + r
    r_sq = np.dot(r, r)
    return np.array([center[0], center[1], center[2], r_sq], dtype=np.float32)


def make_single_tet_scene(seed=42):
    """Create a deterministic single-tet scene matching the Rust test setup."""
    rng = np.random.RandomState(seed)
    radius = 0.6

    # Random vertices in [-radius, radius]^3
    verts = (rng.rand(4, 3).astype(np.float32) - 0.5) * 2.0 * radius

    # Fix winding (ensure positive orientation)
    v0, v1, v2, v3 = verts[0], verts[1], verts[2], verts[3]
    det = np.dot(v1 - v0, np.cross(v2 - v0, v3 - v0))
    if det < 0:
        indices = np.array([0, 1, 3, 2], dtype=np.uint32)
    else:
        indices = np.array([0, 1, 2, 3], dtype=np.uint32)

    # SH degree 0: 1 coefficient per color channel = 3 total
    sh_coeffs = (rng.rand(3).astype(np.float32) * 2.0 - 1.0)
    densities = np.array([rng.rand() * 5.0 + 0.5], dtype=np.float32)
    color_grads = ((rng.rand(3).astype(np.float32) - 0.5) * 0.2)

    # Circumsphere
    i0, i1, i2, i3 = indices
    circumdata = compute_circumsphere(verts[i0], verts[i1], verts[i2], verts[i3])

    return {
        "vertices": verts.ravel(),
        "indices": indices,
        "sh_coeffs": sh_coeffs,
        "densities": densities,
        "color_grads": color_grads,
        "circumdata": circumdata,
    }


def setup_camera(width, height, distance=1.0):
    """Create a camera looking at the origin from distance along +Z.

    Uses the GS/COLMAP convention where camera Z points forward,
    so in-front objects have positive view_z and clip.w > 0.
    The VP matrix is stored in the same row-vector format as the
    training code (vp = Rt^T @ P^T, sent as .ravel()).
    """
    cam_pos = np.array([0.0, 0.0, distance], dtype=np.float32)

    # COLMAP convention: R maps world→camera, rows = camera axes in world
    # Camera at (0,0,d) looking at origin:
    #   right = (1,0,0), down = (0,-1,0), forward = (0,0,-1)
    R = np.array([[1, 0, 0],
                  [0, -1, 0],
                  [0, 0, -1]], dtype=np.float32)
    t = -R @ cam_pos  # = [0, 0, d]

    # GS view matrix: Rt = [R^T | t; 0 0 0 1]
    Rt = np.eye(4, dtype=np.float32)
    Rt[:3, :3] = R.T
    Rt[:3, 3] = t

    # GS projection (z_sign=1, matching getProjectionMatrix)
    fov = 45.0 * np.pi / 180.0
    aspect = width / height
    near, far = 0.01, 100.0
    f_val = 1.0 / np.tan(fov / 2.0)

    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f_val / aspect
    proj[1, 1] = f_val
    proj[2, 2] = far / (far - near)
    proj[2, 3] = -(far * near) / (far - near)
    proj[3, 2] = 1.0

    # GS convention: vp = Rt^T @ proj^T (row-vector format)
    # Shader loads rows as columns → VP_wgsl = vp^T = proj @ Rt
    # So clip = proj @ Rt @ pos, with clip.w > 0 for in-front objects
    vp = Rt.T @ proj.T

    inv_vp = np.linalg.inv(vp).astype(np.float32)

    return cam_pos, vp.ravel().astype(np.float32), inv_vp.ravel().astype(np.float32)


def run_forward_loss(renderer, cam_pos, vp, inv_vp, gt_image):
    """Run forward_tiled and compute L2 loss in float64."""
    cam_np = cam_pos.detach().cpu().numpy().ravel().astype(np.float32)
    vp_np = vp.detach().cpu().numpy().ravel().astype(np.float32)
    inv_vp_np = inv_vp.detach().cpu().numpy().ravel().astype(np.float32)

    image_np = renderer.forward_tiled(cam_np, vp_np, inv_vp_np)
    image = image_np[:, :, :3].astype(np.float64)  # RGB only
    gt = gt_image.astype(np.float64)
    diff = image - gt
    loss = 0.5 * np.sum(diff * diff)
    return loss, image_np


def test_autograd_finite_diff():
    """Compare autograd backward gradients against finite-difference gradients."""
    W, H = 32, 32
    scene = make_single_tet_scene()

    cam_pos_np, vp_np, inv_vp_np = setup_camera(W, H, distance=1.0)

    # Create renderer
    renderer = RMeshRenderer(
        scene["vertices"],
        scene["indices"],
        scene["sh_coeffs"],
        scene["densities"],
        scene["color_grads"],
        scene["circumdata"],
        0,  # sh_degree
        W, H,
    )

    # Create tensors for autograd
    cam_pos = torch.tensor(cam_pos_np, dtype=torch.float32)
    vp = torch.tensor(vp_np, dtype=torch.float32)
    inv_vp = torch.tensor(inv_vp_np, dtype=torch.float32)
    vertices = torch.tensor(scene["vertices"], dtype=torch.float32, requires_grad=True)
    sh_coeffs = torch.tensor(scene["sh_coeffs"], dtype=torch.float32, requires_grad=True)
    densities = torch.tensor(scene["densities"], dtype=torch.float32, requires_grad=True)
    color_grads = torch.tensor(scene["color_grads"], dtype=torch.float32, requires_grad=True)

    # GT image: black (so background doesn't contribute loss)
    gt_image = np.zeros((H, W, 3), dtype=np.float32)

    # --- Analytical gradients via autograd ---
    image = RMeshForward.apply(renderer, cam_pos, vp, inv_vp, vertices, sh_coeffs, densities, color_grads)
    # L2 loss on RGB channels only
    image_rgb = image[:, :, :3]
    gt_tensor = torch.tensor(gt_image, dtype=torch.float32)
    loss = 0.5 * ((image_rgb - gt_tensor) ** 2).sum()
    loss.backward()

    analytical_grads = {
        "sh_coeffs": sh_coeffs.grad.numpy().copy(),
        "densities": densities.grad.numpy().copy(),
        "color_grads": color_grads.grad.numpy().copy(),
        "vertices": vertices.grad.numpy().copy(),
    }

    print(f"Analytical loss: {loss.item():.6e}")
    for name, g in analytical_grads.items():
        print(f"  {name} grad norm: {np.linalg.norm(g):.6e}, values: {g}")

    # --- Numerical gradients via finite differences ---
    eps = 1e-3
    param_groups = [
        ("sh_coeffs", scene["sh_coeffs"].copy()),
        ("densities", scene["densities"].copy()),
        ("color_grads", scene["color_grads"].copy()),
        ("vertices", scene["vertices"].copy()),
    ]

    print(f"\nFinite-difference gradients (eps={eps}):")
    all_pass = True

    for group_name, base_params in param_groups:
        numerical_grad = np.zeros_like(base_params)

        for i in range(len(base_params)):
            # +eps
            params_plus = base_params.copy()
            params_plus[i] += eps
            renderer.update_params(
                scene["vertices"] if group_name != "vertices" else params_plus,
                scene["sh_coeffs"] if group_name != "sh_coeffs" else params_plus,
                scene["densities"] if group_name != "densities" else params_plus,
                scene["color_grads"] if group_name != "color_grads" else params_plus,
            )
            loss_plus, _ = run_forward_loss(renderer, cam_pos, vp, inv_vp, gt_image)

            # -eps
            params_minus = base_params.copy()
            params_minus[i] -= eps
            renderer.update_params(
                scene["vertices"] if group_name != "vertices" else params_minus,
                scene["sh_coeffs"] if group_name != "sh_coeffs" else params_minus,
                scene["densities"] if group_name != "densities" else params_minus,
                scene["color_grads"] if group_name != "color_grads" else params_minus,
            )
            loss_minus, _ = run_forward_loss(renderer, cam_pos, vp, inv_vp, gt_image)

            numerical_grad[i] = (loss_plus - loss_minus) / (2.0 * eps)

        # Restore original params
        renderer.update_params(
            scene["vertices"], scene["sh_coeffs"],
            scene["densities"], scene["color_grads"],
        )

        analytical = analytical_grads[group_name]
        print(f"  {group_name}:")
        print(f"    analytical: {analytical}")
        print(f"    numerical:  {numerical_grad}")

        # Relative error
        for i in range(len(base_params)):
            a = analytical[i]
            n = numerical_grad[i]
            denom = max(abs(a), abs(n), 1e-7)
            rel_err = abs(a - n) / denom
            status = "PASS" if rel_err < 0.1 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"    [{i}] a={a:+.6e} n={n:+.6e} rel_err={rel_err:.4f} {status}")

    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


if __name__ == "__main__":
    passed = test_autograd_finite_diff()
    exit(0 if passed else 1)
