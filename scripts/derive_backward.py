"""
Symbolic derivation and numerical validation of the backward pass
for the tetrahedral volumetric renderer.

Derives derivatives for:
  1. compute_integral (volume rendering integral)
  2. update_pixel_state (alpha blending composition)
  3. Ray-plane intersection (dt/d(vertices))
  4. Color chain (simplified: no SH, no softplus, colors are leaf params)
  5. Full chain composition test (simplified color model)

Each section prints simplified symbolic expressions and validates
them against finite differences.

Usage:
    python derive_backward.py
"""

import sympy as sp
import numpy as np
from sympy import symbols, exp, ln, Abs, Piecewise, Matrix, simplify, diff
from sympy import Function, Symbol, sqrt, cos, sin, atan2, pi
from typing import Callable


# ============================================================================
# Utility: finite difference gradient checker
# ============================================================================

def finite_diff_check(
    f_forward: Callable,
    f_backward: Callable,
    x0: np.ndarray,
    upstream_grad: np.ndarray,
    name: str = "",
    eps: float = 1e-5,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> bool:
    """Check analytical gradient against finite differences.

    Args:
        f_forward: x -> output array
        f_backward: x, upstream_grad -> gradient array (same shape as x)
        x0: point to evaluate at
        upstream_grad: dL/d(output)
        name: label for printing
        eps: finite difference step
        rtol: relative tolerance
        atol: absolute tolerance
    Returns:
        True if check passes
    """
    analytical = f_backward(x0, upstream_grad)
    numerical = np.zeros_like(x0)
    y0 = f_forward(x0)
    for i in range(len(x0)):
        x_plus = x0.copy()
        x_plus[i] += eps
        x_minus = x0.copy()
        x_minus[i] -= eps
        y_plus = f_forward(x_plus)
        y_minus = f_forward(x_minus)
        # dL/dx_i = sum_j (dL/dy_j * dy_j/dx_i)
        numerical[i] = np.sum(upstream_grad * (y_plus - y_minus) / (2 * eps))

    max_err = np.max(np.abs(analytical - numerical))
    max_scale = np.max(np.abs(numerical)) + atol
    rel_err = max_err / max_scale

    passed = rel_err < rtol or max_err < atol
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: max_abs_err={max_err:.2e}, "
          f"rel_err={rel_err:.2e}, max_grad={max_scale:.2e}")
    if not passed:
        print(f"    analytical: {analytical}")
        print(f"    numerical:  {numerical}")
    return passed


# ============================================================================
# Section 1: compute_integral backward
# ============================================================================

def section1_compute_integral():
    print("=" * 70)
    print("SECTION 1: compute_integral backward")
    print("=" * 70)

    od = Symbol('od', positive=True)

    # phi(od) = (1 - exp(-od)) / od
    phi = (1 - exp(-od)) / od

    # Volume integral weights
    alpha_t = exp(-od)
    w0 = phi - alpha_t       # weight for c_end (exit color)
    w1 = 1 - phi             # weight for c_start (entry color)

    # -- Derivatives of weights w.r.t. od --
    dw0_dod = sp.simplify(diff(w0, od))
    dw1_dod = sp.simplify(diff(w1, od))
    dalpha_dod = sp.simplify(diff(1 - alpha_t, od))  # d(1-exp(-od))/dod

    # Also derive dphi/dod for reference
    dphi_dod = sp.simplify(diff(phi, od))

    print("\nForward:")
    print(f"  phi(od) = (1 - exp(-od)) / od")
    print(f"  w0 = phi - exp(-od)")
    print(f"  w1 = 1 - phi")
    print(f"  alpha = 1 - exp(-od)")

    print(f"\nDerivatives w.r.t. od:")
    print(f"  dphi/dod  = {dphi_dod}")
    print(f"  dw0/dod   = {dw0_dod}")
    print(f"  dw1/dod   = {dw1_dod}")
    print(f"  dalpha/dod = {dalpha_dod}")

    print("\nBackward formulas:")
    print("  dL/d(c_end)   = dL/dC * w0      [elementwise vec3]")
    print("  dL/d(c_start) = dL/dC * w1      [elementwise vec3]")
    print("  dL/d(od) = dL/dC . (c_end * dw0_dod + c_start * dw1_dod)")
    print("           + dL/d(alpha) * exp(-od)")

    # -- Numerical validation --
    print("\nNumerical validation:")

    def compute_integral_fwd(x):
        """x = [c_end_r, c_end_g, c_end_b, c_start_r, c_start_g, c_start_b, od]"""
        c_end = x[0:3]
        c_start = x[3:6]
        od_val = x[6]
        if od_val < 1e-8:
            od_val = 1e-8

        alpha_t_val = np.exp(-od_val)
        if abs(od_val) < 1e-6:
            phi_val = 1.0 - od_val * 0.5
        else:
            phi_val = (1.0 - np.exp(-od_val)) / od_val
        w0_val = phi_val - alpha_t_val
        w1_val = 1.0 - phi_val
        C = w0_val * c_end + w1_val * c_start
        alpha_val = 1.0 - alpha_t_val
        return np.array([C[0], C[1], C[2], alpha_val])

    def compute_integral_bwd(x, dL_dout):
        """Analytical backward."""
        c_end = x[0:3]
        c_start = x[3:6]
        od_val = x[6]
        if od_val < 1e-8:
            od_val = 1e-8

        alpha_t_val = np.exp(-od_val)
        if abs(od_val) < 1e-6:
            phi_val = 1.0 - od_val * 0.5
        else:
            phi_val = (1.0 - np.exp(-od_val)) / od_val

        w0_val = phi_val - alpha_t_val
        w1_val = 1.0 - phi_val

        # dphi/dod
        if abs(od_val) < 1e-6:
            dphi_val = -0.5 + od_val / 3.0
        else:
            dphi_val = (np.exp(-od_val) * (1.0 + od_val) - 1.0) / (od_val * od_val)

        dw0_dod_val = dphi_val + alpha_t_val  # dphi + exp(-od) since d(-exp(-od))/dod = exp(-od)
        dw1_dod_val = -dphi_val

        dL_dC = dL_dout[0:3]
        dL_dalpha = dL_dout[3]

        dL_dc_end = dL_dC * w0_val
        dL_dc_start = dL_dC * w1_val
        dL_dod = (np.dot(dL_dC, c_end * dw0_dod_val + c_start * dw1_dod_val)
                  + dL_dalpha * alpha_t_val)

        return np.array([
            dL_dc_end[0], dL_dc_end[1], dL_dc_end[2],
            dL_dc_start[0], dL_dc_start[1], dL_dc_start[2],
            dL_dod
        ])

    np.random.seed(42)
    all_pass = True
    for trial in range(5):
        x0 = np.random.rand(7) * 2.0 + 0.1  # positive values
        upstream = np.random.randn(4)
        ok = finite_diff_check(
            compute_integral_fwd, compute_integral_bwd,
            x0, upstream, name=f"compute_integral trial {trial}"
        )
        all_pass = all_pass and ok

    return all_pass


# ============================================================================
# Section 2: update_pixel_state backward
# ============================================================================

def section2_update_pixel_state():
    print("\n" + "=" * 70)
    print("SECTION 2: update_pixel_state backward")
    print("=" * 70)

    print("\nForward (log-space formulation):")
    print("  T = exp(log_T)")
    print("  new_color = old_color + C_premultiplied * T")
    print("  new_log_T = log_T - od")

    print("\nBackward given dL/d(new_color), dL/d(new_log_T):")
    print("  dL/d(C_premultiplied) = dL/d(new_color) * T       [elementwise vec3]")
    print("  dL/d(od)              = -dL/d(new_log_T)")
    print("  dL/d(old_color)       = dL/d(new_color)            [pass-through]")
    print("  dL/d(old_log_T)       = dL/d(new_log_T) + dot(dL/d(new_color), C) * T")

    print("\nUndo (for reverse iteration):")
    print("  old_log_T = new_log_T + od")
    print("  T = exp(old_log_T)")
    print("  old_color = new_color - C_premultiplied * T")

    # -- Numerical validation --
    print("\nNumerical validation:")

    def update_pixel_fwd(x):
        """x = [old_r, old_g, old_b, log_T, C_r, C_g, C_b, od]"""
        old_color = x[0:3]
        log_T = x[3]
        C = x[4:7]
        od_val = x[7]

        T = np.exp(log_T)
        new_color = old_color + C * T
        new_log_T = log_T - od_val
        return np.array([new_color[0], new_color[1], new_color[2], new_log_T])

    def update_pixel_bwd(x, dL_dout):
        """Analytical backward."""
        old_color = x[0:3]
        log_T = x[3]
        C = x[4:7]
        od_val = x[7]

        T = np.exp(log_T)

        dL_d_new_color = dL_dout[0:3]
        dL_d_new_logT = dL_dout[3]

        dL_d_C = dL_d_new_color * T
        dL_d_od = -dL_d_new_logT
        dL_d_old_color = dL_d_new_color.copy()
        dL_d_old_logT = dL_d_new_logT + np.dot(dL_d_new_color, C) * T

        return np.array([
            dL_d_old_color[0], dL_d_old_color[1], dL_d_old_color[2],
            dL_d_old_logT,
            dL_d_C[0], dL_d_C[1], dL_d_C[2],
            dL_d_od
        ])

    np.random.seed(123)
    all_pass = True
    for trial in range(5):
        x0 = np.concatenate([
            np.random.rand(3),       # old_color
            [-np.random.rand(1)[0] * 2],  # log_T (negative = some absorption)
            np.random.rand(3),       # C_premultiplied
            [np.random.rand(1)[0] * 0.5]  # od
        ])
        upstream = np.random.randn(4)
        ok = finite_diff_check(
            update_pixel_fwd, update_pixel_bwd,
            x0, upstream, name=f"update_pixel_state trial {trial}"
        )
        all_pass = all_pass and ok

    return all_pass


# ============================================================================
# Section 3: Ray-plane intersection backward
# ============================================================================

def section3_intersection():
    print("\n" + "=" * 70)
    print("SECTION 3: Ray-plane intersection backward")
    print("=" * 70)

    # Symbolic derivation for one face
    # Face vertices: va, vb, vc
    # Normal: n = (vc - va) x (vb - va)
    # Numerator: num = n . (va - cam)
    # Denominator: den = n . ray_dir
    # t = num / den

    vax, vay, vaz = symbols('vax vay vaz')
    vbx, vby, vbz = symbols('vbx vby vbz')
    vcx, vcy, vcz = symbols('vcx vcy vcz')
    ox, oy, oz = symbols('ox oy oz')
    dx, dy, dz = symbols('dx dy dz')

    va = Matrix([vax, vay, vaz])
    vb = Matrix([vbx, vby, vbz])
    vc = Matrix([vcx, vcy, vcz])
    cam = Matrix([ox, oy, oz])
    ray_d = Matrix([dx, dy, dz])

    # Normal = (vc - va) x (vb - va)
    edge_c = vc - va
    edge_b = vb - va
    n = edge_c.cross(edge_b)

    num = n.dot(va - cam)
    den = n.dot(ray_d)
    t = num / den

    print("\nForward:")
    print("  n = (vc - va) x (vb - va)")
    print("  num = n . (va - cam)")
    print("  den = n . ray_dir")
    print("  t = num / den")

    # Derive dt/d(va), dt/d(vb), dt/d(vc)
    # Each is a 3-vector
    print("\nSymbolic derivatives (simplified):")

    dt_dva = Matrix([simplify(diff(t, v)) for v in [vax, vay, vaz]])
    dt_dvb = Matrix([simplify(diff(t, v)) for v in [vbx, vby, vbz]])
    dt_dvc = Matrix([simplify(diff(t, v)) for v in [vcx, vcy, vcz]])

    print(f"\n  dt/d(va) = {dt_dva.T}")
    print(f"\n  dt/d(vb) = {dt_dvb.T}")
    print(f"\n  dt/d(vc) = {dt_dvc.T}")

    # Try to find a more compact form using quotient rule
    # t = num/den, so dt/dx = (den * dnum/dx - num * dden/dx) / den^2
    # = (dnum/dx - t * dden/dx) / den
    print("\n\nCompact form using quotient rule:")
    print("  dt/dx = (dnum/dx - t * dden/dx) / den")
    print("  where num = n . (va - cam), den = n . ray_dir")
    print("  and n = (vc - va) x (vb - va)")
    print("")
    print("  For each vertex v_j in {va, vb, vc}:")
    print("    dn/dv_j is a 3x3 Jacobian (cross product derivative)")
    print("    dnum/dv_j = dn/dv_j^T . (va - cam)  +  (n if v_j == va)")
    print("    dden/dv_j = dn/dv_j^T . ray_dir")

    # Derive dn/d(va), dn/d(vb), dn/d(vc) symbolically
    # n = (vc - va) x (vb - va)  =  a x b  where a = vc-va, b = vb-va
    #
    # d(a x b)/d(va_i) = (da/d(va_i)) x b + a x (db/d(va_i))
    #                   = (-e_i) x b + a x (-e_i)
    #                   = -(e_i x b) + (e_i x a)     [since a x (-e_i) = e_i x a]
    #                   = e_i x (a - b)
    #                   = e_i x ((vc - va) - (vb - va))
    #                   = e_i x (vc - vb)
    #
    # The Jacobian element (i,j) = d(n_i)/d(va_j).
    # Column j of the Jacobian = d(n)/d(va_j) = e_j x (vc - vb)  -- WRONG!
    #
    # Actually e_j x (vc-vb) gives the j-th "slice", but we need the matrix [M]
    # where M * e_j = e_j x (vc-vb). That's NOT skew(vc-vb), it's -skew(vc-vb).
    # Because [u]_x * v = u x v, so to get v x u = -u x v = -[u]_x * v.
    # We want M such that M * e_j = e_j x (vc-vb) = -(vc-vb) x e_j = -[vc-vb]_x * e_j.
    #
    # But wait: the Jacobian has element (i,j) = (e_j x (vc-vb))_i.
    # This means ROW i, COL j of the Jacobian = (e_j x (vc-vb))_i.
    # The matrix with this property has column j = e_j x (vc-vb).
    # And [u]_x has column j = u x e_j. So column j of [-u]_x = -u x e_j = e_j x u.
    # Therefore dn/d(va) = [-(vc-vb)]_x = [vb-vc]_x = skew(vb-vc).
    #
    # Similarly:
    #   d(a x b)/d(vb_j) = a x e_j  (only b depends on vb, db/d(vb_j) = e_j)
    #   Jacobian column j = a x e_j = (vc-va) x e_j = [vc-va]_x * e_j
    #   => dn/d(vb) = [vc-va]_x = skew(vc-va)
    #
    #   d(a x b)/d(vc_j) = e_j x b  (only a depends on vc, da/d(vc_j) = e_j)
    #   Jacobian column j = e_j x b = e_j x (vb-va) = [-(vb-va)]_x * e_j = [va-vb]_x * e_j
    #   => dn/d(vc) = [va-vb]_x = skew(va-vb)

    print("\n  Cross-product Jacobians (n = (vc-va) x (vb-va)):")
    print("    dn/d(va) = [vb - vc]_x")
    print("    dn/d(vb) = [vc - va]_x")
    print("    dn/d(vc) = [va - vb]_x")
    print("")
    print("  where [u]_x = [[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]]")

    # Skew symmetric matrix [u]_x
    def skew(u):
        return Matrix([
            [0, -u[2], u[1]],
            [u[2], 0, -u[0]],
            [-u[1], u[0], 0]
        ])

    dn_dva_expected = skew(vb - vc)
    dn_dvb_expected = skew(vc - va)
    dn_dvc_expected = skew(va - vb)

    # Verify against direct differentiation
    n_vec = Matrix([n[0], n[1], n[2]])
    dn_dva_direct = Matrix([
        [diff(n_vec[i], [vax, vay, vaz][j]) for j in range(3)]
        for i in range(3)
    ])

    diff_check = simplify(dn_dva_expected - dn_dva_direct)
    assert diff_check == sp.zeros(3, 3), f"dn/d(va) Jacobian mismatch: {diff_check}"
    print("\n  Verified: dn/d(va) = [vb - vc]_x")

    dn_dvb_direct = Matrix([
        [diff(n_vec[i], [vbx, vby, vbz][j]) for j in range(3)]
        for i in range(3)
    ])
    diff_check = simplify(dn_dvb_expected - dn_dvb_direct)
    assert diff_check == sp.zeros(3, 3), f"dn/d(vb) Jacobian mismatch: {diff_check}"
    print("  Verified: dn/d(vb) = [vc - va]_x")

    dn_dvc_direct = Matrix([
        [diff(n_vec[i], [vcx, vcy, vcz][j]) for j in range(3)]
        for i in range(3)
    ])
    diff_check = simplify(dn_dvc_expected - dn_dvc_direct)
    assert diff_check == sp.zeros(3, 3), f"dn/d(vc) Jacobian mismatch: {diff_check}"
    print("  Verified: dn/d(vc) = [va - vb]_x")

    # Compact dt/d(vertex) formulas via quotient rule:
    # t = num / den
    # dt/d(x) = (dnum/d(x) - t * dden/d(x)) / den
    #
    # For va (appears in both n and (va-cam)):
    #   dnum/d(va) = dn/d(va)^T * (va - cam) + n
    #   dden/d(va) = dn/d(va)^T * ray_dir
    #   dt/d(va) = (dn/d(va)^T * (va - hit) + n) / den
    #
    # [u]_x^T = -[u]_x, so [u]_x^T * v = -(u x v) = v x u
    # dn/d(va)^T * w = [vb-vc]_x^T * w = w x (vb-vc) = (va-hit) x (vb-vc)
    #
    # For vb (appears only in n):
    #   dnum/d(vb) = dn/d(vb)^T * (va - cam)
    #   dt/d(vb) = dn/d(vb)^T * (va - hit) / den
    #            = (va - hit) x (vc - va) / den
    #
    # For vc (appears only in n):
    #   dt/d(vc) = dn/d(vc)^T * (va - hit) / den
    #            = (va - hit) x (va - vb) / den

    print("\n  Full dt/d(vertex) compact formulas:")
    print("    hit = cam + t * ray_dir")
    print("")
    print("    dt/d(va) = ((va - hit) x (vb - vc) + n) / den")
    print("    dt/d(vb) = ((va - hit) x (vc - va)) / den")
    print("    dt/d(vc) = ((va - hit) x (va - vb)) / den")

    # Verify these compact forms match the direct sympy derivatives
    hit = cam + t * ray_d

    dt_dva_compact = ((va - hit).cross(vb - vc) + n) / den
    dt_dvb_compact = ((va - hit).cross(vc - va)) / den
    dt_dvc_compact = ((va - hit).cross(va - vb)) / den

    for i in range(3):
        d1 = simplify(dt_dva[i] - dt_dva_compact[i])
        assert d1 == 0, f"dt_dva mismatch at {i}: {d1}"
    print("\n  Verified: compact dt/d(va) matches symbolic")

    for i in range(3):
        d1 = simplify(dt_dvb[i] - dt_dvb_compact[i])
        assert d1 == 0, f"dt_dvb mismatch at {i}: {d1}"
    print("  Verified: compact dt/d(vb) matches symbolic")

    for i in range(3):
        d1 = simplify(dt_dvc[i] - dt_dvc_compact[i])
        assert d1 == 0, f"dt_dvc mismatch at {i}: {d1}"
    print("  Verified: compact dt/d(vc) matches symbolic")

    # -- Numerical validation --
    print("\nNumerical validation:")

    def cross3(a, b):
        return np.array([
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        ])

    def intersection_fwd(x):
        """x = [va(3), vb(3), vc(3)] with fixed cam, ray_dir"""
        va_v = x[0:3]
        vb_v = x[3:6]
        vc_v = x[6:9]
        cam_v = np.array([0.0, 0.0, -1.0])
        ray_d_v = np.array([0.1, 0.2, 1.0])

        n_v = cross3(vc_v - va_v, vb_v - va_v)
        num_v = np.dot(n_v, va_v - cam_v)
        den_v = np.dot(n_v, ray_d_v)
        if abs(den_v) < 1e-20:
            return np.array([0.0])
        t_v = num_v / den_v
        return np.array([t_v])

    def intersection_bwd(x, dL_dt):
        """Analytical backward using compact form."""
        va_v = x[0:3]
        vb_v = x[3:6]
        vc_v = x[6:9]
        cam_v = np.array([0.0, 0.0, -1.0])
        ray_d_v = np.array([0.1, 0.2, 1.0])

        n_v = cross3(vc_v - va_v, vb_v - va_v)
        num_v = np.dot(n_v, va_v - cam_v)
        den_v = np.dot(n_v, ray_d_v)
        if abs(den_v) < 1e-20:
            return np.zeros(9)
        t_v = num_v / den_v

        hit_v = cam_v + t_v * ray_d_v

        # dt/d(va) = ((va - hit) x (vb - vc) + n) / den
        dt_dva_v = (cross3(va_v - hit_v, vb_v - vc_v) + n_v) / den_v
        # dt/d(vb) = ((va - hit) x (vc - va)) / den
        dt_dvb_v = cross3(va_v - hit_v, vc_v - va_v) / den_v
        # dt/d(vc) = ((va - hit) x (va - vb)) / den
        dt_dvc_v = cross3(va_v - hit_v, va_v - vb_v) / den_v

        dL = dL_dt[0]
        return np.concatenate([
            dL * dt_dva_v,
            dL * dt_dvb_v,
            dL * dt_dvc_v
        ])

    np.random.seed(456)
    all_pass = True
    for trial in range(5):
        # Random tet face vertices
        x0 = np.random.randn(9) * 0.5
        upstream = np.random.randn(1)
        ok = finite_diff_check(
            intersection_fwd, intersection_bwd,
            x0, upstream, name=f"intersection trial {trial}"
        )
        all_pass = all_pass and ok

    return all_pass


# ============================================================================
# Section 4: Color chain backward (simplified: no SH, no softplus)
# ============================================================================

def section4_color_chain():
    print("\n" + "=" * 70)
    print("SECTION 4: Color chain backward (simplified: no SH, no softplus)")
    print("=" * 70)

    print("\nForward color chain (simplified — colors are raw leaf parameters):")
    print("  1. base_offset = grad . (cam - v0)              [scalar]")
    print("  2. base_color  = colors + base_offset            [vec3, scalar broadcast]")
    print("  3. dc_dt       = grad . ray_dir                  [scalar]")
    print("  4. c_start     = max(base_color + dc_dt * t_min, 0)  [vec3]")
    print("     c_end       = max(base_color + dc_dt * t_max, 0)  [vec3]")

    print("\nBackward (given dL/d(c_start), dL/d(c_end)):")
    print("")
    print("  Step 4 backward (max clamp / ReLU):")
    print("    d_c_start_raw = dL/d(c_start) * (1 if raw > 0 else 0)  [per channel]")
    print("    d_c_end_raw   = dL/d(c_end) * (1 if raw > 0 else 0)")
    print("")
    print("  Step 3+2 backward (base_color + dc_dt * t):")
    print("    d_base_color = d_c_start_raw + d_c_end_raw")
    print("    d_dc_dt = sum(d_c_start_raw) * t_min + sum(d_c_end_raw) * t_max")
    print("    d_t_min_color = sum(d_c_start_raw) * dc_dt")
    print("    d_t_max_color = sum(d_c_end_raw) * dc_dt")
    print("")
    print("  Step 2 backward (base_color = colors + scalar offset):")
    print("    d_colors = d_base_color               [pass-through, leaf param]")
    print("    d_base_offset_scalar = sum(d_base_color)")
    print("    d_grad += (cam - v0) * d_base_offset_scalar")
    print("    d_v0   += -grad * d_base_offset_scalar")
    print("")
    print("  Step 3 backward (dc_dt = grad . ray_dir):")
    print("    d_grad += ray_dir * d_dc_dt")
    print("    (ray_dir is not differentiated)")

    # -- Numerical validation --
    print("\nNumerical validation (color chain without intersection):")

    # Fixed ray direction (not differentiated)
    fixed_ray_dir = np.array([0.1, 0.2, 1.0])

    def color_chain_fwd(x):
        """x = [colors(3), grad(3), v0(3), cam(3), t_min, t_max, density]
        Total: 15 elements.  ray_dir is a constant (not differentiated).
        """
        colors = x[0:3]
        grad = x[3:6]
        v0 = x[6:9]
        cam_v = x[9:12]
        ray_dir = fixed_ray_dir
        t_min = x[12]
        t_max = x[13]
        density = x[14]

        # Color chain (simplified: no SH, no softplus)
        base_offset = np.dot(grad, cam_v - v0)
        base_color = colors + base_offset  # scalar broadcast
        dc_dt = np.dot(grad, ray_dir)

        # entry/exit colors
        c_start = np.maximum(base_color + dc_dt * t_min, 0.0)
        c_end = np.maximum(base_color + dc_dt * t_max, 0.0)

        # Volume integral
        dist = max(t_max - t_min, 0.0)
        od = density * dist
        if od < 1e-8:
            od = 1e-8
        alpha_t = np.exp(-od)
        phi_val = (1.0 - np.exp(-od)) / od if abs(od) > 1e-6 else 1.0 - od * 0.5
        w0 = phi_val - alpha_t
        w1 = 1.0 - phi_val
        C = w0 * c_end + w1 * c_start
        alpha = 1.0 - alpha_t
        return np.array([C[0], C[1], C[2], alpha])

    def color_chain_bwd(x, dL_dout):
        """Analytical backward through the simplified color chain."""
        colors = x[0:3]
        grad = x[3:6]
        v0 = x[6:9]
        cam_v = x[9:12]
        ray_dir = fixed_ray_dir
        t_min = x[12]
        t_max = x[13]
        density = x[14]

        # Forward replay
        base_offset = np.dot(grad, cam_v - v0)
        base_color = colors + base_offset
        dc_dt = np.dot(grad, ray_dir)
        c_start_raw = base_color + dc_dt * t_min
        c_end_raw = base_color + dc_dt * t_max
        c_start = np.maximum(c_start_raw, 0.0)
        c_end = np.maximum(c_end_raw, 0.0)
        dist = max(t_max - t_min, 0.0)
        od = density * dist
        if od < 1e-8:
            od = 1e-8
        alpha_t = np.exp(-od)
        phi_val = (1.0 - np.exp(-od)) / od if abs(od) > 1e-6 else 1.0 - od * 0.5
        dphi_val = (np.exp(-od) * (1.0 + od) - 1.0) / (od * od) if abs(od) > 1e-6 else -0.5 + od / 3.0
        w0 = phi_val - alpha_t
        w1 = 1.0 - phi_val
        dw0_dod = dphi_val + alpha_t
        dw1_dod = -dphi_val

        # -- Backward through compute_integral --
        dL_dC = dL_dout[0:3]
        dL_dalpha = dL_dout[3]

        d_c_end = dL_dC * w0
        d_c_start = dL_dC * w1
        d_od = (np.dot(dL_dC, c_end * dw0_dod + c_start * dw1_dod)
                + dL_dalpha * alpha_t)

        # -- Backward through max clamp (ReLU) --
        d_c_start_raw = d_c_start * (c_start_raw > 0).astype(float)
        d_c_end_raw = d_c_end * (c_end_raw > 0).astype(float)

        # -- Backward through od = density * dist, dist = t_max - t_min --
        d_density = d_od * dist
        d_dist = d_od * density
        d_t_min_dist = -d_dist if dist > 0 else 0.0
        d_t_max_dist = d_dist if dist > 0 else 0.0

        # -- Backward through base_color + dc_dt * t --
        d_base_color = d_c_start_raw + d_c_end_raw
        d_dc_dt = np.sum(d_c_start_raw) * t_min + np.sum(d_c_end_raw) * t_max
        d_t_min_color = np.sum(d_c_start_raw) * dc_dt
        d_t_max_color = np.sum(d_c_end_raw) * dc_dt

        d_t_min = d_t_min_color + d_t_min_dist
        d_t_max = d_t_max_color + d_t_max_dist

        # -- Backward through base_color = colors + grad.(cam - v0) --
        d_colors = d_base_color.copy()  # pass-through (leaf param, no softplus)
        d_base_offset_scalar = np.sum(d_base_color)
        d_grad = (cam_v - v0) * d_base_offset_scalar
        d_v0 = -grad * d_base_offset_scalar

        # -- Backward through dc_dt = grad . ray_dir --
        d_grad += ray_dir * d_dc_dt

        # d_cam (for testing — camera is a variable in this test)
        d_cam = grad * d_base_offset_scalar

        return np.array([
            d_colors[0], d_colors[1], d_colors[2],
            d_grad[0], d_grad[1], d_grad[2],
            d_v0[0], d_v0[1], d_v0[2],
            d_cam[0], d_cam[1], d_cam[2],
            d_t_min, d_t_max, d_density
        ])

    np.random.seed(789)
    all_pass = True
    for trial in range(5):
        # Use positive colors to stay above ReLU boundary
        # Keep gradients very small so base_color + dc_dt * t stays positive
        x0 = np.concatenate([
            np.random.rand(3) * 0.5 + 0.5,    # colors (positive, well above 0)
            np.random.randn(3) * 0.005,        # grad (tiny -> color always positive)
            np.random.randn(3) * 0.3,          # v0
            np.array([0.0, 0.0, -2.0]),        # cam
            [0.5 + np.random.rand() * 0.3],    # t_min
            [1.2 + np.random.rand() * 0.3],    # t_max (well separated from t_min)
            [np.random.rand() * 1.5 + 0.3],    # density
        ])
        upstream = np.random.randn(4)

        ok = finite_diff_check(
            color_chain_fwd, color_chain_bwd,
            x0, upstream, name=f"color_chain trial {trial}",
            eps=1e-5, rtol=1e-3  # slightly relaxed for piecewise functions
        )
        all_pass = all_pass and ok

    return all_pass


# ============================================================================
# Section 5: Full chain composition test (simplified: no SH, no softplus)
# ============================================================================

def section5_full_chain():
    print("\n" + "=" * 70)
    print("SECTION 5: Full chain composition test (simplified color model)")
    print("=" * 70)
    print("\nEnd-to-end: single tet, single pixel, full parameter gradients")
    print("  params = [colors(3), density(1), color_grads(3), vertices(12)]")

    def cross3(a, b):
        return np.array([
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        ])

    # Face winding: [[0,2,1],[1,2,3],[0,3,2],[3,0,1]]
    FACES = [[0,2,1],[1,2,3],[0,3,2],[3,0,1]]

    def full_forward(params, cam, ray_dir):
        """Full forward pass for a single tet and single ray.

        params = [colors(3), density(1), color_grads(3), vertices(12)]
        Returns: (C_premultiplied_rgb(3), alpha(1)) = 4 values
        """
        colors = params[0:3]
        density = params[3]
        grad = params[4:7]
        verts = params[7:].reshape(4, 3)

        # Color chain (simplified: no SH, no softplus)
        base_offset = np.dot(grad, cam - verts[0])
        base_color = colors + base_offset  # scalar broadcast
        dc_dt = np.dot(grad, ray_dir)

        # Ray-tet intersection (slab method)
        t_enters = []
        t_exits = []
        for face in FACES:
            va = verts[face[0]]
            vb = verts[face[1]]
            vc = verts[face[2]]
            n = cross3(vc - va, vb - va)
            num = np.dot(n, va - cam)
            den = np.dot(n, ray_dir)
            if abs(den) < 1e-20:
                if num > 0:
                    return np.array([0.0, 0.0, 0.0, 0.0])
                continue
            t = num / den
            if den > 0:
                t_enters.append(t)
            else:
                t_exits.append(t)

        if not t_enters or not t_exits:
            return np.array([0.0, 0.0, 0.0, 0.0])

        t_min = max(t_enters)
        t_max = min(t_exits)
        if t_min >= t_max:
            return np.array([0.0, 0.0, 0.0, 0.0])

        dist = t_max - t_min
        od = density * dist
        if od < 1e-8:
            od = 1e-8

        c_start = np.maximum(base_color + dc_dt * t_min, 0.0)
        c_end = np.maximum(base_color + dc_dt * t_max, 0.0)

        alpha_t = np.exp(-od)
        phi_val = (1.0 - np.exp(-od)) / od if abs(od) > 1e-6 else 1.0 - od * 0.5
        w0 = phi_val - alpha_t
        w1 = 1.0 - phi_val
        C = w0 * c_end + w1 * c_start
        alpha = 1.0 - alpha_t
        return np.array([C[0], C[1], C[2], alpha])

    def full_backward(params, cam, ray_dir, dL_dout):
        """Full analytical backward. Returns gradient w.r.t. params."""
        colors = params[0:3]
        density = params[3]
        grad = params[4:7]
        verts = params[7:].reshape(4, 3)

        # -- Forward replay --
        base_offset = np.dot(grad, cam - verts[0])
        base_color = colors + base_offset
        dc_dt = np.dot(grad, ray_dir)

        # Intersection
        t_enters = []
        t_exits = []
        for fi, face in enumerate(FACES):
            va = verts[face[0]]
            vb = verts[face[1]]
            vc = verts[face[2]]
            n = cross3(vc - va, vb - va)
            num = np.dot(n, va - cam)
            den = np.dot(n, ray_dir)
            if abs(den) < 1e-20:
                continue
            t = num / den
            if den > 0:
                t_enters.append((t, fi))
            else:
                t_exits.append((t, fi))

        if not t_enters or not t_exits:
            return np.zeros_like(params)

        t_min_entry = max(t_enters, key=lambda x: x[0])
        t_max_entry = min(t_exits, key=lambda x: x[0])
        t_min = t_min_entry[0]
        t_max = t_max_entry[0]
        min_face_idx = t_min_entry[1]
        max_face_idx = t_max_entry[1]

        if t_min >= t_max:
            return np.zeros_like(params)

        dist = t_max - t_min
        od = density * dist
        if od < 1e-8:
            od = 1e-8

        c_start_raw = base_color + dc_dt * t_min
        c_end_raw = base_color + dc_dt * t_max
        c_start = np.maximum(c_start_raw, 0.0)
        c_end = np.maximum(c_end_raw, 0.0)

        alpha_t = np.exp(-od)
        phi_val = (1.0 - np.exp(-od)) / od if abs(od) > 1e-6 else 1.0 - od * 0.5
        dphi_val = (np.exp(-od) * (1.0 + od) - 1.0) / (od * od) if abs(od) > 1e-6 else -0.5 + od / 3.0
        w0 = phi_val - alpha_t
        w1 = 1.0 - phi_val
        dw0_dod = dphi_val + alpha_t
        dw1_dod = -dphi_val

        # -- Backward through compute_integral --
        dL_dC = dL_dout[0:3]
        dL_dalpha = dL_dout[3]

        d_c_end = dL_dC * w0
        d_c_start = dL_dC * w1
        d_od = (np.dot(dL_dC, c_end * dw0_dod + c_start * dw1_dod)
                + dL_dalpha * alpha_t)

        # -- Backward through max clamp --
        d_c_start_raw = d_c_start * (c_start_raw > 0).astype(float)
        d_c_end_raw = d_c_end * (c_end_raw > 0).astype(float)

        # -- Backward through od, dist --
        d_density = d_od * dist
        d_dist = d_od * density
        d_t_min = -d_dist  # from dist = t_max - t_min
        d_t_max = d_dist

        # -- Backward through color at entry/exit --
        d_base_color = d_c_start_raw + d_c_end_raw
        d_dc_dt = np.sum(d_c_start_raw) * t_min + np.sum(d_c_end_raw) * t_max
        d_t_min += np.sum(d_c_start_raw) * dc_dt
        d_t_max += np.sum(d_c_end_raw) * dc_dt

        # -- Backward through base_color = colors + grad.(cam - v0) --
        d_colors = d_base_color.copy()  # pass-through (leaf param, no softplus)
        d_base_offset_scalar = np.sum(d_base_color)
        d_grad = (cam - verts[0]) * d_base_offset_scalar
        d_v0_from_base = -grad * d_base_offset_scalar

        # -- Backward through dc_dt = grad . ray_dir --
        d_grad = d_grad + ray_dir * d_dc_dt

        # -- Backward through intersection (t_min and t_max) --
        d_verts = np.zeros((4, 3))

        for t_val, d_t, face_idx in [(t_min, d_t_min, min_face_idx),
                                      (t_max, d_t_max, max_face_idx)]:
            face = FACES[face_idx]
            va = verts[face[0]]
            vb = verts[face[1]]
            vc = verts[face[2]]
            n = cross3(vc - va, vb - va)
            den = np.dot(n, ray_dir)
            hit = cam + t_val * ray_dir

            # dt/d(va) = ((va - hit) x (vb - vc) + n) / den
            dt_dva = (cross3(va - hit, vb - vc) + n) / den
            # dt/d(vb) = ((va - hit) x (vc - va)) / den
            dt_dvb = cross3(va - hit, vc - va) / den
            # dt/d(vc) = ((va - hit) x (va - vb)) / den
            dt_dvc = cross3(va - hit, va - vb) / den

            d_verts[face[0]] += d_t * dt_dva
            d_verts[face[1]] += d_t * dt_dvb
            d_verts[face[2]] += d_t * dt_dvc

        # Combine vertex gradients
        d_verts[0] += d_v0_from_base

        # Assemble output
        d_params = np.zeros_like(params)
        d_params[0:3] = d_colors
        d_params[3] = d_density
        d_params[4:7] = d_grad
        d_params[7:] = d_verts.flatten()

        return d_params

    # Generate a valid test case: tet that the ray passes through its center
    print("\nNumerical validation:")
    print("  Note: vertex gradients may show small discrepancies near face-selection")
    print("  boundaries (where which face determines t_min/t_max can switch).")
    print("  This is inherent to piecewise functions, not a derivative bug.")
    print("  Tests skip unstable configurations where face t-values are too close.")
    np.random.seed(2024)
    all_pass = True
    n_passed = 0

    def compute_face_gap(verts, cam_v, ray_d):
        """Check the gap between the winning and second-best face t-values.
        Returns (gap_enter, gap_exit) or None if ray misses.
        If only 1 enter or 1 exit, that gap is infinite (no ambiguity)."""
        t_enters = []
        t_exits = []
        for face in FACES:
            va = verts[face[0]]
            vb = verts[face[1]]
            vc = verts[face[2]]
            n = cross3(vc - va, vb - va)
            num = np.dot(n, va - cam_v)
            den = np.dot(n, ray_d)
            if abs(den) < 1e-20:
                continue
            t = num / den
            if den > 0:
                t_enters.append(t)
            else:
                t_exits.append(t)
        if not t_enters or not t_exits:
            return None
        # If only 1 enter or 1 exit, gap is infinite (no face-selection ambiguity)
        gap_enter = float('inf')
        gap_exit = float('inf')
        if len(t_enters) >= 2:
            t_enters.sort(reverse=True)
            gap_enter = t_enters[0] - t_enters[1]
        if len(t_exits) >= 2:
            t_exits.sort()
            gap_exit = t_exits[1] - t_exits[0]
        return (gap_enter, gap_exit)

    for trial in range(30):  # try more to get enough stable configs
        if n_passed >= 5:
            break

        # Vary camera and ray direction for diversity
        cam_v = np.array([0.3 * np.random.randn(), 0.3 * np.random.randn(), -5.0])
        # Slightly tilted ray so we don't hit symmetric face configurations
        ray_tilt = np.random.randn(2) * 0.15
        ray_d = np.array([ray_tilt[0], ray_tilt[1], 1.0])
        ray_d = ray_d / np.linalg.norm(ray_d)

        # Large tet with moderate perturbation for irregular shapes
        v0 = np.array([2.0, 0.0, -0.5]) + np.random.randn(3) * 0.15
        v1 = np.array([-1.0, 1.732, -0.5]) + np.random.randn(3) * 0.15
        v2 = np.array([-1.0, -1.732, -0.5]) + np.random.randn(3) * 0.15
        v3 = np.array([0.0, 0.0, 2.5]) + np.random.randn(3) * 0.15

        colors_v = np.random.rand(3) * 0.5 + 0.5  # positive base colors
        density = np.random.rand() * 0.5 + 0.5
        grad_v = np.random.randn(3) * 0.003  # tiny gradient

        verts_arr = np.array([v0, v1, v2, v3])
        gaps = compute_face_gap(verts_arr, cam_v, ray_d)
        if gaps is None or min(gaps) < 0.05:
            # Face t-values too close; finite differences will be unreliable
            continue

        params = np.concatenate([
            colors_v, [density], grad_v,
            v0, v1, v2, v3
        ])

        out = full_forward(params, cam_v, ray_d)
        if np.all(np.abs(out) < 1e-10):
            continue

        upstream = np.random.randn(4)

        def fwd_wrapper(p, _cam=cam_v, _rd=ray_d):
            return full_forward(p, _cam, _rd)

        def bwd_wrapper(p, up, _cam=cam_v, _rd=ray_d):
            return full_backward(p, _cam, _rd, up)

        ok = finite_diff_check(
            fwd_wrapper, bwd_wrapper,
            params, upstream, name=f"full_chain trial {n_passed}",
            eps=1e-5, rtol=5e-3
        )
        all_pass = all_pass and ok
        n_passed += 1

    if n_passed < 3:
        print(f"  WARNING: only {n_passed} stable test configs found (need 3)")
        all_pass = False

    return all_pass


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    results = {}

    results["Section 1: compute_integral"] = section1_compute_integral()
    results["Section 2: update_pixel_state"] = section2_update_pixel_state()
    results["Section 3: intersection"] = section3_intersection()
    results["Section 4: color_chain"] = section4_color_chain()
    results["Section 5: full_chain"] = section5_full_chain()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_ok = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        all_ok = all_ok and passed

    if all_ok:
        print("\nAll checks passed!")
    else:
        print("\nSome checks FAILED!")

    exit(0 if all_ok else 1)
