"""
Test moment shadow map approaches.

Creates random piecewise-constant density functions, computes ground-truth
transmittance T(z) = exp(-integral_0^z sigma(t) dt), then reconstructs T(z)
from moments accumulated via additive blending.

Tests both:
1. Fourier opacity mapping (proper Fourier series on [0,1])
2. Power moments (Hamburger moment problem, Peters & Klein 2015)
"""

import numpy as np
import matplotlib.pyplot as plt


def make_random_density(n_segments=8, z_range=(0.0, 1.0), max_density=10.0, rng=None):
    """Create a random piecewise-constant density function."""
    if rng is None:
        rng = np.random.default_rng()
    knots = np.sort(rng.uniform(z_range[0], z_range[1], n_segments - 1))
    knots = np.concatenate([[z_range[0]], knots, [z_range[1]]])
    densities = rng.uniform(0.0, max_density, n_segments)
    return knots, densities


def ground_truth_transmittance(z_query, knots, densities):
    """Compute T(z) = exp(-integral_0^z sigma(t) dt) exactly."""
    z_query = np.asarray(z_query)
    absorbance = np.zeros_like(z_query)
    for i in range(len(densities)):
        z_lo = knots[i]
        z_hi = knots[i + 1]
        contrib_length = np.clip(z_query - z_lo, 0, z_hi - z_lo)
        absorbance += densities[i] * contrib_length
    return np.exp(-absorbance)


## -----------------------------------------------------------------------
## Fourier Opacity Mapping (proper Fourier series on [0,1])
## -----------------------------------------------------------------------

def accumulate_fourier(knots, densities, n_terms=4, n_substeps=200):
    """Accumulate Fourier coefficients of the opacity density.

    The density sigma(z) on [0,1] has Fourier series:
        sigma(z) = a0/2 + sum_{k=1}^N [a_k cos(2*pi*k*z) + b_k sin(2*pi*k*z)]

    Each thin slab at depth z with opacity alpha contributes:
        a0 += 2 * alpha   (so that a0/2 = total opacity)
        a_k += 2 * alpha * cos(2*pi*k*z)
        b_k += 2 * alpha * sin(2*pi*k*z)

    Returns:
        a0: DC term (total opacity * 2)
        a_k: (n_terms,) cosine coefficients
        b_k: (n_terms,) sine coefficients
    """
    a0 = 0.0
    a_k = np.zeros(n_terms)
    b_k = np.zeros(n_terms)

    for i in range(len(densities)):
        z_lo = knots[i]
        z_hi = knots[i + 1]
        seg_len = z_hi - z_lo
        if seg_len < 1e-12 or densities[i] < 1e-12:
            continue

        dz = seg_len / n_substeps
        for s in range(n_substeps):
            z = z_lo + (s + 0.5) * dz
            od = densities[i] * dz
            alpha = 1.0 - np.exp(-od)

            a0 += 2.0 * alpha
            for k in range(1, n_terms + 1):
                a_k[k-1] += 2.0 * alpha * np.cos(2.0 * np.pi * k * z)
                b_k[k-1] += 2.0 * alpha * np.sin(2.0 * np.pi * k * z)

    return a0, a_k, b_k


def accumulate_fourier_analytic(knots, densities, n_terms=4):
    """Accumulate Fourier coefficients analytically for piecewise-constant density.

    For a constant density rho on [z_lo, z_hi], the Fourier coefficients are
    computed by integrating the basis functions exactly, avoiding sub-stepping error.

    We treat the density as continuous (not discretized into alpha slabs).
    The optical depth contribution is rho * dz, and for thin slabs alpha ≈ od.
    For a continuous density function, we integrate directly:

        a0 = 2 * integral_0^1 sigma(z) dz
        a_k = 2 * integral_0^1 sigma(z) * cos(2*pi*k*z) dz
        b_k = 2 * integral_0^1 sigma(z) * sin(2*pi*k*z) dz
    """
    a0 = 0.0
    a_k = np.zeros(n_terms)
    b_k = np.zeros(n_terms)

    for i in range(len(densities)):
        z_lo = knots[i]
        z_hi = knots[i + 1]
        rho = densities[i]
        seg_len = z_hi - z_lo
        if seg_len < 1e-12 or rho < 1e-12:
            continue

        # DC: integral of rho over [z_lo, z_hi]
        a0 += 2.0 * rho * seg_len

        for k in range(1, n_terms + 1):
            w = 2.0 * np.pi * k
            # integral of rho * cos(w*z) dz from z_lo to z_hi = rho * [sin(w*z)/w]
            a_k[k-1] += 2.0 * rho * (np.sin(w * z_hi) - np.sin(w * z_lo)) / w
            # integral of rho * sin(w*z) dz from z_lo to z_hi = rho * [-cos(w*z)/w]
            b_k[k-1] += 2.0 * rho * (-np.cos(w * z_hi) + np.cos(w * z_lo)) / w

    return a0, a_k, b_k


def reconstruct_transmittance_fourier(z_query, a0, a_k, b_k, use_sigma_window=False):
    """Reconstruct T(z) from Fourier coefficients.

    The reconstructed density is:
        sigma_hat(z) = a0/2 + sum_k [a_k cos(2*pi*k*z) + b_k sin(2*pi*k*z)]

    Absorbance is the integral from 0 to z:
        A(z) = (a0/2)*z + sum_k [a_k * sin(2*pi*k*z)/(2*pi*k) + b_k * (1 - cos(2*pi*k*z))/(2*pi*k)]

    T(z) = exp(-A(z))
    """
    z_query = np.asarray(z_query)
    n_terms = len(a_k)

    A = (a0 / 2.0) * z_query
    for k in range(1, n_terms + 1):
        w = 2.0 * np.pi * k
        ak = a_k[k-1]
        bk = b_k[k-1]

        # Lanczos sigma factor to reduce Gibbs ringing
        if use_sigma_window:
            sigma = np.sin(k * np.pi / (n_terms + 1)) / (k * np.pi / (n_terms + 1))
            ak *= sigma
            bk *= sigma

        # Integral of ak*cos(w*t) from 0 to z = ak * sin(w*z) / w
        # Integral of bk*sin(w*t) from 0 to z = bk * (1 - cos(w*z)) / w
        A += ak * np.sin(w * z_query) / w
        A += bk * (1.0 - np.cos(w * z_query)) / w

    A = np.maximum(A, 0.0)
    return np.exp(-A)


## -----------------------------------------------------------------------
## Power Moments (Hamburger moment problem)
## -----------------------------------------------------------------------

def accumulate_power_moments(knots, densities, n_moments=4, n_substeps=200):
    """Accumulate power moments b_i = sum(alpha * z^i)."""
    moments = np.zeros(n_moments + 1)

    for i in range(len(densities)):
        z_lo = knots[i]
        z_hi = knots[i + 1]
        seg_len = z_hi - z_lo
        if seg_len < 1e-12 or densities[i] < 1e-12:
            continue

        dz = seg_len / n_substeps
        for s in range(n_substeps):
            z = z_lo + (s + 0.5) * dz
            od = densities[i] * dz
            alpha = 1.0 - np.exp(-od)

            for p in range(n_moments + 1):
                moments[p] += alpha * (z ** p)

    return moments


def reconstruct_transmittance_power(z_query, moments, eps=3e-5):
    """Reconstruct T(z) from 4 power moments using 2-node Hamburger solve."""
    z_query = np.asarray(z_query)
    b0 = moments[0]

    if b0 < 1e-10:
        return np.ones_like(z_query)

    m = moments[1:] / b0

    H = np.array([[1.0, m[0]], [m[0], m[1]]])
    H += eps * np.eye(2)
    rhs = np.array([-m[1], -m[2]])

    try:
        c = np.linalg.solve(H, rhs)
    except np.linalg.LinAlgError:
        return np.exp(-b0 * z_query)

    disc = c[1]**2 - 4.0 * c[0]
    if disc < 0:
        z_mean = m[0]
        return np.clip(np.where(z_query < z_mean, 1.0, 1.0 - b0), 0, 1)

    sqrt_disc = np.sqrt(disc)
    z1 = (-c[1] - sqrt_disc) / 2.0
    z2 = (-c[1] + sqrt_disc) / 2.0
    if z1 > z2:
        z1, z2 = z2, z1

    if abs(z2 - z1) < 1e-10:
        return np.clip(np.where(z_query < z1, 1.0, 1.0 - b0), 0, 1)

    w2 = (m[0] - z1) / (z2 - z1)
    w1 = 1.0 - w2

    T = np.ones_like(z_query)
    T -= b0 * w1 * (z_query >= z1).astype(float)
    T -= b0 * w2 * (z_query >= z2).astype(float)
    return np.clip(T, 0, 1)


## -----------------------------------------------------------------------
## Tests
## -----------------------------------------------------------------------

def test_comparison():
    """Statistical comparison across many random scenes."""
    n_trials = 50
    configs = [
        ("Fourier N=4 (substep)", lambda k, d: reconstruct_transmittance_fourier(z, *accumulate_fourier(k, d, 4))),
        ("Fourier N=4 (analytic)", lambda k, d: reconstruct_transmittance_fourier(z, *accumulate_fourier_analytic(k, d, 4))),
        ("Fourier N=8 (analytic)", lambda k, d: reconstruct_transmittance_fourier(z, *accumulate_fourier_analytic(k, d, 8))),
        ("Fourier N=16 (analytic)", lambda k, d: reconstruct_transmittance_fourier(z, *accumulate_fourier_analytic(k, d, 16))),
        ("Fourier N=4 + Lanczos", lambda k, d: reconstruct_transmittance_fourier(z, *accumulate_fourier_analytic(k, d, 4), use_sigma_window=True)),
        ("Fourier N=8 + Lanczos", lambda k, d: reconstruct_transmittance_fourier(z, *accumulate_fourier_analytic(k, d, 8), use_sigma_window=True)),
        ("Power n=2 (4 moments)", lambda k, d: reconstruct_transmittance_power(z, accumulate_power_moments(k, d, 4))),
    ]

    z = np.linspace(0, 1, 1000)

    print("=== Statistical Comparison (50 random scenes, 8 segments) ===\n")
    print(f"{'Method':>30} {'Mean RMSE':>10} {'Max RMSE':>10} {'Mean MaxErr':>12}")
    print("-" * 67)

    for label, recon_fn in configs:
        rmses = []
        max_errs = []
        for seed in range(n_trials):
            rng = np.random.default_rng(seed)
            knots, densities = make_random_density(n_segments=8, rng=rng)
            T_true = ground_truth_transmittance(z, knots, densities)
            T_recon = recon_fn(knots, densities)
            rmses.append(np.sqrt(np.mean((T_true - T_recon) ** 2)))
            max_errs.append(np.max(np.abs(T_true - T_recon)))
        print(f"{label:>30} {np.mean(rmses):>10.4f} {np.max(rmses):>10.4f} {np.mean(max_errs):>12.4f}")

    # Fix z for lambdas
    return z


def test_visual(seeds=[42, 7, 99]):
    """Visual comparison."""
    for seed in seeds:
        rng = np.random.default_rng(seed)
        knots, densities = make_random_density(n_segments=6, rng=rng)
        z = np.linspace(0, 1, 1000)
        T_true = ground_truth_transmittance(z, knots, densities)

        methods = [
            ("Fourier N=4", reconstruct_transmittance_fourier(z, *accumulate_fourier_analytic(knots, densities, 4))),
            ("Fourier N=8", reconstruct_transmittance_fourier(z, *accumulate_fourier_analytic(knots, densities, 8))),
            ("Fourier N=16", reconstruct_transmittance_fourier(z, *accumulate_fourier_analytic(knots, densities, 16))),
            ("Fourier N=8+Lanczos", reconstruct_transmittance_fourier(z, *accumulate_fourier_analytic(knots, densities, 8), use_sigma_window=True)),
            ("Power n=2", reconstruct_transmittance_power(z, accumulate_power_moments(knots, densities, 4))),
        ]

        fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4))
        for ax, (label, T_recon) in zip(axes, methods):
            rmse = np.sqrt(np.mean((T_true - T_recon) ** 2))
            ax.plot(z, T_true, 'k-', linewidth=2, label='Truth')
            ax.plot(z, T_recon, 'r--', linewidth=1.5, label=label)

            ax2 = ax.twinx()
            for i in range(len(densities)):
                ax2.fill_between([knots[i], knots[i+1]], 0, densities[i],
                                 alpha=0.15, color='blue')
            ax2.set_ylim(0, max(densities) * 1.5 if max(densities) > 0 else 1)
            ax2.tick_params(labelright=False)

            ax.set_title(f'{label}\nRMSE={rmse:.4f}')
            ax.legend(loc='upper right', fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(0, 1)

        plt.tight_layout()
        plt.savefig(f'comparison_seed{seed}.png', dpi=150)
        plt.show()
        print(f"Saved comparison_seed{seed}.png")


if __name__ == '__main__':
    # Hack: make z available to lambdas in test_comparison
    z = np.linspace(0, 1, 1000)
    test_comparison()
    test_visual()
