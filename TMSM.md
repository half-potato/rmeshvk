# Trigonometric Moment Shadow Maps (TMSM)

Fourier-based deep shadow maps for cached per-light transmittance. Renders the scene once per light face using order-independent additive blending, stores a compact Fourier representation of the transmittance function T(z), and queries it at arbitrary depths during deferred shading.

## Problem

We need to know how much light reaches a point at depth z along a ray from a light source. The scene contains volumetric tetrahedra with varying density. Each tet absorbs some light, and the total transmittance is the product of individual transmittances along the ray.

A standard shadow map stores a single depth — useless for volumes. We need T(z) as a continuous function per pixel.

## Representation

The opacity density along a light ray is a function sigma(z) on [0,1] (NDC depth from the light's projection). We represent it as a truncated Fourier series:

```
sigma(z) = a0/2 + sum_{k=1}^{N} [ a_k cos(2*pi*k*z) + b_k sin(2*pi*k*z) ]
```

The transmittance is:

```
T(z) = exp(-A(z))
```

where A(z) is the absorbance (integrated density from 0 to z):

```
A(z) = integral_0^z sigma(t) dt
     = (a0/2) * z + sum_{k=1}^{N} [ a_k * sin(2*pi*k*z) / (2*pi*k)
                                    + b_k * (1 - cos(2*pi*k*z)) / (2*pi*k) ]
```

## Generation (per light face, single pass)

Render all tets from the light's viewpoint. No sorting needed. Each fragment at NDC depth z with opacity alpha contributes to the Fourier coefficients via **additive blending** (src=ONE, dst=ONE):

```
a0 += 2 * alpha
a_k += 2 * alpha * cos(2*pi*k*z)    for k = 1..N
b_k += 2 * alpha * sin(2*pi*k*z)    for k = 1..N
```

The opacity alpha of a fragment comes from the volume rendering integral of the tet interval:

```
od = density * distance_through_tet
alpha = 1 - exp(-od)
```

### Storage layout

2N+1 scalar values per pixel. Packed into RGBA16F render targets:

| N | Scalars | RGBA16F targets | Quality (mean RMSE) |
|---|---------|-----------------|---------------------|
| 4 | 9       | 3 (one partially used) | 0.018 |
| 8 | 17      | 5 (one partially used) | 0.008 |

Packing for N=4:
```
RT0: (a0, a1, b1, a2)
RT1: (b2, a3, b3, a4)
RT2: (b4, unused, unused, unused)
```

Half-float (16-bit) precision is sufficient because all basis functions cos/sin are bounded to [-1,1] and alpha is in [0,1], so accumulated values stay in a reasonable range.

## Query (deferred shading)

For a receiver point in world space:

1. Project into light clip space to get NDC depth z in [0,1]
2. Sample the Fourier coefficient textures at the light-space pixel coordinate
3. Reconstruct absorbance:

```
w_k = 2 * pi * k

A(z) = (a0/2) * z
     + sum_{k=1}^{N} [ a_k * sin(w_k * z) / w_k
                      + b_k * (1 - cos(w_k * z)) / w_k ]

A = max(A, 0)    // clamp to avoid negative absorbance from Gibbs oscillation
T = exp(-A)
```

4. Use T as the shadow attenuation factor in the lighting equation

## Primitives

Opaque primitives (cubes, spheres, etc.) are rendered in a pre-pass before the tet pass, into the same Fourier targets plus a Depth32Float buffer:

- Primitive fragment outputs: `(2.0, 2.0*cos(...), 2.0*sin(...), ...)` with alpha=1 (fully opaque contributes maximally to all coefficients). In practice, since primitives are opaque blockers, they write a0 += 2 and the trig terms at their depth.
- Primitive pass uses opaque writes (no blending) and clears targets.
- Tet pass uses additive blending with LoadOp::Load, plus depth testing against primitive depths (read-only, LessEqual) so tets behind primitives are culled.

Alternatively, primitives can simply write alpha=1 to all Fourier targets using the same additive formula as tets, which automatically produces the correct transmittance step at their depth.

## Gibbs oscillation

The truncated Fourier series can oscillate near sharp density transitions, potentially producing negative A(z) (transmittance > 1). This is handled by clamping A(z) >= 0 at query time.

For smooth volumetric density (typical of tet meshes), Gibbs ringing is minimal. The Lanczos sigma window (multiply coefficient k by `sin(k*pi/(N+1)) / (k*pi/(N+1))` at query time) can further suppress ringing but reduces sharpness — not recommended for our use case since volumes are already smooth.

## Validation

`scripts/test_trig_moments.py` tests the approach on random piecewise-constant density functions. Results with analytic Fourier coefficient accumulation:

```
Fourier N=4:  mean RMSE 0.018, max error 6.4%
Fourier N=8:  mean RMSE 0.008, max error 3.3%
Fourier N=16: mean RMSE 0.004, max error 1.9%
```

Compared to power moment shadow maps (Hamburger reconstruction with 4 moments): mean RMSE 0.230, max error 67%. The Fourier approach is an order of magnitude more accurate for volumetric scenes.

## References

- Jansen & Bavoil, "Fourier Opacity Mapping", I3D 2010 — the core technique
- Lokovic & Veach, "Deep Shadow Maps", SIGGRAPH 2000 — the original DSM concept
- Peters & Klein, "Moment Shadow Maps", I3D 2015 — power moment approach (poor fit for volumes)
- Peters & Klein, "Moment Transparency", I3D 2017 — trigonometric moments for OIT (related but different application)
