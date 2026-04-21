# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**rmeshvk** — A wgpu-based differentiable tetrahedral volume renderer for Radiance Meshes. This is the Rust/WebGPU port of the rendering and training backend, designed to replace the Slang/CUDA pipeline from the parent `delaunay_rasterization` project.

It renders Delaunay tetrahedral meshes with spherical harmonics (SH) color, supports forward and backward (gradient) passes entirely on GPU via WGSL compute shaders, and exposes a Python module for PyTorch integration.

## Build & Test

```bash
# Build entire workspace
cargo build

# Run all tests (requires GPU — tests gracefully skip if no adapter found)
cargo test

# Run a specific test
cargo test -p rmesh-render test_center_view

# Run tests for a specific crate
cargo test -p rmesh-render
cargo test -p rmesh-backward

# Build the Python extension (maturin + PyO3)
cd crates/rmesh-python && maturin develop
```

Toolchain: Rust stable (see `rust-toolchain.toml`). No special features or nightly required.

## Workspace Crates (Dependency Order)

```
rmesh-util          ← Shared CPU/GPU types (Uniforms, etc.), WGSL utility shaders (common, intersect, math, SH)
rmesh-sort          ← GPU radix sort pipelines (basic & DRS modes)
rmesh-tile          ← Tile infrastructure (fill, ranges, scan, RTS prefix scan)
    ↓
rmesh-data          ← .rmesh file loading, PCA decompression, circumsphere computation
    ↓
rmesh-render        ← Forward pipeline: project → sort → interval rasterize (wgpu orchestration)
    ↓
rmesh-backward      ← Backward pipeline: backward compute → gradient accumulation
rmesh-error         ← Per-tet error statistics accumulation
    ↓
rmesh-train         ← Training loop (forward + loss + backward + Adam, all GPU)
rmesh-dsm           ← Deep shadow maps (power moments, per-light cubemap atlas)
rmesh-compositor    ← Opaque primitive rendering (cube, sphere, plane, cylinder) + depth compositing
rmesh-neural        ← Neural material models (MLPBRDF, RetroHead via burn)
rmesh-sim           ← Eulerian fluid simulation on tet mesh (Stable Fluids)
rmesh-interact      ← Input handling & transform interactions (translate, rotate, scale)
rmesh-viewer        ← Interactive winit/wgpu viewer (orbit camera, egui UI, loads .rmesh files)
rmesh-viewer-web    ← WebAssembly viewer
rmesh-python        ← PyO3/maturin bindings exposing RMeshRenderer to Python/PyTorch
```

## Architecture

### Rendering Pipeline (Interval Shading)

The interval shading path is the active rendering approach:

1. **Project compute** — Projects tet vertices, evaluates SH color, generates sort keys (`project_compute.wgsl`)
2. **Radix sort** — GPU radix sort of tets by depth (rmesh-sort, 5-pass: count → reduce → scan → scan_add → scatter)
3. **Interval generate** — Compute shader generates screen-space triangles per tet, writes vertex + per-tet data to buffers (`interval_generate.wgsl`)
4. **Interval vertex/fragment** — Hardware rasterization of interval geometry with MRT output: color, depth, normals (`interval_vertex.wgsl`, `interval_fragment.wgsl`)
5. **Deferred shading** — Fullscreen pass combining MRT G-buffer with PBR lighting and DSM shadows (`deferred_shade_frag.wgsl`)

There is also a **mesh shader pathway** (`interval_mesh.wgsl`, `IntervalPipelines`) that replaces steps 3–4 with a single mesh shader dispatch. **This path is untested** — it requires `Features::EXPERIMENTAL_MESH_SHADER` hardware support which is not currently available for testing.

An alternative **interval tiled rasterize** compute path (`interval_tiled_rasterize.wgsl`, `IntervalTiledRasterizePipeline`) also exists for tile-based software rasterization of intervals.

#### Deep Shadow Maps (rmesh-dsm)

Shadows use power moments stored in a per-light cubemap atlas:
- `DsmPipeline` renders interval geometry from each light's perspective
- Fragment shader accumulates power moments (m_0..m_4 = α·z^k) into 3 Rgba16Float MRT targets
- `DsmResolvePipeline` reconstructs transmittance T(z) via Hamburger 2-atom moment reconstruction
- The deferred shading pass samples the DSM atlas for shadow evaluation

#### Compositing (rmesh-compositor)

Opaque primitives (cubes, spheres, planes, cylinders) are rendered via `PrimitivePipeline` and depth-composited with translucent tet volumes via `CompositorPipeline`.

### Legacy / Dead Code Paths

Several older rendering approaches remain in the codebase but are not actively used:

- **Hardware rasterization path** (`forward_vertex.wgsl`, `forward_fragment.wgsl`, `ForwardPipelines`): The original vertex/fragment pipeline. Superseded by interval shading.
- **Quad-based rendering** (`forward_vertex_quad.wgsl`, `forward_fragment_quad.wgsl`, `forward_prepass_compute.wgsl`): Quad billboard approach.
- **Mesh shader forward** (`forward_mesh.wgsl`, `MeshForwardPipelines`): Non-interval mesh shader path.
- **Ray tracing compute** (`raytrace_compute.wgsl`, `RayTracePipeline`): Software ray-tet intersection.
- **Rasterize compute** (`rasterize_compute.wgsl`, `RasterizeComputePipeline`): Software rasterization compute path.
- **16-bit project** (`project_compute_16bit.wgsl`): Half-precision projection variant.
- **Shadow ray gen** (`shadow_ray_gen.wgsl`): Pre-DSM shadow approach.

These are retained for reference and potential future use but are not part of the active pipeline.

### Training Pipeline

1. **Loss compute** — L1/L2/SSIM loss + per-pixel dL/d(image) (`rmesh-train`)
2. **Backward tiled compute** — Reverse-order traversal computing dL/d(params) (`rmesh-backward`)
3. **Error accumulation** — Per-tet error statistics (`rmesh-error`)
4. **Adam compute** — Per-parameter-group Adam optimizer update (`rmesh-train`)

### Key Design Patterns

- **Shared types** (`rmesh-util/src/shared.rs`): `#[repr(C)]` structs with `bytemuck::Pod` matching WGSL `struct` layouts byte-for-byte (std430). All CPU↔GPU data flows through these.
- **WGSL as `include_str!`**: Shaders are embedded at compile time in each crate's `lib.rs`. No runtime shader compilation — change a `.wgsl` file and `cargo build` picks it up.
- **WebGPU 8-buffer limit**: Bind groups split across 2 groups to stay within limits.
- **Subgroup operations**: Tiled/interval shaders use `enable subgroups;` for warp-level shuffles.
- **GPU-only training loop**: `rmesh-train` does forward+loss+backward+Adam without CPU readback in the loop.
- **Indirect dispatch**: Compute shaders write `DrawIndirectCommand` buffers for dynamic dispatch counts.

### Python Integration

`rmesh-python` builds a `_native.cpython-*.so` via maturin. The Python package `rmesh_wgpu` provides:
- `RMeshRenderer` — Rust class with `forward()`, `backward()`, `train_step()`, `update_params()`, `get_params()`
- `autograd.py` — `torch.autograd.Function` wrapper (`RMeshForward`) and `nn.Module` wrapper (`RMeshModule`)

### Tests

Tests live in `crates/rmesh-render/tests/` with a shared `common/mod.rs` module providing:
- CPU reference renderer for comparison
- Random scene generators (`random_single_tet_scene`)
- GPU render helper (gracefully returns `None` if no adapter)
- Image comparison utilities

Test files: `single_tet_test.rs` (forward rendering), `multi_tet_test.rs` (multiple tets), `gradient_test.rs` (finite-difference gradient checks), `cross_renderer_test.rs`, `kernel_tests.rs`, `mrt_test.rs`, `overdraw_stats.rs`.

### Symbolic Backward Derivation

`scripts/derive_backward.py` uses SymPy to derive and validate the backward pass math (volume rendering integral, alpha blending, ray-plane intersection, SH/color chain). Run with `python scripts/derive_backward.py`.

## Data Format

`.rmesh` files are gzip-compressed binary:
- Header: `[vertex_count, tet_count, sh_degree, k_components]` as u32
- Start pose: 8× f32 (position + quaternion + pad)
- Vertices: `[N × 3]` f32, Indices: `[M × 4]` u32
- Densities: `[M]` u8 log-encoded → `exp((val-100)/20)`
- SH: PCA-compressed (mean + basis + weights as f16) → decompressed to direct coefficients on load
- Color gradients: `[M × 3]` f16
