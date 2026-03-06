# rmeshvk

wgpu-based differentiable tetrahedral volume renderer for Radiance Meshes. Rust/WebGPU port of the rendering and training backend, replacing the Slang/CUDA pipeline.

## Prerequisites

- Rust stable (`rustup` — see `rust-toolchain.toml`)
- Python 3.8+ with a virtualenv
- GPU with Vulkan or Metal support
- `maturin` (installed in the venv)

## Build

```bash
# Build entire Rust workspace
cargo build

# Build in release mode
cargo build --release
```

## Tests

```bash
# Run all tests (requires GPU — tests gracefully skip if no adapter)
cargo test

# Run tests for a specific crate
cargo test -p rmesh-render
cargo test -p rmesh-backward

# Run a specific test
cargo test -p rmesh-render test_center_view
```

## Python Extension

The `rmesh-python` crate builds a native Python module (`rmesh_wgpu`) via maturin + PyO3.

```bash
# From the repo root, activate the venv and build:
source .venv/bin/activate
cd crates/rmesh-python
maturin develop          # debug build
maturin develop --release  # optimized build
cd -
```

Verify the install:

```bash
python -c "from rmesh_wgpu import RMeshRenderer; print('OK')"
```

### Running the benchmark

```bash
source .venv/bin/activate
cd crates/rmesh-python && maturin develop --release && cd -
uv run tests/bench_backward.py
```

The benchmark tests forward and backward pass performance at various tet counts (200–10k vertices) at 512x512 resolution. Requires `scipy`, `torch`, and `numpy`.

## Viewer

```bash
cargo run -p rmesh-viewer --release -- <input.rmesh>
```

Controls: left-drag to orbit, scroll to zoom, Escape to quit.

## Crate Overview

| Crate | Purpose |
|-------|---------|
| `rmesh-shaders` | WGSL shader sources + shared CPU/GPU types |
| `rmesh-data` | `.rmesh` file loading, PCA decompression |
| `rmesh-render` | Forward pipeline: compute, sort, rasterize |
| `rmesh-backward` | Backward pipeline: loss, backward compute, Adam |
| `rmesh-train` | GPU-only training loop |
| `rmesh-viewer` | Interactive winit/wgpu viewer |
| `rmesh-python` | PyO3/maturin bindings for Python/PyTorch |

## Python API

```python
from rmesh_wgpu import RMeshRenderer

renderer = RMeshRenderer(
    vertices,    # [N*3] f32
    indices,     # [M*4] u32
    sh_coeffs,   # [M*nc*3] f32
    densities,   # [M] f32
    color_grads,  # [M*3] f32
    circumdata,  # [M*4] f32 (cx, cy, cz, r^2)
    sh_degree,   # int (0-3)
    width, height,
)

image = renderer.forward(cam_pos, vp, inv_vp)       # [H, W, 4] f32
grads = renderer.backward(dl_d_image)                # dict of gradient arrays
loss  = renderer.train_step(cam_pos, vp, inv_vp,     # all-in-one GPU step
                            gt_image, lr_sh, lr_verts,
                            lr_dens, lr_grads, loss_type)
```

PyTorch integration via `rmesh_wgpu.autograd`:

```python
from rmesh_wgpu.autograd import RMeshModule

model = RMeshModule(vertices, indices, sh_coeffs, densities,
                    color_grads, circumdata, sh_degree, width, height)
image = model(cam_pos, vp, inv_vp)  # differentiable
```
