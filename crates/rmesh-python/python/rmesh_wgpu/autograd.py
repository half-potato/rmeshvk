"""PyTorch autograd integration for the wgpu renderer.

Provides:
  - RMeshForward: torch.autograd.Function wrapping forward/backward
  - RMeshModule: nn.Module holding learnable parameters + renderer
"""

import torch
import torch.nn as nn
import numpy as np

from rmesh_wgpu import RMeshRenderer


class RMeshForward(torch.autograd.Function):
    """Custom autograd function that delegates rendering to wgpu.

    Forward: uploads params → runs wgpu forward → returns image tensor.
    Backward: uploads dL/d(image) → runs wgpu backward → returns param gradients.
    """

    @staticmethod
    def forward(ctx, renderer, cam_pos, vp, c2w_intrinsics, vertices, base_colors, densities, color_grads):
        # Upload current parameters to GPU
        renderer.update_params(
            vertices.detach().cpu().numpy().ravel(),
            base_colors.detach().cpu().numpy().ravel(),
            densities.detach().cpu().numpy().ravel(),
            color_grads.detach().cpu().numpy().ravel(),
        )

        # Camera data as flat numpy arrays
        cam_np = cam_pos.detach().cpu().numpy().ravel().astype(np.float32)
        vp_np = vp.detach().cpu().numpy().ravel().astype(np.float32)
        c2w_int_np = c2w_intrinsics.detach().cpu().numpy().ravel().astype(np.float32)

        # Run forward render
        image_np = renderer.forward_tiled(cam_np, vp_np, c2w_int_np)

        ctx.renderer = renderer
        ctx.save_for_backward(vertices, base_colors, densities, color_grads)

        device = vertices.device
        return torch.from_numpy(image_np.copy()).to(device)

    @staticmethod
    def backward(ctx, grad_output):
        dl_d_image = grad_output.detach().cpu().numpy().astype(np.float32)
        grads = ctx.renderer.backward(dl_d_image)

        device = ctx.saved_tensors[0].device

        d_vertices = torch.from_numpy(grads["d_vertices"].copy()).to(device)
        d_base_colors = torch.from_numpy(grads["d_base_colors"].copy()).to(device)
        d_densities = torch.from_numpy(grads["d_densities"].copy()).to(device)
        d_color_grads = torch.from_numpy(grads["d_color_grads"].copy()).to(device)

        return (
            None,  # renderer
            None,  # cam_pos
            None,  # vp
            None,  # c2w_intrinsics
            d_vertices,
            d_base_colors,
            d_densities,
            d_color_grads,
        )


class RMeshModule(nn.Module):
    """nn.Module that wraps RMeshRenderer with learnable parameters.

    Usage:
        model = RMeshModule(vertices, indices, base_colors, densities,
                            color_grads, circumdata, w, h)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        image = model(cam_pos, vp, c2w_intrinsics)
        loss = (image - gt).pow(2).sum()
        loss.backward()
        optimizer.step()
    """

    def __init__(
        self,
        vertices: np.ndarray,
        indices: np.ndarray,
        base_colors: np.ndarray,
        densities: np.ndarray,
        color_grads: np.ndarray,
        circumdata: np.ndarray,
        width: int,
        height: int,
    ):
        super().__init__()

        self.vertices = nn.Parameter(torch.tensor(vertices.ravel(), dtype=torch.float32))
        self.base_colors = nn.Parameter(torch.tensor(base_colors.ravel(), dtype=torch.float32))
        self.densities = nn.Parameter(torch.tensor(densities.ravel(), dtype=torch.float32))
        self.color_grads = nn.Parameter(torch.tensor(color_grads.ravel(), dtype=torch.float32))

        # Non-differentiable data
        self._indices = indices.ravel().astype(np.uint32)
        self._circumdata = circumdata.ravel().astype(np.float32)
        self._width = width
        self._height = height

        # Create the Rust renderer
        self.renderer = RMeshRenderer(
            vertices.ravel().astype(np.float32),
            self._indices,
            base_colors.ravel().astype(np.float32),
            densities.ravel().astype(np.float32),
            color_grads.ravel().astype(np.float32),
            self._circumdata,
            width,
            height,
        )

    def forward(self, cam_pos, vp, c2w_intrinsics):
        """Render an image from the given camera.

        Args:
            cam_pos: [3] tensor, camera position
            vp: [4, 4] tensor, view-projection matrix (column-major when flattened)
            c2w_intrinsics: [16] tensor, camera-to-world rotation + intrinsics

        Returns:
            [H, W, 4] tensor (premultiplied RGBA)
        """
        return RMeshForward.apply(
            self.renderer,
            cam_pos,
            vp,
            c2w_intrinsics,
            self.vertices,
            self.base_colors,
            self.densities,
            self.color_grads,
        )
