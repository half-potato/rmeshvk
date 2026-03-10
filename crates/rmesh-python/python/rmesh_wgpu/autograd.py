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
    def forward(ctx, renderer, cam_pos, vp, inv_vp, vertices, sh_coeffs, densities, color_grads):
        # Upload current parameters to GPU
        renderer.update_params(
            vertices.detach().cpu().numpy().ravel(),
            sh_coeffs.detach().cpu().numpy().ravel(),
            densities.detach().cpu().numpy().ravel(),
            color_grads.detach().cpu().numpy().ravel(),
        )

        # Camera data as flat numpy arrays
        cam_np = cam_pos.detach().cpu().numpy().ravel().astype(np.float32)
        vp_np = vp.detach().cpu().numpy().ravel().astype(np.float32)
        inv_vp_np = inv_vp.detach().cpu().numpy().ravel().astype(np.float32)

        # Run forward render
        image_np = renderer.forward_tiled(cam_np, vp_np, inv_vp_np)

        ctx.renderer = renderer
        ctx.save_for_backward(vertices, sh_coeffs, densities, color_grads)

        device = vertices.device
        return torch.from_numpy(image_np.copy()).to(device)

    @staticmethod
    def backward(ctx, grad_output):
        dl_d_image = grad_output.detach().cpu().numpy().astype(np.float32)
        grads = ctx.renderer.backward(dl_d_image)

        device = ctx.saved_tensors[0].device

        d_vertices = torch.from_numpy(grads["d_vertices"].copy()).to(device)
        d_sh_coeffs = torch.from_numpy(grads["d_sh_coeffs"].copy()).to(device)
        d_densities = torch.from_numpy(grads["d_densities"].copy()).to(device)
        d_color_grads = torch.from_numpy(grads["d_color_grads"].copy()).to(device)

        return (
            None,  # renderer
            None,  # cam_pos
            None,  # vp
            None,  # inv_vp
            d_vertices,
            d_sh_coeffs,
            d_densities,
            d_color_grads,
        )


class RMeshModule(nn.Module):
    """nn.Module that wraps RMeshRenderer with learnable parameters.

    Usage:
        model = RMeshModule(vertices, indices, sh_coeffs, densities,
                            color_grads, circumdata, sh_degree, w, h)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        image = model(cam_pos, vp, inv_vp)
        loss = (image - gt).pow(2).sum()
        loss.backward()
        optimizer.step()
    """

    def __init__(
        self,
        vertices: np.ndarray,
        indices: np.ndarray,
        sh_coeffs: np.ndarray,
        densities: np.ndarray,
        color_grads: np.ndarray,
        circumdata: np.ndarray,
        sh_degree: int,
        width: int,
        height: int,
    ):
        super().__init__()

        self.vertices = nn.Parameter(torch.tensor(vertices.ravel(), dtype=torch.float32))
        self.sh_coeffs = nn.Parameter(torch.tensor(sh_coeffs.ravel(), dtype=torch.float32))
        self.densities = nn.Parameter(torch.tensor(densities.ravel(), dtype=torch.float32))
        self.color_grads = nn.Parameter(torch.tensor(color_grads.ravel(), dtype=torch.float32))

        # Non-differentiable data
        self._indices = indices.ravel().astype(np.uint32)
        self._circumdata = circumdata.ravel().astype(np.float32)
        self._sh_degree = sh_degree
        self._width = width
        self._height = height

        # Create the Rust renderer
        self.renderer = RMeshRenderer(
            vertices.ravel().astype(np.float32),
            self._indices,
            sh_coeffs.ravel().astype(np.float32),
            densities.ravel().astype(np.float32),
            color_grads.ravel().astype(np.float32),
            self._circumdata,
            sh_degree,
            width,
            height,
        )

    def forward(self, cam_pos, vp, inv_vp):
        """Render an image from the given camera.

        Args:
            cam_pos: [3] tensor, camera position
            vp: [4, 4] tensor, view-projection matrix (column-major when flattened)
            inv_vp: [4, 4] tensor, inverse view-projection matrix

        Returns:
            [H, W, 4] tensor (premultiplied RGBA)
        """
        return RMeshForward.apply(
            self.renderer,
            cam_pos,
            vp,
            inv_vp,
            self.vertices,
            self.sh_coeffs,
            self.densities,
            self.color_grads,
        )
