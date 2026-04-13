import os
import tinyplypy
import numpy as np
import argparse
from io import BytesIO
import gzip
import math
from pathlib import Path
import json
import struct
from pyquaternion import Quaternion


C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

def eval_sh(deg: int, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = 0.28209479177387814 * sh[..., 0] + 0.5
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                0.4886025119029199 * y * sh[..., 1] +
                0.4886025119029199 * z * sh[..., 2] -
                0.4886025119029199 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    1.0925484305920792 * xy * sh[..., 4] +
                    -1.0925484305920792 * yz * sh[..., 5] +
                    0.31539156525252005 * (2.0 * zz - xx - yy) * sh[..., 6] +
                    -1.0925484305920792 * xz * sh[..., 7] +
                    0.5462742152960396 * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                -0.5900435899266435 * y * (3 * xx - yy) * sh[..., 9] +
                2.890611442640554 * xy * z * sh[..., 10] +
                -0.4570457994644658 * y * (4 * zz - xx - yy)* sh[..., 11] +
                0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                -0.4570457994644658 * x * (4 * zz - xx - yy) * sh[..., 13] +
                1.445305721320277 * z * (xx - yy) * sh[..., 14] +
                -0.5900435899266435 * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + 2.5033429417967046 * xy * (xx - yy) * sh[..., 16] +
                            -1.7701307697799304 * yz * (3 * xx - yy) * sh[..., 17] +
                            0.9461746957575601 * xy * (7 * zz - 1) * sh[..., 18] +
                            -0.6690465435572892 * yz * (7 * zz - 3) * sh[..., 19] +
                            0.10578554691520431 * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            -0.6690465435572892 * xz * (7 * zz - 3) * sh[..., 21] +
                            0.47308734787878004 * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            -1.7701307697799304 * xz * (xx - 3 * yy) * sh[..., 23] +
                            0.6258357354491761 * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result


def load_sh(tetra_dict):
    sh_names = [k for k in tetra_dict.keys() if k.startswith("sh_")]
    N = tetra_dict[f"sh_0_r"].shape[0]
    sh_coeffs = np.zeros((N, len(sh_names) // 3, 3))
    for i in range(len(sh_names) // 3):
        sh_coeffs[:, i, 0] = tetra_dict[f"sh_{i}_r"]
        sh_coeffs[:, i, 1] = tetra_dict[f"sh_{i}_g"]
        sh_coeffs[:, i, 2] = tetra_dict[f"sh_{i}_b"]
    return sh_coeffs

def calculate_circumcenters(vertices):
    """
    Compute the circumcenter of a tetrahedron.

    Args:
        vertices: Tensor of shape (..., 4, 3) containing the vertices of the tetrahedron(a).
                 The first dimension can be batched.

    Returns:
        circumcenter: Tensor of shape (..., 3) containing the circumcenter coordinates
    """
    # Compute vectors from v0 to other vertices
    a = vertices[..., 1, :] - vertices[..., 0, :]  # v1 - v0
    b = vertices[..., 2, :] - vertices[..., 0, :]  # v2 - v0
    c = vertices[..., 3, :] - vertices[..., 0, :]  # v3 - v0

    # Compute squares of lengths
    aa = np.sum(a * a, axis=-1, keepdims=True)  # |a|^2
    bb = np.sum(b * b, axis=-1, keepdims=True)  # |b|^2
    cc = np.sum(c * c, axis=-1, keepdims=True)  # |c|^2

    # Compute cross products
    cross_bc = np.cross(b, c, axis=-1)
    cross_ca = np.cross(c, a, axis=-1)
    cross_ab = np.cross(a, b, axis=-1)

    # Compute denominator
    denominator = 2.0 * np.sum(a * cross_bc, axis=-1, keepdims=True)

    # Create mask for small denominators
    mask = np.abs(denominator) < 1e-12

    # Compute circumcenter relative to verts[0]
    relative_circumcenter = (
        aa * cross_bc +
        bb * cross_ca +
        cc * cross_ab
    ) / np.where(mask, np.ones_like(denominator), denominator)


    radius = np.linalg.norm(a - relative_circumcenter, axis=-1)

    # Return absolute position
    return vertices[..., 0, :] + relative_circumcenter, radius

def tet_volumes(tets):
    v0 = tets[:, 0]
    v1 = tets[:, 1]
    v2 = tets[:, 2]
    v3 = tets[:, 3]

    a = v1 - v0
    b = v2 - v0
    c = v3 - v0

    mat = np.stack((a, b, c), axis=1)
    det = np.linalg.det(mat)

    vol = det / 6.0
    return vol

def softplus(x, b=10):
    return 0.1*np.log(1+np.exp(10*x))

def compress_matrix(data, k=16):
    # 1. Reshape to (Samples, Features)
    X = data.reshape(data.shape[0], -1) 
    
    # 2. Center the data
    mean_vec = np.mean(X, axis=0)
    X_centered = X - mean_vec

    # 3. Compute Covariance Matrix
    cov_matrix = np.dot(X_centered.T, X_centered) / (X_centered.shape[0] - 1)

    # 4. Eigen Decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 5. Sort eigenvectors (descending)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # 6. Compress: Keep top 'k' components
    # We force k=16 for GPU alignment (vec4 friendly)
    top_vectors = eigenvectors[:, :k]

    # Project data
    compressed_weights = np.dot(X_centered, top_vectors)
    
    # Transpose basis for easier storage: (k, 48)
    basis = top_vectors.T 

    return compressed_weights.astype(np.float16), basis.astype(np.float16), mean_vec.astype(np.float16)

def _process_ply_to_rmesh_buffer(ply_file_path, starting_cam, out_deg):
    """Process PLY to uncompressed .rmesh BytesIO buffer."""
    data = tinyplypy.read_ply(ply_file_path)
    vertices = np.stack([data['vertex']['x'], data['vertex']['y'], data['vertex']['z']], axis=1)
    indices = data['tetrahedron']['indices']
    s = data['tetrahedron']['s']
    mask = np.isnan(s) | (s < 1e-3)

    grd = np.stack([data['tetrahedron']['grd_x'], data['tetrahedron']['grd_y'], data['tetrahedron']['grd_z']], axis=1)

    # Load raw SH
    sh_dat = load_sh(data['tetrahedron']) # Shape: (N, 16, 3)
    # Clip to requested degree
    sh_dat = sh_dat[:, :(out_deg+1)**2]
    sh_dat = np.transpose(sh_dat, (0, 2, 1))

    # Filter masks before compression to avoid compressing garbage
    s = s[~mask]
    grd = grd[~mask]
    sh_dat = sh_dat[~mask]
    indices = indices[~mask]
    vertices = vertices # Vertices are not filtered in your original logic, keeping consistent

    # Compress
    # k=16 components
    k_components = 12
    sh_weights, sh_basis, sh_mean = compress_matrix(sh_dat, k=k_components)

    N = vertices.shape[0]
    M = indices.shape[0]

    # density_t = s.astype(np.float16)
    density_i = np.log(s.clip(min=1e-3))*20+100
    density_i[density_i<=1] = 0
    density_t = np.clip(density_i, 0, 255).astype(np.uint8)

    grd_t = grd.astype(np.float16)

    # sh_weights is already float16 from compress_matrix
    # flatten weights: [w0_tet0, w1_tet0 ... w0_tet1 ...]
    weights_flat = sh_weights.flatten()

    buffer = BytesIO()
    # Header: N, M, Degree, K_Components
    buffer.write(np.array([N, M, out_deg, k_components]).astype(np.uint32).tobytes())
    buffer.write(starting_cam.astype(np.float32).reshape(8).tobytes())
    buffer.write(vertices.tobytes())
    buffer.write(indices.tobytes())
    buffer.write(density_t.tobytes())

    # Align to 4 bytes
    while buffer.tell() % 4 != 0: buffer.write(b'\x00')

    # Write Compression Globals
    buffer.write(sh_mean.tobytes())  # (48,)
    buffer.write(sh_basis.tobytes()) # (16, 48)

    # Write Per-Tet Weights
    buffer.write(weights_flat.tobytes())

    # Align
    while buffer.tell() % 4 != 0: buffer.write(b'\x00')

    buffer.write(grd_t.tobytes())

    return buffer, mask


def process_ply_to_rmesh(ply_file_path, starting_cam, out_deg):
    buffer, _ = _process_ply_to_rmesh_buffer(ply_file_path, starting_cam, out_deg)
    compressed_bytes = gzip.compress(buffer.getvalue(), compresslevel=9)
    return compressed_bytes


# =============================================================================
# Tagged section helpers for extended .rmesh format
# =============================================================================

TAGGED_MAGIC = 0x524D5458  # "RMTX" in little-endian

def _align4(buffer):
    while buffer.tell() % 4 != 0:
        buffer.write(b'\x00')

def write_tagged_section(buffer, tag, data, dtype_code):
    """Write a single tagged section.

    dtype_code: 0=f16, 1=f32, 2=u8, 3=i32
    """
    tag_bytes = tag.encode('ascii')[:16].ljust(16, b'\x00')
    buffer.write(tag_bytes)
    buffer.write(np.array([dtype_code], dtype=np.uint32).tobytes())
    shape = data.shape
    buffer.write(np.array([len(shape)], dtype=np.uint32).tobytes())
    for dim in shape:
        buffer.write(np.array([dim], dtype=np.uint32).tobytes())
    if dtype_code == 0:
        payload = data.astype(np.float16).tobytes()
    elif dtype_code == 1:
        payload = data.astype(np.float32).tobytes()
    elif dtype_code == 2:
        payload = data.astype(np.uint8).tobytes()
    else:
        payload = data.astype(np.int32).tobytes()
    buffer.write(np.array([len(payload)], dtype=np.uint32).tobytes())
    buffer.write(payload)
    _align4(buffer)


def write_mlp_section(buffer, tag, state_dict, layer_prefix):
    """Write MLP weights as a tagged section.

    Binary format:
      num_layers: u32
      per layer:
        in_dim: u32
        out_dim: u32
        has_bias: u8
        weights: f16[out_dim * in_dim]  (row-major)
        bias: f16[out_dim]  (if has_bias)
    """
    import torch
    weight_keys = sorted(
        [k for k in state_dict if k.endswith('.weight') and layer_prefix in k],
        key=lambda k: int(k.split(layer_prefix + '.')[1].split('.')[0])
    )

    layer_data = BytesIO()
    layer_data.write(np.array([len(weight_keys)], dtype=np.uint32).tobytes())

    for wk in weight_keys:
        w = state_dict[wk].float().cpu().numpy()  # [out_dim, in_dim]
        bk = wk.replace('.weight', '.bias')
        has_bias = bk in state_dict
        out_dim, in_dim = w.shape

        layer_data.write(np.array([in_dim], dtype=np.uint32).tobytes())
        layer_data.write(np.array([out_dim], dtype=np.uint32).tobytes())
        layer_data.write(np.array([1 if has_bias else 0], dtype=np.uint8).tobytes())
        layer_data.write(w.astype(np.float16).tobytes())
        if has_bias:
            b = state_dict[bk].float().cpu().numpy()
            layer_data.write(b.astype(np.float16).tobytes())

    raw = layer_data.getvalue()
    # Write as a tagged section with shape = [byte_count]
    tag_bytes = tag.encode('ascii')[:16].ljust(16, b'\x00')
    buffer.write(tag_bytes)
    buffer.write(np.array([0], dtype=np.uint32).tobytes())   # dtype=f16 (nominal)
    buffer.write(np.array([1], dtype=np.uint32).tobytes())   # shape_rank=1
    buffer.write(np.array([len(raw)], dtype=np.uint32).tobytes())  # shape dim
    buffer.write(np.array([len(raw)], dtype=np.uint32).tobytes())  # data_bytes
    buffer.write(raw)
    _align4(buffer)


def process_pbr_to_rmesh(ply_file_path, pbr_dir, starting_cam, out_deg):
    """Process PLY + PBR checkpoint into extended .rmesh with tagged sections."""
    import torch

    # Base rmesh buffer (uncompressed)
    buffer, tet_mask = _process_ply_to_rmesh_buffer(ply_file_path, starting_cam, out_deg)

    _align4(buffer)

    # Write tagged section header
    buffer.write(np.array([TAGGED_MAGIC], dtype=np.uint32).tobytes())

    pbr_dir = Path(pbr_dir)

    # Load checkpoint data
    ckpt = torch.load(pbr_dir / "ckpt.pth", map_location='cpu', weights_only=False)
    ref_sd = torch.load(pbr_dir / "ref_renderer.pt", map_location='cpu', weights_only=False)
    brdf_sd = {k.removeprefix("brdf_model."): v for k, v in ref_sd.items() if k.startswith("brdf_model.")}

    # Load model for material baking and vertex normal computation
    import sys
    sys.path.insert(1, os.path.join(os.path.dirname(__file__), 'radiance_meshes'))
    from radiance_meshes.models.ingp_color import Model
    from renderer.ref_renderer import activate_aux
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model.load_ckpt(pbr_dir, device)

    # Per-tet material channels — load from .npy if available, otherwise bake
    if (pbr_dir / "roughness.npy").exists():
        roughness = np.load(pbr_dir / "roughness.npy")[~tet_mask]
        env_feature = np.load(pbr_dir / "env_feature.npy")[~tet_mask]
        albedo = np.load(pbr_dir / "albedo.npy")[~tet_mask]
    else:
        print("  Baking material channels from checkpoint...")
        model.current_sh_deg = 0
        cam_center = model.center.to(device)
        dummy_cam = type('Cam', (), {'camera_center': cam_center})()
        with torch.no_grad():
            _, cell_values = model.get_cell_values(dummy_cam)
            activated = activate_aux(cell_values)
            roughness = activated[:, 7].cpu().numpy()[~tet_mask]
            env_feature = activated[:, 8:12].cpu().numpy()[~tet_mask]
            albedo = activated[:, 12:15].cpu().numpy()[~tet_mask]

    # Full vertex normals: geometric (density-weighted tet face normals) + learned offsets
    with torch.no_grad():
        vn_full = model.compute_geometric_vertex_normals() + torch.cat([
            model.interior_vertex_normals, model.ext_vertex_normals])
        vn = vn_full.float().cpu().numpy()
    del model

    # Count sections: roughness, env_feature, albedo, vertex_normals, brdf_mlp, retro_head, tone_curve
    num_sections = 7
    buffer.write(np.array([num_sections], dtype=np.uint32).tobytes())

    # Material sections (f16)
    write_tagged_section(buffer, "roughness", roughness, 0)
    write_tagged_section(buffer, "env_feature", env_feature, 0)
    write_tagged_section(buffer, "albedo", albedo, 0)
    write_tagged_section(buffer, "vertex_normals", vn, 0)

    # BRDF MLP weights
    write_mlp_section(buffer, "brdf_mlp", brdf_sd, "mlp")

    # Retro head: weight [1, 4] + bias [1] → combined f16 array [5]
    retro_w = ref_sd['retro_head.weight'].float().cpu().numpy().flatten()  # [4]
    retro_b = ref_sd['retro_head.bias'].float().cpu().numpy().flatten()    # [1]
    retro_combined = np.concatenate([retro_w, retro_b])
    write_tagged_section(buffer, "retro_head", retro_combined, 0)

    # Monotonic spline tone curve: [y_knots..., slope, intercept, intercept_bias]
    # Viewer reconstructs knots_x = linspace(0, 1, n_knots) where n_knots = len - 3
    spline_sd = {k.removeprefix("spatial_spline."): v for k, v in ref_sd.items()
                 if k.startswith("spatial_spline.")}
    log_dy = spline_sd['log_dy'].float()
    dy = torch.nn.functional.softplus(log_dy)
    y_cumsum = torch.cumsum(dy, dim=0)
    y_knots = torch.cat([torch.zeros(1), y_cumsum])
    y_knots = (y_knots / y_knots[-1:].clamp(min=1e-6)).numpy()
    slope = spline_sd['slope'].float().numpy().flatten()
    intercept = spline_sd['intercept'].float().numpy().flatten()
    intercept_bias = np.array([spline_sd['intercept_bias'].item()])
    tone_curve = np.concatenate([y_knots, slope, intercept, intercept_bias])
    write_tagged_section(buffer, "tone_curve", tone_curve, 0)

    compressed = gzip.compress(buffer.getvalue(), compresslevel=9)
    return compressed

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_null_terminated_string(fid):
    """
    Reads characters from a file object until a null terminator is found.
    """
    char_list = []
    while True:
        char = fid.read(1)
        if char == b'' or char == b'\x00': # Stop on null terminator or end of file
            break
        char_list.append(char)
    return b''.join(char_list).decode("utf-8")

def read_extrinsics_binary(path_to_model_file, transform_matrix):
    """
    Reads the images.bin file, converts to GL coordinates, applies transform,
    and exports as [x, y, z, w] for JS compatibility.
    """
    images = []

    # Orthogonalize the transform matrix to prevent skewing
    u, _, vt = np.linalg.svd(transform_matrix[:3, :3])
    transform_matrix[:3, :3] = u @ vt

    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]

            # COLMAP qvec is [w, x, y, z]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]

            image_name = read_null_terminated_string(fid)

            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            fid.seek(24 * num_points2D, 1) # Skip 2D points

            # 1. W2C (World-to-Camera) from COLMAP
            w2c = np.eye(4)
            w2c[:3, :3] = Quaternion(qvec).rotation_matrix
            w2c[:3, 3] = tvec

            # 2. C2W (Camera-to-World)
            c2w = np.linalg.inv(w2c)

            # 3. Coordinate System Conversion: COLMAP (Y-Down, Z-Fwd) -> GL (Y-Up, Z-Back)
            # Flip local Y and Z axes by multiplying the rotation columns
            c2w[:3, 1:3] *= -1

            # 4. Apply scene transformation
            new_c2w = transform_matrix @ c2w

            new_tvec = new_c2w[:3, 3]
            new_q = Quaternion(matrix=new_c2w[:3, :3])

            # 5. Reorder to [x, y, z, w] for JS consumption
            qvec_js = np.array([new_q.x, new_q.y, new_q.z, new_q.w])

            images.append(np.concatenate([new_tvec, qvec_js, np.array([0.0])], axis=0))

    return images

def main():
    parser = argparse.ArgumentParser(description="Convert PLY files to SPLAT format.")
    parser.add_argument(
        "input_files", nargs="+", help="The input PLY files to process."
    )
    parser.add_argument(
        "--output", "-o", default="output.rmesh", help="The output RMESH file."
    )
    parser.add_argument(
        "--degree", "-d", type=int, default=3, help="The degree of the final mesh"
    )
    parser.add_argument(
        "--pbr", action="store_true",
        help="Enable PBR export (expects PBR checkpoint files in the input directory)"
    )
    args = parser.parse_args()
    for input_file in args.input_files:
        input_path = Path(input_file)
        if input_path.is_dir():
            ckpt_dir = input_path
            ply_file = ckpt_dir / "ckpt.ply"
            meta_dir = ckpt_dir
        else:
            ply_file = input_path
            ckpt_dir = input_path.parent
            meta_dir = input_path.parent

        print(f"Processing {ply_file}...")
        transform = np.loadtxt(meta_dir / "transform.txt")
        with (meta_dir / "config.json").open('r') as f:
            js = json.load(f)
            path = Path(js['dataset_path']) / "sparse/0/images.bin"
            extrinsics = read_extrinsics_binary(str(path), transform)
        print(args.pbr)

        if args.pbr:
            print(f"  PBR export from {ckpt_dir}")
            rmesh_data = process_pbr_to_rmesh(
                str(ply_file), str(ckpt_dir), extrinsics[0], 0
            )
        else:
            rmesh_data = process_ply_to_rmesh(str(ply_file), extrinsics[0], args.degree)

        output_file = (
            args.output if len(args.input_files) == 1 else input_file + ".rmesh"
        )
        with open(output_file, "wb") as f:
            f.write(rmesh_data)
        print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
