//! Data loading for .rmesh files and PLY files.
//!
//! .rmesh format (from webrm convert.py):
//!   Header: [vertex_count, tet_count, sh_degree, k_components] as u32
//!   Start pose: [x, y, z, qx, qy, qz, qw, _pad] as f32 (8 floats)
//!   Vertices: [N × 3] f32
//!   Indices: [M × 4] u32
//!   Densities: [M] u8 (log-encoded: density = exp((val-100)/20))
//!   Alignment padding to 4 bytes
//!   SH Mean: [(deg+1)^2 × 3] f16
//!   SH Basis: [k × (deg+1)^2 × 3] f16
//!   SH Weights: [M × k] f16
//!   Alignment padding to 4 bytes
//!   Gradients: [M × 3] f16
//!
//! For training, we decompress PCA back to direct SH coefficients per tet.

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use glam::Vec3;
use half::f16;
use std::io::Read;

/// Loaded scene data, ready for GPU upload.
#[derive(Clone)]
pub struct SceneData {
    /// Vertex positions [N × 3] f32
    pub vertices: Vec<f32>,
    /// Tet vertex indices [M × 4] u32
    pub indices: Vec<u32>,
    /// Direct SH coefficients [M × (deg+1)^2 × 3] f32 (decompressed from PCA if .rmesh)
    pub sh_coeffs: Vec<f32>,
    /// Per-tet density [M] f32
    pub densities: Vec<f32>,
    /// Per-tet color gradient [M × 3] f32
    pub color_grads: Vec<f32>,
    /// Circumsphere data [M × 4] f32 (center_x, center_y, center_z, radius²)
    pub circumdata: Vec<f32>,
    /// Initial camera pose [x, y, z, qx, qy, qz, qw]
    pub start_pose: [f32; 7],
    /// Number of vertices
    pub vertex_count: u32,
    /// Number of tetrahedra
    pub tet_count: u32,
    /// SH degree
    pub sh_degree: u32,
}

impl SceneData {
    /// Number of SH coefficients per channel: (deg+1)^2
    pub fn num_sh_coeffs(&self) -> u32 {
        (self.sh_degree + 1) * (self.sh_degree + 1)
    }
}

/// Load a .rmesh file (gzip-compressed binary).
pub fn load_rmesh(data: &[u8]) -> Result<SceneData> {
    // Decompress gzip
    let decompressed = decompress_gzip(data)?;
    parse_rmesh(&decompressed)
}

/// Load a raw (uncompressed) .rmesh binary.
pub fn load_rmesh_raw(data: &[u8]) -> Result<SceneData> {
    parse_rmesh(data)
}

fn decompress_gzip(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() >= 2 && data[0] == 0x1f && data[1] == 0x8b {
        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder
            .read_to_end(&mut decompressed)
            .context("Failed to decompress gzip data")?;
        Ok(decompressed)
    } else {
        Ok(data.to_vec())
    }
}

fn parse_rmesh(data: &[u8]) -> Result<SceneData> {
    let mut offset = 0usize;

    // Helper to read typed slices
    let read_u32 = |off: &mut usize| -> u32 {
        let val = u32::from_le_bytes(data[*off..*off + 4].try_into().unwrap());
        *off += 4;
        val
    };
    let read_f32 = |off: &mut usize| -> f32 {
        let val = f32::from_le_bytes(data[*off..*off + 4].try_into().unwrap());
        *off += 4;
        val
    };

    // Header
    let vertex_count = read_u32(&mut offset);
    let tet_count = read_u32(&mut offset);
    let sh_degree = read_u32(&mut offset);
    let k_components = read_u32(&mut offset);

    log::info!(
        "rmesh: {} vertices, {} tets, SH degree {}, {} PCA components",
        vertex_count, tet_count, sh_degree, k_components
    );

    // Start pose (8 floats: x, y, z, qx, qy, qz, qw, _pad)
    let mut start_pose = [0.0f32; 7];
    for i in 0..7 {
        start_pose[i] = read_f32(&mut offset);
    }
    let _pad = read_f32(&mut offset); // skip padding float

    // Vertices [N × 3] f32
    let vert_count = (vertex_count * 3) as usize;
    let vertices: Vec<f32> = (0..vert_count).map(|_| read_f32(&mut offset)).collect();

    // Indices [M × 4] u32
    let idx_count = (tet_count * 4) as usize;
    let indices: Vec<u32> = (0..idx_count).map(|_| read_u32(&mut offset)).collect();

    // Densities [M] u8, log-encoded
    let densities_u8: Vec<u8> = data[offset..offset + tet_count as usize].to_vec();
    offset += tet_count as usize;

    // Decode densities: exp((val - 100) / 20)
    let densities: Vec<f32> = densities_u8
        .iter()
        .map(|&v| ((v as f32 - 100.0) / 20.0).exp())
        .collect();

    // Align to 4 bytes
    offset = (offset + 3) & !3;

    // SH data: PCA-compressed
    let num_coeffs = ((sh_degree + 1) * (sh_degree + 1)) as usize;
    let total_dims = num_coeffs * 3; // coeffs × channels

    // SH Mean: [total_dims] f16
    let mean: Vec<f32> = (0..total_dims)
        .map(|_| {
            let val =
                f16::from_le_bytes(data[offset..offset + 2].try_into().unwrap());
            offset += 2;
            val.to_f32()
        })
        .collect();

    // SH Basis: [k × total_dims] f16
    let basis_count = k_components as usize * total_dims;
    let basis: Vec<f32> = (0..basis_count)
        .map(|_| {
            let val =
                f16::from_le_bytes(data[offset..offset + 2].try_into().unwrap());
            offset += 2;
            val.to_f32()
        })
        .collect();

    // SH Weights: [M × k] f16
    let weight_count = tet_count as usize * k_components as usize;
    let weights: Vec<f32> = (0..weight_count)
        .map(|_| {
            let val =
                f16::from_le_bytes(data[offset..offset + 2].try_into().unwrap());
            offset += 2;
            val.to_f32()
        })
        .collect();

    // Align to 4 bytes
    offset = (offset + 3) & !3;

    // Gradients [M × 3] f16
    let grad_count = tet_count as usize * 3;
    let color_grads: Vec<f32> = (0..grad_count)
        .map(|_| {
            let val =
                f16::from_le_bytes(data[offset..offset + 2].try_into().unwrap());
            offset += 2;
            val.to_f32()
        })
        .collect();

    // --- Decompress PCA → direct SH coefficients ---
    // result[tet][feat] = mean[feat] + Σ_k(weight[tet][k] * basis[k][feat])
    let mut sh_coeffs = vec![0.0f32; tet_count as usize * total_dims];

    for tet in 0..tet_count as usize {
        for feat in 0..total_dims {
            let mut val = mean[feat];
            for k in 0..k_components as usize {
                val += weights[tet * k_components as usize + k]
                    * basis[k * total_dims + feat];
            }
            sh_coeffs[tet * total_dims + feat] = val;
        }
    }

    // --- Compute circumspheres ---
    let circumdata = compute_circumspheres(&vertices, &indices, tet_count as usize);

    Ok(SceneData {
        vertices,
        indices,
        sh_coeffs,
        densities,
        color_grads,
        circumdata,
        start_pose,
        vertex_count,
        tet_count,
        sh_degree,
    })
}

/// Compute circumsphere (center + radius²) for each tetrahedron.
/// Matches both webrm's JS worker and delaunay_splatting's stable_power.slang.
fn compute_circumspheres(vertices: &[f32], indices: &[u32], tet_count: usize) -> Vec<f32> {
    let mut circumdata = vec![0.0f32; tet_count * 4];

    for i in 0..tet_count {
        let i0 = indices[i * 4] as usize;
        let i1 = indices[i * 4 + 1] as usize;
        let i2 = indices[i * 4 + 2] as usize;
        let i3 = indices[i * 4 + 3] as usize;

        let v0 = Vec3::new(
            vertices[i0 * 3],
            vertices[i0 * 3 + 1],
            vertices[i0 * 3 + 2],
        );
        let v1 = Vec3::new(
            vertices[i1 * 3],
            vertices[i1 * 3 + 1],
            vertices[i1 * 3 + 2],
        );
        let v2 = Vec3::new(
            vertices[i2 * 3],
            vertices[i2 * 3 + 1],
            vertices[i2 * 3 + 2],
        );
        let v3 = Vec3::new(
            vertices[i3 * 3],
            vertices[i3 * 3 + 1],
            vertices[i3 * 3 + 2],
        );

        let a = v1 - v0;
        let b = v2 - v0;
        let c = v3 - v0;

        let aa = a.dot(a);
        let bb = b.dot(b);
        let cc = c.dot(c);

        let cross_bc = b.cross(c);
        let cross_ca = c.cross(a);
        let cross_ab = a.cross(b);

        let mut denom = 2.0 * a.dot(cross_bc);

        // Degenerate tet handling (matches stable_power.slang)
        if denom.abs() < 1e-12 {
            denom = 1.0;
        }

        let r = (aa * cross_bc + bb * cross_ca + cc * cross_ab) / denom;
        let center = v0 + r;
        let r_sq = r.dot(r);

        circumdata[i * 4] = center.x;
        circumdata[i * 4 + 1] = center.y;
        circumdata[i * 4 + 2] = center.z;
        circumdata[i * 4 + 3] = r_sq;
    }

    circumdata
}

/// Load a ground truth image for training.
pub fn load_image(path: &std::path::Path) -> Result<(Vec<f32>, u32, u32)> {
    let img = image::open(path)
        .with_context(|| format!("Failed to load image: {}", path.display()))?
        .to_rgb32f();
    let (w, h) = img.dimensions();
    let data: Vec<f32> = img.into_raw();
    Ok((data, w, h))
}
