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
use half::f16;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::io::Read;

/// Parallel `chunks_mut` + `enumerate` + `for_each` when the `parallel` feature is
/// enabled, sequential fallback otherwise.
macro_rules! par_chunks_for_each {
    ($data:expr, $chunk_size:expr, $closure:expr) => {{
        #[cfg(feature = "parallel")]
        {
            $data
                .par_chunks_mut($chunk_size)
                .enumerate()
                .for_each($closure);
        }
        #[cfg(not(feature = "parallel"))]
        {
            $data
                .chunks_mut($chunk_size)
                .enumerate()
                .for_each($closure);
        }
    }};
}

/// Loaded scene data, ready for GPU upload.
#[derive(Clone)]
pub struct SceneData {
    /// Vertex positions [N × 3] f32
    pub vertices: Vec<f32>,
    /// Tet vertex indices [M × 4] u32
    pub indices: Vec<u32>,
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
}

/// SH color coefficients loaded from an .rmesh file (separate from scene geometry).
#[derive(Clone)]
pub struct ShCoeffs {
    /// Direct SH coefficients [M × (deg+1)^2 × 3] f32 (decompressed from PCA)
    pub coeffs: Vec<f32>,
    /// SH degree (0–3)
    pub degree: u32,
}

impl ShCoeffs {
    /// Number of SH coefficients per channel: (deg+1)^2
    pub fn num_coeffs(&self) -> u32 {
        (self.degree + 1) * (self.degree + 1)
    }

    /// Stride per tet: num_coeffs * 3 channels
    pub fn stride(&self) -> u32 {
        self.num_coeffs() * 3
    }

    /// Create zeroed-out coefficients (degree 0, all zeros).
    pub fn zero(tet_count: u32) -> Self {
        ShCoeffs {
            coeffs: vec![0.0; tet_count as usize * 3],
            degree: 0,
        }
    }
}

/// Load a .rmesh file (gzip-compressed binary).
pub fn load_rmesh(data: &[u8]) -> Result<(SceneData, ShCoeffs)> {
    let decompressed = decompress_gzip(data)?;
    parse_rmesh(&decompressed)
}

/// Load a raw (uncompressed) .rmesh binary.
pub fn load_rmesh_raw(data: &[u8]) -> Result<(SceneData, ShCoeffs)> {
    parse_rmesh(data)
}

fn decompress_gzip(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() >= 2 && data[0] == 0x1f && data[1] == 0x8b {
        log::info!("Decompressing gzip ({:.1} MB)...", data.len() as f64 / 1e6);
        let t0 = std::time::Instant::now();
        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder
            .read_to_end(&mut decompressed)
            .context("Failed to decompress gzip data")?;
        log::info!(
            "Decompressed: {:.1} MB in {:.2}s",
            decompressed.len() as f64 / 1e6,
            t0.elapsed().as_secs_f64()
        );
        Ok(decompressed)
    } else {
        Ok(data.to_vec())
    }
}

// ---------------------------------------------------------------------------
// Zero-copy slice readers
// ---------------------------------------------------------------------------

/// Read a contiguous &[f32] from LE byte buffer (zero-copy on LE platforms).
fn slice_f32(data: &[u8], off: &mut usize, count: usize) -> Vec<f32> {
    let byte_len = count * 4;
    let bytes = &data[*off..*off + byte_len];
    *off += byte_len;
    // bytemuck can cast aligned slices directly; fall back to per-element for safety
    if bytes.as_ptr() as usize % 4 == 0 {
        bytemuck::cast_slice::<u8, f32>(bytes).to_vec()
    } else {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
}

/// Read a contiguous &[u32] from LE byte buffer.
fn slice_u32(data: &[u8], off: &mut usize, count: usize) -> Vec<u32> {
    let byte_len = count * 4;
    let bytes = &data[*off..*off + byte_len];
    *off += byte_len;
    if bytes.as_ptr() as usize % 4 == 0 {
        bytemuck::cast_slice::<u8, u32>(bytes).to_vec()
    } else {
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
}

/// Bulk f16 → f32 conversion with parallelism for large arrays.
fn slice_f16_to_f32(data: &[u8], off: &mut usize, count: usize) -> Vec<f32> {
    let byte_len = count * 2;
    let bytes = &data[*off..*off + byte_len];
    *off += byte_len;
    // For large arrays (>100K elements), use parallel conversion when available
    if count > 100_000 {
        let mut out = vec![0.0f32; count];
        par_chunks_for_each!(out, 8192, |(chunk_idx, chunk): (usize, &mut [f32])| {
            let start = chunk_idx * 8192;
            for (i, val) in chunk.iter_mut().enumerate() {
                let bi = (start + i) * 2;
                *val = f16::from_le_bytes(bytes[bi..bi + 2].try_into().unwrap()).to_f32();
            }
        });
        out
    } else {
        bytes
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes(c.try_into().unwrap()).to_f32())
            .collect()
    }
}

fn read_u32_val(data: &[u8], off: &mut usize) -> u32 {
    let val = u32::from_le_bytes(data[*off..*off + 4].try_into().unwrap());
    *off += 4;
    val
}

fn read_f32_val(data: &[u8], off: &mut usize) -> f32 {
    let val = f32::from_le_bytes(data[*off..*off + 4].try_into().unwrap());
    *off += 4;
    val
}

// ---------------------------------------------------------------------------
// Main parser
// ---------------------------------------------------------------------------

fn parse_rmesh(data: &[u8]) -> Result<(SceneData, ShCoeffs)> {
    let t_total = std::time::Instant::now();
    let mut offset = 0usize;

    // Header
    let vertex_count = read_u32_val(data, &mut offset);
    let tet_count = read_u32_val(data, &mut offset);
    let sh_degree = read_u32_val(data, &mut offset);
    let k_components = read_u32_val(data, &mut offset);

    log::info!(
        "rmesh: {} vertices, {} tets, SH degree {}, {} PCA components",
        vertex_count, tet_count, sh_degree, k_components
    );

    // Start pose (8 floats: x, y, z, qx, qy, qz, qw, _pad)
    let mut start_pose = [0.0f32; 7];
    for i in 0..7 {
        start_pose[i] = read_f32_val(data, &mut offset);
    }
    let _pad = read_f32_val(data, &mut offset);

    // Vertices [N × 3] f32 — zero-copy
    let t0 = std::time::Instant::now();
    let vertices = slice_f32(data, &mut offset, (vertex_count * 3) as usize);
    let indices = slice_u32(data, &mut offset, (tet_count * 4) as usize);
    log::info!(
        "Vertices + indices: {:.2}s ({:.1} MB)",
        t0.elapsed().as_secs_f64(),
        (vertices.len() * 4 + indices.len() * 4) as f64 / 1e6
    );

    // Densities [M] u8, log-encoded → parallel decode
    let t0 = std::time::Instant::now();
    let densities_u8 = &data[offset..offset + tet_count as usize];
    offset += tet_count as usize;
    let mut densities = vec![0.0f32; tet_count as usize];
    par_chunks_for_each!(densities, 8192, |(ci, chunk): (usize, &mut [f32])| {
        let start = ci * 8192;
        for (i, d) in chunk.iter_mut().enumerate() {
            *d = ((densities_u8[start + i] as f32 - 100.0) / 20.0).exp();
        }
    });

    // Align to 4 bytes
    offset = (offset + 3) & !3;

    // SH data: PCA-compressed
    let num_coeffs = ((sh_degree + 1) * (sh_degree + 1)) as usize;
    let total_dims = num_coeffs * 3;

    // SH Mean + Basis are small, Weights + Grads are large → parallel f16 conversion
    let mean = slice_f16_to_f32(data, &mut offset, total_dims);
    let basis = slice_f16_to_f32(data, &mut offset, k_components as usize * total_dims);
    let weights = slice_f16_to_f32(data, &mut offset, tet_count as usize * k_components as usize);

    offset = (offset + 3) & !3;
    let color_grads = slice_f16_to_f32(data, &mut offset, tet_count as usize * 3);
    log::info!("Densities + SH + gradients: {:.2}s", t0.elapsed().as_secs_f64());

    // --- Parallel PCA decompression ---
    // result[tet][feat] = mean[feat] + Σ_k(weight[tet][k] * basis[k][feat])
    log::info!(
        "Decompressing PCA ({} tets × {} dims × {} components)...",
        tet_count, total_dims, k_components
    );
    let t0 = std::time::Instant::now();
    let k = k_components as usize;
    let mut sh_coeffs = vec![0.0f32; tet_count as usize * total_dims];

    par_chunks_for_each!(sh_coeffs, total_dims, |(tet, chunk): (usize, &mut [f32])| {
        let w_base = tet * k;
        for feat in 0..total_dims {
            let mut val = mean[feat];
            for ki in 0..k {
                val += weights[w_base + ki] * basis[ki * total_dims + feat];
            }
            chunk[feat] = val;
        }
    });
    log::info!("PCA decompression: {:.2}s", t0.elapsed().as_secs_f64());

    // --- Parallel circumsphere computation ---
    log::info!("Computing circumspheres ({} tets)...", tet_count);
    let t0 = std::time::Instant::now();
    let circumdata = compute_circumspheres_parallel(&vertices, &indices, tet_count as usize);
    log::info!("Circumspheres: {:.2}s", t0.elapsed().as_secs_f64());

    log::info!("Total parse time: {:.2}s", t_total.elapsed().as_secs_f64());

    let sh = ShCoeffs {
        coeffs: sh_coeffs,
        degree: sh_degree,
    };

    Ok((
        SceneData {
            vertices,
            indices,
            densities,
            color_grads,
            circumdata,
            start_pose,
            vertex_count,
            tet_count,
        },
        sh,
    ))
}

/// Parallel circumsphere computation using rayon.
/// Matches stable_power.slang: double precision + degenerate tet handling.
fn compute_circumspheres_parallel(
    vertices: &[f32],
    indices: &[u32],
    tet_count: usize,
) -> Vec<f32> {
    use glam::DVec3;

    let mut circumdata = vec![0.0f32; tet_count * 4];

    par_chunks_for_each!(circumdata, 4, |(i, out): (usize, &mut [f32])| {
            let i0 = indices[i * 4] as usize;
            let i1 = indices[i * 4 + 1] as usize;
            let i2 = indices[i * 4 + 2] as usize;
            let i3 = indices[i * 4 + 3] as usize;

            // Double precision to match stable_power.slang
            let v0 = DVec3::new(
                vertices[i0 * 3] as f64,
                vertices[i0 * 3 + 1] as f64,
                vertices[i0 * 3 + 2] as f64,
            );
            let v1 = DVec3::new(
                vertices[i1 * 3] as f64,
                vertices[i1 * 3 + 1] as f64,
                vertices[i1 * 3 + 2] as f64,
            );
            let v2 = DVec3::new(
                vertices[i2 * 3] as f64,
                vertices[i2 * 3 + 1] as f64,
                vertices[i2 * 3 + 2] as f64,
            );
            let v3 = DVec3::new(
                vertices[i3 * 3] as f64,
                vertices[i3 * 3 + 1] as f64,
                vertices[i3 * 3 + 2] as f64,
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

            let denom = 2.0 * a.dot(cross_bc);

            if denom.abs() < 1e-10 {
                // Degenerate tet — match stable_power.slang logic
                let max_dist = (v1 - v0)
                    .length()
                    .max((v2 - v0).length())
                    .max((v3 - v0).length())
                    .max((v2 - v1).length())
                    .max((v3 - v1).length())
                    .max((v3 - v2).length());

                if max_dist < 1e-7 {
                    // Tiny tet: use centroid, radius = min vertex distance
                    let center = 0.25 * (v0 + v1 + v2 + v3);
                    let r = (v0 - center)
                        .length()
                        .min((v1 - center).length())
                        .min((v2 - center).length())
                        .min((v3 - center).length());
                    out[0] = center.x as f32;
                    out[1] = center.y as f32;
                    out[2] = center.z as f32;
                    out[3] = (r * r) as f32;
                } else {
                    // Near-planar tet: mark invalid (centroid, r²=-1e20)
                    // Sort key = |diff|² - r² will be huge → sorted to back
                    let center = 0.25 * (v0 + v1 + v2 + v3);
                    out[0] = center.x as f32;
                    out[1] = center.y as f32;
                    out[2] = center.z as f32;
                    out[3] = 1e20_f32; // huge r² → power always very negative → back of sort
                }
            } else {
                let rel = (aa * cross_bc + bb * cross_ca + cc * cross_ab) / denom;
                let center = v0 + rel;
                out[0] = center.x as f32;
                out[1] = center.y as f32;
                out[2] = center.z as f32;
                out[3] = rel.dot(rel) as f32;
            }
        });

    circumdata
}

// ---------------------------------------------------------------------------
// PLY loader
// ---------------------------------------------------------------------------

/// Load a binary PLY file written by `BaseModel.save2ply()`.
///
/// Format:
///   element vertex N: (float x, float y, float z)
///   element tetrahedron M: (list uchar int indices, float s,
///     float grd_x, grd_y, grd_z, float sh_0_r, sh_0_g, sh_0_b, ...)
///
/// The SH coefficients in the PLY are stored **interleaved by coefficient**:
///   [sh_0_r, sh_0_g, sh_0_b, sh_1_r, sh_1_g, sh_1_b, ...]
/// but the viewer/renderer expects **planar by channel**:
///   [sh_0_r, sh_1_r, ..., sh_N_r, sh_0_g, sh_1_g, ..., sh_N_g, sh_0_b, ...]
pub fn load_ply(data: &[u8]) -> Result<(SceneData, ShCoeffs)> {
    let t_total = std::time::Instant::now();

    // --- Parse ASCII header ---
    let header_end = find_header_end(data).context("Missing end_header in PLY")?;
    let header = std::str::from_utf8(&data[..header_end]).context("Invalid PLY header")?;

    let mut vertex_count = 0u32;
    let mut tet_count = 0u32;
    let mut num_sh = 0u32; // number of SH coefficient groups (sh_0, sh_1, ..., sh_N)
    let mut is_binary_le = false;
    let mut in_tet_element = false;

    for line in header.lines() {
        let line = line.trim();
        if line == "format binary_little_endian 1.0" {
            is_binary_le = true;
        } else if line.starts_with("element vertex ") {
            vertex_count = line[15..].trim().parse().context("Bad vertex count")?;
            in_tet_element = false;
        } else if line.starts_with("element tetrahedron ") {
            tet_count = line[20..].trim().parse().context("Bad tet count")?;
            in_tet_element = true;
        } else if line.starts_with("element ") {
            in_tet_element = false;
        } else if in_tet_element && line.starts_with("property float sh_") && line.ends_with("_r") {
            num_sh += 1;
        }
    }

    anyhow::ensure!(is_binary_le, "Only binary_little_endian PLY is supported");
    anyhow::ensure!(vertex_count > 0 && tet_count > 0, "Empty PLY");

    let sh_degree = (num_sh as f64).sqrt() as u32 - 1;
    let nc = (sh_degree + 1) * (sh_degree + 1);
    anyhow::ensure!(nc == num_sh, "SH count {num_sh} is not a perfect square");

    log::info!(
        "PLY: {} vertices, {} tets, {} SH coefficients (degree {})",
        vertex_count, tet_count, num_sh, sh_degree,
    );

    let mut offset = header_end;

    // --- Read vertices: N × 12 bytes (contiguous f32 x,y,z) ---
    let t0 = std::time::Instant::now();
    let vertices = slice_f32(data, &mut offset, (vertex_count * 3) as usize);
    log::info!("Vertices: {:.2}s", t0.elapsed().as_secs_f64());

    // --- Read tet data in parallel ---
    // Each tet record: [u8:count=4][i32×4:indices][f32:density][f32×3:grad][f32×3×num_sh:SH]
    let num_scalar_props = 1 + 3 + num_sh as usize * 3; // density + grad(3) + SH
    let tet_stride = 1 + 4 * 4 + num_scalar_props * 4; // 1 byte list prefix + indices + floats
    let tet_data = &data[offset..offset + tet_stride * tet_count as usize];
    offset += tet_stride * tet_count as usize;

    log::info!(
        "Parsing {} tets ({:.1} MB, {} bytes/tet)...",
        tet_count,
        tet_data.len() as f64 / 1e6,
        tet_stride,
    );
    let t0 = std::time::Instant::now();

    let nc_usize = nc as usize;
    let total_dims = nc_usize * 3; // SH floats per tet in planar layout

    // Allocate output arrays
    let mut indices = vec![0u32; tet_count as usize * 4];
    let mut densities = vec![0.0f32; tet_count as usize];
    let mut color_grads = vec![0.0f32; tet_count as usize * 3];
    let mut sh_coeffs = vec![0.0f32; tet_count as usize * total_dims];

    // Parallel parse: indices (4 u32 per tet)
    let _n_tets = tet_count as usize;
    par_chunks_for_each!(indices, 4, |(t, idx): (usize, &mut [u32])| {
        let rec = &tet_data[t * tet_stride + 1..]; // skip list count byte
        for k in 0..4 {
            idx[k] = i32::from_le_bytes(rec[k * 4..k * 4 + 4].try_into().unwrap()) as u32;
        }
    });

    // Parallel parse: density + gradients + SH (all scalar properties)
    // Scalar props start at offset 17 within each tet record (1 list_count + 16 indices)
    let scalar_off = 17usize;

    par_chunks_for_each!(densities, 1, |(t, d): (usize, &mut [f32])| {
        let off = t * tet_stride + scalar_off;
        d[0] = f32::from_le_bytes(tet_data[off..off + 4].try_into().unwrap());
    });

    par_chunks_for_each!(color_grads, 3, |(t, g): (usize, &mut [f32])| {
        let off = t * tet_stride + scalar_off + 4;
        for k in 0..3 {
            g[k] = f32::from_le_bytes(tet_data[off + k * 4..off + k * 4 + 4].try_into().unwrap());
        }
    });

    // SH: interleaved [sh_0_r, sh_0_g, sh_0_b, sh_1_r, ...] → planar [R..., G..., B...]
    let sh_byte_off = scalar_off + 16; // density(4) + grad(12)
    par_chunks_for_each!(sh_coeffs, total_dims, |(t, sh_out): (usize, &mut [f32])| {
        let off = t * tet_stride + sh_byte_off;
        for coeff in 0..nc_usize {
            for ch in 0..3usize {
                let src = off + (coeff * 3 + ch) * 4;
                sh_out[ch * nc_usize + coeff] =
                    f32::from_le_bytes(tet_data[src..src + 4].try_into().unwrap());
            }
        }
    });

    log::info!("Tet parsing: {:.2}s", t0.elapsed().as_secs_f64());

    // --- Compute circumspheres ---
    log::info!("Computing circumspheres ({} tets)...", tet_count);
    let t0 = std::time::Instant::now();
    let circumdata = compute_circumspheres_parallel(&vertices, &indices, tet_count as usize);
    log::info!("Circumspheres: {:.2}s", t0.elapsed().as_secs_f64());

    log::info!("Total PLY parse time: {:.2}s", t_total.elapsed().as_secs_f64());

    let sh = ShCoeffs {
        coeffs: sh_coeffs,
        degree: sh_degree,
    };

    Ok((
        SceneData {
            vertices,
            indices,
            densities,
            color_grads,
            circumdata,
            start_pose: [0.0; 7],
            vertex_count,
            tet_count,
        },
        sh,
    ))
}

/// Find the byte offset just past `end_header\n`.
fn find_header_end(data: &[u8]) -> Option<usize> {
    let marker = b"end_header\n";
    data.windows(marker.len())
        .position(|w| w == marker)
        .map(|pos| pos + marker.len())
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
