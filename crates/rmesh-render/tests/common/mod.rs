//! CPU reference renderer for testing GPU pipeline correctness.
//!
//! Implements the same math as the WGSL shaders (forward_compute, forward_vertex,
//! forward_fragment) in pure Rust with glam. No GPU required.

#![allow(dead_code)]

use glam::{Mat4, Vec3, Vec4};
pub use rand::Rng;
pub use rmesh_data::SceneData;

// ---------------------------------------------------------------------------
// SH constants (must match forward_compute.wgsl and shared.rs)
// ---------------------------------------------------------------------------

const C0: f32 = 0.28209479177387814;
const C1: f32 = 0.4886025119029199;
const C2_0: f32 = 1.0925484305920792;
const C2_1: f32 = -1.0925484305920792;
const C2_2: f32 = 0.31539156525252005;
const C2_3: f32 = -1.0925484305920792;
const C2_4: f32 = 0.5462742152960396;
const C3_0: f32 = -0.5900435899266435;
const C3_1: f32 = 2.890611442640554;
const C3_2: f32 = -0.4570457994644658;
const C3_3: f32 = 0.3731763325901154;
const C3_4: f32 = -0.4570457994644658;
const C3_5: f32 = 1.445305721320277;
const C3_6: f32 = -0.5900435899266435;

// Face winding: matches TET_FACES in forward_vertex.wgsl
const TET_FACES: [[usize; 3]; 4] = [[0, 2, 1], [1, 2, 3], [0, 3, 2], [3, 0, 1]];

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

pub fn softplus(x: f32) -> f32 {
    if x > 8.0 {
        x
    } else {
        0.1 * (1.0 + (10.0 * x).exp()).ln()
    }
}

fn phi(x: f32) -> f32 {
    if x.abs() < 1e-6 {
        1.0 - x * 0.5
    } else {
        (1.0 - (-x).exp()) / x
    }
}

pub fn eval_sh(dir: Vec3, sh_degree: u32, coeffs: &[f32]) -> f32 {
    let (x, y, z) = (dir.x, dir.y, dir.z);
    let mut val = C0 * coeffs[0];

    if sh_degree >= 1 {
        val += -C1 * y * coeffs[1];
        val += C1 * z * coeffs[2];
        val += -C1 * x * coeffs[3];
    }
    if sh_degree >= 2 {
        let (xx, yy, zz) = (x * x, y * y, z * z);
        val += C2_0 * x * y * coeffs[4];
        val += C2_1 * y * z * coeffs[5];
        val += C2_2 * (2.0 * zz - xx - yy) * coeffs[6];
        val += C2_3 * x * z * coeffs[7];
        val += C2_4 * (xx - yy) * coeffs[8];

        if sh_degree >= 3 {
            val += C3_0 * y * (3.0 * xx - yy) * coeffs[9];
            val += C3_1 * x * y * z * coeffs[10];
            val += C3_2 * y * (4.0 * zz - xx - yy) * coeffs[11];
            val += C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * coeffs[12];
            val += C3_4 * x * (4.0 * zz - xx - yy) * coeffs[13];
            val += C3_5 * z * (xx - yy) * coeffs[14];
            val += C3_6 * x * (xx - 3.0 * yy) * coeffs[15];
        }
    }
    val
}

// ---------------------------------------------------------------------------
// CPU compute pass: SH eval + color gradient + softplus
// ---------------------------------------------------------------------------

/// Evaluate SH and apply color gradient for a single tet (matches forward_compute.wgsl).
/// Returns (base_color, color_grad_vec) where base_color has offset baked in.
fn compute_tet_color(scene: &SceneData, tet_id: usize, cam_pos: Vec3) -> Vec3 {
    let verts = load_tet_verts(scene, tet_id);
    let centroid = (verts[0] + verts[1] + verts[2] + verts[3]) * 0.25;
    let dir = (centroid - cam_pos).normalize();

    let num_coeffs = ((scene.sh_degree + 1) * (scene.sh_degree + 1)) as usize;
    let sh_stride = num_coeffs * 3;
    let sh_base = tet_id * sh_stride;

    let mut result_color = Vec3::ZERO;
    result_color.x = eval_sh(dir, scene.sh_degree, &scene.sh_coeffs[sh_base..]);
    result_color.y = eval_sh(dir, scene.sh_degree, &scene.sh_coeffs[sh_base + num_coeffs..]);
    result_color.z = eval_sh(
        dir,
        scene.sh_degree,
        &scene.sh_coeffs[sh_base + 2 * num_coeffs..],
    );

    // SH2RGB bias
    result_color += Vec3::splat(0.5);

    // Color gradient offset
    let grad = load_color_grad(scene, tet_id);
    let offset = grad.dot(verts[0] - centroid);
    let input_val = result_color + Vec3::splat(offset);

    // Softplus activation
    Vec3::new(
        softplus(input_val.x),
        softplus(input_val.y),
        softplus(input_val.z),
    )
}

// ---------------------------------------------------------------------------
// CPU fragment: ray-tet intersection + volume integral
// ---------------------------------------------------------------------------

/// Per-pixel ray-tet intersection and volume integral (matches forward_fragment.wgsl).
/// Returns (premultiplied_rgb, alpha).
fn render_tet_pixel(
    cam_pos: Vec3,
    raw_ray_dir: Vec3, // unnormalized ray direction (from vertex interpolation)
    verts: &[Vec3; 4],
    density: f32,
    base_color: Vec3,
    color_grad: Vec3,
) -> [f32; 4] {
    let d = raw_ray_dir.length();

    // Ray-plane intersection for all 4 faces
    let mut numerators = [0.0f32; 4];
    let mut denominators = [0.0f32; 4];

    for (fi, face) in TET_FACES.iter().enumerate() {
        let va = verts[face[0]];
        let vb = verts[face[1]];
        let vc = verts[face[2]];
        let n = (vc - va).cross(vb - va);
        numerators[fi] = n.dot(va - cam_pos);
        denominators[fi] = n.dot(raw_ray_dir);
    }

    // Normalize denominators by ray length (matching fragment shader)
    let plane_denom: [f32; 4] = std::array::from_fn(|i| denominators[i] / d);
    let dc_dt = color_grad.dot(raw_ray_dir) / d;

    // Compute t values
    let all_t: [f32; 4] = std::array::from_fn(|i| numerators[i] / plane_denom[i]);

    // Classify entering/exiting
    let mut t_min = f32::NEG_INFINITY;
    let mut t_max = f32::INFINITY;

    for i in 0..4 {
        if plane_denom[i] > 0.0 {
            // Entering
            t_min = t_min.max(all_t[i]);
        } else if plane_denom[i] < 0.0 {
            // Exiting
            t_max = t_max.min(all_t[i]);
        }
        // denom == 0 → parallel, skip
    }

    let dist = (t_max - t_min).max(0.0);
    let od = density * dist;

    // base_color already has gradient offset to cam baked in (base_color = color + grad·(cam - v0))
    // dc_dt is grad·ray_dir
    let c_start = (base_color + Vec3::splat(dc_dt * t_min)).max(Vec3::ZERO);
    let c_end = (base_color + Vec3::splat(dc_dt * t_max)).max(Vec3::ZERO);

    // Volume integral: compute_integral(c_end, c_start, od)
    // Note: c_end (exit) is c0, c_start (enter) is c1, matching WGSL
    let alpha_t = (-od).exp(); // exp(-od), transmittance through this tet
    let phi_val = phi(od);
    let w0 = phi_val - alpha_t;
    let w1 = 1.0 - phi_val;
    let c = c_end * w0 + c_start * w1;
    let alpha = 1.0 - alpha_t;

    [c.x, c.y, c.z, alpha]
}

// ---------------------------------------------------------------------------
// Scene helpers
// ---------------------------------------------------------------------------

pub fn load_tet_verts(scene: &SceneData, tet_id: usize) -> [Vec3; 4] {
    let i0 = scene.indices[tet_id * 4] as usize;
    let i1 = scene.indices[tet_id * 4 + 1] as usize;
    let i2 = scene.indices[tet_id * 4 + 2] as usize;
    let i3 = scene.indices[tet_id * 4 + 3] as usize;
    [
        Vec3::new(
            scene.vertices[i0 * 3],
            scene.vertices[i0 * 3 + 1],
            scene.vertices[i0 * 3 + 2],
        ),
        Vec3::new(
            scene.vertices[i1 * 3],
            scene.vertices[i1 * 3 + 1],
            scene.vertices[i1 * 3 + 2],
        ),
        Vec3::new(
            scene.vertices[i2 * 3],
            scene.vertices[i2 * 3 + 1],
            scene.vertices[i2 * 3 + 2],
        ),
        Vec3::new(
            scene.vertices[i3 * 3],
            scene.vertices[i3 * 3 + 1],
            scene.vertices[i3 * 3 + 2],
        ),
    ]
}

fn load_color_grad(scene: &SceneData, tet_id: usize) -> Vec3 {
    Vec3::new(
        scene.color_grads[tet_id * 3],
        scene.color_grads[tet_id * 3 + 1],
        scene.color_grads[tet_id * 3 + 2],
    )
}

// ---------------------------------------------------------------------------
// Sorting (CPU back-to-front, matches forward_compute.wgsl depth key)
// ---------------------------------------------------------------------------

/// Sort tet indices back-to-front using circumsphere depth key.
/// Matches WGSL: key = ~bitcast<u32>(dot(diff,diff) - r²), sorted ascending.
pub fn sort_tets_back_to_front(scene: &SceneData, cam_pos: Vec3) -> Vec<u32> {
    let n = scene.tet_count as usize;
    let mut indexed: Vec<(u32, u32)> = (0..n as u32)
        .map(|i| {
            let cx = scene.circumdata[i as usize * 4];
            let cy = scene.circumdata[i as usize * 4 + 1];
            let cz = scene.circumdata[i as usize * 4 + 2];
            let r2 = scene.circumdata[i as usize * 4 + 3];
            let diff = Vec3::new(cx, cy, cz) - cam_pos;
            let depth = diff.dot(diff) - r2;
            let depth_bits = depth.to_bits();
            let key = !depth_bits;
            (key, i)
        })
        .collect();

    // Ascending sort by key (same as GPU radix sort)
    indexed.sort_by_key(|&(k, _)| k);
    indexed.into_iter().map(|(_, idx)| idx).collect()
}

// ---------------------------------------------------------------------------
// Camera ray construction
// ---------------------------------------------------------------------------

/// Compute world-space ray direction for pixel (px, py) matching the vertex shader.
///
/// The vertex shader projects world vertices to clip space. The fragment shader
/// gets `ray_dir = normalize(world_pos - cam)` interpolated across the face.
///
/// For CPU reference, we shoot a ray from cam through the pixel center using
/// the inverse VP matrix to unproject NDC → world.
pub fn pixel_ray_dir(inv_vp: Mat4, cam_pos: Vec3, px: f32, py: f32, w: f32, h: f32) -> Vec3 {
    // NDC: x ∈ [-1,1], y ∈ [-1,1] (wgpu: y=-1 at top, y=+1 at bottom)
    let ndc_x = (2.0 * px + 1.0) / w - 1.0;
    let ndc_y = 1.0 - (2.0 * py + 1.0) / h; // flip y for wgpu convention

    // Unproject a point on the near plane
    let clip_near = Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
    let world_h = inv_vp * clip_near;
    let world_pos = world_h.truncate() / world_h.w;

    (world_pos - cam_pos).normalize()
}

// ---------------------------------------------------------------------------
// Full CPU render
// ---------------------------------------------------------------------------

/// Render the full scene on CPU, returning RGBA per pixel (premultiplied alpha, composited).
/// Image is row-major, top-to-bottom (matching wgpu texture layout).
pub fn cpu_render_scene(
    scene: &SceneData,
    cam_pos: Vec3,
    vp: Mat4,
    inv_vp: Mat4,
    w: u32,
    h: u32,
) -> Vec<[f32; 4]> {
    let n = scene.tet_count as usize;
    let sorted = sort_tets_back_to_front(scene, cam_pos);

    // Pre-compute per-tet: evaluated color, color_grad, density, verts
    struct TetData {
        verts: [Vec3; 4],
        color: Vec3,
        grad: Vec3,
        density: f32,
    }

    let tet_data: Vec<TetData> = (0..n)
        .map(|i| {
            let verts = load_tet_verts(scene, i);
            let color = compute_tet_color(scene, i, cam_pos);
            let grad = load_color_grad(scene, i);
            let density = scene.densities[i];
            TetData {
                verts,
                color,
                grad,
                density,
            }
        })
        .collect();

    // Frustum cull (matching compute shader logic)
    let visible: Vec<bool> = (0..n)
        .map(|i| {
            let td = &tet_data[i];
            let mut min_x = f32::INFINITY;
            let mut max_x = f32::NEG_INFINITY;
            let mut min_y = f32::INFINITY;
            let mut max_y = f32::NEG_INFINITY;
            let mut min_z = f32::INFINITY;

            for v in &td.verts {
                let clip = vp * Vec4::new(v.x, v.y, v.z, 1.0);
                let inv_w = 1.0 / (clip.w + 1e-6);
                let ndc = clip.truncate() * inv_w;
                min_x = min_x.min(ndc.x);
                max_x = max_x.max(ndc.x);
                min_y = min_y.min(ndc.y);
                max_y = max_y.max(ndc.y);
                min_z = min_z.min(ndc.z);
            }

            if max_x < -1.0 || min_x > 1.0 || max_y < -1.0 || min_y > 1.0 || min_z > 1.0 {
                return false;
            }
            let ext_x = (max_x - min_x) * w as f32;
            let ext_y = (max_y - min_y) * h as f32;
            if ext_x * ext_y < 1.0 {
                return false;
            }
            true
        })
        .collect();

    let mut image = vec![[0.0f32; 4]; (w * h) as usize];

    for py in 0..h {
        for px in 0..w {
            let ray_dir = pixel_ray_dir(inv_vp, cam_pos, px as f32, py as f32, w as f32, h as f32);
            let mut accum_r = 0.0f32;
            let mut accum_g = 0.0f32;
            let mut accum_b = 0.0f32;
            let mut accum_a = 0.0f32;

            // Back-to-front compositing with premultiplied alpha
            for &tet_id in &sorted {
                let ti = tet_id as usize;
                if !visible[ti] {
                    continue;
                }
                let td = &tet_data[ti];

                // Reconstruct base_color at cam (matching vertex shader)
                let offset = td.grad.dot(cam_pos - td.verts[0]);
                let base_color = td.color + Vec3::splat(offset);

                let [cr, cg, cb, ca] = render_tet_pixel(
                    cam_pos,
                    ray_dir, // already normalized, d=1
                    &td.verts,
                    td.density,
                    base_color,
                    td.grad,
                );

                // Premultiplied alpha blend: dst = src + dst * (1 - src_alpha)
                accum_r = cr + accum_r * (1.0 - ca);
                accum_g = cg + accum_g * (1.0 - ca);
                accum_b = cb + accum_b * (1.0 - ca);
                accum_a = ca + accum_a * (1.0 - ca);
            }

            image[(py * w + px) as usize] = [accum_r, accum_g, accum_b, accum_a];
        }
    }

    image
}

// ---------------------------------------------------------------------------
// GPU pipeline runner (headless)
// ---------------------------------------------------------------------------

/// Run the full GPU forward pipeline and read back the color image.
/// Returns None if no GPU adapter is available (CI without GPU).
pub fn gpu_render_scene(
    scene: &SceneData,
    cam_pos: Vec3,
    vp: Mat4,
    inv_vp: Mat4,
    w: u32,
    h: u32,
) -> Option<Vec<[f32; 4]>> {
    pollster::block_on(gpu_render_scene_async(scene, cam_pos, vp, inv_vp, w, h))
}

async fn gpu_render_scene_async(
    scene: &SceneData,
    cam_pos: Vec3,
    vp: Mat4,
    inv_vp: Mat4,
    w: u32,
    h: u32,
) -> Option<Vec<[f32; 4]>> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 11,
                    ..wgpu::Limits::downlevel_defaults()
                },
                ..Default::default()
            },
            None,
        )
        .await
        .ok()?;

    let (buffers, pipelines, targets, compute_bg, render_bg, sort_state) =
        rmesh_render::setup_forward(&device, &queue, scene, w, h);

    // Write uniforms
    let uniforms = rmesh_render::make_uniforms(
        vp,
        inv_vp,
        cam_pos,
        w as f32,
        h as f32,
        scene.tet_count,
        scene.sh_degree,
        0,
    );
    queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

    // Record and submit
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_forward"),
    });
    rmesh_render::record_forward_pass(
        &mut encoder,
        &pipelines,
        &buffers,
        &targets,
        &compute_bg,
        &render_bg,
        &sort_state,
        scene.tet_count,
        &queue,
    );

    // Copy color texture to readback buffer
    let bytes_per_pixel = 8u32; // Rgba16Float = 4 × f16 = 8 bytes
    let bytes_per_row = w * bytes_per_pixel;
    // wgpu requires bytes_per_row aligned to 256
    let aligned_bytes_per_row = (bytes_per_row + 255) & !255;

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (aligned_bytes_per_row * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &targets.color_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(aligned_bytes_per_row),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(std::iter::once(encoder.finish()));

    // Map and read back
    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    device.poll(wgpu::Maintain::Wait);
    receiver.recv().unwrap().ok()?;

    let data = slice.get_mapped_range();
    let mut image = vec![[0.0f32; 4]; (w * h) as usize];

    for row in 0..h {
        let row_start = (row * aligned_bytes_per_row) as usize;
        for col in 0..w {
            let pixel_start = row_start + (col * bytes_per_pixel) as usize;
            let r = half::f16::from_le_bytes([data[pixel_start], data[pixel_start + 1]]).to_f32();
            let g =
                half::f16::from_le_bytes([data[pixel_start + 2], data[pixel_start + 3]]).to_f32();
            let b =
                half::f16::from_le_bytes([data[pixel_start + 4], data[pixel_start + 5]]).to_f32();
            let a =
                half::f16::from_le_bytes([data[pixel_start + 6], data[pixel_start + 7]]).to_f32();
            image[(row * w + col) as usize] = [r, g, b, a];
        }
    }

    drop(data);
    readback.unmap();

    Some(image)
}

// ---------------------------------------------------------------------------
// Test scene builders
// ---------------------------------------------------------------------------

/// Build a SceneData from raw vertices, indices, and per-tet parameters.
pub fn build_test_scene(
    vertices: Vec<f32>,
    indices: Vec<u32>,
    sh_coeffs: Vec<f32>,
    densities: Vec<f32>,
    color_grads: Vec<f32>,
    sh_degree: u32,
) -> SceneData {
    let vertex_count = vertices.len() as u32 / 3;
    let tet_count = indices.len() as u32 / 4;
    let circumdata = compute_circumspheres_from_verts(&vertices, &indices, tet_count as usize);

    SceneData {
        vertices,
        indices,
        sh_coeffs,
        densities,
        color_grads,
        circumdata,
        start_pose: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        vertex_count,
        tet_count,
        sh_degree,
    }
}

/// Compute circumspheres (duplicated from rmesh-data to avoid pub visibility issues).
fn compute_circumspheres_from_verts(
    vertices: &[f32],
    indices: &[u32],
    tet_count: usize,
) -> Vec<f32> {
    let mut circumdata = vec![0.0f32; tet_count * 4];
    for i in 0..tet_count {
        let i0 = indices[i * 4] as usize;
        let i1 = indices[i * 4 + 1] as usize;
        let i2 = indices[i * 4 + 2] as usize;
        let i3 = indices[i * 4 + 3] as usize;
        let v0 = Vec3::new(vertices[i0 * 3], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]);
        let v1 = Vec3::new(vertices[i1 * 3], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]);
        let v2 = Vec3::new(vertices[i2 * 3], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]);
        let v3 = Vec3::new(vertices[i3 * 3], vertices[i3 * 3 + 1], vertices[i3 * 3 + 2]);
        let a = v1 - v0;
        let b = v2 - v0;
        let c = v3 - v0;
        let (aa, bb, cc) = (a.dot(a), b.dot(b), c.dot(c));
        let cross_bc = b.cross(c);
        let cross_ca = c.cross(a);
        let cross_ab = a.cross(b);
        let mut denom = 2.0 * a.dot(cross_bc);
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

/// Generate a random tetrahedron centered roughly at origin.
/// Ensures positive orientation (det > 0) so face winding produces inward normals.
pub fn random_tet_vertices<R: Rng>(rng: &mut R, radius: f32) -> ([f32; 12], [u32; 4]) {
    let mut verts = [0.0f32; 12];
    for i in 0..4 {
        verts[i * 3] = (rng.random::<f32>() - 0.5) * 2.0 * radius;
        verts[i * 3 + 1] = (rng.random::<f32>() - 0.5) * 2.0 * radius;
        verts[i * 3 + 2] = (rng.random::<f32>() - 0.5) * 2.0 * radius;
    }
    // Ensure positive orientation: det(v1-v0, v2-v0, v3-v0) > 0
    let v0 = Vec3::new(verts[0], verts[1], verts[2]);
    let v1 = Vec3::new(verts[3], verts[4], verts[5]);
    let v2 = Vec3::new(verts[6], verts[7], verts[8]);
    let v3 = Vec3::new(verts[9], verts[10], verts[11]);
    let det = (v1 - v0).dot((v2 - v0).cross(v3 - v0));
    if det < 0.0 {
        // Swap vertices 2 and 3 to flip orientation
        return (verts, [0, 1, 3, 2]);
    }
    (verts, [0, 1, 2, 3])
}

/// Generate random SH coefficients for degree 0 (just DC term, 1 coeff per channel = 3 total).
pub fn random_sh_degree0<R: Rng>(rng: &mut R, tet_count: usize) -> Vec<f32> {
    (0..tet_count * 3)
        .map(|_| rng.random::<f32>() * 2.0 - 1.0)
        .collect()
}

/// Build a single-tet test scene with random parameters.
pub fn random_single_tet_scene<R: Rng>(rng: &mut R, radius: f32) -> SceneData {
    let (verts, indices) = random_tet_vertices(rng, radius);
    let sh_coeffs = random_sh_degree0(rng, 1);
    let density = vec![rng.random::<f32>() * 5.0 + 0.5];
    let color_grads = vec![
        (rng.random::<f32>() - 0.5) * 0.2,
        (rng.random::<f32>() - 0.5) * 0.2,
        (rng.random::<f32>() - 0.5) * 0.2,
    ];

    build_test_scene(verts.to_vec(), indices.to_vec(), sh_coeffs, density, color_grads, 0)
}

/// Build a simple perspective projection matrix.
pub fn perspective_matrix(fov_y_rad: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    // Reversed-Z infinite far or standard perspective
    // Using standard wgpu perspective (clip z ∈ [0,1])
    let f = 1.0 / (fov_y_rad / 2.0).tan();
    Mat4::from_cols(
        Vec4::new(f / aspect, 0.0, 0.0, 0.0),
        Vec4::new(0.0, f, 0.0, 0.0),
        Vec4::new(0.0, 0.0, far / (far - near), 1.0),
        Vec4::new(0.0, 0.0, -(far * near) / (far - near), 0.0),
    )
}

/// Build a look-at view matrix (Z-up convention).
pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
    let f = (target - eye).normalize();
    let r = f.cross(up).normalize();
    let u = r.cross(f);

    Mat4::from_cols(
        Vec4::new(r.x, u.x, f.x, 0.0),
        Vec4::new(r.y, u.y, f.y, 0.0),
        Vec4::new(r.z, u.z, f.z, 0.0),
        Vec4::new(-r.dot(eye), -u.dot(eye), -f.dot(eye), 1.0),
    )
}

/// Comparison helper: check if two images match within tolerance.
/// Returns (max_abs_diff, mean_abs_diff, num_pixels_compared).
pub fn compare_images(
    cpu: &[[f32; 4]],
    gpu: &[[f32; 4]],
) -> (f32, f32, usize) {
    assert_eq!(cpu.len(), gpu.len());
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    let mut count = 0usize;

    for (c, g) in cpu.iter().zip(gpu.iter()) {
        for ch in 0..4 {
            let diff = (c[ch] - g[ch]).abs();
            max_diff = max_diff.max(diff);
            sum_diff += diff;
            count += 1;
        }
    }

    let mean_diff = if count > 0 {
        sum_diff / count as f32
    } else {
        0.0
    };

    (max_diff, mean_diff, count)
}
