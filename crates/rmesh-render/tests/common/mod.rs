//! CPU reference renderer for testing GPU pipeline correctness.
//!
//! Implements the same math as the WGSL shaders (forward_compute, forward_vertex,
//! forward_fragment) in pure Rust with glam. No GPU required.

#![allow(dead_code, unused_imports)]

use glam::{Mat4, Vec3, Vec4};
pub use rand::Rng;
pub use rmesh_data::SceneData;

// Re-export shared camera utilities from rmesh-util
pub use rmesh_util::camera::{perspective_matrix, look_at, TET_FACES};

// Re-export test utilities from rmesh-util
pub use rmesh_util::test_util::{build_test_scene, random_single_tet_scene};

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

fn phi(x: f32) -> f32 {
    if x.abs() < 1e-6 {
        1.0 - x * 0.5
    } else {
        (1.0 - (-x).exp()) / x
    }
}

// ---------------------------------------------------------------------------
// CPU compute pass: color eval (raw base colors, no activation)
// ---------------------------------------------------------------------------

/// Evaluate per-tet color (matches forward_compute.wgsl).
/// Base color is 0.5 (constant bias from the shader).
/// No activation — raw base colors are passed through, matching the GPU path.
/// Per-pixel ReLU clamping (max(0)) is applied later in render_tet_pixel.
fn compute_tet_color(_scene: &SceneData, _tet_id: usize, _cam_pos: Vec3) -> Vec3 {
    let result_color = Vec3::splat(0.5);

    // In the GPU path, base_colors_buf is passed through directly.
    // The test uses 0.5 as the base color.
    result_color
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
        let v_opp = verts[face[3]];
        let mut n = (vc - va).cross(vb - va);
        // Flip normal to point inward (toward opposite vertex)
        if n.dot(v_opp - va) < 0.0 {
            n = -n;
        }
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
        .await
        .ok()?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::SUBGROUP,
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 16,
                    max_storage_buffer_binding_size: 1 << 30,
                    max_buffer_size: 1 << 30,
                    ..wgpu::Limits::default()
                },
                ..Default::default()
            },
        )
        .await
        .ok()?;

    let zero_base_colors = vec![0.5f32; scene.tet_count as usize * 3];
    let (buffers, _material, pipelines, targets, compute_bg, render_bg) =
        rmesh_render::setup_forward(&device, &queue, scene, &zero_base_colors, &scene.color_grads, w, h);

    // Write uniforms
    let uniforms = rmesh_render::make_uniforms(
        vp,
        inv_vp,
        cam_pos,
        w as f32,
        h as f32,
        scene.tet_count,
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
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
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
// GPU ray trace pipeline runner (headless)
// ---------------------------------------------------------------------------

/// Run the GPU ray trace pipeline and read back the color image.
/// Returns None if no GPU adapter is available.
pub fn gpu_raytrace_scene(
    scene: &SceneData,
    cam_pos: Vec3,
    vp: Mat4,
    inv_vp: Mat4,
    w: u32,
    h: u32,
) -> Option<Vec<[f32; 4]>> {
    pollster::block_on(gpu_raytrace_scene_async(scene, cam_pos, vp, inv_vp, w, h))
}

async fn gpu_raytrace_scene_async(
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
        .await
        .ok()?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 16,
                    max_storage_buffer_binding_size: 1 << 30,
                    max_buffer_size: 1 << 30,
                    ..wgpu::Limits::default()
                },
                ..Default::default()
            },
        )
        .await
        .ok()?;

    // Upload scene
    let color_format = wgpu::TextureFormat::Rgba16Float;
    let aux_format = wgpu::TextureFormat::Rgba32Float;
    let buffers = rmesh_render::SceneBuffers::upload(&device, &queue, scene);
    let zero_base_colors = vec![0.5f32; scene.tet_count as usize * 3];
    let material = rmesh_render::MaterialBuffers::upload(&device, &zero_base_colors, &scene.color_grads, scene.tet_count);
    let pipelines = rmesh_render::ForwardPipelines::new(&device, color_format, aux_format);
    let compute_bg = rmesh_render::create_compute_bind_group(&device, &pipelines, &buffers, &material);

    // Build ray trace data
    let neighbors = rmesh_render::compute_tet_neighbors(&scene.indices, scene.tet_count as usize);
    let bvh = rmesh_render::build_boundary_bvh(
        &scene.vertices, &scene.indices, &neighbors, scene.tet_count as usize,
    );
    let rt_pipeline = rmesh_render::RayTracePipeline::new(&device, w, h);
    let rt_buffers = rmesh_render::RayTraceBuffers::new(&device, &neighbors, &bvh);

    // Determine start tet
    let start_tet = rmesh_render::find_containing_tet(
        &scene.vertices, &scene.indices, scene.tet_count as usize, cam_pos,
    ).map(|t| t as i32).unwrap_or(-1);
    queue.write_buffer(&rt_buffers.start_tet, 0, bytemuck::cast_slice(&[start_tet]));

    let rt_bg = rmesh_render::create_raytrace_bind_group(&device, &rt_pipeline, &buffers, &material, &rt_buffers);

    // Write uniforms
    let uniforms = rmesh_render::make_uniforms(
        vp, inv_vp, cam_pos, w as f32, h as f32, scene.tet_count, 0,
    );
    queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

    // Record: forward compute + raytrace
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_raytrace"),
    });

    // Clear output
    encoder.clear_buffer(&rt_pipeline.rendered_image, 0, None);

    // Forward compute (color eval → colors_buf)
    rmesh_render::record_forward_compute(
        &mut encoder, &pipelines, &buffers, &compute_bg, scene.tet_count, &queue,
    );

    // Ray trace
    rmesh_render::record_raytrace(&mut encoder, &rt_pipeline, &rt_bg, w, h);

    // Readback
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (w as u64) * (h as u64) * 4 * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(
        &rt_pipeline.rendered_image, 0, &readback, 0, readback.size(),
    );
    queue.submit(std::iter::once(encoder.finish()));

    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    receiver.recv().unwrap().ok()?;

    let data = slice.get_mapped_range();
    let floats: &[f32] = bytemuck::cast_slice(&data);
    let mut image = vec![[0.0f32; 4]; (w * h) as usize];
    for i in 0..(w * h) as usize {
        image[i] = [floats[i * 4], floats[i * 4 + 1], floats[i * 4 + 2], floats[i * 4 + 3]];
    }
    drop(data);
    readback.unmap();

    Some(image)
}

// ---------------------------------------------------------------------------
// GPU tiled forward pipeline runner (headless)
// ---------------------------------------------------------------------------

/// Run the GPU tiled forward pipeline and read back the color image.
/// Returns None if no GPU adapter is available.
///
/// This uses the scan-based tile pipeline (same as the Python/training path):
///   forward_compute → scan tile pipeline → radix_sort → tile_ranges → forward_tiled
pub fn gpu_tiled_render_scene(
    scene: &SceneData,
    cam_pos: Vec3,
    vp: Mat4,
    inv_vp: Mat4,
    w: u32,
    h: u32,
) -> Option<Vec<[f32; 4]>> {
    pollster::block_on(gpu_tiled_render_scene_async(scene, cam_pos, vp, inv_vp, w, h))
}

async fn gpu_tiled_render_scene_async(
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
        .await
        .ok()?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::SUBGROUP,
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 16,
                    max_storage_buffer_binding_size: 1 << 30,
                    max_buffer_size: 1 << 30,
                    ..wgpu::Limits::default()
                },
                ..Default::default()
            },
        )
        .await
        .ok()?;

    // Must be 16 to match forward_compute.wgsl (hardcoded tile_size=16.0)
    // and forward_tiled_compute.wgsl (hardcoded 16×16 pixel tiles).
    let tile_size = 16u32;

    // Forward compute setup
    let zero_base_colors = vec![0.5f32; scene.tet_count as usize * 3];
    let (buffers, material, fwd_pipelines, _targets, compute_bg, _render_bg) =
        rmesh_render::setup_forward(&device, &queue, scene, &zero_base_colors, &scene.color_grads, w, h);

    let uniforms = rmesh_render::make_uniforms(
        vp, inv_vp, cam_pos, w as f32, h as f32, scene.tet_count, 0u32,
    );
    queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

    // Run forward compute
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    rmesh_render::record_forward_compute(
        &mut encoder, &fwd_pipelines, &buffers, &compute_bg, scene.tet_count, &queue,
    );
    queue.submit(std::iter::once(encoder.finish()));

    // Tile pipeline setup
    let tile_pipelines = rmesh_tile::TilePipelines::new(&device);
    let radix_pipelines = rmesh_sort::RadixSortPipelines::new(&device);
    let tile_buffers = rmesh_tile::TileBuffers::new(&device, scene.tet_count, w, h, tile_size);
    let radix_state = rmesh_sort::RadixSortState::new(&device, tile_buffers.max_pairs_pow2, 32);
    radix_state.upload_configs(&queue);

    let scan_pipelines = rmesh_tile::ScanPipelines::new(&device);
    let scan_buffers = rmesh_tile::ScanBuffers::new(&device, scene.tet_count);

    let tile_uni = rmesh_util::shared::TileUniforms {
        screen_width: w,
        screen_height: h,
        tile_size,
        tiles_x: tile_buffers.tiles_x,
        tiles_y: tile_buffers.tiles_y,
        num_tiles: tile_buffers.num_tiles,
        visible_tet_count: 0,
        _pad: [0; 5],
    };
    queue.write_buffer(&tile_buffers.tile_uniforms, 0, bytemuck::bytes_of(&tile_uni));

    // Create bind groups
    let prepare_dispatch_bg = rmesh_tile::create_prepare_dispatch_bind_group(
        &device, &scan_pipelines, &buffers.indirect_args, &scan_buffers,
    );
    let rts_bg = rmesh_tile::create_rts_bind_group(
        &device, &scan_pipelines, &buffers.tiles_touched, &scan_buffers,
    );
    let tile_fill_bg = rmesh_tile::create_tile_fill_bind_group(&device, &tile_pipelines, &tile_buffers);
    let tile_gen_scan_bg = rmesh_tile::create_tile_gen_scan_bind_group(
        &device, &scan_pipelines, &tile_buffers,
        &buffers.uniforms, &buffers.vertices, &buffers.indices,
        &buffers.compact_tet_ids, &buffers.circumdata, &buffers.tiles_touched,
        &scan_buffers, &radix_state.num_keys_buf,
    );

    let tile_ranges_bg_a = rmesh_tile::create_tile_ranges_bind_group_with_keys(
        &device, &tile_pipelines, &tile_buffers.tile_sort_keys,
        &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms, &radix_state.num_keys_buf,
    );
    let tile_ranges_bg_b = rmesh_tile::create_tile_ranges_bind_group_with_keys(
        &device, &tile_pipelines, &radix_state.keys_b,
        &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms, &radix_state.num_keys_buf,
    );

    // Forward tiled pipeline
    let fwd_tiled = rmesh_render::ForwardTiledPipeline::new(&device, w, h);
    let fwd_tiled_bg_a = rmesh_render::create_forward_tiled_bind_group(
        &device, &fwd_tiled, &buffers.uniforms,
        &buffers.vertices, &buffers.indices, &material.colors,
        &buffers.densities, &material.color_grads,
        &tile_buffers.tile_sort_values, &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms,
    );
    let fwd_tiled_bg_b = rmesh_render::create_forward_tiled_bind_group(
        &device, &fwd_tiled, &buffers.uniforms,
        &buffers.vertices, &buffers.indices, &material.colors,
        &buffers.densities, &material.color_grads,
        &radix_state.values_b, &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms,
    );

    // Dispatch tiled forward pipeline
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.clear_buffer(&fwd_tiled.rendered_image, 0, None);
    encoder.clear_buffer(&tile_buffers.tile_ranges, 0, None);

    // 1. Scan-based tile pipeline
    rmesh_tile::record_scan_tile_pipeline(
        &mut encoder, &scan_pipelines, &tile_pipelines,
        &prepare_dispatch_bg, &rts_bg,
        &tile_fill_bg, &tile_gen_scan_bg, &scan_buffers, &tile_buffers,
    );

    // 2. Radix sort
    let result_in_b = rmesh_sort::record_radix_sort(
        &mut encoder, &device, &radix_pipelines, &radix_state,
        &tile_buffers.tile_sort_keys, &tile_buffers.tile_sort_values,
    );

    // 3. Tile ranges
    {
        let ranges_bg = if result_in_b { &tile_ranges_bg_b } else { &tile_ranges_bg_a };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tile_ranges"), timestamp_writes: None,
        });
        pass.set_pipeline(&tile_pipelines.tile_ranges_pipeline);
        pass.set_bind_group(0, ranges_bg, &[]);
        let wgs = (tile_buffers.max_pairs_pow2 + 255) / 256;
        pass.dispatch_workgroups(wgs.min(65535), ((wgs + 65534) / 65535).max(1), 1);
    }

    // 4. Forward tiled
    {
        let fwd_bg = if result_in_b { &fwd_tiled_bg_b } else { &fwd_tiled_bg_a };
        rmesh_render::record_forward_tiled(&mut encoder, &fwd_tiled, fwd_bg, tile_buffers.num_tiles);
    }

    queue.submit(std::iter::once(encoder.finish()));

    // Read back rendered image
    let pixel_count = (w * h) as usize;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (pixel_count as u64) * 4 * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(&fwd_tiled.rendered_image, 0, &readback, 0, readback.size());
    queue.submit(std::iter::once(encoder.finish()));

    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    receiver.recv().unwrap().ok()?;

    let data = slice.get_mapped_range();
    let floats: &[f32] = bytemuck::cast_slice(&data);
    let mut image = vec![[0.0f32; 4]; pixel_count];
    for i in 0..pixel_count {
        image[i] = [floats[i * 4], floats[i * 4 + 1], floats[i * 4 + 2], floats[i * 4 + 3]];
    }
    drop(data);
    readback.unmap();

    Some(image)
}

// ---------------------------------------------------------------------------
// Test scene builders
// ---------------------------------------------------------------------------

/// Build a deterministic single-tet scene with known geometry.
/// Regular tet with vertices at (±0.5, ±0.5, ±0.5), density=3.0, zero color_grads.
/// This produces analytically predictable output.
pub fn known_single_tet_scene() -> SceneData {
    let vertices = vec![
        0.5, 0.5, 0.5,    // v0
        -0.5, -0.5, 0.5,  // v1
        -0.5, 0.5, -0.5,  // v2
        0.5, -0.5, -0.5,  // v3
    ];
    let indices = vec![0u32, 1, 2, 3];
    let densities = vec![3.0f32];
    let color_grads = vec![0.0f32; 3];
    build_test_scene(vertices, indices, densities, color_grads)
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
