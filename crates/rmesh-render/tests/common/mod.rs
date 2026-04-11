//! CPU reference renderer for testing GPU pipeline correctness.
//!
//! Implements the same math as the WGSL shaders (project_compute, forward_vertex,
//! forward_fragment) in pure Rust with glam. No GPU required.

#![allow(dead_code, unused_imports)]

use glam::{Mat3, Mat4, Vec3, Vec4};
pub use rand::Rng;
pub use rmesh_data::SceneData;

// Re-export shared camera utilities from rmesh-util
pub use rmesh_util::camera::{perspective_matrix, look_at, pixel_ray_intrinsics, TET_FACES};

// Re-export test utilities from rmesh-util
pub use rmesh_util::test_util::{build_test_scene, random_single_tet_scene};

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

fn phi(x: f32) -> f32 {
    if x.abs() < 0.02 {
        1.0 + x * (-0.5 + x * (1.0 / 6.0 + x * (-1.0 / 24.0)))
    } else {
        (1.0 - (-x).exp()) / x
    }
}

// ---------------------------------------------------------------------------
// CPU compute pass: color eval (raw base colors, no activation)
// ---------------------------------------------------------------------------

/// Evaluate per-tet color (matches project_compute.wgsl).
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

    // Classify entering/exiting (matching rasterize_compute.wgsl logic)
    let mut t_min = f32::NEG_INFINITY;
    let mut t_max = f32::INFINITY;
    let mut valid = true;

    for i in 0..4 {
        let den = plane_denom[i];
        let num = numerators[i];
        // Match GPU: abs(den) < 1e-20 → parallel face
        if den.abs() < 1e-20 {
            // Ray parallel to face; if ray is outside this face, no intersection
            if num > 0.0 {
                valid = false;
            }
            continue;
        }
        let t = num / den;
        if den > 0.0 {
            // Entering
            t_min = t_min.max(t);
        } else {
            // Exiting
            t_max = t_max.min(t);
        }
    }

    if !valid {
        return [0.0; 4];
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
// Sorting (CPU back-to-front, matches project_compute.wgsl depth key)
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

/// Compute world-space ray direction for pixel (px, py) matching rasterize_compute.wgsl.
///
/// Uses camera intrinsics and c2w rotation (pinhole convention: y-down, z-forward).
/// Matches the GPU shader's intrinsics-based ray construction.
pub fn pixel_ray_dir(c2w: Mat3, intrinsics: [f32; 4], cam_pos: Vec3, px: f32, py: f32) -> Vec3 {
    let (_, dir) = pixel_ray_intrinsics(c2w, intrinsics, cam_pos, px, py);
    dir
}

/// Compute c2w rotation and pinhole intrinsics for the test camera.
///
/// Uses the same LH look_at convention (up = Vec3::Z) as the test setup_camera functions.
/// c2w maps from pinhole camera space (x=right, y=down, z=forward) to world space.
pub fn test_camera_c2w_intrinsics(
    eye: Vec3, target: Vec3, fov_y: f32, w: f32, h: f32,
) -> (Mat3, [f32; 4]) {
    let f = (target - eye).normalize();
    let r = f.cross(Vec3::Z).normalize();
    let u = r.cross(f);
    // Pinhole convention: x=right, y=DOWN, z=forward
    let c2w = Mat3::from_cols(r, -u, f);

    let f_val = 1.0 / (fov_y / 2.0).tan();
    let fx = f_val * h / 2.0;
    let fy = f_val * h / 2.0;
    (c2w, [fx, fy, w / 2.0, h / 2.0])
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
    c2w: Mat3,
    intrinsics: [f32; 4],
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
            let ray_dir = pixel_ray_dir(c2w, intrinsics, cam_pos, px as f32, py as f32);
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
    c2w: Mat3,
    intrinsics: [f32; 4],
    w: u32,
    h: u32,
) -> Option<Vec<[f32; 4]>> {
    pollster::block_on(gpu_render_scene_async(scene, cam_pos, vp, c2w, intrinsics, w, h))
}

async fn gpu_render_scene_async(
    scene: &SceneData,
    cam_pos: Vec3,
    vp: Mat4,
    c2w: Mat3,
    intrinsics: [f32; 4],
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
                required_features: wgpu::Features::SUBGROUP | wgpu::Features::SHADER_FLOAT32_ATOMIC,
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 20,
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
    let (buffers, material, pipelines, targets, compute_bg, render_bg) =
        rmesh_render::setup_forward(&device, &queue, scene, &zero_base_colors, &scene.color_grads, w, h);

    // Sort infrastructure for sorted HW raster forward pass (32-bit keys)
    let n_pow2 = scene.tet_count.next_power_of_two();
    let sort_pipelines = rmesh_sort::RadixSortPipelines::new(&device, 1, rmesh_sort::SortBackend::Drs);
    let sort_state = rmesh_sort::RadixSortState::new(&device, n_pow2, 32, 1, rmesh_sort::SortBackend::Drs);
    sort_state.upload_configs(&queue);

    // B-variant render bind group (uses sort_state.values_b)
    let render_bg_b = rmesh_render::create_render_bind_group_with_sort_values(
        &device, &pipelines, &buffers, &material, sort_state.values_b(),
    );

    // Write uniforms
    let uniforms = rmesh_render::make_uniforms(
        vp,
        c2w,
        intrinsics,
        cam_pos,
        w as f32,
        h as f32,
        scene.tet_count,
        0,
        16,
        0.0,
        0,
        0.01,
        100.0,
    );
    queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

    // Create dummy depth texture (all-pass) for forward pass depth attachment
    let dummy_depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test_dummy_depth"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let dummy_depth_view = dummy_depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

    // Record and submit
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_forward"),
    });
    // Clear color and depth before forward pass (which uses LoadOp::Load)
    {
        let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("test_clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &targets.color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &dummy_depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
    }
    rmesh_render::record_sorted_forward_pass(
        &mut encoder,
        &device,
        &pipelines,
        &sort_pipelines,
        &sort_state,
        &buffers,
        &targets,
        &compute_bg,
        &render_bg,
        &render_bg_b,
        scene.tet_count,
        &queue,
        &dummy_depth_view,
        None,
        false,
        None, None, None,
        None,
        true,
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
    c2w: Mat3,
    intrinsics: [f32; 4],
    w: u32,
    h: u32,
) -> Option<Vec<[f32; 4]>> {
    pollster::block_on(gpu_raytrace_scene_async(scene, cam_pos, vp, c2w, intrinsics, w, h))
}

async fn gpu_raytrace_scene_async(
    scene: &SceneData,
    cam_pos: Vec3,
    vp: Mat4,
    c2w: Mat3,
    intrinsics: [f32; 4],
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
                    max_storage_buffers_per_shader_stage: 20,
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
    let buffers = rmesh_render::SceneBuffers::upload(&device, &queue, scene);
    let zero_base_colors = vec![0.5f32; scene.tet_count as usize * 3];
    let material = rmesh_render::MaterialBuffers::upload(&device, &zero_base_colors, &scene.color_grads, scene.tet_count);
    let pipelines = rmesh_render::ForwardPipelines::new(&device, color_format);
    let dummy_sh = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dummy_sh_coeffs"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let compute_bg = rmesh_render::create_compute_bind_group(&device, &pipelines, &buffers, &material, &dummy_sh);

    // Build ray trace data
    let neighbors = rmesh_render::compute_tet_neighbors(&scene.indices, scene.tet_count as usize);
    let bvh = rmesh_render::build_boundary_bvh(
        &scene.vertices, &scene.indices, &neighbors, scene.tet_count as usize,
    );
    let rt_pipeline = rmesh_render::RayTracePipeline::new(&device, w, h, 0);
    let rt_buffers = rmesh_render::RayTraceBuffers::new(&device, &neighbors, &bvh);

    // Determine start tet
    let start_tet = rmesh_render::find_containing_tet(
        &scene.vertices, &scene.indices, scene.tet_count as usize, cam_pos,
    ).map(|t| t as i32).unwrap_or(-1);
    queue.write_buffer(&rt_buffers.start_tet, 0, bytemuck::cast_slice(&[start_tet]));

    let rt_bg = rmesh_render::create_raytrace_bind_group(&device, &rt_pipeline, &buffers, &material, &rt_buffers);

    // Write uniforms
    let uniforms = rmesh_render::make_uniforms(
        vp, c2w, intrinsics, cam_pos, w as f32, h as f32, scene.tet_count, 0, 12, 0.0, 0, 0.01, 100.0,
    );
    queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

    // Record: forward compute + raytrace
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_raytrace"),
    });

    // Clear output
    encoder.clear_buffer(&rt_pipeline.rendered_image, 0, None);

    // Forward compute (color eval → colors_buf)
    rmesh_render::record_project_compute(
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
///   project_compute → scan tile pipeline → radix_sort → tile_ranges → rasterize_compute
pub fn gpu_tiled_render_scene(
    scene: &SceneData,
    cam_pos: Vec3,
    vp: Mat4,
    c2w: Mat3,
    intrinsics: [f32; 4],
    w: u32,
    h: u32,
) -> Option<Vec<[f32; 4]>> {
    pollster::block_on(gpu_tiled_render_scene_async(scene, cam_pos, vp, c2w, intrinsics, w, h))
}

async fn gpu_tiled_render_scene_async(
    scene: &SceneData,
    cam_pos: Vec3,
    vp: Mat4,
    c2w: Mat3,
    intrinsics: [f32; 4],
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
                required_features: wgpu::Features::SUBGROUP | wgpu::Features::SHADER_FLOAT32_ATOMIC,
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 20,
                    max_storage_buffer_binding_size: 1 << 30,
                    max_buffer_size: 1 << 30,
                    ..wgpu::Limits::default()
                },
                ..Default::default()
            },
        )
        .await
        .ok()?;

    let tile_size = 12u32;

    // Forward compute setup
    let zero_base_colors = vec![0.5f32; scene.tet_count as usize * 3];
    let (buffers, material, fwd_pipelines, _targets, compute_bg, _render_bg) =
        rmesh_render::setup_forward(&device, &queue, scene, &zero_base_colors, &scene.color_grads, w, h);

    let uniforms = rmesh_render::make_uniforms(
        vp, c2w, intrinsics, cam_pos, w as f32, h as f32, scene.tet_count, 0u32, 12, 0.0, 0, 0.01, 100.0,
    );
    queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

    // Run forward compute
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    rmesh_render::record_project_compute(
        &mut encoder, &fwd_pipelines, &buffers, &compute_bg, scene.tet_count, &queue,
    );
    queue.submit(std::iter::once(encoder.finish()));

    // Tile pipeline setup
    let tile_pipelines = rmesh_tile::TilePipelines::new(&device);
    let radix_pipelines = rmesh_sort::RadixSortPipelines::new(&device, 2, rmesh_sort::SortBackend::Drs);
    let tile_buffers = rmesh_tile::TileBuffers::new(&device, scene.tet_count, w, h, tile_size);
    let sorting_bits = rmesh_sort::sorting_bits_for_tiles(tile_buffers.num_tiles, rmesh_sort::SortBackend::Drs);
    let radix_state = rmesh_sort::RadixSortState::new(&device, tile_buffers.max_pairs_pow2, sorting_bits, 2, rmesh_sort::SortBackend::Drs);
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
        &scan_buffers, radix_state.num_keys_buf(),
    );

    let tile_ranges_bg_a = rmesh_tile::create_tile_ranges_bind_group_with_keys(
        &device, &tile_pipelines, &tile_buffers.tile_sort_keys,
        &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms, radix_state.num_keys_buf(),
    );
    let tile_ranges_bg_b = rmesh_tile::create_tile_ranges_bind_group_with_keys(
        &device, &tile_pipelines, radix_state.keys_b(),
        &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms, radix_state.num_keys_buf(),
    );

    // Forward tiled pipeline
    let rasterize = rmesh_render::RasterizeComputePipeline::new(&device, w, h, 0);
    let rasterize_bg_a = rmesh_render::create_rasterize_bind_group(
        &device, &rasterize, &buffers.uniforms,
        &buffers.vertices, &buffers.indices, &material.colors,
        &buffers.densities, &material.color_grads,
        &tile_buffers.tile_sort_values, &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms,
    );
    let rasterize_bg_b = rmesh_render::create_rasterize_bind_group(
        &device, &rasterize, &buffers.uniforms,
        &buffers.vertices, &buffers.indices, &material.colors,
        &buffers.densities, &material.color_grads,
        radix_state.values_b(), &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms,
    );

    // Dispatch tiled forward pipeline
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.clear_buffer(&rasterize.rendered_image, 0, None);
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
        let fwd_bg = if result_in_b { &rasterize_bg_b } else { &rasterize_bg_a };
        rmesh_render::record_rasterize_compute(&mut encoder, &rasterize, fwd_bg, tile_buffers.num_tiles);
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
    encoder.copy_buffer_to_buffer(&rasterize.rendered_image, 0, &readback, 0, readback.size());
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
// GPU tiled forward pipeline with overdraw stats readback
// ---------------------------------------------------------------------------

/// Overdraw stats per pixel: [ray_miss, ghost, occluded, useful].
pub type OverdrawStats = [u32; 4];

/// Run the GPU tiled forward pipeline and read back both the color image
/// and the per-pixel overdraw debug stats.
/// Returns None if no GPU adapter is available.
pub fn gpu_tiled_render_with_stats(
    scene: &SceneData,
    cam_pos: Vec3,
    vp: Mat4,
    c2w: Mat3,
    intrinsics: [f32; 4],
    w: u32,
    h: u32,
) -> Option<(Vec<[f32; 4]>, Vec<OverdrawStats>)> {
    pollster::block_on(gpu_tiled_render_with_stats_async(scene, cam_pos, vp, c2w, intrinsics, w, h))
}

async fn gpu_tiled_render_with_stats_async(
    scene: &SceneData,
    cam_pos: Vec3,
    vp: Mat4,
    c2w: Mat3,
    intrinsics: [f32; 4],
    w: u32,
    h: u32,
) -> Option<(Vec<[f32; 4]>, Vec<OverdrawStats>)> {
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
                required_features: wgpu::Features::SUBGROUP | wgpu::Features::SHADER_FLOAT32_ATOMIC,
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 20,
                    max_storage_buffer_binding_size: 1 << 30,
                    max_buffer_size: 1 << 30,
                    ..wgpu::Limits::default()
                },
                ..Default::default()
            },
        )
        .await
        .ok()?;

    let tile_size = 12u32;

    // Forward compute setup
    let zero_base_colors = vec![0.5f32; scene.tet_count as usize * 3];
    let (buffers, material, fwd_pipelines, _targets, compute_bg, _render_bg) =
        rmesh_render::setup_forward(&device, &queue, scene, &zero_base_colors, &scene.color_grads, w, h);

    let uniforms = rmesh_render::make_uniforms(
        vp, c2w, intrinsics, cam_pos, w as f32, h as f32, scene.tet_count, 0u32, 12, 0.0, 0, 0.01, 100.0,
    );
    queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

    // Run forward compute
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    rmesh_render::record_project_compute(
        &mut encoder, &fwd_pipelines, &buffers, &compute_bg, scene.tet_count, &queue,
    );
    queue.submit(std::iter::once(encoder.finish()));

    // Tile pipeline setup
    let tile_pipelines = rmesh_tile::TilePipelines::new(&device);
    let radix_pipelines = rmesh_sort::RadixSortPipelines::new(&device, 2, rmesh_sort::SortBackend::Drs);
    let tile_buffers = rmesh_tile::TileBuffers::new(&device, scene.tet_count, w, h, tile_size);
    let sorting_bits = rmesh_sort::sorting_bits_for_tiles(tile_buffers.num_tiles, rmesh_sort::SortBackend::Drs);
    let radix_state = rmesh_sort::RadixSortState::new(&device, tile_buffers.max_pairs_pow2, sorting_bits, 2, rmesh_sort::SortBackend::Drs);
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
        &scan_buffers, radix_state.num_keys_buf(),
    );

    let tile_ranges_bg_a = rmesh_tile::create_tile_ranges_bind_group_with_keys(
        &device, &tile_pipelines, &tile_buffers.tile_sort_keys,
        &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms, radix_state.num_keys_buf(),
    );
    let tile_ranges_bg_b = rmesh_tile::create_tile_ranges_bind_group_with_keys(
        &device, &tile_pipelines, radix_state.keys_b(),
        &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms, radix_state.num_keys_buf(),
    );

    // Forward tiled pipeline
    let rasterize = rmesh_render::RasterizeComputePipeline::new(&device, w, h, 0);
    let rasterize_bg_a = rmesh_render::create_rasterize_bind_group(
        &device, &rasterize, &buffers.uniforms,
        &buffers.vertices, &buffers.indices, &material.colors,
        &buffers.densities, &material.color_grads,
        &tile_buffers.tile_sort_values, &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms,
    );
    let rasterize_bg_b = rmesh_render::create_rasterize_bind_group(
        &device, &rasterize, &buffers.uniforms,
        &buffers.vertices, &buffers.indices, &material.colors,
        &buffers.densities, &material.color_grads,
        radix_state.values_b(), &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms,
    );

    // Dispatch tiled forward pipeline
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.clear_buffer(&rasterize.rendered_image, 0, None);
    encoder.clear_buffer(&rasterize.debug_image, 0, None);
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
        let fwd_bg = if result_in_b { &rasterize_bg_b } else { &rasterize_bg_a };
        rmesh_render::record_rasterize_compute(&mut encoder, &rasterize, fwd_bg, tile_buffers.num_tiles);
    }

    queue.submit(std::iter::once(encoder.finish()));

    // Read back rendered image + debug stats
    let pixel_count = (w * h) as usize;
    let img_size = (pixel_count as u64) * 4 * 4;
    let readback_img = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback_img"),
        size: img_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let readback_dbg = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback_dbg"),
        size: img_size, // same size: 4 × u32 per pixel
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(&rasterize.rendered_image, 0, &readback_img, 0, img_size);
    encoder.copy_buffer_to_buffer(&rasterize.debug_image, 0, &readback_dbg, 0, img_size);
    queue.submit(std::iter::once(encoder.finish()));

    // Map image buffer
    let slice_img = readback_img.slice(..);
    let (tx1, rx1) = std::sync::mpsc::channel();
    slice_img.map_async(wgpu::MapMode::Read, move |r| { tx1.send(r).unwrap(); });
    let slice_dbg = readback_dbg.slice(..);
    let (tx2, rx2) = std::sync::mpsc::channel();
    slice_dbg.map_async(wgpu::MapMode::Read, move |r| { tx2.send(r).unwrap(); });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx1.recv().unwrap().ok()?;
    rx2.recv().unwrap().ok()?;

    let img_data = slice_img.get_mapped_range();
    let floats: &[f32] = bytemuck::cast_slice(&img_data);
    let mut image = vec![[0.0f32; 4]; pixel_count];
    for i in 0..pixel_count {
        image[i] = [floats[i * 4], floats[i * 4 + 1], floats[i * 4 + 2], floats[i * 4 + 3]];
    }
    drop(img_data);
    readback_img.unmap();

    let dbg_data = slice_dbg.get_mapped_range();
    let uints: &[u32] = bytemuck::cast_slice(&dbg_data);
    let mut stats = vec![[0u32; 4]; pixel_count];
    for i in 0..pixel_count {
        stats[i] = [uints[i * 4], uints[i * 4 + 1], uints[i * 4 + 2], uints[i * 4 + 3]];
    }
    drop(dbg_data);
    readback_dbg.unmap();

    Some((image, stats))
}

// ---------------------------------------------------------------------------
// GPU interval shading pipeline runner (headless, requires mesh shaders)
// ---------------------------------------------------------------------------

/// Run the GPU interval shading pipeline and read back the color image.
/// Returns None if no GPU adapter is available or mesh shaders not supported.
///
/// This uses the interval shading path (Tricard, HPG 2024):
///   project_compute → radix sort → indirect_convert (TETS_PER_GROUP=16) → interval mesh render
pub fn gpu_interval_render_scene(
    _scene: &SceneData,
    _cam_pos: Vec3,
    _vp: Mat4,
    _c2w: Mat3,
    _intrinsics: [f32; 4],
    _w: u32,
    _h: u32,
) -> Option<Vec<[f32; 4]>> {
    // Disabled: mesh shader interval path causes GPU hangs in cross_renderer tests.
    // Use gpu_interval_tiled_render_scene (compute-based) instead.
    None
}

async fn gpu_interval_render_scene_async(
    scene: &SceneData,
    cam_pos: Vec3,
    vp: Mat4,
    c2w: Mat3,
    intrinsics: [f32; 4],
    w: u32,
    h: u32,
) -> Option<Vec<[f32; 4]>> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL,
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok()?;

    // Check adapter supports mesh shaders
    let adapter_features = adapter.features();
    if !adapter_features.contains(wgpu::Features::EXPERIMENTAL_MESH_SHADER) {
        return None;
    }

    // Copy mesh shader limits from adapter (defaults are 0 = disabled)
    let supported_limits = adapter.limits();
    let mut limits = wgpu::Limits {
        max_storage_buffers_per_shader_stage: 20,
        max_storage_buffer_binding_size: 1 << 30,
        max_buffer_size: 1 << 30,
        ..wgpu::Limits::default()
    };
    limits.max_mesh_invocations_per_workgroup = supported_limits.max_mesh_invocations_per_workgroup;
    limits.max_mesh_invocations_per_dimension = supported_limits.max_mesh_invocations_per_dimension;
    limits.max_mesh_output_vertices = supported_limits.max_mesh_output_vertices;
    limits.max_mesh_output_primitives = supported_limits.max_mesh_output_primitives;
    limits.max_mesh_output_layers = supported_limits.max_mesh_output_layers;
    limits.max_mesh_multiview_view_count = supported_limits.max_mesh_multiview_view_count;
    limits.max_task_mesh_workgroup_total_count = supported_limits.max_task_mesh_workgroup_total_count;
    limits.max_task_mesh_workgroups_per_dimension = supported_limits.max_task_mesh_workgroups_per_dimension;

    // Mesh shaders are experimental — must opt in
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::SUBGROUP
                    | wgpu::Features::SHADER_FLOAT32_ATOMIC
                    | wgpu::Features::EXPERIMENTAL_MESH_SHADER,
                required_limits: limits,
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                ..Default::default()
            },
        )
        .await
        .ok()?;

    let color_format = wgpu::TextureFormat::Rgba16Float;
    let zero_base_colors = vec![0.5f32; scene.tet_count as usize * 3];
    let (buffers, material, pipelines, targets, compute_bg, _render_bg) =
        rmesh_render::setup_forward(&device, &queue, scene, &zero_base_colors, &scene.color_grads, w, h);

    // Interval shading pipelines
    let interval_pipelines = rmesh_render::IntervalPipelines::new(&device, color_format);

    // Sort infrastructure (32-bit keys)
    let n_pow2 = scene.tet_count.next_power_of_two();
    let sort_pipelines = rmesh_sort::RadixSortPipelines::new(&device, 1, rmesh_sort::SortBackend::Drs);
    let sort_state = rmesh_sort::RadixSortState::new(&device, n_pow2, 32, 1, rmesh_sort::SortBackend::Drs);
    sort_state.upload_configs(&queue);

    // Interval bind groups (A and B for sort buffer swapping)
    let interval_render_bg_a = rmesh_render::create_interval_render_bind_group(
        &device, &interval_pipelines, &buffers, &material,
    );
    let interval_render_bg_b = rmesh_render::create_interval_render_bind_group_with_sort_values(
        &device, &interval_pipelines, &buffers, &material, sort_state.values_b(),
    );
    let indirect_convert_bg = rmesh_render::create_interval_indirect_convert_bind_group(
        &device, &interval_pipelines, &buffers,
    );

    // Write uniforms
    let uniforms = rmesh_render::make_uniforms(
        vp, c2w, intrinsics, cam_pos,
        w as f32, h as f32,
        scene.tet_count, 0, 16, 0.0, 0,
        0.01, 100.0,
    );
    queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

    // Depth texture for the render pass
    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test_interval_depth"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

    // Record and submit
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_interval"),
    });

    // Clear color and depth
    {
        let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("test_interval_clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &targets.color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
    }

    rmesh_render::record_sorted_interval_forward_pass(
        &mut encoder,
        &device,
        &pipelines,
        &interval_pipelines,
        &sort_pipelines,
        &sort_state,
        &buffers,
        &targets,
        &compute_bg,
        &interval_render_bg_a,
        &interval_render_bg_b,
        &indirect_convert_bg,
        scene.tet_count,
        &queue,
        &depth_view,
        None,
        None,
    );

    // Copy color texture to readback buffer
    let bytes_per_pixel = 8u32; // Rgba16Float = 4 × f16 = 8 bytes
    let bytes_per_row = w * bytes_per_pixel;
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
            let g = half::f16::from_le_bytes([data[pixel_start + 2], data[pixel_start + 3]]).to_f32();
            let b = half::f16::from_le_bytes([data[pixel_start + 4], data[pixel_start + 5]]).to_f32();
            let a = half::f16::from_le_bytes([data[pixel_start + 6], data[pixel_start + 7]]).to_f32();
            image[(row * w + col) as usize] = [r, g, b, a];
        }
    }

    drop(data);
    readback.unmap();

    Some(image)
}

// ---------------------------------------------------------------------------
// GPU interval tiled pipeline runner (headless)
// ---------------------------------------------------------------------------

/// Run the GPU interval tiled forward pipeline and read back the color image.
/// Returns None if no GPU adapter is available.
///
/// This uses the tiled interval path:
///   project_compute → interval_generate → scan tile pipeline → radix_sort
///   → tile_ranges → interval_tiled_rasterize
pub fn gpu_interval_tiled_render_scene(
    scene: &SceneData,
    cam_pos: Vec3,
    vp: Mat4,
    c2w: Mat3,
    intrinsics: [f32; 4],
    w: u32,
    h: u32,
) -> Option<Vec<[f32; 4]>> {
    pollster::block_on(gpu_interval_tiled_render_scene_async(scene, cam_pos, vp, c2w, intrinsics, w, h))
}

async fn gpu_interval_tiled_render_scene_async(
    scene: &SceneData,
    cam_pos: Vec3,
    vp: Mat4,
    c2w: Mat3,
    intrinsics: [f32; 4],
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
                required_features: wgpu::Features::SUBGROUP | wgpu::Features::SHADER_FLOAT32_ATOMIC,
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 20,
                    max_storage_buffer_binding_size: 1 << 30,
                    max_buffer_size: 1 << 30,
                    ..wgpu::Limits::default()
                },
                ..Default::default()
            },
        )
        .await
        .ok()?;

    let tile_size = 12u32;

    // Forward compute setup
    let zero_base_colors = vec![0.5f32; scene.tet_count as usize * 3];
    let (buffers, material, fwd_pipelines, _targets, compute_bg, _render_bg) =
        rmesh_render::setup_forward(&device, &queue, scene, &zero_base_colors, &scene.color_grads, w, h);

    let uniforms = rmesh_render::make_uniforms(
        vp, c2w, intrinsics, cam_pos, w as f32, h as f32, scene.tet_count, 0u32, 12, 0.0, 0, 0.01, 100.0,
    );
    queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

    // Run forward compute
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    rmesh_render::record_project_compute(
        &mut encoder, &fwd_pipelines, &buffers, &compute_bg, scene.tet_count, &queue,
    );
    queue.submit(std::iter::once(encoder.finish()));

    // Interval tiled buffers + pipelines
    let interval_buffers = rmesh_render::IntervalTiledBuffers::new(&device, scene.tet_count);
    let interval_gen = rmesh_render::IntervalGeneratePipeline::new(&device);
    let interval_gen_bg = rmesh_render::create_interval_generate_bind_group(
        &device, &interval_gen, &buffers, &material, &interval_buffers,
    );

    // Tile pipeline setup
    let tile_pipelines = rmesh_tile::TilePipelines::new(&device);
    let radix_pipelines = rmesh_sort::RadixSortPipelines::new(&device, 2, rmesh_sort::SortBackend::Drs);
    let tile_buffers = rmesh_tile::TileBuffers::new(&device, scene.tet_count, w, h, tile_size);
    let sorting_bits = rmesh_sort::sorting_bits_for_tiles(tile_buffers.num_tiles, rmesh_sort::SortBackend::Drs);
    let radix_state = rmesh_sort::RadixSortState::new(&device, tile_buffers.max_pairs_pow2, sorting_bits, 2, rmesh_sort::SortBackend::Drs);
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
        &scan_buffers, radix_state.num_keys_buf(),
    );

    let tile_ranges_bg_a = rmesh_tile::create_tile_ranges_bind_group_with_keys(
        &device, &tile_pipelines, &tile_buffers.tile_sort_keys,
        &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms, radix_state.num_keys_buf(),
    );
    let tile_ranges_bg_b = rmesh_tile::create_tile_ranges_bind_group_with_keys(
        &device, &tile_pipelines, radix_state.keys_b(),
        &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms, radix_state.num_keys_buf(),
    );

    // Interval tiled rasterize (replaces rasterize_compute)
    let interval_rasterize = rmesh_render::IntervalTiledRasterizePipeline::new(&device, w, h, 0);
    let interval_rasterize_bg_a = rmesh_render::create_interval_tiled_rasterize_bind_group(
        &device, &interval_rasterize, &buffers.uniforms, &interval_buffers,
        &tile_buffers.tile_sort_values, &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms,
        &interval_rasterize.aux_data_dummy, &interval_rasterize.aux_image,
    );
    let interval_rasterize_bg_b = rmesh_render::create_interval_tiled_rasterize_bind_group(
        &device, &interval_rasterize, &buffers.uniforms, &interval_buffers,
        radix_state.values_b(), &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms,
        &interval_rasterize.aux_data_dummy, &interval_rasterize.aux_image,
    );

    // Dispatch tiled forward pipeline
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.clear_buffer(&interval_rasterize.rendered_image, 0, None);
    encoder.clear_buffer(&interval_rasterize.xyzd_image, 0, None);
    encoder.clear_buffer(&interval_rasterize.distortion_image, 0, None);
    encoder.clear_buffer(&tile_buffers.tile_ranges, 0, None);

    // 1. Interval generate (needs compact_tet_ids from project_compute)
    rmesh_render::record_interval_generate(
        &mut encoder, &interval_gen, &interval_gen_bg, scene.tet_count,
    );

    // 2. Scan-based tile pipeline
    rmesh_tile::record_scan_tile_pipeline(
        &mut encoder, &scan_pipelines, &tile_pipelines,
        &prepare_dispatch_bg, &rts_bg,
        &tile_fill_bg, &tile_gen_scan_bg, &scan_buffers, &tile_buffers,
    );

    // 3. Radix sort
    let result_in_b = rmesh_sort::record_radix_sort(
        &mut encoder, &device, &radix_pipelines, &radix_state,
        &tile_buffers.tile_sort_keys, &tile_buffers.tile_sort_values,
    );

    // 4. Tile ranges
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

    // 5. Interval tiled rasterize
    {
        let fwd_bg = if result_in_b { &interval_rasterize_bg_b } else { &interval_rasterize_bg_a };
        rmesh_render::record_interval_tiled_rasterize(
            &mut encoder, &interval_rasterize, fwd_bg, tile_buffers.num_tiles,
        );
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
    encoder.copy_buffer_to_buffer(&interval_rasterize.rendered_image, 0, &readback, 0, readback.size());
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

// ---------------------------------------------------------------------------
// Compute-based interval rendering (no mesh shader required)
// ---------------------------------------------------------------------------

/// GPU render using compute-based interval shading (available on all GPUs).
pub fn gpu_compute_interval_render_scene(
    scene: &SceneData,
    cam_pos: Vec3,
    vp: Mat4,
    c2w: Mat3,
    intrinsics: [f32; 4],
    w: u32,
    h: u32,
) -> Option<Vec<[f32; 4]>> {
    pollster::block_on(gpu_compute_interval_render_scene_async(scene, cam_pos, vp, c2w, intrinsics, w, h))
}

async fn gpu_compute_interval_render_scene_async(
    scene: &SceneData,
    cam_pos: Vec3,
    vp: Mat4,
    c2w: Mat3,
    intrinsics: [f32; 4],
    w: u32,
    h: u32,
) -> Option<Vec<[f32; 4]>> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL,
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok()?;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::SUBGROUP
                | wgpu::Features::SHADER_FLOAT32_ATOMIC,
            required_limits: wgpu::Limits {
                max_storage_buffers_per_shader_stage: 20,
                max_storage_buffer_binding_size: 1 << 30,
                max_buffer_size: 1 << 30,
                ..wgpu::Limits::default()
            },
            ..Default::default()
        })
        .await
        .ok()?;

    let color_format = wgpu::TextureFormat::Rgba16Float;
    let zero_base_colors = vec![0.5f32; scene.tet_count as usize * 3];
    let (buffers, material, pipelines, targets, compute_bg, _render_bg) =
        rmesh_render::setup_forward(&device, &queue, scene, &zero_base_colors, &scene.color_grads, w, h);

    // Compute-interval pipelines (no mesh shader needed)
    let ci_pipelines = rmesh_render::ComputeIntervalPipelines::new(&device, color_format);

    // Sort infrastructure
    let n_pow2 = scene.tet_count.next_power_of_two();
    let sort_pipelines = rmesh_sort::RadixSortPipelines::new(&device, 1, rmesh_sort::SortBackend::Drs);
    let sort_state = rmesh_sort::RadixSortState::new(&device, n_pow2, 32, 1, rmesh_sort::SortBackend::Drs);
    sort_state.upload_configs(&queue);

    // Bind groups
    let gen_bg_a = rmesh_render::create_compute_interval_gen_bind_group(
        &device, &ci_pipelines, &buffers, &material,
    );
    let gen_bg_b = rmesh_render::create_compute_interval_gen_bind_group_with_sort_values(
        &device, &ci_pipelines, &buffers, &material, sort_state.values_b(),
    );
    let ci_render_bg = rmesh_render::create_compute_interval_render_bind_group(
        &device, &ci_pipelines, &buffers,
    );
    let ci_convert_bg = rmesh_render::create_compute_interval_indirect_convert_bind_group(
        &device, &ci_pipelines, &buffers,
    );

    // Write uniforms
    let uniforms = rmesh_render::make_uniforms(
        vp, c2w, intrinsics, cam_pos,
        w as f32, h as f32,
        scene.tet_count, 0, 16, 0.0, 0,
        0.01, 100.0,
    );
    queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

    // Depth texture
    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test_ci_depth"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

    // Record and submit
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_ci"),
    });

    // Clear color and depth
    {
        let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("test_ci_clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &targets.color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
    }

    rmesh_render::record_sorted_compute_interval_forward_pass(
        &mut encoder,
        &device,
        &pipelines,
        &ci_pipelines,
        &sort_pipelines,
        &sort_state,
        &buffers,
        &targets,
        &compute_bg,
        &gen_bg_a,
        &gen_bg_b,
        &ci_render_bg,
        &ci_convert_bg,
        scene.tet_count,
        &queue,
        &depth_view,
        None,
        None,
        false,
        true,
    );

    // Copy color texture to readback buffer
    let bytes_per_pixel = 8u32;
    let bytes_per_row = w * bytes_per_pixel;
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
            let g = half::f16::from_le_bytes([data[pixel_start + 2], data[pixel_start + 3]]).to_f32();
            let b = half::f16::from_le_bytes([data[pixel_start + 4], data[pixel_start + 5]]).to_f32();
            let a = half::f16::from_le_bytes([data[pixel_start + 6], data[pixel_start + 7]]).to_f32();
            image[(row * w + col) as usize] = [r, g, b, a];
        }
    }

    drop(data);
    readback.unmap();

    Some(image)
}

// ---------------------------------------------------------------------------
// MRT compositing: CPU reference + GPU readback
// ---------------------------------------------------------------------------

/// Per-tet auxiliary data for MRT testing.
pub struct MrtAuxData {
    /// Packed [M * 8]: roughness, env_f0..3, albedo_r, albedo_g, albedo_b
    pub aux_flat: Vec<f32>,
    /// Per-vertex normals [V * 3]
    pub vertex_normals: Vec<f32>,
}

/// MRT image set.
pub struct MrtImages {
    pub color: Vec<[f32; 4]>,
    pub aux0: Vec<[f32; 4]>,
    pub normals: Vec<[f32; 4]>,
    pub albedo: Vec<[f32; 4]>,
}

fn oct_encode_cpu(n: Vec3) -> [f32; 2] {
    let s = n.x.abs() + n.y.abs() + n.z.abs();
    if s < 1e-12 { return [0.5, 0.5]; }
    let mut px = n.x / s;
    let mut py = n.y / s;
    if n.z < 0.0 {
        let ox = px; let oy = py;
        px = (1.0 - oy.abs()) * if ox >= 0.0 { 1.0 } else { -1.0 };
        py = (1.0 - ox.abs()) * if oy >= 0.0 { 1.0 } else { -1.0 };
    }
    [px * 0.5 + 0.5, py * 0.5 + 0.5]
}

fn pack_2f_cpu(a: f32, b: f32) -> f32 {
    (a.clamp(0.0, 1.0) * 255.0).floor() + b.clamp(0.0, 1.0)
}

/// CPU reference renderer with MRT aux channels.
pub fn cpu_render_scene_with_mrt(
    scene: &SceneData,
    aux: &MrtAuxData,
    cam_pos: Vec3, vp: Mat4, c2w: Mat3, intrinsics: [f32; 4], w: u32, h: u32,
) -> MrtImages {
    let n = scene.tet_count as usize;
    let sorted = sort_tets_back_to_front(scene, cam_pos);

    struct TD { verts: [Vec3; 4], color: Vec3, grad: Vec3, density: f32 }
    let tets: Vec<TD> = (0..n).map(|i| TD {
        verts: load_tet_verts(scene, i),
        color: compute_tet_color(scene, i, cam_pos),
        grad: load_color_grad(scene, i),
        density: scene.densities[i],
    }).collect();

    let visible: Vec<bool> = (0..n).map(|i| {
        let td = &tets[i];
        let (mut mn_x, mut mx_x) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut mn_y, mut mx_y) = (f32::INFINITY, f32::NEG_INFINITY);
        let mut mn_z = f32::INFINITY;
        for v in &td.verts {
            let clip = vp * Vec4::new(v.x, v.y, v.z, 1.0);
            let iw = 1.0 / (clip.w + 1e-6);
            let nd = clip.truncate() * iw;
            mn_x = mn_x.min(nd.x); mx_x = mx_x.max(nd.x);
            mn_y = mn_y.min(nd.y); mx_y = mx_y.max(nd.y);
            mn_z = mn_z.min(nd.z);
        }
        !(mx_x < -1.0 || mn_x > 1.0 || mx_y < -1.0 || mn_y > 1.0 || mn_z > 1.0)
            && (mx_x - mn_x) * w as f32 * (mx_y - mn_y) * h as f32 >= 1.0
    }).collect();

    let npix = (w * h) as usize;
    let mut col = vec![[0.0f32; 4]; npix];
    let mut a0 = vec![[0.0f32; 4]; npix];
    let mut nm = vec![[0.0f32; 4]; npix];
    let mut al = vec![[0.0f32; 4]; npix];

    for py in 0..h {
        for px in 0..w {
            let ray = pixel_ray_dir(c2w, intrinsics, cam_pos, px as f32, py as f32);
            let pi = (py * w + px) as usize;
            for &tid in &sorted {
                let ti = tid as usize;
                if !visible[ti] { continue; }
                let td = &tets[ti];
                let off = td.grad.dot(cam_pos - td.verts[0]);
                let bc = td.color + Vec3::splat(off);
                let [cr, cg, cb, ca] = render_tet_pixel(cam_pos, ray, &td.verts, td.density, bc, td.grad);
                if ca < 1e-7 { continue; }

                let ab = ti * 8;
                let g = |i: usize| if ab + i < aux.aux_flat.len() { aux.aux_flat[ab + i] } else { 0.0 };

                let i0 = scene.indices[ti*4] as usize;
                let i1 = scene.indices[ti*4+1] as usize;
                let i2 = scene.indices[ti*4+2] as usize;
                let i3 = scene.indices[ti*4+3] as usize;
                let vn = |vi: usize| -> Vec3 {
                    if vi*3+2 < aux.vertex_normals.len() {
                        Vec3::new(aux.vertex_normals[vi*3], aux.vertex_normals[vi*3+1], aux.vertex_normals[vi*3+2])
                    } else { Vec3::ZERO }
                };
                let normal = (vn(i0)+vn(i1)+vn(i2)+vn(i3)).normalize_or_zero();
                let [ox, oy] = oct_encode_cpu(normal);
                let ep = pack_2f_cpu(g(3), g(4));

                let blend = |acc: &mut [f32; 4], s: [f32; 4]| {
                    let a = s[3];
                    for c in 0..4 { acc[c] = s[c] + acc[c] * (1.0 - a); }
                };
                blend(&mut col[pi], [cr, cg, cb, ca]);
                blend(&mut a0[pi], [g(0)*ca, g(1)*ca, g(2)*ca, ca]);
                blend(&mut nm[pi], [ox*ca, oy*ca, ep*ca, ca]);
                blend(&mut al[pi], [g(5)*ca, g(6)*ca, g(7)*ca, ca]);
            }
        }
    }
    MrtImages { color: col, aux0: a0, normals: nm, albedo: al }
}

/// GPU compute-interval render with MRT, returning all 4 targets.
pub fn gpu_compute_interval_render_scene_with_mrt(
    scene: &SceneData, aux: &MrtAuxData,
    cam_pos: Vec3, vp: Mat4, c2w: Mat3, intrinsics: [f32; 4], w: u32, h: u32,
) -> Option<MrtImages> {
    pollster::block_on(gpu_ci_mrt_async(scene, aux, cam_pos, vp, c2w, intrinsics, w, h))
}

async fn gpu_ci_mrt_async(
    scene: &SceneData, aux: &MrtAuxData,
    cam_pos: Vec3, vp: Mat4, c2w: Mat3, intrinsics: [f32; 4], w: u32, h: u32,
) -> Option<MrtImages> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL,
        ..Default::default()
    });
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None, force_fallback_adapter: false,
    }).await.ok()?;
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::SUBGROUP | wgpu::Features::SHADER_FLOAT32_ATOMIC,
        required_limits: wgpu::Limits {
            max_storage_buffers_per_shader_stage: 20,
            max_storage_buffer_binding_size: 1 << 30,
            max_buffer_size: 1 << 30,
            ..wgpu::Limits::default()
        },
        ..Default::default()
    }).await.ok()?;

    let color_format = wgpu::TextureFormat::Rgba16Float;
    let base_colors = vec![0.5f32; scene.tet_count as usize * 3];
    let (buffers, material, pipelines, targets, compute_bg, _) =
        rmesh_render::setup_forward(&device, &queue, scene, &base_colors, &scene.color_grads, w, h);
    let ci_pipelines = rmesh_render::ComputeIntervalPipelines::new(&device, color_format);

    let n_pow2 = scene.tet_count.next_power_of_two();
    let sort_pipelines = rmesh_sort::RadixSortPipelines::new(&device, 1, rmesh_sort::SortBackend::Drs);
    let sort_state = rmesh_sort::RadixSortState::new(&device, n_pow2, 32, 1, rmesh_sort::SortBackend::Drs);
    sort_state.upload_configs(&queue);

    use wgpu::util::DeviceExt;
    let aux_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("test_aux"), contents: bytemuck::cast_slice(&aux.aux_flat),
        usage: wgpu::BufferUsages::STORAGE,
    });
    queue.write_buffer(&buffers.vertex_normals, 0, bytemuck::cast_slice(&aux.vertex_normals));

    let gen_bg_a = rmesh_render::create_compute_interval_gen_bind_group(&device, &ci_pipelines, &buffers, &material);
    let gen_bg_b = rmesh_render::create_compute_interval_gen_bind_group_with_sort_values(
        &device, &ci_pipelines, &buffers, &material, sort_state.values_b(),
    );
    let ci_render_bg = rmesh_render::create_compute_interval_render_bind_group_pbr(
        &device, &ci_pipelines, &buffers, &aux_buf, &buffers.indices,
    );
    let ci_convert_bg = rmesh_render::create_compute_interval_indirect_convert_bind_group(&device, &ci_pipelines, &buffers);

    let uniforms = rmesh_render::make_uniforms(vp, c2w, intrinsics, cam_pos, w as f32, h as f32,
        scene.tet_count, 0, 16, 0.0, 0, 0.01, 100.0);
    queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test_mrt_depth"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT, view_formats: &[],
    });
    let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("test_mrt") });
    {
        let clear = wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store };
        let _rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mrt_clear"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment { view: &targets.color_view, resolve_target: None, ops: clear, depth_slice: None }),
                Some(wgpu::RenderPassColorAttachment { view: &targets.aux0_view, resolve_target: None, ops: clear, depth_slice: None }),
                Some(wgpu::RenderPassColorAttachment { view: &targets.normals_view, resolve_target: None, ops: clear, depth_slice: None }),
                Some(wgpu::RenderPassColorAttachment { view: &targets.depth_view, resolve_target: None, ops: clear, depth_slice: None }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
    }

    rmesh_render::record_sorted_compute_interval_forward_pass(
        &mut encoder, &device, &pipelines, &ci_pipelines,
        &sort_pipelines, &sort_state, &buffers, &targets, &compute_bg,
        &gen_bg_a, &gen_bg_b, &ci_render_bg, &ci_convert_bg,
        scene.tet_count, &queue, &depth_view, None, None, false, true,
    );

    let tex_list = [&targets.color_texture, &targets.aux0_texture, &targets.normals_texture, &targets.depth_texture];
    let bpp = 8u32;
    let abpr = ((w * bpp) + 255) & !255;
    let bsz = (abpr * h) as u64;

    let rbs: Vec<wgpu::Buffer> = (0..4).map(|i| device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("rb{i}")), size: bsz,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })).collect();

    for (i, tex) in tex_list.iter().enumerate() {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &rbs[i], layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(abpr), rows_per_image: Some(h) } },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
    }
    queue.submit(std::iter::once(encoder.finish()));

    let mut imgs: Vec<Vec<[f32; 4]>> = Vec::new();
    for rb in &rbs {
        let slice = rb.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().ok()?;
        let data = slice.get_mapped_range();
        let mut img = vec![[0.0f32; 4]; (w*h) as usize];
        for row in 0..h {
            let rs = (row * abpr) as usize;
            for col in 0..w {
                let ps = rs + (col * bpp) as usize;
                let r = half::f16::from_le_bytes([data[ps], data[ps+1]]).to_f32();
                let g = half::f16::from_le_bytes([data[ps+2], data[ps+3]]).to_f32();
                let b = half::f16::from_le_bytes([data[ps+4], data[ps+5]]).to_f32();
                let a = half::f16::from_le_bytes([data[ps+6], data[ps+7]]).to_f32();
                img[(row*w+col) as usize] = [r, g, b, a];
            }
        }
        drop(data);
        rb.unmap();
        imgs.push(img);
    }
    Some(MrtImages { color: imgs.remove(0), aux0: imgs.remove(0), normals: imgs.remove(0), albedo: imgs.remove(0) })
}
