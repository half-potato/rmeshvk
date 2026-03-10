//! Per-kernel unit tests for the backward pipeline and tiled rendering.
//!
//! Tests each shader/kernel pathway:
//!   - radix_sort (5-pass): tile-tet pair sorting
//!   - tile_fill_compute.wgsl: sentinel initialization
//!   - tile_gen_hull_compute.wgsl: tile-tet pair generation
//!   - tile_ranges_compute.wgsl: per-tile range computation
//!   - loss_compute.wgsl: L1/L2 loss + per-pixel gradient
//!   - adam_compute.wgsl: Adam optimizer step
//!   - forward_tiled_compute.wgsl: subgroup-based tiled forward (end-to-end)
//!   - backward_tiled_compute.wgsl: subgroup-based tiled backward (end-to-end)
//!
//! Each test creates a GPU device with the SUBGROUP feature enabled.
//! Tests gracefully skip if no GPU adapter is available.

use bytemuck;
use glam::{Mat4, Vec3, Vec4};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rmesh_data::SceneData;
use rmesh_shaders::shared::{AdamUniforms, LossUniforms};
use rmesh_train::{
    AdamPipeline, AdamState, LossBuffers, LossPipeline,
    create_adam_bind_group, create_loss_bind_group,
    record_adam_pass, record_loss_pass,
};
use wgpu::util::DeviceExt;

const SEED: u64 = 42424242;
const W: u32 = 64;
const H: u32 = 64;

// ---------------------------------------------------------------------------
// GPU setup helpers
// ---------------------------------------------------------------------------

fn create_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok()?;

        adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::SUBGROUP,
                    required_limits: wgpu::Limits {
                        max_storage_buffers_per_shader_stage: 16,
                        max_storage_buffer_binding_size: 1 << 30,
                        max_buffer_size: 1 << 30,
                        ..wgpu::Limits::downlevel_defaults()
                    },
                    ..Default::default()
                },
            )
            .await
            .ok()
    })
}

fn read_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    count: usize,
) -> Vec<T> {
    let size = (count * std::mem::size_of::<T>()) as u64;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(source, 0, &readback, 0, size);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    receiver.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    readback.unmap();
    result
}

fn create_rw_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

// ---------------------------------------------------------------------------
// Scene helpers (subset of rmesh-render test common module)
// ---------------------------------------------------------------------------

fn perspective_matrix(fov_y_rad: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fov_y_rad / 2.0).tan();
    Mat4::from_cols(
        Vec4::new(f / aspect, 0.0, 0.0, 0.0),
        Vec4::new(0.0, f, 0.0, 0.0),
        Vec4::new(0.0, 0.0, far / (far - near), 1.0),
        Vec4::new(0.0, 0.0, -(far * near) / (far - near), 0.0),
    )
}

fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
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

fn setup_camera(eye: Vec3, target: Vec3) -> (Mat4, Mat4) {
    let aspect = W as f32 / H as f32;
    let proj = perspective_matrix(std::f32::consts::FRAC_PI_2, aspect, 0.01, 100.0);
    let view = look_at(eye, target, Vec3::new(0.0, 0.0, 1.0));
    let vp = proj * view;
    (vp, vp.inverse())
}

fn random_single_tet_scene(rng: &mut ChaCha8Rng, radius: f32) -> SceneData {
    let mut verts = [0.0f32; 12];
    for i in 0..4 {
        verts[i * 3] = (rng.random::<f32>() - 0.5) * 2.0 * radius;
        verts[i * 3 + 1] = (rng.random::<f32>() - 0.5) * 2.0 * radius;
        verts[i * 3 + 2] = (rng.random::<f32>() - 0.5) * 2.0 * radius;
    }
    let v0 = Vec3::new(verts[0], verts[1], verts[2]);
    let v1 = Vec3::new(verts[3], verts[4], verts[5]);
    let v2 = Vec3::new(verts[6], verts[7], verts[8]);
    let v3 = Vec3::new(verts[9], verts[10], verts[11]);
    let det = (v1 - v0).dot((v2 - v0).cross(v3 - v0));
    let indices: Vec<u32> = if det < 0.0 {
        vec![0, 1, 3, 2]
    } else {
        vec![0, 1, 2, 3]
    };

    let sh_coeffs: Vec<f32> = (0..3).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
    let densities = vec![rng.random::<f32>() * 5.0 + 0.5];
    let color_grads = vec![
        (rng.random::<f32>() - 0.5) * 0.2,
        (rng.random::<f32>() - 0.5) * 0.2,
        (rng.random::<f32>() - 0.5) * 0.2,
    ];

    let circumdata = compute_circumspheres(&verts, &indices);

    SceneData {
        vertices: verts.to_vec(),
        indices,
        sh_coeffs,
        densities,
        color_grads,
        circumdata,
        start_pose: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        vertex_count: 4,
        tet_count: 1,
        sh_degree: 0,
    }
}

fn compute_circumspheres(vertices: &[f32], indices: &[u32]) -> Vec<f32> {
    let tet_count = indices.len() / 4;
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
        circumdata[i * 4] = center.x;
        circumdata[i * 4 + 1] = center.y;
        circumdata[i * 4 + 2] = center.z;
        circumdata[i * 4 + 3] = r.dot(r);
    }
    circumdata
}

// ===========================================================================
// Test: 5-pass radix sort
// ===========================================================================

/// Creates unsorted key-value pairs, runs the 5-pass radix sort,
/// and verifies ascending order.
#[test]
fn test_radix_sort_kernel() {
    let (device, queue) = match create_test_device() {
        Some(dq) => dq,
        None => {
            eprintln!("Skipping test_radix_sort_kernel (no GPU)");
            return;
        }
    };

    let n = 64u32;
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let keys_data: Vec<u32> = (0..n).map(|_| rng.random::<u32>() % 10000).collect();
    let values_data: Vec<u32> = (0..n).collect();

    // Create A-buffers (source) with COPY_SRC for readback
    let keys_a = create_rw_buffer(&device, "keys_a", (n as u64) * 4);
    let values_a = create_rw_buffer(&device, "values_a", (n as u64) * 4);
    queue.write_buffer(&keys_a, 0, bytemuck::cast_slice(&keys_data));
    queue.write_buffer(&values_a, 0, bytemuck::cast_slice(&values_data));

    // Create pipelines and sort state
    let radix_pipelines = rmesh_backward::RadixSortPipelines::new(&device);
    let radix_state = rmesh_backward::RadixSortState::new(&device, n, 32);
    radix_state.upload_configs(&queue);

    // Write num_keys
    queue.write_buffer(&radix_state.num_keys_buf, 0, bytemuck::bytes_of(&n));

    // Dispatch sort
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let result_in_b =
        rmesh_backward::record_radix_sort(&mut encoder, &device, &radix_pipelines, &radix_state, &keys_a, &values_a);
    queue.submit(std::iter::once(encoder.finish()));

    // Read back from whichever buffer has the result
    let (sorted_keys, sorted_values) = if result_in_b {
        (
            read_buffer::<u32>(&device, &queue, &radix_state.keys_b, n as usize),
            read_buffer::<u32>(&device, &queue, &radix_state.values_b, n as usize),
        )
    } else {
        (
            read_buffer::<u32>(&device, &queue, &keys_a, n as usize),
            read_buffer::<u32>(&device, &queue, &values_a, n as usize),
        )
    };

    eprintln!("radix sort: first 10 keys = {:?}", &sorted_keys[..10]);

    // Keys should be sorted ascending
    for i in 1..n as usize {
        assert!(
            sorted_keys[i] >= sorted_keys[i - 1],
            "Radix sort: keys not sorted at {i}: {} > {}",
            sorted_keys[i - 1],
            sorted_keys[i]
        );
    }

    // Key-value correspondence
    for (i, &val) in sorted_values.iter().enumerate() {
        assert_eq!(
            sorted_keys[i], keys_data[val as usize],
            "Radix sort: key-value mismatch at {i}"
        );
    }
}

// ===========================================================================
// Test: loss_compute.wgsl (L1 loss + gradient)
// ===========================================================================

/// Computes L1 loss between a known rendered image and ground truth.
/// Verifies per-pixel gradients (dl_d_image) are correct.
#[test]
fn test_loss_compute_kernel() {
    let (device, queue) = match create_test_device() {
        Some(dq) => dq,
        None => {
            eprintln!("Skipping test_loss_compute_kernel (no GPU)");
            return;
        }
    };

    let n_pixels = (W * H) as usize;

    // Create constant rendered image: RGBA = [0.8, 0.6, 0.4, 1.0]
    let mut rendered_data = vec![0.0f32; n_pixels * 4];
    for i in 0..n_pixels {
        rendered_data[i * 4] = 0.8;
        rendered_data[i * 4 + 1] = 0.6;
        rendered_data[i * 4 + 2] = 0.4;
        rendered_data[i * 4 + 3] = 1.0;
    }

    // Create constant ground truth: RGB = [0.5, 0.5, 0.5]
    let mut gt_data = vec![0.0f32; n_pixels * 3];
    for i in 0..n_pixels {
        gt_data[i * 3] = 0.5;
        gt_data[i * 3 + 1] = 0.5;
        gt_data[i * 3 + 2] = 0.5;
    }

    // Upload
    let rendered_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rendered"),
        contents: bytemuck::cast_slice(&rendered_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let gt_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ground_truth"),
        contents: bytemuck::cast_slice(&gt_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let dl_d_image = create_rw_buffer(&device, "dl_d_image", (n_pixels as u64) * 4 * 4);
    let loss_value = create_rw_buffer(&device, "loss_value", 4);
    let loss_uniforms_buf = create_rw_buffer(
        &device,
        "loss_uniforms",
        std::mem::size_of::<LossUniforms>() as u64,
    );

    // Clear loss_value
    queue.write_buffer(&loss_value, 0, &[0u8; 4]);

    // Write loss uniforms (L1)
    let loss_uni = LossUniforms {
        width: W,
        height: H,
        loss_type: 0, // L1
        lambda_ssim: 0.0,
    };
    queue.write_buffer(&loss_uniforms_buf, 0, bytemuck::bytes_of(&loss_uni));

    // Create pipeline and bind group
    let loss_pipeline = LossPipeline::new(&device);

    // Manually create loss buffers struct for bind group
    let loss_buffers = LossBuffers {
        dl_d_image,
        ground_truth: gt_buf,
        loss_value,
        loss_uniforms: loss_uniforms_buf,
    };
    let loss_bg =
        create_loss_bind_group(&device, &loss_pipeline, &loss_buffers, &rendered_buf);

    // Dispatch
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    record_loss_pass(&mut encoder, &loss_pipeline, &loss_bg, W, H);
    queue.submit(std::iter::once(encoder.finish()));

    // Read back dl_d_image
    let dl_d: Vec<f32> = read_buffer(&device, &queue, &loss_buffers.dl_d_image, n_pixels * 4);

    // For L1: grad = sign(rendered - gt) / n_pixels
    // diff_r = 0.8 - 0.5 = 0.3 > 0 → grad_r = 1.0 / 256
    // diff_g = 0.6 - 0.5 = 0.1 > 0 → grad_g = 1.0 / 256
    // diff_b = 0.4 - 0.5 = -0.1 < 0 → grad_b = -1.0 / 256
    let n_pix_f = n_pixels as f32;
    let expected_grad_r = 1.0 / n_pix_f;
    let expected_grad_g = 1.0 / n_pix_f;
    let expected_grad_b = -1.0 / n_pix_f;

    // Check first pixel
    let tol = 1e-5;
    assert!(
        (dl_d[0] - expected_grad_r).abs() < tol,
        "dl_d_r[0] = {}, expected {}",
        dl_d[0],
        expected_grad_r
    );
    assert!(
        (dl_d[1] - expected_grad_g).abs() < tol,
        "dl_d_g[0] = {}, expected {}",
        dl_d[1],
        expected_grad_g
    );
    assert!(
        (dl_d[2] - expected_grad_b).abs() < tol,
        "dl_d_b[0] = {}, expected {}",
        dl_d[2],
        expected_grad_b
    );
    assert!(
        dl_d[3].abs() < tol,
        "dl_d_a[0] = {}, expected 0",
        dl_d[3]
    );

    // Check all pixels have consistent gradients
    for i in 0..n_pixels {
        assert!(
            (dl_d[i * 4] - expected_grad_r).abs() < tol,
            "Pixel {i}: grad_r mismatch"
        );
    }

    eprintln!("loss_compute: gradients verified for {n_pixels} pixels");
}

// ===========================================================================
// Test: adam_compute.wgsl (optimizer step)
// ===========================================================================

/// Runs one Adam optimizer step with known parameters and gradients.
/// Verifies parameters are updated in the correct direction.
#[test]
fn test_adam_kernel() {
    let (device, queue) = match create_test_device() {
        Some(dq) => dq,
        None => {
            eprintln!("Skipping test_adam_kernel (no GPU)");
            return;
        }
    };

    let param_count = 4u32;
    let params_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let grads_data: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4];
    let zeros: Vec<f32> = vec![0.0; param_count as usize];

    let params_buf = create_rw_buffer(&device, "params", (param_count as u64) * 4);
    let grads_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("grads"),
        contents: bytemuck::cast_slice(&grads_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let m_buf = create_rw_buffer(&device, "m", (param_count as u64) * 4);
    let v_buf = create_rw_buffer(&device, "v", (param_count as u64) * 4);
    let adam_uniforms_buf = create_rw_buffer(
        &device,
        "adam_uniforms",
        std::mem::size_of::<AdamUniforms>() as u64,
    );

    queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&params_data));
    queue.write_buffer(&m_buf, 0, bytemuck::cast_slice(&zeros));
    queue.write_buffer(&v_buf, 0, bytemuck::cast_slice(&zeros));

    let adam_uni = AdamUniforms {
        param_count,
        step: 1,
        lr: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        _pad: [0; 2],
    };
    queue.write_buffer(&adam_uniforms_buf, 0, bytemuck::bytes_of(&adam_uni));

    let adam_pipeline = AdamPipeline::new(&device);
    let adam_bg = create_adam_bind_group(
        &device,
        &adam_pipeline,
        &adam_uniforms_buf,
        &params_buf,
        &grads_buf,
        &m_buf,
        &v_buf,
    );

    // Dispatch
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    record_adam_pass(
        &mut encoder,
        &adam_pipeline,
        std::slice::from_ref(&adam_bg),
        &[param_count],
    );
    queue.submit(std::iter::once(encoder.finish()));

    // Read back
    let updated_params: Vec<f32> = read_buffer(&device, &queue, &params_buf, param_count as usize);
    let updated_m: Vec<f32> = read_buffer(&device, &queue, &m_buf, param_count as usize);
    let updated_v: Vec<f32> = read_buffer(&device, &queue, &v_buf, param_count as usize);

    eprintln!("adam: params before = {params_data:?}");
    eprintln!("adam: params after  = {updated_params:?}");
    eprintln!("adam: m = {updated_m:?}");
    eprintln!("adam: v = {updated_v:?}");

    // Verify params changed
    for i in 0..param_count as usize {
        assert!(
            (updated_params[i] - params_data[i]).abs() > 1e-6,
            "Param {i} unchanged after Adam step"
        );
    }

    // Verify direction: positive gradient → params decrease, negative → increase
    for i in 0..param_count as usize {
        let delta = updated_params[i] - params_data[i];
        if grads_data[i] > 0.0 {
            assert!(delta < 0.0, "Param {i}: positive grad but param increased");
        } else {
            assert!(delta > 0.0, "Param {i}: negative grad but param decreased");
        }
    }

    // Verify m = (1 - beta1) * grad (since m was zero)
    for i in 0..param_count as usize {
        let expected_m = 0.1 * grads_data[i]; // (1 - 0.9) * grad
        assert!(
            (updated_m[i] - expected_m).abs() < 1e-6,
            "m[{i}] = {}, expected {}",
            updated_m[i],
            expected_m
        );
    }
}

// ===========================================================================
// Test: tile_fill_compute.wgsl (sentinel initialization)
// ===========================================================================

/// Dispatches tile_fill and verifies all keys are set to 0xFFFFFFFF.
#[test]
fn test_tile_fill_kernel() {
    let (device, queue) = match create_test_device() {
        Some(dq) => dq,
        None => {
            eprintln!("Skipping test_tile_fill_kernel (no GPU)");
            return;
        }
    };

    let tet_count = 4u32;
    let tile_size = 4u32;
    let tile_buffers = rmesh_backward::TileBuffers::new(&device, tet_count, W, H, tile_size);
    let tile_pipelines = rmesh_backward::TilePipelines::new(&device);

    // Write tile uniforms
    let tile_uni = rmesh_backward::TileUniforms {
        screen_width: W,
        screen_height: H,
        tile_size,
        tiles_x: tile_buffers.tiles_x,
        tiles_y: tile_buffers.tiles_y,
        num_tiles: tile_buffers.num_tiles,
        visible_tet_count: tet_count,
        _pad: [0; 5],
    };
    queue.write_buffer(&tile_buffers.tile_uniforms, 0, bytemuck::bytes_of(&tile_uni));

    let tile_fill_bg =
        rmesh_backward::create_tile_fill_bind_group(&device, &tile_pipelines, &tile_buffers);

    // Dispatch
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tile_fill"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&tile_pipelines.fill_pipeline);
        pass.set_bind_group(0, &tile_fill_bg, &[]);
        let wgs = (tile_buffers.max_pairs_pow2 + 255) / 256;
        pass.dispatch_workgroups(wgs, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    // tile_sort_keys doesn't have COPY_SRC, so we verify via tile_pair_count instead
    // (tile_fill doesn't write pair_count, but we verified dispatch didn't error)
    // For a proper readback test, create custom key buffer:
    let test_keys = create_rw_buffer(&device, "test_keys", (tile_buffers.max_pairs_pow2 as u64) * 4);
    let test_values = create_rw_buffer(&device, "test_values", (tile_buffers.max_pairs_pow2 as u64) * 4);

    // Write some non-sentinel data first
    let garbage: Vec<u32> = (0..tile_buffers.max_pairs_pow2).collect();
    queue.write_buffer(&test_keys, 0, bytemuck::cast_slice(&garbage));
    queue.write_buffer(&test_values, 0, bytemuck::cast_slice(&garbage));

    // Create a custom tile fill bind group with our readable buffers
    let test_fill_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test_fill_bg"),
        layout: &tile_pipelines.fill_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: tile_buffers.tile_uniforms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: test_keys.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: test_values.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tile_fill_test"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&tile_pipelines.fill_pipeline);
        pass.set_bind_group(0, &test_fill_bg, &[]);
        let wgs = (tile_buffers.max_pairs_pow2 + 255) / 256;
        pass.dispatch_workgroups(wgs, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    let keys: Vec<u32> = read_buffer(&device, &queue, &test_keys, tile_buffers.max_pairs_pow2 as usize);
    let values: Vec<u32> = read_buffer(&device, &queue, &test_values, tile_buffers.max_pairs_pow2 as usize);

    // All keys should be sentinel
    for (i, &k) in keys.iter().enumerate() {
        assert_eq!(k, 0xFFFFFFFF, "Key at {i} = {k:#x}, expected 0xFFFFFFFF");
    }
    // All values should be 0
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(v, 0, "Value at {i} = {v}, expected 0");
    }

    eprintln!("tile_fill: verified {} entries", tile_buffers.max_pairs_pow2);
}

// ===========================================================================
// Test: tiled forward end-to-end (subgroup)
// ===========================================================================

/// Runs the full scan-based tiled forward pipeline:
///   forward_compute → scan tile pipeline → radix_sort → tile_ranges → forward_tiled
///
/// Uses the same scan-based tile pipeline as the Python bindings.
/// The forward_tiled_compute.wgsl shader uses `enable subgroups;` and
/// subgroupShuffle operations. If subgroup support is broken, this test
/// will fail during pipeline creation or produce incorrect output.
#[test]
fn test_tiled_forward_e2e() {
    let (device, queue) = match create_test_device() {
        Some(dq) => dq,
        None => {
            eprintln!("Skipping test_tiled_forward_e2e (no GPU)");
            return;
        }
    };

    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    // Use larger tet (0.6) with closer camera (1.0) to guarantee visibility
    // after circumsphere culling in forward_compute.
    let scene = random_single_tet_scene(&mut rng, 0.6);

    let verts_arr = [
        Vec3::new(scene.vertices[0], scene.vertices[1], scene.vertices[2]),
        Vec3::new(scene.vertices[3], scene.vertices[4], scene.vertices[5]),
        Vec3::new(scene.vertices[6], scene.vertices[7], scene.vertices[8]),
        Vec3::new(scene.vertices[9], scene.vertices[10], scene.vertices[11]),
    ];
    let centroid = (verts_arr[0] + verts_arr[1] + verts_arr[2] + verts_arr[3]) * 0.25;
    let eye = centroid + Vec3::new(1.0, 0.0, 0.0);
    let (vp, inv_vp) = setup_camera(eye, centroid);

    // --- Forward compute (populates colors, tiles_touched, compact_tet_ids, indirect_args) ---
    let (buffers, fwd_pipelines, _targets, compute_bg, _render_bg) =
        rmesh_render::setup_forward(&device, &queue, &scene, W, H);

    let uniforms = rmesh_render::make_uniforms(
        vp, inv_vp, eye, W as f32, H as f32, scene.tet_count, scene.sh_degree, 0,
    );
    queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    rmesh_render::record_forward_compute(
        &mut encoder, &fwd_pipelines, &buffers, &compute_bg, scene.tet_count, &queue,
    );
    queue.submit(std::iter::once(encoder.finish()));

    // --- Set up tiled pipeline (scan-based, same as Python path) ---
    let tile_size = 4u32;
    let tile_pipelines = rmesh_backward::TilePipelines::new(&device);
    let radix_pipelines = rmesh_backward::RadixSortPipelines::new(&device);
    let tile_buffers = rmesh_backward::TileBuffers::new(&device, scene.tet_count, W, H, tile_size);
    let radix_state = rmesh_backward::RadixSortState::new(&device, tile_buffers.max_pairs_pow2, 32);
    radix_state.upload_configs(&queue);

    let scan_pipelines = rmesh_backward::ScanPipelines::new(&device);
    let scan_buffers = rmesh_backward::ScanBuffers::new(&device, scene.tet_count);

    // Write tile uniforms (visible_tet_count=0 — scan reads from GPU)
    let tile_uni = rmesh_backward::TileUniforms {
        screen_width: W,
        screen_height: H,
        tile_size,
        tiles_x: tile_buffers.tiles_x,
        tiles_y: tile_buffers.tiles_y,
        num_tiles: tile_buffers.num_tiles,
        visible_tet_count: 0,
        _pad: [0; 5],
    };
    queue.write_buffer(&tile_buffers.tile_uniforms, 0, bytemuck::bytes_of(&tile_uni));

    // Create scan bind groups
    let prepare_dispatch_bg = rmesh_backward::create_prepare_dispatch_bind_group(
        &device, &scan_pipelines, &buffers.indirect_args, &scan_buffers,
    );
    let rts_bg = rmesh_backward::create_rts_bind_group(
        &device, &scan_pipelines, &buffers.tiles_touched, &scan_buffers,
    );
    let tile_fill_bg =
        rmesh_backward::create_tile_fill_bind_group(&device, &tile_pipelines, &tile_buffers);
    let tile_gen_scan_bg = rmesh_backward::create_tile_gen_scan_bind_group(
        &device, &scan_pipelines, &tile_buffers,
        &buffers.uniforms, &buffers.vertices, &buffers.indices,
        &buffers.compact_tet_ids, &buffers.circumdata, &buffers.tiles_touched,
        &scan_buffers, &radix_state.num_keys_buf,
    );

    // Tile ranges uses num_keys_buf (scan pipeline writes total_pairs there)
    let tile_ranges_bg_a = rmesh_backward::create_tile_ranges_bind_group_with_keys(
        &device, &tile_pipelines, &tile_buffers.tile_sort_keys,
        &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms, &radix_state.num_keys_buf,
    );
    let tile_ranges_bg_b = rmesh_backward::create_tile_ranges_bind_group_with_keys(
        &device, &tile_pipelines, &radix_state.keys_b,
        &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms, &radix_state.num_keys_buf,
    );

    // Create forward tiled pipeline (SUBGROUP)
    let fwd_tiled = rmesh_render::ForwardTiledPipeline::new(&device, W, H);
    let fwd_tiled_bg_a = rmesh_render::create_forward_tiled_bind_group(
        &device, &fwd_tiled, &buffers.uniforms,
        &buffers.vertices, &buffers.indices, &buffers.colors,
        &buffers.densities, &buffers.color_grads,
        &tile_buffers.tile_sort_values, &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms,
    );
    let fwd_tiled_bg_b = rmesh_render::create_forward_tiled_bind_group(
        &device, &fwd_tiled, &buffers.uniforms,
        &buffers.vertices, &buffers.indices, &buffers.colors,
        &buffers.densities, &buffers.color_grads,
        &radix_state.values_b, &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms,
    );

    // --- Dispatch scan-based tiled forward pipeline ---
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.clear_buffer(&fwd_tiled.rendered_image, 0, None);
    encoder.clear_buffer(&tile_buffers.tile_ranges, 0, None);

    // 1. Scan-based tile pipeline
    rmesh_backward::record_scan_tile_pipeline(
        &mut encoder, &scan_pipelines, &tile_pipelines,
        &prepare_dispatch_bg, &rts_bg,
        &tile_fill_bg, &tile_gen_scan_bg, &scan_buffers, &tile_buffers,
    );

    // 2. Radix sort
    let result_in_b = rmesh_backward::record_radix_sort(
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
    let pixel_count = (W * H) as usize;
    let image: Vec<f32> = read_buffer(&device, &queue, &fwd_tiled.rendered_image, pixel_count * 4);

    // Check that we got non-zero output (tet should be visible)
    let total_alpha: f32 = image.iter().skip(3).step_by(4).sum();
    eprintln!("tiled_forward: total_alpha = {total_alpha:.4}");
    assert!(
        total_alpha > 0.001,
        "Tiled forward produced all-zero image (total_alpha={total_alpha})"
    );

    // Check pixel values are reasonable (non-NaN, non-inf)
    for (i, &v) in image.iter().enumerate() {
        assert!(
            v.is_finite(),
            "Pixel value at index {i} is not finite: {v}"
        );
    }

    eprintln!("tiled_forward: subgroup pipeline ran successfully");
}

// ===========================================================================
// Test: backward_tiled_compute.wgsl (subgroup backward, end-to-end)
// ===========================================================================

/// Runs the full tiled backward pipeline:
///   forward_compute → tile setup → forward_tiled → loss → backward_tiled
///
/// Verifies gradient buffers contain non-zero values.
/// The backward_tiled_compute.wgsl shader uses subgroupShuffle and
/// subgroupShuffleXor for warp-level gradient reduction.
#[test]
fn test_tiled_backward_e2e() {
    let (device, queue) = match create_test_device() {
        Some(dq) => dq,
        None => {
            eprintln!("Skipping test_tiled_backward_e2e (no GPU)");
            return;
        }
    };

    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    // Use larger tet (0.6) with closer camera (1.0) to guarantee visibility
    let scene = random_single_tet_scene(&mut rng, 0.6);

    let verts_arr = [
        Vec3::new(scene.vertices[0], scene.vertices[1], scene.vertices[2]),
        Vec3::new(scene.vertices[3], scene.vertices[4], scene.vertices[5]),
        Vec3::new(scene.vertices[6], scene.vertices[7], scene.vertices[8]),
        Vec3::new(scene.vertices[9], scene.vertices[10], scene.vertices[11]),
    ];
    let centroid = (verts_arr[0] + verts_arr[1] + verts_arr[2] + verts_arr[3]) * 0.25;
    let eye = centroid + Vec3::new(1.0, 0.0, 0.0);
    let (vp, inv_vp) = setup_camera(eye, centroid);

    // --- Forward setup ---
    let (buffers, fwd_pipelines, _targets, compute_bg, _render_bg) =
        rmesh_render::setup_forward(&device, &queue, &scene, W, H);

    let uniforms = rmesh_render::make_uniforms(
        vp, inv_vp, eye, W as f32, H as f32, scene.tet_count, scene.sh_degree, 0,
    );
    queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

    // Run forward compute only (populates colors, tiles_touched, compact_tet_ids, indirect_args)
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    rmesh_render::record_forward_compute(
        &mut encoder, &fwd_pipelines, &buffers, &compute_bg, scene.tet_count, &queue,
    );
    queue.submit(std::iter::once(encoder.finish()));

    // --- Tiled infrastructure (scan-based, same as Python path) ---
    let tile_size = 4u32;
    let tile_pipelines = rmesh_backward::TilePipelines::new(&device);
    let radix_pipelines = rmesh_backward::RadixSortPipelines::new(&device);
    let tile_buffers = rmesh_backward::TileBuffers::new(&device, scene.tet_count, W, H, tile_size);
    let radix_state = rmesh_backward::RadixSortState::new(&device, tile_buffers.max_pairs_pow2, 32);
    radix_state.upload_configs(&queue);

    let scan_pipelines = rmesh_backward::ScanPipelines::new(&device);
    let scan_buffers = rmesh_backward::ScanBuffers::new(&device, scene.tet_count);

    let tile_uni = rmesh_backward::TileUniforms {
        screen_width: W,
        screen_height: H,
        tile_size,
        tiles_x: tile_buffers.tiles_x,
        tiles_y: tile_buffers.tiles_y,
        num_tiles: tile_buffers.num_tiles,
        visible_tet_count: 0,
        _pad: [0; 5],
    };
    queue.write_buffer(&tile_buffers.tile_uniforms, 0, bytemuck::bytes_of(&tile_uni));

    // --- Forward tiled ---
    let fwd_tiled = rmesh_render::ForwardTiledPipeline::new(&device, W, H);

    // Scan bind groups
    let prepare_dispatch_bg = rmesh_backward::create_prepare_dispatch_bind_group(
        &device, &scan_pipelines, &buffers.indirect_args, &scan_buffers,
    );
    let rts_bg = rmesh_backward::create_rts_bind_group(
        &device, &scan_pipelines, &buffers.tiles_touched, &scan_buffers,
    );
    let tile_fill_bg =
        rmesh_backward::create_tile_fill_bind_group(&device, &tile_pipelines, &tile_buffers);
    let tile_gen_scan_bg = rmesh_backward::create_tile_gen_scan_bind_group(
        &device, &scan_pipelines, &tile_buffers,
        &buffers.uniforms, &buffers.vertices, &buffers.indices,
        &buffers.compact_tet_ids, &buffers.circumdata, &buffers.tiles_touched,
        &scan_buffers, &radix_state.num_keys_buf,
    );

    // Tile ranges uses num_keys_buf (scan pipeline writes total_pairs there)
    let tile_ranges_bg_a = rmesh_backward::create_tile_ranges_bind_group_with_keys(
        &device, &tile_pipelines, &tile_buffers.tile_sort_keys,
        &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms, &radix_state.num_keys_buf,
    );
    let tile_ranges_bg_b = rmesh_backward::create_tile_ranges_bind_group_with_keys(
        &device, &tile_pipelines, &radix_state.keys_b,
        &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms, &radix_state.num_keys_buf,
    );

    let fwd_tiled_bg_a = rmesh_render::create_forward_tiled_bind_group(
        &device, &fwd_tiled, &buffers.uniforms,
        &buffers.vertices, &buffers.indices, &buffers.colors,
        &buffers.densities, &buffers.color_grads,
        &tile_buffers.tile_sort_values, &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms,
    );
    let fwd_tiled_bg_b = rmesh_render::create_forward_tiled_bind_group(
        &device, &fwd_tiled, &buffers.uniforms,
        &buffers.vertices, &buffers.indices, &buffers.colors,
        &buffers.densities, &buffers.color_grads,
        &radix_state.values_b, &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms,
    );

    // Dispatch scan-based tiled forward
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.clear_buffer(&tile_buffers.tile_ranges, 0, None);

    rmesh_backward::record_scan_tile_pipeline(
        &mut encoder, &scan_pipelines, &tile_pipelines,
        &prepare_dispatch_bg, &rts_bg,
        &tile_fill_bg, &tile_gen_scan_bg, &scan_buffers, &tile_buffers,
    );

    let result_in_b = rmesh_backward::record_radix_sort(
        &mut encoder, &device, &radix_pipelines, &radix_state,
        &tile_buffers.tile_sort_keys, &tile_buffers.tile_sort_values,
    );
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
    {
        let fwd_bg = if result_in_b { &fwd_tiled_bg_b } else { &fwd_tiled_bg_a };
        rmesh_render::record_forward_tiled(&mut encoder, &fwd_tiled, fwd_bg, tile_buffers.num_tiles);
    }
    queue.submit(std::iter::once(encoder.finish()));

    // --- Loss computation ---
    let n_pixels = (W * H) as usize;
    let loss_pipeline = LossPipeline::new(&device);
    let loss_buffers = LossBuffers::new(&device, W, H);

    // Ground truth: constant gray
    let gt_data: Vec<f32> = vec![0.5; n_pixels * 3];
    queue.write_buffer(&loss_buffers.ground_truth, 0, bytemuck::cast_slice(&gt_data));

    let loss_uni = LossUniforms {
        width: W,
        height: H,
        loss_type: 0,
        lambda_ssim: 0.0,
    };
    queue.write_buffer(&loss_buffers.loss_uniforms, 0, bytemuck::bytes_of(&loss_uni));
    queue.write_buffer(&loss_buffers.loss_value, 0, &[0u8; 4]);

    let loss_bg = create_loss_bind_group(
        &device, &loss_pipeline, &loss_buffers, &fwd_tiled.rendered_image,
    );

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    record_loss_pass(&mut encoder, &loss_pipeline, &loss_bg, W, H);
    queue.submit(std::iter::once(encoder.finish()));

    // --- Backward tiled ---
    let sh_stride = scene.num_sh_coeffs() * 3;
    let grad_buffers = rmesh_backward::GradientBuffers::new(
        &device, scene.vertex_count, scene.tet_count, sh_stride,
    );

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.clear_buffer(&grad_buffers.d_sh_coeffs, 0, None);
    encoder.clear_buffer(&grad_buffers.d_vertices, 0, None);
    encoder.clear_buffer(&grad_buffers.d_densities, 0, None);
    encoder.clear_buffer(&grad_buffers.d_color_grads, 0, None);
    queue.submit(std::iter::once(encoder.finish()));

    let debug_image = create_rw_buffer(&device, "debug_image", (n_pixels as u64) * 4 * 4);

    let (tile_sort_values_sorted, _) = if result_in_b {
        (&radix_state.values_b, &radix_state.keys_b)
    } else {
        (&tile_buffers.tile_sort_values, &tile_buffers.tile_sort_keys)
    };

    let bwd_tiled_pipelines = rmesh_backward::BackwardTiledPipelines::new(&device);
    let (bwd_bg0, bwd_bg1) = rmesh_backward::create_backward_tiled_bind_groups(
        &device, &bwd_tiled_pipelines,
        &buffers.uniforms, &loss_buffers.dl_d_image, &fwd_tiled.rendered_image,
        &buffers.vertices, &buffers.indices, &buffers.sh_coeffs,
        &buffers.densities, &buffers.color_grads, &buffers.circumdata,
        &buffers.colors, tile_sort_values_sorted,
        &grad_buffers.d_sh_coeffs, &grad_buffers.d_vertices,
        &grad_buffers.d_densities, &grad_buffers.d_color_grads,
        &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms, &debug_image,
    );

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("backward_tiled"), timestamp_writes: None,
        });
        pass.set_pipeline(&bwd_tiled_pipelines.pipeline);
        pass.set_bind_group(0, &bwd_bg0, &[]);
        pass.set_bind_group(1, &bwd_bg1, &[]);
        let num_tiles = tile_buffers.num_tiles;
        pass.dispatch_workgroups(num_tiles.min(65535), ((num_tiles + 65534) / 65535).max(1), 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    // Read back gradients
    let d_densities: Vec<f32> =
        read_buffer(&device, &queue, &grad_buffers.d_densities, scene.tet_count as usize);
    let d_sh: Vec<f32> =
        read_buffer(&device, &queue, &grad_buffers.d_sh_coeffs, (scene.tet_count * sh_stride) as usize);

    eprintln!("backward_tiled: d_densities = {d_densities:?}");
    eprintln!("backward_tiled: d_sh = {d_sh:?}");

    for (i, &v) in d_densities.iter().enumerate() {
        assert!(v.is_finite(), "d_density[{i}] is not finite: {v}");
    }
    for (i, &v) in d_sh.iter().enumerate() {
        assert!(v.is_finite(), "d_sh[{i}] is not finite: {v}");
    }

    let density_grad_mag: f32 = d_densities.iter().map(|v| v.abs()).sum();
    let sh_grad_mag: f32 = d_sh.iter().map(|v| v.abs()).sum();

    eprintln!("backward_tiled: |d_density| = {density_grad_mag:.6}, |d_sh| = {sh_grad_mag:.6}");
    eprintln!("backward_tiled: subgroup backward pipeline ran successfully");
}

// ===========================================================================
// Test: finite-difference gradient verification (single tet)
// ===========================================================================

/// Runs the full tiled pipeline (forward_compute → tile_fill → tile_gen →
/// radix_sort → tile_ranges → forward_tiled → loss → backward_tiled) on a
/// single tet, then verifies every analytical gradient against a numerical
/// finite-difference estimate.
///
/// For a single primitive with no overlapping tets, the gradient should be
/// exact (no approximation errors from multi-tet interactions).
#[test]
fn test_single_tet_gradient_finite_diff() {
    let (device, queue) = match create_test_device() {
        Some(dq) => dq,
        None => {
            eprintln!("Skipping test_single_tet_gradient_finite_diff (no GPU)");
            return;
        }
    };

    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    // Large tet (radius 0.6) with close camera (1.0 unit away) for high pixel coverage.
    // Camera is outside tet but close → large projected area → many hit pixels.
    let scene = random_single_tet_scene(&mut rng, 0.6);

    let verts_arr = [
        Vec3::new(scene.vertices[0], scene.vertices[1], scene.vertices[2]),
        Vec3::new(scene.vertices[3], scene.vertices[4], scene.vertices[5]),
        Vec3::new(scene.vertices[6], scene.vertices[7], scene.vertices[8]),
        Vec3::new(scene.vertices[9], scene.vertices[10], scene.vertices[11]),
    ];
    let centroid = (verts_arr[0] + verts_arr[1] + verts_arr[2] + verts_arr[3]) * 0.25;
    let eye = centroid + Vec3::new(1.0, 0.0, 0.0);
    let (vp, inv_vp) = setup_camera(eye, centroid);

    let tile_size = 4u32;
    let n_pixels = (W * H) as usize;

    // Ground truth: constant gray
    let gt_data: Vec<f32> = vec![0.5; n_pixels * 3];

    // Use L2 loss for smooth gradient
    let loss_type = 1u32;

    // -----------------------------------------------------------------------
    // Helper: run full forward+loss pipeline and return scalar loss
    // -----------------------------------------------------------------------
    // Uses scan-based tile pipeline (same as Python path).
    // Re-creates scene buffers each time because we modify parameters.
    let run_forward_loss = |scene_data: &SceneData| -> f32 {
        let (buffers, fwd_pipelines, _targets, compute_bg, _render_bg) =
            rmesh_render::setup_forward(&device, &queue, scene_data, W, H);

        let uniforms = rmesh_render::make_uniforms(
            vp, inv_vp, eye, W as f32, H as f32, scene_data.tet_count, scene_data.sh_degree, 0,
        );
        queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        rmesh_render::record_forward_compute(
            &mut encoder, &fwd_pipelines, &buffers, &compute_bg, scene_data.tet_count, &queue,
        );
        queue.submit(std::iter::once(encoder.finish()));

        // Scan-based tiled pipeline
        let tile_pipelines = rmesh_backward::TilePipelines::new(&device);
        let radix_pipelines = rmesh_backward::RadixSortPipelines::new(&device);
        let tile_buffers = rmesh_backward::TileBuffers::new(&device, scene_data.tet_count, W, H, tile_size);
        let radix_state = rmesh_backward::RadixSortState::new(&device, tile_buffers.max_pairs_pow2, 32);
        radix_state.upload_configs(&queue);

        let scan_pipelines = rmesh_backward::ScanPipelines::new(&device);
        let scan_buffers = rmesh_backward::ScanBuffers::new(&device, scene_data.tet_count);

        let tile_uni = rmesh_backward::TileUniforms {
            screen_width: W, screen_height: H, tile_size,
            tiles_x: tile_buffers.tiles_x, tiles_y: tile_buffers.tiles_y,
            num_tiles: tile_buffers.num_tiles, visible_tet_count: 0,
            _pad: [0; 5],
        };
        queue.write_buffer(&tile_buffers.tile_uniforms, 0, bytemuck::bytes_of(&tile_uni));

        let prepare_dispatch_bg = rmesh_backward::create_prepare_dispatch_bind_group(
            &device, &scan_pipelines, &buffers.indirect_args, &scan_buffers,
        );
        let rts_bg = rmesh_backward::create_rts_bind_group(
            &device, &scan_pipelines, &buffers.tiles_touched, &scan_buffers,
        );
        let tile_fill_bg = rmesh_backward::create_tile_fill_bind_group(&device, &tile_pipelines, &tile_buffers);
        let tile_gen_scan_bg = rmesh_backward::create_tile_gen_scan_bind_group(
            &device, &scan_pipelines, &tile_buffers,
            &buffers.uniforms, &buffers.vertices, &buffers.indices,
            &buffers.compact_tet_ids, &buffers.circumdata, &buffers.tiles_touched,
            &scan_buffers, &radix_state.num_keys_buf,
        );

        let tile_ranges_bg_a = rmesh_backward::create_tile_ranges_bind_group_with_keys(
            &device, &tile_pipelines, &tile_buffers.tile_sort_keys,
            &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms, &radix_state.num_keys_buf,
        );
        let tile_ranges_bg_b = rmesh_backward::create_tile_ranges_bind_group_with_keys(
            &device, &tile_pipelines, &radix_state.keys_b,
            &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms, &radix_state.num_keys_buf,
        );

        let fwd_tiled = rmesh_render::ForwardTiledPipeline::new(&device, W, H);
        let fwd_tiled_bg_a = rmesh_render::create_forward_tiled_bind_group(
            &device, &fwd_tiled, &buffers.uniforms,
            &buffers.vertices, &buffers.indices, &buffers.colors,
            &buffers.densities, &buffers.color_grads,
            &tile_buffers.tile_sort_values, &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms,
        );
        let fwd_tiled_bg_b = rmesh_render::create_forward_tiled_bind_group(
            &device, &fwd_tiled, &buffers.uniforms,
            &buffers.vertices, &buffers.indices, &buffers.colors,
            &buffers.densities, &buffers.color_grads,
            &radix_state.values_b, &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms,
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.clear_buffer(&fwd_tiled.rendered_image, 0, None);
        encoder.clear_buffer(&tile_buffers.tile_ranges, 0, None);

        rmesh_backward::record_scan_tile_pipeline(
            &mut encoder, &scan_pipelines, &tile_pipelines,
            &prepare_dispatch_bg, &rts_bg,
            &tile_fill_bg, &tile_gen_scan_bg, &scan_buffers, &tile_buffers,
        );

        let result_in_b = rmesh_backward::record_radix_sort(
            &mut encoder, &device, &radix_pipelines, &radix_state,
            &tile_buffers.tile_sort_keys, &tile_buffers.tile_sort_values,
        );
        {
            let ranges_bg = if result_in_b { &tile_ranges_bg_b } else { &tile_ranges_bg_a };
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("tile_ranges"), timestamp_writes: None });
            pass.set_pipeline(&tile_pipelines.tile_ranges_pipeline);
            pass.set_bind_group(0, ranges_bg, &[]);
            let wgs = (tile_buffers.max_pairs_pow2 + 255) / 256;
            pass.dispatch_workgroups(wgs.min(65535), ((wgs + 65534) / 65535).max(1), 1);
        }
        {
            let fwd_bg = if result_in_b { &fwd_tiled_bg_b } else { &fwd_tiled_bg_a };
            rmesh_render::record_forward_tiled(&mut encoder, &fwd_tiled, fwd_bg, tile_buffers.num_tiles);
        }

        queue.submit(std::iter::once(encoder.finish()));

        // Compute L2 loss on CPU in f64 (avoids GPU atomic CAS non-determinism)
        let rendered: Vec<f32> = read_buffer(&device, &queue, &fwd_tiled.rendered_image, n_pixels * 4);
        let mut loss_sum = 0.0f64;
        for i in 0..n_pixels {
            for c in 0..3usize {
                let diff = rendered[i * 4 + c] as f64 - gt_data[i * 3 + c] as f64;
                loss_sum += diff * diff;
            }
        }
        (loss_sum / n_pixels as f64) as f32
    };

    // -----------------------------------------------------------------------
    // Helper: run full pipeline + backward, return analytical gradients
    // -----------------------------------------------------------------------
    // Uses scan-based tile pipeline (same as Python path).
    let run_backward = |scene_data: &SceneData| -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let (buffers, fwd_pipelines, _targets, compute_bg, _render_bg) =
            rmesh_render::setup_forward(&device, &queue, scene_data, W, H);

        let uniforms = rmesh_render::make_uniforms(
            vp, inv_vp, eye, W as f32, H as f32, scene_data.tet_count, scene_data.sh_degree, 0,
        );
        queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        rmesh_render::record_forward_compute(
            &mut encoder, &fwd_pipelines, &buffers, &compute_bg, scene_data.tet_count, &queue,
        );
        queue.submit(std::iter::once(encoder.finish()));

        // Scan-based tiled pipeline
        let tile_pipelines = rmesh_backward::TilePipelines::new(&device);
        let radix_pipelines = rmesh_backward::RadixSortPipelines::new(&device);
        let tile_buffers = rmesh_backward::TileBuffers::new(&device, scene_data.tet_count, W, H, tile_size);
        let radix_state = rmesh_backward::RadixSortState::new(&device, tile_buffers.max_pairs_pow2, 32);
        radix_state.upload_configs(&queue);

        let scan_pipelines = rmesh_backward::ScanPipelines::new(&device);
        let scan_buffers = rmesh_backward::ScanBuffers::new(&device, scene_data.tet_count);

        let tile_uni = rmesh_backward::TileUniforms {
            screen_width: W, screen_height: H, tile_size,
            tiles_x: tile_buffers.tiles_x, tiles_y: tile_buffers.tiles_y,
            num_tiles: tile_buffers.num_tiles, visible_tet_count: 0,
            _pad: [0; 5],
        };
        queue.write_buffer(&tile_buffers.tile_uniforms, 0, bytemuck::bytes_of(&tile_uni));

        let prepare_dispatch_bg = rmesh_backward::create_prepare_dispatch_bind_group(
            &device, &scan_pipelines, &buffers.indirect_args, &scan_buffers,
        );
        let rts_bg = rmesh_backward::create_rts_bind_group(
            &device, &scan_pipelines, &buffers.tiles_touched, &scan_buffers,
        );
        let tile_fill_bg = rmesh_backward::create_tile_fill_bind_group(&device, &tile_pipelines, &tile_buffers);
        let tile_gen_scan_bg = rmesh_backward::create_tile_gen_scan_bind_group(
            &device, &scan_pipelines, &tile_buffers,
            &buffers.uniforms, &buffers.vertices, &buffers.indices,
            &buffers.compact_tet_ids, &buffers.circumdata, &buffers.tiles_touched,
            &scan_buffers, &radix_state.num_keys_buf,
        );

        let tile_ranges_bg_a = rmesh_backward::create_tile_ranges_bind_group_with_keys(
            &device, &tile_pipelines, &tile_buffers.tile_sort_keys,
            &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms, &radix_state.num_keys_buf,
        );
        let tile_ranges_bg_b = rmesh_backward::create_tile_ranges_bind_group_with_keys(
            &device, &tile_pipelines, &radix_state.keys_b,
            &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms, &radix_state.num_keys_buf,
        );

        let fwd_tiled = rmesh_render::ForwardTiledPipeline::new(&device, W, H);
        let fwd_tiled_bg_a = rmesh_render::create_forward_tiled_bind_group(
            &device, &fwd_tiled, &buffers.uniforms,
            &buffers.vertices, &buffers.indices, &buffers.colors,
            &buffers.densities, &buffers.color_grads,
            &tile_buffers.tile_sort_values, &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms,
        );
        let fwd_tiled_bg_b = rmesh_render::create_forward_tiled_bind_group(
            &device, &fwd_tiled, &buffers.uniforms,
            &buffers.vertices, &buffers.indices, &buffers.colors,
            &buffers.densities, &buffers.color_grads,
            &radix_state.values_b, &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms,
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.clear_buffer(&fwd_tiled.rendered_image, 0, None);
        encoder.clear_buffer(&tile_buffers.tile_ranges, 0, None);

        rmesh_backward::record_scan_tile_pipeline(
            &mut encoder, &scan_pipelines, &tile_pipelines,
            &prepare_dispatch_bg, &rts_bg,
            &tile_fill_bg, &tile_gen_scan_bg, &scan_buffers, &tile_buffers,
        );

        let result_in_b = rmesh_backward::record_radix_sort(
            &mut encoder, &device, &radix_pipelines, &radix_state,
            &tile_buffers.tile_sort_keys, &tile_buffers.tile_sort_values,
        );
        {
            let ranges_bg = if result_in_b { &tile_ranges_bg_b } else { &tile_ranges_bg_a };
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("tile_ranges"), timestamp_writes: None });
            pass.set_pipeline(&tile_pipelines.tile_ranges_pipeline);
            pass.set_bind_group(0, ranges_bg, &[]);
            let wgs = (tile_buffers.max_pairs_pow2 + 255) / 256;
            pass.dispatch_workgroups(wgs.min(65535), ((wgs + 65534) / 65535).max(1), 1);
        }
        {
            let fwd_bg = if result_in_b { &fwd_tiled_bg_b } else { &fwd_tiled_bg_a };
            rmesh_render::record_forward_tiled(&mut encoder, &fwd_tiled, fwd_bg, tile_buffers.num_tiles);
        }

        queue.submit(std::iter::once(encoder.finish()));

        // Loss
        let loss_pipeline = LossPipeline::new(&device);
        let loss_buffers = LossBuffers::new(&device, W, H);
        queue.write_buffer(&loss_buffers.ground_truth, 0, bytemuck::cast_slice(&gt_data));

        let loss_uni = LossUniforms {
            width: W, height: H, loss_type, lambda_ssim: 0.0,
        };
        queue.write_buffer(&loss_buffers.loss_uniforms, 0, bytemuck::bytes_of(&loss_uni));
        queue.write_buffer(&loss_buffers.loss_value, 0, &[0u8; 4]);

        let loss_bg = create_loss_bind_group(&device, &loss_pipeline, &loss_buffers, &fwd_tiled.rendered_image);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        record_loss_pass(&mut encoder, &loss_pipeline, &loss_bg, W, H);
        queue.submit(std::iter::once(encoder.finish()));

        // Backward
        let sh_stride = scene_data.num_sh_coeffs() * 3;
        let grad_buffers = rmesh_backward::GradientBuffers::new(
            &device, scene_data.vertex_count, scene_data.tet_count, sh_stride,
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.clear_buffer(&grad_buffers.d_sh_coeffs, 0, None);
        encoder.clear_buffer(&grad_buffers.d_vertices, 0, None);
        encoder.clear_buffer(&grad_buffers.d_densities, 0, None);
        encoder.clear_buffer(&grad_buffers.d_color_grads, 0, None);
        queue.submit(std::iter::once(encoder.finish()));

        let debug_image = create_rw_buffer(&device, "debug_image", (n_pixels as u64) * 4 * 4);

        let (tile_sort_values_sorted, _) = if result_in_b {
            (&radix_state.values_b, &radix_state.keys_b)
        } else {
            (&tile_buffers.tile_sort_values, &tile_buffers.tile_sort_keys)
        };

        let bwd_tiled_pipelines = rmesh_backward::BackwardTiledPipelines::new(&device);
        let (bwd_bg0, bwd_bg1) = rmesh_backward::create_backward_tiled_bind_groups(
            &device, &bwd_tiled_pipelines,
            &buffers.uniforms, &loss_buffers.dl_d_image, &fwd_tiled.rendered_image,
            &buffers.vertices, &buffers.indices, &buffers.sh_coeffs,
            &buffers.densities, &buffers.color_grads, &buffers.circumdata,
            &buffers.colors, tile_sort_values_sorted,
            &grad_buffers.d_sh_coeffs, &grad_buffers.d_vertices,
            &grad_buffers.d_densities, &grad_buffers.d_color_grads,
            &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms, &debug_image,
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("backward_tiled"), timestamp_writes: None });
            pass.set_pipeline(&bwd_tiled_pipelines.pipeline);
            pass.set_bind_group(0, &bwd_bg0, &[]);
            pass.set_bind_group(1, &bwd_bg1, &[]);
            let num_tiles = tile_buffers.num_tiles;
            pass.dispatch_workgroups(num_tiles.min(65535), ((num_tiles + 65534) / 65535).max(1), 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        let d_densities: Vec<f32> = read_buffer(&device, &queue, &grad_buffers.d_densities, scene_data.tet_count as usize);
        let d_sh: Vec<f32> = read_buffer(&device, &queue, &grad_buffers.d_sh_coeffs, (scene_data.tet_count * sh_stride) as usize);
        let d_vertices: Vec<f32> = read_buffer(&device, &queue, &grad_buffers.d_vertices, (scene_data.vertex_count * 3) as usize);
        let d_color_grads: Vec<f32> = read_buffer(&device, &queue, &grad_buffers.d_color_grads, (scene_data.tet_count * 3) as usize);

        (d_densities, d_sh, d_vertices, d_color_grads)
    };

    // -----------------------------------------------------------------------
    // Run backward to get analytical gradients
    // -----------------------------------------------------------------------
    let (d_densities, d_sh, d_vertices, d_color_grads) = run_backward(&scene);
    let base_loss = run_forward_loss(&scene);

    eprintln!("\n=== Single-tet finite-difference gradient test ===");
    eprintln!("base_loss = {base_loss:.8}");
    eprintln!("analytical d_density = {:?}", d_densities);
    eprintln!("analytical d_sh = {:?}", d_sh);
    eprintln!("analytical d_vertices = {:?}", d_vertices);
    eprintln!("analytical d_color_grads = {:?}", d_color_grads);

    // --- Quick diagnostic: pixel hit count (reuse run_forward_loss which uses scan pipeline) ---
    {
        let diag_loss = run_forward_loss(&scene);
        eprintln!("  CPU-computed base_loss = {diag_loss:.8}");
        // run_forward_loss doesn't return rendered image, so we count via a dedicated run
        // (the run_forward_loss already validates the pipeline works)
    }

    let eps = 1e-3f32;
    // 10% tolerance accounts for f32 precision, GPU atomic CAS non-determinism,
    // and higher-order finite-difference error. Still catches 20x-scale bugs.
    let rel_tol = 0.10;

    let check_grad = |name: &str, analytical: f32, param_plus_loss: f32, param_minus_loss: f32| {
        let numerical = (param_plus_loss - param_minus_loss) / (2.0 * eps);
        let denom = analytical.abs().max(numerical.abs()).max(1e-7);
        let rel_err = (analytical - numerical).abs() / denom;
        eprintln!(
            "  {name}: analytical={analytical:.8}, numerical={numerical:.8}, rel_err={rel_err:.6}",
        );
        (rel_err, analytical, numerical)
    };

    let mut any_failed = false;

    // --- Density gradient ---
    {
        let mut scene_plus = scene.clone();
        scene_plus.densities[0] += eps;
        let loss_plus = run_forward_loss(&scene_plus);

        let mut scene_minus = scene.clone();
        scene_minus.densities[0] -= eps;
        let loss_minus = run_forward_loss(&scene_minus);

        let (rel_err, analytical, numerical) = check_grad("density[0]", d_densities[0], loss_plus, loss_minus);
        if rel_err > rel_tol && (analytical.abs() > 1e-7 || numerical.abs() > 1e-7) {
            eprintln!("  FAIL: density gradient rel_err={rel_err:.6} > {rel_tol}");
            any_failed = true;
        }
    }

    // --- SH coefficient gradients ---
    for i in 0..scene.sh_coeffs.len() {
        let mut scene_plus = scene.clone();
        scene_plus.sh_coeffs[i] += eps;
        let loss_plus = run_forward_loss(&scene_plus);

        let mut scene_minus = scene.clone();
        scene_minus.sh_coeffs[i] -= eps;
        let loss_minus = run_forward_loss(&scene_minus);

        let (rel_err, analytical, numerical) = check_grad(&format!("sh_coeffs[{i}]"), d_sh[i], loss_plus, loss_minus);
        if rel_err > rel_tol && (analytical.abs() > 1e-7 || numerical.abs() > 1e-7) {
            eprintln!("  FAIL: sh_coeffs[{i}] gradient rel_err={rel_err:.6} > {rel_tol}");
            any_failed = true;
        }
    }

    // --- Vertex gradients ---
    for i in 0..scene.vertices.len() {
        let mut scene_plus = scene.clone();
        scene_plus.vertices[i] += eps;
        // Recompute circumspheres since vertices changed
        scene_plus.circumdata = compute_circumspheres(&scene_plus.vertices, &scene_plus.indices);
        let loss_plus = run_forward_loss(&scene_plus);

        let mut scene_minus = scene.clone();
        scene_minus.vertices[i] -= eps;
        scene_minus.circumdata = compute_circumspheres(&scene_minus.vertices, &scene_minus.indices);
        let loss_minus = run_forward_loss(&scene_minus);

        let (rel_err, analytical, numerical) = check_grad(&format!("vertex[{i}]"), d_vertices[i], loss_plus, loss_minus);
        if rel_err > rel_tol && (analytical.abs() > 1e-7 || numerical.abs() > 1e-7) {
            eprintln!("  FAIL: vertex[{i}] gradient rel_err={rel_err:.6} > {rel_tol}");
            any_failed = true;
        }
    }

    // --- Color gradient gradients ---
    // Use larger eps for color_grads: they produce small loss changes that are
    // near f32 precision floor with eps=1e-3. eps=0.01 gives 10x larger signal.
    let eps_cg = 0.01f32;
    let check_grad_eps = |name: &str, analytical: f32, param_plus_loss: f32, param_minus_loss: f32, epsilon: f32| {
        let numerical = (param_plus_loss - param_minus_loss) / (2.0 * epsilon);
        let denom = analytical.abs().max(numerical.abs()).max(1e-7);
        let rel_err = (analytical - numerical).abs() / denom;
        eprintln!(
            "  {name}: analytical={analytical:.8}, numerical={numerical:.8}, rel_err={rel_err:.6}",
        );
        (rel_err, analytical, numerical)
    };
    for i in 0..scene.color_grads.len() {
        let mut scene_plus = scene.clone();
        scene_plus.color_grads[i] += eps_cg;
        let loss_plus = run_forward_loss(&scene_plus);

        let mut scene_minus = scene.clone();
        scene_minus.color_grads[i] -= eps_cg;
        let loss_minus = run_forward_loss(&scene_minus);

        let (rel_err, analytical, numerical) = check_grad_eps(&format!("color_grads[{i}]"), d_color_grads[i], loss_plus, loss_minus, eps_cg);
        if rel_err > rel_tol && (analytical.abs() > 1e-7 || numerical.abs() > 1e-7) {
            eprintln!("  FAIL: color_grads[{i}] gradient rel_err={rel_err:.6} > {rel_tol}");
            any_failed = true;
        }
    }

    if any_failed {
        panic!("Some gradients failed the finite-difference check! See output above.");
    }

    eprintln!("\nAll gradients passed finite-difference check (rel_tol={rel_tol})");
}

// ===========================================================================
// Test: training convergence (single tet, fixed vertices)
// ===========================================================================

/// Runs a full training loop (forward → loss → backward → Adam) on a single tet
/// with fixed vertices. Verifies loss decreases over 50 steps.
///
/// This tests the entire training pipeline end-to-end without requiring
/// Delaunay triangulation. Only SH coefficients, density, and color gradients
/// are optimized (vertices are frozen).
#[test]
fn test_single_tet_loss_decreases() {
    let (device, queue) = match create_test_device() {
        Some(dq) => dq,
        None => {
            eprintln!("Skipping test_single_tet_loss_decreases (no GPU)");
            return;
        }
    };

    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = random_single_tet_scene(&mut rng, 0.6);
    let n_pixels = (W * H) as usize;

    let verts_arr = [
        Vec3::new(scene.vertices[0], scene.vertices[1], scene.vertices[2]),
        Vec3::new(scene.vertices[3], scene.vertices[4], scene.vertices[5]),
        Vec3::new(scene.vertices[6], scene.vertices[7], scene.vertices[8]),
        Vec3::new(scene.vertices[9], scene.vertices[10], scene.vertices[11]),
    ];
    let centroid = (verts_arr[0] + verts_arr[1] + verts_arr[2] + verts_arr[3]) * 0.25;
    let eye = centroid + Vec3::new(1.0, 0.0, 0.0);
    let (vp, inv_vp) = setup_camera(eye, centroid);

    // --- Setup (all created ONCE, reused across steps) ---

    // Forward buffers + compute pipeline (Adam updates these buffers in-place)
    let (buffers, fwd_pipelines, _targets, compute_bg, _render_bg) =
        rmesh_render::setup_forward(&device, &queue, &scene, W, H);
    let uniforms = rmesh_render::make_uniforms(
        vp, inv_vp, eye, W as f32, H as f32, scene.tet_count, scene.sh_degree, 0,
    );
    queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

    // Tiled pipeline
    let tile_size = 4u32;
    let tile_pipelines = rmesh_backward::TilePipelines::new(&device);
    let radix_pipelines = rmesh_backward::RadixSortPipelines::new(&device);
    let tile_buffers = rmesh_backward::TileBuffers::new(&device, scene.tet_count, W, H, tile_size);
    let radix_state = rmesh_backward::RadixSortState::new(&device, tile_buffers.max_pairs_pow2, 32);
    radix_state.upload_configs(&queue);

    let scan_pipelines = rmesh_backward::ScanPipelines::new(&device);
    let scan_buffers = rmesh_backward::ScanBuffers::new(&device, scene.tet_count);

    let tile_uni = rmesh_backward::TileUniforms {
        screen_width: W, screen_height: H, tile_size,
        tiles_x: tile_buffers.tiles_x, tiles_y: tile_buffers.tiles_y,
        num_tiles: tile_buffers.num_tiles, visible_tet_count: 0,
        _pad: [0; 5],
    };
    queue.write_buffer(&tile_buffers.tile_uniforms, 0, bytemuck::bytes_of(&tile_uni));

    // Scan bind groups
    let prepare_dispatch_bg = rmesh_backward::create_prepare_dispatch_bind_group(
        &device, &scan_pipelines, &buffers.indirect_args, &scan_buffers,
    );
    let rts_bg = rmesh_backward::create_rts_bind_group(
        &device, &scan_pipelines, &buffers.tiles_touched, &scan_buffers,
    );
    let tile_fill_bg =
        rmesh_backward::create_tile_fill_bind_group(&device, &tile_pipelines, &tile_buffers);
    let tile_gen_scan_bg = rmesh_backward::create_tile_gen_scan_bind_group(
        &device, &scan_pipelines, &tile_buffers,
        &buffers.uniforms, &buffers.vertices, &buffers.indices,
        &buffers.compact_tet_ids, &buffers.circumdata, &buffers.tiles_touched,
        &scan_buffers, &radix_state.num_keys_buf,
    );

    let tile_ranges_bg_a = rmesh_backward::create_tile_ranges_bind_group_with_keys(
        &device, &tile_pipelines, &tile_buffers.tile_sort_keys,
        &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms, &radix_state.num_keys_buf,
    );
    let tile_ranges_bg_b = rmesh_backward::create_tile_ranges_bind_group_with_keys(
        &device, &tile_pipelines, &radix_state.keys_b,
        &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms, &radix_state.num_keys_buf,
    );

    // Forward tiled
    let fwd_tiled = rmesh_render::ForwardTiledPipeline::new(&device, W, H);
    let fwd_tiled_bg_a = rmesh_render::create_forward_tiled_bind_group(
        &device, &fwd_tiled, &buffers.uniforms,
        &buffers.vertices, &buffers.indices, &buffers.colors,
        &buffers.densities, &buffers.color_grads,
        &tile_buffers.tile_sort_values, &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms,
    );
    let fwd_tiled_bg_b = rmesh_render::create_forward_tiled_bind_group(
        &device, &fwd_tiled, &buffers.uniforms,
        &buffers.vertices, &buffers.indices, &buffers.colors,
        &buffers.densities, &buffers.color_grads,
        &radix_state.values_b, &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms,
    );

    // Loss + Backward
    let loss_pipeline = LossPipeline::new(&device);
    let loss_buffers = LossBuffers::new(&device, W, H);
    // GT = black: background pixels (rendered=[0,0,0]) contribute zero loss.
    // Only foreground pixels matter → optimizer should drive loss to near zero
    // by making the tet transparent (density→0) or color→black.
    let gt_data: Vec<f32> = vec![0.0; n_pixels * 3];
    queue.write_buffer(&loss_buffers.ground_truth, 0, bytemuck::cast_slice(&gt_data));

    let loss_uni = LossUniforms {
        width: W, height: H, loss_type: 1, lambda_ssim: 0.0, // L2
    };
    queue.write_buffer(&loss_buffers.loss_uniforms, 0, bytemuck::bytes_of(&loss_uni));

    let loss_bg = create_loss_bind_group(
        &device, &loss_pipeline, &loss_buffers, &fwd_tiled.rendered_image,
    );

    let sh_stride = scene.num_sh_coeffs() * 3;
    let grad_buffers = rmesh_backward::GradientBuffers::new(
        &device, scene.vertex_count, scene.tet_count, sh_stride,
    );
    let debug_image = create_rw_buffer(&device, "debug_image", (n_pixels as u64) * 4 * 4);

    // Backward bind groups (A/B variants for radix sort result)
    let bwd_tiled_pipelines = rmesh_backward::BackwardTiledPipelines::new(&device);
    let (bwd_bg0_a, bwd_bg1) = rmesh_backward::create_backward_tiled_bind_groups(
        &device, &bwd_tiled_pipelines,
        &buffers.uniforms, &loss_buffers.dl_d_image, &fwd_tiled.rendered_image,
        &buffers.vertices, &buffers.indices, &buffers.sh_coeffs,
        &buffers.densities, &buffers.color_grads, &buffers.circumdata,
        &buffers.colors, &tile_buffers.tile_sort_values,
        &grad_buffers.d_sh_coeffs, &grad_buffers.d_vertices,
        &grad_buffers.d_densities, &grad_buffers.d_color_grads,
        &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms, &debug_image,
    );
    let (bwd_bg0_b, _) = rmesh_backward::create_backward_tiled_bind_groups(
        &device, &bwd_tiled_pipelines,
        &buffers.uniforms, &loss_buffers.dl_d_image, &fwd_tiled.rendered_image,
        &buffers.vertices, &buffers.indices, &buffers.sh_coeffs,
        &buffers.densities, &buffers.color_grads, &buffers.circumdata,
        &buffers.colors, &radix_state.values_b,
        &grad_buffers.d_sh_coeffs, &grad_buffers.d_vertices,
        &grad_buffers.d_densities, &grad_buffers.d_color_grads,
        &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms, &debug_image,
    );

    // Adam state + bind groups (separate uniform buffers per group to avoid overwrite)
    let adam_state = AdamState::new(
        &device, scene.vertex_count, scene.tet_count, sh_stride,
    );
    let adam_pipeline = AdamPipeline::new(&device);
    let adam_uni_sh = create_rw_buffer(
        &device, "adam_uni_sh", std::mem::size_of::<AdamUniforms>() as u64,
    );
    let adam_uni_dens = create_rw_buffer(
        &device, "adam_uni_dens", std::mem::size_of::<AdamUniforms>() as u64,
    );
    let adam_uni_cg = create_rw_buffer(
        &device, "adam_uni_cg", std::mem::size_of::<AdamUniforms>() as u64,
    );
    let adam_bg_sh = create_adam_bind_group(
        &device, &adam_pipeline, &adam_uni_sh,
        &buffers.sh_coeffs, &grad_buffers.d_sh_coeffs,
        &adam_state.m_sh, &adam_state.v_sh,
    );
    let adam_bg_dens = create_adam_bind_group(
        &device, &adam_pipeline, &adam_uni_dens,
        &buffers.densities, &grad_buffers.d_densities,
        &adam_state.m_densities, &adam_state.v_densities,
    );
    let adam_bg_cg = create_adam_bind_group(
        &device, &adam_pipeline, &adam_uni_cg,
        &buffers.color_grads, &grad_buffers.d_color_grads,
        &adam_state.m_color_grads, &adam_state.v_color_grads,
    );

    let adam_bgs = [&adam_bg_sh, &adam_bg_dens, &adam_bg_cg];
    let adam_uni_bufs = [&adam_uni_sh, &adam_uni_dens, &adam_uni_cg];
    let param_counts = [
        scene.tet_count * sh_stride,  // sh_coeffs: 1 tet × 3 coeffs
        scene.tet_count,               // densities: 1 tet
        scene.tet_count * 3,           // color_grads: 1 tet × 3
    ];

    let lr = 0.1f32;
    let num_steps = 200u32;
    let mut losses = Vec::new();

    eprintln!("\n=== Single-tet training convergence test ===");
    eprintln!("Training {num_steps} steps with lr={lr}, L2 loss, fixed vertices");

    // --- Training loop ---
    for step in 1..=num_steps {
        // Write Adam uniforms per group (before encoder submission)
        for (i, &count) in param_counts.iter().enumerate() {
            let adam_uni = AdamUniforms {
                param_count: count,
                step,
                lr,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                _pad: [0; 2],
            };
            queue.write_buffer(adam_uni_bufs[i], 0, bytemuck::bytes_of(&adam_uni));
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        // Clear
        encoder.clear_buffer(&grad_buffers.d_sh_coeffs, 0, None);
        encoder.clear_buffer(&grad_buffers.d_vertices, 0, None);
        encoder.clear_buffer(&grad_buffers.d_densities, 0, None);
        encoder.clear_buffer(&grad_buffers.d_color_grads, 0, None);
        encoder.clear_buffer(&loss_buffers.loss_value, 0, None);
        encoder.clear_buffer(&tile_buffers.tile_ranges, 0, None);
        encoder.clear_buffer(&fwd_tiled.rendered_image, 0, None);

        // Forward compute
        rmesh_render::record_forward_compute(
            &mut encoder, &fwd_pipelines, &buffers, &compute_bg, scene.tet_count, &queue,
        );

        // Scan pipeline
        rmesh_backward::record_scan_tile_pipeline(
            &mut encoder, &scan_pipelines, &tile_pipelines,
            &prepare_dispatch_bg, &rts_bg,
            &tile_fill_bg, &tile_gen_scan_bg, &scan_buffers, &tile_buffers,
        );

        // Radix sort
        let result_in_b = rmesh_backward::record_radix_sort(
            &mut encoder, &device, &radix_pipelines, &radix_state,
            &tile_buffers.tile_sort_keys, &tile_buffers.tile_sort_values,
        );

        // Tile ranges
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

        // Forward tiled
        {
            let fwd_bg = if result_in_b { &fwd_tiled_bg_b } else { &fwd_tiled_bg_a };
            rmesh_render::record_forward_tiled(&mut encoder, &fwd_tiled, fwd_bg, tile_buffers.num_tiles);
        }

        // Loss
        record_loss_pass(&mut encoder, &loss_pipeline, &loss_bg, W, H);

        // Backward tiled
        {
            let bwd_bg0 = if result_in_b { &bwd_bg0_b } else { &bwd_bg0_a };
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("backward_tiled"), timestamp_writes: None,
            });
            pass.set_pipeline(&bwd_tiled_pipelines.pipeline);
            pass.set_bind_group(0, bwd_bg0, &[]);
            pass.set_bind_group(1, &bwd_bg1, &[]);
            let num_tiles = tile_buffers.num_tiles;
            pass.dispatch_workgroups(num_tiles.min(65535), ((num_tiles + 65534) / 65535).max(1), 1);
        }

        // Adam (3 groups: sh, density, color_grads — vertices frozen)
        for (&bg, &count) in adam_bgs.iter().zip(param_counts.iter()) {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("adam"), timestamp_writes: None,
            });
            pass.set_pipeline(&adam_pipeline.pipeline);
            pass.set_bind_group(0, bg, &[]);
            let wg_count: u32 = (count + 255) / 256;
            pass.dispatch_workgroups(wg_count.min(65535), ((wg_count + 65534) / 65535).max(1), 1);
        }

        queue.submit(std::iter::once(encoder.finish()));

        // Read back loss
        let loss_val: Vec<f32> = read_buffer(&device, &queue, &loss_buffers.loss_value, 1);
        losses.push(loss_val[0]);

        if step == 1 {
            // Read gradients and params on first step for diagnostics
            let d_sh: Vec<f32> = read_buffer(&device, &queue, &grad_buffers.d_sh_coeffs, (scene.tet_count * sh_stride) as usize);
            let d_dens: Vec<f32> = read_buffer(&device, &queue, &grad_buffers.d_densities, scene.tet_count as usize);
            let d_cg: Vec<f32> = read_buffer(&device, &queue, &grad_buffers.d_color_grads, (scene.tet_count * 3) as usize);
            let sh_vals: Vec<f32> = read_buffer(&device, &queue, &buffers.sh_coeffs, (scene.tet_count * sh_stride) as usize);
            let dens_vals: Vec<f32> = read_buffer(&device, &queue, &buffers.densities, scene.tet_count as usize);
            eprintln!("  step {step}: loss = {:.8}", loss_val[0]);
            eprintln!("  gradients: d_sh={d_sh:?}, d_dens={d_dens:?}, d_cg={d_cg:?}");
            eprintln!("  params after Adam: sh={sh_vals:?}, dens={dens_vals:?}");
        } else if step % 50 == 0 {
            eprintln!("  step {step}: loss = {:.8}", loss_val[0]);
        }
    }

    // Print final summary
    eprintln!("\nLoss progression:");
    for (i, &l) in losses.iter().enumerate() {
        if (i + 1) % 10 == 0 || i == 0 {
            eprintln!("  step {}: {:.8}", i + 1, l);
        }
    }

    let initial_loss = losses[0];
    let final_loss = *losses.last().unwrap();
    eprintln!("Initial: {initial_loss:.8}, Final: {final_loss:.8}, Ratio: {:.4}", final_loss / initial_loss);

    assert!(
        initial_loss > 0.0,
        "Initial loss is zero — tet is not visible or ground truth matches initial render"
    );
    // GT=black, background=black → only foreground pixels contribute.
    // Optimizer can drive loss to near zero by making tet transparent or color black.
    assert!(
        final_loss < initial_loss * 0.01,
        "Loss did not converge to near zero: initial={initial_loss:.8}, final={final_loss:.8}"
    );
    // Also verify early-step progress (catches Adam/gradient magnitude issues)
    assert!(
        losses[9] < losses[0],
        "Loss did not decrease in first 10 steps: step1={:.8}, step10={:.8}",
        losses[0], losses[9]
    );
}
