//! Large-scene rendering benchmarks (2M vertices, ~10M tets).
//!
//! Measures forward and backward tiled pipeline performance at 1920x1080.
//! Requires a GPU with SUBGROUP support. Skips gracefully if unavailable.

use criterion::{criterion_group, criterion_main, Criterion};
use glam::{Mat4, Vec3};
use rmesh_util::camera::{look_at, perspective_matrix};
use rmesh_util::shared::{LossUniforms, TileUniforms};
use rmesh_util::test_util::{
    grid_tet_scene, print_timestamp_table,
    TimestampRecorder,
};
use rmesh_train::{create_loss_bind_group, record_loss_pass, LossBuffers, LossPipeline};

const W: u32 = 1920;
const H: u32 = 1080;
const TILE_SIZE: u32 = 8;
const GRID_SIZE: u32 = 126; // ~2M verts, ~10M tets

fn setup_camera() -> (Mat4, glam::Mat3, [f32; 4], Vec3) {
    let fov_y = std::f32::consts::FRAC_PI_4;
    let aspect = W as f32 / H as f32;
    let proj = perspective_matrix(fov_y, aspect, 0.01, 100.0);
    let eye = Vec3::new(0.5, 0.5, 3.0);
    let target = Vec3::new(0.5, 0.5, 0.5);
    let up = Vec3::new(0.0, 1.0, 0.0);
    let view = look_at(eye, target, up);
    let vp = proj * view;
    // c2w for pinhole convention (y-down, z-forward)
    let f = (target - eye).normalize();
    let r = f.cross(up).normalize();
    let u = r.cross(f);
    let c2w = glam::Mat3::from_cols(r, -u, f);
    let f_val = 1.0 / (fov_y / 2.0).tan();
    let intrinsics = [f_val * H as f32 / 2.0, f_val * H as f32 / 2.0, W as f32 / 2.0, H as f32 / 2.0];
    (vp, c2w, intrinsics, eye)
}

/// All GPU state needed to run a benchmark iteration.
#[allow(dead_code)]
struct BenchState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    // Scene
    tet_count: u32,
    // Forward project + HW rasterization
    buffers: rmesh_render::SceneBuffers,
    material: rmesh_render::MaterialBuffers,
    fwd_pipelines: rmesh_render::ForwardPipelines,
    targets: rmesh_render::RenderTargets,
    compute_bg: wgpu::BindGroup,
    render_bg: wgpu::BindGroup,
    // Tiled pipeline
    tile_pipelines: rmesh_backward::TilePipelines,
    radix_pipelines: rmesh_backward::RadixSortPipelines,
    tile_buffers: rmesh_backward::TileBuffers,
    radix_state: rmesh_backward::RadixSortState,
    scan_pipelines: rmesh_backward::ScanPipelines,
    scan_buffers: rmesh_backward::ScanBuffers,
    // Bind groups
    prepare_dispatch_bg: wgpu::BindGroup,
    rts_bg: wgpu::BindGroup,
    tile_fill_bg: wgpu::BindGroup,
    tile_gen_scan_bg: wgpu::BindGroup,
    tile_ranges_bg_a: wgpu::BindGroup,
    tile_ranges_bg_b: wgpu::BindGroup,
    // HW raster sort (for sorted forward HW pass, 32-bit keys)
    hw_radix_pipelines: rmesh_backward::RadixSortPipelines,
    hw_sort_state: rmesh_backward::RadixSortState,
    render_bg_b: wgpu::BindGroup,
    dummy_depth_view: wgpu::TextureView,
    // Forward tiled
    rasterize: rmesh_render::RasterizeComputePipeline,
    rasterize_bg_a: wgpu::BindGroup,
    rasterize_bg_b: wgpu::BindGroup,
    // Loss
    loss_pipeline: LossPipeline,
    loss_buffers: LossBuffers,
    loss_bg: wgpu::BindGroup,
    // Backward tiled
    bwd_tiled_pipelines: rmesh_backward::BackwardTiledPipelines,
    grad_buffers: rmesh_backward::GradientBuffers,
    mat_grad_buffers: rmesh_backward::MaterialGradBuffers,
    bwd_bg0_a: wgpu::BindGroup,
    bwd_bg0_b: wgpu::BindGroup,
    bwd_bg1: wgpu::BindGroup,
    // Ray trace
    rt_pipeline: rmesh_render::RayTracePipeline,
    rt_bg: wgpu::BindGroup,
}

fn create_bench_device() -> Option<(wgpu::Device, wgpu::Queue)> {
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

        let supported_limits = adapter.limits();
        let limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 20,
            ..supported_limits
        };

        adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::SUBGROUP
                    | wgpu::Features::SHADER_FLOAT32_ATOMIC
                    | wgpu::Features::TIMESTAMP_QUERY,
                required_limits: limits,
                ..Default::default()
            })
            .await
            .ok()
    })
}

fn create_bench_state() -> Option<BenchState> {
    let (device, queue) = create_bench_device()?;

    eprintln!("Generating grid scene (grid_size={GRID_SIZE})...");
    let scene = grid_tet_scene(GRID_SIZE);
    eprintln!(
        "Scene: {} vertices, {} tets",
        scene.vertex_count, scene.tet_count
    );

    let (vp, c2w, intrinsics, eye) = setup_camera();

    // Forward compute
    let base_colors = vec![0.5f32; scene.tet_count as usize * 3];
    let (buffers, material, fwd_pipelines, targets, compute_bg, render_bg) =
        rmesh_render::setup_forward(
            &device,
            &queue,
            &scene,
            &base_colors,
            &scene.color_grads,
            W,
            H,
        );

    let uniforms = rmesh_render::make_uniforms(
        vp,
        c2w,
        intrinsics,
        eye,
        W as f32,
        H as f32,
        scene.tet_count,
        0u32,
        TILE_SIZE,
        0.0,
        0,
        0.01,
        100.0,
    );
    queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

    // Tiled pipeline
    let tile_pipelines = rmesh_backward::TilePipelines::new(&device);
    let radix_pipelines = rmesh_backward::RadixSortPipelines::new(&device, 2, rmesh_backward::SortBackend::Drs);
    let tile_buffers =
        rmesh_backward::TileBuffers::new(&device, scene.tet_count, W, H, TILE_SIZE);
    let sorting_bits = rmesh_backward::sorting_bits_for_tiles(tile_buffers.num_tiles, rmesh_backward::SortBackend::Drs);
    let radix_state =
        rmesh_backward::RadixSortState::new(&device, tile_buffers.max_pairs_pow2, sorting_bits, 2, rmesh_backward::SortBackend::Drs);
    radix_state.upload_configs(&queue);

    let scan_pipelines = rmesh_backward::ScanPipelines::new(&device);
    let scan_buffers = rmesh_backward::ScanBuffers::new(&device, scene.tet_count);

    let tile_uni = TileUniforms {
        screen_width: W,
        screen_height: H,
        tile_size: TILE_SIZE,
        tiles_x: tile_buffers.tiles_x,
        tiles_y: tile_buffers.tiles_y,
        num_tiles: tile_buffers.num_tiles,
        visible_tet_count: 0,
        _pad: [0; 5],
    };
    queue.write_buffer(
        &tile_buffers.tile_uniforms,
        0,
        bytemuck::bytes_of(&tile_uni),
    );

    // Scan bind groups
    let prepare_dispatch_bg = rmesh_backward::create_prepare_dispatch_bind_group(
        &device,
        &scan_pipelines,
        &buffers.indirect_args,
        &scan_buffers,
    );
    let rts_bg = rmesh_backward::create_rts_bind_group(
        &device,
        &scan_pipelines,
        &buffers.tiles_touched,
        &scan_buffers,
    );
    let tile_fill_bg =
        rmesh_backward::create_tile_fill_bind_group(&device, &tile_pipelines, &tile_buffers);
    let tile_gen_scan_bg = rmesh_backward::create_tile_gen_scan_bind_group(
        &device,
        &scan_pipelines,
        &tile_buffers,
        &buffers.uniforms,
        &buffers.vertices,
        &buffers.indices,
        &buffers.compact_tet_ids,
        &buffers.circumdata,
        &buffers.tiles_touched,
        &scan_buffers,
        radix_state.num_keys_buf(),
    );

    let tile_ranges_bg_a = rmesh_backward::create_tile_ranges_bind_group_with_keys(
        &device,
        &tile_pipelines,
        &tile_buffers.tile_sort_keys,
        &tile_buffers.tile_ranges,
        &tile_buffers.tile_uniforms,
        radix_state.num_keys_buf(),
    );
    let tile_ranges_bg_b = rmesh_backward::create_tile_ranges_bind_group_with_keys(
        &device,
        &tile_pipelines,
        radix_state.keys_b(),
        &tile_buffers.tile_ranges,
        &tile_buffers.tile_uniforms,
        radix_state.num_keys_buf(),
    );

    // HW raster sort infrastructure (sized for tet_count, not tile pairs — 32-bit keys)
    let hw_radix_pipelines = rmesh_backward::RadixSortPipelines::new(&device, 1, rmesh_backward::SortBackend::Drs);
    let hw_n_pow2 = scene.tet_count.next_power_of_two();
    let hw_sort_state = rmesh_backward::RadixSortState::new(&device, hw_n_pow2, 32, 1, rmesh_backward::SortBackend::Drs);
    hw_sort_state.upload_configs(&queue);
    let render_bg_b = rmesh_render::create_render_bind_group_with_sort_values(
        &device, &fwd_pipelines, &buffers, &material, hw_sort_state.values_b(),
    );

    // Dummy depth texture for forward pass depth attachment
    let dummy_depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bench_dummy_depth"),
        size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let dummy_depth_view = dummy_depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

    // Forward tiled
    let rasterize = rmesh_render::RasterizeComputePipeline::new(&device, W, H, 0);
    let rasterize_bg_a = rmesh_render::create_rasterize_bind_group(
        &device,
        &rasterize,
        &buffers.uniforms,
        &buffers.vertices,
        &buffers.indices,
        &material.colors,
        &buffers.densities,
        &material.color_grads,
        &tile_buffers.tile_sort_values,
        &tile_buffers.tile_ranges,
        &tile_buffers.tile_uniforms,
    );
    let rasterize_bg_b = rmesh_render::create_rasterize_bind_group(
        &device,
        &rasterize,
        &buffers.uniforms,
        &buffers.vertices,
        &buffers.indices,
        &material.colors,
        &buffers.densities,
        &material.color_grads,
        radix_state.values_b(),
        &tile_buffers.tile_ranges,
        &tile_buffers.tile_uniforms,
    );

    // Loss
    let loss_pipeline = LossPipeline::new(&device);
    let loss_buffers = LossBuffers::new(&device, W, H);
    let n_pixels = (W * H) as usize;
    let gt_data: Vec<f32> = vec![0.5; n_pixels * 3];
    queue.write_buffer(
        &loss_buffers.ground_truth,
        0,
        bytemuck::cast_slice(&gt_data),
    );
    let loss_uni = LossUniforms {
        width: W,
        height: H,
        loss_type: 1, // L2
        lambda_ssim: 0.0,
    };
    queue.write_buffer(
        &loss_buffers.loss_uniforms,
        0,
        bytemuck::bytes_of(&loss_uni),
    );
    let loss_bg = create_loss_bind_group(
        &device,
        &loss_pipeline,
        &loss_buffers,
        &rasterize.rendered_image,
    );

    // Backward tiled
    let grad_buffers =
        rmesh_backward::GradientBuffers::new(&device, scene.vertex_count, scene.tet_count);
    let mat_grad_buffers = rmesh_backward::MaterialGradBuffers::new(&device, scene.tet_count);

    let bwd_tiled_pipelines = rmesh_backward::BackwardTiledPipelines::new(&device);
    let (bwd_bg0_a, bwd_bg1) = rmesh_backward::create_backward_tiled_bind_groups(
        &device,
        &bwd_tiled_pipelines,
        &buffers.uniforms,
        &loss_buffers.dl_d_image,
        &rasterize.rendered_image,
        &buffers.vertices,
        &buffers.indices,
        &buffers.densities,
        &material.color_grads,
        &material.colors,
        &tile_buffers.tile_sort_values,
        &grad_buffers.d_vertices,
        &grad_buffers.d_densities,
        &mat_grad_buffers.d_color_grads,
        &mat_grad_buffers.d_base_colors,
        &tile_buffers.tile_ranges,
        &tile_buffers.tile_uniforms,
    );
    let (bwd_bg0_b, _) = rmesh_backward::create_backward_tiled_bind_groups(
        &device,
        &bwd_tiled_pipelines,
        &buffers.uniforms,
        &loss_buffers.dl_d_image,
        &rasterize.rendered_image,
        &buffers.vertices,
        &buffers.indices,
        &buffers.densities,
        &material.color_grads,
        &material.colors,
        radix_state.values_b(),
        &grad_buffers.d_vertices,
        &grad_buffers.d_densities,
        &mat_grad_buffers.d_color_grads,
        &mat_grad_buffers.d_base_colors,
        &tile_buffers.tile_ranges,
        &tile_buffers.tile_uniforms,
    );

    // Ray trace
    let neighbors = rmesh_render::compute_tet_neighbors(&scene.indices, scene.tet_count as usize);
    let bvh = rmesh_render::build_boundary_bvh(
        &scene.vertices, &scene.indices, &neighbors, scene.tet_count as usize,
    );
    let rt_pipeline = rmesh_render::RayTracePipeline::new(&device, W, H, 0);
    let rt_buffers = rmesh_render::RayTraceBuffers::new(&device, &neighbors, &bvh);
    let start_tet = rmesh_render::find_containing_tet(
        &scene.vertices, &scene.indices, scene.tet_count as usize, eye,
    ).map(|t| t as i32).unwrap_or(-1);
    queue.write_buffer(&rt_buffers.start_tet, 0, bytemuck::cast_slice(&[start_tet]));
    let rt_bg = rmesh_render::create_raytrace_bind_group(&device, &rt_pipeline, &buffers, &material, &rt_buffers);

    Some(BenchState {
        device,
        queue,
        tet_count: scene.tet_count,
        buffers,
        material,
        fwd_pipelines,
        targets,
        compute_bg,
        render_bg,
        tile_pipelines,
        radix_pipelines,
        tile_buffers,
        radix_state,
        scan_pipelines,
        scan_buffers,
        prepare_dispatch_bg,
        rts_bg,
        tile_fill_bg,
        tile_gen_scan_bg,
        tile_ranges_bg_a,
        tile_ranges_bg_b,
        hw_radix_pipelines,
        hw_sort_state,
        render_bg_b,
        dummy_depth_view,
        rasterize,
        rasterize_bg_a,
        rasterize_bg_b,
        loss_pipeline,
        loss_buffers,
        loss_bg,
        bwd_tiled_pipelines,
        grad_buffers,
        mat_grad_buffers,
        bwd_bg0_a,
        bwd_bg0_b,
        bwd_bg1,
        rt_pipeline,
        rt_bg,
    })
}

/// Record and submit the full forward tiled pipeline.
fn run_forward(s: &BenchState) {
    let mut encoder = s
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    // Forward compute (visibility, tile counting)
    rmesh_render::record_project_compute(
        &mut encoder,
        &s.fwd_pipelines,
        &s.buffers,
        &s.compute_bg,
        s.tet_count,
        &s.queue,
    );

    // Clear tile ranges + rendered image
    encoder.clear_buffer(&s.rasterize.rendered_image, 0, None);
    encoder.clear_buffer(&s.tile_buffers.tile_ranges, 0, None);

    // Scan-based tile pipeline (prepare_dispatch → RTS → tile_fill → tile_gen)
    rmesh_backward::record_scan_tile_pipeline(
        &mut encoder,
        &s.scan_pipelines,
        &s.tile_pipelines,
        &s.prepare_dispatch_bg,
        &s.rts_bg,
        &s.tile_fill_bg,
        &s.tile_gen_scan_bg,
        &s.scan_buffers,
        &s.tile_buffers,
    );

    // Radix sort
    let result_in_b = rmesh_backward::record_radix_sort(
        &mut encoder,
        &s.device,
        &s.radix_pipelines,
        &s.radix_state,
        &s.tile_buffers.tile_sort_keys,
        &s.tile_buffers.tile_sort_values,
    );

    // Tile ranges
    {
        let ranges_bg = if result_in_b {
            &s.tile_ranges_bg_b
        } else {
            &s.tile_ranges_bg_a
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tile_ranges"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&s.tile_pipelines.tile_ranges_pipeline);
        pass.set_bind_group(0, ranges_bg, &[]);
        let wgs = (s.tile_buffers.max_pairs_pow2 + 255) / 256;
        pass.dispatch_workgroups(wgs.min(65535), ((wgs + 65534) / 65535).max(1), 1);
    }

    // Forward tiled rendering
    {
        let fwd_bg = if result_in_b {
            &s.rasterize_bg_b
        } else {
            &s.rasterize_bg_a
        };
        rmesh_render::record_rasterize_compute(
            &mut encoder,
            &s.rasterize,
            fwd_bg,
            s.tile_buffers.num_tiles,
        );
    }

    s.queue.submit(std::iter::once(encoder.finish()));
    let _ = s.device.poll(wgpu::PollType::wait_indefinitely());
}

/// Record and submit the full forward + loss + backward tiled pipeline.
fn run_backward(s: &BenchState) {
    let mut encoder = s
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    // Forward compute
    rmesh_render::record_project_compute(
        &mut encoder,
        &s.fwd_pipelines,
        &s.buffers,
        &s.compute_bg,
        s.tet_count,
        &s.queue,
    );

    // Clear buffers
    encoder.clear_buffer(&s.rasterize.rendered_image, 0, None);
    encoder.clear_buffer(&s.tile_buffers.tile_ranges, 0, None);
    encoder.clear_buffer(&s.grad_buffers.d_vertices, 0, None);
    encoder.clear_buffer(&s.grad_buffers.d_densities, 0, None);
    encoder.clear_buffer(&s.mat_grad_buffers.d_color_grads, 0, None);
    encoder.clear_buffer(&s.mat_grad_buffers.d_base_colors, 0, None);
    encoder.clear_buffer(&s.loss_buffers.loss_value, 0, None);

    // Scan-based tile pipeline
    rmesh_backward::record_scan_tile_pipeline(
        &mut encoder,
        &s.scan_pipelines,
        &s.tile_pipelines,
        &s.prepare_dispatch_bg,
        &s.rts_bg,
        &s.tile_fill_bg,
        &s.tile_gen_scan_bg,
        &s.scan_buffers,
        &s.tile_buffers,
    );

    // Radix sort
    let result_in_b = rmesh_backward::record_radix_sort(
        &mut encoder,
        &s.device,
        &s.radix_pipelines,
        &s.radix_state,
        &s.tile_buffers.tile_sort_keys,
        &s.tile_buffers.tile_sort_values,
    );

    // Tile ranges
    {
        let ranges_bg = if result_in_b {
            &s.tile_ranges_bg_b
        } else {
            &s.tile_ranges_bg_a
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tile_ranges"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&s.tile_pipelines.tile_ranges_pipeline);
        pass.set_bind_group(0, ranges_bg, &[]);
        let wgs = (s.tile_buffers.max_pairs_pow2 + 255) / 256;
        pass.dispatch_workgroups(wgs.min(65535), ((wgs + 65534) / 65535).max(1), 1);
    }

    // Forward tiled rendering
    {
        let fwd_bg = if result_in_b {
            &s.rasterize_bg_b
        } else {
            &s.rasterize_bg_a
        };
        rmesh_render::record_rasterize_compute(
            &mut encoder,
            &s.rasterize,
            fwd_bg,
            s.tile_buffers.num_tiles,
        );
    }

    // Loss
    record_loss_pass(&mut encoder, &s.loss_pipeline, &s.loss_bg, W, H);

    // Backward tiled
    {
        let bwd_bg0 = if result_in_b {
            &s.bwd_bg0_b
        } else {
            &s.bwd_bg0_a
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("backward_tiled"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&s.bwd_tiled_pipelines.pipeline);
        pass.set_bind_group(0, bwd_bg0, &[]);
        pass.set_bind_group(1, &s.bwd_bg1, &[]);
        let num_tiles = s.tile_buffers.num_tiles;
        pass.dispatch_workgroups(
            num_tiles.min(65535),
            ((num_tiles + 65534) / 65535).max(1),
            1,
        );
    }

    s.queue.submit(std::iter::once(encoder.finish()));
    let _ = s.device.poll(wgpu::PollType::wait_indefinitely());
}

/// Run a single forward pass with GPU timestamps and print the breakdown.
fn print_forward_timestamp_breakdown(s: &BenchState) {
    let mut ts = TimestampRecorder::new(&s.device, &s.queue, 18);

    let mut encoder = s
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    // Inline project_compute with timestamp instrumentation
    {
        let reset_cmd = rmesh_render::DrawIndirectCommand {
            vertex_count: 12,
            instance_count: 0,
            first_vertex: 0,
            first_instance: 0,
        };
        s.queue.write_buffer(&s.buffers.indirect_args, 0, bytemuck::bytes_of(&reset_cmd));
        encoder.clear_buffer(&s.buffers.tiles_touched, 0, None);

        let (b, e) = ts.allocate("project_compute");
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("project_compute"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: ts.query_set(),
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            }),
        });
        cpass.set_pipeline(&s.fwd_pipelines.compute_pipeline);
        cpass.set_bind_group(0, &s.compute_bg, &[]);
        let workgroup_size = 64u32;
        let n_pow2 = s.tet_count.next_power_of_two();
        let total_workgroups = (n_pow2 + workgroup_size - 1) / workgroup_size;
        let max_per_dim = 65535u32;
        let dispatch_x = total_workgroups.min(max_per_dim);
        let dispatch_y = (total_workgroups + max_per_dim - 1) / max_per_dim;
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    encoder.clear_buffer(&s.rasterize.rendered_image, 0, None);
    encoder.clear_buffer(&s.tile_buffers.tile_ranges, 0, None);

    // Scan tile pipeline — instrument each sub-pass
    {
        let (b, e) = ts.allocate("prepare_dispatch");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("prepare_dispatch"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: ts.query_set(),
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            }),
        });
        pass.set_pipeline(&s.scan_pipelines.prepare_dispatch_pipeline);
        pass.set_bind_group(0, &s.prepare_dispatch_bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    {
        let (b, e) = ts.allocate("rts_reduce");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rts_reduce"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: ts.query_set(),
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            }),
        });
        pass.set_pipeline(&s.scan_pipelines.rts_reduce_pipeline);
        pass.set_bind_group(0, &s.rts_bg, &[]);
        pass.dispatch_workgroups_indirect(&s.scan_buffers.dispatch_scan, 0);
    }
    {
        let (b, e) = ts.allocate("rts_spine_scan");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rts_spine_scan"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: ts.query_set(),
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            }),
        });
        pass.set_pipeline(&s.scan_pipelines.rts_spine_scan_pipeline);
        pass.set_bind_group(0, &s.rts_bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    {
        let (b, e) = ts.allocate("rts_downsweep");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rts_downsweep"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: ts.query_set(),
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            }),
        });
        pass.set_pipeline(&s.scan_pipelines.rts_downsweep_pipeline);
        pass.set_bind_group(0, &s.rts_bg, &[]);
        pass.dispatch_workgroups_indirect(&s.scan_buffers.dispatch_scan, 0);
    }
    {
        let (b, e) = ts.allocate("tile_fill");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tile_fill"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: ts.query_set(),
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            }),
        });
        pass.set_pipeline(&s.tile_pipelines.fill_pipeline);
        pass.set_bind_group(0, &s.tile_fill_bg, &[]);
        let (fx, fy) = rmesh_tile::dispatch_2d(
            (s.tile_buffers.max_pairs_pow2 + 255) / 256,
        );
        pass.dispatch_workgroups(fx, fy, 1);
    }
    {
        let (b, e) = ts.allocate("tile_gen_scan");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tile_gen_scan"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: ts.query_set(),
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            }),
        });
        pass.set_pipeline(&s.scan_pipelines.tile_gen_scan_pipeline);
        pass.set_bind_group(0, &s.tile_gen_scan_bg, &[]);
        pass.dispatch_workgroups_indirect(&s.scan_buffers.dispatch_tile_gen, 0);
    }

    // Radix sort (5 passes — treated as one unit since record_radix_sort
    // creates its own passes)
    let result_in_b = rmesh_backward::record_radix_sort(
        &mut encoder,
        &s.device,
        &s.radix_pipelines,
        &s.radix_state,
        &s.tile_buffers.tile_sort_keys,
        &s.tile_buffers.tile_sort_values,
    );

    // Tile ranges
    {
        let ranges_bg = if result_in_b {
            &s.tile_ranges_bg_b
        } else {
            &s.tile_ranges_bg_a
        };
        let (b, e) = ts.allocate("tile_ranges");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tile_ranges"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: ts.query_set(),
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            }),
        });
        pass.set_pipeline(&s.tile_pipelines.tile_ranges_pipeline);
        pass.set_bind_group(0, ranges_bg, &[]);
        let wgs = (s.tile_buffers.max_pairs_pow2 + 255) / 256;
        pass.dispatch_workgroups(wgs.min(65535), ((wgs + 65534) / 65535).max(1), 1);
    }

    // Forward tiled
    {
        let fwd_bg = if result_in_b {
            &s.rasterize_bg_b
        } else {
            &s.rasterize_bg_a
        };
        let (b, e) = ts.allocate("rasterize_compute");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rasterize_compute"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: ts.query_set(),
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            }),
        });
        pass.set_pipeline(s.rasterize.pipeline());
        pass.set_bind_group(0, fwd_bg, &[]);
        pass.set_bind_group(1, &s.rasterize.aux_bind_group, &[]);
        let (x, y) = rmesh_tile::dispatch_2d(s.tile_buffers.num_tiles);
        pass.dispatch_workgroups(x, y, 1);
    }

    ts.resolve(&mut encoder);
    s.queue.submit(std::iter::once(encoder.finish()));
    let _ = s.device.poll(wgpu::PollType::wait_indefinitely());

    let results = ts.read_results(&s.device, &s.queue);
    eprintln!("\n=== GPU Timestamp Breakdown (Forward) ===");
    print_timestamp_table(&results);
}

/// Run only the forward compute shader (projection, culling, tile counting).
fn run_project_compute(s: &BenchState) {
    let mut encoder = s
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    rmesh_render::record_project_compute(
        &mut encoder,
        &s.fwd_pipelines,
        &s.buffers,
        &s.compute_bg,
        s.tet_count,
        &s.queue,
    );

    s.queue.submit(std::iter::once(encoder.finish()));
    let _ = s.device.poll(wgpu::PollType::wait_indefinitely());
}

/// Run only the forward tiled rendering kernel (assumes tile data is populated).
fn run_rasterize_only(s: &BenchState, result_in_b: bool) {
    let mut encoder = s
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    encoder.clear_buffer(&s.rasterize.rendered_image, 0, None);

    let fwd_bg = if result_in_b {
        &s.rasterize_bg_b
    } else {
        &s.rasterize_bg_a
    };
    rmesh_render::record_rasterize_compute(
        &mut encoder,
        &s.rasterize,
        fwd_bg,
        s.tile_buffers.num_tiles,
    );

    s.queue.submit(std::iter::once(encoder.finish()));
    let _ = s.device.poll(wgpu::PollType::wait_indefinitely());
}

/// Run the full HW vertex-fragment rasterization pipeline (project_compute + sort + render pass).
fn run_forward_hw_rasterize(s: &BenchState) {
    let mut encoder = s
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    // Clear color and depth before forward pass
    {
        let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("bench_clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &s.targets.color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &s.dummy_depth_view,
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
        &s.device,
        &s.fwd_pipelines,
        &s.hw_radix_pipelines,
        &s.hw_sort_state,
        &s.buffers,
        &s.targets,
        &s.compute_bg,
        &s.render_bg,
        &s.render_bg_b,
        s.tet_count,
        &s.queue,
        &s.dummy_depth_view,
        None,
        false,
        None, None, None,
        None,
        true,
    );

    s.queue.submit(std::iter::once(encoder.finish()));
    let _ = s.device.poll(wgpu::PollType::wait_indefinitely());
}

/// Run only the loss compute kernel (assumes rendered_image is populated).
fn run_loss_only(s: &BenchState) {
    let mut encoder = s
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    encoder.clear_buffer(&s.loss_buffers.loss_value, 0, None);
    record_loss_pass(&mut encoder, &s.loss_pipeline, &s.loss_bg, W, H);

    s.queue.submit(std::iter::once(encoder.finish()));
    let _ = s.device.poll(wgpu::PollType::wait_indefinitely());
}

/// Run only the backward tiled kernel (assumes forward + loss are already done).
fn run_backward_only(s: &BenchState, result_in_b: bool) {
    let mut encoder = s
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    encoder.clear_buffer(&s.grad_buffers.d_vertices, 0, None);
    encoder.clear_buffer(&s.grad_buffers.d_densities, 0, None);
    encoder.clear_buffer(&s.mat_grad_buffers.d_color_grads, 0, None);
    encoder.clear_buffer(&s.mat_grad_buffers.d_base_colors, 0, None);

    let bwd_bg0 = if result_in_b {
        &s.bwd_bg0_b
    } else {
        &s.bwd_bg0_a
    };
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("backward_rasterize"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&s.bwd_tiled_pipelines.pipeline);
    pass.set_bind_group(0, bwd_bg0, &[]);
    pass.set_bind_group(1, &s.bwd_bg1, &[]);
    let num_tiles = s.tile_buffers.num_tiles;
    pass.dispatch_workgroups(
        num_tiles.min(65535),
        ((num_tiles + 65534) / 65535).max(1),
        1,
    );
    drop(pass);

    s.queue.submit(std::iter::once(encoder.finish()));
    let _ = s.device.poll(wgpu::PollType::wait_indefinitely());
}

fn bench_forward(c: &mut Criterion) {
    let state = match create_bench_state() {
        Some(s) => s,
        None => {
            eprintln!("Skipping bench_forward (no GPU with SUBGROUP + TIMESTAMP_QUERY)");
            return;
        }
    };

    eprintln!(
        "Tiles: {}x{} = {}, max_pairs_pow2 = {}",
        state.tile_buffers.tiles_x,
        state.tile_buffers.tiles_y,
        state.tile_buffers.num_tiles,
        state.tile_buffers.max_pairs_pow2
    );

    // Warmup: run one full forward pass to populate tile data
    run_forward(&state);

    // Full forward pipeline (compute + scan + sort + tile_ranges + tiled rendering)
    c.bench_function("forward_rasterize_2M", |b| {
        b.iter(|| run_forward(&state));
    });

    // Forward compute only (projection, culling, tile counting)
    c.bench_function("project_compute_2M", |b| {
        b.iter(|| run_project_compute(&state));
    });

    // Forward tiled rendering only (tile data already populated from warmup)
    // Determine which buffer the sort result lands in via a dry run
    let result_in_b = {
        let mut encoder = state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let r = rmesh_backward::record_radix_sort(
            &mut encoder,
            &state.device,
            &state.radix_pipelines,
            &state.radix_state,
            &state.tile_buffers.tile_sort_keys,
            &state.tile_buffers.tile_sort_values,
        );
        state.queue.submit(std::iter::once(encoder.finish()));
        let _ = state.device.poll(wgpu::PollType::wait_indefinitely());
        r
    };
    c.bench_function("rasterize_compute_2M", |b| {
        b.iter(|| run_rasterize_only(&state, result_in_b));
    });

    // HW vertex-fragment rasterization pipeline (project_compute + render pass)
    c.bench_function("forward_hw_rasterize_2M", |b| {
        b.iter(|| run_forward_hw_rasterize(&state));
    });

    // GPU timestamp breakdown (single run after criterion)
    print_forward_timestamp_breakdown(&state);
}

/// Run a single backward pass with GPU timestamps on each pass.
fn print_backward_timestamp_breakdown(s: &BenchState) {
    let mut ts = TimestampRecorder::new(&s.device, &s.queue, 22);

    let mut encoder = s
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    // Inline project_compute with timestamp instrumentation
    {
        let reset_cmd = rmesh_render::DrawIndirectCommand {
            vertex_count: 12,
            instance_count: 0,
            first_vertex: 0,
            first_instance: 0,
        };
        s.queue.write_buffer(&s.buffers.indirect_args, 0, bytemuck::bytes_of(&reset_cmd));
        encoder.clear_buffer(&s.buffers.tiles_touched, 0, None);

        let (b, e) = ts.allocate("project_compute");
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("project_compute"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: ts.query_set(),
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            }),
        });
        cpass.set_pipeline(&s.fwd_pipelines.compute_pipeline);
        cpass.set_bind_group(0, &s.compute_bg, &[]);
        let workgroup_size = 64u32;
        let n_pow2 = s.tet_count.next_power_of_two();
        let total_workgroups = (n_pow2 + workgroup_size - 1) / workgroup_size;
        let max_per_dim = 65535u32;
        let dispatch_x = total_workgroups.min(max_per_dim);
        let dispatch_y = (total_workgroups + max_per_dim - 1) / max_per_dim;
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    // Clears
    encoder.clear_buffer(&s.rasterize.rendered_image, 0, None);
    encoder.clear_buffer(&s.tile_buffers.tile_ranges, 0, None);
    encoder.clear_buffer(&s.grad_buffers.d_vertices, 0, None);
    encoder.clear_buffer(&s.grad_buffers.d_densities, 0, None);
    encoder.clear_buffer(&s.mat_grad_buffers.d_color_grads, 0, None);
    encoder.clear_buffer(&s.mat_grad_buffers.d_base_colors, 0, None);
    encoder.clear_buffer(&s.loss_buffers.loss_value, 0, None);

    // Scan tile pipeline (record without timestamps — it's fast)
    rmesh_backward::record_scan_tile_pipeline(
        &mut encoder,
        &s.scan_pipelines,
        &s.tile_pipelines,
        &s.prepare_dispatch_bg,
        &s.rts_bg,
        &s.tile_fill_bg,
        &s.tile_gen_scan_bg,
        &s.scan_buffers,
        &s.tile_buffers,
    );

    // Radix sort
    let result_in_b = rmesh_backward::record_radix_sort(
        &mut encoder,
        &s.device,
        &s.radix_pipelines,
        &s.radix_state,
        &s.tile_buffers.tile_sort_keys,
        &s.tile_buffers.tile_sort_values,
    );

    // Tile ranges
    {
        let ranges_bg = if result_in_b {
            &s.tile_ranges_bg_b
        } else {
            &s.tile_ranges_bg_a
        };
        let (b, e) = ts.allocate("tile_ranges");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tile_ranges"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: ts.query_set(),
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            }),
        });
        pass.set_pipeline(&s.tile_pipelines.tile_ranges_pipeline);
        pass.set_bind_group(0, ranges_bg, &[]);
        let wgs = (s.tile_buffers.max_pairs_pow2 + 255) / 256;
        pass.dispatch_workgroups(wgs.min(65535), ((wgs + 65534) / 65535).max(1), 1);
    }

    // Forward tiled
    {
        let fwd_bg = if result_in_b {
            &s.rasterize_bg_b
        } else {
            &s.rasterize_bg_a
        };
        let (b, e) = ts.allocate("rasterize_compute");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rasterize_compute"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: ts.query_set(),
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            }),
        });
        pass.set_pipeline(s.rasterize.pipeline());
        pass.set_bind_group(0, fwd_bg, &[]);
        pass.set_bind_group(1, &s.rasterize.aux_bind_group, &[]);
        let (x, y) = rmesh_tile::dispatch_2d(s.tile_buffers.num_tiles);
        pass.dispatch_workgroups(x, y, 1);
    }

    // Loss
    {
        let (b, e) = ts.allocate("loss");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("loss"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: ts.query_set(),
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            }),
        });
        pass.set_pipeline(&s.loss_pipeline.pipeline);
        pass.set_bind_group(0, &s.loss_bg, &[]);
        pass.dispatch_workgroups((W + 15) / 16, (H + 15) / 16, 1);
    }

    // Backward tiled
    {
        let bwd_bg0 = if result_in_b {
            &s.bwd_bg0_b
        } else {
            &s.bwd_bg0_a
        };
        let (b, e) = ts.allocate("backward_tiled");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("backward_tiled"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: ts.query_set(),
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            }),
        });
        pass.set_pipeline(&s.bwd_tiled_pipelines.pipeline);
        pass.set_bind_group(0, bwd_bg0, &[]);
        pass.set_bind_group(1, &s.bwd_bg1, &[]);
        let num_tiles = s.tile_buffers.num_tiles;
        pass.dispatch_workgroups(
            num_tiles.min(65535),
            ((num_tiles + 65534) / 65535).max(1),
            1,
        );
    }

    ts.resolve(&mut encoder);
    s.queue.submit(std::iter::once(encoder.finish()));
    let _ = s.device.poll(wgpu::PollType::wait_indefinitely());

    let results = ts.read_results(&s.device, &s.queue);
    eprintln!("\n=== GPU Timestamp Breakdown (Backward) ===");
    print_timestamp_table(&results);
}

fn bench_backward(c: &mut Criterion) {
    let state = match create_bench_state() {
        Some(s) => s,
        None => {
            eprintln!("Skipping bench_backward (no GPU with SUBGROUP + TIMESTAMP_QUERY)");
            return;
        }
    };

    // Warmup: run full forward+backward to populate all buffers
    run_backward(&state);

    // Full backward pipeline (project + scan + sort + rasterize + loss + backward)
    c.bench_function("backward_tiled_2M", |b| {
        b.iter(|| run_backward(&state));
    });

    // Determine which buffer the sort result lands in
    let result_in_b = {
        let mut encoder = state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let r = rmesh_backward::record_radix_sort(
            &mut encoder,
            &state.device,
            &state.radix_pipelines,
            &state.radix_state,
            &state.tile_buffers.tile_sort_keys,
            &state.tile_buffers.tile_sort_values,
        );
        state.queue.submit(std::iter::once(encoder.finish()));
        let _ = state.device.poll(wgpu::PollType::wait_indefinitely());
        r
    };

    // Loss compute only (assumes rendered_image populated from warmup)
    c.bench_function("loss_compute_2M", |b| {
        b.iter(|| run_loss_only(&state));
    });

    // Backward rasterize only (assumes forward + loss populated from warmup)
    c.bench_function("backward_rasterize_2M", |b| {
        b.iter(|| run_backward_only(&state, result_in_b));
    });

    // GPU timestamp breakdown (single run after criterion)
    print_backward_timestamp_breakdown(&state);
}

// ---------------------------------------------------------------------------
// Interval shading benchmark (mesh shaders — separate device)
// ---------------------------------------------------------------------------

/// GPU state for the interval shading benchmark.
/// Separate from BenchState because it needs a device with mesh shader support.
#[allow(dead_code)]
struct IntervalBenchState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    tet_count: u32,
    buffers: rmesh_render::SceneBuffers,
    material: rmesh_render::MaterialBuffers,
    fwd_pipelines: rmesh_render::ForwardPipelines,
    targets: rmesh_render::RenderTargets,
    compute_bg: wgpu::BindGroup,
    interval_pipelines: rmesh_render::IntervalPipelines,
    sort_pipelines: rmesh_backward::RadixSortPipelines,
    sort_state: rmesh_backward::RadixSortState,
    interval_render_bg_a: wgpu::BindGroup,
    interval_render_bg_b: wgpu::BindGroup,
    indirect_convert_bg: wgpu::BindGroup,
    depth_view: wgpu::TextureView,
}

fn create_interval_bench_state() -> Option<IntervalBenchState> {
    let (device, queue) = pollster::block_on(async {
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

        let adapter_features = adapter.features();
        if !adapter_features.contains(wgpu::Features::EXPERIMENTAL_MESH_SHADER) {
            return None;
        }

        // naga's MSL backend doesn't support mesh shader translation — skip on Metal
        if adapter.get_info().backend == wgpu::Backend::Metal {
            return None;
        }

        let supported_limits = adapter.limits();
        let limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 20,
            ..supported_limits
        };

        adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::SUBGROUP
                    | wgpu::Features::SHADER_FLOAT32_ATOMIC
                    | wgpu::Features::TIMESTAMP_QUERY
                    | wgpu::Features::EXPERIMENTAL_MESH_SHADER,
                required_limits: limits,
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                ..Default::default()
            })
            .await
            .ok()
    })?;

    eprintln!("Generating interval grid scene (grid_size={GRID_SIZE})...");
    let scene = grid_tet_scene(GRID_SIZE);
    eprintln!(
        "Interval scene: {} vertices, {} tets",
        scene.vertex_count, scene.tet_count
    );

    let (vp, c2w, intrinsics, eye) = setup_camera();

    let color_format = wgpu::TextureFormat::Rgba16Float;
    let base_colors = vec![0.5f32; scene.tet_count as usize * 3];
    let (buffers, material, fwd_pipelines, targets, compute_bg, _render_bg) =
        rmesh_render::setup_forward(
            &device,
            &queue,
            &scene,
            &base_colors,
            &scene.color_grads,
            W,
            H,
        );

    let uniforms = rmesh_render::make_uniforms(
        vp, c2w, intrinsics, eye,
        W as f32, H as f32,
        scene.tet_count, 0u32, 16, 0.0, 0,
        0.01, 100.0,
    );
    queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

    // Interval shading pipelines
    let interval_pipelines = rmesh_render::IntervalPipelines::new(&device, color_format);

    // Sort infrastructure (32-bit keys, 1 payload — tet-level sort)
    let n_pow2 = scene.tet_count.next_power_of_two();
    let sort_pipelines =
        rmesh_backward::RadixSortPipelines::new(&device, 1, rmesh_backward::SortBackend::Drs);
    let sort_state = rmesh_backward::RadixSortState::new(
        &device, n_pow2, 32, 1, rmesh_backward::SortBackend::Drs,
    );
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

    // Depth texture for the render pass
    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bench_interval_depth"),
        size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

    Some(IntervalBenchState {
        device,
        queue,
        tet_count: scene.tet_count,
        buffers,
        material,
        fwd_pipelines,
        targets,
        compute_bg,
        interval_pipelines,
        sort_pipelines,
        sort_state,
        interval_render_bg_a,
        interval_render_bg_b,
        indirect_convert_bg,
        depth_view,
    })
}

/// Record and submit the full interval shading pipeline.
fn run_forward_interval(s: &IntervalBenchState) {
    let mut encoder = s
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    // Clear color and depth
    {
        let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("bench_interval_clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &s.targets.color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &s.depth_view,
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
        &s.device,
        &s.fwd_pipelines,
        &s.interval_pipelines,
        &s.sort_pipelines,
        &s.sort_state,
        &s.buffers,
        &s.targets,
        &s.compute_bg,
        &s.interval_render_bg_a,
        &s.interval_render_bg_b,
        &s.indirect_convert_bg,
        s.tet_count,
        &s.queue,
        &s.depth_view,
        None,
        None,
    );

    s.queue.submit(std::iter::once(encoder.finish()));
    let _ = s.device.poll(wgpu::PollType::wait_indefinitely());
}

fn bench_interval(c: &mut Criterion) {
    let state = match create_interval_bench_state() {
        Some(s) => s,
        None => {
            eprintln!("Skipping bench_interval (no GPU with working mesh shader support)");
            return;
        }
    };

    // Warmup
    run_forward_interval(&state);

    c.bench_function("forward_interval_2M", |b| {
        b.iter(|| run_forward_interval(&state));
    });
}

// ---------------------------------------------------------------------------
// Interval tiled benchmark (compute-only, no mesh shaders required)
// ---------------------------------------------------------------------------

/// GPU state for the interval tiled benchmark.
#[allow(dead_code)]
struct IntervalTiledBenchState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    tet_count: u32,
    // Forward
    buffers: rmesh_render::SceneBuffers,
    material: rmesh_render::MaterialBuffers,
    fwd_pipelines: rmesh_render::ForwardPipelines,
    compute_bg: wgpu::BindGroup,
    // Interval tiled
    interval_buffers: rmesh_render::IntervalTiledBuffers,
    interval_gen: rmesh_render::IntervalGeneratePipeline,
    interval_gen_bg: wgpu::BindGroup,
    interval_rasterize: rmesh_render::IntervalTiledRasterizePipeline,
    interval_rasterize_bg_a: wgpu::BindGroup,
    interval_rasterize_bg_b: wgpu::BindGroup,
    // Tile infrastructure
    tile_pipelines: rmesh_backward::TilePipelines,
    radix_pipelines: rmesh_backward::RadixSortPipelines,
    tile_buffers: rmesh_backward::TileBuffers,
    radix_state: rmesh_backward::RadixSortState,
    scan_pipelines: rmesh_backward::ScanPipelines,
    scan_buffers: rmesh_backward::ScanBuffers,
    prepare_dispatch_bg: wgpu::BindGroup,
    rts_bg: wgpu::BindGroup,
    tile_fill_bg: wgpu::BindGroup,
    tile_gen_scan_bg: wgpu::BindGroup,
    tile_ranges_bg_a: wgpu::BindGroup,
    tile_ranges_bg_b: wgpu::BindGroup,
    // Loss
    loss_pipeline: LossPipeline,
    loss_buffers: LossBuffers,
    loss_bg: wgpu::BindGroup,
    // Backward interval tiled
    bwd_interval_tiled: rmesh_backward::BackwardIntervalTiledPipeline,
    interval_grad_buffers: rmesh_backward::IntervalGradBuffers,
    bwd_it_bg0_a: wgpu::BindGroup,
    bwd_it_bg0_b: wgpu::BindGroup,
    bwd_it_bg1: wgpu::BindGroup,
    // Chain-back
    chain_back: rmesh_backward::IntervalChainBackPipeline,
    chain_back_bg0: wgpu::BindGroup,
    chain_back_bg1: wgpu::BindGroup,
    // Final gradient buffers
    grad_buffers: rmesh_backward::GradientBuffers,
    mat_grad_buffers: rmesh_backward::MaterialGradBuffers,
}

fn create_interval_tiled_bench_state() -> Option<IntervalTiledBenchState> {
    let (device, queue) = create_bench_device()?;

    eprintln!("Generating interval tiled grid scene (grid_size={GRID_SIZE})...");
    let scene = grid_tet_scene(GRID_SIZE);
    eprintln!(
        "Interval tiled scene: {} vertices, {} tets",
        scene.vertex_count, scene.tet_count
    );

    let (vp, c2w, intrinsics, eye) = setup_camera();

    let base_colors = vec![0.5f32; scene.tet_count as usize * 3];
    let (buffers, material, fwd_pipelines, _targets, compute_bg, _render_bg) =
        rmesh_render::setup_forward(&device, &queue, &scene, &base_colors, &scene.color_grads, W, H);

    let uniforms = rmesh_render::make_uniforms(
        vp, c2w, intrinsics, eye, W as f32, H as f32,
        scene.tet_count, 0u32, TILE_SIZE, 0.0, 0, 0.01, 100.0,
    );
    queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

    // Interval tiled buffers + pipelines
    let interval_buffers = rmesh_render::IntervalTiledBuffers::new(&device, scene.tet_count);
    let interval_gen = rmesh_render::IntervalGeneratePipeline::new(&device);
    let interval_gen_bg = rmesh_render::create_interval_generate_bind_group(
        &device, &interval_gen, &buffers, &material, &interval_buffers,
    );

    // Tile infrastructure
    let tile_pipelines = rmesh_backward::TilePipelines::new(&device);
    let radix_pipelines = rmesh_backward::RadixSortPipelines::new(&device, 2, rmesh_backward::SortBackend::Drs);
    let tile_buffers = rmesh_backward::TileBuffers::new(&device, scene.tet_count, W, H, TILE_SIZE);
    let sorting_bits = rmesh_backward::sorting_bits_for_tiles(tile_buffers.num_tiles, rmesh_backward::SortBackend::Drs);
    let radix_state = rmesh_backward::RadixSortState::new(&device, tile_buffers.max_pairs_pow2, sorting_bits, 2, rmesh_backward::SortBackend::Drs);
    radix_state.upload_configs(&queue);

    let scan_pipelines = rmesh_backward::ScanPipelines::new(&device);
    let scan_buffers = rmesh_backward::ScanBuffers::new(&device, scene.tet_count);

    let tile_uni = TileUniforms {
        screen_width: W, screen_height: H, tile_size: TILE_SIZE,
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
        &scan_buffers, radix_state.num_keys_buf(),
    );

    let tile_ranges_bg_a = rmesh_backward::create_tile_ranges_bind_group_with_keys(
        &device, &tile_pipelines, &tile_buffers.tile_sort_keys,
        &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms, radix_state.num_keys_buf(),
    );
    let tile_ranges_bg_b = rmesh_backward::create_tile_ranges_bind_group_with_keys(
        &device, &tile_pipelines, radix_state.keys_b(),
        &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms, radix_state.num_keys_buf(),
    );

    // Interval tiled rasterize
    let interval_rasterize = rmesh_render::IntervalTiledRasterizePipeline::new(&device, W, H, 0);
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

    // Loss
    let loss_pipeline = LossPipeline::new(&device);
    let loss_buffers = LossBuffers::new(&device, W, H);
    let n_pixels = (W * H) as usize;
    let gt_data: Vec<f32> = vec![0.5; n_pixels * 3];
    queue.write_buffer(&loss_buffers.ground_truth, 0, bytemuck::cast_slice(&gt_data));
    let loss_uni = LossUniforms { width: W, height: H, loss_type: 1, lambda_ssim: 0.0 };
    queue.write_buffer(&loss_buffers.loss_uniforms, 0, bytemuck::bytes_of(&loss_uni));
    let loss_bg = create_loss_bind_group(&device, &loss_pipeline, &loss_buffers, &interval_rasterize.rendered_image);

    // Backward interval tiled
    let interval_grad_buffers = rmesh_backward::IntervalGradBuffers::new(&device, scene.tet_count, scene.vertex_count, scene.tet_count, 0);
    let grad_buffers = rmesh_backward::GradientBuffers::new(&device, scene.vertex_count, scene.tet_count);
    let mat_grad_buffers = rmesh_backward::MaterialGradBuffers::new(&device, scene.tet_count);

    let bwd_interval_tiled = rmesh_backward::BackwardIntervalTiledPipeline::new(&device, 0);
    // Zero-initialized gradient buffers for xyzd/distortion (no loss on these in bench)
    let n_px = (W as u64) * (H as u64);
    let zero_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let dl_d_xyzd = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dl_d_xyzd_zero"), size: n_px * 4 * 4, usage: zero_usage, mapped_at_creation: false,
    });
    let dl_d_distortion = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dl_d_distortion_zero"), size: n_px * 5 * 4, usage: zero_usage, mapped_at_creation: false,
    });

    let (bwd_it_bg0_a, bwd_it_bg1) = rmesh_backward::create_backward_interval_tiled_bind_groups(
        &device, &bwd_interval_tiled,
        &buffers.uniforms, &loss_buffers.dl_d_image, &interval_rasterize.rendered_image,
        &interval_buffers.interval_verts, &interval_buffers.interval_tet_data,
        &interval_buffers.interval_meta, &tile_buffers.tile_sort_values,
        &interval_grad_buffers.d_interval_verts, &interval_grad_buffers.d_interval_tet_data,
        &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms,
        &dl_d_xyzd, &dl_d_distortion,
        &interval_rasterize.xyzd_image, &interval_rasterize.distortion_image,
        &interval_rasterize.aux_data_dummy,
        &interval_rasterize.aux_data_dummy,
        &interval_grad_buffers.d_aux_data,
    );
    let (bwd_it_bg0_b, _) = rmesh_backward::create_backward_interval_tiled_bind_groups(
        &device, &bwd_interval_tiled,
        &buffers.uniforms, &loss_buffers.dl_d_image, &interval_rasterize.rendered_image,
        &interval_buffers.interval_verts, &interval_buffers.interval_tet_data,
        &interval_buffers.interval_meta, radix_state.values_b(),
        &interval_grad_buffers.d_interval_verts, &interval_grad_buffers.d_interval_tet_data,
        &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms,
        &dl_d_xyzd, &dl_d_distortion,
        &interval_rasterize.xyzd_image, &interval_rasterize.distortion_image,
        &interval_rasterize.aux_data_dummy,
        &interval_rasterize.aux_data_dummy,
        &interval_grad_buffers.d_aux_data,
    );

    // Chain-back
    let chain_back = rmesh_backward::IntervalChainBackPipeline::new(&device);
    let (chain_back_bg0, chain_back_bg1) = rmesh_backward::create_interval_chain_back_bind_groups(
        &device, &chain_back,
        &buffers.uniforms, &buffers.vertices, &buffers.indices,
        &material.color_grads, &buffers.compact_tet_ids, &buffers.indirect_args,
        &interval_buffers.interval_meta,
        &interval_grad_buffers.d_interval_verts, &interval_grad_buffers.d_interval_tet_data,
        &buffers.vertex_normals,
        &grad_buffers.d_vertices, &grad_buffers.d_densities,
        &mat_grad_buffers.d_color_grads, &mat_grad_buffers.d_base_colors,
        &interval_grad_buffers.d_vertex_normals,
    );

    Some(IntervalTiledBenchState {
        device, queue, tet_count: scene.tet_count,
        buffers, material, fwd_pipelines, compute_bg,
        interval_buffers, interval_gen, interval_gen_bg,
        interval_rasterize, interval_rasterize_bg_a, interval_rasterize_bg_b,
        tile_pipelines, radix_pipelines, tile_buffers, radix_state,
        scan_pipelines, scan_buffers,
        prepare_dispatch_bg, rts_bg, tile_fill_bg, tile_gen_scan_bg,
        tile_ranges_bg_a, tile_ranges_bg_b,
        loss_pipeline, loss_buffers, loss_bg,
        bwd_interval_tiled, interval_grad_buffers,
        bwd_it_bg0_a, bwd_it_bg0_b, bwd_it_bg1,
        chain_back, chain_back_bg0, chain_back_bg1,
        grad_buffers, mat_grad_buffers,
    })
}

fn run_forward_interval_tiled(s: &IntervalTiledBenchState) {
    let mut encoder = s.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    // Forward compute
    rmesh_render::record_project_compute(
        &mut encoder, &s.fwd_pipelines, &s.buffers, &s.compute_bg, s.tet_count, &s.queue,
    );

    // Clear
    encoder.clear_buffer(&s.interval_rasterize.rendered_image, 0, None);
    encoder.clear_buffer(&s.tile_buffers.tile_ranges, 0, None);

    // Interval generate
    rmesh_render::record_interval_generate(
        &mut encoder, &s.interval_gen, &s.interval_gen_bg, s.tet_count,
    );

    // Scan tile pipeline
    rmesh_backward::record_scan_tile_pipeline(
        &mut encoder, &s.scan_pipelines, &s.tile_pipelines,
        &s.prepare_dispatch_bg, &s.rts_bg,
        &s.tile_fill_bg, &s.tile_gen_scan_bg, &s.scan_buffers, &s.tile_buffers,
    );

    // Radix sort
    let result_in_b = rmesh_backward::record_radix_sort(
        &mut encoder, &s.device, &s.radix_pipelines, &s.radix_state,
        &s.tile_buffers.tile_sort_keys, &s.tile_buffers.tile_sort_values,
    );

    // Tile ranges
    {
        let ranges_bg = if result_in_b { &s.tile_ranges_bg_b } else { &s.tile_ranges_bg_a };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tile_ranges"), timestamp_writes: None,
        });
        pass.set_pipeline(&s.tile_pipelines.tile_ranges_pipeline);
        pass.set_bind_group(0, ranges_bg, &[]);
        let wgs = (s.tile_buffers.max_pairs_pow2 + 255) / 256;
        pass.dispatch_workgroups(wgs.min(65535), ((wgs + 65534) / 65535).max(1), 1);
    }

    // Interval tiled rasterize
    {
        let fwd_bg = if result_in_b { &s.interval_rasterize_bg_b } else { &s.interval_rasterize_bg_a };
        rmesh_render::record_interval_tiled_rasterize(
            &mut encoder, &s.interval_rasterize, fwd_bg, s.tile_buffers.num_tiles,
        );
    }

    s.queue.submit(std::iter::once(encoder.finish()));
    let _ = s.device.poll(wgpu::PollType::wait_indefinitely());
}

fn run_backward_interval_tiled(s: &IntervalTiledBenchState) {
    let mut encoder = s.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    // Forward compute
    rmesh_render::record_project_compute(
        &mut encoder, &s.fwd_pipelines, &s.buffers, &s.compute_bg, s.tet_count, &s.queue,
    );

    // Clear
    encoder.clear_buffer(&s.interval_rasterize.rendered_image, 0, None);
    encoder.clear_buffer(&s.tile_buffers.tile_ranges, 0, None);
    encoder.clear_buffer(&s.interval_grad_buffers.d_interval_verts, 0, None);
    encoder.clear_buffer(&s.interval_grad_buffers.d_interval_tet_data, 0, None);
    encoder.clear_buffer(&s.grad_buffers.d_vertices, 0, None);
    encoder.clear_buffer(&s.grad_buffers.d_densities, 0, None);
    encoder.clear_buffer(&s.mat_grad_buffers.d_color_grads, 0, None);
    encoder.clear_buffer(&s.mat_grad_buffers.d_base_colors, 0, None);
    encoder.clear_buffer(&s.loss_buffers.loss_value, 0, None);

    // Interval generate
    rmesh_render::record_interval_generate(
        &mut encoder, &s.interval_gen, &s.interval_gen_bg, s.tet_count,
    );

    // Scan tile pipeline
    rmesh_backward::record_scan_tile_pipeline(
        &mut encoder, &s.scan_pipelines, &s.tile_pipelines,
        &s.prepare_dispatch_bg, &s.rts_bg,
        &s.tile_fill_bg, &s.tile_gen_scan_bg, &s.scan_buffers, &s.tile_buffers,
    );

    // Radix sort
    let result_in_b = rmesh_backward::record_radix_sort(
        &mut encoder, &s.device, &s.radix_pipelines, &s.radix_state,
        &s.tile_buffers.tile_sort_keys, &s.tile_buffers.tile_sort_values,
    );

    // Tile ranges
    {
        let ranges_bg = if result_in_b { &s.tile_ranges_bg_b } else { &s.tile_ranges_bg_a };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tile_ranges"), timestamp_writes: None,
        });
        pass.set_pipeline(&s.tile_pipelines.tile_ranges_pipeline);
        pass.set_bind_group(0, ranges_bg, &[]);
        let wgs = (s.tile_buffers.max_pairs_pow2 + 255) / 256;
        pass.dispatch_workgroups(wgs.min(65535), ((wgs + 65534) / 65535).max(1), 1);
    }

    // Interval tiled rasterize
    {
        let fwd_bg = if result_in_b { &s.interval_rasterize_bg_b } else { &s.interval_rasterize_bg_a };
        rmesh_render::record_interval_tiled_rasterize(
            &mut encoder, &s.interval_rasterize, fwd_bg, s.tile_buffers.num_tiles,
        );
    }

    // Loss
    record_loss_pass(&mut encoder, &s.loss_pipeline, &s.loss_bg, W, H);

    // Backward interval tiled
    {
        let bwd_bg0 = if result_in_b { &s.bwd_it_bg0_b } else { &s.bwd_it_bg0_a };
        rmesh_backward::record_backward_interval_tiled(
            &mut encoder, &s.bwd_interval_tiled, bwd_bg0, &s.bwd_it_bg1, s.tile_buffers.num_tiles,
        );
    }

    // Chain-back
    rmesh_backward::record_interval_chain_back(
        &mut encoder, &s.chain_back, &s.chain_back_bg0, &s.chain_back_bg1, s.tet_count,
    );

    s.queue.submit(std::iter::once(encoder.finish()));
    let _ = s.device.poll(wgpu::PollType::wait_indefinitely());
}

fn bench_interval_tiled(c: &mut Criterion) {
    let state = match create_interval_tiled_bench_state() {
        Some(s) => s,
        None => {
            eprintln!("Skipping bench_interval_tiled (no GPU with SUBGROUP + TIMESTAMP_QUERY)");
            return;
        }
    };

    // Warmup
    run_forward_interval_tiled(&state);

    c.bench_function("forward_interval_tiled_2M", |b| {
        b.iter(|| run_forward_interval_tiled(&state));
    });

    // Full backward
    run_backward_interval_tiled(&state);

    c.bench_function("backward_interval_tiled_2M", |b| {
        b.iter(|| run_backward_interval_tiled(&state));
    });
}

// ---------------------------------------------------------------------------
// Ray trace benchmark
// ---------------------------------------------------------------------------

fn run_raytrace(s: &BenchState) {
    let mut encoder = s
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    // Forward compute (color eval)
    rmesh_render::record_project_compute(
        &mut encoder,
        &s.fwd_pipelines,
        &s.buffers,
        &s.compute_bg,
        s.tet_count,
        &s.queue,
    );

    // Clear output
    encoder.clear_buffer(&s.rt_pipeline.rendered_image, 0, None);

    // Ray trace
    rmesh_render::record_raytrace(&mut encoder, &s.rt_pipeline, &s.rt_bg, W, H);

    s.queue.submit(std::iter::once(encoder.finish()));
    let _ = s.device.poll(wgpu::PollType::wait_indefinitely());
}

fn bench_raytrace(c: &mut Criterion) {
    let state = match create_bench_state() {
        Some(s) => s,
        None => {
            eprintln!("Skipping bench_raytrace (no GPU with SUBGROUP + TIMESTAMP_QUERY)");
            return;
        }
    };

    // Warmup
    run_raytrace(&state);

    c.bench_function("raytrace_2M", |b| {
        b.iter(|| run_raytrace(&state));
    });
}

criterion_group! {
    name = tiled_pipeline;
    config = Criterion::default().sample_size(20);
    targets = bench_forward, bench_backward
}
criterion_group! {
    name = interval_pipeline;
    config = Criterion::default().sample_size(20);
    targets = bench_interval
}
criterion_group! {
    name = interval_tiled_pipeline;
    config = Criterion::default().sample_size(20);
    targets = bench_interval_tiled
}
criterion_group! {
    name = raytrace_pipeline;
    config = Criterion::default().sample_size(20);
    targets = bench_raytrace
}
criterion_main!(tiled_pipeline, interval_pipeline, interval_tiled_pipeline, raytrace_pipeline);
