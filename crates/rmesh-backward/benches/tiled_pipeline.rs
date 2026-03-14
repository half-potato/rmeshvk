//! Large-scene rendering benchmarks (2M vertices, ~10M tets).
//!
//! Measures forward and backward tiled pipeline performance at 1920x1080.
//! Requires a GPU with SUBGROUP support. Skips gracefully if unavailable.

use criterion::{criterion_group, criterion_main, Criterion};
use glam::{Mat4, Vec3};
use rmesh_util::camera::{look_at, perspective_matrix};
use rmesh_util::shared::{LossUniforms, TileUniforms};
use rmesh_util::test_util::{
    create_rw_buffer, create_timestamp_device, grid_tet_scene, print_timestamp_table,
    TimestampRecorder,
};
use rmesh_train::{create_loss_bind_group, record_loss_pass, LossBuffers, LossPipeline};

const W: u32 = 1920;
const H: u32 = 1080;
const TILE_SIZE: u32 = 16;
const GRID_SIZE: u32 = 126; // ~2M verts, ~10M tets

fn setup_camera() -> (Mat4, Mat4, Vec3) {
    let aspect = W as f32 / H as f32;
    let proj = perspective_matrix(std::f32::consts::FRAC_PI_4, aspect, 0.01, 100.0);
    let eye = Vec3::new(0.5, 0.5, 3.0);
    let target = Vec3::new(0.5, 0.5, 0.5);
    let view = look_at(eye, target, Vec3::new(0.0, 1.0, 0.0));
    let vp = proj * view;
    (vp, vp.inverse(), eye)
}

/// All GPU state needed to run a benchmark iteration.
#[allow(dead_code)]
struct BenchState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    // Scene
    tet_count: u32,
    // Forward compute
    buffers: rmesh_render::SceneBuffers,
    material: rmesh_render::MaterialBuffers,
    fwd_pipelines: rmesh_render::ForwardPipelines,
    compute_bg: wgpu::BindGroup,
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
    // Forward tiled
    fwd_tiled: rmesh_render::ForwardTiledPipeline,
    fwd_tiled_bg_a: wgpu::BindGroup,
    fwd_tiled_bg_b: wgpu::BindGroup,
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
    debug_image: wgpu::Buffer,
}

fn create_bench_state() -> Option<BenchState> {
    let (device, queue) = create_timestamp_device()?;

    eprintln!("Generating grid scene (grid_size={GRID_SIZE})...");
    let scene = grid_tet_scene(GRID_SIZE);
    eprintln!(
        "Scene: {} vertices, {} tets",
        scene.vertex_count, scene.tet_count
    );

    let (vp, inv_vp, eye) = setup_camera();

    // Forward compute
    let base_colors = vec![0.5f32; scene.tet_count as usize * 3];
    let (buffers, material, fwd_pipelines, _targets, compute_bg, _render_bg) =
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
        inv_vp,
        eye,
        W as f32,
        H as f32,
        scene.tet_count,
        0u32,
    );
    queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

    // Tiled pipeline
    let tile_pipelines = rmesh_backward::TilePipelines::new(&device);
    let radix_pipelines = rmesh_backward::RadixSortPipelines::new(&device);
    let tile_buffers =
        rmesh_backward::TileBuffers::new(&device, scene.tet_count, W, H, TILE_SIZE);
    let radix_state =
        rmesh_backward::RadixSortState::new(&device, tile_buffers.max_pairs_pow2, 32);
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
        &radix_state.num_keys_buf,
    );

    let tile_ranges_bg_a = rmesh_backward::create_tile_ranges_bind_group_with_keys(
        &device,
        &tile_pipelines,
        &tile_buffers.tile_sort_keys,
        &tile_buffers.tile_ranges,
        &tile_buffers.tile_uniforms,
        &radix_state.num_keys_buf,
    );
    let tile_ranges_bg_b = rmesh_backward::create_tile_ranges_bind_group_with_keys(
        &device,
        &tile_pipelines,
        &radix_state.keys_b,
        &tile_buffers.tile_ranges,
        &tile_buffers.tile_uniforms,
        &radix_state.num_keys_buf,
    );

    // Forward tiled
    let fwd_tiled = rmesh_render::ForwardTiledPipeline::new(&device, W, H);
    let fwd_tiled_bg_a = rmesh_render::create_forward_tiled_bind_group(
        &device,
        &fwd_tiled,
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
    let fwd_tiled_bg_b = rmesh_render::create_forward_tiled_bind_group(
        &device,
        &fwd_tiled,
        &buffers.uniforms,
        &buffers.vertices,
        &buffers.indices,
        &material.colors,
        &buffers.densities,
        &material.color_grads,
        &radix_state.values_b,
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
        &fwd_tiled.rendered_image,
    );

    // Backward tiled
    let grad_buffers =
        rmesh_backward::GradientBuffers::new(&device, scene.vertex_count, scene.tet_count);
    let mat_grad_buffers = rmesh_backward::MaterialGradBuffers::new(&device, scene.tet_count);
    let debug_image = create_rw_buffer(&device, "debug_image", (n_pixels as u64) * 4 * 4);

    let bwd_tiled_pipelines = rmesh_backward::BackwardTiledPipelines::new(&device);
    let (bwd_bg0_a, bwd_bg1) = rmesh_backward::create_backward_tiled_bind_groups(
        &device,
        &bwd_tiled_pipelines,
        &buffers.uniforms,
        &loss_buffers.dl_d_image,
        &fwd_tiled.rendered_image,
        &buffers.vertices,
        &buffers.indices,
        &buffers.densities,
        &material.color_grads,
        &buffers.circumdata,
        &material.colors,
        &tile_buffers.tile_sort_values,
        &material.base_colors,
        &grad_buffers.d_vertices,
        &grad_buffers.d_densities,
        &mat_grad_buffers.d_color_grads,
        &mat_grad_buffers.d_base_colors,
        &tile_buffers.tile_ranges,
        &tile_buffers.tile_uniforms,
        &debug_image,
    );
    let (bwd_bg0_b, _) = rmesh_backward::create_backward_tiled_bind_groups(
        &device,
        &bwd_tiled_pipelines,
        &buffers.uniforms,
        &loss_buffers.dl_d_image,
        &fwd_tiled.rendered_image,
        &buffers.vertices,
        &buffers.indices,
        &buffers.densities,
        &material.color_grads,
        &buffers.circumdata,
        &material.colors,
        &radix_state.values_b,
        &material.base_colors,
        &grad_buffers.d_vertices,
        &grad_buffers.d_densities,
        &mat_grad_buffers.d_color_grads,
        &mat_grad_buffers.d_base_colors,
        &tile_buffers.tile_ranges,
        &tile_buffers.tile_uniforms,
        &debug_image,
    );

    Some(BenchState {
        device,
        queue,
        tet_count: scene.tet_count,
        buffers,
        material,
        fwd_pipelines,
        compute_bg,
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
        fwd_tiled,
        fwd_tiled_bg_a,
        fwd_tiled_bg_b,
        loss_pipeline,
        loss_buffers,
        loss_bg,
        bwd_tiled_pipelines,
        grad_buffers,
        mat_grad_buffers,
        bwd_bg0_a,
        bwd_bg0_b,
        bwd_bg1,
        debug_image,
    })
}

/// Record and submit the full forward tiled pipeline.
fn run_forward(s: &BenchState) {
    let mut encoder = s
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    // Forward compute (visibility, tile counting)
    rmesh_render::record_forward_compute(
        &mut encoder,
        &s.fwd_pipelines,
        &s.buffers,
        &s.compute_bg,
        s.tet_count,
        &s.queue,
    );

    // Clear tile ranges + rendered image
    encoder.clear_buffer(&s.fwd_tiled.rendered_image, 0, None);
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
            &s.fwd_tiled_bg_b
        } else {
            &s.fwd_tiled_bg_a
        };
        rmesh_render::record_forward_tiled(
            &mut encoder,
            &s.fwd_tiled,
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
    rmesh_render::record_forward_compute(
        &mut encoder,
        &s.fwd_pipelines,
        &s.buffers,
        &s.compute_bg,
        s.tet_count,
        &s.queue,
    );

    // Clear buffers
    encoder.clear_buffer(&s.fwd_tiled.rendered_image, 0, None);
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
            &s.fwd_tiled_bg_b
        } else {
            &s.fwd_tiled_bg_a
        };
        rmesh_render::record_forward_tiled(
            &mut encoder,
            &s.fwd_tiled,
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
    let mut ts = TimestampRecorder::new(&s.device, &s.queue, 16);

    let mut encoder = s
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    // We can't use record_forward_compute directly since it creates its own
    // passes without timestamp writes. Instead, replicate the logic with
    // timestamps on the passes we can instrument.

    // For the scan pipeline and later stages, we instrument each pass.
    // First, run forward_compute without timestamps (it's a single pass).
    rmesh_render::record_forward_compute(
        &mut encoder,
        &s.fwd_pipelines,
        &s.buffers,
        &s.compute_bg,
        s.tet_count,
        &s.queue,
    );

    encoder.clear_buffer(&s.fwd_tiled.rendered_image, 0, None);
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
            &s.fwd_tiled_bg_b
        } else {
            &s.fwd_tiled_bg_a
        };
        let (b, e) = ts.allocate("forward_tiled");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("forward_tiled"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: ts.query_set(),
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            }),
        });
        pass.set_pipeline(s.fwd_tiled.pipeline());
        pass.set_bind_group(0, fwd_bg, &[]);
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

    // Warmup: run one forward pass before benchmarking
    run_forward(&state);

    c.bench_function("forward_tiled_2M", |b| {
        b.iter(|| run_forward(&state));
    });

    // GPU timestamp breakdown (single run after criterion)
    print_forward_timestamp_breakdown(&state);
}

fn bench_backward(c: &mut Criterion) {
    let state = match create_bench_state() {
        Some(s) => s,
        None => {
            eprintln!("Skipping bench_backward (no GPU with SUBGROUP + TIMESTAMP_QUERY)");
            return;
        }
    };

    // Warmup
    run_backward(&state);

    c.bench_function("backward_tiled_2M", |b| {
        b.iter(|| run_backward(&state));
    });
}

criterion_group! {
    name = tiled_pipeline;
    config = Criterion::default().sample_size(20);
    targets = bench_forward, bench_backward
}
criterion_main!(tiled_pipeline);
