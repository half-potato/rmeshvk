//! Tile-size sweep benchmark.
//!
//! Patches WGSL shaders for each tile size (8, 12, 16, 20, 24, 32),
//! builds fresh pipelines, and measures forward+backward GPU time.
//!
//! Usage:
//!   cargo run -p rmesh-backward --release --example tile_size_sweep

use glam::{Mat4, Vec3};
use rmesh_util::camera::{look_at, perspective_matrix};
use rmesh_util::shared::{LossUniforms, TileUniforms};
use rmesh_util::test_util::{
    create_timestamp_device, grid_tet_scene, TimestampRecorder,
};
use rmesh_train::{create_loss_bind_group, LossBuffers, LossPipeline};

const W: u32 = 1920;
const H: u32 = 1080;
const GRID_SIZE: u32 = 126; // ~2M verts, ~10M tets
const WARMUP_ITERS: u32 = 3;
const BENCH_ITERS: u32 = 5;

const TILE_SIZES: &[u32] = &[8, 12, 16, 20, 24, 32];

// Shader sources (base tile size = 16)
const PROJECT_COMPUTE_SRC: &str = include_str!("../../../crates/rmesh-render/src/wgsl/project_compute.wgsl");
const RASTERIZE_COMPUTE_SRC: &str = include_str!("../../../crates/rmesh-render/src/wgsl/rasterize_compute.wgsl");
const BWD_TILED_SRC: &str = include_str!("../../../crates/rmesh-backward/src/wgsl/backward_tiled_compute.wgsl");

// ---------------------------------------------------------------------------
// Shader patching
// ---------------------------------------------------------------------------

/// Patch a tiled shader (forward_tiled or backward_tiled) from tile_size=16 to `ts`.
fn patch_tiled_shader(src: &str, ts: u32) -> String {
    let ts_sq = ts * ts;
    let ts_m1 = ts as i32 - 1;

    src
        // Shared memory declarations
        .replace(
            "array<vec4<f32>, 256>",
            &format!("array<vec4<f32>, {ts_sq}>"),
        )
        .replace(
            "array<f32, 256>",
            &format!("array<f32, {ts_sq}>"),
        )
        .replace("array<i32, 16>", &format!("array<i32, {ts}>"))
        .replace("array<u32, 17>", &format!("array<u32, {}>", ts + 1))
        // Tile offset: tile_x * 16u, tile_y * 16u
        .replace("* 16u", &format!("* {ts}u"))
        // Pixel loop bounds: i < 256u
        .replace("< 256u", &format!("< {ts_sq}u"))
        // Row/lane bounds: lane < 16u, r < 16u
        .replace("< 16u", &format!("< {ts}u"))
        // Prefix sum read: sm_prefix[16u]
        .replace("sm_prefix[16u]", &format!("sm_prefix[{ts}u]"))
        // Pixel index: row * 16u + col → already covered by "* 16u" above
        // Write-out: i / 16u, i % 16u
        .replace("/ 16u", &format!("/ {ts}u"))
        .replace("% 16u", &format!("% {ts}u"))
        // Max column bound: min(..., 15)
        .replace(", 15)", &format!(", {ts_m1})"))
}

/// Patch project_compute.wgsl tile_size constant.
fn patch_project_compute(src: &str, ts: u32) -> String {
    src.replace(
        "let tile_size = 16.0;",
        &format!("let tile_size = {ts}.0;"),
    )
}

// ---------------------------------------------------------------------------
// Pipeline creation from patched sources
// ---------------------------------------------------------------------------

/// Create a compute pipeline from WGSL source with a given bind group layout.
fn create_compute_pipeline(
    device: &wgpu::Device,
    source: &str,
    label: &str,
    layout: &wgpu::PipelineLayout,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Build the forward compute pipeline layout (13 bindings, matching ForwardPipelines).
fn fwd_compute_layout(device: &wgpu::Device) -> (wgpu::BindGroupLayout, wgpu::PipelineLayout) {
    let read_only = [
        true, true, true, true, true, true,   // 0-5 read-only
        false, false, false, false,            // 6-9 read-write
        false, false,                          // 10-11 read-write
        true,                                  // 12 read-only (base_colors)
    ];
    let entries: Vec<wgpu::BindGroupLayoutEntry> = read_only
        .iter()
        .enumerate()
        .map(|(i, &ro)| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: ro },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fwd_compute_bgl"),
        entries: &entries,
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("fwd_compute_pl"),
        bind_group_layouts: &[&bgl],
        immediate_size: 0,
    });
    (bgl, pl)
}

/// Build the forward tiled pipeline layout (10 bindings, 1 group).
fn rasterize_layout(device: &wgpu::Device) -> (wgpu::BindGroupLayout, wgpu::PipelineLayout) {
    let read_only = [true, true, true, true, true, true, true, true, true, false];
    let entries: Vec<wgpu::BindGroupLayoutEntry> = read_only
        .iter()
        .enumerate()
        .map(|(i, &ro)| rmesh_tile::storage_entry(i as u32, ro))
        .collect();
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("rasterize_bgl"),
        entries: &entries,
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("rasterize_pl"),
        bind_group_layouts: &[&bgl],
        immediate_size: 0,
    });
    (bgl, pl)
}

/// Build the backward tiled pipeline layout (2 groups: 9 + 6 bindings).
fn bwd_tiled_layout(
    device: &wgpu::Device,
) -> (wgpu::BindGroupLayout, wgpu::BindGroupLayout, wgpu::PipelineLayout) {
    let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bwd_tiled_bgl_0"),
        entries: &(0..9)
            .map(|i| rmesh_tile::storage_entry(i, true))
            .collect::<Vec<_>>(),
    });
    let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bwd_tiled_bgl_1"),
        entries: &[
            rmesh_tile::storage_entry(0, false),
            rmesh_tile::storage_entry(1, false),
            rmesh_tile::storage_entry(2, false),
            rmesh_tile::storage_entry(3, true),
            rmesh_tile::storage_entry(4, true),
            rmesh_tile::storage_entry(5, false),
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bwd_tiled_pl"),
        bind_group_layouts: &[&bgl0, &bgl1],
        immediate_size: 0,
    });
    (bgl0, bgl1, pl)
}

// ---------------------------------------------------------------------------
// Per-tile-size state
// ---------------------------------------------------------------------------

struct TileSizeState {
    tile_size: u32,
    // Patched pipelines
    fwd_compute_pipeline: wgpu::ComputePipeline,
    rasterize_pipeline: wgpu::ComputePipeline,
    bwd_tiled_pipeline: wgpu::ComputePipeline,
    // Tile infrastructure
    tile_buffers: rmesh_backward::TileBuffers,
    radix_state: rmesh_backward::RadixSortState,
    // Bind groups
    fwd_compute_bg: wgpu::BindGroup,
    tile_fill_bg: wgpu::BindGroup,
    tile_gen_scan_bg: wgpu::BindGroup,
    tile_ranges_bg_a: wgpu::BindGroup,
    tile_ranges_bg_b: wgpu::BindGroup,
    rasterize_bg_a: wgpu::BindGroup,
    rasterize_bg_b: wgpu::BindGroup,
    bwd_bg0_a: wgpu::BindGroup,
    bwd_bg0_b: wgpu::BindGroup,
    bwd_bg1: wgpu::BindGroup,
    loss_bg: wgpu::BindGroup,
    // Rendered image
    rendered_image: wgpu::Buffer,
}

// ---------------------------------------------------------------------------
// Shared state (device, scene, scan pipelines — tile-size-independent)
// ---------------------------------------------------------------------------

struct SharedState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    tet_count: u32,
    scene_buffers: rmesh_render::SceneBuffers,
    material_buffers: rmesh_render::MaterialBuffers,
    // Scan pipelines (tile-size-independent)
    scan_pipelines: rmesh_backward::ScanPipelines,
    scan_buffers: rmesh_backward::ScanBuffers,
    tile_pipelines: rmesh_backward::TilePipelines,
    radix_pipelines: rmesh_backward::RadixSortPipelines,
    prepare_dispatch_bg: wgpu::BindGroup,
    rts_bg: wgpu::BindGroup,
    // Gradient buffers
    grad_buffers: rmesh_backward::GradientBuffers,
    mat_grad_buffers: rmesh_backward::MaterialGradBuffers,
    // Loss
    loss_pipeline: LossPipeline,
    loss_buffers: LossBuffers,
}

fn create_shared_state() -> Option<SharedState> {
    let (device, queue) = create_timestamp_device()?;

    eprintln!("Generating grid scene (grid_size={GRID_SIZE})...");
    let scene = grid_tet_scene(GRID_SIZE);
    eprintln!(
        "Scene: {} vertices, {} tets",
        scene.vertex_count, scene.tet_count
    );

    let (vp, inv_vp, eye) = setup_camera();

    let scene_buffers = rmesh_render::SceneBuffers::upload(&device, &queue, &scene);
    let base_colors = vec![0.5f32; scene.tet_count as usize * 3];
    let material_buffers = rmesh_render::MaterialBuffers::upload(
        &device,
        &base_colors,
        &scene.color_grads,
        scene.tet_count,
    );

    let uniforms = rmesh_render::make_uniforms(
        vp, inv_vp, eye, W as f32, H as f32, scene.tet_count, 0u32,
    );
    queue.write_buffer(&scene_buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

    // Scan pipelines (shared across tile sizes)
    let scan_pipelines = rmesh_backward::ScanPipelines::new(&device);
    let scan_buffers = rmesh_backward::ScanBuffers::new(&device, scene.tet_count);
    let tile_pipelines = rmesh_backward::TilePipelines::new(&device);
    let radix_pipelines = rmesh_backward::RadixSortPipelines::new(&device);

    let prepare_dispatch_bg = rmesh_backward::create_prepare_dispatch_bind_group(
        &device,
        &scan_pipelines,
        &scene_buffers.indirect_args,
        &scan_buffers,
    );
    let rts_bg = rmesh_backward::create_rts_bind_group(
        &device,
        &scan_pipelines,
        &scene_buffers.tiles_touched,
        &scan_buffers,
    );

    // Gradient buffers
    let grad_buffers =
        rmesh_backward::GradientBuffers::new(&device, scene.vertex_count, scene.tet_count);
    let mat_grad_buffers = rmesh_backward::MaterialGradBuffers::new(&device, scene.tet_count);

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

    Some(SharedState {
        device,
        queue,
        tet_count: scene.tet_count,
        scene_buffers,
        material_buffers,
        scan_pipelines,
        scan_buffers,
        tile_pipelines,
        radix_pipelines,
        prepare_dispatch_bg,
        rts_bg,
        grad_buffers,
        mat_grad_buffers,
        loss_pipeline,
        loss_buffers,
    })
}

fn create_tile_size_state(shared: &SharedState, tile_size: u32) -> TileSizeState {
    let device = &shared.device;
    let queue = &shared.queue;
    let n_pixels = (W as u64) * (H as u64);

    // Patch shaders
    let fwd_compute_src = patch_project_compute(PROJECT_COMPUTE_SRC, tile_size);
    let rasterize_src = patch_tiled_shader(RASTERIZE_COMPUTE_SRC, tile_size);
    let bwd_tiled_src = patch_tiled_shader(BWD_TILED_SRC, tile_size);

    // Create pipeline layouts
    let (fwd_compute_bgl, fwd_compute_pl) = fwd_compute_layout(device);
    let (rasterize_bgl, rasterize_pl) = rasterize_layout(device);
    let (bwd_bgl0, bwd_bgl1, bwd_pl) = bwd_tiled_layout(device);

    // Create pipelines from patched sources
    let fwd_compute_pipeline = create_compute_pipeline(
        device,
        &fwd_compute_src,
        &format!("fwd_compute_ts{tile_size}"),
        &fwd_compute_pl,
    );
    let rasterize_pipeline = create_compute_pipeline(
        device,
        &rasterize_src,
        &format!("rasterize_ts{tile_size}"),
        &rasterize_pl,
    );
    let bwd_tiled_pipeline = create_compute_pipeline(
        device,
        &bwd_tiled_src,
        &format!("bwd_tiled_ts{tile_size}"),
        &bwd_pl,
    );

    // Tile buffers for this tile size
    let tile_buffers = rmesh_backward::TileBuffers::new(
        device,
        shared.tet_count,
        W,
        H,
        tile_size,
    );
    let radix_state = rmesh_backward::RadixSortState::new(
        device,
        tile_buffers.max_pairs_pow2,
        32,
    );
    radix_state.upload_configs(queue);

    let tile_uni = TileUniforms {
        screen_width: W,
        screen_height: H,
        tile_size,
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

    // Forward compute bind group (uses the same layout as ForwardPipelines)
    let sb = &shared.scene_buffers;
    let mb = &shared.material_buffers;
    let fwd_compute_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fwd_compute_bg"),
        layout: &fwd_compute_bgl,
        entries: &[
            buf_entry(0, &sb.uniforms),
            buf_entry(1, &sb.vertices),
            buf_entry(2, &sb.indices),
            buf_entry(3, &sb.densities),
            buf_entry(4, &mb.color_grads),
            buf_entry(5, &sb.circumdata),
            buf_entry(6, &mb.colors),
            buf_entry(7, &sb.sort_keys),
            buf_entry(8, &sb.sort_values),
            buf_entry(9, &sb.indirect_args),
            buf_entry(10, &sb.tiles_touched),
            buf_entry(11, &sb.compact_tet_ids),
            buf_entry(12, &mb.base_colors),
        ],
    });

    // Tile fill bind group
    let tile_fill_bg = rmesh_backward::create_tile_fill_bind_group(
        device,
        &shared.tile_pipelines,
        &tile_buffers,
    );

    // Tile gen scan bind group
    let tile_gen_scan_bg = rmesh_backward::create_tile_gen_scan_bind_group(
        device,
        &shared.scan_pipelines,
        &tile_buffers,
        &sb.uniforms,
        &sb.vertices,
        &sb.indices,
        &sb.compact_tet_ids,
        &sb.circumdata,
        &sb.tiles_touched,
        &shared.scan_buffers,
        &radix_state.num_keys_buf,
    );

    // Tile ranges bind groups (A and B)
    let tile_ranges_bg_a = rmesh_backward::create_tile_ranges_bind_group_with_keys(
        device,
        &shared.tile_pipelines,
        &tile_buffers.tile_sort_keys,
        &tile_buffers.tile_ranges,
        &tile_buffers.tile_uniforms,
        &radix_state.num_keys_buf,
    );
    let tile_ranges_bg_b = rmesh_backward::create_tile_ranges_bind_group_with_keys(
        device,
        &shared.tile_pipelines,
        &radix_state.keys_b,
        &tile_buffers.tile_ranges,
        &tile_buffers.tile_uniforms,
        &radix_state.num_keys_buf,
    );

    // Rendered image buffer
    let rendered_image = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rendered_image"),
        size: n_pixels * 4 * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Forward tiled bind groups (A and B)
    let rasterize_bg_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rasterize_bg_a"),
        layout: &rasterize_bgl,
        entries: &[
            buf_entry(0, &sb.uniforms),
            buf_entry(1, &sb.vertices),
            buf_entry(2, &sb.indices),
            buf_entry(3, &mb.colors),
            buf_entry(4, &sb.densities),
            buf_entry(5, &mb.color_grads),
            buf_entry(6, &tile_buffers.tile_sort_values),
            buf_entry(7, &tile_buffers.tile_ranges),
            buf_entry(8, &tile_buffers.tile_uniforms),
            buf_entry(9, &rendered_image),
        ],
    });
    let rasterize_bg_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rasterize_bg_b"),
        layout: &rasterize_bgl,
        entries: &[
            buf_entry(0, &sb.uniforms),
            buf_entry(1, &sb.vertices),
            buf_entry(2, &sb.indices),
            buf_entry(3, &mb.colors),
            buf_entry(4, &sb.densities),
            buf_entry(5, &mb.color_grads),
            buf_entry(6, &radix_state.values_b),
            buf_entry(7, &tile_buffers.tile_ranges),
            buf_entry(8, &tile_buffers.tile_uniforms),
            buf_entry(9, &rendered_image),
        ],
    });

    // Loss bind group
    let loss_bg = create_loss_bind_group(
        device,
        &shared.loss_pipeline,
        &shared.loss_buffers,
        &rendered_image,
    );

    // Backward tiled bind groups (A and B)
    let bwd_bg0_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bwd_bg0_a"),
        layout: &bwd_bgl0,
        entries: &[
            buf_entry(0, &sb.uniforms),
            buf_entry(1, &shared.loss_buffers.dl_d_image),
            buf_entry(2, &rendered_image),
            buf_entry(3, &sb.vertices),
            buf_entry(4, &sb.indices),
            buf_entry(5, &sb.densities),
            buf_entry(6, &mb.color_grads),
            buf_entry(7, &mb.colors),
            buf_entry(8, &tile_buffers.tile_sort_values),
        ],
    });
    let bwd_bg0_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bwd_bg0_b"),
        layout: &bwd_bgl0,
        entries: &[
            buf_entry(0, &sb.uniforms),
            buf_entry(1, &shared.loss_buffers.dl_d_image),
            buf_entry(2, &rendered_image),
            buf_entry(3, &sb.vertices),
            buf_entry(4, &sb.indices),
            buf_entry(5, &sb.densities),
            buf_entry(6, &mb.color_grads),
            buf_entry(7, &mb.colors),
            buf_entry(8, &radix_state.values_b),
        ],
    });
    let bwd_bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bwd_bg1"),
        layout: &bwd_bgl1,
        entries: &[
            buf_entry(0, &shared.grad_buffers.d_vertices),
            buf_entry(1, &shared.grad_buffers.d_densities),
            buf_entry(2, &shared.mat_grad_buffers.d_color_grads),
            buf_entry(3, &tile_buffers.tile_ranges),
            buf_entry(4, &tile_buffers.tile_uniforms),
            buf_entry(5, &shared.mat_grad_buffers.d_base_colors),
        ],
    });

    TileSizeState {
        tile_size,
        fwd_compute_pipeline,
        rasterize_pipeline,
        bwd_tiled_pipeline,
        tile_buffers,
        radix_state,
        fwd_compute_bg,
        tile_fill_bg,
        tile_gen_scan_bg,
        tile_ranges_bg_a,
        tile_ranges_bg_b,
        rasterize_bg_a,
        rasterize_bg_b,
        bwd_bg0_a,
        bwd_bg0_b,
        bwd_bg1,
        loss_bg,
        rendered_image,
    }
}

fn buf_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn setup_camera() -> (Mat4, Mat4, Vec3) {
    let aspect = W as f32 / H as f32;
    let proj = perspective_matrix(std::f32::consts::FRAC_PI_4, aspect, 0.01, 100.0);
    let eye = Vec3::new(0.5, 0.5, 3.0);
    let target = Vec3::new(0.5, 0.5, 0.5);
    let view = look_at(eye, target, Vec3::new(0.0, 1.0, 0.0));
    let vp = proj * view;
    (vp, vp.inverse(), eye)
}

// ---------------------------------------------------------------------------
// Run pipeline
// ---------------------------------------------------------------------------

fn record_forward_backward(
    shared: &SharedState,
    ts: &TileSizeState,
    encoder: &mut wgpu::CommandEncoder,
    recorder: &mut Option<&mut TimestampRecorder>,
) {
    let device = &shared.device;
    let sb = &shared.scene_buffers;

    // Reset indirect args
    let reset_cmd = rmesh_render::DrawIndirectCommand {
        vertex_count: 12,
        instance_count: 0,
        first_vertex: 0,
        first_instance: 0,
    };
    shared
        .queue
        .write_buffer(&sb.indirect_args, 0, bytemuck::bytes_of(&reset_cmd));
    encoder.clear_buffer(&sb.tiles_touched, 0, None);

    // Forward compute (patched for this tile size)
    {
        let tw = recorder.as_mut().map(|r| {
            let (b, e) = r.allocate("project_compute");
            wgpu::ComputePassTimestampWrites {
                query_set: r.query_set(),
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            }
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("project_compute"),
            timestamp_writes: tw,
        });
        pass.set_pipeline(&ts.fwd_compute_pipeline);
        pass.set_bind_group(0, &ts.fwd_compute_bg, &[]);
        let wg_size = 64u32;
        let n_pow2 = shared.tet_count.next_power_of_two();
        let total_wg = (n_pow2 + wg_size - 1) / wg_size;
        let (x, y) = rmesh_tile::dispatch_2d(total_wg);
        pass.dispatch_workgroups(x, y, 1);
    }

    // Clear buffers
    encoder.clear_buffer(&ts.rendered_image, 0, None);
    encoder.clear_buffer(&ts.tile_buffers.tile_ranges, 0, None);
    encoder.clear_buffer(&shared.grad_buffers.d_vertices, 0, None);
    encoder.clear_buffer(&shared.grad_buffers.d_densities, 0, None);
    encoder.clear_buffer(&shared.mat_grad_buffers.d_color_grads, 0, None);
    encoder.clear_buffer(&shared.mat_grad_buffers.d_base_colors, 0, None);
    encoder.clear_buffer(&shared.loss_buffers.loss_value, 0, None);

    // Scan tile pipeline
    rmesh_backward::record_scan_tile_pipeline(
        encoder,
        &shared.scan_pipelines,
        &shared.tile_pipelines,
        &shared.prepare_dispatch_bg,
        &shared.rts_bg,
        &ts.tile_fill_bg,
        &ts.tile_gen_scan_bg,
        &shared.scan_buffers,
        &ts.tile_buffers,
    );

    // Radix sort
    let result_in_b = rmesh_backward::record_radix_sort(
        encoder,
        device,
        &shared.radix_pipelines,
        &ts.radix_state,
        &ts.tile_buffers.tile_sort_keys,
        &ts.tile_buffers.tile_sort_values,
    );

    // Tile ranges
    {
        let ranges_bg = if result_in_b {
            &ts.tile_ranges_bg_b
        } else {
            &ts.tile_ranges_bg_a
        };
        let tw = recorder.as_mut().map(|r| {
            let (b, e) = r.allocate("tile_ranges");
            wgpu::ComputePassTimestampWrites {
                query_set: r.query_set(),
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            }
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tile_ranges"),
            timestamp_writes: tw,
        });
        pass.set_pipeline(&shared.tile_pipelines.tile_ranges_pipeline);
        pass.set_bind_group(0, ranges_bg, &[]);
        let wgs = (ts.tile_buffers.max_pairs_pow2 + 255) / 256;
        let (x, y) = rmesh_tile::dispatch_2d(wgs);
        pass.dispatch_workgroups(x, y, 1);
    }

    // Forward tiled (patched)
    {
        let fwd_bg = if result_in_b {
            &ts.rasterize_bg_b
        } else {
            &ts.rasterize_bg_a
        };
        let tw = recorder.as_mut().map(|r| {
            let (b, e) = r.allocate("rasterize_compute");
            wgpu::ComputePassTimestampWrites {
                query_set: r.query_set(),
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            }
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rasterize_compute"),
            timestamp_writes: tw,
        });
        pass.set_pipeline(&ts.rasterize_pipeline);
        pass.set_bind_group(0, fwd_bg, &[]);
        let (x, y) = rmesh_tile::dispatch_2d(ts.tile_buffers.num_tiles);
        pass.dispatch_workgroups(x, y, 1);
    }

    // Loss
    {
        let tw = recorder.as_mut().map(|r| {
            let (b, e) = r.allocate("loss");
            wgpu::ComputePassTimestampWrites {
                query_set: r.query_set(),
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            }
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("loss"),
            timestamp_writes: tw,
        });
        pass.set_pipeline(&shared.loss_pipeline.pipeline);
        pass.set_bind_group(0, &ts.loss_bg, &[]);
        pass.dispatch_workgroups((W + 15) / 16, (H + 15) / 16, 1);
    }

    // Backward tiled (patched)
    {
        let bwd_bg0 = if result_in_b {
            &ts.bwd_bg0_b
        } else {
            &ts.bwd_bg0_a
        };
        let tw = recorder.as_mut().map(|r| {
            let (b, e) = r.allocate("backward_tiled");
            wgpu::ComputePassTimestampWrites {
                query_set: r.query_set(),
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            }
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("backward_tiled"),
            timestamp_writes: tw,
        });
        pass.set_pipeline(&ts.bwd_tiled_pipeline);
        pass.set_bind_group(0, bwd_bg0, &[]);
        pass.set_bind_group(1, &ts.bwd_bg1, &[]);
        let (x, y) = rmesh_tile::dispatch_2d(ts.tile_buffers.num_tiles);
        pass.dispatch_workgroups(x, y, 1);
    }
}

fn run_once(shared: &SharedState, ts: &TileSizeState) {
    let mut encoder = shared
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    record_forward_backward(shared, ts, &mut encoder, &mut None);
    shared.queue.submit(std::iter::once(encoder.finish()));
    let _ = shared.device.poll(wgpu::PollType::wait_indefinitely());
}

fn run_timed(shared: &SharedState, ts: &TileSizeState) -> Vec<(String, f64)> {
    let mut recorder = TimestampRecorder::new(&shared.device, &shared.queue, 20);
    let mut encoder = shared
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    record_forward_backward(shared, ts, &mut encoder, &mut Some(&mut recorder));
    recorder.resolve(&mut encoder);
    shared.queue.submit(std::iter::once(encoder.finish()));
    let _ = shared.device.poll(wgpu::PollType::wait_indefinitely());
    recorder.read_results(&shared.device, &shared.queue)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let shared = match create_shared_state() {
        Some(s) => s,
        None => {
            eprintln!("No GPU with SUBGROUP + TIMESTAMP_QUERY + SHADER_FLOAT32_ATOMIC found.");
            std::process::exit(1);
        }
    };

    // Results: (tile_size, fwd_ms, bwd_ms, total_ms)
    let mut results: Vec<(u32, f64, f64, f64, u32)> = Vec::new();

    for &tile_size in TILE_SIZES {
        eprint!("tile_size={tile_size:>2} ... ");

        // Check constraint: tile_size must be ≤ warp size (32)
        if tile_size > 32 {
            eprintln!("SKIP (> warp size 32)");
            continue;
        }

        let ts = create_tile_size_state(&shared, tile_size);
        let num_tiles = ts.tile_buffers.num_tiles;

        // Warmup
        for _ in 0..WARMUP_ITERS {
            run_once(&shared, &ts);
        }

        // Collect timings
        let mut fwd_times = Vec::new();
        let mut bwd_times = Vec::new();
        let mut total_times = Vec::new();

        for _ in 0..BENCH_ITERS {
            let timings = run_timed(&shared, &ts);
            let mut fwd_ms = 0.0;
            let mut bwd_ms = 0.0;
            let mut total = 0.0;
            for (name, ms) in &timings {
                total += ms;
                if name == "rasterize_compute" {
                    fwd_ms = *ms;
                } else if name == "backward_tiled" {
                    bwd_ms = *ms;
                }
            }
            fwd_times.push(fwd_ms);
            bwd_times.push(bwd_ms);
            total_times.push(total);
        }

        // Use median
        fwd_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        bwd_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        total_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = BENCH_ITERS as usize / 2;
        let fwd = fwd_times[mid];
        let bwd = bwd_times[mid];
        let total = total_times[mid];

        eprintln!(
            "tiles={num_tiles:>6}  fwd={fwd:>8.2}ms  bwd={bwd:>8.2}ms  total={total:>8.2}ms"
        );
        results.push((tile_size, fwd, bwd, total, num_tiles));

        // Print detailed breakdown for last run
        let timings = run_timed(&shared, &ts);
        for (name, ms) in &timings {
            eprintln!("  {name:<20} {ms:>8.3} ms");
        }
        eprintln!();
    }

    // Summary table
    eprintln!("\n{}", "=".repeat(80));
    eprintln!(
        "{:<10} {:>8} {:>10} {:>10} {:>10} {:>8}",
        "tile_size", "tiles", "fwd (ms)", "bwd (ms)", "total(ms)", "bwd/fwd"
    );
    eprintln!("{}", "-".repeat(80));
    for &(ts, fwd, bwd, total, num_tiles) in &results {
        let ratio = if fwd > 0.0 { bwd / fwd } else { 0.0 };
        eprintln!(
            "{ts:<10} {num_tiles:>8} {fwd:>10.2} {bwd:>10.2} {total:>10.2} {ratio:>8.2}x"
        );
    }
    eprintln!("{}", "-".repeat(80));

    // Find best
    if let Some(&(best_ts, _, _, best_total, _)) = results.iter().min_by(|a, b| a.3.partial_cmp(&b.3).unwrap()) {
        eprintln!("\nBest total: tile_size={best_ts} ({best_total:.2} ms)");
    }
}
