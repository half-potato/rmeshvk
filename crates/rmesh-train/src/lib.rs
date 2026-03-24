//! Training loop orchestration.
//!
//! Coordinates the full training pipeline:
//!   1. Forward compute → sort → render
//!   2. Loss computation
//!   3. Backward compute
//!   4. Adam optimizer update
//!
//! All steps are GPU command buffer dispatches — no CPU readback in the loop.

use anyhow::Result;
use glam::{Mat4, Quat, Vec3};
use rmesh_backward::{
    create_backward_tiled_bind_groups, BackwardTiledPipelines,
    GradientBuffers, MaterialGradBuffers, TileUniforms,
    RadixSortPipelines, RadixSortState, sorting_bits_for_tiles, SortBackend,
    TileBuffers, TilePipelines,
    ScanPipelines, ScanBuffers,
    create_tile_fill_bind_group, create_tile_ranges_bind_group_with_keys,
    create_prepare_dispatch_bind_group, create_rts_bind_group,
    create_tile_gen_scan_bind_group,
    record_scan_tile_pipeline,
};
use rmesh_data::SceneData;
use rmesh_render::{
    create_compute_bind_group, create_rasterize_bind_group, record_project_compute,
    record_rasterize_compute,
    ForwardPipelines, MaterialBuffers, RasterizeComputePipeline, SceneBuffers, Uniforms,
};
use rmesh_util::shared::{AdamUniforms, LossUniforms};

// Re-export these types so downstream crates (rmesh-python) can use them.
pub use rmesh_util::shared::AdamUniforms as AdamUniformsType;
pub use rmesh_util::shared::LossUniforms as LossUniformsType;

// WGSL shader sources.
const LOSS_COMPUTE_WGSL: &str = include_str!("wgsl/loss_compute.wgsl");
const ADAM_COMPUTE_WGSL: &str = include_str!("wgsl/adam_compute.wgsl");

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn dispatch_2d(total_workgroups: u32) -> (u32, u32) {
    if total_workgroups <= 65535 {
        (total_workgroups, 1)
    } else {
        let x = 65535u32;
        let y = (total_workgroups + x - 1) / x;
        (x, y)
    }
}

fn create_storage_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

// ===========================================================================
// Loss buffers and pipeline
// ===========================================================================

/// Buffers for loss computation.
pub struct LossBuffers {
    /// Per-pixel gradient dL/d(pixel): [H x W x 4] f32
    pub dl_d_image: wgpu::Buffer,
    /// Ground truth image: [H x W x 3] f32
    pub ground_truth: wgpu::Buffer,
    /// Scalar loss value: [1] f32
    pub loss_value: wgpu::Buffer,
    /// Loss uniforms
    pub loss_uniforms: wgpu::Buffer,
}

impl LossBuffers {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let n_pixels = (width as u64) * (height as u64);
        Self {
            dl_d_image: create_storage_buffer(device, "dl_d_image", n_pixels * 4 * 4),
            ground_truth: create_storage_buffer(device, "ground_truth", n_pixels * 3 * 4),
            loss_value: create_storage_buffer(device, "loss_value", 4),
            loss_uniforms: create_storage_buffer(
                device,
                "loss_uniforms",
                std::mem::size_of::<LossUniforms>() as u64,
            ),
        }
    }
}

/// Loss compute pipeline.
pub struct LossPipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl LossPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let loss_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("loss_compute"),
            source: wgpu::ShaderSource::Wgsl(LOSS_COMPUTE_WGSL.into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("loss_bind_group_layout"),
                entries: &[
                    storage_entry(0, true),  // uniforms
                    storage_entry(1, true),  // rendered
                    storage_entry(2, true),  // ground_truth
                    storage_entry(3, false), // dl_d_image
                ],
            });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("loss_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                immediate_size: 0,
            });

        let pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("loss_pipeline"),
                layout: Some(&pipeline_layout),
                module: &loss_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self { pipeline, bind_group_layout }
    }
}

/// Create the bind group for the loss compute pass.
pub fn create_loss_bind_group(
    device: &wgpu::Device,
    pipeline: &LossPipeline,
    loss_buffers: &LossBuffers,
    rendered_buf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("loss_bind_group"),
        layout: &pipeline.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: loss_buffers.loss_uniforms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rendered_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: loss_buffers.ground_truth.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: loss_buffers.dl_d_image.as_entire_binding(),
            },
        ],
    })
}

/// Record the loss computation pass.
pub fn record_loss_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &LossPipeline,
    loss_bg: &wgpu::BindGroup,
    width: u32,
    height: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("loss_pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipeline.pipeline);
    pass.set_bind_group(0, loss_bg, &[]);
    pass.dispatch_workgroups((width + 15) / 16, (height + 15) / 16, 1);
}

// ===========================================================================
// Adam optimizer
// ===========================================================================

/// Adam optimizer state for geometry parameters.
pub struct AdamState {
    pub m_vertices: wgpu::Buffer,
    pub v_vertices: wgpu::Buffer,
    pub m_densities: wgpu::Buffer,
    pub v_densities: wgpu::Buffer,
}

impl AdamState {
    pub fn new(
        device: &wgpu::Device,
        vertex_count: u32,
        tet_count: u32,
    ) -> Self {
        let vert_size = (vertex_count as u64) * 3 * 4;
        let density_size = (tet_count as u64) * 4;

        Self {
            m_vertices: create_storage_buffer(device, "adam_m_vertices", vert_size),
            v_vertices: create_storage_buffer(device, "adam_v_vertices", vert_size),
            m_densities: create_storage_buffer(device, "adam_m_densities", density_size),
            v_densities: create_storage_buffer(device, "adam_v_densities", density_size),
        }
    }
}

/// Adam optimizer state for material parameters.
pub struct MaterialAdamState {
    pub m_color_grads: wgpu::Buffer,
    pub v_color_grads: wgpu::Buffer,
}

impl MaterialAdamState {
    pub fn new(device: &wgpu::Device, tet_count: u32) -> Self {
        let grad_size = (tet_count as u64) * 3 * 4;
        Self {
            m_color_grads: create_storage_buffer(device, "adam_m_color_grads", grad_size),
            v_color_grads: create_storage_buffer(device, "adam_v_color_grads", grad_size),
        }
    }
}

/// Adam optimizer pipeline.
pub struct AdamPipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl AdamPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let adam_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("adam_compute"),
            source: wgpu::ShaderSource::Wgsl(ADAM_COMPUTE_WGSL.into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("adam_bind_group_layout"),
                entries: &[
                    storage_entry(0, true),  // uniforms
                    storage_entry(1, false), // params
                    storage_entry(2, true),  // grads
                    storage_entry(3, false), // m
                    storage_entry(4, false), // v
                ],
            });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("adam_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                immediate_size: 0,
            });

        let pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("adam_pipeline"),
                layout: Some(&pipeline_layout),
                module: &adam_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self { pipeline, bind_group_layout }
    }
}

/// Create a bind group for a single Adam optimizer parameter group.
pub fn create_adam_bind_group(
    device: &wgpu::Device,
    pipeline: &AdamPipeline,
    uniforms_buf: &wgpu::Buffer,
    params: &wgpu::Buffer,
    grads: &wgpu::Buffer,
    m: &wgpu::Buffer,
    v: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("adam_bind_group"),
        layout: &pipeline.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: grads.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: m.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: v.as_entire_binding(),
            },
        ],
    })
}

/// Record Adam optimizer updates for all parameter groups.
pub fn record_adam_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &AdamPipeline,
    adam_bgs: &[wgpu::BindGroup],
    param_counts: &[u32],
) {
    for (bg, &count) in adam_bgs.iter().zip(param_counts) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("adam_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, bg, &[]);
        let (ax, ay) = dispatch_2d((count + 255) / 256);
        pass.dispatch_workgroups(ax, ay, 1);
    }
}

// ===========================================================================
// Training configuration and loop
// ===========================================================================

/// Training configuration.
pub struct TrainConfig {
    pub epochs: u32,
    pub lr_vertices: f32,
    pub lr_density: f32,
    pub lr_color_grad: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub loss_type: u32, // 0 = L1, 1 = L2
    pub render_width: u32,
    pub render_height: u32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            lr_vertices: 1e-4,
            lr_density: 1e-2,
            lr_color_grad: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            loss_type: 0,
            render_width: 1920,
            render_height: 1080,
        }
    }
}

/// A training camera view with associated ground truth.
pub struct TrainingView {
    pub position: Vec3,
    pub rotation: Quat,
    pub fov_y: f32,
    pub gt_image: Vec<f32>, // [H × W × 3] f32
    pub width: u32,
    pub height: u32,
}

impl TrainingView {
    /// Build view-projection matrix for this camera.
    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        let view = Mat4::from_rotation_translation(self.rotation, self.position).inverse();
        let proj = Mat4::perspective_rh(self.fov_y, aspect, 0.01, 1000.0);
        proj * view
    }
}

/// Run the training loop (tiled forward + tiled backward).
pub fn train(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    scene: &SceneData,
    views: &[TrainingView],
    config: &TrainConfig,
) -> Result<()> {
    let color_format = wgpu::TextureFormat::Rgba16Float;
    let tile_size = 12u32;

    // Create pipelines
    let fwd_pipelines = ForwardPipelines::new(device, color_format);
    let bwd_tiled_pipelines = BackwardTiledPipelines::new(device);
    let loss_pipeline = LossPipeline::new(device);
    let adam_pipeline = AdamPipeline::new(device);

    // Upload scene + material data
    let buffers = SceneBuffers::upload(device, queue, scene);
    let default_base_colors = vec![0.5f32; scene.tet_count as usize * 3];
    let material = MaterialBuffers::upload(device, &default_base_colors, &scene.color_grads, scene.tet_count);

    // Allocate gradient + optimizer state
    let grads = GradientBuffers::new(device, scene.vertex_count, scene.tet_count);
    let mat_grads = MaterialGradBuffers::new(device, scene.tet_count);
    let adam = AdamState::new(device, scene.vertex_count, scene.tet_count);
    let mat_adam = MaterialAdamState::new(device, scene.tet_count);

    // Dummy sh_coeffs buffer (training uses sh_degree=0 / base_colors path)
    let dummy_sh = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dummy_sh_coeffs"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    // Forward compute bind group
    let compute_bg = create_compute_bind_group(device, &fwd_pipelines, &buffers, &material, &dummy_sh);

    // Tiled forward pipeline
    let rasterize = RasterizeComputePipeline::new(device, config.render_width, config.render_height, 0);

    // Tile infrastructure
    let tile_pipelines = TilePipelines::new(device);
    let tile_buffers = TileBuffers::new(device, scene.tet_count, config.render_width, config.render_height, tile_size);
    let sorting_bits = sorting_bits_for_tiles(tile_buffers.num_tiles, SortBackend::Drs);
    let radix_pipelines = RadixSortPipelines::new(device, 2, SortBackend::Drs);
    let radix_state = RadixSortState::new(device, tile_buffers.max_pairs_pow2, sorting_bits, 2, SortBackend::Drs);
    radix_state.upload_configs(queue);

    let tile_fill_bg = create_tile_fill_bind_group(device, &tile_pipelines, &tile_buffers);

    // A/B tile ranges bind groups
    let tile_ranges_bg = create_tile_ranges_bind_group_with_keys(
        device, &tile_pipelines,
        &tile_buffers.tile_sort_keys, &tile_buffers.tile_ranges,
        &tile_buffers.tile_uniforms, radix_state.num_keys_buf(),
    );
    let tile_ranges_bg_b = create_tile_ranges_bind_group_with_keys(
        device, &tile_pipelines,
        radix_state.keys_b(), &tile_buffers.tile_ranges,
        &tile_buffers.tile_uniforms, radix_state.num_keys_buf(),
    );

    // Forward tiled bind groups (A and B)
    let rasterize_bg = create_rasterize_bind_group(
        device, &rasterize, &buffers.uniforms, &buffers.vertices,
        &buffers.indices, &material.colors, &buffers.densities,
        &material.color_grads, &tile_buffers.tile_sort_values,
        &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms,
    );
    let rasterize_bg_b = create_rasterize_bind_group(
        device, &rasterize, &buffers.uniforms, &buffers.vertices,
        &buffers.indices, &material.colors, &buffers.densities,
        &material.color_grads, radix_state.values_b(),
        &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms,
    );

    // Scan pipeline infrastructure
    let scan_pipelines = ScanPipelines::new(device);
    let scan_buffers = ScanBuffers::new(device, scene.tet_count);
    let prepare_dispatch_bg = create_prepare_dispatch_bind_group(
        device, &scan_pipelines, &buffers.indirect_args, &scan_buffers,
    );
    let rts_bg = create_rts_bind_group(
        device, &scan_pipelines, &buffers.tiles_touched, &scan_buffers,
    );
    let tile_gen_scan_bg = create_tile_gen_scan_bind_group(
        device, &scan_pipelines, &tile_buffers, &buffers.uniforms,
        &buffers.vertices, &buffers.indices, &buffers.compact_tet_ids,
        &buffers.circumdata, &buffers.tiles_touched, &scan_buffers,
        radix_state.num_keys_buf(),
    );

    // Loss buffers + bind group (using rasterize.rendered_image)
    let loss_buffers = LossBuffers::new(device, config.render_width, config.render_height);
    let loss_bg = create_loss_bind_group(device, &loss_pipeline, &loss_buffers, &rasterize.rendered_image);

    // Backward tiled bind groups (A and B, using rasterize.rendered_image)
    let (bwd_tiled_bg0, bwd_tiled_bg1) = create_backward_tiled_bind_groups(
        device, &bwd_tiled_pipelines, &buffers.uniforms,
        &loss_buffers.dl_d_image, &rasterize.rendered_image,
        &buffers.vertices, &buffers.indices, &buffers.densities,
        &material.color_grads, &material.colors,
        &tile_buffers.tile_sort_values,
        &grads.d_vertices, &grads.d_densities,
        &mat_grads.d_color_grads, &mat_grads.d_base_colors,
        &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms,
    );
    let (bwd_tiled_bg0_b, _) = create_backward_tiled_bind_groups(
        device, &bwd_tiled_pipelines, &buffers.uniforms,
        &loss_buffers.dl_d_image, &rasterize.rendered_image,
        &buffers.vertices, &buffers.indices, &buffers.densities,
        &material.color_grads, &material.colors,
        radix_state.values_b(),
        &grads.d_vertices, &grads.d_densities,
        &mat_grads.d_color_grads, &mat_grads.d_base_colors,
        &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms,
    );

    // Adam bind groups — one per parameter group
    let adam_uniforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("adam_uniforms"),
        size: std::mem::size_of::<AdamUniforms>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let adam_bgs = [
        create_adam_bind_group(
            device, &adam_pipeline, &adam_uniforms_buf,
            &buffers.vertices, &grads.d_vertices,
            &adam.m_vertices, &adam.v_vertices,
        ),
        create_adam_bind_group(
            device, &adam_pipeline, &adam_uniforms_buf,
            &buffers.densities, &grads.d_densities,
            &adam.m_densities, &adam.v_densities,
        ),
        create_adam_bind_group(
            device, &adam_pipeline, &adam_uniforms_buf,
            &material.color_grads, &mat_grads.d_color_grads,
            &mat_adam.m_color_grads, &mat_adam.v_color_grads,
        ),
    ];

    let param_counts = [
        scene.vertex_count * 3,
        scene.tet_count,
        scene.tet_count * 3,
    ];
    let learning_rates = [
        config.lr_vertices,
        config.lr_density,
        config.lr_color_grad,
    ];

    log::info!(
        "Training: {} tets, {} vertices, {} views, {} epochs",
        scene.tet_count, scene.vertex_count, views.len(), config.epochs
    );

    let mut step = 0u32;

    for epoch in 0..config.epochs {
        for (view_idx, view) in views.iter().enumerate() {
            let aspect = view.width as f32 / view.height as f32;
            let vp = view.view_projection(aspect);

            // Extract c2w rotation from inverse view matrix
            // view_projection uses Mat4::perspective_rh (RH convention)
            let inv_view = Mat4::from_rotation_translation(view.rotation, view.position);
            let cam_right = inv_view.col(0).truncate();
            let cam_up = inv_view.col(1).truncate();
            let cam_back = inv_view.col(2).truncate();
            // Pinhole convention: x=right, y=DOWN, z=FORWARD
            let c2w = glam::Mat3::from_cols(cam_right, -cam_up, -cam_back);

            // Intrinsics from FOV
            let f_val = 1.0 / (view.fov_y / 2.0).tan();
            let fx = f_val * view.height as f32 / 2.0;
            let fy = fx;
            let intrinsics = [fx, fy, view.width as f32 / 2.0, view.height as f32 / 2.0];

            let vp_cols = vp.to_cols_array_2d();
            let pos = view.position.to_array();

            let uniforms = Uniforms {
                vp_col0: vp_cols[0],
                vp_col1: vp_cols[1],
                vp_col2: vp_cols[2],
                vp_col3: vp_cols[3],
                c2w_col0: [c2w.col(0).x, c2w.col(0).y, c2w.col(0).z, 0.0],
                c2w_col1: [c2w.col(1).x, c2w.col(1).y, c2w.col(1).z, 0.0],
                c2w_col2: [c2w.col(2).x, c2w.col(2).y, c2w.col(2).z, 0.0],
                intrinsics,
                cam_pos_pad: [pos[0], pos[1], pos[2], 0.0],
                screen_width: view.width as f32,
                screen_height: view.height as f32,
                tet_count: scene.tet_count,
                step,
                tile_size_u: tile_size,
                ray_mode: 0,
                min_t: 0.0,
                sh_degree: 0,
                near_plane: 0.01,
                far_plane: 1000.0,
                _pad1: [0; 2],
            };

            // Upload uniforms
            queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

            // Upload tile uniforms
            let tile_uni = TileUniforms {
                screen_width: config.render_width,
                screen_height: config.render_height,
                tile_size,
                tiles_x: tile_buffers.tiles_x,
                tiles_y: tile_buffers.tiles_y,
                num_tiles: tile_buffers.num_tiles,
                visible_tet_count: 0,
                _pad: [0; 5],
            };
            queue.write_buffer(&tile_buffers.tile_uniforms, 0, bytemuck::bytes_of(&tile_uni));

            // Upload ground truth + loss uniforms
            queue.write_buffer(&loss_buffers.ground_truth, 0, bytemuck::cast_slice(&view.gt_image));
            let loss_uni = LossUniforms {
                width: view.width,
                height: view.height,
                loss_type: config.loss_type,
                lambda_ssim: 0.0,
            };
            queue.write_buffer(&loss_buffers.loss_uniforms, 0, bytemuck::bytes_of(&loss_uni));

            // Single encoder: clear + forward compute + scan + sort + tile_ranges
            //                 + forward tiled + loss + backward tiled + adam
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("train_step"),
            });

            // Clear buffers
            encoder.clear_buffer(&grads.d_vertices, 0, None);
            encoder.clear_buffer(&grads.d_densities, 0, None);
            encoder.clear_buffer(&mat_grads.d_base_colors, 0, None);
            encoder.clear_buffer(&mat_grads.d_color_grads, 0, None);
            encoder.clear_buffer(&loss_buffers.loss_value, 0, None);
            encoder.clear_buffer(&tile_buffers.tile_ranges, 0, None);
            encoder.clear_buffer(&rasterize.rendered_image, 0, None);

            // Forward compute (SH eval + cull + tiles_touched + compact_tet_ids)
            record_project_compute(
                &mut encoder, &fwd_pipelines, &buffers, &compute_bg,
                scene.tet_count, queue,
            );

            // Scan-based tile pipeline
            record_scan_tile_pipeline(
                &mut encoder, &scan_pipelines, &tile_pipelines,
                &prepare_dispatch_bg, &rts_bg, &tile_fill_bg,
                &tile_gen_scan_bg, &scan_buffers, &tile_buffers,
            );

            // Radix sort
            let result_in_b = rmesh_backward::record_radix_sort(
                &mut encoder, device, &radix_pipelines, &radix_state,
                &tile_buffers.tile_sort_keys, &tile_buffers.tile_sort_values,
            );

            let (ranges_bg, fwd_bg, bwd_bg0) = if result_in_b {
                (&tile_ranges_bg_b, &rasterize_bg_b, &bwd_tiled_bg0_b)
            } else {
                (&tile_ranges_bg, &rasterize_bg, &bwd_tiled_bg0)
            };

            // Tile ranges
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("tile_ranges"), timestamp_writes: None,
                });
                pass.set_pipeline(&tile_pipelines.tile_ranges_pipeline);
                pass.set_bind_group(0, ranges_bg, &[]);
                let total = (tile_buffers.max_pairs_pow2 + 255) / 256;
                let (x, y) = dispatch_2d(total);
                pass.dispatch_workgroups(x, y, 1);
            }

            // Forward tiled compute
            record_rasterize_compute(&mut encoder, &rasterize, fwd_bg, tile_buffers.num_tiles);

            // Loss compute
            record_loss_pass(&mut encoder, &loss_pipeline, &loss_bg, view.width, view.height);

            // Backward tiled compute
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("backward_tiled"), timestamp_writes: None,
                });
                pass.set_pipeline(&bwd_tiled_pipelines.pipeline);
                pass.set_bind_group(0, bwd_bg0, &[]);
                pass.set_bind_group(1, &bwd_tiled_bg1, &[]);
                let (x, y) = dispatch_2d(tile_buffers.num_tiles);
                pass.dispatch_workgroups(x, y, 1);
            }

            // Adam passes
            step += 1;
            for (i, (bg, &count)) in adam_bgs.iter().zip(param_counts.iter()).enumerate() {
                let adam_uni = AdamUniforms {
                    param_count: count,
                    step,
                    lr: learning_rates[i],
                    beta1: config.beta1,
                    beta2: config.beta2,
                    epsilon: config.epsilon,
                    _pad: [0; 2],
                };
                queue.write_buffer(&adam_uniforms_buf, 0, bytemuck::bytes_of(&adam_uni));

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("adam"), timestamp_writes: None,
                });
                pass.set_pipeline(&adam_pipeline.pipeline);
                pass.set_bind_group(0, bg, &[]);
                let (ax, ay) = dispatch_2d((count + 255) / 256);
                pass.dispatch_workgroups(ax, ay, 1);
            }

            queue.submit(std::iter::once(encoder.finish()));

            if step % 100 == 0 {
                log::info!(
                    "Step {}, epoch {}, view {}/{}",
                    step, epoch, view_idx + 1, views.len()
                );
            }
        }
    }

    Ok(())
}
