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
    create_backward_bind_groups, BackwardPipelines, GradientBuffers, MaterialGradBuffers,
};
use rmesh_data::SceneData;
use rmesh_render::{
    create_compute_bind_group, create_render_bind_group, record_tex_to_buffer,
    ForwardPipelines, MaterialBuffers, RenderTargets, SceneBuffers, TexToBufferPipeline, Uniforms,
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

/// Run the training loop.
pub fn train(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    scene: &SceneData,
    views: &[TrainingView],
    config: &TrainConfig,
) -> Result<()> {
    let color_format = wgpu::TextureFormat::Rgba16Float;
    let aux_format = wgpu::TextureFormat::Rgba32Float;

    // Create pipelines
    let fwd_pipelines = ForwardPipelines::new(device, color_format, aux_format);
    let bwd_pipelines = BackwardPipelines::new(device);
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

    // Create render targets
    let targets = RenderTargets::new(device, config.render_width, config.render_height);

    // Create bind groups for forward pass
    let compute_bg = create_compute_bind_group(device, &fwd_pipelines, &buffers, &material);
    let render_bg = create_render_bind_group(device, &fwd_pipelines, &buffers, &material);

    // Loss buffers
    let loss_buffers = LossBuffers::new(device, config.render_width, config.render_height);

    // Tex-to-buffer pipeline
    let ttb = TexToBufferPipeline::new(
        device,
        queue,
        &targets.color_view,
        config.render_width,
        config.render_height,
    );

    // Create bind groups for backward pass
    let loss_bg = create_loss_bind_group(device, &loss_pipeline, &loss_buffers, &ttb.rendered_image);
    let (bwd_bg0, bwd_bg1) =
        create_backward_bind_groups(device, &bwd_pipelines, &buffers, &material, &loss_buffers.dl_d_image, &grads, &mat_grads, &ttb.rendered_image);

    // Adam bind groups — one per parameter group
    let adam_uniforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("adam_uniforms"),
        size: std::mem::size_of::<AdamUniforms>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let adam_bgs = [
        create_adam_bind_group(
            device,
            &adam_pipeline,
            &adam_uniforms_buf,
            &buffers.vertices,
            &grads.d_vertices,
            &adam.m_vertices,
            &adam.v_vertices,
        ),
        create_adam_bind_group(
            device,
            &adam_pipeline,
            &adam_uniforms_buf,
            &buffers.densities,
            &grads.d_densities,
            &adam.m_densities,
            &adam.v_densities,
        ),
        create_adam_bind_group(
            device,
            &adam_pipeline,
            &adam_uniforms_buf,
            &material.color_grads,
            &mat_grads.d_color_grads,
            &mat_adam.m_color_grads,
            &mat_adam.v_color_grads,
        ),
    ];

    let vert_param_count = scene.vertex_count * 3;
    let density_param_count = scene.tet_count;
    let grad_param_count = scene.tet_count * 3;

    let param_counts = [
        vert_param_count,
        density_param_count,
        grad_param_count,
    ];
    let learning_rates = [
        config.lr_vertices,
        config.lr_density,
        config.lr_color_grad,
    ];

    log::info!(
        "Training: {} tets, {} vertices, {} views, {} epochs",
        scene.tet_count,
        scene.vertex_count,
        views.len(),
        config.epochs
    );

    let mut step = 0u32;

    for epoch in 0..config.epochs {
        for (view_idx, view) in views.iter().enumerate() {
            let aspect = view.width as f32 / view.height as f32;
            let vp = view.view_projection(aspect);
            let inv_vp = vp.inverse();

            let vp_cols = vp.to_cols_array_2d();
            let inv_cols = inv_vp.to_cols_array_2d();
            let pos = view.position.to_array();

            let uniforms = Uniforms {
                vp_col0: vp_cols[0],
                vp_col1: vp_cols[1],
                vp_col2: vp_cols[2],
                vp_col3: vp_cols[3],
                inv_vp_col0: inv_cols[0],
                inv_vp_col1: inv_cols[1],
                inv_vp_col2: inv_cols[2],
                inv_vp_col3: inv_cols[3],
                cam_pos_pad: [pos[0], pos[1], pos[2], 0.0],
                screen_width: view.width as f32,
                screen_height: view.height as f32,
                tet_count: scene.tet_count,
                step,
                _pad1: [0; 8],
            };

            // Upload uniforms
            queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

            // Upload ground truth
            queue.write_buffer(
                &loss_buffers.ground_truth,
                0,
                bytemuck::cast_slice(&view.gt_image),
            );

            // Upload loss uniforms
            let loss_uni = LossUniforms {
                width: view.width,
                height: view.height,
                loss_type: config.loss_type,
                lambda_ssim: 0.0,
            };
            queue.write_buffer(
                &loss_buffers.loss_uniforms,
                0,
                bytemuck::bytes_of(&loss_uni),
            );

            // Clear gradient buffers
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("clear grads"),
                });
            encoder.clear_buffer(&grads.d_vertices, 0, None);
            encoder.clear_buffer(&grads.d_densities, 0, None);
            encoder.clear_buffer(&mat_grads.d_color_grads, 0, None);
            encoder.clear_buffer(&loss_buffers.loss_value, 0, None);
            queue.submit(std::iter::once(encoder.finish()));

            // Forward pass
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("forward"),
                });
            rmesh_render::record_forward_pass(
                &mut encoder,
                &fwd_pipelines,
                &buffers,
                &targets,
                &compute_bg,
                &render_bg,
                scene.tet_count,
                queue,
            );
            record_tex_to_buffer(&mut encoder, &ttb);
            queue.submit(std::iter::once(encoder.finish()));

            // Loss pass
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("loss"),
                });
            record_loss_pass(
                &mut encoder,
                &loss_pipeline,
                &loss_bg,
                view.width,
                view.height,
            );
            queue.submit(std::iter::once(encoder.finish()));

            // Backward pass
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("backward"),
                });
            rmesh_backward::record_backward_pass(
                &mut encoder,
                &bwd_pipelines,
                &bwd_bg0,
                &bwd_bg1,
                view.width,
                view.height,
            );
            queue.submit(std::iter::once(encoder.finish()));

            // Adam pass — one dispatch per parameter group
            for (i, (bg, &count)) in adam_bgs.iter().zip(param_counts.iter()).enumerate() {
                step += 1;
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

                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("adam"),
                    });
                record_adam_pass(
                    &mut encoder,
                    &adam_pipeline,
                    std::slice::from_ref(bg),
                    &[count],
                );
                queue.submit(std::iter::once(encoder.finish()));
            }

            if step % 100 == 0 {
                log::info!(
                    "Step {}, epoch {}, view {}/{}",
                    step,
                    epoch,
                    view_idx + 1,
                    views.len()
                );
            }
        }
    }

    Ok(())
}
