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
    create_adam_bind_group, create_backward_bind_groups, create_loss_bind_group, AdamState,
    BackwardPipelines, GradientBuffers, LossBuffers,
};
use rmesh_data::SceneData;
use rmesh_render::{
    create_compute_bind_group, create_render_bind_group, record_tex_to_buffer,
    ForwardPipelines, RenderTargets, SceneBuffers, SortState, TexToBufferPipeline, Uniforms,
};
use rmesh_shaders::shared::{AdamUniforms, LossUniforms};

/// Training configuration.
pub struct TrainConfig {
    pub epochs: u32,
    pub lr_sh: f32,
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
            lr_sh: 1e-3,
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

    // Upload scene data
    let buffers = SceneBuffers::upload(device, queue, scene);

    // Allocate gradient + optimizer state
    let sh_stride = scene.num_sh_coeffs() * 3;
    let grads = GradientBuffers::new(device, scene.vertex_count, scene.tet_count, sh_stride);
    let adam = AdamState::new(device, scene.vertex_count, scene.tet_count, sh_stride);

    // Create render targets
    let targets = RenderTargets::new(device, config.render_width, config.render_height);

    // Create sort state
    let sort_state = SortState::new(
        device,
        &fwd_pipelines,
        &buffers.sort_keys,
        &buffers.sort_values,
        scene.tet_count,
    );

    // Create bind groups for forward pass
    let compute_bg = create_compute_bind_group(device, &fwd_pipelines, &buffers);
    let render_bg = create_render_bind_group(device, &fwd_pipelines, &buffers);

    // Loss buffers
    let loss_buffers = LossBuffers::new(device, config.render_width, config.render_height);

    // Tex-to-buffer pipeline: converts Rgba16Float render target to f32 storage buffer
    let ttb = TexToBufferPipeline::new(
        device,
        queue,
        &targets.color_view,
        config.render_width,
        config.render_height,
    );

    // Create bind groups for backward pass
    let loss_bg = create_loss_bind_group(device, &bwd_pipelines, &loss_buffers, &ttb.rendered_image);
    let (bwd_bg0, bwd_bg1) =
        create_backward_bind_groups(device, &bwd_pipelines, &buffers, &loss_buffers, &grads, &ttb.rendered_image);

    // Adam bind groups — one per parameter group
    // Each group: (uniforms_buf, params, grads, m, v)
    let adam_uniforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("adam_uniforms"),
        size: std::mem::size_of::<AdamUniforms>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Parameter groups: SH, vertices, densities, color_grads
    let adam_bgs = [
        create_adam_bind_group(
            device,
            &bwd_pipelines,
            &adam_uniforms_buf,
            &buffers.sh_coeffs,
            &grads.d_sh_coeffs,
            &adam.m_sh,
            &adam.v_sh,
        ),
        create_adam_bind_group(
            device,
            &bwd_pipelines,
            &adam_uniforms_buf,
            &buffers.vertices,
            &grads.d_vertices,
            &adam.m_vertices,
            &adam.v_vertices,
        ),
        create_adam_bind_group(
            device,
            &bwd_pipelines,
            &adam_uniforms_buf,
            &buffers.densities,
            &grads.d_densities,
            &adam.m_densities,
            &adam.v_densities,
        ),
        create_adam_bind_group(
            device,
            &bwd_pipelines,
            &adam_uniforms_buf,
            &buffers.color_grads,
            &grads.d_color_grads,
            &adam.m_color_grads,
            &adam.v_color_grads,
        ),
    ];

    let sh_param_count = scene.tet_count * sh_stride;
    let vert_param_count = scene.vertex_count * 3;
    let density_param_count = scene.tet_count;
    let grad_param_count = scene.tet_count * 3;

    let param_counts = [
        sh_param_count,
        vert_param_count,
        density_param_count,
        grad_param_count,
    ];
    let learning_rates = [
        config.lr_sh,
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
                sh_degree: scene.sh_degree,
                step,
                _pad1: [0; 7],
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
            encoder.clear_buffer(&grads.d_sh_coeffs, 0, None);
            encoder.clear_buffer(&grads.d_vertices, 0, None);
            encoder.clear_buffer(&grads.d_densities, 0, None);
            encoder.clear_buffer(&grads.d_color_grads, 0, None);
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
                &sort_state,
                scene.tet_count,
                queue,
            );
            // Convert f16 render target texture to f32 storage buffer for loss/backward
            record_tex_to_buffer(&mut encoder, &ttb);
            queue.submit(std::iter::once(encoder.finish()));

            // Loss pass
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("loss"),
                });
            rmesh_backward::record_loss_pass(
                &mut encoder,
                &bwd_pipelines,
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
                rmesh_backward::record_adam_pass(
                    &mut encoder,
                    &bwd_pipelines,
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
