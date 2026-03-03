//! Backward pass orchestration (wgpu port).
//!
//! Manages:
//!   - Loss computation dispatch
//!   - Backward compute dispatch (2 bind groups for WebGPU 8-buffer limit)
//!   - Gradient buffer management
//!   - Adam optimizer dispatch

use rmesh_render::SceneBuffers;
use rmesh_shaders::shared;
use wgpu;

// Re-export shared uniform types for downstream crates.
pub use shared::AdamUniforms;
pub use shared::LossUniforms;

/// Gradient buffers -- one per trainable parameter group.
pub struct GradientBuffers {
    pub d_sh_coeffs: wgpu::Buffer,
    pub d_vertices: wgpu::Buffer,
    pub d_densities: wgpu::Buffer,
    pub d_color_grads: wgpu::Buffer,
}

/// Adam optimizer state -- first and second moments per parameter group.
pub struct AdamState {
    // SH coefficients
    pub m_sh: wgpu::Buffer,
    pub v_sh: wgpu::Buffer,
    // Vertices
    pub m_vertices: wgpu::Buffer,
    pub v_vertices: wgpu::Buffer,
    // Densities
    pub m_densities: wgpu::Buffer,
    pub v_densities: wgpu::Buffer,
    // Color gradients
    pub m_color_grads: wgpu::Buffer,
    pub v_color_grads: wgpu::Buffer,
}

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

/// Loss + backward + optimizer pipelines and their bind group layouts.
pub struct BackwardPipelines {
    pub loss_pipeline: wgpu::ComputePipeline,
    pub loss_bind_group_layout: wgpu::BindGroupLayout,

    pub backward_pipeline: wgpu::ComputePipeline,
    pub backward_bind_group_layout_0: wgpu::BindGroupLayout,
    pub backward_bind_group_layout_1: wgpu::BindGroupLayout,

    pub adam_pipeline: wgpu::ComputePipeline,
    pub adam_bind_group_layout: wgpu::BindGroupLayout,
}

// ---------------------------------------------------------------------------
// Buffer creation helpers
// ---------------------------------------------------------------------------

/// Create a zero-initialized storage buffer with COPY_DST (for clearing) and COPY_SRC (for readback).
fn create_storage_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Create a storage buffer with COPY_DST and COPY_SRC (for GPU readback).
fn create_storage_buffer_readable(
    device: &wgpu::Device,
    label: &str,
    size: u64,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

// ---------------------------------------------------------------------------
// GradientBuffers
// ---------------------------------------------------------------------------

impl GradientBuffers {
    /// Allocate zero-initialized gradient buffers.
    ///
    /// * `vertex_count` -- number of unique vertices
    /// * `tet_count`    -- number of tetrahedra
    /// * `sh_stride`    -- floats per tet in the SH buffer (num_coeffs * 3)
    pub fn new(
        device: &wgpu::Device,
        vertex_count: u32,
        tet_count: u32,
        sh_stride: u32,
    ) -> Self {
        Self {
            d_sh_coeffs: create_storage_buffer(
                device,
                "d_sh_coeffs",
                (tet_count as u64) * (sh_stride as u64) * 4,
            ),
            d_vertices: create_storage_buffer(
                device,
                "d_vertices",
                (vertex_count as u64) * 3 * 4,
            ),
            d_densities: create_storage_buffer(
                device,
                "d_densities",
                (tet_count as u64) * 4,
            ),
            d_color_grads: create_storage_buffer(
                device,
                "d_color_grads",
                (tet_count as u64) * 3 * 4,
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// AdamState
// ---------------------------------------------------------------------------

impl AdamState {
    /// Allocate zero-initialized Adam state buffers.
    pub fn new(
        device: &wgpu::Device,
        vertex_count: u32,
        tet_count: u32,
        sh_stride: u32,
    ) -> Self {
        let sh_size = (tet_count as u64) * (sh_stride as u64) * 4;
        let vert_size = (vertex_count as u64) * 3 * 4;
        let density_size = (tet_count as u64) * 4;
        let grad_size = (tet_count as u64) * 3 * 4;

        Self {
            m_sh: create_storage_buffer(device, "adam_m_sh", sh_size),
            v_sh: create_storage_buffer(device, "adam_v_sh", sh_size),
            m_vertices: create_storage_buffer(device, "adam_m_vertices", vert_size),
            v_vertices: create_storage_buffer(device, "adam_v_vertices", vert_size),
            m_densities: create_storage_buffer(device, "adam_m_densities", density_size),
            v_densities: create_storage_buffer(device, "adam_v_densities", density_size),
            m_color_grads: create_storage_buffer(device, "adam_m_color_grads", grad_size),
            v_color_grads: create_storage_buffer(device, "adam_v_color_grads", grad_size),
        }
    }
}

// ---------------------------------------------------------------------------
// LossBuffers
// ---------------------------------------------------------------------------

impl LossBuffers {
    /// Allocate loss computation buffers.
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let n_pixels = (width as u64) * (height as u64);

        Self {
            dl_d_image: create_storage_buffer(
                device,
                "dl_d_image",
                n_pixels * 4 * 4, // RGBA f32
            ),
            ground_truth: create_storage_buffer(
                device,
                "ground_truth",
                n_pixels * 3 * 4, // RGB f32
            ),
            loss_value: create_storage_buffer_readable(
                device,
                "loss_value",
                4, // single f32
            ),
            loss_uniforms: create_storage_buffer(
                device,
                "loss_uniforms",
                std::mem::size_of::<LossUniforms>() as u64,
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// BackwardPipelines
// ---------------------------------------------------------------------------

impl BackwardPipelines {
    /// Create all compute pipelines from WGSL source.
    pub fn new(device: &wgpu::Device) -> Self {
        // ----- Loss pipeline -----
        let loss_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("loss_compute"),
            source: wgpu::ShaderSource::Wgsl(rmesh_shaders::LOSS_COMPUTE_WGSL.into()),
        });

        let loss_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("loss_bind_group_layout"),
                entries: &[
                    // @binding(0) uniforms (read)
                    storage_entry(0, true),
                    // @binding(1) rendered (read)
                    storage_entry(1, true),
                    // @binding(2) ground_truth (read)
                    storage_entry(2, true),
                    // @binding(3) dl_d_image (read_write)
                    storage_entry(3, false),
                    // @binding(4) loss_value (read_write)
                    storage_entry(4, false),
                ],
            });

        let loss_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("loss_pipeline_layout"),
                bind_group_layouts: &[&loss_bind_group_layout],
                push_constant_ranges: &[],
            });

        let loss_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("loss_pipeline"),
                layout: Some(&loss_pipeline_layout),
                module: &loss_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // ----- Backward pipeline (2 bind groups) -----
        let backward_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("backward_compute"),
            source: wgpu::ShaderSource::Wgsl(rmesh_shaders::BACKWARD_COMPUTE_WGSL.into()),
        });

        // Group 0: 11 read-only bindings
        let backward_bind_group_layout_0 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("backward_bind_group_layout_0"),
                entries: &[
                    storage_entry(0, true),  // uniforms
                    storage_entry(1, true),  // dl_d_image
                    storage_entry(2, true),  // rendered_image
                    storage_entry(3, true),  // vertices
                    storage_entry(4, true),  // indices
                    storage_entry(5, true),  // sh_coeffs
                    storage_entry(6, true),  // densities
                    storage_entry(7, true),  // color_grads
                    storage_entry(8, true),  // circumdata
                    storage_entry(9, true),  // colors
                    storage_entry(10, true), // sorted_indices
                ],
            });

        // Group 1: 4 read-write bindings
        let backward_bind_group_layout_1 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("backward_bind_group_layout_1"),
                entries: &[
                    storage_entry(0, false), // d_sh_coeffs
                    storage_entry(1, false), // d_vertices
                    storage_entry(2, false), // d_densities
                    storage_entry(3, false), // d_color_grads
                ],
            });

        let backward_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("backward_pipeline_layout"),
                bind_group_layouts: &[
                    &backward_bind_group_layout_0,
                    &backward_bind_group_layout_1,
                ],
                push_constant_ranges: &[],
            });

        let backward_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("backward_pipeline"),
                layout: Some(&backward_pipeline_layout),
                module: &backward_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // ----- Adam pipeline -----
        let adam_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("adam_compute"),
            source: wgpu::ShaderSource::Wgsl(rmesh_shaders::ADAM_COMPUTE_WGSL.into()),
        });

        let adam_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("adam_bind_group_layout"),
                entries: &[
                    // @binding(0) uniforms (read)
                    storage_entry(0, true),
                    // @binding(1) params (read_write)
                    storage_entry(1, false),
                    // @binding(2) grads (read)
                    storage_entry(2, true),
                    // @binding(3) m (read_write)
                    storage_entry(3, false),
                    // @binding(4) v (read_write)
                    storage_entry(4, false),
                ],
            });

        let adam_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("adam_pipeline_layout"),
                bind_group_layouts: &[&adam_bind_group_layout],
                push_constant_ranges: &[],
            });

        let adam_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("adam_pipeline"),
                layout: Some(&adam_pipeline_layout),
                module: &adam_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            loss_pipeline,
            loss_bind_group_layout,
            backward_pipeline,
            backward_bind_group_layout_0,
            backward_bind_group_layout_1,
            adam_pipeline,
            adam_bind_group_layout,
        }
    }
}

// ---------------------------------------------------------------------------
// Bind group creation
// ---------------------------------------------------------------------------

/// Create the bind group for the loss compute pass.
///
/// Bindings:
///   0 - loss_uniforms (read)
///   1 - rendered image (read)
///   2 - ground_truth (read)
///   3 - dl_d_image (read_write)
///   4 - loss_value (read_write)
pub fn create_loss_bind_group(
    device: &wgpu::Device,
    pipelines: &BackwardPipelines,
    loss_buffers: &LossBuffers,
    rendered_buf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("loss_bind_group"),
        layout: &pipelines.loss_bind_group_layout,
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
            wgpu::BindGroupEntry {
                binding: 4,
                resource: loss_buffers.loss_value.as_entire_binding(),
            },
        ],
    })
}

/// Create bind groups for the backward compute pass.
///
/// Returns `(group0, group1)` where:
///   - group0: 11 read-only bindings (scene + loss data)
///   - group1: 4 read-write gradient output bindings
///
/// `rendered_image` is the [W x H x 4] f32 buffer from the tex-to-buffer pass.
pub fn create_backward_bind_groups(
    device: &wgpu::Device,
    pipelines: &BackwardPipelines,
    scene_buffers: &SceneBuffers,
    loss_buffers: &LossBuffers,
    grad_buffers: &GradientBuffers,
    rendered_image: &wgpu::Buffer,
) -> (wgpu::BindGroup, wgpu::BindGroup) {
    let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("backward_bind_group_0"),
        layout: &pipelines.backward_bind_group_layout_0,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: scene_buffers.uniforms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: loss_buffers.dl_d_image.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: rendered_image.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: scene_buffers.vertices.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: scene_buffers.indices.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: scene_buffers.sh_coeffs.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: scene_buffers.densities.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: scene_buffers.color_grads.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: scene_buffers.circumdata.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: scene_buffers.colors.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: scene_buffers.sort_values.as_entire_binding(),
            },
        ],
    });

    let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("backward_bind_group_1"),
        layout: &pipelines.backward_bind_group_layout_1,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: grad_buffers.d_sh_coeffs.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: grad_buffers.d_vertices.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: grad_buffers.d_densities.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: grad_buffers.d_color_grads.as_entire_binding(),
            },
        ],
    });

    (bg0, bg1)
}

/// Create a bind group for a single Adam optimizer parameter group.
///
/// Bindings:
///   0 - uniforms (read)
///   1 - params (read_write)
///   2 - grads (read)
///   3 - m (read_write)
///   4 - v (read_write)
pub fn create_adam_bind_group(
    device: &wgpu::Device,
    pipelines: &BackwardPipelines,
    uniforms_buf: &wgpu::Buffer,
    params: &wgpu::Buffer,
    grads: &wgpu::Buffer,
    m: &wgpu::Buffer,
    v: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("adam_bind_group"),
        layout: &pipelines.adam_bind_group_layout,
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

// ---------------------------------------------------------------------------
// Recording functions
// ---------------------------------------------------------------------------

/// Record the loss computation pass.
///
/// Dispatches over pixels in 16x16 workgroups.
pub fn record_loss_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &BackwardPipelines,
    loss_bg: &wgpu::BindGroup,
    width: u32,
    height: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("loss_pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipelines.loss_pipeline);
    pass.set_bind_group(0, loss_bg, &[]);
    pass.dispatch_workgroups((width + 15) / 16, (height + 15) / 16, 1);
    // pass is dropped here, ending the pass
}

/// Record the backward compute pass.
///
/// Dispatches over pixels in 16x16 workgroups. Uses two bind groups to stay
/// within the WebGPU per-bind-group buffer limit.
pub fn record_backward_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &BackwardPipelines,
    bg0: &wgpu::BindGroup,
    bg1: &wgpu::BindGroup,
    width: u32,
    height: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("backward_pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipelines.backward_pipeline);
    pass.set_bind_group(0, bg0, &[]);
    pass.set_bind_group(1, bg1, &[]);
    pass.dispatch_workgroups((width + 15) / 16, (height + 15) / 16, 1);
    // pass is dropped here, ending the pass
}

/// Record Adam optimizer updates for all parameter groups.
///
/// Each bind group corresponds to a parameter group (SH, vertices, densities,
/// color gradients). Each group is dispatched separately with 256-thread
/// workgroups.
pub fn record_adam_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &BackwardPipelines,
    adam_bgs: &[wgpu::BindGroup],
    param_counts: &[u32],
) {
    for (bg, &count) in adam_bgs.iter().zip(param_counts) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("adam_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.adam_pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups((count + 255) / 256, 1, 1);
        // pass is dropped here, ending the pass
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Helper to create a storage buffer bind group layout entry.
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
