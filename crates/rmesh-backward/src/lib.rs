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
pub use shared::TileUniforms;

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

// ===========================================================================
// Tiled backward pass infrastructure
// ===========================================================================

/// GPU buffers for the tiled backward pass.
pub struct TileBuffers {
    pub tile_sort_keys: wgpu::Buffer,
    pub tile_sort_values: wgpu::Buffer,
    pub tile_pair_count: wgpu::Buffer,
    pub tile_ranges: wgpu::Buffer,
    pub tile_uniforms: wgpu::Buffer,
    pub max_pairs: u32,
    pub max_pairs_pow2: u32,
    pub tiles_x: u32,
    pub tiles_y: u32,
    pub num_tiles: u32,
}

impl TileBuffers {
    /// Allocate tile buffers.
    ///
    /// `max_pairs = tet_count * 16` — estimated max tile-tet pairs.
    pub fn new(device: &wgpu::Device, tet_count: u32, width: u32, height: u32, tile_size: u32) -> Self {
        let tiles_x = (width + tile_size - 1) / tile_size;
        let tiles_y = (height + tile_size - 1) / tile_size;
        let num_tiles = tiles_x * tiles_y;

        let max_pairs = tet_count * 16;
        let max_pairs_pow2 = (max_pairs as u64).next_power_of_two() as u32;

        let tile_sort_keys = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile_sort_keys"),
            size: (max_pairs_pow2 as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tile_sort_values = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile_sort_values"),
            size: (max_pairs_pow2 as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tile_pair_count = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile_pair_count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let tile_ranges = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile_ranges"),
            size: (num_tiles as u64) * 2 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tile_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile_uniforms"),
            size: std::mem::size_of::<TileUniforms>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            tile_sort_keys,
            tile_sort_values,
            tile_pair_count,
            tile_ranges,
            tile_uniforms,
            max_pairs,
            max_pairs_pow2,
            tiles_x,
            tiles_y,
            num_tiles,
        }
    }
}

/// Pipelines for the tiled backward pass.
pub struct TilePipelines {
    pub fill_pipeline: wgpu::ComputePipeline,
    pub fill_bind_group_layout: wgpu::BindGroupLayout,
    pub tile_gen_pipeline: wgpu::ComputePipeline,
    pub tile_gen_bind_group_layout: wgpu::BindGroupLayout,
    pub tile_ranges_pipeline: wgpu::ComputePipeline,
    pub tile_ranges_bind_group_layout: wgpu::BindGroupLayout,
    pub backward_tiled_pipeline: wgpu::ComputePipeline,
    pub backward_tiled_bg_layout_0: wgpu::BindGroupLayout,
    pub backward_tiled_bg_layout_1: wgpu::BindGroupLayout,
}

impl TilePipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        // ----- Tile fill pipeline (3 bindings) -----
        let fill_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tile_fill_compute"),
            source: wgpu::ShaderSource::Wgsl(rmesh_shaders::TILE_FILL_WGSL.into()),
        });
        let fill_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tile_fill_bgl"),
                entries: &[
                    storage_entry(0, true),  // tile_uniforms
                    storage_entry(1, false), // tile_sort_keys
                    storage_entry(2, false), // tile_sort_values
                ],
            });
        let fill_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("tile_fill_pl"),
                bind_group_layouts: &[&fill_bind_group_layout],
                push_constant_ranges: &[],
            });
        let fill_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("tile_fill_pipeline"),
                layout: Some(&fill_pipeline_layout),
                module: &fill_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // ----- Tile gen pipeline (9 bindings) -----
        let tile_gen_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tile_gen_compute"),
            source: wgpu::ShaderSource::Wgsl(rmesh_shaders::TILE_GEN_WGSL.into()),
        });
        let tile_gen_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tile_gen_bgl"),
                entries: &[
                    storage_entry(0, true),  // tile_uniforms
                    storage_entry(1, true),  // main uniforms
                    storage_entry(2, true),  // vertices
                    storage_entry(3, true),  // indices
                    storage_entry(4, true),  // sorted_indices
                    storage_entry(5, true),  // circumdata
                    storage_entry(6, false), // tile_sort_keys
                    storage_entry(7, false), // tile_sort_values
                    storage_entry(8, false), // tile_pair_count
                ],
            });
        let tile_gen_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("tile_gen_pl"),
                bind_group_layouts: &[&tile_gen_bind_group_layout],
                push_constant_ranges: &[],
            });
        let tile_gen_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("tile_gen_pipeline"),
                layout: Some(&tile_gen_pipeline_layout),
                module: &tile_gen_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // ----- Tile ranges pipeline (4 bindings) -----
        let tile_ranges_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tile_ranges_compute"),
            source: wgpu::ShaderSource::Wgsl(rmesh_shaders::TILE_RANGES_WGSL.into()),
        });
        let tile_ranges_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tile_ranges_bgl"),
                entries: &[
                    storage_entry(0, true),  // tile_sort_keys
                    storage_entry(1, false), // tile_ranges
                    storage_entry(2, true),  // tile_uniforms
                    storage_entry(3, true),  // tile_pair_count
                ],
            });
        let tile_ranges_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("tile_ranges_pl"),
                bind_group_layouts: &[&tile_ranges_bind_group_layout],
                push_constant_ranges: &[],
            });
        let tile_ranges_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("tile_ranges_pipeline"),
                layout: Some(&tile_ranges_pipeline_layout),
                module: &tile_ranges_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // ----- Backward tiled pipeline (2 bind groups: 11 + 6) -----
        let backward_tiled_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("backward_tiled_compute"),
            source: wgpu::ShaderSource::Wgsl(rmesh_shaders::BACKWARD_TILED_WGSL.into()),
        });

        // Group 0: 11 read-only bindings
        let backward_tiled_bg_layout_0 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("backward_tiled_bgl_0"),
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
                    storage_entry(9, true),  // colors_buf
                    storage_entry(10, true), // tile_sort_values
                ],
            });

        // Group 1: 7 bindings (5 rw + 2 read)
        let backward_tiled_bg_layout_1 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("backward_tiled_bgl_1"),
                entries: &[
                    storage_entry(0, false), // d_sh_coeffs
                    storage_entry(1, false), // d_vertices
                    storage_entry(2, false), // d_densities
                    storage_entry(3, false), // d_color_grads
                    storage_entry(4, true),  // tile_ranges
                    storage_entry(5, true),  // tile_uniforms
                    storage_entry(6, false), // debug_image
                ],
            });

        let backward_tiled_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("backward_tiled_pl"),
                bind_group_layouts: &[&backward_tiled_bg_layout_0, &backward_tiled_bg_layout_1],
                push_constant_ranges: &[],
            });

        let backward_tiled_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("backward_tiled_pipeline"),
                layout: Some(&backward_tiled_pipeline_layout),
                module: &backward_tiled_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            fill_pipeline,
            fill_bind_group_layout,
            tile_gen_pipeline,
            tile_gen_bind_group_layout,
            tile_ranges_pipeline,
            tile_ranges_bind_group_layout,
            backward_tiled_pipeline,
            backward_tiled_bg_layout_0,
            backward_tiled_bg_layout_1,
        }
    }
}

// ---------------------------------------------------------------------------
// Tile bind group creation
// ---------------------------------------------------------------------------

pub fn create_tile_fill_bind_group(
    device: &wgpu::Device,
    pipelines: &TilePipelines,
    tile_buffers: &TileBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tile_fill_bg"),
        layout: &pipelines.fill_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: tile_buffers.tile_uniforms.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: tile_buffers.tile_sort_keys.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: tile_buffers.tile_sort_values.as_entire_binding() },
        ],
    })
}

pub fn create_tile_gen_bind_group(
    device: &wgpu::Device,
    pipelines: &TilePipelines,
    tile_buffers: &TileBuffers,
    scene_buffers: &SceneBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tile_gen_bg"),
        layout: &pipelines.tile_gen_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: tile_buffers.tile_uniforms.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: scene_buffers.uniforms.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: scene_buffers.vertices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: scene_buffers.indices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: scene_buffers.sort_values.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: scene_buffers.circumdata.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: tile_buffers.tile_sort_keys.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: tile_buffers.tile_sort_values.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: tile_buffers.tile_pair_count.as_entire_binding() },
        ],
    })
}

pub fn create_tile_ranges_bind_group(
    device: &wgpu::Device,
    pipelines: &TilePipelines,
    tile_buffers: &TileBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tile_ranges_bg"),
        layout: &pipelines.tile_ranges_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: tile_buffers.tile_sort_keys.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: tile_buffers.tile_ranges.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: tile_buffers.tile_uniforms.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: tile_buffers.tile_pair_count.as_entire_binding() },
        ],
    })
}

/// Create the backward tiled bind groups using the A (primary) sort buffers.
///
/// Returns `(bg0, bg1)`.
pub fn create_backward_tiled_bind_groups(
    device: &wgpu::Device,
    pipelines: &TilePipelines,
    scene_buffers: &SceneBuffers,
    loss_buffers: &LossBuffers,
    grad_buffers: &GradientBuffers,
    rendered_image: &wgpu::Buffer,
    tile_sort_values: &wgpu::Buffer,
    tile_ranges: &wgpu::Buffer,
    tile_uniforms: &wgpu::Buffer,
    debug_image: &wgpu::Buffer,
) -> (wgpu::BindGroup, wgpu::BindGroup) {
    let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("backward_tiled_bg_0"),
        layout: &pipelines.backward_tiled_bg_layout_0,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: scene_buffers.uniforms.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: loss_buffers.dl_d_image.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: rendered_image.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: scene_buffers.vertices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: scene_buffers.indices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: scene_buffers.sh_coeffs.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: scene_buffers.densities.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: scene_buffers.color_grads.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: scene_buffers.circumdata.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: scene_buffers.colors.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: tile_sort_values.as_entire_binding() },
        ],
    });

    let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("backward_tiled_bg_1"),
        layout: &pipelines.backward_tiled_bg_layout_1,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: grad_buffers.d_sh_coeffs.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: grad_buffers.d_vertices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: grad_buffers.d_densities.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: grad_buffers.d_color_grads.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: tile_ranges.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: tile_uniforms.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: debug_image.as_entire_binding() },
        ],
    });

    (bg0, bg1)
}

/// Create tile_ranges bind group pointing at specific sort key buffer.
pub fn create_tile_ranges_bind_group_with_keys(
    device: &wgpu::Device,
    pipelines: &TilePipelines,
    tile_sort_keys: &wgpu::Buffer,
    tile_ranges: &wgpu::Buffer,
    tile_uniforms: &wgpu::Buffer,
    tile_pair_count: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tile_ranges_bg"),
        layout: &pipelines.tile_ranges_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: tile_sort_keys.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: tile_ranges.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: tile_uniforms.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: tile_pair_count.as_entire_binding() },
        ],
    })
}

// ===========================================================================
// Radix sort infrastructure (replaces bitonic tile sort)
// ===========================================================================

/// Radix sort constants (must match WGSL shaders).
const RADIX_WG: u32 = 256;
const RADIX_ELEMENTS_PER_THREAD: u32 = 4;
const RADIX_BLOCK_SIZE: u32 = RADIX_WG * RADIX_ELEMENTS_PER_THREAD; // 1024
const RADIX_BIN_COUNT: u32 = 16;

/// Pipelines for the 5-stage radix sort.
pub struct RadixSortPipelines {
    pub count_pipeline: wgpu::ComputePipeline,
    pub count_bgl: wgpu::BindGroupLayout,
    pub reduce_pipeline: wgpu::ComputePipeline,
    pub reduce_bgl: wgpu::BindGroupLayout,
    pub scan_pipeline: wgpu::ComputePipeline,
    pub scan_bgl: wgpu::BindGroupLayout,
    pub scan_add_pipeline: wgpu::ComputePipeline,
    pub scan_add_bgl: wgpu::BindGroupLayout,
    pub scatter_pipeline: wgpu::ComputePipeline,
    pub scatter_bgl: wgpu::BindGroupLayout,
}

impl RadixSortPipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        // Count: config(r), num_keys(r), src(r), counts(rw)
        let count_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("radix_sort_count"),
            source: wgpu::ShaderSource::Wgsl(rmesh_shaders::RADIX_SORT_COUNT_WGSL.into()),
        });
        let count_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("radix_count_bgl"),
            entries: &[
                storage_entry(0, true),  // config
                storage_entry(1, true),  // num_keys
                storage_entry(2, true),  // src
                storage_entry(3, false), // counts
            ],
        });
        let count_pipeline = make_compute_pipeline(device, "radix_count", &count_shader, &[&count_bgl]);

        // Reduce: num_keys(r), counts(r), reduced(rw)
        let reduce_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("radix_sort_reduce"),
            source: wgpu::ShaderSource::Wgsl(rmesh_shaders::RADIX_SORT_REDUCE_WGSL.into()),
        });
        let reduce_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("radix_reduce_bgl"),
            entries: &[
                storage_entry(0, true),  // num_keys
                storage_entry(1, true),  // counts
                storage_entry(2, false), // reduced
            ],
        });
        let reduce_pipeline = make_compute_pipeline(device, "radix_reduce", &reduce_shader, &[&reduce_bgl]);

        // Scan: num_keys(r), reduced(rw)
        let scan_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("radix_sort_scan"),
            source: wgpu::ShaderSource::Wgsl(rmesh_shaders::RADIX_SORT_SCAN_WGSL.into()),
        });
        let scan_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("radix_scan_bgl"),
            entries: &[
                storage_entry(0, true),  // num_keys
                storage_entry(1, false), // reduced
            ],
        });
        let scan_pipeline = make_compute_pipeline(device, "radix_scan", &scan_shader, &[&scan_bgl]);

        // ScanAdd: num_keys(r), reduced(r), counts(rw)
        let scan_add_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("radix_sort_scan_add"),
            source: wgpu::ShaderSource::Wgsl(rmesh_shaders::RADIX_SORT_SCAN_ADD_WGSL.into()),
        });
        let scan_add_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("radix_scan_add_bgl"),
            entries: &[
                storage_entry(0, true),  // num_keys
                storage_entry(1, true),  // reduced
                storage_entry(2, false), // counts
            ],
        });
        let scan_add_pipeline = make_compute_pipeline(device, "radix_scan_add", &scan_add_shader, &[&scan_add_bgl]);

        // Scatter: config(r), num_keys(r), src(r), values(r), counts(r), out(rw), out_values(rw)
        let scatter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("radix_sort_scatter"),
            source: wgpu::ShaderSource::Wgsl(rmesh_shaders::RADIX_SORT_SCATTER_WGSL.into()),
        });
        let scatter_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("radix_scatter_bgl"),
            entries: &[
                storage_entry(0, true),  // config
                storage_entry(1, true),  // num_keys
                storage_entry(2, true),  // src
                storage_entry(3, true),  // values
                storage_entry(4, true),  // counts
                storage_entry(5, false), // out
                storage_entry(6, false), // out_values
            ],
        });
        let scatter_pipeline = make_compute_pipeline(device, "radix_scatter", &scatter_shader, &[&scatter_bgl]);

        Self {
            count_pipeline,
            count_bgl,
            reduce_pipeline,
            reduce_bgl,
            scan_pipeline,
            scan_bgl,
            scan_add_pipeline,
            scan_add_bgl,
            scatter_pipeline,
            scatter_bgl,
        }
    }
}

/// Buffers and bind groups for the radix sort.
pub struct RadixSortState {
    /// Ping-pong key buffers (A = primary from TileBuffers, B = alternate)
    pub keys_b: wgpu::Buffer,
    pub values_b: wgpu::Buffer,
    /// Per-workgroup histogram counts
    pub counts: wgpu::Buffer,
    /// Reduced partial sums
    pub reduced: wgpu::Buffer,
    /// Number of keys to sort (1 u32, written by CPU before sort)
    pub num_keys_buf: wgpu::Buffer,
    /// Per-pass config uniform buffers (one per pass, contains shift amount)
    pub config_buffers: Vec<wgpu::Buffer>,
    /// Max workgroups needed
    pub max_num_wgs: u32,
    /// Number of sorting bits (32 for full sort, can be less)
    pub sorting_bits: u32,
}

impl RadixSortState {
    pub fn new(device: &wgpu::Device, max_pairs: u32, sorting_bits: u32) -> Self {
        let max_num_wgs = (max_pairs + RADIX_BLOCK_SIZE - 1) / RADIX_BLOCK_SIZE;

        let keys_b = create_storage_buffer(device, "radix_keys_b", (max_pairs as u64) * 4);
        let values_b = create_storage_buffer(device, "radix_values_b", (max_pairs as u64) * 4);

        // counts: [BIN_COUNT * max_num_wgs] u32
        let counts = create_storage_buffer(
            device,
            "radix_counts",
            (RADIX_BIN_COUNT as u64) * (max_num_wgs as u64) * 4,
        );

        // reduced: [BLOCK_SIZE] u32 (enough for single-workgroup scan)
        let reduced = create_storage_buffer(device, "radix_reduced", (RADIX_BLOCK_SIZE as u64) * 4);

        let num_keys_buf = create_storage_buffer(device, "radix_num_keys", 4);

        let num_passes = (sorting_bits + 3) / 4;
        let config_buffers: Vec<wgpu::Buffer> = (0..num_passes)
            .map(|pass| {
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("radix_config_{pass}")),
                    size: 4, // single u32
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                // We'll write the shift value at init time
                buf
            })
            .collect();

        Self {
            keys_b,
            values_b,
            counts,
            reduced,
            num_keys_buf,
            config_buffers,
            max_num_wgs,
            sorting_bits,
        }
    }

    /// Write per-pass config (shift values) to GPU buffers. Call once at init.
    pub fn upload_configs(&self, queue: &wgpu::Queue) {
        let num_passes = (self.sorting_bits + 3) / 4;
        for pass in 0..num_passes {
            let shift = pass * 4;
            queue.write_buffer(&self.config_buffers[pass as usize], 0, bytemuck::bytes_of(&shift));
        }
    }
}

/// Compute 2D dispatch dimensions that stay within the 65535 limit per dimension.
fn dispatch_2d(total_workgroups: u32) -> (u32, u32) {
    if total_workgroups <= 65535 {
        (total_workgroups, 1)
    } else {
        let x = 65535u32;
        let y = (total_workgroups + x - 1) / x;
        (x, y)
    }
}

/// Record a complete radix sort of tile_sort_keys/tile_sort_values.
///
/// After sorting, the result ends up in either the primary (A) or alternate (B) buffers
/// depending on the number of passes. Returns `true` if result is in B buffers.
pub fn record_radix_sort(
    encoder: &mut wgpu::CommandEncoder,
    device: &wgpu::Device,
    pipelines: &RadixSortPipelines,
    state: &RadixSortState,
    keys_a: &wgpu::Buffer,
    values_a: &wgpu::Buffer,
) -> bool {
    let num_passes = (state.sorting_bits + 3) / 4;
    let max_num_wgs = state.max_num_wgs;
    let num_reduce_wgs = RADIX_BIN_COUNT * ((max_num_wgs + RADIX_BLOCK_SIZE - 1) / RADIX_BLOCK_SIZE);

    let (count_dx, count_dy) = dispatch_2d(max_num_wgs);
    let (reduce_dx, reduce_dy) = dispatch_2d(num_reduce_wgs);

    // We alternate between A and B buffers each pass
    for pass in 0..num_passes {
        let even = pass % 2 == 0;
        let (src_keys, src_vals, dst_keys, dst_vals) = if even {
            (keys_a, values_a, &state.keys_b, &state.values_b)
        } else {
            (&state.keys_b, &state.values_b, keys_a, values_a)
        };

        let config_buf = &state.config_buffers[pass as usize];

        // 1. Count
        {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("radix_count_bg"),
                layout: &pipelines.count_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: config_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: state.num_keys_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: src_keys.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: state.counts.as_entire_binding() },
                ],
            });
            let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("radix_count"),
                timestamp_writes: None,
            });
            p.set_pipeline(&pipelines.count_pipeline);
            p.set_bind_group(0, &bg, &[]);
            p.dispatch_workgroups(count_dx, count_dy, 1);
        }

        // 2. Reduce
        {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("radix_reduce_bg"),
                layout: &pipelines.reduce_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: state.num_keys_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: state.counts.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: state.reduced.as_entire_binding() },
                ],
            });
            let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("radix_reduce"),
                timestamp_writes: None,
            });
            p.set_pipeline(&pipelines.reduce_pipeline);
            p.set_bind_group(0, &bg, &[]);
            p.dispatch_workgroups(reduce_dx, reduce_dy, 1);
        }

        // 3. Scan (single workgroup)
        {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("radix_scan_bg"),
                layout: &pipelines.scan_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: state.num_keys_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: state.reduced.as_entire_binding() },
                ],
            });
            let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("radix_scan"),
                timestamp_writes: None,
            });
            p.set_pipeline(&pipelines.scan_pipeline);
            p.set_bind_group(0, &bg, &[]);
            p.dispatch_workgroups(1, 1, 1);
        }

        // 4. Scan Add
        {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("radix_scan_add_bg"),
                layout: &pipelines.scan_add_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: state.num_keys_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: state.reduced.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: state.counts.as_entire_binding() },
                ],
            });
            let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("radix_scan_add"),
                timestamp_writes: None,
            });
            p.set_pipeline(&pipelines.scan_add_pipeline);
            p.set_bind_group(0, &bg, &[]);
            p.dispatch_workgroups(reduce_dx, reduce_dy, 1);
        }

        // 5. Scatter
        {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("radix_scatter_bg"),
                layout: &pipelines.scatter_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: config_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: state.num_keys_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: src_keys.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: src_vals.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: state.counts.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: dst_keys.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: dst_vals.as_entire_binding() },
                ],
            });
            let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("radix_scatter"),
                timestamp_writes: None,
            });
            p.set_pipeline(&pipelines.scatter_pipeline);
            p.set_bind_group(0, &bg, &[]);
            p.dispatch_workgroups(count_dx, count_dy, 1);
        }
    }

    // Return true if result ended up in B buffers (odd number of passes)
    num_passes % 2 != 0
}

// ---------------------------------------------------------------------------
// Tiled backward recording
// ---------------------------------------------------------------------------

/// Record the full tiled backward pass.
///
/// Stages: tile_fill → tile_gen → radix_sort → tile_ranges → backward_tiled
pub fn record_tiled_backward(
    encoder: &mut wgpu::CommandEncoder,
    device: &wgpu::Device,
    tile_pipelines: &TilePipelines,
    radix_pipelines: &RadixSortPipelines,
    radix_state: &RadixSortState,
    tile_fill_bg: &wgpu::BindGroup,
    tile_gen_bg: &wgpu::BindGroup,
    tile_ranges_bg: &wgpu::BindGroup,
    bwd_tiled_bg0: &wgpu::BindGroup,
    bwd_tiled_bg1: &wgpu::BindGroup,
    tile_buffers: &TileBuffers,
    // Alternate bind groups for when sort result ends up in B buffers
    tile_ranges_bg_b: &wgpu::BindGroup,
    bwd_tiled_bg0_b: &wgpu::BindGroup,
) {
    // 0. Fill sort buffers with sentinels (0xFFFFFFFF keys)
    // Required so the radix sort (which processes max_pairs entries) doesn't
    // mix stale/random data with the valid tile-tet pairs from tile_gen.
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tile_fill"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&tile_pipelines.fill_pipeline);
        pass.set_bind_group(0, tile_fill_bg, &[]);
        pass.dispatch_workgroups((tile_buffers.max_pairs_pow2 + 255) / 256, 1, 1);
    }

    // 1. Generate tile-tet pairs
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tile_gen"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&tile_pipelines.tile_gen_pipeline);
        pass.set_bind_group(0, tile_gen_bg, &[]);
        let tet_count_upper = tile_buffers.max_pairs / 16;
        pass.dispatch_workgroups((tet_count_upper + 63) / 64, 1, 1);
    }

    // 2. Radix sort tile keys/values
    let result_in_b = record_radix_sort(
        encoder,
        device,
        radix_pipelines,
        radix_state,
        &tile_buffers.tile_sort_keys,
        &tile_buffers.tile_sort_values,
    );

    // Pick bind groups based on which buffers hold the sorted result
    let (ranges_bg, bwd_bg0) = if result_in_b {
        (tile_ranges_bg_b, bwd_tiled_bg0_b)
    } else {
        (tile_ranges_bg, bwd_tiled_bg0)
    };

    // 3. Compute per-tile ranges
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tile_ranges"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&tile_pipelines.tile_ranges_pipeline);
        pass.set_bind_group(0, ranges_bg, &[]);
        pass.dispatch_workgroups((tile_buffers.max_pairs + 255) / 256, 1, 1);
    }

    // 4. Tiled backward pass
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("backward_tiled"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&tile_pipelines.backward_tiled_pipeline);
        pass.set_bind_group(0, bwd_bg0, &[]);
        pass.set_bind_group(1, bwd_tiled_bg1, &[]);
        pass.dispatch_workgroups(tile_buffers.tiles_x, tile_buffers.tiles_y, 1);
    }
}

/// Helper to create a compute pipeline from a single bind group layout.
fn make_compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    shader: &wgpu::ShaderModule,
    bgls: &[&wgpu::BindGroupLayout],
) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label}_pl")),
        bind_group_layouts: bgls,
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&format!("{label}_pipeline")),
        layout: Some(&layout),
        module: shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}
