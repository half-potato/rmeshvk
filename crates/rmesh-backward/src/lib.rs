//! Backward pass orchestration (wgpu port).
//!
//! Manages:
//!   - Backward compute dispatch (2 bind groups for WebGPU 8-buffer limit)
//!   - Backward tiled compute dispatch
//!   - Gradient buffer management

use rmesh_render::{MaterialBuffers, SceneBuffers};
use rmesh_shaders::shared;
use wgpu;

// Re-export shared uniform types for downstream crates.
pub use shared::TileUniforms;

// Re-export sort types (moved to rmesh-sort crate).
pub use rmesh_sort::{
    RadixSortPipelines, RadixSortState, record_radix_sort,
};

// Re-export tile types (moved to rmesh-tile crate).
pub use rmesh_tile::{
    TileBuffers, TilePipelines,
    ScanPipelines, ScanBuffers,
    create_tile_fill_bind_group, create_tile_ranges_bind_group,
    create_tile_ranges_bind_group_with_keys,
    create_prepare_dispatch_bind_group, create_rts_bind_group,
    create_tile_gen_scan_bind_group,
    record_scan_tile_pipeline,
};

// WGSL shader sources, embedded from crate-local files.
const BACKWARD_COMPUTE_WGSL: &str = include_str!("wgsl/backward_compute.wgsl");
const BACKWARD_TILED_WGSL: &str = include_str!("wgsl/backward_tiled_compute.wgsl");

/// Gradient buffers for scene geometry parameters.
pub struct GradientBuffers {
    pub d_vertices: wgpu::Buffer,
    pub d_densities: wgpu::Buffer,
}

/// Gradient buffers for material/appearance parameters.
pub struct MaterialGradBuffers {
    pub d_coeffs: wgpu::Buffer,
    pub d_color_grads: wgpu::Buffer,
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

// ---------------------------------------------------------------------------
// GradientBuffers
// ---------------------------------------------------------------------------

impl GradientBuffers {
    /// Allocate zero-initialized geometry gradient buffers.
    pub fn new(
        device: &wgpu::Device,
        vertex_count: u32,
        tet_count: u32,
    ) -> Self {
        Self {
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
        }
    }
}

impl MaterialGradBuffers {
    /// Allocate zero-initialized material gradient buffers.
    pub fn new(
        device: &wgpu::Device,
        tet_count: u32,
        sh_stride: u32,
    ) -> Self {
        Self {
            d_coeffs: create_storage_buffer(
                device,
                "d_sh_coeffs",
                (tet_count as u64) * (sh_stride as u64) * 4,
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
// BackwardPipelines (backward compute only)
// ---------------------------------------------------------------------------

/// Backward compute pipeline and its bind group layouts.
pub struct BackwardPipelines {
    pub backward_pipeline: wgpu::ComputePipeline,
    pub backward_bind_group_layout_0: wgpu::BindGroupLayout,
    pub backward_bind_group_layout_1: wgpu::BindGroupLayout,
}

impl BackwardPipelines {
    /// Create backward compute pipeline from WGSL source.
    pub fn new(device: &wgpu::Device) -> Self {
        let backward_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("backward_compute"),
            source: wgpu::ShaderSource::Wgsl(BACKWARD_COMPUTE_WGSL.into()),
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
                immediate_size: 0,
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

        Self {
            backward_pipeline,
            backward_bind_group_layout_0,
            backward_bind_group_layout_1,
        }
    }
}

// ===========================================================================
// Backward tiled pipeline
// ===========================================================================

/// Pipeline for the backward tiled pass (warp-per-tile gradient computation).
pub struct BackwardTiledPipelines {
    pub pipeline: wgpu::ComputePipeline,
    pub bg_layout_0: wgpu::BindGroupLayout,
    pub bg_layout_1: wgpu::BindGroupLayout,
}

impl BackwardTiledPipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("backward_tiled_compute"),
            source: wgpu::ShaderSource::Wgsl(BACKWARD_TILED_WGSL.into()),
        });

        // Group 0: 11 read-only bindings
        let bg_layout_0 =
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
        let bg_layout_1 =
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

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("backward_tiled_pl"),
                bind_group_layouts: &[&bg_layout_0, &bg_layout_1],
                immediate_size: 0,
            });

        let pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("backward_tiled_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self { pipeline, bg_layout_0, bg_layout_1 }
    }
}

// ---------------------------------------------------------------------------
// Bind group creation
// ---------------------------------------------------------------------------

/// Create bind groups for the backward compute pass.
///
/// Returns `(group0, group1)` where:
///   - group0: 11 read-only bindings (scene + material + loss data)
///   - group1: 4 read-write gradient output bindings
///
/// `rendered_image` is the [W x H x 4] f32 buffer from the tex-to-buffer pass.
pub fn create_backward_bind_groups(
    device: &wgpu::Device,
    pipelines: &BackwardPipelines,
    scene_buffers: &SceneBuffers,
    material: &MaterialBuffers,
    dl_d_image: &wgpu::Buffer,
    grad_buffers: &GradientBuffers,
    mat_grad_buffers: &MaterialGradBuffers,
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
                resource: dl_d_image.as_entire_binding(),
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
                resource: material.coeffs.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: scene_buffers.densities.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: material.color_grads.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: scene_buffers.circumdata.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: material.colors.as_entire_binding(),
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
                resource: mat_grad_buffers.d_coeffs.as_entire_binding(),
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
                resource: mat_grad_buffers.d_color_grads.as_entire_binding(),
            },
        ],
    });

    (bg0, bg1)
}

/// Create the backward tiled bind groups.
///
/// Returns `(bg0, bg1)`.
pub fn create_backward_tiled_bind_groups(
    device: &wgpu::Device,
    pipelines: &BackwardTiledPipelines,
    uniforms: &wgpu::Buffer,
    dl_d_image: &wgpu::Buffer,
    rendered_image: &wgpu::Buffer,
    vertices: &wgpu::Buffer,
    indices: &wgpu::Buffer,
    sh_coeffs: &wgpu::Buffer,
    densities: &wgpu::Buffer,
    color_grads: &wgpu::Buffer,
    circumdata: &wgpu::Buffer,
    colors: &wgpu::Buffer,
    tile_sort_values: &wgpu::Buffer,
    d_sh_coeffs: &wgpu::Buffer,
    d_vertices: &wgpu::Buffer,
    d_densities: &wgpu::Buffer,
    d_color_grads: &wgpu::Buffer,
    tile_ranges: &wgpu::Buffer,
    tile_uniforms: &wgpu::Buffer,
    debug_image: &wgpu::Buffer,
) -> (wgpu::BindGroup, wgpu::BindGroup) {
    let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("backward_tiled_bg_0"),
        layout: &pipelines.bg_layout_0,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dl_d_image.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: rendered_image.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: vertices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: indices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: sh_coeffs.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: densities.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: color_grads.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: circumdata.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: colors.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: tile_sort_values.as_entire_binding() },
        ],
    });

    let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("backward_tiled_bg_1"),
        layout: &pipelines.bg_layout_1,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: d_sh_coeffs.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: d_vertices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: d_densities.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: d_color_grads.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: tile_ranges.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: tile_uniforms.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: debug_image.as_entire_binding() },
        ],
    });

    (bg0, bg1)
}

// ---------------------------------------------------------------------------
// Recording functions
// ---------------------------------------------------------------------------

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
