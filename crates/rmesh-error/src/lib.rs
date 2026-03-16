//! Per-tet error statistics accumulation (wgpu compute).
//!
//! Mirrors the tiled traversal of `rasterize_compute.wgsl` but accumulates
//! per-tet error statistics via `atomic<f32>` instead of compositing colors.
//! Used during densification to decide which vertices to clone/split.

use rmesh_tile::{dispatch_2d, storage_entry};

const ERROR_TILED_WGSL: &str = include_str!("wgsl/error_tiled_compute.wgsl");

// ---------------------------------------------------------------------------
// Buffer types
// ---------------------------------------------------------------------------

/// Per-tet error accumulation buffers (atomic outputs).
pub struct ErrorBuffers {
    /// `[T * 16]` f32 atomic — per-tet error statistics.
    pub tet_err: wgpu::Buffer,
    /// `[T * 2]` i32 atomic — per-tet pixel count + max contribution.
    pub tet_count: wgpu::Buffer,
}

/// Per-pixel error input buffers (uploaded from CPU each call).
pub struct ErrorInputBuffers {
    /// `[H * W]` f32 — per-pixel L1 error.
    pub pixel_err: wgpu::Buffer,
    /// `[H * W]` f32 — per-pixel SSIM error.
    pub ssim_err: wgpu::Buffer,
}

// ---------------------------------------------------------------------------
// Buffer creation helpers
// ---------------------------------------------------------------------------

fn create_storage_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

impl ErrorBuffers {
    pub fn new(device: &wgpu::Device, tet_count: u32) -> Self {
        Self {
            tet_err: create_storage_buffer(
                device,
                "tet_err",
                (tet_count as u64) * 16 * 4,
            ),
            tet_count: create_storage_buffer(
                device,
                "tet_count",
                (tet_count as u64) * 2 * 4,
            ),
        }
    }
}

impl ErrorInputBuffers {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let n_pixels = (width as u64) * (height as u64);
        Self {
            pixel_err: create_storage_buffer(device, "pixel_err", n_pixels * 4),
            ssim_err: create_storage_buffer(device, "ssim_err", n_pixels * 4),
        }
    }
}

// ===========================================================================
// Error tiled pipeline
// ===========================================================================

/// Pipeline for the error tiled pass (per-tet error statistics).
pub struct ErrorPipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub bg_layout_0: wgpu::BindGroupLayout,
    pub bg_layout_1: wgpu::BindGroupLayout,
}

impl ErrorPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("error_tiled_compute"),
            source: wgpu::ShaderSource::Wgsl(ERROR_TILED_WGSL.into()),
        });

        // Group 0: 9 read-only bindings (scene + tile data)
        let bg_layout_0 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("error_tiled_bgl_0"),
                entries: &[
                    storage_entry(0, true),  // uniforms
                    storage_entry(1, true),  // vertices
                    storage_entry(2, true),  // indices
                    storage_entry(3, true),  // colors_buf
                    storage_entry(4, true),  // densities
                    storage_entry(5, true),  // color_grads
                    storage_entry(6, true),  // tile_sort_values
                    storage_entry(7, true),  // tile_ranges
                    storage_entry(8, true),  // tile_uniforms
                ],
            });

        // Group 1: 2 read + 2 read-write
        let bg_layout_1 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("error_tiled_bgl_1"),
                entries: &[
                    storage_entry(0, true),  // pixel_err
                    storage_entry(1, true),  // ssim_err
                    storage_entry(2, false), // tet_err (rw atomic)
                    storage_entry(3, false), // tet_count (rw atomic)
                ],
            });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("error_tiled_pl"),
                bind_group_layouts: &[&bg_layout_0, &bg_layout_1],
                immediate_size: 0,
            });

        let pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("error_tiled_pipeline"),
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

/// Create the error tiled group 0 bind group (scene + tile data).
///
/// Call twice with different `tile_sort_values` for A/B sort buffer variants.
pub fn create_error_bg0(
    device: &wgpu::Device,
    pipeline: &ErrorPipeline,
    uniforms: &wgpu::Buffer,
    vertices: &wgpu::Buffer,
    indices: &wgpu::Buffer,
    colors: &wgpu::Buffer,
    densities: &wgpu::Buffer,
    color_grads: &wgpu::Buffer,
    tile_sort_values: &wgpu::Buffer,
    tile_ranges: &wgpu::Buffer,
    tile_uniforms: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("error_tiled_bg_0"),
        layout: &pipeline.bg_layout_0,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: vertices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: indices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: colors.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: densities.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: color_grads.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: tile_sort_values.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: tile_ranges.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: tile_uniforms.as_entire_binding() },
        ],
    })
}

/// Create the error tiled group 1 bind group (error inputs + outputs).
pub fn create_error_bg1(
    device: &wgpu::Device,
    pipeline: &ErrorPipeline,
    input_buffers: &ErrorInputBuffers,
    error_buffers: &ErrorBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("error_tiled_bg_1"),
        layout: &pipeline.bg_layout_1,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input_buffers.pixel_err.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: input_buffers.ssim_err.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: error_buffers.tet_err.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: error_buffers.tet_count.as_entire_binding() },
        ],
    })
}

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------

/// Record the error tiled compute pass.
///
/// Dispatches 1 workgroup (32 threads) per tile.
pub fn record_error_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &ErrorPipeline,
    bg0: &wgpu::BindGroup,
    bg1: &wgpu::BindGroup,
    num_tiles: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("error_tiled"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipeline.pipeline);
    pass.set_bind_group(0, bg0, &[]);
    pass.set_bind_group(1, bg1, &[]);
    let (x, y) = dispatch_2d(num_tiles);
    pass.dispatch_workgroups(x, y, 1);
}
