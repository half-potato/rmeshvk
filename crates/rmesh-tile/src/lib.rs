//! Tile infrastructure shared between forward and backward tiled passes.
//!
//! Contains:
//!   - Tile buffers and pipelines (fill, ranges)
//!   - Scan pipelines (RTS prefix scan, prepare dispatch, tile gen)
//!   - Recording helpers

use rmesh_util::shared::TileUniforms;

// Re-export shared types.
pub use rmesh_util::shared::TileUniforms as TileUniformsType;

// WGSL shader sources.
const TILE_FILL_WGSL: &str = include_str!("wgsl/tile_fill_compute.wgsl");
const TILE_RANGES_WGSL: &str = include_str!("wgsl/tile_ranges_compute.wgsl");
const TILE_GEN_SCAN_WGSL: &str = include_str!("wgsl/tile_gen_scan_compute.wgsl");
const PREPARE_DISPATCH_WGSL: &str = include_str!("wgsl/prepare_dispatch.wgsl");
const RTS_WGSL: &str = include_str!("wgsl/rts.wgsl");

// ---------------------------------------------------------------------------
// Helpers (crate-local duplicates of common patterns)
// ---------------------------------------------------------------------------

pub fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

pub fn make_compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    shader: &wgpu::ShaderModule,
    bgls: &[&wgpu::BindGroupLayout],
) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label}_pl")),
        bind_group_layouts: bgls,
        immediate_size: 0,
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

/// Compute 2D dispatch dimensions that stay within the 65535 limit per dimension.
pub fn dispatch_2d(total_workgroups: u32) -> (u32, u32) {
    if total_workgroups <= 65535 {
        (total_workgroups, 1)
    } else {
        let x = 65535u32;
        let y = (total_workgroups + x - 1) / x;
        (x, y)
    }
}

fn buf_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

// ===========================================================================
// Tile buffers
// ===========================================================================

/// GPU buffers for the tiled pass.
pub struct TileBuffers {
    pub tile_sort_keys: wgpu::Buffer,
    pub tile_sort_values: wgpu::Buffer,
    pub tile_pair_count: wgpu::Buffer,
    pub tile_ranges: wgpu::Buffer,
    pub tile_uniforms: wgpu::Buffer,
    pub max_pairs_pow2: u32,
    pub tiles_x: u32,
    pub tiles_y: u32,
    pub num_tiles: u32,
}

impl TileBuffers {
    /// Allocate tile buffers.
    ///
    /// Sort buffer size is `next_power_of_two(tet_count * num_tiles)` (each tet can
    /// touch every tile in the worst case), capped to avoid excessive allocation.
    pub fn new(device: &wgpu::Device, tet_count: u32, width: u32, height: u32, tile_size: u32) -> Self {
        let tiles_x = (width + tile_size - 1) / tile_size;
        let tiles_y = (height + tile_size - 1) / tile_size;
        let num_tiles = tiles_x * tiles_y;

        // Each tet can touch up to num_tiles tiles. Use tet_count * num_tiles as the
        // upper bound, but cap per-tet estimate at 256 to avoid huge allocations.
        let tiles_per_tet = (num_tiles as u64).min(256);
        let max_pairs_pow2 = ((tet_count as u64 * tiles_per_tet).max(num_tiles as u64)).next_power_of_two() as u32;

        let tile_sort_keys = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile_sort_keys"),
            size: (max_pairs_pow2 as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let tile_sort_values = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile_sort_values"),
            size: (max_pairs_pow2 as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
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
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
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
            max_pairs_pow2,
            tiles_x,
            tiles_y,
            num_tiles,
        }
    }
}

// ===========================================================================
// Tile pipelines (fill + ranges)
// ===========================================================================

/// Pipelines for the shared tile infrastructure (fill sentinel values + compute tile ranges).
pub struct TilePipelines {
    pub fill_pipeline: wgpu::ComputePipeline,
    pub fill_bind_group_layout: wgpu::BindGroupLayout,
    pub tile_ranges_pipeline: wgpu::ComputePipeline,
    pub tile_ranges_bind_group_layout: wgpu::BindGroupLayout,
}

impl TilePipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        // ----- Tile fill pipeline (3 bindings) -----
        let fill_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tile_fill_compute"),
            source: wgpu::ShaderSource::Wgsl(TILE_FILL_WGSL.into()),
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
                immediate_size: 0,
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

        // ----- Tile ranges pipeline (4 bindings) -----
        let tile_ranges_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tile_ranges_compute"),
            source: wgpu::ShaderSource::Wgsl(TILE_RANGES_WGSL.into()),
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
                immediate_size: 0,
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

        Self {
            fill_pipeline,
            fill_bind_group_layout,
            tile_ranges_pipeline,
            tile_ranges_bind_group_layout,
        }
    }
}

// ===========================================================================
// Scan pipelines (RTS prefix scan + tile gen)
// ===========================================================================

/// Pipelines for RTS prefix scan, prepare_dispatch, and scan-based tile gen.
pub struct ScanPipelines {
    pub prepare_dispatch_pipeline: wgpu::ComputePipeline,
    pub prepare_dispatch_bgl: wgpu::BindGroupLayout,
    pub rts_reduce_pipeline: wgpu::ComputePipeline,
    pub rts_spine_scan_pipeline: wgpu::ComputePipeline,
    pub rts_downsweep_pipeline: wgpu::ComputePipeline,
    pub rts_bgl: wgpu::BindGroupLayout,
    pub tile_gen_scan_pipeline: wgpu::ComputePipeline,
    pub tile_gen_scan_bgl: wgpu::BindGroupLayout,
}

impl ScanPipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        // Prepare dispatch
        let prepare_dispatch_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prepare_dispatch"),
            source: wgpu::ShaderSource::Wgsl(PREPARE_DISPATCH_WGSL.into()),
        });
        let prepare_dispatch_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prepare_dispatch_bgl"),
            entries: &[
                storage_entry(0, true),  // indirect_args
                storage_entry(1, false), // dispatch_scan
                storage_entry(2, false), // dispatch_tile_gen
                storage_entry(3, false), // visible_count_out
                storage_entry(4, false), // rts_info
            ],
        });
        let prepare_dispatch_pipeline = make_compute_pipeline(device, "prepare_dispatch", &prepare_dispatch_shader, &[&prepare_dispatch_bgl]);

        // RTS (Reduce Then Scan) — 3 entry points sharing one bind group layout
        let rts_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rts"),
            source: wgpu::ShaderSource::Wgsl(RTS_WGSL.into()),
        });
        let rts_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rts_bgl"),
            entries: &[
                storage_entry(0, true),  // info (InfoStruct)
                storage_entry(1, false), // scan_in (tiles_touched as vec4<u32>)
                storage_entry(2, false), // scan_out (pair_offsets as vec4<u32>)
                storage_entry(3, false), // scan_bump (unused, 1 u32)
                storage_entry(4, false), // reduction (spine buffer)
                storage_entry(5, false), // misc (unused, 1 u32)
            ],
        });
        let rts_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rts_pl"),
            bind_group_layouts: &[&rts_bgl],
            immediate_size: 0,
        });
        let rts_reduce_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rts_reduce_pipeline"),
            layout: Some(&rts_layout),
            module: &rts_shader,
            entry_point: Some("reduce"),
            compilation_options: Default::default(),
            cache: None,
        });
        let rts_spine_scan_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rts_spine_scan_pipeline"),
            layout: Some(&rts_layout),
            module: &rts_shader,
            entry_point: Some("spine_scan"),
            compilation_options: Default::default(),
            cache: None,
        });
        let rts_downsweep_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rts_downsweep_pipeline"),
            layout: Some(&rts_layout),
            module: &rts_shader,
            entry_point: Some("downsweep"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Tile gen scan
        let tile_gen_scan_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tile_gen_scan"),
            source: wgpu::ShaderSource::Wgsl(TILE_GEN_SCAN_WGSL.into()),
        });
        let tile_gen_scan_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tile_gen_scan_bgl"),
            entries: &[
                storage_entry(0, true),  // tile_uniforms
                storage_entry(1, true),  // uniforms
                storage_entry(2, true),  // vertices
                storage_entry(3, true),  // indices
                storage_entry(4, true),  // compact_tet_ids
                storage_entry(5, true),  // circumdata
                storage_entry(6, false), // tile_sort_keys
                storage_entry(7, false), // tile_sort_values
                storage_entry(8, true),  // pair_offsets
                storage_entry(9, true),  // tiles_touched
                storage_entry(10, true), // visible_count
                storage_entry(11, false), // num_keys_out
            ],
        });
        let tile_gen_scan_pipeline = make_compute_pipeline(device, "tile_gen_scan", &tile_gen_scan_shader, &[&tile_gen_scan_bgl]);

        Self {
            prepare_dispatch_pipeline,
            prepare_dispatch_bgl,
            rts_reduce_pipeline,
            rts_spine_scan_pipeline,
            rts_downsweep_pipeline,
            rts_bgl,
            tile_gen_scan_pipeline,
            tile_gen_scan_bgl,
        }
    }
}

/// Buffers for RTS prefix scan and prepare_dispatch.
pub struct ScanBuffers {
    pub dispatch_scan: wgpu::Buffer,
    pub dispatch_tile_gen: wgpu::Buffer,
    pub visible_count: wgpu::Buffer,
    pub pair_offsets: wgpu::Buffer,
    pub rts_uniform: wgpu::Buffer,
    pub rts_reduction: wgpu::Buffer,
    pub rts_scan_bump: wgpu::Buffer,
    pub rts_misc: wgpu::Buffer,
}

impl ScanBuffers {
    pub fn new(device: &wgpu::Device, max_visible: u32) -> Self {
        let max_vec_size = ((max_visible as u64) + 3) / 4;
        let max_thread_blocks = ((max_vec_size + 1023) / 1024).max(1) as u32;

        let dispatch_scan = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dispatch_scan"),
            size: 12,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let dispatch_tile_gen = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dispatch_tile_gen"),
            size: 12,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let visible_count = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visible_count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let pair_offsets = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pair_offsets"),
            size: max_vec_size * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let rts_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rts_uniform"),
            size: 12,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let rts_reduction = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rts_reduction"),
            size: (max_thread_blocks as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let rts_scan_bump = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rts_scan_bump"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let rts_misc = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rts_misc"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            dispatch_scan,
            dispatch_tile_gen,
            visible_count,
            pair_offsets,
            rts_uniform,
            rts_reduction,
            rts_scan_bump,
            rts_misc,
        }
    }
}

// ---------------------------------------------------------------------------
// Bind group creation
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

/// Create the prepare_dispatch bind group.
pub fn create_prepare_dispatch_bind_group(
    device: &wgpu::Device,
    pipelines: &ScanPipelines,
    indirect_args: &wgpu::Buffer,
    scan_buffers: &ScanBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("prepare_dispatch_bg"),
        layout: &pipelines.prepare_dispatch_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: indirect_args.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: scan_buffers.dispatch_scan.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: scan_buffers.dispatch_tile_gen.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: scan_buffers.visible_count.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: scan_buffers.rts_uniform.as_entire_binding() },
        ],
    })
}

/// Create the RTS (Reduce Then Scan) bind group.
pub fn create_rts_bind_group(
    device: &wgpu::Device,
    pipelines: &ScanPipelines,
    tiles_touched: &wgpu::Buffer,
    scan_buffers: &ScanBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rts_bg"),
        layout: &pipelines.rts_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: scan_buffers.rts_uniform.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: tiles_touched.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: scan_buffers.pair_offsets.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: scan_buffers.rts_scan_bump.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: scan_buffers.rts_reduction.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: scan_buffers.rts_misc.as_entire_binding() },
        ],
    })
}

/// Create the scan-based tile gen bind group.
pub fn create_tile_gen_scan_bind_group(
    device: &wgpu::Device,
    pipelines: &ScanPipelines,
    tile_buffers: &TileBuffers,
    uniforms: &wgpu::Buffer,
    vertices: &wgpu::Buffer,
    indices: &wgpu::Buffer,
    compact_tet_ids: &wgpu::Buffer,
    circumdata: &wgpu::Buffer,
    tiles_touched: &wgpu::Buffer,
    scan_buffers: &ScanBuffers,
    num_keys_buf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tile_gen_scan_bg"),
        layout: &pipelines.tile_gen_scan_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: tile_buffers.tile_uniforms.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: uniforms.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: vertices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: indices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: compact_tet_ids.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: circumdata.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: tile_buffers.tile_sort_keys.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: tile_buffers.tile_sort_values.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: scan_buffers.pair_offsets.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: tiles_touched.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: scan_buffers.visible_count.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: num_keys_buf.as_entire_binding() },
        ],
    })
}

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------

/// Record the full scan-based tile pipeline using RTS (Reduce Then Scan).
///
/// Stages: prepare_dispatch → rts_reduce → rts_spine_scan → rts_downsweep → tile_fill → tile_gen_scan
pub fn record_scan_tile_pipeline(
    encoder: &mut wgpu::CommandEncoder,
    scan_pipelines: &ScanPipelines,
    tile_pipelines: &TilePipelines,
    prepare_dispatch_bg: &wgpu::BindGroup,
    rts_bg: &wgpu::BindGroup,
    tile_fill_bg: &wgpu::BindGroup,
    tile_gen_scan_bg: &wgpu::BindGroup,
    scan_buffers: &ScanBuffers,
    tile_buffers: &TileBuffers,
) {
    // 1. Prepare dispatch args + RTS info
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("prepare_dispatch"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&scan_pipelines.prepare_dispatch_pipeline);
        pass.set_bind_group(0, prepare_dispatch_bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    // 2. RTS reduce
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rts_reduce"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&scan_pipelines.rts_reduce_pipeline);
        pass.set_bind_group(0, rts_bg, &[]);
        pass.dispatch_workgroups_indirect(&scan_buffers.dispatch_scan, 0);
    }

    // 3. RTS spine scan (single workgroup)
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rts_spine_scan"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&scan_pipelines.rts_spine_scan_pipeline);
        pass.set_bind_group(0, rts_bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    // 4. RTS downsweep
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rts_downsweep"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&scan_pipelines.rts_downsweep_pipeline);
        pass.set_bind_group(0, rts_bg, &[]);
        pass.dispatch_workgroups_indirect(&scan_buffers.dispatch_scan, 0);
    }

    // 5. Tile fill (sentinel keys)
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tile_fill"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&tile_pipelines.fill_pipeline);
        pass.set_bind_group(0, tile_fill_bg, &[]);
        let (fx, fy) = dispatch_2d((tile_buffers.max_pairs_pow2 + 255) / 256);
        pass.dispatch_workgroups(fx, fy, 1);
    }

    // 6. Tile gen (scan-based)
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tile_gen_scan"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&scan_pipelines.tile_gen_scan_pipeline);
        pass.set_bind_group(0, tile_gen_scan_bg, &[]);
        pass.dispatch_workgroups_indirect(&scan_buffers.dispatch_tile_gen, 0);
    }
}
