//! Sort infrastructure shared between forward and backward passes.
//!
//! Contains:
//!   - 5-pass radix sort pipeline, state, and recording (radix_sort_*.wgsl)
const RADIX_SORT_COUNT_WGSL: &str = include_str!("wgsl/radix_sort_count.wgsl");
const RADIX_SORT_REDUCE_WGSL: &str = include_str!("wgsl/radix_sort_reduce.wgsl");
const RADIX_SORT_SCAN_WGSL: &str = include_str!("wgsl/radix_sort_scan.wgsl");
const RADIX_SORT_SCAN_ADD_WGSL: &str = include_str!("wgsl/radix_sort_scan_add.wgsl");
const RADIX_SORT_SCATTER_WGSL: &str = include_str!("wgsl/radix_sort_scatter.wgsl");

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Helper to create a storage buffer bind group layout entry.
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

/// Helper to create a compute pipeline from a single shader and bind group layouts.
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

/// Create a zero-initialized storage buffer with COPY_DST and COPY_SRC.
pub fn create_storage_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

// ===========================================================================
// 5-pass radix sort
// ===========================================================================

/// Radix sort constants (must match WGSL shaders).
pub const RADIX_WG: u32 = 256;
pub const RADIX_ELEMENTS_PER_THREAD: u32 = 4;
pub const RADIX_BLOCK_SIZE: u32 = RADIX_WG * RADIX_ELEMENTS_PER_THREAD; // 1024
pub const RADIX_BIN_COUNT: u32 = 16;

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
            source: wgpu::ShaderSource::Wgsl(RADIX_SORT_COUNT_WGSL.into()),
        });
        let count_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("radix_count_bgl"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
            ],
        });
        let count_pipeline = make_compute_pipeline(device, "radix_count", &count_shader, &[&count_bgl]);

        // Reduce: num_keys(r), counts(r), reduced(rw)
        let reduce_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("radix_sort_reduce"),
            source: wgpu::ShaderSource::Wgsl(RADIX_SORT_REDUCE_WGSL.into()),
        });
        let reduce_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("radix_reduce_bgl"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
            ],
        });
        let reduce_pipeline = make_compute_pipeline(device, "radix_reduce", &reduce_shader, &[&reduce_bgl]);

        // Scan: num_keys(r), reduced(rw)
        let scan_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("radix_sort_scan"),
            source: wgpu::ShaderSource::Wgsl(RADIX_SORT_SCAN_WGSL.into()),
        });
        let scan_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("radix_scan_bgl"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, false),
            ],
        });
        let scan_pipeline = make_compute_pipeline(device, "radix_scan", &scan_shader, &[&scan_bgl]);

        // ScanAdd: num_keys(r), reduced(r), counts(rw)
        let scan_add_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("radix_sort_scan_add"),
            source: wgpu::ShaderSource::Wgsl(RADIX_SORT_SCAN_ADD_WGSL.into()),
        });
        let scan_add_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("radix_scan_add_bgl"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
            ],
        });
        let scan_add_pipeline = make_compute_pipeline(device, "radix_scan_add", &scan_add_shader, &[&scan_add_bgl]);

        // Scatter: config(r), num_keys(r), src(r), values(r), counts(r), out(rw), out_values(rw)
        let scatter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("radix_sort_scatter"),
            source: wgpu::ShaderSource::Wgsl(RADIX_SORT_SCATTER_WGSL.into()),
        });
        let scatter_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("radix_scatter_bgl"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, false),
                storage_entry(6, false),
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
    pub keys_b: wgpu::Buffer,
    pub values_b: wgpu::Buffer,
    pub counts: wgpu::Buffer,
    pub reduced: wgpu::Buffer,
    pub num_keys_buf: wgpu::Buffer,
    pub config_buffers: Vec<wgpu::Buffer>,
    pub max_num_wgs: u32,
    pub sorting_bits: u32,
}

impl RadixSortState {
    pub fn new(device: &wgpu::Device, sort_buf_size: u32, sorting_bits: u32) -> Self {
        let max_num_wgs = (sort_buf_size + RADIX_BLOCK_SIZE - 1) / RADIX_BLOCK_SIZE;

        let keys_b = create_storage_buffer(device, "radix_keys_b", (sort_buf_size as u64) * 4);
        let values_b = create_storage_buffer(device, "radix_values_b", (sort_buf_size as u64) * 4);

        let counts = create_storage_buffer(
            device,
            "radix_counts",
            (RADIX_BIN_COUNT as u64) * (max_num_wgs as u64) * 4,
        );

        let reduced = create_storage_buffer(device, "radix_reduced", (RADIX_BLOCK_SIZE as u64) * 4);

        let num_keys_buf = create_storage_buffer(device, "radix_num_keys", 4);

        let num_passes = (sorting_bits + 3) / 4;
        let config_buffers: Vec<wgpu::Buffer> = (0..num_passes)
            .map(|pass| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("radix_config_{pass}")),
                    size: 4,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
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

/// Record a complete radix sort of keys/values.
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

    // Zero the reduced buffer so the scan shader (which always reads 1024 elements)
    // doesn't read uninitialized memory when num_reduce_wgs < 1024.
    encoder.clear_buffer(&state.reduced, 0, None);

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
