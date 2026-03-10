//! Forward rendering pipeline orchestration (wgpu).
//!
//! Sets up the wgpu compute and render pipelines for the forward pass:
//!   1. Compute pass: SH eval, cull, depth key generation
//!   2. Render pass: Hardware rasterization with MRT output
//!
//! All GPU buffer management and bind group creation lives here.

use glam::{Mat4, Vec3};
use rmesh_data::SceneData;
use std::collections::HashMap;
use wgpu::util::DeviceExt;

// Re-export shared types for CPU-side use.
pub use rmesh_shaders::shared::{BVHNode, DrawIndirectCommand, Uniforms};

// Re-export tile types (moved to rmesh-tile crate).
pub use rmesh_tile::{
    TileBuffers, TilePipelines,
    ScanPipelines, ScanBuffers,
    create_tile_fill_bind_group,
    create_tile_ranges_bind_group, create_tile_ranges_bind_group_with_keys,
    create_prepare_dispatch_bind_group, create_rts_bind_group,
    create_tile_gen_scan_bind_group,
    record_scan_tile_pipeline,
    dispatch_2d,
};

// WGSL shader sources, embedded from crate-local files.
const FORWARD_COMPUTE_WGSL: &str = include_str!("wgsl/forward_compute.wgsl");
const FORWARD_VERTEX_WGSL: &str = include_str!("wgsl/forward_vertex.wgsl");
const FORWARD_FRAGMENT_WGSL: &str = include_str!("wgsl/forward_fragment.wgsl");
const TEX_TO_BUFFER_WGSL: &str = include_str!("wgsl/tex_to_buffer.wgsl");
const RAYTRACE_COMPUTE_WGSL: &str = include_str!("wgsl/raytrace_compute.wgsl");
const FORWARD_TILED_WGSL: &str = include_str!("wgsl/forward_tiled_compute.wgsl");

// ---------------------------------------------------------------------------
// GPU Buffers
// ---------------------------------------------------------------------------

/// All GPU buffers for the forward rendering pipeline.
pub struct SceneBuffers {
    /// Vertex positions [N x 3] f32
    pub vertices: wgpu::Buffer,
    /// Tet vertex indices [M x 4] u32
    pub indices: wgpu::Buffer,
    /// Direct SH coefficients [M x (deg+1)^2 x 3] f32
    pub sh_coeffs: wgpu::Buffer,
    /// Per-tet density [M] f32
    pub densities: wgpu::Buffer,
    /// Per-tet color gradient [M x 3] f32
    pub color_grads: wgpu::Buffer,
    /// Circumsphere data [M x 4] f32 (cx, cy, cz, r^2)
    pub circumdata: wgpu::Buffer,
    /// Evaluated per-tet color [M x 3] f32 (written by compute)
    pub colors: wgpu::Buffer,
    /// Sort keys [M] u32 (written by compute, sorted in place)
    pub sort_keys: wgpu::Buffer,
    /// Sort values [M] u32 (written by compute, sorted in place)
    pub sort_values: wgpu::Buffer,
    /// Indirect draw arguments (DrawIndirectCommand, 16 bytes)
    pub indirect_args: wgpu::Buffer,
    /// Per-frame uniforms (Uniforms struct)
    pub uniforms: wgpu::Buffer,
    /// Tiles touched per visible tet [M] u32 (written by compute at vis_idx)
    pub tiles_touched: wgpu::Buffer,
    /// Compact visible tet IDs [M] u32 (written by compute at vis_idx)
    pub compact_tet_ids: wgpu::Buffer,
}

impl SceneBuffers {
    /// Upload scene data to GPU buffers.
    pub fn upload(device: &wgpu::Device, _queue: &wgpu::Queue, scene: &SceneData) -> Self {
        let storage_copy = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        // Trainable parameter buffers need COPY_DST (for upload) and COPY_SRC (for readback).
        let trainable = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertices"),
            contents: bytemuck::cast_slice(&scene.vertices),
            usage: trainable,
        });

        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("indices"),
            contents: bytemuck::cast_slice(&scene.indices),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let sh_coeffs = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sh_coeffs"),
            contents: bytemuck::cast_slice(&scene.sh_coeffs),
            usage: trainable,
        });

        let densities = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("densities"),
            contents: bytemuck::cast_slice(&scene.densities),
            usage: trainable,
        });

        let color_grads = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("color_grads"),
            contents: bytemuck::cast_slice(&scene.color_grads),
            usage: trainable,
        });

        let circumdata = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("circumdata"),
            contents: bytemuck::cast_slice(&scene.circumdata),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let m = scene.tet_count as u64;
        let n_pow2 = (scene.tet_count as u64).next_power_of_two();

        let colors = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("colors"),
            size: m * 3 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Sort buffers padded to next power of 2.
        let sort_keys = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sort_keys"),
            size: n_pow2 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let sort_values = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sort_values"),
            size: n_pow2 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let indirect_args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("indirect_args"),
            size: 16, // DrawIndirectCommand = 4 x u32
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: storage_copy,
            mapped_at_creation: false,
        });

        // Pad to multiple of 16 bytes for vec4<u32> binding in RTS prefix scan
        let tiles_touched = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tiles_touched"),
            size: ((m + 3) / 4) * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let compact_tet_ids = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compact_tet_ids"),
            size: m * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            vertices,
            indices,
            sh_coeffs,
            densities,
            color_grads,
            circumdata,
            colors,
            sort_keys,
            sort_values,
            indirect_args,
            uniforms,
            tiles_touched,
            compact_tet_ids,
        }
    }
}

// ---------------------------------------------------------------------------
// Pipelines
// ---------------------------------------------------------------------------

/// Compiled pipelines for the forward pass.
pub struct ForwardPipelines {
    pub compute_pipeline: wgpu::ComputePipeline,
    pub compute_bind_group_layout: wgpu::BindGroupLayout,
    pub render_pipeline: wgpu::RenderPipeline,
    pub render_bind_group_layout: wgpu::BindGroupLayout,
}

/// Helper: create N storage buffer layout entries for the given visibility.
fn storage_entries(count: u32, visibility: wgpu::ShaderStages, read_only: &[bool]) -> Vec<wgpu::BindGroupLayoutEntry> {
    (0..count)
        .map(|i| wgpu::BindGroupLayoutEntry {
            binding: i,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: read_only[i as usize],
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect()
}

impl ForwardPipelines {
    /// Create all three pipelines from WGSL shader sources.
    ///
    /// `color_format`: texture format for the main color attachment (e.g. Rgba16Float).
    /// `aux_format`: texture format for the auxiliary attachment (e.g. Rgba32Float).
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        aux_format: wgpu::TextureFormat,
    ) -> Self {
        // ----- Compute pipeline (13 bindings) -----
        // Bindings 0-6: read-only storage (uniforms, vertices, indices, sh_coeffs, densities, color_grads, circumdata)
        // Bindings 7-12: read-write storage (colors, sort_keys, sort_values, indirect_args, tiles_touched, compact_tet_ids)
        let compute_read_only = [
            true, true, true, true, true, true, true, // 0-6 read-only
            false, false, false, false,                // 7-10 read-write
            false, false,                              // 11-12 read-write (tiles_touched, compact_tet_ids)
        ];
        let compute_entries = storage_entries(
            13,
            wgpu::ShaderStages::COMPUTE,
            &compute_read_only,
        );
        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("compute_bind_group_layout"),
                entries: &compute_entries,
            });
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute_pipeline_layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                immediate_size: 0,
            });
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forward_compute.wgsl"),
            source: wgpu::ShaderSource::Wgsl(FORWARD_COMPUTE_WGSL.into()),
        });
        let compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("forward_compute_pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &compute_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // ----- Render pipeline (7 bindings) -----
        // Bindings: uniforms, vertices, indices, colors, densities, color_grads, sorted_indices
        // All read-only from the vertex/fragment perspective.
        let render_read_only = [true; 7];
        let render_entries = storage_entries(
            7,
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            &render_read_only,
        );
        let render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("render_bind_group_layout"),
                entries: &render_entries,
            });
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("render_pipeline_layout"),
                bind_group_layouts: &[&render_bind_group_layout],
                immediate_size: 0,
            });
        let vertex_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forward_vertex.wgsl"),
            source: wgpu::ShaderSource::Wgsl(FORWARD_VERTEX_WGSL.into()),
        });
        let fragment_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forward_fragment.wgsl"),
            source: wgpu::ShaderSource::Wgsl(FORWARD_FRAGMENT_WGSL.into()),
        });

        // Premultiplied alpha blend for color attachment 0
        let premul_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };

        let render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("forward_render_pipeline"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &vertex_shader,
                    entry_point: Some("main"),
                    buffers: &[], // No vertex buffers -- all data from storage
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back), // Only front-facing → 1 fragment per pixel
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None, // No depth test -- back-to-front alpha blending
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &fragment_shader,
                    entry_point: Some("main"),
                    targets: &[
                        // Color attachment 0: premultiplied alpha blend
                        Some(wgpu::ColorTargetState {
                            format: color_format,
                            blend: Some(premul_blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        // Color attachment 1 (aux): no blending
                        Some(wgpu::ColorTargetState {
                            format: aux_format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                    ],
                    compilation_options: Default::default(),
                }),
                multiview_mask: None,
                cache: None,
            });

        Self {
            compute_pipeline,
            compute_bind_group_layout,
            render_pipeline,
            render_bind_group_layout,
        }
    }
}

// ---------------------------------------------------------------------------
// Render Targets
// ---------------------------------------------------------------------------

/// The render target textures for MRT output.
pub struct RenderTargets {
    /// Main color output texture (e.g. Rgba16Float)
    pub color_texture: wgpu::Texture,
    /// Auxiliary output texture (Rgba32Float)
    pub aux0_texture: wgpu::Texture,
    /// View into color texture
    pub color_view: wgpu::TextureView,
    /// View into aux texture
    pub aux0_view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
}

impl RenderTargets {
    /// Create render target textures at the given resolution.
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let color_format = wgpu::TextureFormat::Rgba16Float;
        let aux_format = wgpu::TextureFormat::Rgba32Float;

        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("color_target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: color_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let aux0_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("aux0_target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: aux_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let aux0_view = aux0_texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            color_texture,
            aux0_texture,
            color_view,
            aux0_view,
            width,
            height,
        }
    }
}

// ---------------------------------------------------------------------------
// Tex-to-Buffer Pipeline
// ---------------------------------------------------------------------------

/// Pipeline that converts the Rgba16Float render target to an f32 storage buffer.
///
/// The forward render pass outputs to a texture, but the loss and backward shaders
/// need a flat `array<f32>` storage buffer. This compute pipeline bridges the gap.
pub struct TexToBufferPipeline {
    pipeline: wgpu::ComputePipeline,
    _bind_group_layout: wgpu::BindGroupLayout,
    _params_buffer: wgpu::Buffer,
    /// The output storage buffer: [W x H x 4] f32
    pub rendered_image: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
}

impl TexToBufferPipeline {
    /// Create the pipeline and allocate the rendered_image buffer.
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        color_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tex_to_buffer.wgsl"),
            source: wgpu::ShaderSource::Wgsl(TEX_TO_BUFFER_WGSL.into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tex_to_buffer_bind_group_layout"),
                entries: &[
                    // @binding(0) texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // @binding(1) output buffer (read_write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // @binding(2) params (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tex_to_buffer_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tex_to_buffer_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let rendered_image = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rendered_image"),
            size: (width as u64) * (height as u64) * 4 * 4, // RGBA f32
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_data: [u32; 2] = [width, height];
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tex_to_buffer_params"),
            contents: bytemuck::cast_slice(&params_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tex_to_buffer_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rendered_image.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            pipeline,
            _bind_group_layout: bind_group_layout,
            _params_buffer: params_buffer,
            rendered_image,
            bind_group,
            width,
            height,
        }
    }
}

/// Record the tex-to-buffer conversion dispatch.
///
/// Should be called after the render pass finishes, within the same command encoder.
pub fn record_tex_to_buffer(
    encoder: &mut wgpu::CommandEncoder,
    ttb: &TexToBufferPipeline,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("tex_to_buffer"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&ttb.pipeline);
    pass.set_bind_group(0, &ttb.bind_group, &[]);
    pass.dispatch_workgroups(
        (ttb.width + 15) / 16,
        (ttb.height + 15) / 16,
        1,
    );
}

// ---------------------------------------------------------------------------
// Bind Groups
// ---------------------------------------------------------------------------

/// Create the compute bind group (13 bindings).
///
/// Binding order matches `forward_compute.wgsl`:
///   0: uniforms, 1: vertices, 2: indices, 3: sh_coeffs,
///   4: densities, 5: color_grads, 6: circumdata,
///   7: colors, 8: sort_keys, 9: sort_values, 10: indirect_args,
///   11: tiles_touched, 12: compact_tet_ids
pub fn create_compute_bind_group(
    device: &wgpu::Device,
    pipelines: &ForwardPipelines,
    buffers: &SceneBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute_bind_group"),
        layout: &pipelines.compute_bind_group_layout,
        entries: &[
            buf_entry(0, &buffers.uniforms),
            buf_entry(1, &buffers.vertices),
            buf_entry(2, &buffers.indices),
            buf_entry(3, &buffers.sh_coeffs),
            buf_entry(4, &buffers.densities),
            buf_entry(5, &buffers.color_grads),
            buf_entry(6, &buffers.circumdata),
            buf_entry(7, &buffers.colors),
            buf_entry(8, &buffers.sort_keys),
            buf_entry(9, &buffers.sort_values),
            buf_entry(10, &buffers.indirect_args),
            buf_entry(11, &buffers.tiles_touched),
            buf_entry(12, &buffers.compact_tet_ids),
        ],
    })
}

/// Create the render bind group (7 bindings).
///
/// Binding order matches `forward_vertex.wgsl`:
///   0: uniforms, 1: vertices, 2: indices, 3: colors,
///   4: densities, 5: color_grads, 6: sorted_indices (= sort_values)
pub fn create_render_bind_group(
    device: &wgpu::Device,
    pipelines: &ForwardPipelines,
    buffers: &SceneBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("render_bind_group"),
        layout: &pipelines.render_bind_group_layout,
        entries: &[
            buf_entry(0, &buffers.uniforms),
            buf_entry(1, &buffers.vertices),
            buf_entry(2, &buffers.indices),
            buf_entry(3, &buffers.colors),
            buf_entry(4, &buffers.densities),
            buf_entry(5, &buffers.color_grads),
            buf_entry(6, &buffers.sort_values), // sorted indices
        ],
    })
}

/// Shorthand for a full-buffer bind group entry.
fn buf_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

// ---------------------------------------------------------------------------
// Command Recording
// ---------------------------------------------------------------------------

/// Record only the forward compute pass (no sort).
///
/// Stages:
///   1. Reset indirect args (vertex_count=12, instance_count=0)
///   2. Compute pass (SH eval + cull + depth keys + tiles_touched + compact_tet_ids)
///
/// Use this with the scan-based tile pipeline which reads compact_tet_ids
/// directly instead of relying on sorted sort_values.
pub fn record_forward_compute(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &ForwardPipelines,
    buffers: &SceneBuffers,
    compute_bg: &wgpu::BindGroup,
    tet_count: u32,
    queue: &wgpu::Queue,
) {
    let reset_cmd = DrawIndirectCommand {
        vertex_count: 12,
        instance_count: 0,
        first_vertex: 0,
        first_instance: 0,
    };
    queue.write_buffer(&buffers.indirect_args, 0, bytemuck::bytes_of(&reset_cmd));

    // Clear tiles_touched so RTS vec4 padding elements are zero
    encoder.clear_buffer(&buffers.tiles_touched, 0, None);

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("forward_compute"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipelines.compute_pipeline);
        cpass.set_bind_group(0, compute_bg, &[]);

        let workgroup_size = 64u32;
        let n_pow2 = tet_count.next_power_of_two();
        let total_workgroups = (n_pow2 + workgroup_size - 1) / workgroup_size;
        let max_per_dim = 65535u32;
        let dispatch_x = total_workgroups.min(max_per_dim);
        let dispatch_y = (total_workgroups + max_per_dim - 1) / max_per_dim;
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }
}

/// Record just the forward compute pass (no sort, no hardware rasterization).
///
/// Stages:
///   1. Reset indirect args (vertex_count=12, instance_count=0)
///   2. Compute pass (SH eval + cull + depth keys)
pub fn record_forward_compute_and_sort(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &ForwardPipelines,
    buffers: &SceneBuffers,
    compute_bg: &wgpu::BindGroup,
    tet_count: u32,
    queue: &wgpu::Queue,
) {
    // ----- 1. Reset indirect args -----
    let reset_cmd = DrawIndirectCommand {
        vertex_count: 12,
        instance_count: 0,
        first_vertex: 0,
        first_instance: 0,
    };
    queue.write_buffer(&buffers.indirect_args, 0, bytemuck::bytes_of(&reset_cmd));

    // ----- 2. Compute pass -----
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("forward_compute"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipelines.compute_pipeline);
        cpass.set_bind_group(0, compute_bg, &[]);

        let workgroup_size = 64u32;
        let n_pow2 = tet_count.next_power_of_two();
        let total_workgroups = (n_pow2 + workgroup_size - 1) / workgroup_size;
        let max_per_dim = 65535u32;
        let dispatch_x = total_workgroups.min(max_per_dim);
        let dispatch_y = (total_workgroups + max_per_dim - 1) / max_per_dim;
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }
}

/// Record the full forward pass into a command encoder.
///
/// Stages:
///   1. Reset indirect args (vertex_count=12, instance_count=0)
///   2. Compute pass (SH eval + cull + depth keys)
///   3. Render pass (hardware rasterization with MRT, draw_indirect)
///
/// The caller must have already written the Uniforms into `buffers.uniforms`
/// via `queue.write_buffer` before calling this function.
pub fn record_forward_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &ForwardPipelines,
    buffers: &SceneBuffers,
    targets: &RenderTargets,
    compute_bg: &wgpu::BindGroup,
    render_bg: &wgpu::BindGroup,
    tet_count: u32,
    queue: &wgpu::Queue,
) {
    // ----- 1. Reset indirect args -----
    // vertex_count=12 (4 tri faces), instance_count=0 (compute pass will atomicAdd)
    let reset_cmd = DrawIndirectCommand {
        vertex_count: 12,
        instance_count: 0,
        first_vertex: 0,
        first_instance: 0,
    };
    queue.write_buffer(&buffers.indirect_args, 0, bytemuck::bytes_of(&reset_cmd));

    // ----- 2. Compute pass -----
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("forward_compute"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipelines.compute_pipeline);
        cpass.set_bind_group(0, compute_bg, &[]);

        // Dispatch for n_pow2 threads: tet_count threads do real work,
        // padding threads (tet_count..n_pow2-1) initialize sort buffers.
        let workgroup_size = 64u32;
        let n_pow2 = tet_count.next_power_of_two();
        let total_workgroups = (n_pow2 + workgroup_size - 1) / workgroup_size;
        let max_per_dim = 65535u32;
        let dispatch_x = total_workgroups.min(max_per_dim);
        let dispatch_y = (total_workgroups + max_per_dim - 1) / max_per_dim;
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    // ----- 3. Render pass -----
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("forward_render"),
            color_attachments: &[
                // Attachment 0: main color (premultiplied alpha blend)
                Some(wgpu::RenderPassColorAttachment {
                    view: &targets.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
                // Attachment 1: auxiliary (no blending, overwrite)
                Some(wgpu::RenderPassColorAttachment {
                    view: &targets.aux0_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
            ],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_viewport(
            0.0,
            0.0,
            targets.width as f32,
            targets.height as f32,
            0.0,
            1.0,
        );
        rpass.set_scissor_rect(0, 0, targets.width, targets.height);
        rpass.set_pipeline(&pipelines.render_pipeline);
        rpass.set_bind_group(0, render_bg, &[]);

        // Draw indirect -- instance_count comes from compute pass atomicAdd
        rpass.draw_indirect(&buffers.indirect_args, 0);
    }
}

// ---------------------------------------------------------------------------
// High-level helpers
// ---------------------------------------------------------------------------

/// Convenience: set up everything needed for a forward frame.
///
/// Returns (SceneBuffers, ForwardPipelines, RenderTargets, compute_bg, render_bg).
pub fn setup_forward(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    scene: &SceneData,
    width: u32,
    height: u32,
) -> (
    SceneBuffers,
    ForwardPipelines,
    RenderTargets,
    wgpu::BindGroup,
    wgpu::BindGroup,
) {
    let color_format = wgpu::TextureFormat::Rgba16Float;
    let aux_format = wgpu::TextureFormat::Rgba32Float;

    let buffers = SceneBuffers::upload(device, queue, scene);
    let pipelines = ForwardPipelines::new(device, color_format, aux_format);
    let targets = RenderTargets::new(device, width, height);

    let compute_bg = create_compute_bind_group(device, &pipelines, &buffers);
    let render_bg = create_render_bind_group(device, &pipelines, &buffers);

    (buffers, pipelines, targets, compute_bg, render_bg)
}

/// Build a `Uniforms` struct from camera matrices and scene metadata.
pub fn make_uniforms(
    vp: Mat4,
    inv_vp: Mat4,
    cam_pos: glam::Vec3,
    screen_width: f32,
    screen_height: f32,
    tet_count: u32,
    sh_degree: u32,
    step: u32,
) -> Uniforms {
    Uniforms {
        vp_col0: vp.col(0).into(),
        vp_col1: vp.col(1).into(),
        vp_col2: vp.col(2).into(),
        vp_col3: vp.col(3).into(),
        inv_vp_col0: inv_vp.col(0).into(),
        inv_vp_col1: inv_vp.col(1).into(),
        inv_vp_col2: inv_vp.col(2).into(),
        inv_vp_col3: inv_vp.col(3).into(),
        cam_pos_pad: [cam_pos.x, cam_pos.y, cam_pos.z, 0.0],
        screen_width,
        screen_height,
        tet_count,
        sh_degree,
        step,
        _pad1: [0; 7],
    }
}

// ---------------------------------------------------------------------------
// Ray Tracing: Tet Neighbors
// ---------------------------------------------------------------------------

/// Tet neighbor adjacency: `neighbors[tet_id * 4 + face_idx]` = neighbor tet or -1.
pub fn compute_tet_neighbors(indices: &[u32], tet_count: usize) -> Vec<i32> {
    use rmesh_shaders::shared::TET_FACE_INDICES;

    let mut neighbors = vec![-1i32; tet_count * 4];
    let mut face_map: HashMap<[u32; 3], (usize, usize)> = HashMap::with_capacity(tet_count * 4);

    for tet_id in 0..tet_count {
        for face_idx in 0..4usize {
            let fi_base = face_idx * 3;
            let vi0 = indices[tet_id * 4 + TET_FACE_INDICES[fi_base] as usize];
            let vi1 = indices[tet_id * 4 + TET_FACE_INDICES[fi_base + 1] as usize];
            let vi2 = indices[tet_id * 4 + TET_FACE_INDICES[fi_base + 2] as usize];

            let mut key = [vi0, vi1, vi2];
            key.sort();

            if let Some(&(other_tet, other_face)) = face_map.get(&key) {
                neighbors[tet_id * 4 + face_idx] = other_tet as i32;
                neighbors[other_tet * 4 + other_face] = tet_id as i32;
                face_map.remove(&key);
            } else {
                face_map.insert(key, (tet_id, face_idx));
            }
        }
    }

    neighbors
}

// ---------------------------------------------------------------------------
// Ray Tracing: BVH Builder
// ---------------------------------------------------------------------------

/// BVH data: nodes + packed boundary face array.
pub struct BVHData {
    pub nodes: Vec<BVHNode>,
    pub boundary_faces: Vec<u32>,
}

/// 30-bit Morton code for 10-bit x,y,z.
fn morton_3d(x: u32, y: u32, z: u32) -> u32 {
    fn expand_bits(mut v: u32) -> u32 {
        v = (v | (v << 16)) & 0x030000FF;
        v = (v | (v << 8)) & 0x0300F00F;
        v = (v | (v << 4)) & 0x030C30C3;
        v = (v | (v << 2)) & 0x09249249;
        v
    }
    expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2)
}

/// Build a BVH over boundary faces (faces where `neighbors[...] == -1`).
pub fn build_boundary_bvh(
    vertices: &[f32],
    indices: &[u32],
    neighbors: &[i32],
    tet_count: usize,
) -> BVHData {
    use rmesh_shaders::shared::TET_FACE_INDICES;

    // Collect boundary faces
    let mut boundary_faces: Vec<u32> = Vec::new();
    let mut centroids: Vec<[f32; 3]> = Vec::new();
    let mut aabbs: Vec<([f32; 3], [f32; 3])> = Vec::new();

    for tet_id in 0..tet_count {
        for face_idx in 0..4usize {
            if neighbors[tet_id * 4 + face_idx] != -1 {
                continue;
            }
            let packed = ((tet_id as u32) << 2) | (face_idx as u32);
            boundary_faces.push(packed);

            let fi_base = face_idx * 3;
            let vi0 = indices[tet_id * 4 + TET_FACE_INDICES[fi_base] as usize] as usize;
            let vi1 = indices[tet_id * 4 + TET_FACE_INDICES[fi_base + 1] as usize] as usize;
            let vi2 = indices[tet_id * 4 + TET_FACE_INDICES[fi_base + 2] as usize] as usize;

            let v0 = [vertices[vi0 * 3], vertices[vi0 * 3 + 1], vertices[vi0 * 3 + 2]];
            let v1 = [vertices[vi1 * 3], vertices[vi1 * 3 + 1], vertices[vi1 * 3 + 2]];
            let v2 = [vertices[vi2 * 3], vertices[vi2 * 3 + 1], vertices[vi2 * 3 + 2]];

            centroids.push([
                (v0[0] + v1[0] + v2[0]) / 3.0,
                (v0[1] + v1[1] + v2[1]) / 3.0,
                (v0[2] + v1[2] + v2[2]) / 3.0,
            ]);
            aabbs.push((
                [v0[0].min(v1[0]).min(v2[0]), v0[1].min(v1[1]).min(v2[1]), v0[2].min(v1[2]).min(v2[2])],
                [v0[0].max(v1[0]).max(v2[0]), v0[1].max(v1[1]).max(v2[1]), v0[2].max(v1[2]).max(v2[2])],
            ));
        }
    }

    let n_faces = boundary_faces.len();
    if n_faces == 0 {
        return BVHData {
            nodes: vec![BVHNode {
                aabb_min: [0.0; 3], left_or_face: -1, aabb_max: [0.0; 3], right_or_count: 0,
            }],
            boundary_faces,
        };
    }

    // Sort faces by Morton code of centroid
    let mut scene_min = [f32::INFINITY; 3];
    let mut scene_max = [f32::NEG_INFINITY; 3];
    for c in &centroids {
        for i in 0..3 {
            scene_min[i] = scene_min[i].min(c[i]);
            scene_max[i] = scene_max[i].max(c[i]);
        }
    }
    let scene_extent = [
        (scene_max[0] - scene_min[0]).max(1e-10),
        (scene_max[1] - scene_min[1]).max(1e-10),
        (scene_max[2] - scene_min[2]).max(1e-10),
    ];

    let mut sorted_indices: Vec<usize> = (0..n_faces).collect();
    sorted_indices.sort_by_key(|&i| {
        let nx = ((centroids[i][0] - scene_min[0]) / scene_extent[0] * 1023.0) as u32;
        let ny = ((centroids[i][1] - scene_min[1]) / scene_extent[1] * 1023.0) as u32;
        let nz = ((centroids[i][2] - scene_min[2]) / scene_extent[2] * 1023.0) as u32;
        morton_3d(nx.min(1023), ny.min(1023), nz.min(1023))
    });

    // Reorder boundary_faces and aabbs by sorted order
    let orig_faces = boundary_faces.clone();
    let orig_aabbs = aabbs.clone();
    for (new_i, &old_i) in sorted_indices.iter().enumerate() {
        boundary_faces[new_i] = orig_faces[old_i];
        aabbs[new_i] = orig_aabbs[old_i];
    }

    // Build binary BVH
    struct BuildTask {
        start: usize,
        end: usize,
        node_idx: usize,
    }

    let mut nodes: Vec<BVHNode> = Vec::with_capacity(2 * n_faces);
    // Reserve root
    nodes.push(BVHNode { aabb_min: [0.0; 3], left_or_face: 0, aabb_max: [0.0; 3], right_or_count: 0 });
    let mut stack = vec![BuildTask { start: 0, end: n_faces, node_idx: 0 }];

    while let Some(task) = stack.pop() {
        let count = task.end - task.start;
        let mut amin = [f32::INFINITY; 3];
        let mut amax = [f32::NEG_INFINITY; 3];
        for i in task.start..task.end {
            for d in 0..3 {
                amin[d] = amin[d].min(aabbs[i].0[d]);
                amax[d] = amax[d].max(aabbs[i].1[d]);
            }
        }

        if count <= 4 {
            nodes[task.node_idx] = BVHNode {
                aabb_min: amin,
                left_or_face: -(task.start as i32 + 1),
                aabb_max: amax,
                right_or_count: count as i32,
            };
        } else {
            let mid = task.start + count / 2;
            let left_idx = nodes.len();
            nodes.push(BVHNode { aabb_min: [0.0; 3], left_or_face: 0, aabb_max: [0.0; 3], right_or_count: 0 });
            let right_idx = nodes.len();
            nodes.push(BVHNode { aabb_min: [0.0; 3], left_or_face: 0, aabb_max: [0.0; 3], right_or_count: 0 });

            nodes[task.node_idx] = BVHNode {
                aabb_min: amin,
                left_or_face: left_idx as i32,
                aabb_max: amax,
                right_or_count: right_idx as i32,
            };

            stack.push(BuildTask { start: task.start, end: mid, node_idx: left_idx });
            stack.push(BuildTask { start: mid, end: task.end, node_idx: right_idx });
        }
    }

    BVHData { nodes, boundary_faces }
}

// ---------------------------------------------------------------------------
// Ray Tracing: Containment Test
// ---------------------------------------------------------------------------

/// Find the tet containing `point`, or None if outside the mesh.
/// Brute-force barycentric test — O(N_tets), single query per frame.
pub fn find_containing_tet(
    vertices: &[f32],
    indices: &[u32],
    tet_count: usize,
    point: Vec3,
) -> Option<u32> {
    for tet_id in 0..tet_count {
        let vi = [
            indices[tet_id * 4] as usize,
            indices[tet_id * 4 + 1] as usize,
            indices[tet_id * 4 + 2] as usize,
            indices[tet_id * 4 + 3] as usize,
        ];
        let v = [
            Vec3::new(vertices[vi[0] * 3], vertices[vi[0] * 3 + 1], vertices[vi[0] * 3 + 2]),
            Vec3::new(vertices[vi[1] * 3], vertices[vi[1] * 3 + 1], vertices[vi[1] * 3 + 2]),
            Vec3::new(vertices[vi[2] * 3], vertices[vi[2] * 3 + 1], vertices[vi[2] * 3 + 2]),
            Vec3::new(vertices[vi[3] * 3], vertices[vi[3] * 3 + 1], vertices[vi[3] * 3 + 2]),
        ];

        let d = v[1] - v[0];
        let e = v[2] - v[0];
        let f = v[3] - v[0];
        let p = point - v[0];

        let det = d.dot(e.cross(f));
        if det.abs() < 1e-20 {
            continue;
        }
        let inv_det = 1.0 / det;

        let u = p.dot(e.cross(f)) * inv_det;
        let v_coord = d.dot(p.cross(f)) * inv_det;
        let w = d.dot(e.cross(p)) * inv_det;

        let eps = -1e-6;
        if u >= eps && v_coord >= eps && w >= eps && (u + v_coord + w) <= 1.0 + 1e-6 {
            return Some(tet_id as u32);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Ray Tracing: Pipeline + Buffers
// ---------------------------------------------------------------------------

/// Compute-based ray tracing pipeline with adjacency traversal.
pub struct RayTracePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// The output storage buffer: [W x H x 4] f32
    pub rendered_image: wgpu::Buffer,
    pub width: u32,
    pub height: u32,
}

impl RayTracePipeline {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raytrace_compute.wgsl"),
            source: wgpu::ShaderSource::Wgsl(RAYTRACE_COMPUTE_WGSL.into()),
        });

        let read_only = [true, true, true, true, true, true, true, false, true, true, true];
        let entries = storage_entries(11, wgpu::ShaderStages::COMPUTE, &read_only);

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("raytrace_bgl"),
                entries: &entries,
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raytrace_pl"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("raytrace_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let rendered_image = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raytrace_rendered_image"),
            size: (width as u64) * (height as u64) * 4 * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self { pipeline, bind_group_layout, rendered_image, width, height }
    }

    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    pub fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }
}

/// GPU buffers for ray trace adjacency data.
pub struct RayTraceBuffers {
    pub tet_neighbors: wgpu::Buffer,
    pub bvh_nodes: wgpu::Buffer,
    pub boundary_faces: wgpu::Buffer,
    pub start_tet: wgpu::Buffer,
}

impl RayTraceBuffers {
    pub fn new(device: &wgpu::Device, neighbors: &[i32], bvh: &BVHData) -> Self {
        let tet_neighbors = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tet_neighbors"),
            contents: bytemuck::cast_slice(neighbors),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bvh_data = if bvh.nodes.is_empty() {
            vec![BVHNode { aabb_min: [0.0; 3], left_or_face: -1, aabb_max: [0.0; 3], right_or_count: 0 }]
        } else {
            bvh.nodes.clone()
        };
        let bvh_nodes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bvh_nodes"),
            contents: bytemuck::cast_slice(&bvh_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bf_data = if bvh.boundary_faces.is_empty() { vec![0u32] } else { bvh.boundary_faces.clone() };
        let boundary_faces = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("boundary_faces"),
            contents: bytemuck::cast_slice(&bf_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let start_tet = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("start_tet"),
            contents: bytemuck::cast_slice(&[-1i32]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        Self { tet_neighbors, bvh_nodes, boundary_faces, start_tet }
    }
}

/// Create the ray trace bind group.
pub fn create_raytrace_bind_group(
    device: &wgpu::Device,
    rt_pipeline: &RayTracePipeline,
    scene_buffers: &SceneBuffers,
    rt_buffers: &RayTraceBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("raytrace_bg"),
        layout: rt_pipeline.bind_group_layout(),
        entries: &[
            buf_entry(0, &scene_buffers.uniforms),
            buf_entry(1, &scene_buffers.vertices),
            buf_entry(2, &scene_buffers.indices),
            buf_entry(3, &scene_buffers.colors),
            buf_entry(4, &scene_buffers.densities),
            buf_entry(5, &scene_buffers.color_grads),
            buf_entry(6, &rt_buffers.tet_neighbors),
            buf_entry(7, &rt_pipeline.rendered_image),
            buf_entry(8, &rt_buffers.bvh_nodes),
            buf_entry(9, &rt_buffers.boundary_faces),
            buf_entry(10, &rt_buffers.start_tet),
        ],
    })
}

/// Record the ray trace compute pass dispatch.
pub fn record_raytrace(
    encoder: &mut wgpu::CommandEncoder,
    rt_pipeline: &RayTracePipeline,
    bind_group: &wgpu::BindGroup,
    width: u32,
    height: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("raytrace"),
        timestamp_writes: None,
    });
    pass.set_pipeline(rt_pipeline.pipeline());
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
}

// ===========================================================================
// Forward tiled pipeline
// ===========================================================================

/// Compute-based forward renderer using tiles with warp-per-tile.
///
/// Requires `wgpu::Features::SUBGROUPS` on the device.
/// Renders directly to an f32 storage buffer (no texture intermediate).
pub struct ForwardTiledPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// The output storage buffer: [W x H x 4] f32
    pub rendered_image: wgpu::Buffer,
    pub width: u32,
    pub height: u32,
}

impl ForwardTiledPipeline {
    /// Create the forward tiled pipeline and allocate the rendered_image buffer.
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forward_tiled_compute.wgsl"),
            source: wgpu::ShaderSource::Wgsl(FORWARD_TILED_WGSL.into()),
        });

        // 10 bindings: uniforms, vertices, indices, colors, densities, color_grads,
        //              tile_sort_values, tile_ranges, tile_uniforms, rendered_image
        let read_only = [true, true, true, true, true, true, true, true, true, false];
        let entries: Vec<wgpu::BindGroupLayoutEntry> = read_only
            .iter()
            .enumerate()
            .map(|(i, &ro)| rmesh_tile::storage_entry(i as u32, ro))
            .collect();

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("forward_tiled_bgl"),
                entries: &entries,
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("forward_tiled_pl"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("forward_tiled_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let rendered_image = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fwd_tiled_rendered_image"),
            size: (width as u64) * (height as u64) * 4 * 4, // RGBA f32
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group_layout,
            rendered_image,
            width,
            height,
        }
    }

    /// Get the bind group layout (for creating bind groups externally).
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Get the pipeline (for recording dispatches).
    pub fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }
}

/// Create the forward tiled bind group.
///
/// Binding order matches `forward_tiled_compute.wgsl`:
///   0: uniforms, 1: vertices, 2: indices, 3: colors,
///   4: densities, 5: color_grads, 6: tile_sort_values,
///   7: tile_ranges, 8: tile_uniforms, 9: rendered_image
pub fn create_forward_tiled_bind_group(
    device: &wgpu::Device,
    fwd_tiled: &ForwardTiledPipeline,
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
        label: Some("forward_tiled_bg"),
        layout: fwd_tiled.bind_group_layout(),
        entries: &[
            buf_entry(0, uniforms),
            buf_entry(1, vertices),
            buf_entry(2, indices),
            buf_entry(3, colors),
            buf_entry(4, densities),
            buf_entry(5, color_grads),
            buf_entry(6, tile_sort_values),
            buf_entry(7, tile_ranges),
            buf_entry(8, tile_uniforms),
            buf_entry(9, &fwd_tiled.rendered_image),
        ],
    })
}

/// Record the forward tiled compute pass dispatch.
///
/// Dispatches one workgroup per tile (32 threads each).
pub fn record_forward_tiled(
    encoder: &mut wgpu::CommandEncoder,
    fwd_tiled: &ForwardTiledPipeline,
    bind_group: &wgpu::BindGroup,
    num_tiles: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("forward_tiled"),
        timestamp_writes: None,
    });
    pass.set_pipeline(fwd_tiled.pipeline());
    pass.set_bind_group(0, bind_group, &[]);
    let (x, y) = dispatch_2d(num_tiles);
    pass.dispatch_workgroups(x, y, 1);
}
