//! Deep shadow map rendering for tetrahedral radiance meshes.
//!
//! Produces a per-pixel transmittance map by rasterizing the interval
//! decomposition of each visible tet using a minimal fragment shader that
//! only evaluates the volume rendering alpha. Hardware premultiplied-alpha
//! blending (back-to-front) accumulates the total transmittance.
//!
//! # Pipeline flow
//!
//! The DSM pipeline reuses the compute-interval generation and indirect-convert
//! stages from [`rmesh_render::ComputeIntervalPipelines`], then draws with its
//! own lightweight render pipeline:
//!
//! 1. `interval_compute.wgsl` — tet → screen triangles (reused)
//! 2. `interval_indirect_convert.wgsl` — dispatch/draw args (reused)
//! 3. `dsm_fragment.wgsl` — alpha-only volume integral (this crate)
//!
//! # Output
//!
//! The output texture's alpha channel holds `1 - T`, where `T` is the total
//! transmittance. Read `T = 1 - alpha` for shadow attenuation.

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3};
use rmesh_compositor::{PrimitiveGeometry, PrimitiveVertex};
use rmesh_interact::Primitive;
use rmesh_render::{
    ComputeIntervalPipelines, DrawIndirectCommand, ForwardPipelines, GpuLight,
    MaterialBuffers, SceneBuffers, Uniforms, dispatch_2d,
};

const INTERVAL_VERTEX_WGSL: &str = include_str!("wgsl/interval_vertex.wgsl");
const DSM_FRAGMENT_WGSL: &str = include_str!("wgsl/dsm_fragment.wgsl");
const DSM_FOURIER_FRAGMENT_WGSL: &str = include_str!("wgsl/dsm_fourier_fragment.wgsl");
const DSM_PRIMITIVE_WGSL: &str = include_str!("wgsl/dsm_primitive.wgsl");
const DSM_RESOLVE_WGSL: &str = include_str!("wgsl/dsm_resolve.wgsl");

/// Number of Fourier terms (N=4 → 9 coefficients → 3 RGBA targets).
pub const FOURIER_N: u32 = 4;
/// Number of MRT targets for Fourier coefficient storage.
pub const FOURIER_MRT_COUNT: usize = 3;

/// DSM output format.
const DSM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Shorthand for a full-buffer bind group entry.
fn buf_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// Render pipeline for deep shadow map generation.
///
/// Uses `interval_vertex.wgsl` (shared with the full interval pipeline) and a
/// stripped-down `dsm_fragment.wgsl` that outputs only alpha.
pub struct DsmPipeline {
    pub render_pipeline: wgpu::RenderPipeline,
    pub render_bg_layout: wgpu::BindGroupLayout,
}

impl DsmPipeline {
    /// Create the DSM render pipeline.
    ///
    /// `color_format` should match the output texture (e.g. `Rgba16Float`).
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        // 3 read-only storage bindings: uniforms, verts, tet_data
        let render_entries: Vec<wgpu::BindGroupLayoutEntry> = (0..3)
            .map(|i| wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();

        let render_bg_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("dsm_render_bgl"),
                entries: &render_entries,
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dsm_render_pl"),
            bind_group_layouts: &[&render_bg_layout],
            immediate_size: 0,
        });

        let vertex_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dsm_interval_vertex.wgsl"),
            source: wgpu::ShaderSource::Wgsl(INTERVAL_VERTEX_WGSL.into()),
        });

        let fragment_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dsm_fourier_fragment.wgsl"),
            source: wgpu::ShaderSource::Wgsl(DSM_FOURIER_FRAGMENT_WGSL.into()),
        });

        // Additive blending for Fourier coefficient accumulation
        let additive_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };

        let additive_target = Some(wgpu::ColorTargetState {
            format: color_format,
            blend: Some(additive_blend),
            write_mask: wgpu::ColorWrites::ALL,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dsm_render_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_shader,
                entry_point: Some("main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &fragment_shader,
                entry_point: Some("main"),
                targets: &[
                    additive_target.clone(), // RT0: a0, a1, b1, a2
                    additive_target.clone(), // RT1: b2, a3, b3, a4
                    additive_target,         // RT2: b4, 0, 0, 0
                ],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        Self {
            render_pipeline,
            render_bg_layout,
        }
    }
}

/// Create the DSM render bind group (3 storage bindings).
///
/// Binding order matches `interval_vertex.wgsl`:
///   0: uniforms, 1: interval_vertex_buf, 2: interval_tet_data_buf
pub fn create_dsm_render_bind_group(
    device: &wgpu::Device,
    pipeline: &DsmPipeline,
    uniforms: &wgpu::Buffer,
    interval_vertex_buf: &wgpu::Buffer,
    interval_tet_data_buf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dsm_render_bg"),
        layout: &pipeline.render_bg_layout,
        entries: &[
            buf_entry(0, uniforms),
            buf_entry(1, interval_vertex_buf),
            buf_entry(2, interval_tet_data_buf),
        ],
    })
}

/// Record a DSM render pass.
///
/// Clears the output to zero (full transmittance) and draws the interval
/// triangles with premultiplied-alpha blending. After the pass, the alpha
/// channel of `output_view` contains `1 - T_total`.
///
/// * `index_buf` — the static fan index buffer (`interval_fan_index_buf`, 12 u32s)
/// * `indirect_args_buf` — the `interval_args_buf`; draw-indexed-indirect args
///   start at byte offset 12 (skipping the 3 dispatch u32s)
/// * `output_view` — texture view for the DSM output
/// * `width`, `height` — output dimensions (for viewport/scissor)
pub fn record_dsm_render(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &DsmPipeline,
    bind_group: &wgpu::BindGroup,
    index_buf: &wgpu::Buffer,
    indirect_args_buf: &wgpu::Buffer,
    fourier_views: &[&wgpu::TextureView; FOURIER_MRT_COUNT],
    depth_view: &wgpu::TextureView,
    width: u32,
    height: u32,
) {
    let load_ops = wgpu::Operations {
        load: wgpu::LoadOp::Load, // preserve primitive coefficients
        store: wgpu::StoreOp::Store,
    };
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("dsm_render"),
        color_attachments: &[
            Some(wgpu::RenderPassColorAttachment {
                view: fourier_views[0], resolve_target: None, ops: load_ops, depth_slice: None,
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: fourier_views[1], resolve_target: None, ops: load_ops, depth_slice: None,
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: fourier_views[2], resolve_target: None, ops: load_ops, depth_slice: None,
            }),
        ],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });

    rpass.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
    rpass.set_scissor_rect(0, 0, width, height);
    rpass.set_pipeline(&pipeline.render_pipeline);
    rpass.set_bind_group(0, bind_group, &[]);
    rpass.set_index_buffer(index_buf.slice(..), wgpu::IndexFormat::Uint32);
    // Draw-indexed-indirect args at byte offset 12 (skip 3 dispatch u32s).
    rpass.draw_indexed_indirect(indirect_args_buf, 12);
}

// ---------------------------------------------------------------------------
// DSM Primitive Pipeline
// ---------------------------------------------------------------------------

/// Uniform data for one DSM primitive draw call, matching `PrimitiveUniformsPadded`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct DsmPrimUniform {
    vp: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    color: [f32; 4],
    _pad: [f32; 28],
}

const DSM_PRIM_UNIFORM_ALIGN: u64 = 256;
const DSM_PRIM_MAX: u64 = 256;

/// Pipeline for rendering opaque primitives into DSM (depth + full occlusion).
pub struct DsmPrimitivePipeline {
    pub render_pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

impl DsmPrimitivePipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dsm_primitive.wgsl"),
            source: wgpu::ShaderSource::Wgsl(DSM_PRIMITIVE_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dsm_prim_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<DsmPrimUniform>() as u64,
                    ),
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dsm_prim_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dsm_prim_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<PrimitiveVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 12,
                            shader_location: 1,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[
                    Some(wgpu::ColorTargetState { format: DSM_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: DSM_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: DSM_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                ],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dsm_prim_uniforms"),
            size: DSM_PRIM_UNIFORM_ALIGN * DSM_PRIM_MAX,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dsm_prim_bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<DsmPrimUniform>() as u64),
                }),
            }],
        });

        Self { render_pipeline, bind_group_layout, uniform_buffer, bind_group }
    }
}

/// Record a primitive pre-pass for one DSM face.
///
/// Clears color to TRANSPARENT and depth to 1.0, then draws opaque primitives
/// from the light's viewpoint. Primitives write alpha=1 (full occlusion) and
/// their depth, so subsequent tet intervals are culled behind them.
pub fn record_dsm_primitive_pass(
    encoder: &mut wgpu::CommandEncoder,
    queue: &wgpu::Queue,
    pipeline: &DsmPrimitivePipeline,
    geometry: &PrimitiveGeometry,
    fourier_views: &[&wgpu::TextureView; FOURIER_MRT_COUNT],
    depth_view: &wgpu::TextureView,
    primitives: &[Primitive],
    light_vp: &Mat4,
    near: f32,
    far: f32,
    width: u32,
    height: u32,
) {
    let clear_color = wgpu::Operations {
        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        store: wgpu::StoreOp::Store,
    };
    let clear_depth = wgpu::Operations {
        load: wgpu::LoadOp::Clear(1.0),
        store: wgpu::StoreOp::Store,
    };

    let color_attachments: [Option<wgpu::RenderPassColorAttachment>; FOURIER_MRT_COUNT] = std::array::from_fn(|i| {
        Some(wgpu::RenderPassColorAttachment {
            view: fourier_views[i],
            resolve_target: None,
            ops: clear_color,
            depth_slice: None,
        })
    });

    if primitives.is_empty() {
        let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("dsm_prim_clear"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(clear_depth),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        return;
    }

    // Write uniforms
    let vp_cols = light_vp.to_cols_array_2d();
    let count = primitives.len().min(DSM_PRIM_MAX as usize);

    for (i, prim) in primitives.iter().take(count).enumerate() {
        let model = prim.transform.model_matrix();
        let u = DsmPrimUniform {
            vp: vp_cols,
            model: model.to_cols_array_2d(),
            color: [near, far, 0.0, 0.0],
            _pad: [0.0; 28],
        };
        queue.write_buffer(
            &pipeline.uniform_buffer,
            i as u64 * DSM_PRIM_UNIFORM_ALIGN,
            bytemuck::bytes_of(&u),
        );
    }

    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("dsm_primitives"),
        color_attachments: &color_attachments,
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(clear_depth),
            stencil_ops: None,
        }),
        ..Default::default()
    });

    rpass.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
    rpass.set_scissor_rect(0, 0, width, height);
    rpass.set_pipeline(&pipeline.render_pipeline);
    rpass.set_vertex_buffer(0, geometry.vertex_buffer.slice(..));

    for (i, prim) in primitives.iter().take(count).enumerate() {
        let offset = i as u32 * DSM_PRIM_UNIFORM_ALIGN as u32;
        rpass.set_bind_group(0, &pipeline.bind_group, &[offset]);
        let slice = geometry.kinds[prim.kind.index()];
        rpass.draw(slice.offset..slice.offset + slice.count, 0..1);
    }
}

// ---------------------------------------------------------------------------
// DSM Resolve Pipeline (Fourier → T(z_query))
// ---------------------------------------------------------------------------

/// Uniform for the resolve pass: just a query depth.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct DsmResolveUniforms {
    pub z_query: f32,
    pub near: f32,
    pub far: f32,
    pub _pad: f32,
}

/// Fullscreen resolve pass: reads 3 Fourier textures, reconstructs T(z_query).
pub struct DsmResolvePipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub uniforms_buf: wgpu::Buffer,
}

impl DsmResolvePipeline {
    pub fn new(device: &wgpu::Device, output_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dsm_resolve.wgsl"),
            source: wgpu::ShaderSource::Wgsl(DSM_RESOLVE_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dsm_resolve_bgl"),
            entries: &[
                // 0: uniforms (z_query)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<DsmResolveUniforms>() as u64,
                        ),
                    },
                    count: None,
                },
                // 1-3: Fourier textures
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dsm_resolve_pl"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dsm_resolve_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let uniforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dsm_resolve_uniforms"),
            size: std::mem::size_of::<DsmResolveUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self { pipeline, bind_group_layout, uniforms_buf }
    }
}

/// Create the resolve bind group (uniform + 3 Fourier textures).
pub fn create_dsm_resolve_bind_group(
    device: &wgpu::Device,
    resolve: &DsmResolvePipeline,
    fourier_views: &[&wgpu::TextureView; FOURIER_MRT_COUNT],
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dsm_resolve_bg"),
        layout: &resolve.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: resolve.uniforms_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(fourier_views[0]),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(fourier_views[1]),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(fourier_views[2]),
            },
        ],
    })
}

/// Record resolve passes for all 6 cubemap faces in a cross layout.
///
/// Layout (4×3 grid, each cell = face_size × face_size):
/// ```
///        [+Y]
///  [-X]  [+Z]  [+X]  [-Z]
///        [-Y]
/// ```
pub fn record_dsm_resolve_cubemap_cross(
    encoder: &mut wgpu::CommandEncoder,
    device: &wgpu::Device,
    resolve: &DsmResolvePipeline,
    queue: &wgpu::Queue,
    atlas: &DsmAtlas,
    light_index: usize,
    z_query: f32,
    near: f32,
    far: f32,
    output_view: &wgpu::TextureView,
    output_width: u32,
    output_height: u32,
) {
    // Simple 3x2 grid: top row = faces 0,1,2; bottom row = faces 3,4,5
    // Face 0 (+X), Face 1 (-X), Face 2 (+Y), Face 3 (-Y), Face 4 (+Z), Face 5 (-Z)
    let face_w = output_width / 3;
    let face_h = output_height / 2;
    let face_offset = atlas.face_offsets.get(light_index).copied().unwrap_or(0) as usize;
    let face_count = atlas.face_counts.get(light_index).copied().unwrap_or(0) as usize;

    let positions: [(u32, u32); 6] = [
        (0, 0), (1, 0), (2, 0),  // +X, -X, +Y
        (0, 1), (1, 1), (2, 1),  // -Y, +Z, -Z
    ];

    queue.write_buffer(
        &resolve.uniforms_buf,
        0,
        bytemuck::bytes_of(&DsmResolveUniforms {
            z_query,
            near,
            far,
            _pad: 0.0,
        }),
    );

    // Clear output
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("dsm_cross_clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.2, g: 0.2, b: 0.2, a: 1.0 }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
    }

    // Render each face into its cross position
    for fi in 0..face_count.min(6) {
        let layer = face_offset + fi;
        if layer >= atlas.fourier_layer_views[0].len() { break; }

        let layer_views: [&wgpu::TextureView; FOURIER_MRT_COUNT] =
            std::array::from_fn(|m| &atlas.fourier_layer_views[m][layer]);

        let resolve_bg = create_dsm_resolve_bind_group(device, resolve, &layer_views);

        let (col, row) = positions[fi];
        let x = col * face_w;
        let y = row * face_h;

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("dsm_cross_face"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        rpass.set_viewport(x as f32, y as f32, face_w as f32, face_h as f32, 0.0, 1.0);
        rpass.set_pipeline(&resolve.pipeline);
        rpass.set_bind_group(0, &resolve_bg, &[]);
        rpass.draw(0..3, 0..1);
    }
}

/// Record the resolve pass: reads Fourier textures, writes T(z_query) to output.
pub fn record_dsm_resolve(
    encoder: &mut wgpu::CommandEncoder,
    resolve: &DsmResolvePipeline,
    bind_group: &wgpu::BindGroup,
    output_view: &wgpu::TextureView,
) {
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("dsm_resolve"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: output_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        ..Default::default()
    });
    rpass.set_pipeline(&resolve.pipeline);
    rpass.set_bind_group(0, bind_group, &[]);
    rpass.draw(0..3, 0..1);
}

// ---------------------------------------------------------------------------
// Per-Light DSM Cache
// ---------------------------------------------------------------------------

/// Cubemap face directions: +X, -X, +Y, -Y, +Z, -Z.
const CUBEMAP_DIRS: [(Vec3, Vec3); 6] = [
    (Vec3::X,     Vec3::NEG_Y),   // +X, up = -Y
    (Vec3::NEG_X, Vec3::NEG_Y),   // -X, up = -Y
    (Vec3::Y,     Vec3::Z),       // +Y, up = +Z
    (Vec3::NEG_Y, Vec3::NEG_Z),   // -Y, up = -Z
    (Vec3::Z,     Vec3::NEG_Y),   // +Z, up = -Y
    (Vec3::NEG_Z, Vec3::NEG_Y),   // -Z, up = -Y
];

/// Per-light shadow metadata for the deferred shader (matches WGSL `ShadowLight`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowLightMeta {
    /// VP matrices for each cubemap face (6 faces for point lights, 1 for spot/dir).
    pub vp_matrices: [[[f32; 4]; 4]; 6],
    pub face_offset: u32,
    pub face_count: u32,
    pub near: f32,
    pub far: f32,
    pub light_type: u32,
    pub _pad: [u32; 3],
}

/// Cached DSM atlas: texture arrays for all light faces + per-light metadata.
///
/// Fourier coefficients are stored in 3 `texture_2d_array` textures (one per MRT).
/// Each layer corresponds to one face of one light (point lights have 6 faces,
/// spot/directional have 1). DSM generation renders directly into per-layer views.
pub struct DsmAtlas {
    /// 3 texture arrays (one per Fourier MRT), with `total_layers` layers each.
    pub fourier_arrays: [wgpu::Texture; FOURIER_MRT_COUNT],
    /// D2Array views for shader binding.
    pub fourier_array_views: [wgpu::TextureView; FOURIER_MRT_COUNT],
    /// Per-layer D2 views for reading individual faces (resolve inspector).
    pub fourier_layer_views: [Vec<wgpu::TextureView>; FOURIER_MRT_COUNT],
    /// Staging textures for rendering one face at a time (then copied into array layers).
    pub staging_fourier: [wgpu::Texture; FOURIER_MRT_COUNT],
    pub staging_fourier_views: [wgpu::TextureView; FOURIER_MRT_COUNT],
    pub staging_depth: wgpu::Texture,
    pub staging_depth_view: wgpu::TextureView,
    /// Per-light shadow metadata storage buffer.
    pub meta_buf: wgpu::Buffer,
    /// Per-light face offsets (cumulative).
    pub face_offsets: Vec<u32>,
    /// Face count per light.
    pub face_counts: Vec<u32>,
    pub num_lights: u32,
    pub total_layers: u32,
    pub resolution: u32,
    /// Scratch uniform buffer for light viewpoint.
    pub scratch_uniforms: wgpu::Buffer,
}

impl DsmAtlas {
    /// Allocate atlas textures for the given light types.
    ///
    /// `light_types[i]`: 0 = point (6 faces), else = 1 face.
    pub fn new(device: &wgpu::Device, resolution: u32, light_types: &[u32]) -> Self {
        let mut face_offsets = Vec::with_capacity(light_types.len());
        let mut face_counts = Vec::with_capacity(light_types.len());
        let mut total_layers = 0u32;
        for &lt in light_types {
            face_offsets.push(total_layers);
            let fc = if lt == 0 { 6u32 } else { 1u32 };
            face_counts.push(fc);
            total_layers += fc;
        }
        // Ensure at least 1 layer (for dummy atlas when no lights)
        let array_layers = total_layers.max(1);

        let fourier_arrays: [wgpu::Texture; FOURIER_MRT_COUNT] = std::array::from_fn(|m| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("dsm_atlas_mrt{m}")),
                size: wgpu::Extent3d {
                    width: resolution,
                    height: resolution,
                    depth_or_array_layers: array_layers,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DSM_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        });

        let fourier_array_views: [wgpu::TextureView; FOURIER_MRT_COUNT] = std::array::from_fn(|m| {
            fourier_arrays[m].create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        });

        let fourier_layer_views: [Vec<wgpu::TextureView>; FOURIER_MRT_COUNT] = std::array::from_fn(|m| {
            (0..total_layers).map(|layer| {
                fourier_arrays[m].create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            }).collect()
        });

        // Staging textures: single 2D textures for rendering one face at a time
        let staging_fourier: [wgpu::Texture; FOURIER_MRT_COUNT] = std::array::from_fn(|m| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("dsm_staging_mrt{m}")),
                size: wgpu::Extent3d {
                    width: resolution,
                    height: resolution,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DSM_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        });
        let staging_fourier_views: [wgpu::TextureView; FOURIER_MRT_COUNT] = std::array::from_fn(|m| {
            staging_fourier[m].create_view(&wgpu::TextureViewDescriptor::default())
        });
        let staging_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dsm_staging_depth"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let staging_depth_view = staging_depth.create_view(&wgpu::TextureViewDescriptor::default());

        let meta_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dsm_shadow_meta"),
            size: (16 * std::mem::size_of::<ShadowLightMeta>()) as u64, // MAX_LIGHTS
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scratch_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dsm_scratch_uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            fourier_arrays,
            fourier_array_views,
            fourier_layer_views,
            staging_fourier,
            staging_fourier_views,
            staging_depth,
            staging_depth_view,
            meta_buf,
            face_offsets,
            face_counts,
            num_lights: light_types.len() as u32,
            total_layers,
            resolution,
            scratch_uniforms,
        }
    }

    /// Create a dummy atlas (1x1, 1 layer) for when no DSM is available.
    pub fn new_dummy(device: &wgpu::Device) -> Self {
        Self::new(device, 1, &[])
    }

    /// Fill metadata buffer with VP matrices and face offsets for all lights.
    pub fn populate_metadata(
        &self,
        queue: &wgpu::Queue,
        lights: &[GpuLight],
        near: f32,
        far: f32,
    ) {
        let mut metas = vec![ShadowLightMeta::zeroed(); 16]; // MAX_LIGHTS
        for (li, light) in lights.iter().enumerate() {
            if li >= 16 { break; }
            let fc = self.face_counts[li] as usize;
            let mut vps = [[[0.0f32; 4]; 4]; 6];
            for fi in 0..fc {
                let (vp, _c2w) = build_light_vp(light, fi, near, far);
                vps[fi] = vp.to_cols_array_2d();
            }
            metas[li] = ShadowLightMeta {
                vp_matrices: vps,
                face_offset: self.face_offsets[li],
                face_count: fc as u32,
                near,
                far,
                light_type: light.light_type,
                _pad: [0; 3],
            };
        }
        queue.write_buffer(&self.meta_buf, 0, bytemuck::cast_slice(&metas));
    }
}

/// Build a view-projection matrix for a light face.
///
/// For point lights (type 0), `face_index` selects the cubemap face (0..6).
/// For spot lights (type 1), uses the light's direction and outer cone angle.
/// For directional lights (type 2), uses an orthographic projection along direction.
pub fn build_light_vp(
    light: &GpuLight,
    face_index: usize,
    near: f32,
    far: f32,
) -> (Mat4, Mat3) {
    let pos = Vec3::from(light.position);

    let (forward, up) = match light.light_type {
        0 => {
            // Point light cubemap face
            CUBEMAP_DIRS[face_index]
        }
        1 => {
            // Spot light — look along direction
            let dir = Vec3::from(light.direction).normalize_or_zero();
            let tentative_up = if dir.y.abs() > 0.99 { Vec3::X } else { Vec3::Y };
            (dir, tentative_up)
        }
        _ => {
            // Directional light — look along direction
            let dir = Vec3::from(light.direction).normalize_or_zero();
            let tentative_up = if dir.y.abs() > 0.99 { Vec3::X } else { Vec3::Y };
            (dir, tentative_up)
        }
    };

    let view = Mat4::look_to_rh(pos, forward, up);

    let proj = match light.light_type {
        0 => {
            // Cubemap face: 90° square FOV
            Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, near, far)
        }
        1 => {
            // Spot light: FOV = 2 * outer_angle
            let fov = (2.0 * light.outer_angle).max(0.01);
            Mat4::perspective_rh(fov, 1.0, near, far)
        }
        _ => {
            // Directional: orthographic (scene-sized box)
            Mat4::orthographic_rh(-10.0, 10.0, -10.0, 10.0, near, far)
        }
    };

    // Camera-to-world rotation = inverse of the 3×3 part of view
    let view3 = Mat3::from_cols(
        view.col(0).truncate(),
        view.col(1).truncate(),
        view.col(2).truncate(),
    );
    let c2w = view3.transpose();

    (proj * view, c2w)
}

/// Generate deep shadow maps for all active lights.
///
/// For each light:
///   1. Forward compute (sort keys from light position)
///   2. Radix sort (once per light — valid for all cubemap faces)
///   3. Indirect convert
///   4. Per face: interval gen + DSM render
pub fn generate_dsm_for_lights(
    atlas: &DsmAtlas,
    encoder: &mut wgpu::CommandEncoder,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    dsm_pipeline: &DsmPipeline,
    dsm_prim_pipeline: &DsmPrimitivePipeline,
    prim_geometry: &PrimitiveGeometry,
    primitives: &[Primitive],
    fwd_pipelines: &ForwardPipelines,
    ci_pipelines: &ComputeIntervalPipelines,
    sort_pipelines: &rmesh_sort::RadixSortPipelines,
    sort_state: &rmesh_sort::RadixSortState,
    buffers: &SceneBuffers,
    material: &MaterialBuffers,
    sh_coeffs_buf: &wgpu::Buffer,
    lights: &[GpuLight],
    num_lights: u32,
    tet_count: u32,
    near: f32,
    far: f32,
    render_tets: bool,
) {
    let res = atlas.resolution;
    let n_pow2 = tet_count.next_power_of_two();

    // Intrinsics for 90° square FOV at DSM resolution (used for cubemap + spot ray_scale)
    let half = res as f32 / 2.0;

    for li in 0..num_lights.min(lights.len() as u32) {
        let light = &lights[li as usize];
        let face_count = atlas.face_counts[li as usize] as usize;
        let face_offset = atlas.face_offsets[li as usize] as usize;
        let pos = Vec3::from(light.position);

        // --- Sort pass (once per light position) ---
        let (first_vp, first_c2w) = build_light_vp(light, 0, near, far);

        let intrinsics = [half, half, half, half];

        let uniforms = rmesh_render::make_uniforms(
            first_vp,
            first_c2w,
            intrinsics,
            pos,
            res as f32,
            res as f32,
            tet_count,
            0,
            4,
            0.0,
            0,
            near,
            far,
        );
        queue.write_buffer(
            &atlas.scratch_uniforms,
            0,
            bytemuck::bytes_of(&uniforms),
        );

        // For DSM (OIT additive blending), no sort needed — just identity mapping.
        if render_tets {
            // sort_values = [0, 1, 2, ..., tet_count-1]
            let identity: Vec<u32> = (0..tet_count).collect();
            queue.write_buffer(&buffers.sort_values, 0, bytemuck::cast_slice(&identity));

            // Set up indirect_args (for forward compute compatibility)
            let args = DrawIndirectCommand {
                vertex_count: 12,
                instance_count: tet_count,
                first_vertex: 0,
                first_instance: 0,
            };
            queue.write_buffer(&buffers.indirect_args, 0, bytemuck::bytes_of(&args));

            // Set up interval_args_buf with 2D dispatch for large tet counts
            let wg_size = 64u32;
            let total_wg = (tet_count + wg_size - 1) / wg_size;
            let dispatch_x = total_wg.min(65535);
            let dispatch_y = (total_wg + 65534) / 65535;
            let interval_args: [u32; 8] = [
                dispatch_x, dispatch_y, 1,         // compute dispatch args (2D)
                tet_count * 12, 1, 0, 0, 0,        // draw-indexed-indirect args
            ];
            queue.write_buffer(&buffers.interval_args_buf, 0, bytemuck::cast_slice(&interval_args));
        }

        // --- Per-face: primitive pre-pass + optional tet DSM render ---
        // Submit + recreate encoder between faces because queue.write_buffer
        // for scratch_uniforms must take effect before the next face's GPU work.
        for fi in 0..face_count {
            let (face_vp, face_c2w) = build_light_vp(light, fi, near, far);

            // For 90° FOV cubemap faces, fx = fy = half (tan(45°) = 1).
            // Don't extract from VP matrix — it's only correct for Z-forward cameras.
            let face_intrinsics = [half, half, half, half];

            let face_uniforms = rmesh_render::make_uniforms(
                face_vp,
                face_c2w,
                face_intrinsics,
                pos,
                res as f32,
                res as f32,
                tet_count,
                0,
                4,
                0.0,
                0,
                near,
                far,
            );
            // Submit previous work so the new write_buffer takes effect
            let old_encoder = std::mem::replace(encoder, device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("dsm_face") },
            ));
            queue.submit(std::iter::once(old_encoder.finish()));
            queue.write_buffer(
                &atlas.scratch_uniforms,
                0,
                bytemuck::bytes_of(&face_uniforms),
            );

            // Re-initialize buffers for this face (interval gen modifies them)
            if render_tets {
                // Reset indirect_args (instance_count = all tets)
                let args = DrawIndirectCommand {
                    vertex_count: 12,
                    instance_count: tet_count,
                    first_vertex: 0,
                    first_instance: 0,
                };
                queue.write_buffer(&buffers.indirect_args, 0, bytemuck::bytes_of(&args));

                // Reset interval_args_buf with 2D dispatch for large tet counts
                let wg_size = 64u32;
                let total_wg = (tet_count + wg_size - 1) / wg_size;
                let dispatch_x = total_wg.min(65535);
                let dispatch_y = (total_wg + 65534) / 65535;
                let interval_args: [u32; 8] = [
                    dispatch_x, dispatch_y, 1,
                    tet_count * 12, 1, 0, 0, 0,
                ];
                queue.write_buffer(&buffers.interval_args_buf, 0, bytemuck::cast_slice(&interval_args));

                // Reset sort_values to identity
                let identity: Vec<u32> = (0..tet_count).collect();
                queue.write_buffer(&buffers.sort_values, 0, bytemuck::cast_slice(&identity));
            }

            // Render to staging textures
            let staging_views: [&wgpu::TextureView; FOURIER_MRT_COUNT] = std::array::from_fn(|m| {
                &atlas.staging_fourier_views[m]
            });

            // Primitive pre-pass: clear color+depth, draw opaque primitives
            record_dsm_primitive_pass(
                encoder,
                queue,
                dsm_prim_pipeline,
                prim_geometry,
                &staging_views,
                &atlas.staging_depth_view,
                primitives,
                &face_vp,
                near,
                far,
                res,
                res,
            );

            // DSM tet render pass (loads color+depth from primitive pass)
            if render_tets {
                let gen_bg = create_dsm_interval_gen_bg(
                    device,
                    ci_pipelines,
                    buffers,
                    material,
                    &buffers.sort_values,
                    &atlas.scratch_uniforms,
                );
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("dsm_interval_gen"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&ci_pipelines.gen_pipeline);
                    cpass.set_bind_group(0, &gen_bg, &[]);
                    cpass.dispatch_workgroups_indirect(&buffers.interval_args_buf, 0);
                }

                let render_bg = create_dsm_render_bind_group(
                    device,
                    dsm_pipeline,
                    &atlas.scratch_uniforms,
                    &buffers.interval_vertex_buf,
                    &buffers.interval_tet_data_buf,
                );
                record_dsm_render(
                    encoder,
                    dsm_pipeline,
                    &render_bg,
                    &buffers.interval_fan_index_buf,
                    &buffers.interval_args_buf,
                    &staging_views,
                    &atlas.staging_depth_view,
                    res,
                    res,
                );
            }

            // Copy staging textures into atlas array layers
            let layer = face_offset + fi;
            let copy_size = wgpu::Extent3d { width: res, height: res, depth_or_array_layers: 1 };
            for m in 0..FOURIER_MRT_COUNT {
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &atlas.staging_fourier[m],
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &atlas.fourier_arrays[m],
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: layer as u32 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    copy_size,
                );
            }
        }
    }
}

/// Create a HW projection compute bind group with scratch uniforms.
///
/// Mirrors `rmesh_render::create_hw_compute_bind_group` but uses a custom
/// uniforms buffer instead of `buffers.uniforms`.
fn create_dsm_hw_compute_bg(
    device: &wgpu::Device,
    fwd_pipelines: &ForwardPipelines,
    buffers: &SceneBuffers,
    material: &MaterialBuffers,
    sh_coeffs_buf: &wgpu::Buffer,
    scratch_uniforms: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dsm_hw_compute_bg"),
        layout: &fwd_pipelines.hw_compute_bind_group_layout,
        entries: &[
            buf_entry(0, scratch_uniforms),
            buf_entry(1, &buffers.vertices),
            buf_entry(2, &buffers.indices),
            buf_entry(3, &buffers.circumdata),
            buf_entry(4, &buffers.sort_keys),
            buf_entry(5, &buffers.sort_values),
            buf_entry(6, &buffers.indirect_args),
            buf_entry(7, &material.colors),
            buf_entry(8, &material.base_colors),
            buf_entry(9, &material.color_grads),
            buf_entry(10, sh_coeffs_buf),
        ],
    })
}

/// Create the indirect-convert bind group.
fn create_dsm_indirect_convert_bg(
    device: &wgpu::Device,
    ci_pipelines: &ComputeIntervalPipelines,
    buffers: &SceneBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dsm_indirect_convert_bg"),
        layout: &ci_pipelines.indirect_convert_bg_layout,
        entries: &[
            buf_entry(0, &buffers.indirect_args),
            buf_entry(1, &buffers.interval_args_buf),
        ],
    })
}

/// Create the interval-gen bind group with scratch uniforms and explicit sort_values.
fn create_dsm_interval_gen_bg(
    device: &wgpu::Device,
    ci_pipelines: &ComputeIntervalPipelines,
    buffers: &SceneBuffers,
    material: &MaterialBuffers,
    sort_values: &wgpu::Buffer,
    scratch_uniforms: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dsm_interval_gen_bg"),
        layout: &ci_pipelines.gen_bg_layout,
        entries: &[
            buf_entry(0, scratch_uniforms),
            buf_entry(1, &buffers.vertices),
            buf_entry(2, &buffers.indices),
            buf_entry(3, &material.colors),
            buf_entry(4, &buffers.densities),
            buf_entry(5, &material.color_grads),
            buf_entry(6, sort_values),
            buf_entry(7, &buffers.indirect_args),
            buf_entry(8, &buffers.interval_vertex_buf),
            buf_entry(9, &buffers.interval_tet_data_buf),
            buf_entry(10, &buffers.vertex_normals),
        ],
    })
}
