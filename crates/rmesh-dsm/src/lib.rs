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

/// Per-light DSM entry. Point lights have 6 face textures (cubemap),
/// spot/directional lights have 1.
pub struct DsmEntry {
    /// Fourier coefficient textures: [mrt_index][face_index].
    pub fourier_textures: [Vec<wgpu::Texture>; FOURIER_MRT_COUNT],
    pub fourier_views: [Vec<wgpu::TextureView>; FOURIER_MRT_COUNT],
    pub depth_textures: Vec<wgpu::Texture>,
    pub depth_views: Vec<wgpu::TextureView>,
}

/// Cached deep shadow map textures for all active lights.
pub struct DsmLightMaps {
    pub entries: Vec<DsmEntry>,
    pub resolution: u32,
    /// Scratch uniform buffer for light viewpoint (avoids clobbering camera uniforms).
    pub scratch_uniforms: wgpu::Buffer,
}

impl DsmLightMaps {
    /// Allocate DSM textures for `num_lights` lights.
    ///
    /// `light_types[i]` determines face count: 0 (point) → 6 faces, else → 1 face.
    pub fn new(device: &wgpu::Device, resolution: u32, light_types: &[u32]) -> Self {
        let mut entries = Vec::with_capacity(light_types.len());
        for (i, &lt) in light_types.iter().enumerate() {
            let face_count = if lt == 0 { 6 } else { 1 };
            let mut fourier_textures: [Vec<wgpu::Texture>; FOURIER_MRT_COUNT] = std::array::from_fn(|_| Vec::with_capacity(face_count));
            let mut fourier_views: [Vec<wgpu::TextureView>; FOURIER_MRT_COUNT] = std::array::from_fn(|_| Vec::with_capacity(face_count));
            let mut depth_textures = Vec::with_capacity(face_count);
            let mut depth_views = Vec::with_capacity(face_count);
            for f in 0..face_count {
                for m in 0..FOURIER_MRT_COUNT {
                    let tex = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some(&format!("dsm_light{i}_face{f}_mrt{m}")),
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
                            | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    });
                    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
                    fourier_textures[m].push(tex);
                    fourier_views[m].push(view);
                }

                let dtex = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(&format!("dsm_depth_light{i}_face{f}")),
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
                let dview = dtex.create_view(&wgpu::TextureViewDescriptor::default());
                depth_textures.push(dtex);
                depth_views.push(dview);
            }
            entries.push(DsmEntry { fourier_textures, fourier_views, depth_textures, depth_views });
        }

        let scratch_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dsm_scratch_uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            entries,
            resolution,
            scratch_uniforms,
        }
    }
}

/// Build a view-projection matrix for a light face.
///
/// For point lights (type 0), `face_index` selects the cubemap face (0..6).
/// For spot lights (type 1), uses the light's direction and outer cone angle.
/// For directional lights (type 2), uses an orthographic projection along direction.
fn build_light_vp(
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
    maps: &DsmLightMaps,
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
) {
    let res = maps.resolution;
    let n_pow2 = tet_count.next_power_of_two();

    // Intrinsics for 90° square FOV at DSM resolution (used for cubemap + spot ray_scale)
    let half = res as f32 / 2.0;

    for li in 0..num_lights.min(lights.len() as u32) {
        let light = &lights[li as usize];
        let entry = &maps.entries[li as usize];
        let face_count = entry.depth_views.len();
        let pos = Vec3::from(light.position);

        // --- Sort pass (once per light position) ---
        // Use the first face VP for the forward compute (sort only uses cam_pos)
        let (first_vp, first_c2w) = build_light_vp(light, 0, near, far);

        // Compute intrinsics matching the projection
        let fx = first_vp.col(0).x * half;
        let fy = first_vp.col(1).y * half;
        let intrinsics = [fx.abs(), fy.abs(), half, half];

        let uniforms = rmesh_render::make_uniforms(
            first_vp,
            first_c2w,
            intrinsics,
            pos,
            res as f32,
            res as f32,
            tet_count,
            0,  // step
            4,  // tile_size (unused for DSM)
            0.0,
            0,  // sh_degree (unused for DSM)
            near,
            far,
        );
        queue.write_buffer(
            &maps.scratch_uniforms,
            0,
            bytemuck::bytes_of(&uniforms),
        );

        // Reset indirect args + interval_args_buf
        let reset_cmd = DrawIndirectCommand {
            vertex_count: 12,
            instance_count: 0,
            first_vertex: 0,
            first_instance: 0,
        };
        queue.write_buffer(&buffers.indirect_args, 0, bytemuck::bytes_of(&reset_cmd));
        queue.write_buffer(&buffers.interval_args_buf, 0, bytemuck::cast_slice(&[0u32; 8]));
        queue.write_buffer(sort_state.num_keys_buf(), 0, bytemuck::bytes_of(&n_pow2));

        // Create compute bind group with scratch uniforms
        let compute_bg = create_dsm_hw_compute_bg(
            device,
            fwd_pipelines,
            buffers,
            material,
            sh_coeffs_buf,
            &maps.scratch_uniforms,
        );

        // Forward compute (projection + sort keys)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dsm_project_compute"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&fwd_pipelines.hw_compute_pipeline);
            cpass.set_bind_group(0, &compute_bg, &[]);
            let workgroup_size = 64u32;
            let total_workgroups = (n_pow2 + workgroup_size - 1) / workgroup_size;
            let (dx, dy) = dispatch_2d(total_workgroups);
            cpass.dispatch_workgroups(dx, dy, 1);
        }

        // Radix sort
        let result_in_b = rmesh_sort::record_radix_sort(
            encoder,
            device,
            sort_pipelines,
            sort_state,
            &buffers.sort_keys,
            &buffers.sort_values,
        );

        // Indirect convert
        let convert_bg = create_dsm_indirect_convert_bg(
            device,
            ci_pipelines,
            buffers,
        );
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dsm_indirect_convert"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&ci_pipelines.indirect_convert_pipeline);
            cpass.set_bind_group(0, &convert_bg, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        // --- Per-face: interval gen + DSM render ---
        for fi in 0..face_count {
            let (face_vp, face_c2w) = build_light_vp(light, fi, near, far);

            let fx = face_vp.col(0).x * half;
            let fy = face_vp.col(1).y * half;
            let face_intrinsics = [fx.abs(), fy.abs(), half, half];

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
            queue.write_buffer(
                &maps.scratch_uniforms,
                0,
                bytemuck::bytes_of(&face_uniforms),
            );

            // Interval gen compute
            let sort_values = if result_in_b {
                sort_state.values_b()
            } else {
                &buffers.sort_values
            };
            let gen_bg = create_dsm_interval_gen_bg(
                device,
                ci_pipelines,
                buffers,
                material,
                sort_values,
                &maps.scratch_uniforms,
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

            // Build per-face Fourier view array
            let face_fourier_views: [&wgpu::TextureView; FOURIER_MRT_COUNT] = std::array::from_fn(|m| {
                &entry.fourier_views[m][fi]
            });

            // Primitive pre-pass: clear color+depth, draw opaque primitives
            record_dsm_primitive_pass(
                encoder,
                queue,
                dsm_prim_pipeline,
                prim_geometry,
                &face_fourier_views,
                &entry.depth_views[fi],
                primitives,
                &face_vp,
                near,
                far,
                res,
                res,
            );

            // DSM tet render pass (loads color+depth from primitive pass)
            let render_bg = create_dsm_render_bind_group(
                device,
                dsm_pipeline,
                &maps.scratch_uniforms,
                &buffers.interval_vertex_buf,
                &buffers.interval_tet_data_buf,
            );
            record_dsm_render(
                encoder,
                dsm_pipeline,
                &render_bg,
                &buffers.interval_fan_index_buf,
                &buffers.interval_args_buf,
                &face_fourier_views,
                &entry.depth_views[fi],
                res,
                res,
            );
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
