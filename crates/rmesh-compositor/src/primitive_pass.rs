//! Render pass that draws opaque primitives to their own color + depth targets.

use bytemuck::{Pod, Zeroable};
use crate::geometry::{PrimitiveGeometry, PrimitiveVertex};
use rmesh_interact::Primitive;

const PRIMITIVE_WGSL: &str = include_str!("wgsl/primitive.wgsl");
const PRIMITIVE_MRT_WGSL: &str = include_str!("wgsl/primitive_mrt.wgsl");

/// Uniform data for one primitive draw call, padded to 256 bytes for dynamic offsets.
/// Layout: vp(64) + model(64) + color(16) + pad(112) = 256 bytes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PrimitiveUniformsPadded {
    pub vp: [[f32; 4]; 4],    // 64 bytes
    pub model: [[f32; 4]; 4], // 64 bytes
    pub color: [f32; 4],      // 16 bytes
    pub _pad: [f32; 28],      // 112 bytes → total 256
}

const UNIFORM_ALIGN: u64 = 256;
const MAX_PRIMITIVES: u64 = 256;

/// Default primitive colors (indexed by PrimitiveKind::index()).
const PRIMITIVE_COLORS: [[f32; 4]; 4] = [
    [0.8, 0.3, 0.2, 1.0], // Cube: red-ish
    [0.2, 0.5, 0.8, 1.0], // Sphere: blue-ish
    [0.3, 0.7, 0.3, 1.0], // Plane: green-ish
    [0.7, 0.5, 0.2, 1.0], // Cylinder: orange-ish
];

/// Offscreen color + depth targets for primitive rendering.
pub struct PrimitiveTargets {
    pub color_texture: wgpu::Texture,
    pub color_view: wgpu::TextureView,
    pub depth_texture: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
}

impl PrimitiveTargets {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("prim_color"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("prim_depth"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self { color_texture, color_view, depth_texture, depth_view }
    }
}

/// Additional MRT target views for primitive rendering when deferred shading is active.
pub struct MrtViews<'a> {
    pub aux0_view: &'a wgpu::TextureView,
    pub normals_view: &'a wgpu::TextureView,
    pub albedo_view: &'a wgpu::TextureView,
}

/// Pipeline and resources for rendering primitives.
pub struct PrimitivePipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub pipeline_mrt: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

impl PrimitivePipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("primitive.wgsl"),
            source: wgpu::ShaderSource::Wgsl(PRIMITIVE_WGSL.into()),
        });
        let shader_mrt = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("primitive_mrt.wgsl"),
            source: wgpu::ShaderSource::Wgsl(PRIMITIVE_MRT_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prim_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<PrimitiveUniformsPadded>() as u64,
                    ),
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prim_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let color_format = wgpu::TextureFormat::Rgba16Float;

        let vertex_state = wgpu::VertexBufferLayout {
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
        };

        let depth_stencil = wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };

        let primitive_state = wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        };

        // Single-target pipeline (non-MRT)
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("prim_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_state.clone()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: primitive_state,
            depth_stencil: Some(depth_stencil.clone()),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // MRT pipeline (4 color targets matching interval pipeline)
        let opaque_target = Some(wgpu::ColorTargetState {
            format: color_format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        });
        let pipeline_mrt = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("prim_pipeline_mrt"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_mrt,
                entry_point: Some("vs_main"),
                buffers: &[vertex_state],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_mrt,
                entry_point: Some("fs_main"),
                targets: &[
                    opaque_target.clone(), // location(0): color/plaster
                    opaque_target.clone(), // location(1): aux0
                    opaque_target.clone(), // location(2): normals
                    opaque_target,         // location(3): albedo
                ],
                compilation_options: Default::default(),
            }),
            primitive: primitive_state,
            depth_stencil: Some(depth_stencil),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prim_uniforms"),
            size: UNIFORM_ALIGN * MAX_PRIMITIVES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prim_bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(
                        std::mem::size_of::<PrimitiveUniformsPadded>() as u64,
                    ),
                }),
            }],
        });

        Self { pipeline, pipeline_mrt, bind_group_layout, uniform_buffer, bind_group }
    }
}

/// Record a primitive render pass.
///
/// When `mrt_views` is `Some`, renders to 4 MRT targets (color + aux0 + normals + albedo)
/// for deferred shading. Otherwise renders to a single color target.
pub fn record_primitive_pass(
    encoder: &mut wgpu::CommandEncoder,
    queue: &wgpu::Queue,
    pipeline: &PrimitivePipeline,
    geometry: &PrimitiveGeometry,
    color_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    primitives: &[Primitive],
    vp: &glam::Mat4,
    mrt_views: Option<MrtViews>,
) {
    let clear_ops = wgpu::Operations {
        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        store: wgpu::StoreOp::Store,
    };

    let color_attachment = Some(wgpu::RenderPassColorAttachment {
        view: color_view,
        resolve_target: None,
        ops: clear_ops,
        depth_slice: None,
    });

    // Build color attachments based on MRT mode
    let mrt_attachments;
    let single_attachments;
    let color_attachments: &[Option<wgpu::RenderPassColorAttachment>] = if let Some(ref mrt) = mrt_views {
        mrt_attachments = [
            color_attachment,
            Some(wgpu::RenderPassColorAttachment {
                view: mrt.aux0_view,
                resolve_target: None,
                ops: clear_ops,
                depth_slice: None,
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: mrt.normals_view,
                resolve_target: None,
                ops: clear_ops,
                depth_slice: None,
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: mrt.albedo_view,
                resolve_target: None,
                ops: clear_ops,
                depth_slice: None,
            }),
        ];
        &mrt_attachments
    } else {
        single_attachments = [color_attachment];
        &single_attachments
    };

    if primitives.is_empty() {
        // Still clear the targets
        let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("prim_clear"),
            color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        return;
    }

    // Write uniforms for each primitive
    let vp_cols = vp.to_cols_array_2d();
    let count = primitives.len().min(MAX_PRIMITIVES as usize);

    for (i, prim) in primitives.iter().take(count).enumerate() {
        let model = prim.transform.model_matrix();
        let model_cols = model.to_cols_array_2d();
        let color = PRIMITIVE_COLORS[prim.kind.index()];

        let u = PrimitiveUniformsPadded {
            vp: vp_cols,
            model: model_cols,
            color,
            _pad: [0.0; 28],
        };

        queue.write_buffer(
            &pipeline.uniform_buffer,
            i as u64 * UNIFORM_ALIGN,
            bytemuck::bytes_of(&u),
        );
    }

    // Record render pass
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("primitives"),
        color_attachments,
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        ..Default::default()
    });

    let active_pipeline = if mrt_views.is_some() {
        &pipeline.pipeline_mrt
    } else {
        &pipeline.pipeline
    };
    rpass.set_pipeline(active_pipeline);
    rpass.set_vertex_buffer(0, geometry.vertex_buffer.slice(..));

    for (i, prim) in primitives.iter().take(count).enumerate() {
        let offset = (i as u64 * UNIFORM_ALIGN) as u32;
        rpass.set_bind_group(0, &pipeline.bind_group, &[offset]);

        let slice = &geometry.kinds[prim.kind.index()];
        rpass.draw(slice.offset..(slice.offset + slice.count), 0..1);
    }
}
