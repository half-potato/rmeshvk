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

const INTERVAL_VERTEX_WGSL: &str = include_str!("wgsl/interval_vertex.wgsl");
const DSM_FRAGMENT_WGSL: &str = include_str!("wgsl/dsm_fragment.wgsl");

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
            label: Some("dsm_fragment.wgsl"),
            source: wgpu::ShaderSource::Wgsl(DSM_FRAGMENT_WGSL.into()),
        });

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
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &fragment_shader,
                entry_point: Some("main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(premul_blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
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
    output_view: &wgpu::TextureView,
    width: u32,
    height: u32,
) {
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("dsm_render"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: output_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
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
