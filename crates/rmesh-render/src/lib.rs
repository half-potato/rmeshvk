//! Forward rendering pipeline orchestration (wgpu).
//!
//! Sets up the wgpu compute and render pipelines for the forward pass:
//!   1. Compute pass: SH eval, cull, depth key generation
//!   2. Render pass: Hardware rasterization with MRT output
//!
//! All GPU buffer management and bind group creation lives here.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use rmesh_data::SceneData;
use std::collections::HashMap;
use wgpu::util::DeviceExt;

// Re-export shared types for CPU-side use.
pub use rmesh_util::shared::{BVHNode, DrawIndirectCommand, Uniforms};

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
const PROJECT_COMPUTE_WGSL: &str = include_str!("wgsl/project_compute.wgsl");
const PROJECT_COMPUTE_HW_WGSL: &str = include_str!("wgsl/project_compute_hw.wgsl");
const FORWARD_VERTEX_QUAD_WGSL: &str = include_str!("wgsl/forward_vertex_quad.wgsl");
const FORWARD_PREPASS_COMPUTE_WGSL: &str = include_str!("wgsl/forward_prepass_compute.wgsl");
const FORWARD_VERTEX_WGSL: &str = include_str!("wgsl/forward_vertex.wgsl");
const FORWARD_FRAGMENT_WGSL: &str = include_str!("wgsl/forward_fragment.wgsl");
const TEX_TO_BUFFER_WGSL: &str = include_str!("wgsl/tex_to_buffer.wgsl");
const BLIT_WGSL: &str = include_str!("wgsl/blit.wgsl");
const RAYTRACE_COMPUTE_WGSL: &str = include_str!("wgsl/raytrace_compute.wgsl");
const RASTERIZE_COMPUTE_WGSL: &str = include_str!("wgsl/rasterize_compute.wgsl");
const LOCATE_COMPUTE_WGSL: &str = include_str!("wgsl/locate_compute.wgsl");
const FORWARD_MESH_WGSL: &str = include_str!("wgsl/forward_mesh.wgsl");
const INDIRECT_CONVERT_WGSL: &str = include_str!("wgsl/indirect_convert.wgsl");
const INTERVAL_MESH_WGSL: &str = include_str!("wgsl/interval_mesh.wgsl");
const INTERVAL_FRAGMENT_WGSL: &str = include_str!("wgsl/interval_fragment.wgsl");
const INTERVAL_COMPUTE_WGSL: &str = include_str!("wgsl/interval_compute.wgsl");
const INTERVAL_VERTEX_WGSL: &str = include_str!("wgsl/interval_vertex.wgsl");
const INTERVAL_INDIRECT_CONVERT_WGSL: &str = include_str!("wgsl/interval_indirect_convert.wgsl");

// ---------------------------------------------------------------------------
// GPU Buffers
// ---------------------------------------------------------------------------

/// GPU buffers for scene geometry (independent of material/appearance model).
pub struct SceneBuffers {
    /// Vertex positions [N x 3] f32
    pub vertices: wgpu::Buffer,
    /// Tet vertex indices [M x 4] u32
    pub indices: wgpu::Buffer,
    /// Per-tet density [M] f32
    pub densities: wgpu::Buffer,
    /// Circumsphere data [M x 4] f32 (cx, cy, cz, r^2)
    pub circumdata: wgpu::Buffer,
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
    /// Mesh shader indirect dispatch args [3] u32 (x, y, z workgroup counts)
    pub mesh_indirect_args: wgpu::Buffer,
    /// Precomputed per-tet data for quad renderer [M × 8] vec4<f32> (128 bytes/tet)
    pub precomputed: wgpu::Buffer,
    /// Compute-interval vertex buffer [M × 5 × 2] vec4<f32> (160 bytes/tet)
    pub interval_vertex_buf: wgpu::Buffer,
    /// Compute-interval per-tet flat data [M] vec4<f32> (16 bytes/tet)
    pub interval_tet_data_buf: wgpu::Buffer,
    /// Static fan index buffer [12] u32 — shared across all tets via instanced draw
    pub interval_fan_index_buf: wgpu::Buffer,
    /// Combined compute dispatch + draw-indexed-indirect args [8] u32 (32 bytes)
    pub interval_args_buf: wgpu::Buffer,
}

/// GPU buffers for per-tet material/appearance data (pluggable per rendering mode).
pub struct MaterialBuffers {
    /// Per-tet base color [M x 3] f32 (uploaded from model, input to softplus)
    pub base_colors: wgpu::Buffer,
    /// Per-tet color gradient [M x 3] f32
    pub color_grads: wgpu::Buffer,
    /// Evaluated per-tet color [M x 3] f32 (written by project_compute)
    pub colors: wgpu::Buffer,
}

impl SceneBuffers {
    /// Upload scene geometry to GPU buffers.
    pub fn upload(device: &wgpu::Device, _queue: &wgpu::Queue, scene: &SceneData) -> Self {
        let storage_copy = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
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

        let densities = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("densities"),
            contents: bytemuck::cast_slice(&scene.densities),
            usage: trainable,
        });

        let circumdata = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("circumdata"),
            contents: bytemuck::cast_slice(&scene.circumdata),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let m = scene.tet_count as u64;
        let n_pow2 = (scene.tet_count as u64).next_power_of_two();

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
            size: 16,
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

        let mesh_indirect_args = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_indirect_args"),
            contents: bytemuck::cast_slice(&[0u32, 1u32, 1u32]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
        });

        // Precomputed buffer: 10 × vec4<f32> = 160 bytes per tet
        let precomputed = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("precomputed"),
            size: m * 10 * 16, // 160 bytes per tet
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Compute-interval vertex buffer: 5 verts × 2 vec4s × 16 bytes = 160 bytes/tet
        let interval_vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("interval_vertex_buf"),
            size: m * 5 * 2 * 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Compute-interval per-tet flat data: 1 vec4 × 16 bytes = 16 bytes/tet
        let interval_tet_data_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("interval_tet_data_buf"),
            size: m * 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Static fan index buffer: M × 12 u32s, created once (never written per frame).
        // Fan pattern per tet i: (i*5+0,i*5+1,i*5+4), (i*5+1,i*5+2,i*5+4),
        //                        (i*5+2,i*5+3,i*5+4), (i*5+3,i*5+0,i*5+4)
        let fan_indices: Vec<u32> = (0..m as u32)
            .flat_map(|i| {
                let b = i * 5;
                [
                    b, b + 1, b + 4,
                    b + 1, b + 2, b + 4,
                    b + 2, b + 3, b + 4,
                    b + 3, b, b + 4,
                ]
            })
            .collect();
        let interval_fan_index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("interval_fan_index_buf"),
            contents: bytemuck::cast_slice(&fan_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Combined dispatch + draw-indexed-indirect args: 8 × u32 = 32 bytes
        let interval_args_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("interval_args_buf"),
            size: 32,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            vertices,
            indices,
            densities,
            circumdata,
            sort_keys,
            sort_values,
            indirect_args,
            uniforms,
            tiles_touched,
            compact_tet_ids,
            mesh_indirect_args,
            precomputed,
            interval_vertex_buf,
            interval_tet_data_buf,
            interval_fan_index_buf,
            interval_args_buf,
        }
    }
}

impl MaterialBuffers {
    /// Upload material data to GPU buffers.
    ///
    /// * `base_colors` — per-tet base color `[M × 3]` f32 (pre-softplus)
    /// * `color_grads` — per-tet color gradient `[M × 3]` f32
    /// * `tet_count` — number of tetrahedra
    pub fn upload(
        device: &wgpu::Device,
        base_colors: &[f32],
        color_grads: &[f32],
        tet_count: u32,
    ) -> Self {
        let trainable = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        let base_colors_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("base_colors"),
            contents: bytemuck::cast_slice(base_colors),
            usage: trainable,
        });

        let color_grads_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("color_grads"),
            contents: bytemuck::cast_slice(color_grads),
            usage: trainable,
        });

        let colors = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("colors"),
            size: (tet_count as u64) * 3 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        Self {
            base_colors: base_colors_buf,
            color_grads: color_grads_buf,
            colors,
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
    /// Lean HW-only projection compute (no tile counting)
    pub hw_compute_pipeline: wgpu::ComputePipeline,
    pub hw_compute_bind_group_layout: wgpu::BindGroupLayout,
    pub render_pipeline: wgpu::RenderPipeline,
    pub render_bind_group_layout: wgpu::BindGroupLayout,
    /// Quad-based render pipeline (4 verts/tet via triangle strip, reads precomputed buffer)
    pub quad_render_pipeline: wgpu::RenderPipeline,
    /// Bind group layout for quad render (6 bindings: uniforms, precomputed, sorted_indices, colors, densities, color_grads)
    pub quad_render_bg_layout: wgpu::BindGroupLayout,
    /// Compute prepass pipeline (precomputes clip positions + normals for quad path)
    pub prepass_compute_pipeline: wgpu::ComputePipeline,
    pub prepass_bg_layout: wgpu::BindGroupLayout,
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
    /// `color_format`: texture format for all color attachments (Rgba16Float).
    /// Total bytes per sample must not exceed 32 bytes (4 * 8 = 32 for Rgba16Float).
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        // ----- Compute pipeline (14 bindings) -----
        // Bindings 0-5: read-only storage (uniforms, vertices, indices, densities, color_grads, circumdata)
        // Bindings 6-11: read-write storage (colors, sort_keys, sort_values, indirect_args, tiles_touched, compact_tet_ids)
        // Binding 12: read-only storage (base_colors)
        // Binding 13: read-only storage (sh_coeffs, f16-packed)
        let compute_read_only = [
            true, true, true, true, true, true, // 0-5 read-only
            false, false, false, false,          // 6-9 read-write
            false, false,                        // 10-11 read-write (tiles_touched, compact_tet_ids)
            true,                                // 12 read-only (base_colors)
            true,                                // 13 read-only (sh_coeffs)
        ];
        let compute_entries = storage_entries(
            14,
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
            label: Some("project_compute.wgsl"),
            source: wgpu::ShaderSource::Wgsl(PROJECT_COMPUTE_WGSL.into()),
        });
        let compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("project_compute_pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &compute_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // ----- HW projection compute pipeline (11 bindings, no tile work) -----
        // Bindings: uniforms(r), vertices(r), indices(r), circumdata(r),
        //           sort_keys(rw), sort_values(rw), indirect_args(rw),
        //           colors(rw), base_colors(r), color_grads(r), sh_coeffs(r)
        let hw_compute_read_only = [
            true, true, true, true, // 0-3 read-only
            false, false, false,    // 4-6 read-write
            false, true,            // 7-8: colors(rw), base_colors(r)
            true,                   // 9: color_grads(r)
            true,                   // 10: sh_coeffs(r)
        ];
        let hw_compute_entries = storage_entries(
            11,
            wgpu::ShaderStages::COMPUTE,
            &hw_compute_read_only,
        );
        let hw_compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("hw_compute_bind_group_layout"),
                entries: &hw_compute_entries,
            });
        let hw_compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("hw_compute_pipeline_layout"),
                bind_group_layouts: &[&hw_compute_bind_group_layout],
                immediate_size: 0,
            });
        let hw_compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("project_compute_hw.wgsl"),
            source: wgpu::ShaderSource::Wgsl(PROJECT_COMPUTE_HW_WGSL.into()),
        });
        let hw_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("project_compute_hw_pipeline"),
                layout: Some(&hw_compute_pipeline_layout),
                module: &hw_compute_shader,
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
                        // Color attachment 0: premultiplied alpha blend
                        Some(wgpu::ColorTargetState {
                            format: color_format,
                            blend: Some(premul_blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        // Color attachment 1 (aux): no blending
                        Some(wgpu::ColorTargetState {
                            format: color_format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        // Color attachment 2 (normals): premultiplied alpha blend
                        // Uses color_format (Rgba16Float) because Rgba32Float is not blendable
                        Some(wgpu::ColorTargetState {
                            format: color_format,
                            blend: Some(premul_blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        // Color attachment 3 (depth): premultiplied alpha blend
                        Some(wgpu::ColorTargetState {
                            format: color_format,
                            blend: Some(premul_blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                    ],
                    compilation_options: Default::default(),
                }),
                multiview_mask: None,
                cache: None,
            });

        // ----- Prepass compute pipeline (9 bindings) -----
        // Bindings: uniforms(r), vertices(r), indices(r), sorted_indices(r),
        //           indirect_args(r), precomputed(rw), colors(r), densities(r), color_grads(r)
        let prepass_read_only = [
            true, true, true, true, true, // 0-4 read-only
            false,                         // 5 read-write (precomputed)
            true, true, true,              // 6-8 read-only (colors, densities, color_grads)
        ];
        let prepass_entries = storage_entries(
            9,
            wgpu::ShaderStages::COMPUTE,
            &prepass_read_only,
        );
        let prepass_bg_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("prepass_bg_layout"),
                entries: &prepass_entries,
            });
        let prepass_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("prepass_pipeline_layout"),
                bind_group_layouts: &[&prepass_bg_layout],
                immediate_size: 0,
            });
        let prepass_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forward_prepass_compute.wgsl"),
            source: wgpu::ShaderSource::Wgsl(FORWARD_PREPASS_COMPUTE_WGSL.into()),
        });
        let prepass_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("prepass_compute_pipeline"),
                layout: Some(&prepass_pipeline_layout),
                module: &prepass_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // ----- Quad render bind group layout (2 bindings, reads precomputed buffer) -----
        // Bindings: uniforms(r), precomputed(r)
        // Vertex shader reads only precomputed. Fragment shader reads uniforms for intrinsics.
        let quad_render_read_only = [true; 2];
        let quad_render_entries = storage_entries(
            2,
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            &quad_render_read_only,
        );
        let quad_render_bg_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("quad_render_bg_layout"),
                entries: &quad_render_entries,
            });
        let quad_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("quad_render_pipeline_layout"),
                bind_group_layouts: &[&quad_render_bg_layout],
                immediate_size: 0,
            });

        // Quad-based render pipeline (triangle strip, 4 verts/tet)
        // Shares forward_fragment.wgsl — vertex output matches exactly.
        let quad_vertex_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forward_vertex_quad.wgsl"),
            source: wgpu::ShaderSource::Wgsl(FORWARD_VERTEX_QUAD_WGSL.into()),
        });
        let quad_render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("forward_quad_render_pipeline"),
                layout: Some(&quad_render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &quad_vertex_shader,
                    entry_point: Some("main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None, // No face culling — quad is screen-aligned
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
                        Some(wgpu::ColorTargetState {
                            format: color_format,
                            blend: Some(premul_blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: color_format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: color_format,
                            blend: Some(premul_blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: color_format,
                            blend: Some(premul_blend),
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
            hw_compute_pipeline,
            hw_compute_bind_group_layout,
            render_pipeline,
            render_bind_group_layout,
            quad_render_pipeline,
            quad_render_bg_layout,
            prepass_compute_pipeline,
            prepass_bg_layout,
        }
    }
}

// ---------------------------------------------------------------------------
// Mesh Shader Pipelines (optional, requires EXPERIMENTAL_MESH_SHADER)
// ---------------------------------------------------------------------------

/// Compiled pipelines for the mesh shader forward pass.
pub struct MeshForwardPipelines {
    pub mesh_render_pipeline: wgpu::RenderPipeline,
    pub mesh_render_bg_layout: wgpu::BindGroupLayout,
    pub indirect_convert_pipeline: wgpu::ComputePipeline,
    pub indirect_convert_bg_layout: wgpu::BindGroupLayout,
}

impl MeshForwardPipelines {
    /// Create mesh shader pipelines. Requires `Features::EXPERIMENTAL_MESH_SHADER`.
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        // ----- Mesh render pipeline (8 read-only storage bindings) -----
        // Bindings 0-6: same as vertex shader render bind group
        // Binding 7: indirect_args (read visible_count)
        let mesh_read_only = [true; 8];
        let mesh_entries = storage_entries(
            8,
            wgpu::ShaderStages::MESH,
            &mesh_read_only,
        );
        let mesh_render_bg_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mesh_render_bg_layout"),
                entries: &mesh_entries,
            });
        let mesh_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("mesh_pipeline_layout"),
                bind_group_layouts: &[&mesh_render_bg_layout],
                immediate_size: 0,
            });
        let mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forward_mesh.wgsl"),
            source: wgpu::ShaderSource::Wgsl(FORWARD_MESH_WGSL.into()),
        });
        let fragment_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forward_fragment.wgsl"),
            source: wgpu::ShaderSource::Wgsl(FORWARD_FRAGMENT_WGSL.into()),
        });

        // Same blend state as ForwardPipelines
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

        let mesh_render_pipeline =
            device.create_mesh_pipeline(&wgpu::MeshPipelineDescriptor {
                label: Some("mesh_forward_render_pipeline"),
                layout: Some(&mesh_pipeline_layout),
                task: None,
                mesh: wgpu::MeshState {
                    module: &mesh_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
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
                        Some(wgpu::ColorTargetState {
                            format: color_format,
                            blend: Some(premul_blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: color_format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: color_format,
                            blend: Some(premul_blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: color_format,
                            blend: Some(premul_blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                    ],
                    compilation_options: Default::default(),
                }),
                multiview: None,
                cache: None,
            });

        // ----- Indirect convert compute pipeline (2 bindings) -----
        let indirect_convert_bg_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("indirect_convert_bg_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
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
                ],
            });
        let indirect_convert_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("indirect_convert_pipeline_layout"),
                bind_group_layouts: &[&indirect_convert_bg_layout],
                immediate_size: 0,
            });
        let indirect_convert_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("indirect_convert.wgsl"),
                source: wgpu::ShaderSource::Wgsl(INDIRECT_CONVERT_WGSL.into()),
            });
        let indirect_convert_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("indirect_convert_pipeline"),
                layout: Some(&indirect_convert_layout),
                module: &indirect_convert_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            mesh_render_pipeline,
            mesh_render_bg_layout,
            indirect_convert_pipeline,
            indirect_convert_bg_layout,
        }
    }
}

// ---------------------------------------------------------------------------
// Interval Shading Pipelines (requires EXPERIMENTAL_MESH_SHADER)
// ---------------------------------------------------------------------------

/// Compiled pipelines for the interval shading forward pass.
///
/// Decomposes each tet into non-overlapping screen-space triangles with
/// interpolated front/back NDC depths. Single color output (no MRT).
pub struct IntervalPipelines {
    pub mesh_render_pipeline: wgpu::RenderPipeline,
    pub mesh_render_bg_layout: wgpu::BindGroupLayout,
    pub indirect_convert_pipeline: wgpu::ComputePipeline,
    pub indirect_convert_bg_layout: wgpu::BindGroupLayout,
}

impl IntervalPipelines {
    /// Create interval shading pipelines. Requires `Features::EXPERIMENTAL_MESH_SHADER`.
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        // ----- Mesh render pipeline (8 read-only storage bindings) -----
        // Same bindings as MeshForwardPipelines: uniforms, vertices, indices, colors,
        // densities, color_grads, sorted_indices, indirect_args
        let mesh_read_only = [true; 8];
        let mesh_entries = storage_entries(
            8,
            wgpu::ShaderStages::MESH | wgpu::ShaderStages::FRAGMENT,
            &mesh_read_only,
        );
        let mesh_render_bg_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("interval_mesh_render_bg_layout"),
                entries: &mesh_entries,
            });
        let mesh_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("interval_mesh_pipeline_layout"),
                bind_group_layouts: &[&mesh_render_bg_layout],
                immediate_size: 0,
            });
        let mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("interval_mesh.wgsl"),
            source: wgpu::ShaderSource::Wgsl(INTERVAL_MESH_WGSL.into()),
        });
        let fragment_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("interval_fragment.wgsl"),
            source: wgpu::ShaderSource::Wgsl(INTERVAL_FRAGMENT_WGSL.into()),
        });

        // Premultiplied alpha blend for single color attachment
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

        let mesh_render_pipeline =
            device.create_mesh_pipeline(&wgpu::MeshPipelineDescriptor {
                label: Some("interval_mesh_render_pipeline"),
                layout: Some(&mesh_pipeline_layout),
                task: None,
                mesh: wgpu::MeshState {
                    module: &mesh_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None, // No face culling — interval triangles are screen-aligned
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
                        // Single color attachment: premultiplied alpha blend
                        Some(wgpu::ColorTargetState {
                            format: color_format,
                            blend: Some(premul_blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                    ],
                    compilation_options: Default::default(),
                }),
                multiview: None,
                cache: None,
            });

        // ----- Indirect convert compute pipeline (2 bindings) -----
        // Reuses indirect_convert.wgsl with TETS_PER_GROUP overridden to 16
        let indirect_convert_bg_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("interval_indirect_convert_bg_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
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
                ],
            });
        let indirect_convert_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("interval_indirect_convert_pipeline_layout"),
                bind_group_layouts: &[&indirect_convert_bg_layout],
                immediate_size: 0,
            });
        let indirect_convert_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("interval_indirect_convert.wgsl"),
                source: wgpu::ShaderSource::Wgsl(INDIRECT_CONVERT_WGSL.into()),
            });
        // Override TETS_PER_GROUP to 16 for interval path
        let indirect_convert_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("interval_indirect_convert_pipeline"),
                layout: Some(&indirect_convert_layout),
                module: &indirect_convert_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("TETS_PER_GROUP", 16.0)],
                    ..Default::default()
                },
                cache: None,
            });

        Self {
            mesh_render_pipeline,
            mesh_render_bg_layout,
            indirect_convert_pipeline,
            indirect_convert_bg_layout,
        }
    }
}

// ---------------------------------------------------------------------------
// Compute-Based Interval Shading Pipelines (no mesh shader required)
// ---------------------------------------------------------------------------

/// Compiled pipelines for compute-based interval shading.
///
/// Replaces mesh shader with compute → vertex/fragment draw, making interval
/// shading available on all GPUs.
pub struct ComputeIntervalPipelines {
    pub gen_pipeline: wgpu::ComputePipeline,
    pub gen_bg_layout: wgpu::BindGroupLayout,
    pub render_pipeline: wgpu::RenderPipeline,
    pub render_bg_layout: wgpu::BindGroupLayout,
    pub indirect_convert_pipeline: wgpu::ComputePipeline,
    pub indirect_convert_bg_layout: wgpu::BindGroupLayout,
}

impl ComputeIntervalPipelines {
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        // ----- Gen compute pipeline (10 bindings) -----
        // Bindings 0-7: read-only, 8-9: read-write
        let gen_read_only = [
            true, true, true, true, true, true, true, true, // 0-7 read-only
            false, false, // 8-9 read-write (out_vertices, out_tet_data)
        ];
        let gen_entries = storage_entries(10, wgpu::ShaderStages::COMPUTE, &gen_read_only);
        let gen_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute_interval_gen_bg_layout"),
            entries: &gen_entries,
        });
        let gen_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compute_interval_gen_pipeline_layout"),
            bind_group_layouts: &[&gen_bg_layout],
            immediate_size: 0,
        });
        let gen_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("interval_compute.wgsl"),
            source: wgpu::ShaderSource::Wgsl(INTERVAL_COMPUTE_WGSL.into()),
        });
        let gen_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute_interval_gen_pipeline"),
            layout: Some(&gen_pipeline_layout),
            module: &gen_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ----- Render pipeline (3 read-only storage bindings) -----
        // 0: uniforms, 1: interval_vertex_buf, 2: interval_tet_data_buf
        let render_read_only = [true; 3];
        let render_entries = storage_entries(
            3,
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            &render_read_only,
        );
        let render_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute_interval_render_bg_layout"),
            entries: &render_entries,
        });
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compute_interval_render_pipeline_layout"),
            bind_group_layouts: &[&render_bg_layout],
            immediate_size: 0,
        });

        let vertex_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("interval_vertex.wgsl"),
            source: wgpu::ShaderSource::Wgsl(INTERVAL_VERTEX_WGSL.into()),
        });
        let fragment_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("interval_fragment.wgsl"),
            source: wgpu::ShaderSource::Wgsl(INTERVAL_FRAGMENT_WGSL.into()),
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
            label: Some("compute_interval_render_pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_shader,
                entry_point: Some("main"),
                buffers: &[], // No vertex buffers — reads from storage
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // No face culling — interval triangles are screen-aligned
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
                    Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(premul_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        // ----- Indirect convert compute pipeline (2 bindings) -----
        let indirect_convert_bg_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("compute_interval_indirect_convert_bg_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
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
                ],
            });
        let indirect_convert_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute_interval_indirect_convert_pipeline_layout"),
                bind_group_layouts: &[&indirect_convert_bg_layout],
                immediate_size: 0,
            });
        let indirect_convert_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("interval_indirect_convert.wgsl"),
                source: wgpu::ShaderSource::Wgsl(INTERVAL_INDIRECT_CONVERT_WGSL.into()),
            });
        let indirect_convert_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("compute_interval_indirect_convert_pipeline"),
                layout: Some(&indirect_convert_layout),
                module: &indirect_convert_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            gen_pipeline,
            gen_bg_layout,
            render_pipeline,
            render_bg_layout,
            indirect_convert_pipeline,
            indirect_convert_bg_layout,
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
    /// Auxiliary output texture (Rgba16Float)
    pub aux0_texture: wgpu::Texture,
    /// Normals output texture (Rgba16Float, premultiplied alpha blend)
    pub normals_texture: wgpu::Texture,
    /// Depth output texture (Rgba16Float, premultiplied alpha blend)
    pub depth_texture: wgpu::Texture,
    /// View into color texture
    pub color_view: wgpu::TextureView,
    /// View into aux texture
    pub aux0_view: wgpu::TextureView,
    /// View into normals texture
    pub normals_view: wgpu::TextureView,
    /// View into depth texture
    pub depth_view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
}

impl RenderTargets {
    /// Create render target textures at the given resolution.
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let color_format = wgpu::TextureFormat::Rgba16Float;

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
            format: color_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        // Normals and depth use Rgba16Float (blendable) instead of Rgba32Float
        let normals_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("normals_target"),
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

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth_target"),
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

        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let aux0_view = aux0_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let normals_view = normals_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            color_texture,
            aux0_texture,
            normals_texture,
            depth_texture,
            color_view,
            aux0_view,
            normals_view,
            depth_view,
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

/// Create the compute bind group (14 bindings).
///
/// Binding order matches `project_compute.wgsl`:
///   0: uniforms, 1: vertices, 2: indices, 3: densities,
///   4: color_grads, 5: circumdata,
///   6: colors, 7: sort_keys, 8: sort_values, 9: indirect_args,
///   10: tiles_touched, 11: compact_tet_ids, 12: base_colors,
///   13: sh_coeffs (f16-packed)
pub fn create_compute_bind_group(
    device: &wgpu::Device,
    pipelines: &ForwardPipelines,
    buffers: &SceneBuffers,
    material: &MaterialBuffers,
    sh_coeffs_buf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute_bind_group"),
        layout: &pipelines.compute_bind_group_layout,
        entries: &[
            buf_entry(0, &buffers.uniforms),
            buf_entry(1, &buffers.vertices),
            buf_entry(2, &buffers.indices),
            buf_entry(3, &buffers.densities),
            buf_entry(4, &material.color_grads),
            buf_entry(5, &buffers.circumdata),
            buf_entry(6, &material.colors),
            buf_entry(7, &buffers.sort_keys),
            buf_entry(8, &buffers.sort_values),
            buf_entry(9, &buffers.indirect_args),
            buf_entry(10, &buffers.tiles_touched),
            buf_entry(11, &buffers.compact_tet_ids),
            buf_entry(12, &material.base_colors),
            buf_entry(13, sh_coeffs_buf),
        ],
    })
}

/// Create the HW projection compute bind group (11 bindings, no tile data).
///
/// Binding order matches `project_compute_hw.wgsl`:
///   0: uniforms, 1: vertices, 2: indices, 3: circumdata,
///   4: sort_keys, 5: sort_values, 6: indirect_args,
///   7: colors, 8: base_colors, 9: color_grads, 10: sh_coeffs
pub fn create_hw_compute_bind_group(
    device: &wgpu::Device,
    pipelines: &ForwardPipelines,
    buffers: &SceneBuffers,
    material: &MaterialBuffers,
    sh_coeffs_buf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hw_compute_bind_group"),
        layout: &pipelines.hw_compute_bind_group_layout,
        entries: &[
            buf_entry(0, &buffers.uniforms),
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

/// Create the render bind group (7 bindings).
///
/// Binding order matches `forward_vertex.wgsl`:
///   0: uniforms, 1: vertices, 2: indices, 3: colors,
///   4: densities, 5: color_grads, 6: sorted_indices (= sort_values)
pub fn create_render_bind_group(
    device: &wgpu::Device,
    pipelines: &ForwardPipelines,
    buffers: &SceneBuffers,
    material: &MaterialBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("render_bind_group"),
        layout: &pipelines.render_bind_group_layout,
        entries: &[
            buf_entry(0, &buffers.uniforms),
            buf_entry(1, &buffers.vertices),
            buf_entry(2, &buffers.indices),
            buf_entry(3, &material.colors),
            buf_entry(4, &buffers.densities),
            buf_entry(5, &material.color_grads),
            buf_entry(6, &buffers.sort_values), // sorted indices
        ],
    })
}

/// Create a render bind group with an explicit sort_values buffer.
///
/// Same as [`create_render_bind_group`] but binding 6 uses a caller-provided
/// `sort_values` buffer instead of `buffers.sort_values`. This is needed when
/// the radix sort result ends up in the alternate (B) buffer.
pub fn create_render_bind_group_with_sort_values(
    device: &wgpu::Device,
    pipelines: &ForwardPipelines,
    buffers: &SceneBuffers,
    material: &MaterialBuffers,
    sort_values: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("render_bind_group_sort_b"),
        layout: &pipelines.render_bind_group_layout,
        entries: &[
            buf_entry(0, &buffers.uniforms),
            buf_entry(1, &buffers.vertices),
            buf_entry(2, &buffers.indices),
            buf_entry(3, &material.colors),
            buf_entry(4, &buffers.densities),
            buf_entry(5, &material.color_grads),
            buf_entry(6, sort_values),
        ],
    })
}

/// Create a prepass compute bind group (9 bindings).
///
/// Binding order matches `forward_prepass_compute.wgsl`:
///   0: uniforms, 1: vertices, 2: indices, 3: sorted_indices,
///   4: indirect_args, 5: precomputed, 6: colors, 7: densities, 8: color_grads
pub fn create_prepass_bind_group(
    device: &wgpu::Device,
    pipelines: &ForwardPipelines,
    buffers: &SceneBuffers,
    material: &MaterialBuffers,
    sorted_indices: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("prepass_bind_group"),
        layout: &pipelines.prepass_bg_layout,
        entries: &[
            buf_entry(0, &buffers.uniforms),
            buf_entry(1, &buffers.vertices),
            buf_entry(2, &buffers.indices),
            buf_entry(3, sorted_indices),
            buf_entry(4, &buffers.indirect_args),
            buf_entry(5, &buffers.precomputed),
            buf_entry(6, &material.colors),
            buf_entry(7, &buffers.densities),
            buf_entry(8, &material.color_grads),
        ],
    })
}

/// Create a quad render bind group (2 bindings).
///
/// Binding order matches `forward_vertex_quad.wgsl`:
///   0: uniforms, 1: precomputed
pub fn create_quad_render_bind_group(
    device: &wgpu::Device,
    pipelines: &ForwardPipelines,
    buffers: &SceneBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("quad_render_bind_group"),
        layout: &pipelines.quad_render_bg_layout,
        entries: &[
            buf_entry(0, &buffers.uniforms),
            buf_entry(1, &buffers.precomputed),
        ],
    })
}

/// Create the mesh shader render bind group (8 bindings).
///
/// Binding order matches `forward_mesh.wgsl`:
///   0: uniforms, 1: vertices, 2: indices, 3: colors,
///   4: densities, 5: color_grads, 6: sorted_indices, 7: indirect_args
pub fn create_mesh_render_bind_group(
    device: &wgpu::Device,
    mesh_pipelines: &MeshForwardPipelines,
    buffers: &SceneBuffers,
    material: &MaterialBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mesh_render_bind_group"),
        layout: &mesh_pipelines.mesh_render_bg_layout,
        entries: &[
            buf_entry(0, &buffers.uniforms),
            buf_entry(1, &buffers.vertices),
            buf_entry(2, &buffers.indices),
            buf_entry(3, &material.colors),
            buf_entry(4, &buffers.densities),
            buf_entry(5, &material.color_grads),
            buf_entry(6, &buffers.sort_values),  // sorted indices
            buf_entry(7, &buffers.indirect_args),
        ],
    })
}

/// Create a mesh render bind group with an explicit sort_values buffer.
///
/// Same as [`create_mesh_render_bind_group`] but binding 6 uses a caller-provided
/// `sort_values` buffer (the radix sort alternate B buffer).
pub fn create_mesh_render_bind_group_with_sort_values(
    device: &wgpu::Device,
    mesh_pipelines: &MeshForwardPipelines,
    buffers: &SceneBuffers,
    material: &MaterialBuffers,
    sort_values: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mesh_render_bind_group_sort_b"),
        layout: &mesh_pipelines.mesh_render_bg_layout,
        entries: &[
            buf_entry(0, &buffers.uniforms),
            buf_entry(1, &buffers.vertices),
            buf_entry(2, &buffers.indices),
            buf_entry(3, &material.colors),
            buf_entry(4, &buffers.densities),
            buf_entry(5, &material.color_grads),
            buf_entry(6, sort_values),
            buf_entry(7, &buffers.indirect_args),
        ],
    })
}

/// Create the indirect convert bind group (2 bindings).
pub fn create_indirect_convert_bind_group(
    device: &wgpu::Device,
    mesh_pipelines: &MeshForwardPipelines,
    buffers: &SceneBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("indirect_convert_bind_group"),
        layout: &mesh_pipelines.indirect_convert_bg_layout,
        entries: &[
            buf_entry(0, &buffers.indirect_args),
            buf_entry(1, &buffers.mesh_indirect_args),
        ],
    })
}

/// Create the interval mesh shader render bind group (8 bindings).
///
/// Binding order matches `interval_mesh.wgsl`:
///   0: uniforms, 1: vertices, 2: indices, 3: colors,
///   4: densities, 5: color_grads, 6: sorted_indices, 7: indirect_args
pub fn create_interval_render_bind_group(
    device: &wgpu::Device,
    interval_pipelines: &IntervalPipelines,
    buffers: &SceneBuffers,
    material: &MaterialBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("interval_render_bind_group"),
        layout: &interval_pipelines.mesh_render_bg_layout,
        entries: &[
            buf_entry(0, &buffers.uniforms),
            buf_entry(1, &buffers.vertices),
            buf_entry(2, &buffers.indices),
            buf_entry(3, &material.colors),
            buf_entry(4, &buffers.densities),
            buf_entry(5, &material.color_grads),
            buf_entry(6, &buffers.sort_values),  // sorted indices
            buf_entry(7, &buffers.indirect_args),
        ],
    })
}

/// Create an interval render bind group with an explicit sort_values buffer.
pub fn create_interval_render_bind_group_with_sort_values(
    device: &wgpu::Device,
    interval_pipelines: &IntervalPipelines,
    buffers: &SceneBuffers,
    material: &MaterialBuffers,
    sort_values: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("interval_render_bind_group_sort_b"),
        layout: &interval_pipelines.mesh_render_bg_layout,
        entries: &[
            buf_entry(0, &buffers.uniforms),
            buf_entry(1, &buffers.vertices),
            buf_entry(2, &buffers.indices),
            buf_entry(3, &material.colors),
            buf_entry(4, &buffers.densities),
            buf_entry(5, &material.color_grads),
            buf_entry(6, sort_values),
            buf_entry(7, &buffers.indirect_args),
        ],
    })
}

/// Create the interval indirect convert bind group (2 bindings).
pub fn create_interval_indirect_convert_bind_group(
    device: &wgpu::Device,
    interval_pipelines: &IntervalPipelines,
    buffers: &SceneBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("interval_indirect_convert_bind_group"),
        layout: &interval_pipelines.indirect_convert_bg_layout,
        entries: &[
            buf_entry(0, &buffers.indirect_args),
            buf_entry(1, &buffers.mesh_indirect_args),
        ],
    })
}

/// Create the compute-interval gen bind group (10 bindings).
///
/// Binding order matches `interval_compute.wgsl`:
///   0: uniforms, 1: vertices, 2: indices, 3: colors, 4: densities,
///   5: color_grads, 6: sorted_indices (sort_values), 7: indirect_args,
///   8: interval_vertex_buf, 9: interval_tet_data_buf
pub fn create_compute_interval_gen_bind_group(
    device: &wgpu::Device,
    pipelines: &ComputeIntervalPipelines,
    buffers: &SceneBuffers,
    material: &MaterialBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute_interval_gen_bg"),
        layout: &pipelines.gen_bg_layout,
        entries: &[
            buf_entry(0, &buffers.uniforms),
            buf_entry(1, &buffers.vertices),
            buf_entry(2, &buffers.indices),
            buf_entry(3, &material.colors),
            buf_entry(4, &buffers.densities),
            buf_entry(5, &material.color_grads),
            buf_entry(6, &buffers.sort_values), // sorted indices
            buf_entry(7, &buffers.indirect_args),
            buf_entry(8, &buffers.interval_vertex_buf),
            buf_entry(9, &buffers.interval_tet_data_buf),
        ],
    })
}

/// Create a compute-interval gen bind group with an explicit sort_values buffer (B swap).
pub fn create_compute_interval_gen_bind_group_with_sort_values(
    device: &wgpu::Device,
    pipelines: &ComputeIntervalPipelines,
    buffers: &SceneBuffers,
    material: &MaterialBuffers,
    sort_values: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute_interval_gen_bg_sort_b"),
        layout: &pipelines.gen_bg_layout,
        entries: &[
            buf_entry(0, &buffers.uniforms),
            buf_entry(1, &buffers.vertices),
            buf_entry(2, &buffers.indices),
            buf_entry(3, &material.colors),
            buf_entry(4, &buffers.densities),
            buf_entry(5, &material.color_grads),
            buf_entry(6, sort_values),
            buf_entry(7, &buffers.indirect_args),
            buf_entry(8, &buffers.interval_vertex_buf),
            buf_entry(9, &buffers.interval_tet_data_buf),
        ],
    })
}

/// Create the compute-interval render bind group (3 bindings).
///
/// Binding order matches `interval_vertex.wgsl`:
///   0: uniforms, 1: interval_vertex_buf, 2: interval_tet_data_buf
pub fn create_compute_interval_render_bind_group(
    device: &wgpu::Device,
    pipelines: &ComputeIntervalPipelines,
    buffers: &SceneBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute_interval_render_bg"),
        layout: &pipelines.render_bg_layout,
        entries: &[
            buf_entry(0, &buffers.uniforms),
            buf_entry(1, &buffers.interval_vertex_buf),
            buf_entry(2, &buffers.interval_tet_data_buf),
        ],
    })
}

/// Create the compute-interval indirect convert bind group (2 bindings).
pub fn create_compute_interval_indirect_convert_bind_group(
    device: &wgpu::Device,
    pipelines: &ComputeIntervalPipelines,
    buffers: &SceneBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute_interval_indirect_convert_bg"),
        layout: &pipelines.indirect_convert_bg_layout,
        entries: &[
            buf_entry(0, &buffers.indirect_args),
            buf_entry(1, &buffers.interval_args_buf),
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
pub fn record_project_compute(
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
            label: Some("project_compute"),
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
pub fn record_project_compute_and_sort(
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
            label: Some("project_compute"),
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
    depth_view: &wgpu::TextureView,
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
            label: Some("project_compute"),
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
                // Attachment 0: main color (premultiplied alpha blend, load to preserve primitives)
                Some(wgpu::RenderPassColorAttachment {
                    view: &targets.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
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
                // Attachment 2: normals (premultiplied alpha blend)
                Some(wgpu::RenderPassColorAttachment {
                    view: &targets.normals_view,
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
                // Attachment 3: depth (premultiplied alpha blend)
                Some(wgpu::RenderPassColorAttachment {
                    view: &targets.depth_view,
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

/// Record a sorted forward pass: project_compute → radix sort → render.
///
/// Unlike [`record_forward_pass`], this inserts a radix sort between the
/// compute and render passes so that tets are drawn back-to-front.
/// The sort uses ascending order on `~depth_bits` keys written by
/// `project_compute`, which gives correct back-to-front compositing with
/// the existing premultiplied alpha blend state (src=One, dst=OneMinusSrcAlpha).
///
/// The caller must provide two render bind groups:
/// - `render_bg_a`: uses `buffers.sort_values` (primary A buffer)
/// - `render_bg_b`: uses `sort_state.values_b` (alternate B buffer)
///
/// The function selects the correct bind group based on which buffer
/// holds the sorted result (depends on number of radix sort passes).
pub fn record_sorted_forward_pass(
    encoder: &mut wgpu::CommandEncoder,
    device: &wgpu::Device,
    pipelines: &ForwardPipelines,
    sort_pipelines: &rmesh_sort::RadixSortPipelines,
    sort_state: &rmesh_sort::RadixSortState,
    buffers: &SceneBuffers,
    targets: &RenderTargets,
    compute_bg: &wgpu::BindGroup,
    render_bg_a: &wgpu::BindGroup,
    render_bg_b: &wgpu::BindGroup,
    tet_count: u32,
    queue: &wgpu::Queue,
    depth_view: &wgpu::TextureView,
    hw_compute_bg: Option<&wgpu::BindGroup>,
    use_quad: bool,
    prepass_bg_a: Option<&wgpu::BindGroup>,
    prepass_bg_b: Option<&wgpu::BindGroup>,
    quad_render_bg: Option<&wgpu::BindGroup>,
    profiler: Option<&wgpu::QuerySet>,
) {
    // ----- 1. Reset indirect args -----
    let reset_cmd = DrawIndirectCommand {
        vertex_count: if use_quad { 4 } else { 12 },
        instance_count: 0,
        first_vertex: 0,
        first_instance: 0,
    };
    queue.write_buffer(&buffers.indirect_args, 0, bytemuck::bytes_of(&reset_cmd));

    // Write sort element count to radix sort's num_keys_buf.
    // project_compute dispatches n_pow2 threads; padding threads write
    // sort_keys = 0xFFFFFFFF which sorts to the end.
    let n_pow2 = tet_count.next_power_of_two();
    queue.write_buffer(sort_state.num_keys_buf(), 0, bytemuck::bytes_of(&n_pow2));

    // ----- 2. Compute pass -----
    // Use lean HW projection shader when available (no tile counting work)
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("project_compute"),
            timestamp_writes: profiler.map(|qs| wgpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }),
        });
        if let Some(hw_bg) = hw_compute_bg {
            cpass.set_pipeline(&pipelines.hw_compute_pipeline);
            cpass.set_bind_group(0, hw_bg, &[]);
        } else {
            cpass.set_pipeline(&pipelines.compute_pipeline);
            cpass.set_bind_group(0, compute_bg, &[]);
        }

        let workgroup_size = 64u32;
        let total_workgroups = (n_pow2 + workgroup_size - 1) / workgroup_size;
        let max_per_dim = 65535u32;
        let dispatch_x = total_workgroups.min(max_per_dim);
        let dispatch_y = (total_workgroups + max_per_dim - 1) / max_per_dim;
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    // ----- 3. Radix sort (back-to-front via ascending ~depth_bits) -----
    let result_in_b = rmesh_sort::record_radix_sort(
        encoder, device, sort_pipelines, sort_state,
        &buffers.sort_keys, &buffers.sort_values,
    );

    // ----- 3.5. Compute prepass (quad path only) -----
    if use_quad {
        let prepass_bg = if result_in_b {
            prepass_bg_b.expect("prepass_bg_b required for quad path")
        } else {
            prepass_bg_a.expect("prepass_bg_a required for quad path")
        };
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("forward_prepass"),
            timestamp_writes: profiler.map(|qs| wgpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(2),
                end_of_pass_write_index: Some(3),
            }),
        });
        cpass.set_pipeline(&pipelines.prepass_compute_pipeline);
        cpass.set_bind_group(0, prepass_bg, &[]);
        // Dispatch enough threads: tet_count is upper bound on visible tets
        let workgroup_size = 64u32;
        let total_workgroups = (tet_count + workgroup_size - 1) / workgroup_size;
        let max_per_dim = 65535u32;
        let dispatch_x = total_workgroups.min(max_per_dim);
        let dispatch_y = (total_workgroups + max_per_dim - 1) / max_per_dim;
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    // ----- 4. Render pass -----
    let render_bg = if use_quad {
        quad_render_bg.expect("quad_render_bg required for quad path")
    } else if result_in_b {
        render_bg_b
    } else {
        render_bg_a
    };
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("forward_render"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &targets.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
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
                // Attachment 2: normals (premultiplied alpha blend)
                Some(wgpu::RenderPassColorAttachment {
                    view: &targets.normals_view,
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
                // Attachment 3: depth (premultiplied alpha blend)
                Some(wgpu::RenderPassColorAttachment {
                    view: &targets.depth_view,
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
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: profiler.map(|qs| wgpu::RenderPassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(4),
                end_of_pass_write_index: Some(5),
            }),
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
        if use_quad {
            rpass.set_pipeline(&pipelines.quad_render_pipeline);
        } else {
            rpass.set_pipeline(&pipelines.render_pipeline);
        }
        rpass.set_bind_group(0, render_bg, &[]);

        rpass.draw_indirect(&buffers.indirect_args, 0);
    }
}

/// Record a sorted forward pass using mesh shaders instead of hardware vertex rasterization.
///
/// Steps 1-3 are identical to [`record_sorted_forward_pass`]:
///   1. Reset indirect args
///   2. Compute pass (SH eval + cull + depth keys)
///   3. Radix sort
///   3.5. Indirect convert: turns `indirect_args.instance_count` into mesh dispatch args
///   4. Render pass with `draw_mesh_tasks_indirect`
pub fn record_sorted_mesh_forward_pass(
    encoder: &mut wgpu::CommandEncoder,
    device: &wgpu::Device,
    fwd_pipelines: &ForwardPipelines,
    mesh_pipelines: &MeshForwardPipelines,
    sort_pipelines: &rmesh_sort::RadixSortPipelines,
    sort_state: &rmesh_sort::RadixSortState,
    buffers: &SceneBuffers,
    targets: &RenderTargets,
    compute_bg: &wgpu::BindGroup,
    mesh_render_bg_a: &wgpu::BindGroup,
    mesh_render_bg_b: &wgpu::BindGroup,
    indirect_convert_bg: &wgpu::BindGroup,
    tet_count: u32,
    queue: &wgpu::Queue,
    depth_view: &wgpu::TextureView,
    hw_compute_bg: Option<&wgpu::BindGroup>,
    profiler: Option<&wgpu::QuerySet>,
) {
    // ----- 1. Reset indirect args -----
    let reset_cmd = DrawIndirectCommand {
        vertex_count: 12,
        instance_count: 0,
        first_vertex: 0,
        first_instance: 0,
    };
    queue.write_buffer(&buffers.indirect_args, 0, bytemuck::bytes_of(&reset_cmd));
    // Reset mesh dispatch args to safe values (0 workgroups)
    queue.write_buffer(
        &buffers.mesh_indirect_args,
        0,
        bytemuck::cast_slice(&[0u32, 1u32, 1u32]),
    );

    let n_pow2 = tet_count.next_power_of_two();
    queue.write_buffer(sort_state.num_keys_buf(), 0, bytemuck::bytes_of(&n_pow2));

    // ----- 2. Compute pass -----
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("project_compute"),
            timestamp_writes: profiler.map(|qs| wgpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }),
        });
        if let Some(hw_bg) = hw_compute_bg {
            cpass.set_pipeline(&fwd_pipelines.hw_compute_pipeline);
            cpass.set_bind_group(0, hw_bg, &[]);
        } else {
            cpass.set_pipeline(&fwd_pipelines.compute_pipeline);
            cpass.set_bind_group(0, compute_bg, &[]);
        }

        let workgroup_size = 64u32;
        let total_workgroups = (n_pow2 + workgroup_size - 1) / workgroup_size;
        let max_per_dim = 65535u32;
        let dispatch_x = total_workgroups.min(max_per_dim);
        let dispatch_y = (total_workgroups + max_per_dim - 1) / max_per_dim;
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    // ----- 3. Radix sort -----
    let result_in_b = rmesh_sort::record_radix_sort(
        encoder, device, sort_pipelines, sort_state,
        &buffers.sort_keys, &buffers.sort_values,
    );

    // ----- 3.5. Indirect convert: instance_count → mesh dispatch args -----
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("indirect_convert"),
            timestamp_writes: profiler.map(|qs| wgpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(2),
                end_of_pass_write_index: Some(3),
            }),
        });
        cpass.set_pipeline(&mesh_pipelines.indirect_convert_pipeline);
        cpass.set_bind_group(0, indirect_convert_bg, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }

    // ----- 4. Mesh shader render pass -----
    let mesh_bg = if result_in_b { mesh_render_bg_b } else { mesh_render_bg_a };
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mesh_forward_render"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &targets.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &targets.aux0_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0, g: 0.0, b: 0.0, a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &targets.normals_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0, g: 0.0, b: 0.0, a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &targets.depth_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0, g: 0.0, b: 0.0, a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
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
            timestamp_writes: profiler.map(|qs| wgpu::RenderPassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(4),
                end_of_pass_write_index: Some(5),
            }),
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
        rpass.set_pipeline(&mesh_pipelines.mesh_render_pipeline);
        rpass.set_bind_group(0, mesh_bg, &[]);

        rpass.draw_mesh_tasks_indirect(&buffers.mesh_indirect_args, 0);
    }
}

/// Record a sorted forward pass using the interval shading path.
///
/// Steps 1-3 are identical to [`record_sorted_mesh_forward_pass`]:
///   1. Reset indirect args
///   2. Compute pass (SH eval + cull + depth keys)
///   3. Radix sort
///   3.5. Indirect convert (TETS_PER_GROUP=16)
///   4. Render pass with interval mesh shader + single color output
pub fn record_sorted_interval_forward_pass(
    encoder: &mut wgpu::CommandEncoder,
    device: &wgpu::Device,
    fwd_pipelines: &ForwardPipelines,
    interval_pipelines: &IntervalPipelines,
    sort_pipelines: &rmesh_sort::RadixSortPipelines,
    sort_state: &rmesh_sort::RadixSortState,
    buffers: &SceneBuffers,
    targets: &RenderTargets,
    compute_bg: &wgpu::BindGroup,
    interval_render_bg_a: &wgpu::BindGroup,
    interval_render_bg_b: &wgpu::BindGroup,
    indirect_convert_bg: &wgpu::BindGroup,
    tet_count: u32,
    queue: &wgpu::Queue,
    depth_view: &wgpu::TextureView,
    hw_compute_bg: Option<&wgpu::BindGroup>,
    profiler: Option<&wgpu::QuerySet>,
) {
    // ----- 1. Reset indirect args -----
    let reset_cmd = DrawIndirectCommand {
        vertex_count: 12,
        instance_count: 0,
        first_vertex: 0,
        first_instance: 0,
    };
    queue.write_buffer(&buffers.indirect_args, 0, bytemuck::bytes_of(&reset_cmd));
    queue.write_buffer(
        &buffers.mesh_indirect_args,
        0,
        bytemuck::cast_slice(&[0u32, 1u32, 1u32]),
    );

    let n_pow2 = tet_count.next_power_of_two();
    queue.write_buffer(sort_state.num_keys_buf(), 0, bytemuck::bytes_of(&n_pow2));

    // ----- 2. Compute pass -----
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("project_compute"),
            timestamp_writes: profiler.map(|qs| wgpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }),
        });
        if let Some(hw_bg) = hw_compute_bg {
            cpass.set_pipeline(&fwd_pipelines.hw_compute_pipeline);
            cpass.set_bind_group(0, hw_bg, &[]);
        } else {
            cpass.set_pipeline(&fwd_pipelines.compute_pipeline);
            cpass.set_bind_group(0, compute_bg, &[]);
        }

        let workgroup_size = 64u32;
        let total_workgroups = (n_pow2 + workgroup_size - 1) / workgroup_size;
        let max_per_dim = 65535u32;
        let dispatch_x = total_workgroups.min(max_per_dim);
        let dispatch_y = (total_workgroups + max_per_dim - 1) / max_per_dim;
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    // ----- 3. Radix sort -----
    let result_in_b = rmesh_sort::record_radix_sort(
        encoder, device, sort_pipelines, sort_state,
        &buffers.sort_keys, &buffers.sort_values,
    );

    // ----- 3.5. Indirect convert: instance_count → mesh dispatch args (TETS_PER_GROUP=16) -----
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("interval_indirect_convert"),
            timestamp_writes: profiler.map(|qs| wgpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(2),
                end_of_pass_write_index: Some(3),
            }),
        });
        cpass.set_pipeline(&interval_pipelines.indirect_convert_pipeline);
        cpass.set_bind_group(0, indirect_convert_bg, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }

    // ----- 4. Interval shading render pass (single color output) -----
    let render_bg = if result_in_b { interval_render_bg_b } else { interval_render_bg_a };
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("interval_forward_render"),
            color_attachments: &[
                // Single color attachment: premultiplied alpha blend
                Some(wgpu::RenderPassColorAttachment {
                    view: &targets.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
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
            timestamp_writes: profiler.map(|qs| wgpu::RenderPassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(4),
                end_of_pass_write_index: Some(5),
            }),
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
        rpass.set_pipeline(&interval_pipelines.mesh_render_pipeline);
        rpass.set_bind_group(0, render_bg, &[]);

        rpass.draw_mesh_tasks_indirect(&buffers.mesh_indirect_args, 0);
    }
}

/// Record a sorted forward pass using compute-based interval shading (no mesh shader needed).
///
/// Steps:
///   1. Reset indirect args + interval_args_buf
///   2. Compute pass (SH eval + cull + depth keys)
///   3. Radix sort
///   4. Indirect convert: writes both compute dispatch + draw-indexed-indirect args
///   5. Compute pass: interval_compute generates vertices + per-tet data
///   6. Render pass with instanced indexed draw + interval_fragment (single color output)
pub fn record_sorted_compute_interval_forward_pass(
    encoder: &mut wgpu::CommandEncoder,
    device: &wgpu::Device,
    fwd_pipelines: &ForwardPipelines,
    ci_pipelines: &ComputeIntervalPipelines,
    sort_pipelines: &rmesh_sort::RadixSortPipelines,
    sort_state: &rmesh_sort::RadixSortState,
    buffers: &SceneBuffers,
    targets: &RenderTargets,
    compute_bg: &wgpu::BindGroup,
    gen_bg_a: &wgpu::BindGroup,
    gen_bg_b: &wgpu::BindGroup,
    ci_render_bg: &wgpu::BindGroup,
    ci_convert_bg: &wgpu::BindGroup,
    tet_count: u32,
    queue: &wgpu::Queue,
    depth_view: &wgpu::TextureView,
    hw_compute_bg: Option<&wgpu::BindGroup>,
    profiler: Option<&wgpu::QuerySet>,
) {
    // ----- 1. Reset indirect args + interval_args_buf -----
    let reset_cmd = DrawIndirectCommand {
        vertex_count: 12,
        instance_count: 0,
        first_vertex: 0,
        first_instance: 0,
    };
    queue.write_buffer(&buffers.indirect_args, 0, bytemuck::bytes_of(&reset_cmd));
    queue.write_buffer(
        &buffers.interval_args_buf,
        0,
        bytemuck::cast_slice(&[0u32; 8]),
    );

    let n_pow2 = tet_count.next_power_of_two();
    queue.write_buffer(sort_state.num_keys_buf(), 0, bytemuck::bytes_of(&n_pow2));

    // ----- 2. Compute pass (projection + SH eval) -----
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("project_compute"),
            timestamp_writes: profiler.map(|qs| wgpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }),
        });
        if let Some(hw_bg) = hw_compute_bg {
            cpass.set_pipeline(&fwd_pipelines.hw_compute_pipeline);
            cpass.set_bind_group(0, hw_bg, &[]);
        } else {
            cpass.set_pipeline(&fwd_pipelines.compute_pipeline);
            cpass.set_bind_group(0, compute_bg, &[]);
        }

        let workgroup_size = 64u32;
        let total_workgroups = (n_pow2 + workgroup_size - 1) / workgroup_size;
        let max_per_dim = 65535u32;
        let dispatch_x = total_workgroups.min(max_per_dim);
        let dispatch_y = (total_workgroups + max_per_dim - 1) / max_per_dim;
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    // ----- 3. Radix sort -----
    let result_in_b = rmesh_sort::record_radix_sort(
        encoder, device, sort_pipelines, sort_state,
        &buffers.sort_keys, &buffers.sort_values,
    );

    // ----- 4. Indirect convert → combined dispatch + draw args -----
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_interval_indirect_convert"),
            timestamp_writes: profiler.map(|qs| wgpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(2),
                end_of_pass_write_index: Some(3),
            }),
        });
        cpass.set_pipeline(&ci_pipelines.indirect_convert_pipeline);
        cpass.set_bind_group(0, ci_convert_bg, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }

    // ----- 5. Compute pass: generate interval vertices + indices -----
    {
        let gen_bg = if result_in_b { gen_bg_b } else { gen_bg_a };
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_interval_gen"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ci_pipelines.gen_pipeline);
        cpass.set_bind_group(0, gen_bg, &[]);
        cpass.dispatch_workgroups_indirect(&buffers.interval_args_buf, 0);
    }

    // ----- 6. Render pass (single color output) -----
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("compute_interval_render"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &targets.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
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
            timestamp_writes: profiler.map(|qs| wgpu::RenderPassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(4),
                end_of_pass_write_index: Some(5),
            }),
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
        rpass.set_pipeline(&ci_pipelines.render_pipeline);
        rpass.set_bind_group(0, ci_render_bg, &[]);
        rpass.set_index_buffer(buffers.interval_fan_index_buf.slice(..), wgpu::IndexFormat::Uint32);
        // draw-indexed-indirect args start at byte offset 12 (skip 3 dispatch u32s)
        rpass.draw_indexed_indirect(&buffers.interval_args_buf, 12);
    }
}

// ---------------------------------------------------------------------------
// High-level helpers
// ---------------------------------------------------------------------------

/// Convenience: set up everything needed for a forward frame.
///
/// Returns (SceneBuffers, MaterialBuffers, ForwardPipelines, RenderTargets, compute_bg, render_bg).
pub fn setup_forward(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    scene: &SceneData,
    base_colors: &[f32],
    color_grads: &[f32],
    width: u32,
    height: u32,
) -> (
    SceneBuffers,
    MaterialBuffers,
    ForwardPipelines,
    RenderTargets,
    wgpu::BindGroup,
    wgpu::BindGroup,
) {
    let color_format = wgpu::TextureFormat::Rgba16Float;

    let buffers = SceneBuffers::upload(device, queue, scene);
    let material = MaterialBuffers::upload(device, base_colors, color_grads, scene.tet_count);
    let pipelines = ForwardPipelines::new(device, color_format);
    let targets = RenderTargets::new(device, width, height);

    // Dummy sh_coeffs buffer (sh_degree=0 path uses base_colors instead)
    let dummy_sh = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dummy_sh_coeffs"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let compute_bg = create_compute_bind_group(device, &pipelines, &buffers, &material, &dummy_sh);
    let render_bg = create_render_bind_group(device, &pipelines, &buffers, &material);

    (buffers, material, pipelines, targets, compute_bg, render_bg)
}

/// Build a `Uniforms` struct from camera matrices and scene metadata.
pub fn make_uniforms(
    vp: Mat4,
    c2w: glam::Mat3,
    intrinsics: [f32; 4],
    cam_pos: glam::Vec3,
    screen_width: f32,
    screen_height: f32,
    tet_count: u32,
    step: u32,
    tile_size: u32,
    min_t: f32,
    sh_degree: u32,
    near_plane: f32,
    far_plane: f32,
) -> Uniforms {
    Uniforms {
        vp_col0: vp.col(0).into(),
        vp_col1: vp.col(1).into(),
        vp_col2: vp.col(2).into(),
        vp_col3: vp.col(3).into(),
        c2w_col0: [c2w.col(0).x, c2w.col(0).y, c2w.col(0).z, 0.0],
        c2w_col1: [c2w.col(1).x, c2w.col(1).y, c2w.col(1).z, 0.0],
        c2w_col2: [c2w.col(2).x, c2w.col(2).y, c2w.col(2).z, 0.0],
        intrinsics,
        cam_pos_pad: [cam_pos.x, cam_pos.y, cam_pos.z, 0.0],
        screen_width,
        screen_height,
        tet_count,
        step,
        tile_size_u: tile_size,
        ray_mode: 0,
        min_t,
        sh_degree,
        near_plane,
        far_plane,
        _pad1: [0; 2],
    }
}

// ---------------------------------------------------------------------------
// Ray Tracing: Tet Neighbors
// ---------------------------------------------------------------------------

/// Tet neighbor adjacency: `neighbors[tet_id * 4 + face_idx]` = neighbor tet or -1.
pub fn compute_tet_neighbors(indices: &[u32], tet_count: usize) -> Vec<i32> {
    use rmesh_util::shared::TET_FACE_INDICES;

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
    use rmesh_util::shared::TET_FACE_INDICES;

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
    aux_bind_group_layout: wgpu::BindGroupLayout,
    /// The output storage buffer: [W x H x 4] f32
    pub rendered_image: wgpu::Buffer,
    /// Auxiliary output buffer: [W x H x AUX_STRIDE] f32
    pub aux_image: wgpu::Buffer,
    /// Default aux bind group (group 1) with dummy aux_data
    aux_bind_group: wgpu::BindGroup,
    pub width: u32,
    pub height: u32,
    pub aux_dim: u32,
    pub aux_stride: u32,
}

impl RayTracePipeline {
    pub fn new(device: &wgpu::Device, width: u32, height: u32, aux_dim: u32) -> Self {
        let aux_stride = 8 + aux_dim;

        // String-substitute AUX_DIM and AUX_ACC_SIZE in shader source
        let source = RAYTRACE_COMPUTE_WGSL
            .replace("/*AUX_DIM*/0u", &format!("{}u", aux_dim))
            .replace("/*AUX_ACC_SIZE*/1u", &format!("{}u", aux_dim.max(1)));

        let shader = rmesh_util::compose::create_shader_module(
            device, "raytrace_compute.wgsl", &source,
        ).expect("Failed to compose raytrace_compute.wgsl");

        // Group 0: 13 bindings (0-6 read, 7 rw, 8-12 read)
        let read_only = [true, true, true, true, true, true, true, false, true, true, true, true, true];
        let entries = storage_entries(13, wgpu::ShaderStages::COMPUTE, &read_only);

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("raytrace_bgl"),
                entries: &entries,
            });

        // Group 1: aux_image (rw), aux_data (read)
        let aux_entries = storage_entries(2, wgpu::ShaderStages::COMPUTE, &[false, true]);
        let aux_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("raytrace_aux_bgl"),
                entries: &aux_entries,
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raytrace_pl"),
            bind_group_layouts: &[&bind_group_layout, &aux_bind_group_layout],
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

        let aux_image = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raytrace_aux_image"),
            size: (width as u64) * (height as u64) * (aux_stride as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Dummy aux_data buffer (4 bytes minimum)
        let aux_data_dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raytrace_aux_data_dummy"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let aux_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raytrace_aux_bg_default"),
            layout: &aux_bind_group_layout,
            entries: &[
                buf_entry(0, &aux_image),
                buf_entry(1, &aux_data_dummy),
            ],
        });

        Self {
            pipeline, bind_group_layout, aux_bind_group_layout,
            rendered_image, aux_image, aux_bind_group,
            width, height, aux_dim, aux_stride,
        }
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
    /// Per-pixel ray origins [W*H*3] f32 (used when ray_mode=1)
    pub ray_origins: wgpu::Buffer,
    /// Per-pixel ray directions [W*H*3] f32 (used when ray_mode=1)
    pub ray_dirs: wgpu::Buffer,
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

        // Placeholder buffers for ray_origins and ray_dirs (4 bytes each, minimum valid)
        let ray_origins = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ray_origins"),
            contents: bytemuck::cast_slice(&[0.0f32]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let ray_dirs = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ray_dirs"),
            contents: bytemuck::cast_slice(&[0.0f32]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        Self { tet_neighbors, bvh_nodes, boundary_faces, start_tet, ray_origins, ray_dirs }
    }
}

/// Create the ray trace bind group.
pub fn create_raytrace_bind_group(
    device: &wgpu::Device,
    rt_pipeline: &RayTracePipeline,
    scene_buffers: &SceneBuffers,
    material: &MaterialBuffers,
    rt_buffers: &RayTraceBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("raytrace_bg"),
        layout: rt_pipeline.bind_group_layout(),
        entries: &[
            buf_entry(0, &scene_buffers.uniforms),
            buf_entry(1, &scene_buffers.vertices),
            buf_entry(2, &scene_buffers.indices),
            buf_entry(3, &material.colors),
            buf_entry(4, &scene_buffers.densities),
            buf_entry(5, &material.color_grads),
            buf_entry(6, &rt_buffers.tet_neighbors),
            buf_entry(7, &rt_pipeline.rendered_image),
            buf_entry(8, &rt_buffers.bvh_nodes),
            buf_entry(9, &rt_buffers.boundary_faces),
            buf_entry(10, &rt_buffers.start_tet),
            buf_entry(11, &rt_buffers.ray_origins),
            buf_entry(12, &rt_buffers.ray_dirs),
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
    pass.set_bind_group(1, &rt_pipeline.aux_bind_group, &[]);
    pass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
}

// ===========================================================================
// Forward tiled pipeline
// ===========================================================================

/// Compute-based forward renderer using tiles with warp-per-tile.
///
/// Requires `wgpu::Features::SUBGROUPS` on the device.
/// Renders directly to an f32 storage buffer (no texture intermediate).
pub struct RasterizeComputePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    aux_bind_group_layout: wgpu::BindGroupLayout,
    /// The output storage buffer: [W x H x 4] f32
    pub rendered_image: wgpu::Buffer,
    /// Auxiliary output buffer: [W x H x AUX_STRIDE] f32
    pub aux_image: wgpu::Buffer,
    /// Debug stats output buffer: [W x H x 4] u32 (ray_miss, ghost, occluded, useful)
    pub debug_image: wgpu::Buffer,
    /// Default aux bind group (group 1) with dummy aux_data + debug_image
    pub aux_bind_group: wgpu::BindGroup,
    pub width: u32,
    pub height: u32,
    pub aux_dim: u32,
    pub aux_stride: u32,
}

impl RasterizeComputePipeline {
    /// Create the forward tiled pipeline and allocate the rendered_image buffer.
    pub fn new(device: &wgpu::Device, width: u32, height: u32, aux_dim: u32) -> Self {
        let aux_stride = 8 + aux_dim;

        // String-substitute AUX_DIM and SM_AUX_SIZE in shader source
        let source = RASTERIZE_COMPUTE_WGSL
            .replace("/*AUX_DIM*/0u", &format!("{}u", aux_dim))
            .replace("/*SM_AUX_SIZE*/1u", &format!("{}u", (256 * aux_dim).max(1)));

        let shader = rmesh_util::compose::create_shader_module(
            device, "rasterize_compute.wgsl", &source,
        ).expect("Failed to compose rasterize_compute.wgsl");

        // Group 0: 10 bindings
        let read_only = [true, true, true, true, true, true, true, true, true, false];
        let entries: Vec<wgpu::BindGroupLayoutEntry> = read_only
            .iter()
            .enumerate()
            .map(|(i, &ro)| rmesh_tile::storage_entry(i as u32, ro))
            .collect();

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rasterize_compute_bgl"),
                entries: &entries,
            });

        // Group 1: aux_image (rw), aux_data (read), debug_image (rw)
        let aux_entries: Vec<wgpu::BindGroupLayoutEntry> = [false, true, false]
            .iter()
            .enumerate()
            .map(|(i, &ro)| rmesh_tile::storage_entry(i as u32, ro))
            .collect();
        let aux_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rasterize_aux_bgl"),
                entries: &aux_entries,
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rasterize_compute_pl"),
            bind_group_layouts: &[&bind_group_layout, &aux_bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rasterize_compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let rendered_image = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rasterize_rendered_image"),
            size: (width as u64) * (height as u64) * 4 * 4, // RGBA f32
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let aux_image = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rasterize_aux_image"),
            size: (width as u64) * (height as u64) * (aux_stride as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Dummy aux_data buffer (4 bytes minimum)
        let aux_data_dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rasterize_aux_data_dummy"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Debug stats buffer: 4 u32 per pixel (ray_miss, ghost, occluded, useful)
        let debug_image = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rasterize_debug_image"),
            size: (width as u64) * (height as u64) * 4 * 4, // 4 × u32 per pixel
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let aux_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rasterize_aux_bg_default"),
            layout: &aux_bind_group_layout,
            entries: &[
                buf_entry(0, &aux_image),
                buf_entry(1, &aux_data_dummy),
                buf_entry(2, &debug_image),
            ],
        });

        Self {
            pipeline,
            bind_group_layout,
            aux_bind_group_layout,
            rendered_image,
            aux_image,
            debug_image,
            aux_bind_group,
            width,
            height,
            aux_dim,
            aux_stride,
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
/// Binding order matches `rasterize_compute.wgsl`:
///   0: uniforms, 1: vertices, 2: indices, 3: colors,
///   4: densities, 5: color_grads, 6: tile_sort_values,
///   7: tile_ranges, 8: tile_uniforms, 9: rendered_image
pub fn create_rasterize_bind_group(
    device: &wgpu::Device,
    rasterize: &RasterizeComputePipeline,
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
        label: Some("rasterize_compute_bg"),
        layout: rasterize.bind_group_layout(),
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
            buf_entry(9, &rasterize.rendered_image),
        ],
    })
}

/// Record the forward tiled compute pass dispatch.
///
/// Dispatches one workgroup per tile (32 threads each).
pub fn record_rasterize_compute(
    encoder: &mut wgpu::CommandEncoder,
    rasterize: &RasterizeComputePipeline,
    bind_group: &wgpu::BindGroup,
    num_tiles: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("rasterize_compute"),
        timestamp_writes: None,
    });
    pass.set_pipeline(rasterize.pipeline());
    pass.set_bind_group(0, bind_group, &[]);
    pass.set_bind_group(1, &rasterize.aux_bind_group, &[]);
    let (x, y) = dispatch_2d(num_tiles);
    pass.dispatch_workgroups(x, y, 1);
}

// ===========================================================================
// Point Location Pipeline (adjacency walking)
// ===========================================================================

/// Uniforms for the locate compute shader.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LocateUniforms {
    pub num_queries: u32,
    pub hint_tet: i32,   // global hint, or -1 to use per-query hint_tets
    pub tet_count: u32,
    pub _pad: u32,
}

/// GPU pipeline for point-in-tet location via adjacency walking.
pub struct LocatePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl LocatePipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("locate_compute.wgsl"),
            source: wgpu::ShaderSource::Wgsl(LOCATE_COMPUTE_WGSL.into()),
        });

        // 7 bindings: 0-5 read, 6 read_write
        let read_only = [true, true, true, true, true, true, false];
        let entries = storage_entries(7, wgpu::ShaderStages::COMPUTE, &read_only);

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("locate_bgl"),
                entries: &entries,
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("locate_pl"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("locate_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self { pipeline, bind_group_layout }
    }

    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    pub fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }
}

/// Create the locate bind group.
///
/// Binding order matches `locate_compute.wgsl`:
///   0: locate_uniforms, 1: vertices, 2: indices, 3: tet_neighbors,
///   4: query_points, 5: hint_tets, 6: result_tets
pub fn create_locate_bind_group(
    device: &wgpu::Device,
    pipeline: &LocatePipeline,
    locate_uniforms_buf: &wgpu::Buffer,
    vertices: &wgpu::Buffer,
    indices: &wgpu::Buffer,
    tet_neighbors: &wgpu::Buffer,
    query_points: &wgpu::Buffer,
    hint_tets: &wgpu::Buffer,
    result_tets: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("locate_bg"),
        layout: pipeline.bind_group_layout(),
        entries: &[
            buf_entry(0, locate_uniforms_buf),
            buf_entry(1, vertices),
            buf_entry(2, indices),
            buf_entry(3, tet_neighbors),
            buf_entry(4, query_points),
            buf_entry(5, hint_tets),
            buf_entry(6, result_tets),
        ],
    })
}

/// Record the locate compute pass dispatch.
pub fn record_locate(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &LocatePipeline,
    bind_group: &wgpu::BindGroup,
    num_queries: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("locate"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline.pipeline());
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups((num_queries + 63) / 64, 1, 1);
}

// ---------------------------------------------------------------------------
// Point Location: CPU Walking
// ---------------------------------------------------------------------------

/// Find the tet containing `point` by walking from `hint_tet`.
///
/// Same algorithm as the GPU shader but on CPU. Falls back to brute-force
/// `find_containing_tet()` if the walk exceeds 512 steps.
///
/// Vertex-to-face mapping: vertex k -> face opposite vertex k.
///   vertex 0 -> face 1, vertex 1 -> face 2,
///   vertex 2 -> face 3, vertex 3 -> face 0.
pub fn find_containing_tet_walk(
    vertices: &[f32],
    indices: &[u32],
    neighbors: &[i32],
    tet_count: usize,
    point: Vec3,
    hint_tet: usize,
) -> Option<u32> {
    const VERTEX_TO_FACE: [usize; 4] = [1, 2, 3, 0];
    const MAX_ITERS: usize = 512;
    const EPS: f32 = -1e-6;

    let mut current = if hint_tet < tet_count { hint_tet } else { 0 };

    for _ in 0..MAX_ITERS {
        let vi = [
            indices[current * 4] as usize,
            indices[current * 4 + 1] as usize,
            indices[current * 4 + 2] as usize,
            indices[current * 4 + 3] as usize,
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
            // Degenerate tet — fall back to brute force
            return find_containing_tet(vertices, indices, tet_count, point);
        }
        let inv_det = 1.0 / det;

        let u = p.dot(e.cross(f)) * inv_det; // bary for v1
        let vc = d.dot(p.cross(f)) * inv_det; // bary for v2
        let w = d.dot(e.cross(p)) * inv_det; // bary for v3
        let s = 1.0 - u - vc - w; // bary for v0

        // Check containment
        if s >= EPS && u >= EPS && vc >= EPS && w >= EPS {
            return Some(current as u32);
        }

        // Find most negative barycentric
        let barys = [s, u, vc, w];
        let mut min_idx = 0;
        for k in 1..4 {
            if barys[k] < barys[min_idx] {
                min_idx = k;
            }
        }

        let face_idx = VERTEX_TO_FACE[min_idx];
        let neighbor = neighbors[current * 4 + face_idx];

        if neighbor < 0 {
            return None; // Outside mesh
        }

        current = neighbor as usize;
    }

    // Walk exhausted — fall back to brute force
    find_containing_tet(vertices, indices, tet_count, point)
}

// ---------------------------------------------------------------------------
// Blit Pipeline (Rgba16Float → sRGB swapchain)
// ---------------------------------------------------------------------------

/// Pipeline that blits the Rgba16Float render target to the sRGB swapchain
/// via a fullscreen triangle.
pub struct BlitPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub sampler: wgpu::Sampler,
}

impl BlitPipeline {
    /// Create the blit pipeline targeting `target_format` (e.g. Bgra8UnormSrgb).
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit.wgsl"),
            source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("blit_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit_pl"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("blit_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            pipeline,
            bind_group_layout,
            sampler,
        }
    }
}

/// Create a bind group for the blit pipeline from a source texture view.
pub fn create_blit_bind_group(
    device: &wgpu::Device,
    blit: &BlitPipeline,
    source_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("blit_bg"),
        layout: &blit.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(source_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&blit.sampler),
            },
        ],
    })
}

/// Record a blit render pass: fullscreen triangle sampling `source` to `target_view`.
pub fn record_blit(
    encoder: &mut wgpu::CommandEncoder,
    blit: &BlitPipeline,
    bind_group: &wgpu::BindGroup,
    target_view: &wgpu::TextureView,
) {
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("blit"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target_view,
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
    rpass.set_pipeline(&blit.pipeline);
    rpass.set_bind_group(0, bind_group, &[]);
    rpass.draw(0..3, 0..1); // fullscreen triangle
}
