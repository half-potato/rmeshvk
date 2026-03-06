//! Forward rendering pipeline orchestration (wgpu).
//!
//! Sets up the wgpu compute and render pipelines for the forward pass:
//!   1. Compute pass: SH eval, cull, depth key generation
//!   2. Sort pass: Bitonic sort of tets back-to-front
//!   3. Render pass: Hardware rasterization with MRT output
//!
//! All GPU buffer management and bind group creation lives here.

use glam::Mat4;
use rmesh_data::SceneData;
use wgpu::util::DeviceExt;

// Re-export shared types for CPU-side use.
pub use rmesh_shaders::shared::{DrawIndirectCommand, SortUniforms, Uniforms};

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

        // Sort buffers padded to next power of 2 for correct bitonic sort.
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

        let tiles_touched = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tiles_touched"),
            size: m * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
    pub sort_pipeline: wgpu::ComputePipeline,
    pub sort_bind_group_layout: wgpu::BindGroupLayout,
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
            source: wgpu::ShaderSource::Wgsl(rmesh_shaders::FORWARD_COMPUTE_WGSL.into()),
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
            source: wgpu::ShaderSource::Wgsl(rmesh_shaders::FORWARD_VERTEX_WGSL.into()),
        });
        let fragment_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forward_fragment.wgsl"),
            source: wgpu::ShaderSource::Wgsl(rmesh_shaders::FORWARD_FRAGMENT_WGSL.into()),
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

        // ----- Sort pipeline (3 bindings) -----
        // Binding 0: SortUniforms (read-only storage)
        // Binding 1: keys (read-write)
        // Binding 2: values (read-write)
        let sort_read_only = [true, false, false];
        let sort_entries = storage_entries(
            3,
            wgpu::ShaderStages::COMPUTE,
            &sort_read_only,
        );
        let sort_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sort_bind_group_layout"),
                entries: &sort_entries,
            });
        let sort_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sort_pipeline_layout"),
                bind_group_layouts: &[&sort_bind_group_layout],
                immediate_size: 0,
            });
        let sort_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("radix_sort.wgsl"),
            source: wgpu::ShaderSource::Wgsl(rmesh_shaders::RADIX_SORT_WGSL.into()),
        });
        let sort_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("sort_pipeline"),
                layout: Some(&sort_pipeline_layout),
                module: &sort_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            compute_pipeline,
            compute_bind_group_layout,
            render_pipeline,
            render_bind_group_layout,
            sort_pipeline,
            sort_bind_group_layout,
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
            source: wgpu::ShaderSource::Wgsl(rmesh_shaders::TEX_TO_BUFFER_WGSL.into()),
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

// ---------------------------------------------------------------------------
// Forward Tiled Compute Pipeline (4x4 tiles, subgroup-based)
// ---------------------------------------------------------------------------

/// Compute-based forward renderer using 4x4 tiles with warp-per-tile.
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
            source: wgpu::ShaderSource::Wgsl(rmesh_shaders::FORWARD_TILED_WGSL.into()),
        });

        // 10 bindings: uniforms, vertices, indices, colors, densities, color_grads,
        //              tile_sort_values, tile_ranges, tile_uniforms, rendered_image
        let read_only = [true, true, true, true, true, true, true, true, true, false];
        let entries = storage_entries(10, wgpu::ShaderStages::COMPUTE, &read_only);

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
    scene_buffers: &SceneBuffers,
    tile_sort_values: &wgpu::Buffer,
    tile_ranges: &wgpu::Buffer,
    tile_uniforms: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("forward_tiled_bg"),
        layout: fwd_tiled.bind_group_layout(),
        entries: &[
            buf_entry(0, &scene_buffers.uniforms),
            buf_entry(1, &scene_buffers.vertices),
            buf_entry(2, &scene_buffers.indices),
            buf_entry(3, &scene_buffers.colors),
            buf_entry(4, &scene_buffers.densities),
            buf_entry(5, &scene_buffers.color_grads),
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
    // Use 2D dispatch if num_tiles exceeds 65535
    if num_tiles <= 65535 {
        pass.dispatch_workgroups(num_tiles, 1, 1);
    } else {
        let x = 65535u32;
        let y = (num_tiles + x - 1) / x;
        pass.dispatch_workgroups(x, y, 1);
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

/// Create a sort bind group for a specific offset within the sort uniform buffer.
///
/// Binding order matches `radix_sort.wgsl`:
///   0: sort_uniforms (at given offset, SortUniforms size),
///   1: keys, 2: values
pub fn create_sort_bind_group(
    device: &wgpu::Device,
    pipelines: &ForwardPipelines,
    sort_uniform_buffer: &wgpu::Buffer,
    sort_keys: &wgpu::Buffer,
    sort_values: &wgpu::Buffer,
    offset: wgpu::BufferAddress,
) -> wgpu::BindGroup {
    let uniform_size = std::mem::size_of::<SortUniforms>() as wgpu::BufferAddress;
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sort_bind_group"),
        layout: &pipelines.sort_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: sort_uniform_buffer,
                    offset,
                    size: wgpu::BufferSize::new(uniform_size),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: sort_keys.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: sort_values.as_entire_binding(),
            },
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
// Sort State
// ---------------------------------------------------------------------------

/// Pre-computed bitonic sort state.
///
/// Contains a single GPU buffer with all (stage, step_size) uniform pairs
/// at 256-byte aligned offsets, and one bind group per sort dispatch.
pub struct SortState {
    /// Buffer holding all padded SortUniforms entries.
    pub uniform_buffer: wgpu::Buffer,
    /// One bind group per sort dispatch step (references uniform_buffer at the
    /// appropriate offset, plus the sort_keys and sort_values buffers).
    pub bind_groups: Vec<wgpu::BindGroup>,
    /// Number of sort dispatches. Same as `bind_groups.len()`.
    pub step_count: usize,
    /// Workgroup count per sort dispatch: ceil(tet_count / 256).
    pub dispatch_x: u32,
}

/// Minimum alignment for storage buffer offsets. The wgpu spec requires
/// `minStorageBufferOffsetAlignment`, which is at most 256 on all backends.
const SORT_UNIFORM_ALIGNMENT: wgpu::BufferAddress = 256;

impl SortState {
    /// Build the sort state for a given tet count.
    ///
    /// Pre-computes all bitonic sort (stage, step_size) pairs, writes them into
    /// a single padded buffer, and creates one bind group per pair.
    pub fn new(
        device: &wgpu::Device,
        pipelines: &ForwardPipelines,
        sort_keys: &wgpu::Buffer,
        sort_values: &wgpu::Buffer,
        tet_count: u32,
    ) -> Self {
        // Enumerate all (stage, step_size) pairs for the full bitonic network.
        let n_pow2 = tet_count.next_power_of_two();
        let mut pairs: Vec<SortUniforms> = Vec::new();

        let mut k = 2u32;
        while k <= n_pow2 {
            // stage = log2(k) - 1
            let stage = (k as f32).log2() as u32 - 1;
            let mut j = k >> 1;
            while j > 0 {
                let step_bit = (j as f32).log2() as u32;
                // Use n_pow2 as count so ALL comparisons happen in valid memory.
                pairs.push(SortUniforms {
                    count: n_pow2,
                    stage,
                    step_size: step_bit,
                    _pad: 0,
                });
                j >>= 1;
            }
            k <<= 1;
        }

        let step_count = pairs.len();

        // Write all SortUniforms into a single buffer with SORT_UNIFORM_ALIGNMENT padding.
        let buf_size = step_count as wgpu::BufferAddress * SORT_UNIFORM_ALIGNMENT;
        let mut data = vec![0u8; buf_size as usize];
        let uniform_bytes = std::mem::size_of::<SortUniforms>();

        for (i, su) in pairs.iter().enumerate() {
            let offset = i as usize * SORT_UNIFORM_ALIGNMENT as usize;
            data[offset..offset + uniform_bytes].copy_from_slice(bytemuck::bytes_of(su));
        }

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sort_uniforms"),
            contents: &data,
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Create one bind group per pair, each pointing at the appropriate
        // offset within the uniform buffer.
        let bind_groups: Vec<wgpu::BindGroup> = (0..step_count)
            .map(|i| {
                let offset = i as wgpu::BufferAddress * SORT_UNIFORM_ALIGNMENT;
                create_sort_bind_group(
                    device,
                    pipelines,
                    &uniform_buffer,
                    sort_keys,
                    sort_values,
                    offset,
                )
            })
            .collect();

        // Dispatch for n_pow2 elements (padded buffers).
        let dispatch_x = (n_pow2 + 255) / 256;

        Self {
            uniform_buffer,
            bind_groups,
            step_count,
            dispatch_x,
        }
    }
}

// ---------------------------------------------------------------------------
// TileSortState — bitonic sort for tile-tet pairs
// ---------------------------------------------------------------------------

/// Bitonic sort state for tile-tet pair sorting.
///
/// Same pattern as `SortState` but operates on separately-sized buffers.
/// The sort pipeline itself is shared with the forward sort (ForwardPipelines).
pub struct TileSortState {
    pub uniform_buffer: wgpu::Buffer,
    pub bind_groups: Vec<wgpu::BindGroup>,
    pub step_count: usize,
    pub dispatch_x: u32,
}

impl TileSortState {
    /// Build the tile sort state for the given max_pairs_pow2 count.
    ///
    /// Reuses the sort pipeline and bind group layout from ForwardPipelines.
    pub fn new(
        device: &wgpu::Device,
        pipelines: &ForwardPipelines,
        tile_sort_keys: &wgpu::Buffer,
        tile_sort_values: &wgpu::Buffer,
        max_pairs_pow2: u32,
    ) -> Self {
        let n_pow2 = max_pairs_pow2;
        let mut pairs: Vec<SortUniforms> = Vec::new();

        let mut k = 2u32;
        while k <= n_pow2 {
            let stage = (k as f32).log2() as u32 - 1;
            let mut j = k >> 1;
            while j > 0 {
                let step_bit = (j as f32).log2() as u32;
                pairs.push(SortUniforms {
                    count: n_pow2,
                    stage,
                    step_size: step_bit,
                    _pad: 0,
                });
                j >>= 1;
            }
            k <<= 1;
        }

        let step_count = pairs.len();

        let buf_size = step_count as wgpu::BufferAddress * SORT_UNIFORM_ALIGNMENT;
        let mut data = vec![0u8; buf_size as usize];
        let uniform_bytes = std::mem::size_of::<SortUniforms>();

        for (i, su) in pairs.iter().enumerate() {
            let offset = i as usize * SORT_UNIFORM_ALIGNMENT as usize;
            data[offset..offset + uniform_bytes].copy_from_slice(bytemuck::bytes_of(su));
        }

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tile_sort_uniforms"),
            contents: &data,
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bind_groups: Vec<wgpu::BindGroup> = (0..step_count)
            .map(|i| {
                let offset = i as wgpu::BufferAddress * SORT_UNIFORM_ALIGNMENT;
                create_sort_bind_group(
                    device,
                    pipelines,
                    &uniform_buffer,
                    tile_sort_keys,
                    tile_sort_values,
                    offset,
                )
            })
            .collect();

        let dispatch_x = (n_pow2 + 255) / 256;

        Self {
            uniform_buffer,
            bind_groups,
            step_count,
            dispatch_x,
        }
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
/// directly instead of relying on bitonic-sorted sort_values.
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

/// Record the compute + bitonic sort portion of the forward pass (no hardware rasterization).
///
/// Stages:
///   1. Reset indirect args (vertex_count=12, instance_count=0)
///   2. Compute pass (SH eval + cull + depth keys)
///   3. Sort pass (bitonic sort, multiple dispatches)
///
/// After this, `sort_values[0..visible_tet_count-1]` contains visible tet IDs
/// sorted by depth, which is needed by tile_gen.
pub fn record_forward_compute_and_sort(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &ForwardPipelines,
    buffers: &SceneBuffers,
    compute_bg: &wgpu::BindGroup,
    sort_state: &SortState,
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

    // ----- 3. Sort pass (bitonic sort) -----
    {
        let mut spass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bitonic_sort"),
            timestamp_writes: None,
        });
        spass.set_pipeline(&pipelines.sort_pipeline);

        for i in 0..sort_state.step_count {
            spass.set_bind_group(0, &sort_state.bind_groups[i], &[]);
            spass.dispatch_workgroups(sort_state.dispatch_x, 1, 1);
        }
    }
}

/// Record the full forward pass into a command encoder.
///
/// Stages:
///   1. Reset indirect args (vertex_count=12, instance_count=0)
///   2. Compute pass (SH eval + cull + depth keys)
///   3. Sort pass (bitonic sort, multiple dispatches)
///   4. Render pass (hardware rasterization with MRT, draw_indirect)
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
    sort_state: &SortState,
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

    // ----- 3. Sort pass (bitonic sort -- multiple dispatches) -----
    // Each dispatch needs its own bind group (with different SortUniforms offset).
    // wgpu handles compute-to-compute barriers automatically between passes.
    {
        let mut spass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bitonic_sort"),
            timestamp_writes: None,
        });
        spass.set_pipeline(&pipelines.sort_pipeline);

        for i in 0..sort_state.step_count {
            spass.set_bind_group(0, &sort_state.bind_groups[i], &[]);
            spass.dispatch_workgroups(sort_state.dispatch_x, 1, 1);
        }
    }

    // ----- 4. Render pass -----
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
/// Returns (SceneBuffers, ForwardPipelines, RenderTargets, compute_bg, render_bg, SortState).
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
    SortState,
) {
    let color_format = wgpu::TextureFormat::Rgba16Float;
    let aux_format = wgpu::TextureFormat::Rgba32Float;

    let buffers = SceneBuffers::upload(device, queue, scene);
    let pipelines = ForwardPipelines::new(device, color_format, aux_format);
    let targets = RenderTargets::new(device, width, height);

    let compute_bg = create_compute_bind_group(device, &pipelines, &buffers);
    let render_bg = create_render_bind_group(device, &pipelines, &buffers);
    let sort_state = SortState::new(
        device,
        &pipelines,
        &buffers.sort_keys,
        &buffers.sort_values,
        scene.tet_count,
    );

    (buffers, pipelines, targets, compute_bg, render_bg, sort_state)
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
