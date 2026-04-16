//! rmesh-viewer: Interactive wgpu viewer for .rmesh files.
//!
//! Usage:
//!   rmesh-viewer <input.rmesh>
//!
//! Controls:
//!   Left-drag: Orbit
//!   Middle-drag: Pan
//!   Right-drag (vertical): Zoom
//!   Scroll: Zoom
//!   Escape: Quit

use anyhow::{Context, Result};
use glam::Vec3;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use rmesh_interact::{
    InteractContext, InteractEvent, InteractKey, InteractResult, Primitive,
    TransformInteraction,
};
use rmesh_sim::{FluidParams, FluidSim};
use rmesh_util::camera::Camera;
use wgpu::util::DeviceExt;

use rmesh_render::{
    BlitPipeline, ForwardPipelines, MaterialBuffers,
    MeshForwardPipelines, IntervalPipelines, ComputeIntervalPipelines, RenderTargets,
    SceneBuffers,
    create_compute_bind_group, create_hw_compute_bind_group, create_render_bind_group,
    create_render_bind_group_with_sort_values,
    create_blit_bind_group,
    create_indirect_convert_bind_group, create_mesh_render_bind_group,
    create_mesh_render_bind_group_with_sort_values,
    create_prepass_bind_group, create_quad_render_bind_group,
    create_interval_render_bind_group, create_interval_render_bind_group_with_sort_values,
    create_interval_indirect_convert_bind_group,
    create_compute_interval_gen_bind_group,
    create_compute_interval_gen_bind_group_with_sort_values,
    create_compute_interval_render_bind_group,
    create_compute_interval_indirect_convert_bind_group,
};
use rmesh_compositor::{PrimitiveGeometry, PrimitivePipeline, PrimitiveTargets};

mod gpu_state;
mod render;
use gpu_state::*;

/// Create an Rgba32Float texture for copying the raytrace buffer output to blit.
fn create_rt_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("rt_output_texture"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    camera: Camera,
    scene_data: rmesh_data::SceneData,
    sh_coeffs: rmesh_data::ShCoeffs,
    left_pressed: bool,
    middle_pressed: bool,
    right_pressed: bool,
    shift_pressed: bool,
    last_mouse: (f64, f64),
    // egui + FPS
    egui_ctx: egui::Context,
    frame_times: VecDeque<std::time::Instant>,
    fps: f64,
    loaded_path: Option<PathBuf>,
    pending_load: Option<PathBuf>,
    vsync: bool,
    render_mode: RenderMode,
    mesh_shader_supported: bool,
    // Fluid simulation
    fluid_enabled: bool,
    fluid_params: FluidParams,
    // Rendering options
    show_primitives: bool,
    sort_16bit: bool,
    // Transform interaction
    interaction: TransformInteraction,
    primitives: Vec<Primitive>,
    next_primitive_id: u32,
    // Deferred PBR shading
    deferred_enabled: bool,
    deferred_debug_mode: u32,
    ambient: f32,
    dsm_query_depth: f32,
    /// Cached light state for DSM dirty detection.
    cached_dsm_lights: Vec<rmesh_render::GpuLight>,
    cached_dsm_num_lights: u32,
    pbr_data: Option<rmesh_data::PbrData>,
    // Ray trace CPU-side state
    rt_neighbors_cpu: Vec<i32>,
    rt_start_tet_hint: i32,
    rt_locate_ms: f32,
}

impl App {
    fn new(scene: rmesh_data::SceneData, sh: rmesh_data::ShCoeffs, pbr: Option<rmesh_data::PbrData>) -> Self {
        let pos = Vec3::new(scene.start_pose[0], scene.start_pose[1], scene.start_pose[2]);
        let cam_pos = if pos.length() < 0.001 {
            Vec3::new(0.0, 3.0, -2.0)
        } else {
            pos
        };

        Self {
            window: None,
            gpu: None,
            camera: Camera::new(cam_pos),
            scene_data: scene,
            sh_coeffs: sh,
            left_pressed: false,
            middle_pressed: false,
            right_pressed: false,
            shift_pressed: false,
            last_mouse: (0.0, 0.0),
            egui_ctx: egui::Context::default(),
            frame_times: VecDeque::with_capacity(120),
            fps: 0.0,
            loaded_path: None,
            pending_load: None,
            vsync: true,
            render_mode: RenderMode::Regular,
            mesh_shader_supported: false,
            fluid_enabled: false,
            fluid_params: FluidParams::default(),
            show_primitives: false,
            sort_16bit: true,
            interaction: TransformInteraction::new(),
            primitives: Vec::new(),
            next_primitive_id: 1,
            deferred_enabled: pbr.is_some(),
            deferred_debug_mode: 0,
            ambient: 0.05,
            dsm_query_depth: 1.0,
            cached_dsm_lights: Vec::new(),
            cached_dsm_num_lights: 0,
            pbr_data: pbr,
            rt_neighbors_cpu: Vec::new(),
            rt_start_tet_hint: -1,
            rt_locate_ms: 0.0,
        }
    }

    fn init_gpu(&mut self, window: Arc<Window>) {
        let t_total = std::time::Instant::now();
        let size = window.inner_size();

        log::info!("Requesting GPU adapter...");
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let (adapter, device, queue, mesh_shader_supported, subgroup_supported) = pollster::block_on(async {
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: Some(&surface),
                    force_fallback_adapter: false,
                })
                .await
                .expect("No suitable GPU adapter found");

            log::info!("GPU: {:?}", adapter.get_info().name);

            let adapter_features = adapter.features();
            let backend = adapter.get_info().backend;
            let subgroup_supported = adapter_features.contains(wgpu::Features::SUBGROUP);
            let mesh_shader_supported = subgroup_supported
                && adapter_features.contains(wgpu::Features::EXPERIMENTAL_MESH_SHADER)
                && backend != wgpu::Backend::Metal; // naga MSL backend doesn't implement mesh shaders
            log::info!("Subgroup support: {}", subgroup_supported);
            log::info!("Mesh shader support: {}", mesh_shader_supported);

            let mut required_features = wgpu::Features::SHADER_FLOAT32_ATOMIC
                | wgpu::Features::TIMESTAMP_QUERY;
            if subgroup_supported {
                required_features |= wgpu::Features::SUBGROUP;
            }
            if mesh_shader_supported {
                required_features |= wgpu::Features::EXPERIMENTAL_MESH_SHADER;
            }

            let mut limits = wgpu::Limits::default();
            limits.max_storage_buffers_per_shader_stage = 20;
            limits.max_storage_buffer_binding_size = 2 * 1024 * 1024 * 1024 - 4; // 1 GB
            limits.max_buffer_size = 2 * 1024 * 1024 * 1024 - 4; // 1 GB

            // Copy mesh shader limits from adapter (they default to 0 = disabled)
            if mesh_shader_supported {
                let supported = adapter.limits();
                limits.max_mesh_invocations_per_workgroup = supported.max_mesh_invocations_per_workgroup;
                limits.max_mesh_invocations_per_dimension = supported.max_mesh_invocations_per_dimension;
                limits.max_mesh_output_vertices = supported.max_mesh_output_vertices;
                limits.max_mesh_output_primitives = supported.max_mesh_output_primitives;
                limits.max_mesh_output_layers = supported.max_mesh_output_layers;
                limits.max_mesh_multiview_view_count = supported.max_mesh_multiview_view_count;
                limits.max_task_mesh_workgroup_total_count = supported.max_task_mesh_workgroup_total_count;
                limits.max_task_mesh_workgroups_per_dimension = supported.max_task_mesh_workgroups_per_dimension;
            }

            // SAFETY: We opt into experimental features (mesh shaders) and accept
            // that the API surface may change in future wgpu releases.
            let experimental = if mesh_shader_supported {
                unsafe { wgpu::ExperimentalFeatures::enabled() }
            } else {
                wgpu::ExperimentalFeatures::disabled()
            };

            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("rmesh device"),
                        required_features,
                        required_limits: limits,
                        experimental_features: experimental,
                        ..Default::default()
                    },
                )
                .await
                .expect("Failed to create device");

            (adapter, device, queue, mesh_shader_supported, subgroup_supported)
        });
        self.mesh_shader_supported = mesh_shader_supported;
        self.render_mode = RenderMode::IntervalShader;

        let surface_caps = surface.get_capabilities(&adapter);
        // Prefer non-sRGB surface — blit shader applies linear→sRGB manually.
        // Using an sRGB surface would double-gamma.
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| !f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 3,
        };
        surface.configure(&device, &surface_config);

        let color_format = wgpu::TextureFormat::Rgba16Float;

        log::info!("Compiling shader pipelines...");
        let t0 = std::time::Instant::now();
        let pipelines = ForwardPipelines::new(&device, color_format);
        let blit_pipeline = BlitPipeline::new(&device, surface_format);
        let mesh_pipelines = if mesh_shader_supported {
            log::info!("Compiling mesh shader pipelines...");
            Some(MeshForwardPipelines::new(&device, color_format))
        } else {
            None
        };
        let interval_pipelines = if mesh_shader_supported {
            log::info!("Compiling interval shading pipelines...");
            Some(IntervalPipelines::new(&device, color_format))
        } else {
            None
        };
        let compute_interval_pipelines = ComputeIntervalPipelines::new(&device, color_format);
        log::info!("Pipelines compiled: {:.2}s", t0.elapsed().as_secs_f64());

        log::info!("Uploading scene buffers ({} tets)...", self.scene_data.tet_count);
        let t0 = std::time::Instant::now();
        let buffers = SceneBuffers::upload(&device, &queue, &self.scene_data);

        // Allocate base_colors with zeros — GPU SH eval will fill them
        let zero_colors = vec![0.0f32; self.scene_data.tet_count as usize * 3];
        let material = MaterialBuffers::upload(
            &device,
            &zero_colors,
            &self.scene_data.color_grads,
            self.scene_data.tet_count,
        );

        // Upload SH coefficients to GPU as f16-packed u32 array
        let sh_total_dims = ((self.sh_coeffs.degree + 1) * (self.sh_coeffs.degree + 1)) as usize * 3;
        let sh_coeffs_packed = pack_sh_coeffs_f16(&self.sh_coeffs.coeffs, sh_total_dims);
        let sh_coeffs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sh_coeffs"),
            contents: bytemuck::cast_slice(&sh_coeffs_packed),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let sh_degree = self.sh_coeffs.degree;

        // Upload PBR aux data if available: [M * 8] f32 packed as [roughness, env_feat(4), albedo(3)]
        let aux_data_buf = if let Some(ref pbr) = self.pbr_data {
            let tet_count = self.scene_data.tet_count as usize;
            let mut aux = vec![0.0f32; tet_count * 8];
            for t in 0..tet_count {
                aux[t * 8] = if t < pbr.roughness.len() { pbr.roughness[t] } else { 0.5 };
                for c in 0..4 {
                    aux[t * 8 + 1 + c] = if t * 4 + c < pbr.env_feature.len() { pbr.env_feature[t * 4 + c] } else { 0.0 };
                }
                for c in 0..3 {
                    aux[t * 8 + 5 + c] = if t * 3 + c < pbr.albedo.len() { pbr.albedo[t * 3 + c] } else { 0.5 };
                }
            }
            // Override vertex normals from PBR data
            if !pbr.vertex_normals.is_empty() {
                queue.write_buffer(&buffers.vertex_normals, 0, bytemuck::cast_slice(&pbr.vertex_normals));
            }
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("aux_data"),
                contents: bytemuck::cast_slice(&aux),
                usage: wgpu::BufferUsages::STORAGE,
            });
            log::info!("Uploaded PBR aux data: {} tets × 8 channels", tet_count);
            Some(buf)
        } else {
            None
        };

        log::info!("Buffers uploaded: {:.2}s", t0.elapsed().as_secs_f64());

        let targets = RenderTargets::new(&device, size.width.max(1), size.height.max(1));

        // Radix sort state — use DRS (subgroup-based) when available, else Basic
        let sort_backend = if subgroup_supported {
            rmesh_sort::SortBackend::Drs
        } else {
            rmesh_sort::SortBackend::Basic
        };
        log::info!("Creating radix sort pipelines (backend: {:?})...", sort_backend);
        let t0 = std::time::Instant::now();
        let sort_pipelines = rmesh_sort::RadixSortPipelines::new(&device, 1, sort_backend);
        let n_pow2 = (self.scene_data.tet_count as u32).next_power_of_two();
        let sort_state = rmesh_sort::RadixSortState::new(&device, n_pow2, 32, 1, sort_backend);
        sort_state.upload_configs(&queue);
        let sort_state_16bit = rmesh_sort::RadixSortState::new(&device, n_pow2, 16, 1, sort_backend);
        sort_state_16bit.upload_configs(&queue);
        log::info!("Sort pipelines: {:.2}s", t0.elapsed().as_secs_f64());

        // Fluid simulation: lazily initialized when first enabled (saves ~500MB GPU memory)

        let compute_bg = create_compute_bind_group(&device, &pipelines, &buffers, &material, &sh_coeffs_buf);
        let hw_compute_bg = create_hw_compute_bind_group(&device, &pipelines, &buffers, &material, &sh_coeffs_buf);
        let render_bg = create_render_bind_group(&device, &pipelines, &buffers, &material);
        let render_bg_b = create_render_bind_group_with_sort_values(
            &device,
            &pipelines,
            &buffers,
            &material,
            sort_state.values_b(),
        );

        // Mesh shader bind groups (if supported)
        let (mesh_render_bg_a, mesh_render_bg_b, indirect_convert_bg) =
            if let Some(ref mp) = mesh_pipelines {
                let a = create_mesh_render_bind_group(&device, mp, &buffers, &material);
                let b = create_mesh_render_bind_group_with_sort_values(
                    &device, mp, &buffers, &material, sort_state.values_b(),
                );
                let ic = create_indirect_convert_bind_group(&device, mp, &buffers);
                (Some(a), Some(b), Some(ic))
            } else {
                (None, None, None)
            };

        // Interval shading bind groups (if supported)
        let (interval_render_bg_a, interval_render_bg_b, interval_indirect_convert_bg) =
            if let Some(ref ip) = interval_pipelines {
                let a = create_interval_render_bind_group(&device, ip, &buffers, &material);
                let b = create_interval_render_bind_group_with_sort_values(
                    &device, ip, &buffers, &material, sort_state.values_b(),
                );
                let ic = create_interval_indirect_convert_bind_group(&device, ip, &buffers);
                (Some(a), Some(b), Some(ic))
            } else {
                (None, None, None)
            };

        // Compute-interval bind groups (always available)
        let compute_interval_gen_bg_a = create_compute_interval_gen_bind_group(
            &device, &compute_interval_pipelines, &buffers, &material,
        );
        let compute_interval_gen_bg_b = create_compute_interval_gen_bind_group_with_sort_values(
            &device, &compute_interval_pipelines, &buffers, &material, sort_state.values_b(),
        );
        // 16-bit sort variant: gen bind groups use sort_state_16bit's values_b
        let compute_interval_gen_bg_a_16bit = create_compute_interval_gen_bind_group(
            &device, &compute_interval_pipelines, &buffers, &material,
        );
        let compute_interval_gen_bg_b_16bit = create_compute_interval_gen_bind_group_with_sort_values(
            &device, &compute_interval_pipelines, &buffers, &material, sort_state_16bit.values_b(),
        );
        let compute_interval_render_bg = if let Some(ref aux_buf) = aux_data_buf {
            rmesh_render::create_compute_interval_render_bind_group_pbr(
                &device, &compute_interval_pipelines, &buffers, aux_buf, &buffers.indices,
            )
        } else {
            create_compute_interval_render_bind_group(
                &device, &compute_interval_pipelines, &buffers,
            )
        };
        let compute_interval_convert_bg = create_compute_interval_indirect_convert_bind_group(
            &device, &compute_interval_pipelines, &buffers,
        );

        // Quad prepass + render bind groups (A/B for sort result location)
        let prepass_bg_a = create_prepass_bind_group(
            &device, &pipelines, &buffers, &material, &buffers.sort_values,
        );
        let prepass_bg_b = create_prepass_bind_group(
            &device, &pipelines, &buffers, &material, sort_state.values_b(),
        );
        let quad_render_bg = create_quad_render_bind_group(
            &device, &pipelines, &buffers,
        );

        let blit_bg = create_blit_bind_group(&device, &blit_pipeline, &targets.color_view);

        // Primitive setup (depth used for hardware early-z culling in forward pass)
        let primitive_geometry = PrimitiveGeometry::new(&device);
        let primitive_pipeline = PrimitivePipeline::new(&device);
        let primitive_targets = PrimitiveTargets::new(&device, size.width.max(1), size.height.max(1));

        // Instance count readback buffer
        let instance_count_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_count_readback"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // GPU timestamp profiling
        let ts_period_ns = queue.get_timestamp_period();
        let ts_query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("gpu_profiler"),
            ty: wgpu::QueryType::Timestamp,
            count: TS_QUERY_COUNT,
        });
        let ts_resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ts_resolve"),
            size: (TS_QUERY_COUNT as u64) * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let ts_readback = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(if i == 0 { "ts_readback_a" } else { "ts_readback_b" }),
                size: (TS_QUERY_COUNT as u64) * 8,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        });
        let ts_readback_ready: [std::sync::Arc<std::sync::atomic::AtomicBool>; 2] =
            std::array::from_fn(|_| std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)));
        let ts_readback_mapped: [std::sync::Arc<std::sync::atomic::AtomicBool>; 2] =
            std::array::from_fn(|_| std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)));

        // egui setup
        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            surface_format,
            egui_wgpu::RendererOptions::default(),
        );
        let egui_state = egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        // Deferred PBR shading pipeline (only when PBR data is loaded)
        let has_pbr = self.pbr_data.is_some();
        let (deferred_pipeline, deferred_bg, deferred_output, deferred_output_view, deferred_blit_bg, deferred_dsm_dummy_bg) = if has_pbr {
            log::info!("Creating deferred PBR shading pipeline...");
            let dp = rmesh_render::DeferredShadePipeline::new(&device, color_format);
            let bg = rmesh_render::create_deferred_bind_group(&device, &dp, &targets, &primitive_targets.depth_view);
            // Dummy DSM bind group (1x1 atlas, no lights)
            let dummy_atlas = rmesh_dsm::DsmAtlas::new_dummy(&device);
            let dummy_dsm_bg = rmesh_render::create_deferred_dsm_bind_group(
                &device, &dp, &dummy_atlas.fourier_array_views, &dummy_atlas.meta_buf,
            );
            // Separate output texture (can't read+write color_view in same pass)
            let out_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("deferred_output"),
                size: wgpu::Extent3d { width: size.width.max(1), height: size.height.max(1), depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: color_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let out_view = out_tex.create_view(&wgpu::TextureViewDescriptor::default());
            let d_blit_bg = create_blit_bind_group(&device, &blit_pipeline, &out_view);
            (Some(dp), Some(bg), Some(out_tex), Some(out_view), Some(d_blit_bg), Some(dummy_dsm_bg))
        } else {
            (None, None, None, None, None, None)
        };

        // Ray trace pipeline
        log::info!("Building ray trace data...");
        let t0 = std::time::Instant::now();
        let rt_neighbors = rmesh_render::compute_tet_neighbors(
            &self.scene_data.indices,
            self.scene_data.tet_count as usize,
        );
        let rt_bvh = rmesh_render::build_boundary_bvh(
            &self.scene_data.vertices,
            &self.scene_data.indices,
            &rt_neighbors,
            self.scene_data.tet_count as usize,
        );
        let rt_pipeline = rmesh_render::RayTracePipeline::new(
            &device,
            size.width.max(1),
            size.height.max(1),
            0,
        );
        let rt_buffers = rmesh_render::RayTraceBuffers::new(&device, &rt_neighbors, &rt_bvh);
        let start_tet = rmesh_render::find_containing_tet(
            &self.scene_data.vertices,
            &self.scene_data.indices,
            self.scene_data.tet_count as usize,
            self.camera.position,
        ).map(|t| t as i32).unwrap_or(-1);
        queue.write_buffer(&rt_buffers.start_tet, 0, bytemuck::cast_slice(&[start_tet]));
        self.rt_neighbors_cpu = rt_neighbors;
        self.rt_start_tet_hint = start_tet;
        let rt_bg = rmesh_render::create_raytrace_bind_group(
            &device, &rt_pipeline, &buffers, &material, &rt_buffers,
        );
        let (rt_texture, rt_texture_view) = create_rt_texture(&device, size.width.max(1), size.height.max(1));
        let rt_blit_pipeline = rmesh_render::BlitPipelineNonFiltering::new(&device, surface_format);
        let rt_blit_bg = rmesh_render::create_blit_nf_bind_group(&device, &rt_blit_pipeline, &rt_texture_view);
        log::info!("Ray trace data: {:.2}s", t0.elapsed().as_secs_f64());

        // DSM debug view (Fourier deep shadow map from camera perspective)
        let dsm_pipeline = rmesh_dsm::DsmPipeline::new(&device, color_format);
        let dsm_prim_pipeline = rmesh_dsm::DsmPrimitivePipeline::new(&device);
        let dsm_resolve_pipeline = rmesh_dsm::DsmResolvePipeline::new(&device, color_format);
        let (dsm_fourier_textures, dsm_fourier_views, dsm_depth_texture, dsm_depth_view,
             dsm_resolve_output, dsm_resolve_output_view) = {
            let w = size.width.max(1);
            let h = size.height.max(1);
            let make_fourier_tex = |idx: usize| {
                let tex = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(&format!("dsm_fourier_{idx}")),
                    size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                    mip_level_count: 1, sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: color_format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
                (tex, view)
            };
            let (ft0, fv0) = make_fourier_tex(0);
            let (ft1, fv1) = make_fourier_tex(1);
            let (ft2, fv2) = make_fourier_tex(2);
            let dtex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("dsm_debug_depth"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let dview = dtex.create_view(&wgpu::TextureViewDescriptor::default());
            let resolve_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("dsm_resolve_output"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: color_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let resolve_view = resolve_tex.create_view(&wgpu::TextureViewDescriptor::default());
            ([ft0, ft1, ft2], [fv0, fv1, fv2], dtex, dview, resolve_tex, resolve_view)
        };
        let dsm_render_bg = rmesh_dsm::create_dsm_render_bind_group(
            &device, &dsm_pipeline,
            &buffers.uniforms,
            &buffers.interval_vertex_buf,
            &buffers.interval_tet_data_buf,
        );
        let dsm_fourier_view_refs: [&wgpu::TextureView; rmesh_dsm::FOURIER_MRT_COUNT] =
            std::array::from_fn(|i| &dsm_fourier_views[i]);
        let dsm_resolve_bg = rmesh_dsm::create_dsm_resolve_bind_group(
            &device, &dsm_resolve_pipeline, &dsm_fourier_view_refs,
        );
        let dsm_blit_bg = create_blit_bind_group(&device, &blit_pipeline, &dsm_resolve_output_view);

        log::info!("GPU init total: {:.2}s", t_total.elapsed().as_secs_f64());

        self.gpu = Some(GpuState {
            device,
            queue,
            surface,
            surface_config,
            pipelines,
            blit_pipeline,
            sort_pipelines,
            sort_state,
            sort_state_16bit,
            sort_backend,
            buffers,
            material_buffers: material,
            targets,
            compute_bg,
            hw_compute_bg,
            render_bg,
            render_bg_b,
            blit_bg,
            tet_count: self.scene_data.tet_count,
            sh_coeffs_buf,
            sh_degree,
            pending_reconfigure: false,
            mesh_pipelines,
            mesh_render_bg_a,
            mesh_render_bg_b,
            indirect_convert_bg,
            interval_pipelines,
            interval_render_bg_a,
            interval_render_bg_b,
            interval_indirect_convert_bg,
            compute_interval_pipelines,
            compute_interval_gen_bg_a,
            compute_interval_gen_bg_b,
            compute_interval_gen_bg_a_16bit,
            compute_interval_gen_bg_b_16bit,
            compute_interval_render_bg,
            compute_interval_convert_bg,
            prepass_bg_a,
            prepass_bg_b,
            quad_render_bg,
            fluid_sim: None,
            tet_neighbors_buf: None,
            primitive_geometry,
            primitive_pipeline,
            primitive_targets,
            egui_renderer,
            egui_state,
            instance_count_readback,
            visible_instance_count: 0,
            ts_query_set,
            ts_resolve_buf,
            ts_readback,
            ts_readback_ready,
            ts_readback_mapped,
            ts_frame: 0,
            ts_period_ns,
            gpu_times_ms: GpuTimings::default(),
            cpu_times_ms: CpuTimings::default(),
            instance_count_ready: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            instance_count_mapped: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            aux_data_buf,
            deferred_pipeline,
            deferred_bg,
            deferred_output,
            deferred_output_view,
            deferred_blit_bg,
            has_pbr_data: has_pbr,
            dsm_pipeline,
            dsm_prim_pipeline,
            dsm_resolve_pipeline,
            dsm_fourier_textures,
            dsm_fourier_views,
            dsm_depth_texture,
            dsm_depth_view,
            dsm_resolve_output,
            dsm_resolve_output_view,
            dsm_render_bg,
            dsm_resolve_bg,
            dsm_blit_bg,
            dsm_atlas: None,
            deferred_dsm_bg: None,
            deferred_dsm_dummy_bg: deferred_dsm_dummy_bg,
            rt_pipeline,
            rt_buffers,
            rt_bg,
            rt_texture,
            rt_texture_view,
            rt_blit_pipeline,
            rt_blit_bg,
        });
    }

    // render() and update_fps() are in render.rs
    // (The old ~720-line render body has been deleted from this file)

    fn load_file(&mut self, path: &std::path::Path) {
        log::info!("Loading: {}", path.display());
        let file_data = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                log::error!("Failed to read {}: {}", path.display(), e);
                return;
            }
        };

        let is_ply = path
            .extension()
            .map_or(false, |ext| ext.eq_ignore_ascii_case("ply"));

        let (scene, sh, pbr) = if is_ply {
            match rmesh_data::load_ply(&file_data) {
                Ok(r) => r,
                Err(e) => {
                    log::error!("Failed to parse PLY: {}", e);
                    return;
                }
            }
        } else {
            match rmesh_data::load_rmesh(&file_data)
                .or_else(|_| rmesh_data::load_rmesh_raw(&file_data))
            {
                Ok(r) => r,
                Err(e) => {
                    log::error!("Failed to parse scene: {}", e);
                    return;
                }
            }
        };

        log::info!(
            "Loaded: {} vertices, {} tets, SH degree {}",
            scene.vertex_count,
            scene.tet_count,
            sh.degree,
        );

        // Update scene data
        self.scene_data = scene;
        self.sh_coeffs = sh;
        self.pbr_data = pbr;
        self.deferred_enabled = self.pbr_data.is_some();
        self.loaded_path = Some(path.to_path_buf());

        // Reset camera
        let pos = Vec3::new(
            self.scene_data.start_pose[0],
            self.scene_data.start_pose[1],
            self.scene_data.start_pose[2],
        );
        if pos.length() > 0.001 {
            self.camera = Camera::new(pos);
        }

        // Rebuild GPU buffers
        if let Some(gpu) = &mut self.gpu {
            gpu.buffers = SceneBuffers::upload(&gpu.device, &gpu.queue, &self.scene_data);

            let zero_colors = vec![0.0f32; self.scene_data.tet_count as usize * 3];
            gpu.material_buffers = MaterialBuffers::upload(
                &gpu.device,
                &zero_colors,
                &self.scene_data.color_grads,
                self.scene_data.tet_count,
            );
            gpu.tet_count = self.scene_data.tet_count;

            // Recreate SH coeffs buffer (f16-packed)
            let sh_total_dims = ((self.sh_coeffs.degree + 1) * (self.sh_coeffs.degree + 1)) as usize * 3;
        let sh_coeffs_packed = pack_sh_coeffs_f16(&self.sh_coeffs.coeffs, sh_total_dims);
            gpu.sh_coeffs_buf =
                gpu.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("sh_coeffs"),
                        contents: bytemuck::cast_slice(&sh_coeffs_packed),
                        usage: wgpu::BufferUsages::STORAGE,
                    });
            gpu.sh_degree = self.sh_coeffs.degree;

            // Recreate sort state for new tet count
            let n_pow2 = (gpu.tet_count as u32).next_power_of_two();
            gpu.sort_state = rmesh_sort::RadixSortState::new(&gpu.device, n_pow2, 32, 1, gpu.sort_backend);
            gpu.sort_state.upload_configs(&gpu.queue);
            gpu.sort_state_16bit = rmesh_sort::RadixSortState::new(&gpu.device, n_pow2, 16, 1, gpu.sort_backend);
            gpu.sort_state_16bit.upload_configs(&gpu.queue);

            // Recreate bind groups
            gpu.compute_bg = create_compute_bind_group(
                &gpu.device,
                &gpu.pipelines,
                &gpu.buffers,
                &gpu.material_buffers,
                &gpu.sh_coeffs_buf,
            );
            gpu.hw_compute_bg = create_hw_compute_bind_group(
                &gpu.device,
                &gpu.pipelines,
                &gpu.buffers,
                &gpu.material_buffers,
                &gpu.sh_coeffs_buf,
            );
            gpu.render_bg = create_render_bind_group(
                &gpu.device,
                &gpu.pipelines,
                &gpu.buffers,
                &gpu.material_buffers,
            );
            gpu.render_bg_b = create_render_bind_group_with_sort_values(
                &gpu.device,
                &gpu.pipelines,
                &gpu.buffers,
                &gpu.material_buffers,
                gpu.sort_state.values_b(),
            );

            // Recreate quad prepass + render bind groups
            gpu.prepass_bg_a = create_prepass_bind_group(
                &gpu.device, &gpu.pipelines, &gpu.buffers, &gpu.material_buffers,
                &gpu.buffers.sort_values,
            );
            gpu.prepass_bg_b = create_prepass_bind_group(
                &gpu.device, &gpu.pipelines, &gpu.buffers, &gpu.material_buffers,
                gpu.sort_state.values_b(),
            );
            gpu.quad_render_bg = create_quad_render_bind_group(
                &gpu.device, &gpu.pipelines, &gpu.buffers,
            );

            // Recreate mesh shader bind groups
            if let Some(ref mp) = gpu.mesh_pipelines {
                gpu.mesh_render_bg_a = Some(create_mesh_render_bind_group(
                    &gpu.device, mp, &gpu.buffers, &gpu.material_buffers,
                ));
                gpu.mesh_render_bg_b = Some(create_mesh_render_bind_group_with_sort_values(
                    &gpu.device, mp, &gpu.buffers, &gpu.material_buffers,
                    gpu.sort_state.values_b(),
                ));
                gpu.indirect_convert_bg = Some(create_indirect_convert_bind_group(
                    &gpu.device, mp, &gpu.buffers,
                ));
            }

            // Recreate interval shading bind groups
            if let Some(ref ip) = gpu.interval_pipelines {
                gpu.interval_render_bg_a = Some(create_interval_render_bind_group(
                    &gpu.device, ip, &gpu.buffers, &gpu.material_buffers,
                ));
                gpu.interval_render_bg_b = Some(create_interval_render_bind_group_with_sort_values(
                    &gpu.device, ip, &gpu.buffers, &gpu.material_buffers,
                    gpu.sort_state.values_b(),
                ));
                gpu.interval_indirect_convert_bg = Some(create_interval_indirect_convert_bind_group(
                    &gpu.device, ip, &gpu.buffers,
                ));
            }

            // Recreate compute-interval bind groups
            gpu.compute_interval_gen_bg_a = create_compute_interval_gen_bind_group(
                &gpu.device, &gpu.compute_interval_pipelines, &gpu.buffers, &gpu.material_buffers,
            );
            gpu.compute_interval_gen_bg_b = create_compute_interval_gen_bind_group_with_sort_values(
                &gpu.device, &gpu.compute_interval_pipelines, &gpu.buffers, &gpu.material_buffers,
                gpu.sort_state.values_b(),
            );
            gpu.compute_interval_gen_bg_a_16bit = create_compute_interval_gen_bind_group(
                &gpu.device, &gpu.compute_interval_pipelines, &gpu.buffers, &gpu.material_buffers,
            );
            gpu.compute_interval_gen_bg_b_16bit = create_compute_interval_gen_bind_group_with_sort_values(
                &gpu.device, &gpu.compute_interval_pipelines, &gpu.buffers, &gpu.material_buffers,
                gpu.sort_state_16bit.values_b(),
            );
            // Upload PBR aux data if available
            gpu.has_pbr_data = self.pbr_data.is_some();
            if let Some(ref pbr) = self.pbr_data {
                let tc = self.scene_data.tet_count as usize;
                let mut aux = vec![0.0f32; tc * 8];
                for t in 0..tc {
                    aux[t * 8] = if t < pbr.roughness.len() { pbr.roughness[t] } else { 0.5 };
                    for c in 0..4 {
                        aux[t * 8 + 1 + c] = if t * 4 + c < pbr.env_feature.len() { pbr.env_feature[t * 4 + c] } else { 0.0 };
                    }
                    for c in 0..3 {
                        aux[t * 8 + 5 + c] = if t * 3 + c < pbr.albedo.len() { pbr.albedo[t * 3 + c] } else { 0.5 };
                    }
                }
                if !pbr.vertex_normals.is_empty() {
                    gpu.queue.write_buffer(&gpu.buffers.vertex_normals, 0, bytemuck::cast_slice(&pbr.vertex_normals));
                }
                let aux_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("aux_data"),
                    contents: bytemuck::cast_slice(&aux),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                gpu.compute_interval_render_bg = rmesh_render::create_compute_interval_render_bind_group_pbr(
                    &gpu.device, &gpu.compute_interval_pipelines, &gpu.buffers, &aux_buf, &gpu.buffers.indices,
                );
                gpu.aux_data_buf = Some(aux_buf);

                // Recreate deferred pipeline
                let color_format = wgpu::TextureFormat::Rgba16Float;
                let dp = rmesh_render::DeferredShadePipeline::new(&gpu.device, color_format);
                gpu.deferred_bg = Some(rmesh_render::create_deferred_bind_group(&gpu.device, &dp, &gpu.targets, &gpu.primitive_targets.depth_view));
                // Reset DSM state (will be regenerated on next frame with lights)
                let dummy_atlas = rmesh_dsm::DsmAtlas::new_dummy(&gpu.device);
                gpu.deferred_dsm_dummy_bg = Some(rmesh_render::create_deferred_dsm_bind_group(
                    &gpu.device, &dp, &dummy_atlas.fourier_array_views, &dummy_atlas.meta_buf,
                ));
                gpu.deferred_dsm_bg = None;
                gpu.dsm_atlas = None;
                self.cached_dsm_lights.clear();
                self.cached_dsm_num_lights = 0;
                let w = gpu.surface_config.width;
                let h = gpu.surface_config.height;
                let out_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("deferred_output"),
                    size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                    mip_level_count: 1, sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: color_format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let out_view = out_tex.create_view(&wgpu::TextureViewDescriptor::default());
                gpu.deferred_blit_bg = Some(create_blit_bind_group(&gpu.device, &gpu.blit_pipeline, &out_view));
                gpu.deferred_output = Some(out_tex);
                gpu.deferred_output_view = Some(out_view);
                gpu.deferred_pipeline = Some(dp);
            } else {
                gpu.compute_interval_render_bg = create_compute_interval_render_bind_group(
                    &gpu.device, &gpu.compute_interval_pipelines, &gpu.buffers,
                );
                gpu.aux_data_buf = None;
                gpu.deferred_pipeline = None;
                gpu.deferred_bg = None;
                gpu.deferred_output = None;
                gpu.deferred_output_view = None;
                gpu.deferred_blit_bg = None;
            }
            gpu.compute_interval_convert_bg = create_compute_interval_indirect_convert_bind_group(
                &gpu.device, &gpu.compute_interval_pipelines, &gpu.buffers,
            );

            // Recreate fluid simulation
            let neighbors = rmesh_render::compute_tet_neighbors(
                &self.scene_data.indices,
                self.scene_data.tet_count as usize,
            );
            let tet_neighbors_buf =
                gpu.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("tet_neighbors"),
                        contents: bytemuck::cast_slice(&neighbors),
                        usage: wgpu::BufferUsages::STORAGE,
                    });
            let mut fluid_sim = FluidSim::new(&gpu.device, self.scene_data.tet_count);
            {
                let mut encoder =
                    gpu.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("precompute_geometry_reload"),
                        });
                fluid_sim.precompute_geometry(
                    &gpu.device,
                    &mut encoder,
                    &gpu.buffers.vertices,
                    &gpu.buffers.indices,
                    &tet_neighbors_buf,
                );
                gpu.queue.submit(std::iter::once(encoder.finish()));
            }
            fluid_sim.compute_mesh_bbox(&gpu.device, &gpu.queue);
            fluid_sim.log_precompute_stats(&gpu.device, &gpu.queue);
            self.fluid_params = fluid_sim.default_params_for_mesh();
            gpu.fluid_sim = Some(fluid_sim);
            gpu.tet_neighbors_buf = Some(tet_neighbors_buf);

            // Recreate ray trace state (reuse neighbors from fluid sim)
            let rt_bvh = rmesh_render::build_boundary_bvh(
                &self.scene_data.vertices,
                &self.scene_data.indices,
                &neighbors,
                self.scene_data.tet_count as usize,
            );
            let w = gpu.surface_config.width;
            let h = gpu.surface_config.height;
            gpu.rt_pipeline = rmesh_render::RayTracePipeline::new(&gpu.device, w, h, 0);
            gpu.rt_buffers = rmesh_render::RayTraceBuffers::new(&gpu.device, &neighbors, &rt_bvh);
            let start_tet = rmesh_render::find_containing_tet(
                &self.scene_data.vertices,
                &self.scene_data.indices,
                self.scene_data.tet_count as usize,
                self.camera.position,
            ).map(|t| t as i32).unwrap_or(-1);
            gpu.queue.write_buffer(&gpu.rt_buffers.start_tet, 0, bytemuck::cast_slice(&[start_tet]));
            self.rt_neighbors_cpu = neighbors.clone();
            self.rt_start_tet_hint = start_tet;
            gpu.rt_bg = rmesh_render::create_raytrace_bind_group(
                &gpu.device, &gpu.rt_pipeline, &gpu.buffers, &gpu.material_buffers, &gpu.rt_buffers,
            );
            let (rt_tex, rt_view) = create_rt_texture(&gpu.device, w, h);
            gpu.rt_blit_bg = rmesh_render::create_blit_nf_bind_group(&gpu.device, &gpu.rt_blit_pipeline, &rt_view);
            gpu.rt_texture = rt_tex;
            gpu.rt_texture_view = rt_view;
        }

        // Update window title
        if let Some(window) = &self.window {
            let name = path
                .file_name()
                .map_or("rmesh viewer".into(), |n| format!("rmesh viewer - {}", n.to_string_lossy()));
            window.set_title(&name);
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        let gpu = match &mut self.gpu {
            Some(g) => g,
            None => return,
        };
        if new_size.width > 0 && new_size.height > 0 {
            gpu.surface_config.width = new_size.width;
            gpu.surface_config.height = new_size.height;
            gpu.surface.configure(&gpu.device, &gpu.surface_config);
            gpu.targets = RenderTargets::new(&gpu.device, new_size.width, new_size.height);
            // Recreate blit bind group since color_view changed
            gpu.blit_bg = create_blit_bind_group(
                &gpu.device,
                &gpu.blit_pipeline,
                &gpu.targets.color_view,
            );

            // Recreate primitive depth target
            gpu.primitive_targets = PrimitiveTargets::new(&gpu.device, new_size.width, new_size.height);

            // Recreate deferred resources since MRT texture views changed
            if let Some(ref dp) = gpu.deferred_pipeline {
                gpu.deferred_bg = Some(rmesh_render::create_deferred_bind_group(
                    &gpu.device, dp, &gpu.targets, &gpu.primitive_targets.depth_view,
                ));
                let out_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("deferred_output"),
                    size: wgpu::Extent3d { width: new_size.width, height: new_size.height, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let out_view = out_tex.create_view(&wgpu::TextureViewDescriptor::default());
                gpu.deferred_blit_bg = Some(create_blit_bind_group(&gpu.device, &gpu.blit_pipeline, &out_view));
                gpu.deferred_output = Some(out_tex);
                gpu.deferred_output_view = Some(out_view);
            }

            // Recreate DSM debug textures
            {
                let w = new_size.width;
                let h = new_size.height;
                let color_format = wgpu::TextureFormat::Rgba16Float;
                for i in 0..rmesh_dsm::FOURIER_MRT_COUNT {
                    gpu.dsm_fourier_textures[i] = gpu.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some(&format!("dsm_fourier_{i}")),
                        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                        mip_level_count: 1, sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: color_format,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    });
                    gpu.dsm_fourier_views[i] = gpu.dsm_fourier_textures[i].create_view(&wgpu::TextureViewDescriptor::default());
                }
                gpu.dsm_depth_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("dsm_debug_depth"),
                    size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                    mip_level_count: 1, sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
                gpu.dsm_depth_view = gpu.dsm_depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
                gpu.dsm_resolve_output = gpu.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("dsm_resolve_output"),
                    size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                    mip_level_count: 1, sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: color_format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                gpu.dsm_resolve_output_view = gpu.dsm_resolve_output.create_view(&wgpu::TextureViewDescriptor::default());
                gpu.dsm_render_bg = rmesh_dsm::create_dsm_render_bind_group(
                    &gpu.device, &gpu.dsm_pipeline,
                    &gpu.buffers.uniforms,
                    &gpu.buffers.interval_vertex_buf,
                    &gpu.buffers.interval_tet_data_buf,
                );
                let fv_refs: [&wgpu::TextureView; rmesh_dsm::FOURIER_MRT_COUNT] =
                    std::array::from_fn(|i| &gpu.dsm_fourier_views[i]);
                gpu.dsm_resolve_bg = rmesh_dsm::create_dsm_resolve_bind_group(
                    &gpu.device, &gpu.dsm_resolve_pipeline, &fv_refs,
                );
                gpu.dsm_blit_bg = create_blit_bind_group(&gpu.device, &gpu.blit_pipeline, &gpu.dsm_resolve_output_view);
            }

            // Recreate ray trace pipeline + texture for new size
            gpu.rt_pipeline = rmesh_render::RayTracePipeline::new(
                &gpu.device, new_size.width, new_size.height, 0,
            );
            gpu.rt_bg = rmesh_render::create_raytrace_bind_group(
                &gpu.device, &gpu.rt_pipeline, &gpu.buffers, &gpu.material_buffers, &gpu.rt_buffers,
            );
            let (rt_tex, rt_view) = create_rt_texture(&gpu.device, new_size.width, new_size.height);
            gpu.rt_blit_bg = rmesh_render::create_blit_nf_bind_group(&gpu.device, &gpu.rt_blit_pipeline, &rt_view);
            gpu.rt_texture = rt_tex;
            gpu.rt_texture_view = rt_view;
        }
    }
}

/// Map winit KeyCode to InteractKey.
fn winit_key_to_interact(key: KeyCode) -> Option<InteractKey> {
    match key {
        KeyCode::KeyG => Some(InteractKey::G),
        KeyCode::KeyS => Some(InteractKey::S),
        KeyCode::KeyR => Some(InteractKey::R),
        KeyCode::KeyX => Some(InteractKey::X),
        KeyCode::KeyY => Some(InteractKey::Y),
        KeyCode::KeyZ => Some(InteractKey::Z),
        KeyCode::ShiftLeft | KeyCode::ShiftRight => Some(InteractKey::Shift),
        KeyCode::Enter | KeyCode::NumpadEnter => Some(InteractKey::Enter),
        KeyCode::Escape => Some(InteractKey::Escape),
        KeyCode::Backspace => Some(InteractKey::Backspace),
        KeyCode::Delete => Some(InteractKey::Delete),
        KeyCode::Tab => Some(InteractKey::Tab),
        _ => None,
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let title = self
            .loaded_path
            .as_ref()
            .and_then(|p| p.file_name())
            .map_or("rmesh viewer".into(), |n| {
                format!("rmesh viewer - {}", n.to_string_lossy())
            });
        let attrs = Window::default_attributes()
            .with_title(title)
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.init_gpu(window.clone());
        self.window = Some(window);
        log::info!(
            "Viewer initialized: {} tets, {} vertices (SH degree {})",
            self.scene_data.tet_count,
            self.scene_data.vertex_count,
            self.sh_coeffs.degree,
        );
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Feed events to egui first
        if let Some(gpu) = &mut self.gpu {
            if let Some(window) = &self.window {
                let _ = gpu.egui_state.on_window_event(window, &event);
            }
        }

        // Check if egui wants keyboard/pointer (from previous frame)
        let egui_wants_pointer = self.egui_ctx.egui_wants_pointer_input();
        let egui_wants_keyboard = self.egui_ctx.egui_wants_keyboard_input();

        // Build interaction context for the state machine
        let interact_ctx = {
            let (w, h) = self.gpu.as_ref().map_or((800, 600), |g| {
                (g.surface_config.width, g.surface_config.height)
            });
            let aspect = w as f32 / h as f32;
            InteractContext {
                view_matrix: self.camera.view_matrix(),
                proj_matrix: self.camera.projection_matrix(aspect),
                viewport_width: w as f32,
                viewport_height: h as f32,
            }
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput {
                event: ref key_event,
                ..
            } if !egui_wants_keyboard => {
                let PhysicalKey::Code(code) = key_event.physical_key else {
                    return;
                };

                // Track shift state for shift+left-click pan
                if matches!(code, KeyCode::ShiftLeft | KeyCode::ShiftRight) {
                    self.shift_pressed = key_event.state == ElementState::Pressed;
                }

                // Try interaction system first
                let mut consumed = false;
                if let Some(ikey) = winit_key_to_interact(code) {
                    let ie = match key_event.state {
                        ElementState::Pressed => InteractEvent::KeyDown(ikey),
                        ElementState::Released => InteractEvent::KeyUp(ikey),
                    };
                    let result = self.interaction.process_event(
                        &ie,
                        &mut self.primitives,
                        &interact_ctx,
                    );
                    consumed = result != InteractResult::NotConsumed;
                }

                // CharInput for numeric entry (only on press)
                if !consumed && key_event.state == ElementState::Pressed {
                    if let Some(ref text) = key_event.text {
                        for ch in text.chars() {
                            if matches!(ch, '0'..='9' | '.' | '-') {
                                let ie = InteractEvent::CharInput(ch);
                                let result = self.interaction.process_event(
                                    &ie,
                                    &mut self.primitives,
                                    &interact_ctx,
                                );
                                if result != InteractResult::NotConsumed {
                                    consumed = true;
                                }
                            }
                        }
                    }
                }

                // Fallback: Escape quits if not consumed by interaction
                if !consumed
                    && code == KeyCode::Escape
                    && key_event.state == ElementState::Pressed
                {
                    event_loop.exit();
                }
            }

            WindowEvent::Resized(size) => {
                self.resize(size);
            }

            WindowEvent::MouseInput { state, button, .. } if !egui_wants_pointer => {
                // Feed mouse buttons to interaction system
                let mb = match button {
                    MouseButton::Left => Some(rmesh_interact::MouseButton::Left),
                    MouseButton::Middle => Some(rmesh_interact::MouseButton::Middle),
                    MouseButton::Right => Some(rmesh_interact::MouseButton::Right),
                    _ => None,
                };
                if let Some(mb) = mb {
                    let ie = match state {
                        ElementState::Pressed => InteractEvent::MouseDown { button: mb },
                        ElementState::Released => InteractEvent::MouseUp { button: mb },
                    };
                    let result = self.interaction.process_event(
                        &ie,
                        &mut self.primitives,
                        &interact_ctx,
                    );
                    if result == InteractResult::NotConsumed {
                        // Only update camera button state if not consumed
                        match button {
                            MouseButton::Left => self.left_pressed = state == ElementState::Pressed,
                            MouseButton::Middle => self.middle_pressed = state == ElementState::Pressed,
                            MouseButton::Right => self.right_pressed = state == ElementState::Pressed,
                            _ => {}
                        }
                    }
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                let dx = position.x - self.last_mouse.0;
                let dy = position.y - self.last_mouse.1;

                if !egui_wants_pointer {
                    if self.interaction.is_active() {
                        // Feed mouse movement to interaction system
                        let ie = InteractEvent::MouseMove {
                            dx: dx as f32,
                            dy: dy as f32,
                        };
                        self.interaction.process_event(
                            &ie,
                            &mut self.primitives,
                            &interact_ctx,
                        );
                    } else {
                        if self.left_pressed && self.shift_pressed {
                            self.camera.pan(dx as f32, dy as f32);
                        } else if self.left_pressed {
                            self.camera.orbit(dx as f32, dy as f32);
                        }
                        if self.middle_pressed {
                            self.camera.pan(dx as f32, dy as f32);
                        }
                        if self.right_pressed {
                            self.camera.zoom(dy as f32);
                        }
                    }
                }

                self.last_mouse = (position.x, position.y);
            }

            WindowEvent::MouseWheel { delta, .. } if !egui_wants_pointer => {
                if !self.interaction.is_active() {
                    let scroll = match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                        winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                    };
                    self.camera.zoom(-scroll);
                }
            }

            WindowEvent::RedrawRequested => {
                self.render();
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            _ => {}
        }
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: rmesh-viewer <input.rmesh|input.ply>");
        std::process::exit(1);
    }

    let scene_path = PathBuf::from(&args[1]);
    log::info!("Loading: {}", scene_path.display());

    let file_data = std::fs::read(&scene_path)
        .with_context(|| format!("Failed to read {}", scene_path.display()))?;

    let is_ply = scene_path
        .extension()
        .map_or(false, |ext| ext.eq_ignore_ascii_case("ply"));

    let (scene, sh, pbr) = if is_ply {
        rmesh_data::load_ply(&file_data).context("Failed to parse PLY file")?
    } else {
        rmesh_data::load_rmesh(&file_data)
            .or_else(|_| rmesh_data::load_rmesh_raw(&file_data))
            .context("Failed to parse scene file")?
    };

    log::info!(
        "Scene: {} vertices, {} tets, SH degree {}, PBR: {}",
        scene.vertex_count,
        scene.tet_count,
        sh.degree,
        if pbr.is_some() { "yes" } else { "no" },
    );

    let event_loop = EventLoop::new().context("Failed to create event loop")?;
    let mut app = App::new(scene, sh, pbr);
    app.loaded_path = Some(scene_path);
    event_loop.run_app(&mut app)?;

    Ok(())
}
