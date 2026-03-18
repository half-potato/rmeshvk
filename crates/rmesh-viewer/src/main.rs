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
    InteractContext, InteractEvent, InteractKey, InteractResult, Primitive, PrimitiveKind,
    TransformInteraction,
};
use rmesh_sim::{FluidParams, FluidSim};
use rmesh_util::camera::Camera;
use rmesh_util::sh_eval::{ShEvalPipeline, ShEvalUniforms};
use wgpu::util::DeviceExt;

use rayon::prelude::*;
use rmesh_render::{
    create_compute_bind_group, create_hw_compute_bind_group, create_render_bind_group,
    create_render_bind_group_with_sort_values, BlitPipeline, ForwardPipelines, MaterialBuffers,
    MeshForwardPipelines, RenderTargets, SceneBuffers, Uniforms, create_blit_bind_group,
    create_indirect_convert_bind_group, create_mesh_render_bind_group,
    create_mesh_render_bind_group_with_sort_values, record_blit,
    create_prepass_bind_group, create_quad_render_bind_group,
};
use rmesh_compositor::{
    PrimitiveGeometry, PrimitivePipeline, PrimitiveTargets,
    record_primitive_pass,
};

// SH basis function constants — kept for CPU reference fallback (evaluate_sh_colors).
#[allow(dead_code)]
const C0: f32 = 0.28209479;
#[allow(dead_code)]
const C1: f32 = 0.48860251;
#[allow(dead_code)]
const C2: [f32; 5] = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
];
#[allow(dead_code)]
const C3: [f32; 7] = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
];

struct GpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    pipelines: ForwardPipelines,
    blit_pipeline: BlitPipeline,
    sort_pipelines: rmesh_sort::RadixSortPipelines,
    sort_state: rmesh_sort::RadixSortState,
    buffers: SceneBuffers,
    material_buffers: MaterialBuffers,
    targets: RenderTargets,
    compute_bg: wgpu::BindGroup,
    hw_compute_bg: wgpu::BindGroup,
    render_bg: wgpu::BindGroup,
    render_bg_b: wgpu::BindGroup,
    blit_bg: wgpu::BindGroup,
    tet_count: u32,
    // GPU SH evaluation
    sh_eval_pipeline: ShEvalPipeline,
    sh_eval_bg: wgpu::BindGroup,
    sh_coeffs_buf: wgpu::Buffer,
    sh_eval_uniforms_buf: wgpu::Buffer,
    sh_degree: u32,
    pending_reconfigure: bool,
    // Mesh shader (optional)
    mesh_pipelines: Option<MeshForwardPipelines>,
    mesh_render_bg_a: Option<wgpu::BindGroup>,
    mesh_render_bg_b: Option<wgpu::BindGroup>,
    indirect_convert_bg: Option<wgpu::BindGroup>,
    // Quad prepass bind groups (A/B for sort result location)
    prepass_bg_a: wgpu::BindGroup,
    prepass_bg_b: wgpu::BindGroup,
    // Quad render bind group (no A/B needed — only reads uniforms + precomputed)
    quad_render_bg: wgpu::BindGroup,
    // Fluid simulation
    fluid_sim: Option<FluidSim>,
    tet_neighbors_buf: Option<wgpu::Buffer>,
    // Primitives (depth-first compositing via hardware early-z)
    primitive_geometry: PrimitiveGeometry,
    primitive_pipeline: PrimitivePipeline,
    primitive_targets: PrimitiveTargets,
    // egui
    egui_renderer: egui_wgpu::Renderer,
    egui_state: egui_winit::State,
    // Instance count readback (for debugging)
    instance_count_readback: wgpu::Buffer,
    visible_instance_count: u32,
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
    use_mesh_shader: bool,
    mesh_shader_supported: bool,
    // Fluid simulation
    fluid_enabled: bool,
    fluid_params: FluidParams,
    // Rendering options
    show_primitives: bool,
    use_quad_shader: bool,
    // Transform interaction
    interaction: TransformInteraction,
    primitives: Vec<Primitive>,
    next_primitive_id: u32,
}

impl App {
    fn new(scene: rmesh_data::SceneData, sh: rmesh_data::ShCoeffs) -> Self {
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
            use_mesh_shader: false,
            mesh_shader_supported: false,
            fluid_enabled: false,
            fluid_params: FluidParams::default(),
            show_primitives: false,
            use_quad_shader: false,
            interaction: TransformInteraction::new(),
            primitives: Vec::new(),
            next_primitive_id: 1,
        }
    }

    /// CPU reference SH evaluation (kept as fallback, not called per-frame).
    #[allow(dead_code)]
    fn evaluate_sh_colors(&self) -> Vec<f32> {
        let t_count = self.scene_data.tet_count as usize;
        let degree = self.sh_coeffs.degree as usize;
        let nc = (degree + 1) * (degree + 1); // numCoeffs per channel
        let stride = nc * 3; // total floats per tet
        let cam_pos = self.camera.position;
        let sh = &self.sh_coeffs.coeffs;
        let verts = &self.scene_data.vertices;
        let indices = &self.scene_data.indices;
        let grads = &self.scene_data.color_grads;

        let mut base_colors = vec![0.0f32; t_count * 3];
        base_colors
            .par_chunks_mut(3)
            .enumerate()
            .for_each(|(t, rgb)| {
                let sh_base = t * stride;
                if sh_base + stride > sh.len() {
                    rgb[0] = 0.5;
                    rgb[1] = 0.5;
                    rgb[2] = 0.5;
                    return;
                }

                // Load v0 and centroid for gradient offset (matching webrm)
                let i0 = indices[t * 4] as usize;
                let i1 = indices[t * 4 + 1] as usize;
                let i2 = indices[t * 4 + 2] as usize;
                let i3 = indices[t * 4 + 3] as usize;
                let v0 = Vec3::new(verts[i0*3], verts[i0*3+1], verts[i0*3+2]);
                let v1 = Vec3::new(verts[i1*3], verts[i1*3+1], verts[i1*3+2]);
                let v2 = Vec3::new(verts[i2*3], verts[i2*3+1], verts[i2*3+2]);
                let v3 = Vec3::new(verts[i3*3], verts[i3*3+1], verts[i3*3+2]);
                let centroid = (v0 + v1 + v2 + v3) * 0.25;

                // Direction: centroid - camPos (matches webrm convention)
                let dir = (centroid - cam_pos).normalize_or_zero();
                let (x, y, z) = (dir.x, dir.y, dir.z);

                // Planar layout: sh[sh_base + channel * nc + coeff]
                // Evaluate per-channel SH
                for c in 0..3usize {
                    let ch_base = sh_base + c * nc;

                    // Degree 0: C0 * sh_dc
                    let mut val = C0 * sh[ch_base];

                    // Degree 1: -C1*y, +C1*z, -C1*x
                    if degree >= 1 {
                        val -= C1 * y * sh[ch_base + 1];
                        val += C1 * z * sh[ch_base + 2];
                        val -= C1 * x * sh[ch_base + 3];
                    }

                    // Degree 2
                    if degree >= 2 {
                        let xx = x * x;
                        let yy = y * y;
                        let zz = z * z;
                        let xy = x * y;
                        let yz = y * z;
                        let xz = x * z;
                        val += C2[0] * xy * sh[ch_base + 4];
                        val += C2[1] * yz * sh[ch_base + 5];
                        val += C2[2] * (2.0 * zz - xx - yy) * sh[ch_base + 6];
                        val += C2[3] * xz * sh[ch_base + 7];
                        val += C2[4] * (xx - yy) * sh[ch_base + 8];
                    }

                    // Degree 3
                    if degree >= 3 {
                        let xx = x * x;
                        let yy = y * y;
                        let zz = z * z;
                        val += C3[0] * y * (3.0 * xx - yy) * sh[ch_base + 9];
                        val += C3[1] * x * y * z * sh[ch_base + 10];
                        val += C3[2] * y * (4.0 * zz - xx - yy) * sh[ch_base + 11];
                        val += C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[ch_base + 12];
                        val += C3[4] * x * (4.0 * zz - xx - yy) * sh[ch_base + 13];
                        val += C3[5] * z * (xx - yy) * sh[ch_base + 14];
                        val += C3[6] * x * (xx - 3.0 * yy) * sh[ch_base + 15];
                    }

                    rgb[c] = val + 0.5;
                }

                // Add gradient offset at v0 (matching webrm: dot(grad, v0 - centroid))
                let grad = Vec3::new(grads[t*3], grads[t*3+1], grads[t*3+2]);
                let offset = grad.dot(v0 - centroid);
                rgb[0] += offset;
                rgb[1] += offset;
                rgb[2] += offset;

                // Softplus activation (beta=10), matching webrm:
                //   sp = 0.1 * log(1.0 + exp(10.0 * x))
                for c in 0..3usize {
                    rgb[c] = 0.1 * (1.0 + (10.0 * rgb[c]).exp()).ln();
                }
            });

        base_colors
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

        let (adapter, device, queue, mesh_shader_supported) = pollster::block_on(async {
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
            let subgroup_supported = adapter_features.contains(wgpu::Features::SUBGROUP)
                && backend != wgpu::Backend::Metal; // naga MSL doesn't support subgroups yet
            let mesh_shader_supported = subgroup_supported
                && adapter_features.contains(wgpu::Features::EXPERIMENTAL_MESH_SHADER);
            log::info!("Subgroup support: {}", subgroup_supported);
            log::info!("Mesh shader support: {}", mesh_shader_supported);

            let mut required_features = wgpu::Features::SHADER_FLOAT32_ATOMIC;
            if subgroup_supported {
                required_features |= wgpu::Features::SUBGROUP;
            }
            if mesh_shader_supported {
                required_features |= wgpu::Features::EXPERIMENTAL_MESH_SHADER;
            }

            let mut limits = wgpu::Limits::default();
            limits.max_storage_buffers_per_shader_stage = 16;
            limits.max_storage_buffer_binding_size = 1024 * 1024 * 1024; // 1 GB
            limits.max_buffer_size = 1024 * 1024 * 1024; // 1 GB

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

            (adapter, device, queue, mesh_shader_supported)
        });
        self.mesh_shader_supported = mesh_shader_supported;
        self.use_mesh_shader = mesh_shader_supported;

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
            desired_maximum_frame_latency: 2,
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

        // Upload SH coefficients to GPU (one-time, only changes on file reload)
        let sh_coeffs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sh_coeffs"),
            contents: bytemuck::cast_slice(&self.sh_coeffs.coeffs),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // SH eval uniforms buffer (updated each frame with cam_pos)
        let sh_eval_uniforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sh_eval_uniforms"),
            size: std::mem::size_of::<ShEvalUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        log::info!("Compiling SH eval pipeline...");
        let sh_eval_pipeline = ShEvalPipeline::new(&device);
        let sh_eval_bg = sh_eval_pipeline.create_bind_group(
            &device,
            &sh_eval_uniforms_buf,
            &buffers.vertices,
            &buffers.indices,
            &sh_coeffs_buf,
            &material.color_grads,
            &material.base_colors,
        );
        let sh_degree = self.sh_coeffs.degree;

        log::info!("Buffers uploaded: {:.2}s", t0.elapsed().as_secs_f64());

        let targets = RenderTargets::new(&device, size.width.max(1), size.height.max(1));

        // Radix sort state
        log::info!("Creating radix sort pipelines...");
        let t0 = std::time::Instant::now();
        let sort_pipelines = rmesh_sort::RadixSortPipelines::new(&device, 1);
        let n_pow2 = (self.scene_data.tet_count as u32).next_power_of_two();
        let sort_state = rmesh_sort::RadixSortState::new(&device, n_pow2, 32, 1);
        sort_state.upload_configs(&queue);
        log::info!("Sort pipelines: {:.2}s", t0.elapsed().as_secs_f64());

        // Fluid simulation: lazily initialized when first enabled (saves ~500MB GPU memory)

        let compute_bg = create_compute_bind_group(&device, &pipelines, &buffers, &material);
        let hw_compute_bg = create_hw_compute_bind_group(&device, &pipelines, &buffers, &material);
        let render_bg = create_render_bind_group(&device, &pipelines, &buffers, &material);
        let render_bg_b = create_render_bind_group_with_sort_values(
            &device,
            &pipelines,
            &buffers,
            &material,
            &sort_state.values_b,
        );

        // Mesh shader bind groups (if supported)
        let (mesh_render_bg_a, mesh_render_bg_b, indirect_convert_bg) =
            if let Some(ref mp) = mesh_pipelines {
                let a = create_mesh_render_bind_group(&device, mp, &buffers, &material);
                let b = create_mesh_render_bind_group_with_sort_values(
                    &device, mp, &buffers, &material, &sort_state.values_b,
                );
                let ic = create_indirect_convert_bind_group(&device, mp, &buffers);
                (Some(a), Some(b), Some(ic))
            } else {
                (None, None, None)
            };

        // Quad prepass + render bind groups (A/B for sort result location)
        let prepass_bg_a = create_prepass_bind_group(
            &device, &pipelines, &buffers, &material, &buffers.sort_values,
        );
        let prepass_bg_b = create_prepass_bind_group(
            &device, &pipelines, &buffers, &material, &sort_state.values_b,
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
            buffers,
            material_buffers: material,
            targets,
            compute_bg,
            hw_compute_bg,
            render_bg,
            render_bg_b,
            blit_bg,
            tet_count: self.scene_data.tet_count,
            sh_eval_pipeline,
            sh_eval_bg,
            sh_coeffs_buf,
            sh_eval_uniforms_buf,
            sh_degree,
            pending_reconfigure: false,
            mesh_pipelines,
            mesh_render_bg_a,
            mesh_render_bg_b,
            indirect_convert_bg,
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
        });
    }

    fn update_fps(&mut self) {
        let now = std::time::Instant::now();
        self.frame_times.push_back(now);
        // Keep a 1-second window
        while let Some(&front) = self.frame_times.front() {
            if now.duration_since(front).as_secs_f64() > 1.0 {
                self.frame_times.pop_front();
            } else {
                break;
            }
        }
        self.fps = self.frame_times.len() as f64;
    }

    fn render(&mut self) {
        self.update_fps();

        if self.gpu.is_none() {
            return;
        }

        let (w, h) = {
            let g = self.gpu.as_ref().unwrap();
            (g.surface_config.width, g.surface_config.height)
        };
        let aspect = w as f32 / h as f32;

        // Camera matrices (view_matrix updates self.camera.position)
        let view_mat = self.camera.view_matrix();
        let proj_mat = self.camera.projection_matrix(aspect);
        let vp = proj_mat * view_mat;

        let inv_view = view_mat.inverse();
        let cam_right = inv_view.col(0).truncate();
        let cam_up = inv_view.col(1).truncate();
        let cam_back = inv_view.col(2).truncate();
        let c2w = glam::Mat3::from_cols(cam_right, -cam_up, -cam_back);

        let f_val = 1.0 / (self.camera.fov_y / 2.0).tan();
        let fx = f_val * h as f32 / 2.0;
        let fy = f_val * h as f32 / 2.0;
        let intrinsics = [fx, fy, w as f32 / 2.0, h as f32 / 2.0];

        let vp_cols = vp.to_cols_array_2d();
        let pos = self.camera.position.to_array();
        let fps = self.fps;
        let visible_count = self.gpu.as_ref().map_or(0, |g| g.visible_instance_count);

        let gpu = self.gpu.as_mut().unwrap();

        // Apply deferred surface reconfigure (must happen before get_current_texture)
        if gpu.pending_reconfigure {
            gpu.pending_reconfigure = false;
            gpu.surface_config.present_mode = if self.vsync {
                wgpu::PresentMode::Fifo
            } else {
                wgpu::PresentMode::Mailbox
            };
            gpu.surface.configure(&gpu.device, &gpu.surface_config);
        }

        let output = match gpu.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                gpu.surface.configure(&gpu.device, &gpu.surface_config);
                return;
            }
            Err(e) => {
                log::error!("Surface error: {:?}", e);
                return;
            }
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let uniforms = Uniforms {
            vp_col0: vp_cols[0],
            vp_col1: vp_cols[1],
            vp_col2: vp_cols[2],
            vp_col3: vp_cols[3],
            c2w_col0: [c2w.col(0).x, c2w.col(0).y, c2w.col(0).z, 0.0],
            c2w_col1: [c2w.col(1).x, c2w.col(1).y, c2w.col(1).z, 0.0],
            c2w_col2: [c2w.col(2).x, c2w.col(2).y, c2w.col(2).z, 0.0],
            intrinsics,
            cam_pos_pad: [pos[0], pos[1], pos[2], 0.0],
            screen_width: w as f32,
            screen_height: h as f32,
            tet_count: gpu.tet_count,
            step: 0,
            tile_size_u: 12,
            ray_mode: 0,
            min_t: 0.0,
            _pad1: [0; 5],
        };

        gpu.queue
            .write_buffer(&gpu.buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

        // Update SH eval uniforms and dispatch GPU SH evaluation
        let sh_uniforms = ShEvalUniforms {
            cam_pos: [pos[0], pos[1], pos[2], 0.0],
            tet_count: gpu.tet_count,
            sh_degree: gpu.sh_degree,
            _pad: [0; 2],
        };
        gpu.queue.write_buffer(
            &gpu.sh_eval_uniforms_buf,
            0,
            bytemuck::bytes_of(&sh_uniforms),
        );

        // --- egui frame ---
        let window = self.window.as_ref().unwrap();
        let raw_input = gpu.egui_state.take_egui_input(window);
        let tet_count = gpu.tet_count;
        let mut open_file = false;
        let mut vsync = self.vsync;
        let mut use_mesh_shader = self.use_mesh_shader;
        let mesh_shader_supported = self.mesh_shader_supported;
        let mut fluid_enabled = self.fluid_enabled;
        let mut show_primitives = self.show_primitives;
        let mut use_quad_shader = self.use_quad_shader;
        let mut fluid_reset = false;
        let fluid_params = &mut self.fluid_params;
        let interact_display = self.interaction.display_info();
        let interact_selected = self.interaction.selected();
        let mut new_selected = interact_selected;
        let mut add_primitive: Option<PrimitiveKind> = None;
        let mut delete_selected = false;
        let primitives_ref = &self.primitives;
        // Get mesh bbox for dynamic slider ranges
        let (fluid_bbox_min, fluid_bbox_max) = if let Some(ref sim) = gpu.fluid_sim {
            (sim.mesh_bbox_min, sim.mesh_bbox_max)
        } else {
            ([-5.0; 3], [5.0; 3])
        };
        let fluid_extent = {
            let dx = fluid_bbox_max[0] - fluid_bbox_min[0];
            let dy = fluid_bbox_max[1] - fluid_bbox_min[1];
            let dz = fluid_bbox_max[2] - fluid_bbox_min[2];
            dx.max(dy).max(dz).max(0.1)
        };
        let margin = fluid_extent * 0.5;
        #[allow(deprecated)]
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
                egui::menu::bar(ui, |ui| {
                    ui.menu_button("File", |ui| {
                        if ui.button("Open...").clicked() {
                            open_file = true;
                            ui.close_menu();
                        }
                    });
                    ui.menu_button("Settings", |ui| {
                        ui.checkbox(&mut vsync, "Vsync");
                        ui.add_enabled(
                            mesh_shader_supported,
                            egui::Checkbox::new(&mut use_mesh_shader, "Mesh Shader"),
                        );
                        ui.checkbox(&mut use_quad_shader, "Quad Shader");
                        ui.checkbox(&mut fluid_enabled, "Fluid Sim");
                        ui.checkbox(&mut show_primitives, "Show Primitives");
                    });
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(format!(
                            "FPS: {:.0}  |  {} visible / {} tets  |  {}x{}",
                            fps, visible_count, tet_count, w, h
                        ));
                    });
                });
            });

            // Primitives panel
            egui::SidePanel::left("primitives_panel").default_width(160.0).show(ctx, |ui| {
                ui.heading("Primitives");
                ui.horizontal(|ui| {
                    if ui.button("Cube").clicked() {
                        add_primitive = Some(PrimitiveKind::Cube);
                    }
                    if ui.button("Sphere").clicked() {
                        add_primitive = Some(PrimitiveKind::Sphere);
                    }
                });
                ui.horizontal(|ui| {
                    if ui.button("Plane").clicked() {
                        add_primitive = Some(PrimitiveKind::Plane);
                    }
                    if ui.button("Cylinder").clicked() {
                        add_primitive = Some(PrimitiveKind::Cylinder);
                    }
                });
                ui.separator();
                for (i, prim) in primitives_ref.iter().enumerate() {
                    let selected = new_selected == Some(i);
                    if ui.selectable_label(selected, &prim.name).clicked() {
                        new_selected = if selected { None } else { Some(i) };
                    }
                }
                if new_selected.is_some() {
                    ui.separator();
                    if ui.button("Delete").clicked() {
                        delete_selected = true;
                    }
                }
            });

            // Transform HUD overlay (centered at bottom)
            if let Some(ref info) = interact_display {
                egui::TopBottomPanel::bottom("interact_hud").show(ctx, |ui| {
                    ui.horizontal_centered(|ui| {
                        ui.label(
                            egui::RichText::new(info.mode.label())
                                .size(16.0)
                                .strong(),
                        );
                        if let Some(axis_label) = info.axis.label() {
                            ui.label(
                                egui::RichText::new(axis_label)
                                    .size(16.0)
                                    .color(egui::Color32::from_rgb(100, 180, 255)),
                            );
                        }
                        if !info.numeric_text.is_empty() {
                            ui.label(
                                egui::RichText::new(&info.numeric_text)
                                    .size(16.0)
                                    .monospace(),
                            );
                        }
                    });
                });
            }

            // Fluid simulation panel (only shown when enabled)
            if fluid_enabled {
                egui::SidePanel::right("fluid_panel").default_width(200.0).show(ctx, |ui| {
                    ui.heading("Fluid Simulation");
                    if ui.button("Reset").clicked() {
                        fluid_reset = true;
                    }
                    ui.separator();
                    ui.add(egui::Slider::new(&mut fluid_params.dt, 0.001..=0.1).text("dt").logarithmic(true));
                    ui.add(egui::Slider::new(&mut fluid_params.viscosity, 0.0..=0.1).text("Viscosity").logarithmic(true));
                    ui.add(egui::Slider::new(&mut fluid_params.buoyancy, 0.0..=20.0).text("Buoyancy"));
                    ui.add(egui::Slider::new(&mut fluid_params.density_scale, 0.1..=100.0).text("Density Scale").logarithmic(true));
                    ui.separator();
                    ui.label("Source");
                    ui.add(egui::Slider::new(&mut fluid_params.source_pos[0], (fluid_bbox_min[0] - margin)..=(fluid_bbox_max[0] + margin)).text("X"));
                    ui.add(egui::Slider::new(&mut fluid_params.source_pos[1], (fluid_bbox_min[1] - margin)..=(fluid_bbox_max[1] + margin)).text("Y"));
                    ui.add(egui::Slider::new(&mut fluid_params.source_pos[2], (fluid_bbox_min[2] - margin)..=(fluid_bbox_max[2] + margin)).text("Z"));
                    ui.add(egui::Slider::new(&mut fluid_params.source_radius, 0.001..=(fluid_extent * 0.5)).text("Radius"));
                    ui.add(egui::Slider::new(&mut fluid_params.source_strength, 0.0..=50.0).text("Strength"));
                    ui.separator();
                    ui.label("Solver");
                    ui.add(egui::Slider::new(&mut fluid_params.diffuse_iterations, 1..=100).text("Diffuse Iters"));
                    ui.add(egui::Slider::new(&mut fluid_params.pressure_iterations, 1..=200).text("Pressure Iters"));
                });
            }
        });
        if open_file {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Scene", &["rmesh", "ply"])
                .pick_file()
            {
                self.pending_load = Some(path);
            }
        }
        if vsync != self.vsync {
            self.vsync = vsync;
            gpu.pending_reconfigure = true;
        }
        self.use_mesh_shader = use_mesh_shader;
        self.use_quad_shader = use_quad_shader;
        self.fluid_enabled = fluid_enabled;
        self.show_primitives = show_primitives;

        // Handle primitive add/delete/selection
        if let Some(kind) = add_primitive {
            let name = format!("{}.{:03}", kind.label(), self.next_primitive_id);
            self.next_primitive_id += 1;
            self.primitives.push(Primitive::new(kind, name));
            let idx = self.primitives.len() - 1;
            self.interaction.set_selected(Some(idx));
        } else if delete_selected {
            if let Some(idx) = self.interaction.selected() {
                if idx < self.primitives.len() {
                    self.primitives.remove(idx);
                    self.interaction.set_selected(None);
                }
            }
        } else {
            self.interaction.set_selected(new_selected);
        }

        if fluid_reset {
            if let Some(ref mut sim) = gpu.fluid_sim {
                sim.reset(&gpu.queue);
                log::info!("[fluid] Reset simulation state");
            }
        }
        gpu.egui_state
            .handle_platform_output(window, full_output.platform_output);

        let paint_jobs = self.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen_desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [w, h],
            pixels_per_point: full_output.pixels_per_point,
        };

        // Update egui textures
        for (id, image_delta) in &full_output.textures_delta.set {
            gpu.egui_renderer
                .update_texture(&gpu.device, &gpu.queue, *id, image_delta);
        }

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("forward pass"),
            });

        gpu.egui_renderer
            .update_buffers(&gpu.device, &gpu.queue, &mut encoder, &paint_jobs, &screen_desc);

        // GPU SH evaluation (replaces CPU evaluate_sh_colors + write_buffer)
        gpu.sh_eval_pipeline
            .record(&mut encoder, &gpu.sh_eval_bg, gpu.tet_count);

        // Fluid simulation step (if enabled)
        if self.fluid_enabled {
            // Lazy init: create FluidSim on first enable
            if gpu.fluid_sim.is_none() {
                log::info!("[fluid] Initializing fluid simulation...");
                let t0_fluid = std::time::Instant::now();
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
                let mut fluid_sim = FluidSim::new(&gpu.device, gpu.tet_count);
                {
                    let mut precompute_encoder =
                        gpu.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("precompute_geometry"),
                            });
                    fluid_sim.precompute_geometry(
                        &gpu.device,
                        &mut precompute_encoder,
                        &gpu.buffers.vertices,
                        &gpu.buffers.indices,
                        &tet_neighbors_buf,
                    );
                    gpu.queue.submit(std::iter::once(precompute_encoder.finish()));
                }
                fluid_sim.compute_mesh_bbox(&gpu.device, &gpu.queue);
                fluid_sim.log_precompute_stats(&gpu.device, &gpu.queue);
                // Auto-set fluid params to match mesh geometry
                self.fluid_params = fluid_sim.default_params_for_mesh();
                log::info!(
                    "[fluid] Auto-set source_pos=[{:.3},{:.3},{:.3}] radius={:.4} (mesh center=[{:.3},{:.3},{:.3}] extent={:.3})",
                    self.fluid_params.source_pos[0], self.fluid_params.source_pos[1], self.fluid_params.source_pos[2],
                    self.fluid_params.source_radius,
                    fluid_sim.mesh_center()[0], fluid_sim.mesh_center()[1], fluid_sim.mesh_center()[2],
                    fluid_sim.mesh_extent(),
                );
                log::info!(
                    "[fluid] Fluid sim initialized: {:.2}s",
                    t0_fluid.elapsed().as_secs_f64()
                );
                gpu.fluid_sim = Some(fluid_sim);
                gpu.tet_neighbors_buf = Some(tet_neighbors_buf);
            }

            if let (Some(ref mut sim), Some(ref neighbors_buf)) =
                (&mut gpu.fluid_sim, &gpu.tet_neighbors_buf)
            {
                let should_log = sim.step_count < 3 || sim.step_count % 60 == 0;
                sim.step(
                    &gpu.device,
                    &mut encoder,
                    &gpu.queue,
                    &self.fluid_params,
                    neighbors_buf,
                    &gpu.buffers.densities,
                    &gpu.material_buffers.base_colors,
                );
                // Log on first few frames + every 60 after that.
                // Must submit first since step() only records commands.
                if should_log {
                    // Submit the encoder so far, log, then create a fresh encoder
                    gpu.queue.submit(std::iter::once(encoder.finish()));
                    sim.log_step_stats(
                        &gpu.device,
                        &gpu.queue,
                        &gpu.buffers.densities,
                        &gpu.material_buffers.base_colors,
                    );
                    encoder = gpu.device.create_command_encoder(
                        &wgpu::CommandEncoderDescriptor {
                            label: Some("forward pass (post-fluid)"),
                        },
                    );
                }
            }
        }

        // 1. Primitive pass first: renders to volume color target + primitive depth target
        //    (clears both even if no primitives, so forward pass can use LoadOp::Load)
        {
            // Temporarily apply preview transform so the grabbed object follows the mouse
            let preview_restore = if self.show_primitives {
                if let Some(idx) = self.interaction.selected() {
                    let ctx = InteractContext {
                        view_matrix: view_mat,
                        proj_matrix: proj_mat,
                        viewport_width: w as f32,
                        viewport_height: h as f32,
                    };
                    if let Some(preview) = self.interaction.preview_transform(&ctx) {
                        if idx < self.primitives.len() {
                            let original = self.primitives[idx].transform;
                            self.primitives[idx].transform = preview;
                            Some((idx, original))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            record_primitive_pass(
                &mut encoder,
                &gpu.queue,
                &gpu.primitive_pipeline,
                &gpu.primitive_geometry,
                &gpu.targets.color_view,
                &gpu.primitive_targets.depth_view,
                if self.show_primitives { &self.primitives } else { &[] },
                &vp,
            );

            // Restore original transform after rendering
            if let Some((idx, original)) = preview_restore {
                self.primitives[idx].transform = original;
            }
        }

        // 2. Sorted forward pass: compute → radix sort → render
        //    Color uses LoadOp::Load (preserves primitive colors), depth test culls behind primitives
        if self.use_mesh_shader
            && gpu.mesh_pipelines.is_some()
            && gpu.mesh_render_bg_a.is_some()
        {
            rmesh_render::record_sorted_mesh_forward_pass(
                &mut encoder,
                &gpu.device,
                &gpu.pipelines,
                gpu.mesh_pipelines.as_ref().unwrap(),
                &gpu.sort_pipelines,
                &gpu.sort_state,
                &gpu.buffers,
                &gpu.targets,
                &gpu.compute_bg,
                gpu.mesh_render_bg_a.as_ref().unwrap(),
                gpu.mesh_render_bg_b.as_ref().unwrap(),
                gpu.indirect_convert_bg.as_ref().unwrap(),
                gpu.tet_count,
                &gpu.queue,
                &gpu.primitive_targets.depth_view,
                Some(&gpu.hw_compute_bg),
            );
        } else {
            rmesh_render::record_sorted_forward_pass(
                &mut encoder,
                &gpu.device,
                &gpu.pipelines,
                &gpu.sort_pipelines,
                &gpu.sort_state,
                &gpu.buffers,
                &gpu.targets,
                &gpu.compute_bg,
                &gpu.render_bg,
                &gpu.render_bg_b,
                gpu.tet_count,
                &gpu.queue,
                &gpu.primitive_targets.depth_view,
                Some(&gpu.hw_compute_bg),
                self.use_quad_shader,
                Some(&gpu.prepass_bg_a),
                Some(&gpu.prepass_bg_b),
                Some(&gpu.quad_render_bg),
            );
        }

        // Copy instance_count (offset 4 in DrawIndirectArgs) to readback buffer
        encoder.copy_buffer_to_buffer(
            &gpu.buffers.indirect_args, 4,
            &gpu.instance_count_readback, 0,
            4,
        );

        // 3. Blit directly — no compositor needed
        record_blit(&mut encoder, &gpu.blit_pipeline, &gpu.blit_bg, &view);

        // egui render pass (overlay on swapchain)
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            }).forget_lifetime();
            gpu.egui_renderer.render(&mut rpass, &paint_jobs, &screen_desc);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));

        // Readback visible instance count (blocking, but only 4 bytes)
        {
            let slice = gpu.instance_count_readback.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let data = slice.get_mapped_range();
            gpu.visible_instance_count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            drop(data);
            gpu.instance_count_readback.unmap();
        }

        // Free egui textures after submit
        for id in &full_output.textures_delta.free {
            gpu.egui_renderer.free_texture(id);
        }

        output.present();

        // Handle deferred file load
        if let Some(path) = self.pending_load.take() {
            self.load_file(&path);
        }
    }

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

        let (scene, sh) = if is_ply {
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

            // Recreate SH coeffs buffer and bind group
            gpu.sh_coeffs_buf =
                gpu.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("sh_coeffs"),
                        contents: bytemuck::cast_slice(&self.sh_coeffs.coeffs),
                        usage: wgpu::BufferUsages::STORAGE,
                    });
            gpu.sh_degree = self.sh_coeffs.degree;
            gpu.sh_eval_bg = gpu.sh_eval_pipeline.create_bind_group(
                &gpu.device,
                &gpu.sh_eval_uniforms_buf,
                &gpu.buffers.vertices,
                &gpu.buffers.indices,
                &gpu.sh_coeffs_buf,
                &gpu.material_buffers.color_grads,
                &gpu.material_buffers.base_colors,
            );

            // Recreate sort state for new tet count
            let n_pow2 = (gpu.tet_count as u32).next_power_of_two();
            gpu.sort_state = rmesh_sort::RadixSortState::new(&gpu.device, n_pow2, 32, 1);
            gpu.sort_state.upload_configs(&gpu.queue);

            // Recreate bind groups
            gpu.compute_bg = create_compute_bind_group(
                &gpu.device,
                &gpu.pipelines,
                &gpu.buffers,
                &gpu.material_buffers,
            );
            gpu.hw_compute_bg = create_hw_compute_bind_group(
                &gpu.device,
                &gpu.pipelines,
                &gpu.buffers,
                &gpu.material_buffers,
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
                &gpu.sort_state.values_b,
            );

            // Recreate quad prepass + render bind groups
            gpu.prepass_bg_a = create_prepass_bind_group(
                &gpu.device, &gpu.pipelines, &gpu.buffers, &gpu.material_buffers,
                &gpu.buffers.sort_values,
            );
            gpu.prepass_bg_b = create_prepass_bind_group(
                &gpu.device, &gpu.pipelines, &gpu.buffers, &gpu.material_buffers,
                &gpu.sort_state.values_b,
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
                    &gpu.sort_state.values_b,
                ));
                gpu.indirect_convert_bg = Some(create_indirect_convert_bind_group(
                    &gpu.device, mp, &gpu.buffers,
                ));
            }

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

    let (scene, sh) = if is_ply {
        rmesh_data::load_ply(&file_data).context("Failed to parse PLY file")?
    } else {
        rmesh_data::load_rmesh(&file_data)
            .or_else(|_| rmesh_data::load_rmesh_raw(&file_data))
            .context("Failed to parse scene file")?
    };

    log::info!(
        "Scene: {} vertices, {} tets, SH degree {}",
        scene.vertex_count,
        scene.tet_count,
        sh.degree,
    );

    let event_loop = EventLoop::new().context("Failed to create event loop")?;
    let mut app = App::new(scene, sh);
    app.loaded_path = Some(scene_path);
    event_loop.run_app(&mut app)?;

    Ok(())
}
