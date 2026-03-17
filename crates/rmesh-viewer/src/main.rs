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
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use rmesh_util::camera::Camera;
use rmesh_util::sh_eval::{ShEvalPipeline, ShEvalUniforms};
use wgpu::util::DeviceExt;

use rayon::prelude::*;
use rmesh_render::{
    create_compute_bind_group, create_render_bind_group,
    create_render_bind_group_with_sort_values, BlitPipeline, ForwardPipelines, MaterialBuffers,
    RenderTargets, SceneBuffers, Uniforms, create_blit_bind_group, record_blit,
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
    // egui
    egui_renderer: egui_wgpu::Renderer,
    egui_state: egui_winit::State,
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
    last_mouse: (f64, f64),
    // egui + FPS
    egui_ctx: egui::Context,
    frame_times: VecDeque<std::time::Instant>,
    fps: f64,
    loaded_path: Option<PathBuf>,
    pending_load: Option<PathBuf>,
    vsync: bool,
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
            last_mouse: (0.0, 0.0),
            egui_ctx: egui::Context::default(),
            frame_times: VecDeque::with_capacity(120),
            fps: 0.0,
            loaded_path: None,
            pending_load: None,
            vsync: true,
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

        let (adapter, device, queue) = pollster::block_on(async {
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: Some(&surface),
                    force_fallback_adapter: false,
                })
                .await
                .expect("No suitable GPU adapter found");

            log::info!("GPU: {:?}", adapter.get_info().name);

            let mut limits = wgpu::Limits::default();
            limits.max_storage_buffers_per_shader_stage = 16;
            limits.max_storage_buffer_binding_size = 1024 * 1024 * 1024; // 1 GB
            limits.max_buffer_size = 1024 * 1024 * 1024; // 1 GB

            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("rmesh device"),
                        required_features: wgpu::Features::SUBGROUP
                            | wgpu::Features::SHADER_FLOAT32_ATOMIC,
                        required_limits: limits,
                        ..Default::default()
                    },
                )
                .await
                .expect("Failed to create device");

            (adapter, device, queue)
        });

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

        let compute_bg = create_compute_bind_group(&device, &pipelines, &buffers, &material);
        let render_bg = create_render_bind_group(&device, &pipelines, &buffers, &material);
        let render_bg_b = create_render_bind_group_with_sort_values(
            &device,
            &pipelines,
            &buffers,
            &material,
            &sort_state.values_b,
        );

        let blit_bg = create_blit_bind_group(&device, &blit_pipeline, &targets.color_view);

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
            egui_renderer,
            egui_state,
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
                    });
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(format!(
                            "FPS: {:.0}  |  {} tets  |  {}x{}",
                            fps, tet_count, w, h
                        ));
                    });
                });
            });
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

        // Sorted forward pass: compute → radix sort → HW render
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
        );

        // Blit Rgba16Float render target to sRGB swapchain
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
        }
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

        // Check if egui wants the pointer (from previous frame)
        let egui_wants_pointer = self.egui_ctx.egui_wants_pointer_input();

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => event_loop.exit(),

            WindowEvent::Resized(size) => {
                self.resize(size);
            }

            WindowEvent::MouseInput { state, button, .. } if !egui_wants_pointer => match button {
                MouseButton::Left => {
                    self.left_pressed = state == ElementState::Pressed;
                }
                MouseButton::Middle => {
                    self.middle_pressed = state == ElementState::Pressed;
                }
                MouseButton::Right => {
                    self.right_pressed = state == ElementState::Pressed;
                }
                _ => {}
            },

            WindowEvent::CursorMoved { position, .. } => {
                let dx = position.x - self.last_mouse.0;
                let dy = position.y - self.last_mouse.1;

                if !egui_wants_pointer {
                    if self.left_pressed {
                        self.camera.orbit(dx as f32, dy as f32);
                    }
                    if self.middle_pressed {
                        self.camera.pan(dx as f32, dy as f32);
                    }
                    if self.right_pressed {
                        self.camera.zoom(dy as f32);
                    }
                }

                self.last_mouse = (position.x, position.y);
            }

            WindowEvent::MouseWheel { delta, .. } if !egui_wants_pointer => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                };
                self.camera.zoom(-scroll);
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
