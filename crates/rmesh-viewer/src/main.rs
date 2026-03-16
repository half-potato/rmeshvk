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
use glam::{Mat4, Vec3};
use std::path::PathBuf;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use rmesh_render::{
    create_compute_bind_group, create_render_bind_group,
    create_render_bind_group_with_sort_values, BlitPipeline, ForwardPipelines, MaterialBuffers,
    RenderTargets, SceneBuffers, Uniforms, create_blit_bind_group, record_blit,
};

/// SH degree-0 normalization constant.
const C0: f32 = 0.28209479;
/// SH degree-1 normalization constant.
const C1: f32 = 0.48860251;

/// Orbit camera matching the webrm Camera class.
struct Camera {
    position: Vec3,
    orbit_target: Vec3,
    orbit_distance: f32,
    orbit_yaw: f32,
    orbit_pitch: f32,
    fov_y: f32,
    near_z: f32,
    far_z: f32,
}

impl Camera {
    fn new(position: Vec3) -> Self {
        let distance = position.length();
        let pitch = (position.z / distance).asin();
        let yaw = position.y.atan2(position.x);

        Self {
            position,
            orbit_target: Vec3::ZERO,
            orbit_distance: distance,
            orbit_yaw: yaw,
            orbit_pitch: pitch,
            fov_y: 50.0_f32.to_radians(),
            near_z: 0.01,
            far_z: 1000.0,
        }
    }

    fn view_matrix(&mut self) -> Mat4 {
        let d = self.orbit_distance;
        let yaw = self.orbit_yaw;
        let pitch = self.orbit_pitch;

        let eye = self.orbit_target
            + Vec3::new(
                d * pitch.cos() * yaw.cos(),
                d * pitch.cos() * yaw.sin(),
                d * pitch.sin(),
            );

        self.position = eye;

        // Z-up look-at
        let up = Vec3::new(0.0, 0.0, -1.0);
        Mat4::look_at_rh(eye, self.orbit_target, up)
    }

    fn projection_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, aspect, self.near_z, self.far_z)
    }

    fn orbit(&mut self, dx: f32, dy: f32) {
        let sensitivity = 0.004;
        self.orbit_yaw -= dx * sensitivity;
        self.orbit_pitch += dy * sensitivity;
        let limit = std::f32::consts::FRAC_PI_2 - 0.001;
        self.orbit_pitch = self.orbit_pitch.clamp(-limit, limit);
    }

    fn zoom(&mut self, delta: f32) {
        self.orbit_distance = (self.orbit_distance + delta * 0.01).max(0.1);
    }

    fn pan(&mut self, dx: f32, dy: f32) {
        let sensitivity = 0.002 * self.orbit_distance;
        let yaw = self.orbit_yaw;
        let pitch = self.orbit_pitch;

        // Right vector (perpendicular to forward in XY plane)
        let right = Vec3::new(-yaw.sin(), yaw.cos(), 0.0);
        // Up vector (perpendicular to forward and right)
        let up = Vec3::new(
            -pitch.sin() * yaw.cos(),
            -pitch.sin() * yaw.sin(),
            pitch.cos(),
        );

        self.orbit_target += sensitivity * (-dx * right + dy * up);
    }
}

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
        }
    }

    /// Evaluate SH coefficients to base colors for the current camera position.
    fn evaluate_sh_colors(&self) -> Vec<f32> {
        let t_count = self.scene_data.tet_count as usize;
        let mut base_colors = vec![0.0f32; t_count * 3];
        let degree = self.sh_coeffs.degree;
        let stride = self.sh_coeffs.stride() as usize;
        let cam_pos = self.camera.position;

        for t in 0..t_count {
            let sh_base = t * stride;
            if sh_base + 2 >= self.sh_coeffs.coeffs.len() {
                // Fallback: gray
                base_colors[t * 3] = 0.5;
                base_colors[t * 3 + 1] = 0.5;
                base_colors[t * 3 + 2] = 0.5;
                continue;
            }

            // DC component (degree 0)
            let mut r = C0 * self.sh_coeffs.coeffs[sh_base];
            let mut g = C0 * self.sh_coeffs.coeffs[sh_base + 1];
            let mut b = C0 * self.sh_coeffs.coeffs[sh_base + 2];

            if degree >= 1 && stride >= 12 {
                // Compute direction from circumcenter to camera
                let cx = self.scene_data.circumdata[t * 4];
                let cy = self.scene_data.circumdata[t * 4 + 1];
                let cz = self.scene_data.circumdata[t * 4 + 2];
                let dir = (cam_pos - Vec3::new(cx, cy, cz)).normalize_or_zero();
                let x = dir.x;
                let y = dir.y;
                let z = dir.z;

                // SH degree 1: coeffs layout is [dc_r, dc_g, dc_b, y_r, y_g, y_b, z_r, z_g, z_b, x_r, x_g, x_b]
                r += C1 * (y * self.sh_coeffs.coeffs[sh_base + 3]
                    + z * self.sh_coeffs.coeffs[sh_base + 6]
                    + x * self.sh_coeffs.coeffs[sh_base + 9]);
                g += C1 * (y * self.sh_coeffs.coeffs[sh_base + 4]
                    + z * self.sh_coeffs.coeffs[sh_base + 7]
                    + x * self.sh_coeffs.coeffs[sh_base + 10]);
                b += C1 * (y * self.sh_coeffs.coeffs[sh_base + 5]
                    + z * self.sh_coeffs.coeffs[sh_base + 8]
                    + x * self.sh_coeffs.coeffs[sh_base + 11]);
            }

            // Add 0.5 bias (SH DC is centered around 0, base_colors go through softplus in shader)
            base_colors[t * 3] = r + 0.5;
            base_colors[t * 3 + 1] = g + 0.5;
            base_colors[t * 3 + 2] = b + 0.5;
        }

        base_colors
    }

    fn init_gpu(&mut self, window: Arc<Window>) {
        let size = window.inner_size();
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

            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("rmesh device"),
                        required_features: wgpu::Features::SUBGROUP
                            | wgpu::Features::SHADER_FLOAT32_ATOMIC,
                        required_limits: wgpu::Limits::default(),
                        ..Default::default()
                    },
                )
                .await
                .expect("Failed to create device");

            (adapter, device, queue)
        });

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
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

        let pipelines = ForwardPipelines::new(&device, color_format);
        let blit_pipeline = BlitPipeline::new(&device, surface_format);

        let buffers = SceneBuffers::upload(&device, &queue, &self.scene_data);

        // Evaluate SH colors instead of flat gray
        let base_colors = self.evaluate_sh_colors();
        let material = MaterialBuffers::upload(
            &device,
            &base_colors,
            &self.scene_data.color_grads,
            self.scene_data.tet_count,
        );

        let targets = RenderTargets::new(&device, size.width.max(1), size.height.max(1));

        // Radix sort state
        let sort_pipelines = rmesh_sort::RadixSortPipelines::new(&device);
        let n_pow2 = (self.scene_data.tet_count as u32).next_power_of_two();
        let sort_state = rmesh_sort::RadixSortState::new(&device, n_pow2, 32);
        sort_state.upload_configs(&queue);

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
        });
    }

    fn render(&mut self) {
        let gpu = match &self.gpu {
            Some(g) => g,
            None => return,
        };

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

        let w = gpu.surface_config.width;
        let h = gpu.surface_config.height;
        let aspect = w as f32 / h as f32;

        let view_mat = self.camera.view_matrix();
        let proj_mat = self.camera.projection_matrix(aspect);
        let vp = proj_mat * view_mat;

        // Extract c2w rotation from inverse view matrix
        // RH convention: camera axes are x=right, y=up, z=backward
        // Pinhole convention: x=right, y=DOWN, z=FORWARD
        let inv_view = view_mat.inverse();
        let cam_right = inv_view.col(0).truncate();
        let cam_up = inv_view.col(1).truncate();
        let cam_back = inv_view.col(2).truncate();
        let c2w = glam::Mat3::from_cols(cam_right, -cam_up, -cam_back);

        // Intrinsics from FOV
        let f_val = 1.0 / (self.camera.fov_y / 2.0).tan();
        let fx = f_val * h as f32 / 2.0;
        let fy = f_val * h as f32 / 2.0;
        let intrinsics = [fx, fy, w as f32 / 2.0, h as f32 / 2.0];

        let vp_cols = vp.to_cols_array_2d();
        let pos = self.camera.position.to_array();

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

        // Upload view-dependent SH colors
        let base_colors = self.evaluate_sh_colors();
        gpu.queue.write_buffer(
            &gpu.material_buffers.base_colors,
            0,
            bytemuck::cast_slice(&base_colors),
        );

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("forward pass"),
            });

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

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
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
        let attrs = Window::default_attributes()
            .with_title("rmesh viewer")
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

            WindowEvent::MouseInput { state, button, .. } => match button {
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

                if self.left_pressed {
                    self.camera.orbit(dx as f32, dy as f32);
                }
                if self.middle_pressed {
                    self.camera.pan(dx as f32, dy as f32);
                }
                if self.right_pressed {
                    self.camera.zoom(dy as f32);
                }

                self.last_mouse = (position.x, position.y);
            }

            WindowEvent::MouseWheel { delta, .. } => {
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
        eprintln!("Usage: rmesh-viewer <input.rmesh>");
        std::process::exit(1);
    }

    let scene_path = PathBuf::from(&args[1]);
    log::info!("Loading: {}", scene_path.display());

    let file_data = std::fs::read(&scene_path)
        .with_context(|| format!("Failed to read {}", scene_path.display()))?;
    let (scene, sh) = rmesh_data::load_rmesh(&file_data)
        .or_else(|_| rmesh_data::load_rmesh_raw(&file_data))
        .context("Failed to parse scene file")?;

    log::info!(
        "Scene: {} vertices, {} tets, SH degree {}",
        scene.vertex_count,
        scene.tet_count,
        sh.degree,
    );

    let event_loop = EventLoop::new().context("Failed to create event loop")?;
    let mut app = App::new(scene, sh);
    event_loop.run_app(&mut app)?;

    Ok(())
}
