//! rmesh-viewer-web: WebAssembly viewer for .rmesh files.
//!
//! Exposes a `WebViewer` struct to JavaScript via wasm-bindgen.
//! Uses the same rendering pipeline as the native rmesh-viewer.

use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;

use rmesh_render::{
    create_blit_bind_group, create_compute_bind_group, create_render_bind_group,
    create_render_bind_group_with_sort_values, record_blit, BlitPipeline, ForwardPipelines,
    MaterialBuffers, RenderTargets, SceneBuffers, Uniforms,
};
use rmesh_util::camera::Camera;
use rmesh_util::sh_eval::{ShEvalPipeline, ShEvalUniforms};

use glam::Vec3;

#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).ok();
}

/// Internal state for a loaded scene (separated so `WebViewer` can exist without a scene).
#[allow(dead_code)]
struct SceneState {
    buffers: SceneBuffers,
    material: MaterialBuffers,
    compute_bg: wgpu::BindGroup,
    render_bg: wgpu::BindGroup,
    render_bg_b: wgpu::BindGroup,
    blit_bg: wgpu::BindGroup,
    sort_state: rmesh_sort::RadixSortState,
    sh_eval_bg: wgpu::BindGroup,
    sh_coeffs_buf: wgpu::Buffer,
    sh_eval_uniforms_buf: wgpu::Buffer,
    sh_degree: u32,
    tet_count: u32,
}

#[wasm_bindgen]
pub struct WebViewer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    camera: Camera,
    pipelines: ForwardPipelines,
    sort_pipelines: rmesh_sort::RadixSortPipelines,
    blit_pipeline: BlitPipeline,
    targets: RenderTargets,
    sh_eval_pipeline: ShEvalPipeline,
    scene: Option<SceneState>,
}

#[wasm_bindgen]
impl WebViewer {
    /// Create a viewer attached to a `<canvas>` element by its DOM id. Async — must `await` in JS.
    pub async fn create(canvas_id: &str) -> Result<WebViewer, JsValue> {
        let window = web_sys::window().ok_or("no window")?;
        let document = window.document().ok_or("no document")?;
        let canvas = document
            .get_element_by_id(canvas_id)
            .ok_or_else(|| format!("no element with id '{canvas_id}'"))?;
        let canvas: web_sys::HtmlCanvasElement = canvas
            .dyn_into()
            .map_err(|_| "element is not a canvas")?;

        let width = canvas.client_width().max(1) as u32;
        let height = canvas.client_height().max(1) as u32;
        canvas.set_width(width);
        canvas.set_height(height);

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });

        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|e| format!("create_surface: {e}"))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("request_adapter: {e}"))?;

        log::info!("GPU: {:?}", adapter.get_info().name);

        let adapter_limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("rmesh-web device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter_limits.clone(),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("request_device: {e}"))?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| !f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let color_format = wgpu::TextureFormat::Rgba16Float;

        log::info!("Compiling shader pipelines...");
        let pipelines = ForwardPipelines::new(&device, color_format);
        let blit_pipeline = BlitPipeline::new(&device, surface_format);
        let sort_pipelines = rmesh_sort::RadixSortPipelines::new(&device, 1);
        let sh_eval_pipeline = ShEvalPipeline::new(&device);
        let targets = RenderTargets::new(&device, width, height);

        let camera = Camera::new(Vec3::new(0.0, 3.0, -2.0));

        log::info!("WebViewer initialized ({width}×{height})");

        Ok(WebViewer {
            device,
            queue,
            surface,
            surface_config,
            camera,
            pipelines,
            sort_pipelines,
            blit_pipeline,
            targets,
            sh_eval_pipeline,
            scene: None,
        })
    }

    /// Load a `.rmesh` file from a `Uint8Array`. Call after construction.
    pub fn load_rmesh(&mut self, data: &[u8]) -> Result<(), JsValue> {
        let (scene_data, sh_coeffs) = rmesh_data::load_rmesh(data)
            .or_else(|_| rmesh_data::load_rmesh_raw(data))
            .map_err(|e| format!("parse error: {e}"))?;

        log::info!(
            "Loaded: {} vertices, {} tets, SH degree {}",
            scene_data.vertex_count,
            scene_data.tet_count,
            sh_coeffs.degree,
        );

        self.upload_scene(scene_data, sh_coeffs);
        Ok(())
    }

    /// Load a `.ply` file from a `Uint8Array`.
    pub fn load_ply(&mut self, data: &[u8]) -> Result<(), JsValue> {
        let (scene_data, sh_coeffs) =
            rmesh_data::load_ply(data).map_err(|e| format!("PLY parse error: {e}"))?;

        log::info!(
            "Loaded PLY: {} vertices, {} tets, SH degree {}",
            scene_data.vertex_count,
            scene_data.tet_count,
            sh_coeffs.degree,
        );

        self.upload_scene(scene_data, sh_coeffs);
        Ok(())
    }

    /// Render one frame. JS calls this from `requestAnimationFrame`.
    pub fn render(&mut self) -> Result<(), JsValue> {
        let scene = match &self.scene {
            Some(s) => s,
            None => return Ok(()), // nothing loaded yet
        };

        let w = self.surface_config.width;
        let h = self.surface_config.height;
        let aspect = w as f32 / h as f32;

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
            tet_count: scene.tet_count,
            step: 0,
            tile_size_u: 12,
            ray_mode: 0,
            min_t: 0.0,
            _pad1: [0; 5],
        };

        self.queue
            .write_buffer(&scene.buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

        // SH eval uniforms
        let sh_uniforms = ShEvalUniforms {
            cam_pos: [pos[0], pos[1], pos[2], 0.0],
            tet_count: scene.tet_count,
            sh_degree: scene.sh_degree,
            _pad: [0; 2],
        };
        self.queue.write_buffer(
            &scene.sh_eval_uniforms_buf,
            0,
            bytemuck::bytes_of(&sh_uniforms),
        );

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.surface_config);
                return Ok(());
            }
            Err(e) => return Err(format!("surface error: {e:?}").into()),
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("web forward pass"),
            });

        // GPU SH evaluation
        self.sh_eval_pipeline
            .record(&mut encoder, &scene.sh_eval_bg, scene.tet_count);

        // Sorted forward pass: compute → radix sort → HW render
        rmesh_render::record_sorted_forward_pass(
            &mut encoder,
            &self.device,
            &self.pipelines,
            &self.sort_pipelines,
            &scene.sort_state,
            &scene.buffers,
            &self.targets,
            &scene.compute_bg,
            &scene.render_bg,
            &scene.render_bg_b,
            scene.tet_count,
            &self.queue,
        );

        // Blit to swapchain
        record_blit(&mut encoder, &self.blit_pipeline, &scene.blit_bg, &view);

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    /// Resize the rendering surface (call when canvas size changes).
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);
        self.targets = RenderTargets::new(&self.device, width, height);

        // Recreate blit bind group since color_view changed
        if let Some(scene) = &mut self.scene {
            scene.blit_bg = create_blit_bind_group(
                &self.device,
                &self.blit_pipeline,
                &self.targets.color_view,
            );
        }
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.camera.orbit(dx, dy);
    }
    pub fn pan(&mut self, dx: f32, dy: f32) {
        self.camera.pan(dx, dy);
    }
    pub fn zoom(&mut self, delta: f32) {
        self.camera.zoom(delta);
    }

    /// Returns true if a scene is currently loaded.
    pub fn has_scene(&self) -> bool {
        self.scene.is_some()
    }
}

// Private helpers (not exported to JS)
impl WebViewer {
    fn upload_scene(&mut self, scene_data: rmesh_data::SceneData, sh_coeffs: rmesh_data::ShCoeffs) {
        let tet_count = scene_data.tet_count;

        // Reset camera to scene start pose
        let pos = Vec3::new(
            scene_data.start_pose[0],
            scene_data.start_pose[1],
            scene_data.start_pose[2],
        );
        if pos.length() > 0.001 {
            self.camera = Camera::new(pos);
        }

        let buffers = SceneBuffers::upload(&self.device, &self.queue, &scene_data);

        let zero_colors = vec![0.0f32; tet_count as usize * 3];
        let material = MaterialBuffers::upload(
            &self.device,
            &zero_colors,
            &scene_data.color_grads,
            tet_count,
        );

        let sh_coeffs_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sh_coeffs"),
                contents: bytemuck::cast_slice(&sh_coeffs.coeffs),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let sh_eval_uniforms_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sh_eval_uniforms"),
            size: std::mem::size_of::<ShEvalUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sh_eval_bg = self.sh_eval_pipeline.create_bind_group(
            &self.device,
            &sh_eval_uniforms_buf,
            &buffers.vertices,
            &buffers.indices,
            &sh_coeffs_buf,
            &material.color_grads,
            &material.base_colors,
        );

        let n_pow2 = (tet_count as u32).next_power_of_two();
        let sort_state = rmesh_sort::RadixSortState::new(&self.device, n_pow2, 32, 1);
        sort_state.upload_configs(&self.queue);

        let compute_bg =
            create_compute_bind_group(&self.device, &self.pipelines, &buffers, &material);
        let render_bg =
            create_render_bind_group(&self.device, &self.pipelines, &buffers, &material);
        let render_bg_b = create_render_bind_group_with_sort_values(
            &self.device,
            &self.pipelines,
            &buffers,
            &material,
            &sort_state.values_b,
        );
        let blit_bg = create_blit_bind_group(
            &self.device,
            &self.blit_pipeline,
            &self.targets.color_view,
        );

        self.scene = Some(SceneState {
            buffers,
            material,
            compute_bg,
            render_bg,
            render_bg_b,
            blit_bg,
            sort_state,
            sh_eval_bg,
            sh_coeffs_buf,
            sh_eval_uniforms_buf,
            sh_degree: sh_coeffs.degree,
            tet_count,
        });
    }
}
