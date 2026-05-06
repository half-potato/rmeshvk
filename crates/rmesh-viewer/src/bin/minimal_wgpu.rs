use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    frame_count: u64,
}

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("minimal_wgpu")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600));
        let window = Arc::new(event_loop.create_window(attrs).unwrap());

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
                .expect("no adapter");
            let info = adapter.get_info();
            log::info!("adapter: {:?} backend={:?}", info.name, info.backend);

            let caps = surface.get_capabilities(&adapter);
            log::info!("surface formats: {:?}", caps.formats);
            log::info!("surface present_modes: {:?}", caps.present_modes);
            log::info!("surface alpha_modes: {:?}", caps.alpha_modes);

            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .expect("no device");
            (adapter, device, queue)
        });

        let caps = surface.get_capabilities(&adapter);
        // Match the viewer: prefer non-sRGB, AutoNoVsync, frame_latency=3
        let format = caps.formats.iter().copied().find(|f| !f.is_srgb()).unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 3,
        };
        log::info!("configuring surface: {}x{} fmt={:?} present={:?} alpha={:?}",
            config.width, config.height, config.format, config.present_mode, config.alpha_mode);
        surface.configure(&device, &config);
        log::info!("surface configured");

        self.window = Some(window.clone());
        self.state = Some(State { surface, device, queue, config, frame_count: 0 });
        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else { return; };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                state.config.width = size.width.max(1);
                state.config.height = size.height.max(1);
                state.surface.configure(&state.device, &state.config);
            }
            WindowEvent::RedrawRequested => {
                let t = std::time::Instant::now();
                let frame = match state.surface.get_current_texture() {
                    Ok(f) => f,
                    Err(e) => {
                        log::error!("frame {} acquire error: {:?} (waited {:.1}ms)",
                            state.frame_count, e, t.elapsed().as_secs_f64() * 1000.0);
                        if let Some(w) = &self.window { w.request_redraw(); }
                        return;
                    }
                };
                let dt = t.elapsed().as_secs_f64() * 1000.0;
                let view = frame.texture.create_view(&Default::default());
                let mut enc = state.device.create_command_encoder(&Default::default());
                {
                    let _rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("clear"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.1, g: 0.4, b: 0.7, a: 1.0 }),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                }
                state.queue.submit(std::iter::once(enc.finish()));
                frame.present();
                state.frame_count += 1;
                if state.frame_count <= 5 || state.frame_count % 60 == 0 {
                    log::info!("frame {} acquired in {:.1}ms", state.frame_count, dt);
                }
                if let Some(w) = &self.window { w.request_redraw(); }
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let event_loop = EventLoop::new().unwrap();
    let mut app = App { window: None, state: None };
    event_loop.run_app(&mut app).unwrap();
}
