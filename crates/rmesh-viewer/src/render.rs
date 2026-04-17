//! Per-frame rendering: GPU readback, egui UI, command encoding, submit/present.

use std::sync::Arc;
use wgpu::util::DeviceExt;

use rmesh_render::{Uniforms, record_blit};
use rmesh_compositor::record_primitive_pass;
use rmesh_interact::{InteractContext, Primitive, PrimitiveKind};
use rmesh_sim::FluidSim;

use crate::gpu_state::*;
use crate::App;

impl App {
    pub(crate) fn update_fps(&mut self) {
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

    pub(crate) fn render(&mut self) {
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
        let gpu_times = self.gpu.as_ref().map_or(GpuTimings::default(), |g| g.gpu_times_ms.clone());
        let cpu_times = self.gpu.as_ref().map_or(CpuTimings::default(), |g| g.cpu_times_ms.clone());

        let gpu = self.gpu.as_mut().unwrap();

        let t_frame_start = std::time::Instant::now();

        // --- Non-blocking readback of previous frame's data ---
        {
            use std::sync::atomic::Ordering;
            gpu.device.poll(wgpu::PollType::Poll).ok();

            // Read previous frame's GPU timestamps
            let prev = (gpu.ts_frame + 1) % 2;
            if gpu.ts_readback_ready[prev].load(Ordering::Acquire) {
                let slice = gpu.ts_readback[prev].slice(..);
                let data = slice.get_mapped_range();
                let ts: &[u64] = bytemuck::cast_slice(&data);
                let p = gpu.ts_period_ns as f64 / 1_000_000.0; // convert to ms

                let project = (ts[1].wrapping_sub(ts[0])) as f64 * p;
                let prepass = (ts[3].wrapping_sub(ts[2])) as f64 * p;
                let sort = if ts[2] > ts[1] && ts[2] < ts[4] {
                    (ts[2].wrapping_sub(ts[1])) as f64 * p
                } else {
                    (ts[4].wrapping_sub(ts[1])) as f64 * p
                };
                let render = (ts[5].wrapping_sub(ts[4])) as f64 * p;
                let sh = (ts[7].wrapping_sub(ts[6])) as f64 * p;
                let total = sh + project + sort + prepass + render;

                gpu.gpu_times_ms = GpuTimings {
                    sh_eval_ms: sh as f32,
                    project_ms: project as f32,
                    sort_ms: sort as f32,
                    prepass_ms: prepass as f32,
                    render_ms: render as f32,
                    total_ms: total as f32,
                };

                drop(data);
                gpu.ts_readback[prev].unmap();
                gpu.ts_readback_ready[prev].store(false, Ordering::Release);
                gpu.ts_readback_mapped[prev].store(false, Ordering::Release);
            }

            // Read previous frame's instance count
            if gpu.instance_count_ready.load(Ordering::Acquire) {
                let slice = gpu.instance_count_readback.slice(..);
                let data = slice.get_mapped_range();
                gpu.visible_instance_count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                drop(data);
                gpu.instance_count_readback.unmap();
                gpu.instance_count_ready.store(false, Ordering::Release);
                gpu.instance_count_mapped.store(false, Ordering::Release);
            }
        }
        let t_after_poll = std::time::Instant::now();

        // Apply deferred surface reconfigure (must happen before get_current_texture)
        if gpu.pending_reconfigure {
            gpu.pending_reconfigure = false;
            gpu.surface_config.present_mode = if self.vsync {
                wgpu::PresentMode::Fifo
            } else {
                wgpu::PresentMode::Immediate
            };
            log::info!("Reconfiguring surface: present_mode={:?} frame_latency={}",
                gpu.surface_config.present_mode,
                gpu.surface_config.desired_maximum_frame_latency);
            gpu.surface.configure(&gpu.device, &gpu.surface_config);
        }

        let t_before_acquire = std::time::Instant::now();
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
        let t_after_acquire = std::time::Instant::now();

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
            sh_degree: gpu.sh_degree,
            near_plane: self.camera.near_z,
            far_plane: self.camera.far_z,
            _pad1: [0; 2],
        };

        gpu.queue
            .write_buffer(&gpu.buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

        let t_before_egui = std::time::Instant::now();

        // --- egui frame ---
        let window = self.window.as_ref().unwrap();
        let raw_input = gpu.egui_state.take_egui_input(window);
        let tet_count = gpu.tet_count;
        let mut open_file = false;
        let mut vsync = self.vsync;
        let mut render_mode = self.render_mode;
        let mesh_shader_supported = self.mesh_shader_supported;
        let mut fluid_enabled = self.fluid_enabled;
        let mut show_primitives = self.show_primitives;
        let mut show_scene = self.show_scene;
        let mut sort_16bit = self.sort_16bit;
        let mut fluid_reset = false;
        let fluid_params = &mut self.fluid_params;
        let interact_display = self.interaction.display_info();
        let interact_selected = self.interaction.selected();
        let mut new_selected = interact_selected;
        let mut add_primitive: Option<PrimitiveKind> = None;
        let mut delete_selected = false;
        let mut add_shadow_test_scene = false;
        let primitives_ref = &self.primitives;
        let has_pbr = gpu.has_pbr_data;
        let rt_locate_ms = self.rt_locate_ms;
        let mut deferred_enabled = self.deferred_enabled;
        let mut ambient = self.ambient;
        let mut deferred_debug_mode = self.deferred_debug_mode;
        let mut dsm_query_depth = self.dsm_query_depth;
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
                        ui.label("Renderer");
                        ui.radio_value(&mut render_mode, RenderMode::Regular, "Regular");
                        ui.radio_value(&mut render_mode, RenderMode::Quad, "Quad");
                        if mesh_shader_supported {
                            ui.radio_value(&mut render_mode, RenderMode::MeshShader, "Mesh Shader");
                        }
                        ui.radio_value(&mut render_mode, RenderMode::IntervalShader, "Interval Shader");
                        ui.radio_value(&mut render_mode, RenderMode::RayTrace, "Ray Trace");
                        ui.checkbox(&mut fluid_enabled, "Fluid Sim");
                        ui.checkbox(&mut show_primitives, "Show Primitives");
                        ui.checkbox(&mut show_scene, "Show Scene");
                        if render_mode == RenderMode::IntervalShader {
                            ui.checkbox(&mut sort_16bit, "16-bit Sort");
                        }
                        ui.separator();
                        ui.add_enabled(has_pbr, egui::Checkbox::new(&mut deferred_enabled, "Deferred Shading"));
                        if deferred_enabled && has_pbr {
                            ui.add(egui::Slider::new(&mut ambient, 0.0..=1.0).text("Ambient"));
                            let debug_labels = [
                                "Final", "Raw Albedo", "True Albedo", "Normals",
                                "Roughness", "Env Feature", "Depth", "Specular",
                                "Diffuse", "Shadow", "Retro", "Lambda",
                                "Plaster", "Alpha", "Primitives", "DSM",
                            ];
                            ui.separator();
                            ui.label("Debug Layer");
                            for (i, label) in debug_labels.iter().enumerate() {
                                ui.radio_value(&mut deferred_debug_mode, i as u32, *label);
                            }
                            if deferred_debug_mode == 15 {
                                ui.separator();
                                ui.add(egui::Slider::new(&mut dsm_query_depth, self.camera.near_z..=self.camera.far_z)
                                    .logarithmic(true)
                                    .text("DSM Depth"));
                            }
                        }
                    });
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let rt_info = if render_mode == RenderMode::RayTrace {
                            format!("  |  Locate:{:.2}ms", rt_locate_ms)
                        } else {
                            String::new()
                        };
                        ui.label(format!(
                            "FPS: {:.0}  |  {} vis / {} tets  |  GPU: {:.1}ms (SH:{:.1} Proj:{:.1} Sort:{:.1} Pre:{:.1} Rend:{:.1})  |  CPU: {:.1}ms (Poll:{:.1} Acq:{:.1} Egui:{:.1} Enc:{:.1} Sub:{:.1} Pres:{:.1}){}",
                            fps, visible_count, tet_count,
                            gpu_times.total_ms,
                            gpu_times.sh_eval_ms,
                            gpu_times.project_ms,
                            gpu_times.sort_ms,
                            gpu_times.prepass_ms,
                            gpu_times.render_ms,
                            cpu_times.total_ms,
                            cpu_times.poll_readback_ms,
                            cpu_times.acquire_ms,
                            cpu_times.egui_ms,
                            cpu_times.encode_ms,
                            cpu_times.submit_ms,
                            cpu_times.present_ms,
                            rt_info,
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
                ui.horizontal(|ui| {
                    if ui.button("Point Light").clicked() {
                        add_primitive = Some(PrimitiveKind::PointLight);
                    }
                    if ui.button("Spot Light").clicked() {
                        add_primitive = Some(PrimitiveKind::SpotLight);
                    }
                });
                if ui.button("Add Shadow Test Scene").clicked() {
                    add_shadow_test_scene = true;
                }
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
        self.render_mode = render_mode;
        self.fluid_enabled = fluid_enabled;
        self.show_primitives = show_primitives;
        if show_scene != self.show_scene {
            // Invalidate DSM cache when scene visibility changes
            self.cached_dsm_lights.clear();
            self.cached_dsm_num_lights = 0;
        }
        self.show_scene = show_scene;
        self.sort_16bit = sort_16bit;
        self.deferred_enabled = deferred_enabled;
        self.ambient = ambient;
        self.deferred_debug_mode = deferred_debug_mode;
        self.dsm_query_depth = dsm_query_depth;

        // Handle shadow test scene
        if add_shadow_test_scene {
            use rmesh_interact::{Transform, PrimitiveKind};
            self.primitives.clear();
            self.interaction.set_selected(None);

            // Point light at origin
            let mut light = Primitive::new(PrimitiveKind::PointLight, "TestLight");
            light.transform.position = glam::Vec3::new(0.0, 0.0, 0.0);
            self.primitives.push(light);

            // Small occluder cube in +X direction
            let mut occluder = Primitive::new(PrimitiveKind::Cube, "Occluder_+X");
            occluder.transform.position = glam::Vec3::new(2.0, 0.0, 0.0);
            occluder.transform.scale = glam::Vec3::splat(0.5);
            self.primitives.push(occluder);

            // 6 large surface cubes in each axis direction
            let dirs: [(glam::Vec3, &str); 6] = [
                (glam::Vec3::X, "+X"),
                (glam::Vec3::NEG_X, "-X"),
                (glam::Vec3::Y, "+Y"),
                (glam::Vec3::NEG_Y, "-Y"),
                (glam::Vec3::Z, "+Z"),
                (glam::Vec3::NEG_Z, "-Z"),
            ];
            for (dir, label) in dirs {
                let mut wall = Primitive::new(PrimitiveKind::Cube, format!("Wall_{label}"));
                wall.transform.position = dir * 5.0;
                wall.transform.scale = glam::Vec3::splat(2.0);
                self.primitives.push(wall);
            }

            self.show_primitives = true;
            self.next_primitive_id = 10;
        }

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

        let t_after_egui = std::time::Instant::now();

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("forward pass"),
            });

        gpu.egui_renderer
            .update_buffers(&gpu.device, &gpu.queue, &mut encoder, &paint_jobs, &screen_desc);

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

            let mrt_views = if self.deferred_enabled && gpu.has_pbr_data {
                Some(rmesh_compositor::MrtViews {
                    aux0_view: &gpu.targets.aux0_view,
                    normals_view: &gpu.targets.normals_view,
                    albedo_view: &gpu.targets.depth_view,
                })
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
                mrt_views,
            );

            // Restore original transform after rendering
            if let Some((idx, original)) = preview_restore {
                self.primitives[idx].transform = original;
            }
        }

        // 2. Sorted forward pass: compute → radix sort → render
        //    Color uses LoadOp::Load (preserves primitive colors), depth test culls behind primitives
        let mrt_enabled = self.deferred_enabled && gpu.has_pbr_data;
        let skip_volume = !self.show_scene || (self.deferred_enabled && self.deferred_debug_mode == 14);
        if skip_volume && mrt_enabled {
            // Clear MRT targets when volume is skipped to avoid ghosting
            let clear_ops = wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            };
            let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mrt_clear"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment { view: &gpu.targets.color_view, resolve_target: None, ops: clear_ops, depth_slice: None }),
                    Some(wgpu::RenderPassColorAttachment { view: &gpu.targets.aux0_view, resolve_target: None, ops: clear_ops, depth_slice: None }),
                    Some(wgpu::RenderPassColorAttachment { view: &gpu.targets.normals_view, resolve_target: None, ops: clear_ops, depth_slice: None }),
                    Some(wgpu::RenderPassColorAttachment { view: &gpu.targets.depth_view, resolve_target: None, ops: clear_ops, depth_slice: None }),
                ],
                depth_stencil_attachment: None,
                ..Default::default()
            });
        }
        if !skip_volume { match self.render_mode {
            RenderMode::IntervalShader => {
                let (sort_st, gen_a, gen_b) = if self.sort_16bit {
                    (&gpu.sort_state_16bit, &gpu.compute_interval_gen_bg_a_16bit, &gpu.compute_interval_gen_bg_b_16bit)
                } else {
                    (&gpu.sort_state, &gpu.compute_interval_gen_bg_a, &gpu.compute_interval_gen_bg_b)
                };
                rmesh_render::record_sorted_compute_interval_forward_pass(
                    &mut encoder,
                    &gpu.device,
                    &gpu.pipelines,
                    &gpu.compute_interval_pipelines,
                    &gpu.sort_pipelines,
                    sort_st,
                    &gpu.buffers,
                    &gpu.targets,
                    &gpu.compute_bg,
                    gen_a,
                    gen_b,
                    &gpu.compute_interval_render_bg,
                    &gpu.compute_interval_convert_bg,
                    gpu.tet_count,
                    &gpu.queue,
                    &gpu.primitive_targets.depth_view,
                    Some(&gpu.hw_compute_bg),
                    Some(&gpu.ts_query_set),
                    self.sort_16bit,
                    mrt_enabled,
                );
            }
            RenderMode::MeshShader
                if gpu.mesh_pipelines.is_some()
                    && gpu.mesh_render_bg_a.is_some() =>
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
                    Some(&gpu.ts_query_set),
                    mrt_enabled,
                );
            }
            RenderMode::RayTrace => {
                // Update start_tet for current camera position
                let t_locate = std::time::Instant::now();
                let hint = if self.rt_start_tet_hint >= 0 { self.rt_start_tet_hint as usize } else { 0 };
                let new_start = rmesh_render::find_containing_tet_walk(
                    &self.scene_data.vertices,
                    &self.scene_data.indices,
                    &self.rt_neighbors_cpu,
                    self.scene_data.tet_count as usize,
                    self.camera.position,
                    hint,
                ).map(|t| t as i32).unwrap_or(-1);
                let locate_ms = t_locate.elapsed().as_secs_f32() * 1000.0;
                let alpha = 0.1f32;
                self.rt_locate_ms = self.rt_locate_ms * (1.0 - alpha) + locate_ms * alpha;
                self.rt_start_tet_hint = new_start;
                gpu.queue.write_buffer(&gpu.rt_buffers.start_tet, 0, bytemuck::cast_slice(&[new_start]));

                // Forward compute (SH eval + color computation)
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("rt_project_compute"),
                        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                            query_set: &gpu.ts_query_set,
                            beginning_of_pass_write_index: Some(0),
                            end_of_pass_write_index: Some(1),
                        }),
                    });
                    cpass.set_pipeline(&gpu.pipelines.compute_pipeline);
                    cpass.set_bind_group(0, &gpu.compute_bg, &[]);
                    let n_pow2 = gpu.tet_count.next_power_of_two();
                    let wgs = (n_pow2 + 63) / 64;
                    cpass.dispatch_workgroups(wgs.min(65535), ((wgs + 65534) / 65535).max(1), 1);
                }

                // Clear raytrace output
                encoder.clear_buffer(&gpu.rt_pipeline.rendered_image, 0, None);

                // Ray trace compute pass (timestamped)
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("raytrace"),
                        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                            query_set: &gpu.ts_query_set,
                            beginning_of_pass_write_index: Some(4),
                            end_of_pass_write_index: Some(5),
                        }),
                    });
                    cpass.set_pipeline(gpu.rt_pipeline.pipeline());
                    cpass.set_bind_group(0, &gpu.rt_bg, &[]);
                    cpass.set_bind_group(1, &gpu.rt_pipeline.aux_bind_group, &[]);
                    cpass.dispatch_workgroups((w + 7) / 8, (h + 7) / 8, 1);
                }

                // Copy raytrace buffer → Rgba32Float texture for blitting
                encoder.copy_buffer_to_texture(
                    wgpu::TexelCopyBufferInfo {
                        buffer: &gpu.rt_pipeline.rendered_image,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(w * 16), // 4 channels * 4 bytes/f32
                            rows_per_image: Some(h),
                        },
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &gpu.rt_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                );
            }
            _ => {
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
                    matches!(self.render_mode, RenderMode::Quad),
                    Some(&gpu.prepass_bg_a),
                    Some(&gpu.prepass_bg_b),
                    Some(&gpu.quad_render_bg),
                    Some(&gpu.ts_query_set),
                    mrt_enabled,
                );
            }
        } } // end skip_volume

        // Copy instance_count to readback — skip if buffer has a pending/completed map
        if !gpu.instance_count_mapped.load(std::sync::atomic::Ordering::Acquire) {
            encoder.copy_buffer_to_buffer(
                &gpu.buffers.indirect_args, 4,
                &gpu.instance_count_readback, 0,
                4,
            );
        }

        // Submit forward pass before deferred to avoid read-after-write stall on hw depth
        let use_deferred = self.deferred_enabled && gpu.deferred_pipeline.is_some();
        if use_deferred {
            gpu.queue.submit(std::iter::once(encoder.finish()));
            encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("deferred+blit"),
            });
        }

        // 3. Deferred shading pass (if enabled + PBR data present)
        if use_deferred {
            if let (Some(ref deferred), Some(ref bg), Some(ref out_view)) =
                (&gpu.deferred_pipeline, &gpu.deferred_bg, &gpu.deferred_output_view)
            {
                // Collect lights from PointLight primitives
                let mut gpu_lights = [rmesh_render::GpuLight::default(); rmesh_render::MAX_LIGHTS];
                let mut num_lights = 0u32;
                for prim in &self.primitives {
                    if (num_lights as usize) >= rmesh_render::MAX_LIGHTS {
                        break;
                    }
                    match prim.kind {
                        PrimitiveKind::PointLight => {
                            gpu_lights[num_lights as usize] = rmesh_render::GpuLight {
                                position: prim.transform.position.to_array(),
                                light_type: 0, // point
                                color: [1.0, 1.0, 1.0],
                                intensity: 1.0,
                                direction: [0.0, 0.0, -1.0],
                                inner_angle: 0.0,
                                outer_angle: 0.0,
                                _pad: [0.0; 3],
                            };
                            num_lights += 1;
                        }
                        PrimitiveKind::SpotLight => {
                            let forward = prim.transform.rotation * glam::Vec3::NEG_Z;
                            gpu_lights[num_lights as usize] = rmesh_render::GpuLight {
                                position: prim.transform.position.to_array(),
                                light_type: 1, // spot
                                color: [1.0, 1.0, 1.0],
                                intensity: 1.0,
                                direction: forward.to_array(),
                                inner_angle: 20.0_f32.to_radians(),
                                outer_angle: 35.0_f32.to_radians(),
                                _pad: [0.0; 3],
                            };
                            num_lights += 1;
                        }
                        _ => {}
                    }
                }

                // DSM shadows only for actual scene lights (before adding default camera light)
                let scene_light_count = num_lights;
                let scene_lights = &gpu_lights[..scene_light_count as usize];
                let lights_changed = scene_light_count != self.cached_dsm_num_lights
                    || scene_lights != &self.cached_dsm_lights[..];

                if lights_changed {
                    if scene_light_count == 0 {
                        // No scene lights → disable DSM
                        gpu.dsm_atlas = None;
                        gpu.deferred_dsm_bg = None;
                    } else {
                        // Filter out light primitives from DSM occluders
                        let dsm_primitives: Vec<_> = if self.show_primitives {
                            self.primitives.iter().filter(|p| {
                                !matches!(p.kind, PrimitiveKind::PointLight | PrimitiveKind::SpotLight)
                            }).cloned().collect()
                        } else {
                            Vec::new()
                        };

                        log::debug!("DSM: {} primitives for shadow generation", dsm_primitives.len());
                        // Reallocate DsmAtlas if light count or types changed
                        let need_realloc = gpu.dsm_atlas.as_ref().map_or(true, |atlas| {
                            atlas.num_lights != scene_light_count
                        });
                        if need_realloc {
                            let light_types: Vec<u32> = scene_lights.iter().map(|l| l.light_type).collect();
                            gpu.dsm_atlas = Some(rmesh_dsm::DsmAtlas::new(
                                &gpu.device, 512, &light_types,
                            ));
                        }

                        // Generate DSMs for scene lights only
                        if let Some(ref atlas) = gpu.dsm_atlas {
                            rmesh_dsm::generate_dsm_for_lights(
                                atlas,
                                &mut encoder,
                                &gpu.device,
                                &gpu.queue,
                                &gpu.dsm_pipeline,
                                &gpu.dsm_prim_pipeline,
                                &gpu.primitive_geometry,
                                &dsm_primitives,
                                &gpu.pipelines,
                                &gpu.compute_interval_pipelines,
                                &gpu.sort_pipelines,
                                &gpu.sort_state,
                                &gpu.buffers,
                                &gpu.material_buffers,
                                &gpu.sh_coeffs_buf,
                                scene_lights,
                                scene_light_count,
                                gpu.tet_count,
                                0.1,
                                20.0,
                                self.show_scene,
                            );
                            atlas.populate_metadata(
                                &gpu.queue, scene_lights,
                                0.1, 20.0,
                            );

                            // Create DSM bind group from atlas
                            gpu.deferred_dsm_bg = Some(rmesh_render::create_deferred_dsm_bind_group(
                                &gpu.device, deferred,
                                &atlas.fourier_array_views, &atlas.meta_buf,
                            ));
                        }
                    }

                    // Wait for DSM GPU work to complete before continuing
                    // (prevents shared buffer conflicts with the forward pass)
                    gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();

                    // Update cached state
                    self.cached_dsm_lights = scene_lights.to_vec();
                    self.cached_dsm_num_lights = scene_light_count;
                }

                // Upload lights
                gpu.queue.write_buffer(
                    &deferred.light_buf, 0,
                    bytemuck::cast_slice(&gpu_lights),
                );

                // Upload deferred uniforms — DSM only for scene lights
                let dsm_enabled = if gpu.dsm_atlas.is_some() { 1u32 } else { 0u32 };
                let deferred_uniforms = rmesh_render::DeferredUniforms {
                    inv_vp: vp.inverse().to_cols_array_2d(),
                    cam_pos: pos,
                    num_lights,
                    width: w,
                    height: h,
                    ambient: self.ambient,
                    debug_mode: self.deferred_debug_mode,
                    near_plane: self.camera.near_z,
                    far_plane: self.camera.far_z,
                    dsm_enabled,
                    _pad: 0.0,
                };
                gpu.queue.write_buffer(
                    &deferred.uniforms_buf, 0,
                    bytemuck::bytes_of(&deferred_uniforms),
                );

                // Record deferred shade pass — writes to separate output texture
                let dsm_bg = gpu.deferred_dsm_bg.as_ref()
                    .or(gpu.deferred_dsm_dummy_bg.as_ref())
                    .unwrap();
                rmesh_render::record_deferred_shade(
                    &mut encoder, deferred, bg, dsm_bg, out_view,
                );
            }
        }

        // 3b. DSM debug view
        let use_dsm_debug = self.deferred_enabled && self.deferred_debug_mode == 15;
        if use_dsm_debug {
            // Camera-perspective DSM debug (original mode 15)
            let fourier_views: [&wgpu::TextureView; rmesh_dsm::FOURIER_MRT_COUNT] =
                std::array::from_fn(|i| &gpu.dsm_fourier_views[i]);

            rmesh_dsm::record_dsm_primitive_pass(
                &mut encoder,
                &gpu.queue,
                &gpu.dsm_prim_pipeline,
                &gpu.primitive_geometry,
                &fourier_views,
                &gpu.dsm_depth_view,
                if self.show_primitives { &self.primitives } else { &[] },
                &vp,
                self.camera.near_z,
                self.camera.far_z,
                w,
                h,
            );

            rmesh_dsm::record_dsm_render(
                &mut encoder,
                &gpu.dsm_pipeline,
                &gpu.dsm_render_bg,
                &gpu.buffers.interval_fan_index_buf,
                &gpu.buffers.interval_args_buf,
                &fourier_views,
                &gpu.dsm_depth_view,
                w,
                h,
            );

            gpu.queue.write_buffer(
                &gpu.dsm_resolve_pipeline.uniforms_buf,
                0,
                bytemuck::bytes_of(&rmesh_dsm::DsmResolveUniforms {
                    z_query: self.dsm_query_depth,
                    near: self.camera.near_z,
                    far: self.camera.far_z,
                    _pad: 0.0,
                }),
            );
            rmesh_dsm::record_dsm_resolve(
                &mut encoder,
                &gpu.dsm_resolve_pipeline,
                &gpu.dsm_resolve_bg,
                &gpu.dsm_resolve_output_view,
            );
        }

        // 4. Blit to swapchain — from raytrace output, DSM debug, deferred output, or color_view
        if self.render_mode == RenderMode::RayTrace {
            rmesh_render::record_blit_nf(
                &mut encoder, &gpu.rt_blit_pipeline, &gpu.rt_blit_bg, &view,
            );
        } else if use_dsm_debug {
            record_blit(&mut encoder, &gpu.blit_pipeline, &gpu.dsm_blit_bg, &view);
        } else {
            let blit_bg = if use_deferred {
                gpu.deferred_blit_bg.as_ref().unwrap_or(&gpu.blit_bg)
            } else {
                &gpu.blit_bg
            };
            record_blit(&mut encoder, &gpu.blit_pipeline, blit_bg, &view);
        }

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

        // Resolve GPU timestamps and copy to readback buffer (skip if target still mapped/pending)
        let cur_rb = gpu.ts_frame % 2;
        let ts_buf_free = !gpu.ts_readback_mapped[cur_rb].load(std::sync::atomic::Ordering::Acquire);
        if ts_buf_free {
            encoder.resolve_query_set(
                &gpu.ts_query_set,
                0..TS_QUERY_COUNT,
                &gpu.ts_resolve_buf,
                0,
            );
            encoder.copy_buffer_to_buffer(
                &gpu.ts_resolve_buf, 0,
                &gpu.ts_readback[cur_rb], 0,
                (TS_QUERY_COUNT as u64) * 8,
            );
        }

        let t_before_submit = std::time::Instant::now();
        gpu.queue.submit(std::iter::once(encoder.finish()));
        let t_after_submit = std::time::Instant::now();

        // Async map of this frame's timestamp readback (will be ready next frame)
        if ts_buf_free {
            use std::sync::atomic::Ordering;
            gpu.ts_readback_mapped[cur_rb].store(true, Ordering::Release);
            let ready = gpu.ts_readback_ready[cur_rb].clone();
            gpu.ts_readback[cur_rb]
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    if result.is_ok() {
                        ready.store(true, Ordering::Release);
                    }
                });
        }
        // Always advance frame counter so we alternate slots
        gpu.ts_frame += 1;

        // Async map instance count readback (will be ready next frame)
        // Only map if we actually copied data (i.e., buffer wasn't already mapped)
        if !gpu.instance_count_mapped.load(std::sync::atomic::Ordering::Acquire) {
            use std::sync::atomic::Ordering;
            gpu.instance_count_mapped.store(true, Ordering::Release);
            let ready = gpu.instance_count_ready.clone();
            gpu.instance_count_readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    if result.is_ok() {
                        ready.store(true, Ordering::Release);
                    }
                });
        }

        // Free egui textures after submit
        for id in &full_output.textures_delta.free {
            gpu.egui_renderer.free_texture(id);
        }

        let t_before_present = std::time::Instant::now();
        output.present();
        let t_after_present = std::time::Instant::now();

        // EMA smoothing (alpha = 0.05 for stable readout)
        let alpha = 0.05f32;
        let mix = |old: f32, new: f32| old * (1.0 - alpha) + new * alpha;
        let prev = &gpu.cpu_times_ms;
        gpu.cpu_times_ms = CpuTimings {
            poll_readback_ms: mix(prev.poll_readback_ms, (t_after_poll - t_frame_start).as_secs_f32() * 1000.0),
            acquire_ms: mix(prev.acquire_ms, (t_after_acquire - t_before_acquire).as_secs_f32() * 1000.0),
            egui_ms: mix(prev.egui_ms, (t_after_egui - t_before_egui).as_secs_f32() * 1000.0),
            encode_ms: mix(prev.encode_ms, (t_before_submit - t_after_egui).as_secs_f32() * 1000.0),
            submit_ms: mix(prev.submit_ms, (t_after_submit - t_before_submit).as_secs_f32() * 1000.0),
            present_ms: mix(prev.present_ms, (t_after_present - t_before_present).as_secs_f32() * 1000.0),
            total_ms: mix(prev.total_ms, (t_after_present - t_frame_start).as_secs_f32() * 1000.0),
        };

        // Handle deferred file load
        if let Some(path) = self.pending_load.take() {
            self.load_file(&path);
        }
    }
}
