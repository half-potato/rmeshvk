//! Eulerian fluid simulation on a fixed tetrahedral mesh.
//!
//! Implements incompressible Navier-Stokes via the Stable Fluids algorithm
//! (Stam 1999) discretized with finite volumes on the unstructured tet mesh.
//!
//! Per-timestep pipeline:
//!   1. Advect (velocity + density)
//!   2. Diffuse (Jacobi iterations)
//!   3. Forces (gravity, buoyancy, smoke injection)
//!   4. Divergence
//!   5. Pressure solve (Jacobi iterations)
//!   6. Project (subtract pressure gradient)
//!   7. To render (map density/velocity to rendering buffers)

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// Shader sources (embedded at compile time)
// ---------------------------------------------------------------------------

const PRECOMPUTE_WGSL: &str = include_str!("shaders/precompute_geometry.wgsl");
const ADVECT_WGSL: &str = include_str!("shaders/fluid_advect.wgsl");
const DIFFUSE_WGSL: &str = include_str!("shaders/fluid_diffuse.wgsl");
const FORCES_WGSL: &str = include_str!("shaders/fluid_forces.wgsl");
const DIVERGENCE_WGSL: &str = include_str!("shaders/fluid_divergence.wgsl");
const PRESSURE_WGSL: &str = include_str!("shaders/fluid_pressure.wgsl");
const PROJECT_WGSL: &str = include_str!("shaders/fluid_project.wgsl");
const TO_RENDER_WGSL: &str = include_str!("shaders/fluid_to_render.wgsl");

// ---------------------------------------------------------------------------
// FluidUniforms — matches WGSL struct layout (std430)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct FluidUniforms {
    pub dt: f32,
    pub viscosity: f32,
    pub tet_count: u32,
    pub jacobi_iter: u32,
    pub gravity: [f32; 4],
    pub source_pos: [f32; 4], // xyz + radius in w
    pub source_strength: f32,
    pub buoyancy: f32,
    pub density_scale: f32,
    pub _pad: u32,
}

// ---------------------------------------------------------------------------
// FluidParams — runtime parameters passed to step()
// ---------------------------------------------------------------------------

pub struct FluidParams {
    pub dt: f32,
    pub viscosity: f32,
    pub gravity: [f32; 3],
    pub source_pos: [f32; 3],
    pub source_radius: f32,
    pub source_strength: f32,
    pub buoyancy: f32,
    pub density_scale: f32,
    pub diffuse_iterations: u32,
    pub pressure_iterations: u32,
}

impl Default for FluidParams {
    fn default() -> Self {
        Self {
            dt: 0.016,
            viscosity: 0.0001,
            gravity: [0.0, -9.8, 0.0],
            source_pos: [0.0, -0.5, 0.0],
            source_radius: 0.1,
            source_strength: 5.0,
            buoyancy: 1.0,
            density_scale: 10.0,
            diffuse_iterations: 30,
            pressure_iterations: 60,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const WORKGROUP_SIZE: u32 = 256;

fn dispatch_count(tet_count: u32) -> u32 {
    (tet_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn create_storage_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(4), // wgpu requires non-zero
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn buf_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    layouts: &[&wgpu::BindGroupLayout],
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{}_layout", label)),
        bind_group_layouts: layouts,
        immediate_size: 0,
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

// ---------------------------------------------------------------------------
// FluidSim
// ---------------------------------------------------------------------------

pub struct FluidSim {
    // Pipelines
    precompute_pipeline: wgpu::ComputePipeline,
    advect_pipeline: wgpu::ComputePipeline,
    diffuse_pipeline: wgpu::ComputePipeline,
    forces_pipeline: wgpu::ComputePipeline,
    divergence_pipeline: wgpu::ComputePipeline,
    pressure_pipeline: wgpu::ComputePipeline,
    project_pipeline: wgpu::ComputePipeline,
    to_render_pipeline: wgpu::ComputePipeline,

    // Bind group layouts
    precompute_bgl_0: wgpu::BindGroupLayout,
    precompute_bgl_1: wgpu::BindGroupLayout,
    sim_geo_bgl: wgpu::BindGroupLayout,      // shared geometry BGL for sim shaders
    advect_state_bgl: wgpu::BindGroupLayout,
    diffuse_state_bgl: wgpu::BindGroupLayout,
    forces_state_bgl: wgpu::BindGroupLayout,
    divergence_state_bgl: wgpu::BindGroupLayout,
    pressure_state_bgl: wgpu::BindGroupLayout,
    project_state_bgl: wgpu::BindGroupLayout,
    to_render_bgl_0: wgpu::BindGroupLayout,
    to_render_bgl_1: wgpu::BindGroupLayout,
    to_render_bgl_2: wgpu::BindGroupLayout,

    // Precomputed geometry buffers
    pub tet_volumes: wgpu::Buffer,
    pub face_geo: wgpu::Buffer,
    pub tet_centers: wgpu::Buffer,

    // Simulation state (ping-pong pairs)
    velocity: [wgpu::Buffer; 2],
    density: [wgpu::Buffer; 2],
    pressure: [wgpu::Buffer; 2],
    divergence: wgpu::Buffer,

    // Uniforms
    uniforms_buf: wgpu::Buffer,

    tet_count: u32,
    pub step_count: u32,

    // Mesh bounding box (computed after precompute_geometry)
    pub mesh_bbox_min: [f32; 3],
    pub mesh_bbox_max: [f32; 3],
}

/// Statistics for a GPU buffer readback.
#[derive(Debug)]
pub struct BufferStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub nan_count: usize,
    pub inf_count: usize,
    pub nonzero_count: usize,
    pub count: usize,
}

impl std::fmt::Display for BufferStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "min={:.6} max={:.6} mean={:.6} nonzero={}/{} nan={} inf={}",
            self.min, self.max, self.mean,
            self.nonzero_count, self.count,
            self.nan_count, self.inf_count,
        )
    }
}

fn compute_stats(data: &[f32]) -> BufferStats {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut nan_count = 0;
    let mut inf_count = 0;
    let mut nonzero_count = 0;
    for &v in data {
        if v.is_nan() {
            nan_count += 1;
            continue;
        }
        if v.is_infinite() {
            inf_count += 1;
            continue;
        }
        if v != 0.0 {
            nonzero_count += 1;
        }
        if v < min { min = v; }
        if v > max { max = v; }
        sum += v as f64;
    }
    let count = data.len();
    let mean = if count > nan_count + inf_count {
        (sum / (count - nan_count - inf_count) as f64) as f32
    } else {
        0.0
    };
    BufferStats { min, max, mean, nan_count, inf_count, nonzero_count, count }
}

/// Read back a GPU buffer as f32 values.
fn readback_f32(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<f32> {
    let size = buffer.size();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("debug_readback_staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("debug_readback"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| { tx.send(result).unwrap(); });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    result
}

impl FluidSim {
    pub fn new(device: &wgpu::Device, tet_count: u32) -> Self {
        let m = tet_count as u64;

        // --- Buffers ---
        let tet_volumes = create_storage_buffer(device, "fluid_tet_volumes", m * 4);
        let face_geo = create_storage_buffer(device, "fluid_face_geo", m * 4 * 16); // M*4 vec4<f32>
        let tet_centers = create_storage_buffer(device, "fluid_tet_centers", m * 16); // M vec4<f32>

        let velocity = [
            create_storage_buffer(device, "fluid_velocity_0", m * 16),
            create_storage_buffer(device, "fluid_velocity_1", m * 16),
        ];
        let density = [
            create_storage_buffer(device, "fluid_density_0", m * 4),
            create_storage_buffer(device, "fluid_density_1", m * 4),
        ];
        let pressure = [
            create_storage_buffer(device, "fluid_pressure_0", m * 4),
            create_storage_buffer(device, "fluid_pressure_1", m * 4),
        ];
        let divergence = create_storage_buffer(device, "fluid_divergence", m * 4);

        let uniforms_buf = create_storage_buffer(
            device,
            "fluid_uniforms",
            std::mem::size_of::<FluidUniforms>() as u64,
        );

        // --- Bind group layouts ---

        // Precompute BG0: uniforms(ro), vertices(ro), indices(ro), tet_neighbors(ro)
        let precompute_bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("precompute_bgl_0"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
            ],
        });

        // Precompute BG1: tet_volumes(rw), face_geo(rw), tet_centers(rw)
        let precompute_bgl_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("precompute_bgl_1"),
            entries: &[
                storage_entry(0, false),
                storage_entry(1, false),
                storage_entry(2, false),
            ],
        });

        // Shared geometry BGL for sim shaders:
        // 0: uniforms(ro), 1: tet_neighbors(ro), 2: face_geo(ro), 3: tet_volumes(ro), 4: tet_centers(ro)
        let sim_geo_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sim_geo_bgl"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
            ],
        });

        // Advect state BG1: velocity(ro), density(ro), velocity_tmp(rw), density_tmp(rw)
        let advect_state_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("advect_state_bgl"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, false),
            ],
        });

        // Diffuse state BG1: velocity(ro), velocity_tmp(rw)
        let diffuse_state_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("diffuse_state_bgl"),
                entries: &[storage_entry(0, true), storage_entry(1, false)],
            });

        // Forces state BG1: velocity(rw), density(rw)
        let forces_state_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("forces_state_bgl"),
            entries: &[storage_entry(0, false), storage_entry(1, false)],
        });

        // Divergence state BG1: velocity(ro), divergence(rw)
        let divergence_state_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("divergence_state_bgl"),
                entries: &[storage_entry(0, true), storage_entry(1, false)],
            });

        // Pressure state BG1: pressure(ro), pressure_tmp(rw), divergence(ro)
        let pressure_state_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pressure_state_bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, false),
                    storage_entry(2, true),
                ],
            });

        // Project state BG1: velocity(rw), pressure(ro)
        let project_state_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("project_state_bgl"),
                entries: &[storage_entry(0, false), storage_entry(1, true)],
            });

        // To-render BG0: uniforms(ro)
        let to_render_bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("to_render_bgl_0"),
            entries: &[storage_entry(0, true)],
        });

        // To-render BG1: density(ro), velocity(ro)
        let to_render_bgl_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("to_render_bgl_1"),
            entries: &[storage_entry(0, true), storage_entry(1, true)],
        });

        // To-render BG2: out_densities(rw), out_colors(rw)
        let to_render_bgl_2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("to_render_bgl_2"),
            entries: &[storage_entry(0, false), storage_entry(1, false)],
        });

        // --- Pipelines ---
        // Note: sim shaders that don't need tet_centers can still bind to the 5-entry layout
        // since the shader simply won't access binding 4.

        let precompute_pipeline = create_compute_pipeline(
            device,
            "precompute_geometry",
            PRECOMPUTE_WGSL,
            &[&precompute_bgl_0, &precompute_bgl_1],
        );

        let advect_pipeline = create_compute_pipeline(
            device,
            "fluid_advect",
            ADVECT_WGSL,
            &[&sim_geo_bgl, &advect_state_bgl],
        );

        let diffuse_pipeline = create_compute_pipeline(
            device,
            "fluid_diffuse",
            DIFFUSE_WGSL,
            &[&sim_geo_bgl, &diffuse_state_bgl],
        );

        let forces_pipeline = create_compute_pipeline(
            device,
            "fluid_forces",
            FORCES_WGSL,
            &[&sim_geo_bgl, &forces_state_bgl],
        );

        let divergence_pipeline = create_compute_pipeline(
            device,
            "fluid_divergence",
            DIVERGENCE_WGSL,
            &[&sim_geo_bgl, &divergence_state_bgl],
        );

        let pressure_pipeline = create_compute_pipeline(
            device,
            "fluid_pressure",
            PRESSURE_WGSL,
            &[&sim_geo_bgl, &pressure_state_bgl],
        );

        let project_pipeline = create_compute_pipeline(
            device,
            "fluid_project",
            PROJECT_WGSL,
            &[&sim_geo_bgl, &project_state_bgl],
        );

        let to_render_pipeline = create_compute_pipeline(
            device,
            "fluid_to_render",
            TO_RENDER_WGSL,
            &[&to_render_bgl_0, &to_render_bgl_1, &to_render_bgl_2],
        );

        Self {
            precompute_pipeline,
            advect_pipeline,
            diffuse_pipeline,
            forces_pipeline,
            divergence_pipeline,
            pressure_pipeline,
            project_pipeline,
            to_render_pipeline,
            precompute_bgl_0,
            precompute_bgl_1,
            sim_geo_bgl,
            advect_state_bgl,
            diffuse_state_bgl,
            forces_state_bgl,
            divergence_state_bgl,
            pressure_state_bgl,
            project_state_bgl,
            to_render_bgl_0,
            to_render_bgl_1,
            to_render_bgl_2,
            tet_volumes,
            face_geo,
            tet_centers,
            velocity,
            density,
            pressure,
            divergence,
            uniforms_buf,
            tet_count,
            step_count: 0,
            mesh_bbox_min: [0.0; 3],
            mesh_bbox_max: [0.0; 3],
        }
    }

    /// Precompute tet geometry (volumes, centroids, face normals/areas).
    /// Call once after loading scene data.
    pub fn precompute_geometry(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        vertices: &wgpu::Buffer,
        indices: &wgpu::Buffer,
        tet_neighbors: &wgpu::Buffer,
    ) {
        // Write tet_count into uniforms for the precompute shader
        let uniforms = FluidUniforms {
            tet_count: self.tet_count,
            ..Zeroable::zeroed()
        };

        // Upload uniforms via a temporary init buffer since encoder can't write_buffer
        let uniforms_staging = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("precompute_uniforms_staging"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        encoder.copy_buffer_to_buffer(
            &uniforms_staging,
            0,
            &self.uniforms_buf,
            0,
            std::mem::size_of::<FluidUniforms>() as u64,
        );

        let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("precompute_bg0"),
            layout: &self.precompute_bgl_0,
            entries: &[
                buf_entry(0, &self.uniforms_buf),
                buf_entry(1, vertices),
                buf_entry(2, indices),
                buf_entry(3, tet_neighbors),
            ],
        });

        let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("precompute_bg1"),
            layout: &self.precompute_bgl_1,
            entries: &[
                buf_entry(0, &self.tet_volumes),
                buf_entry(1, &self.face_geo),
                buf_entry(2, &self.tet_centers),
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("precompute_geometry"),
            ..Default::default()
        });
        pass.set_pipeline(&self.precompute_pipeline);
        pass.set_bind_group(0, &bg0, &[]);
        pass.set_bind_group(1, &bg1, &[]);
        pass.dispatch_workgroups(dispatch_count(self.tet_count), 1, 1);
    }

    /// Compute and store the mesh bounding box from tet centers.
    /// Call after precompute_geometry + queue submit.
    pub fn compute_mesh_bbox(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let centers = readback_f32(device, queue, &self.tet_centers);
        let mut bbox_min = [f32::INFINITY; 3];
        let mut bbox_max = [f32::NEG_INFINITY; 3];
        for chunk in centers.chunks_exact(4) {
            for i in 0..3 {
                if chunk[i] < bbox_min[i] { bbox_min[i] = chunk[i]; }
                if chunk[i] > bbox_max[i] { bbox_max[i] = chunk[i]; }
            }
        }
        self.mesh_bbox_min = bbox_min;
        self.mesh_bbox_max = bbox_max;
    }

    /// Returns the mesh center (midpoint of bounding box).
    pub fn mesh_center(&self) -> [f32; 3] {
        [
            (self.mesh_bbox_min[0] + self.mesh_bbox_max[0]) * 0.5,
            (self.mesh_bbox_min[1] + self.mesh_bbox_max[1]) * 0.5,
            (self.mesh_bbox_min[2] + self.mesh_bbox_max[2]) * 0.5,
        ]
    }

    /// Returns the mesh extent (max of bbox dimensions).
    pub fn mesh_extent(&self) -> f32 {
        let dx = self.mesh_bbox_max[0] - self.mesh_bbox_min[0];
        let dy = self.mesh_bbox_max[1] - self.mesh_bbox_min[1];
        let dz = self.mesh_bbox_max[2] - self.mesh_bbox_min[2];
        dx.max(dy).max(dz)
    }

    /// Generate default FluidParams with source placed at the bottom-center of the mesh.
    pub fn default_params_for_mesh(&self) -> FluidParams {
        let center = self.mesh_center();
        let extent = self.mesh_extent();
        // Place source at bottom-center (lower 25% in Y)
        let source_y = self.mesh_bbox_min[1] + (self.mesh_bbox_max[1] - self.mesh_bbox_min[1]) * 0.25;
        FluidParams {
            source_pos: [center[0], source_y, center[2]],
            source_radius: extent * 0.05, // 5% of mesh extent
            ..FluidParams::default()
        }
    }

    /// Log precomputed geometry stats (call after precompute_geometry + queue submit).
    pub fn log_precompute_stats(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let volumes = readback_f32(device, queue, &self.tet_volumes);
        let face_geo_data = readback_f32(device, queue, &self.face_geo);
        let centers = readback_f32(device, queue, &self.tet_centers);

        log::info!("[fluid] Precompute stats for {} tets:", self.tet_count);
        log::info!("[fluid]   tet_volumes: {}", compute_stats(&volumes));

        // Compute bounding box of tet centers (vec4 layout: x,y,z,w per tet)
        let mut bbox_min = [f32::INFINITY; 3];
        let mut bbox_max = [f32::NEG_INFINITY; 3];
        for chunk in centers.chunks_exact(4) {
            for i in 0..3 {
                if chunk[i] < bbox_min[i] { bbox_min[i] = chunk[i]; }
                if chunk[i] > bbox_max[i] { bbox_max[i] = chunk[i]; }
            }
        }
        let bbox_center = [
            (bbox_min[0] + bbox_max[0]) * 0.5,
            (bbox_min[1] + bbox_max[1]) * 0.5,
            (bbox_min[2] + bbox_max[2]) * 0.5,
        ];
        let bbox_size = [
            bbox_max[0] - bbox_min[0],
            bbox_max[1] - bbox_min[1],
            bbox_max[2] - bbox_min[2],
        ];
        log::info!(
            "[fluid]   tet_centers bbox: min=[{:.3},{:.3},{:.3}] max=[{:.3},{:.3},{:.3}]",
            bbox_min[0], bbox_min[1], bbox_min[2],
            bbox_max[0], bbox_max[1], bbox_max[2],
        );
        log::info!(
            "[fluid]   tet_centers bbox center=[{:.3},{:.3},{:.3}] size=[{:.3},{:.3},{:.3}]",
            bbox_center[0], bbox_center[1], bbox_center[2],
            bbox_size[0], bbox_size[1], bbox_size[2],
        );
        log::info!("[fluid]   tet_centers: {}", compute_stats(&centers));

        // face_geo is M*4 vec4s = M*16 floats. Separate xyz (area-weighted normal) from w (laplacian coeff).
        let mut normals_mag = Vec::with_capacity(self.tet_count as usize * 4);
        let mut coeffs = Vec::with_capacity(self.tet_count as usize * 4);
        for chunk in face_geo_data.chunks_exact(4) {
            let nx = chunk[0];
            let ny = chunk[1];
            let nz = chunk[2];
            normals_mag.push((nx*nx + ny*ny + nz*nz).sqrt());
            coeffs.push(chunk[3]);
        }
        log::info!("[fluid]   face_geo |n*A|: {}", compute_stats(&normals_mag));
        log::info!("[fluid]   face_geo coeff (A/d): {}", compute_stats(&coeffs));

        // Check how many boundary faces (coeff == 0)
        let boundary_count = coeffs.iter().filter(|&&c| c == 0.0).count();
        let total_faces = coeffs.len();
        log::info!(
            "[fluid]   boundary faces: {}/{} ({:.1}%)",
            boundary_count, total_faces,
            100.0 * boundary_count as f64 / total_faces as f64
        );
    }

    /// Run one fluid simulation timestep.
    ///
    /// Writes results into the scene's `densities_buf` and `colors_buf` for rendering.
    pub fn step(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        params: &FluidParams,
        tet_neighbors: &wgpu::Buffer,
        densities_buf: &wgpu::Buffer,
        colors_buf: &wgpu::Buffer,
    ) {
        let wg = dispatch_count(self.tet_count);

        if self.step_count < 3 {
            // On first step, count how many tets overlap the source
            if self.step_count == 0 {
                let centers = readback_f32(device, queue, &self.tet_centers);
                let sp = params.source_pos;
                let r = params.source_radius;
                let mut overlap_count = 0u32;
                for chunk in centers.chunks_exact(4) {
                    let dx = chunk[0] - sp[0];
                    let dy = chunk[1] - sp[1];
                    let dz = chunk[2] - sp[2];
                    if (dx*dx + dy*dy + dz*dz).sqrt() < r {
                        overlap_count += 1;
                    }
                }
                log::info!(
                    "[fluid] Source overlap: {} tets within radius {} of [{},{},{}]",
                    overlap_count, r, sp[0], sp[1], sp[2],
                );
            }
            log::info!(
                "[fluid] step {} params: dt={} visc={} gravity=[{},{},{}] \
                 source_pos=[{},{},{}] r={} strength={} buoyancy={} density_scale={} \
                 diffuse_iters={} pressure_iters={}",
                self.step_count, params.dt, params.viscosity,
                params.gravity[0], params.gravity[1], params.gravity[2],
                params.source_pos[0], params.source_pos[1], params.source_pos[2],
                params.source_radius, params.source_strength,
                params.buoyancy, params.density_scale,
                params.diffuse_iterations, params.pressure_iterations,
            );
        }

        // --- Upload uniforms ---
        let uniforms = FluidUniforms {
            dt: params.dt,
            viscosity: params.viscosity,
            tet_count: self.tet_count,
            jacobi_iter: 0,
            gravity: [params.gravity[0], params.gravity[1], params.gravity[2], 0.0],
            source_pos: [
                params.source_pos[0],
                params.source_pos[1],
                params.source_pos[2],
                params.source_radius,
            ],
            source_strength: params.source_strength,
            buoyancy: params.buoyancy,
            density_scale: params.density_scale,
            _pad: 0,
        };
        queue.write_buffer(&self.uniforms_buf, 0, bytemuck::bytes_of(&uniforms));

        // --- Create shared geometry bind group ---
        let geo_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim_geo_bg"),
            layout: &self.sim_geo_bgl,
            entries: &[
                buf_entry(0, &self.uniforms_buf),
                buf_entry(1, tet_neighbors),
                buf_entry(2, &self.face_geo),
                buf_entry(3, &self.tet_volumes),
                buf_entry(4, &self.tet_centers),
            ],
        });

        // --- 1. Advect ---
        // Read velocity[0], density[0] → write velocity[1], density[1]
        {
            let state_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("advect_state"),
                layout: &self.advect_state_bgl,
                entries: &[
                    buf_entry(0, &self.velocity[0]),
                    buf_entry(1, &self.density[0]),
                    buf_entry(2, &self.velocity[1]),
                    buf_entry(3, &self.density[1]),
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("advect"),
                ..Default::default()
            });
            pass.set_pipeline(&self.advect_pipeline);
            pass.set_bind_group(0, &geo_bg, &[]);
            pass.set_bind_group(1, &state_bg, &[]);
            pass.dispatch_workgroups(wg, 1, 1);
        }

        // After advect: velocity[1] and density[1] hold current state.
        // Copy back to [0] for next stages.
        encoder.copy_buffer_to_buffer(
            &self.velocity[1],
            0,
            &self.velocity[0],
            0,
            self.tet_count as u64 * 16,
        );
        encoder.copy_buffer_to_buffer(
            &self.density[1],
            0,
            &self.density[0],
            0,
            self.tet_count as u64 * 4,
        );

        // --- 2. Diffuse (Jacobi iterations) ---
        // Ping-pong between velocity[0] and velocity[1]
        for i in 0..params.diffuse_iterations {
            let src = (i % 2) as usize;
            let dst = 1 - src;

            let state_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("diffuse_state"),
                layout: &self.diffuse_state_bgl,
                entries: &[
                    buf_entry(0, &self.velocity[src]),
                    buf_entry(1, &self.velocity[dst]),
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("diffuse"),
                ..Default::default()
            });
            pass.set_pipeline(&self.diffuse_pipeline);
            pass.set_bind_group(0, &geo_bg, &[]);
            pass.set_bind_group(1, &state_bg, &[]);
            pass.dispatch_workgroups(wg, 1, 1);
        }

        // After diffuse: result is in velocity[diffuse_iterations % 2].
        // Ensure it's in velocity[0] for subsequent stages.
        if params.diffuse_iterations % 2 == 1 {
            encoder.copy_buffer_to_buffer(
                &self.velocity[1],
                0,
                &self.velocity[0],
                0,
                self.tet_count as u64 * 16,
            );
        }

        // --- 3. Forces ---
        // In-place update of velocity[0] and density[0]
        {
            let state_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("forces_state"),
                layout: &self.forces_state_bgl,
                entries: &[
                    buf_entry(0, &self.velocity[0]),
                    buf_entry(1, &self.density[0]),
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("forces"),
                ..Default::default()
            });
            pass.set_pipeline(&self.forces_pipeline);
            pass.set_bind_group(0, &geo_bg, &[]);
            pass.set_bind_group(1, &state_bg, &[]);
            pass.dispatch_workgroups(wg, 1, 1);
        }

        // --- 4. Divergence ---
        {
            let state_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("divergence_state"),
                layout: &self.divergence_state_bgl,
                entries: &[
                    buf_entry(0, &self.velocity[0]),
                    buf_entry(1, &self.divergence),
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("divergence"),
                ..Default::default()
            });
            pass.set_pipeline(&self.divergence_pipeline);
            pass.set_bind_group(0, &geo_bg, &[]);
            pass.set_bind_group(1, &state_bg, &[]);
            pass.dispatch_workgroups(wg, 1, 1);
        }

        // --- 5. Pressure solve (Jacobi iterations) ---
        // Clear pressure[0] to zero before solving
        queue.write_buffer(
            &self.pressure[0],
            0,
            &vec![0u8; self.tet_count as usize * 4],
        );

        for i in 0..params.pressure_iterations {
            let src = (i % 2) as usize;
            let dst = 1 - src;

            let state_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pressure_state"),
                layout: &self.pressure_state_bgl,
                entries: &[
                    buf_entry(0, &self.pressure[src]),
                    buf_entry(1, &self.pressure[dst]),
                    buf_entry(2, &self.divergence),
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pressure_solve"),
                ..Default::default()
            });
            pass.set_pipeline(&self.pressure_pipeline);
            pass.set_bind_group(0, &geo_bg, &[]);
            pass.set_bind_group(1, &state_bg, &[]);
            pass.dispatch_workgroups(wg, 1, 1);
        }

        // After pressure solve: result in pressure[pressure_iterations % 2].
        // Ensure it's in pressure[0].
        if params.pressure_iterations % 2 == 1 {
            encoder.copy_buffer_to_buffer(
                &self.pressure[1],
                0,
                &self.pressure[0],
                0,
                self.tet_count as u64 * 4,
            );
        }

        // --- 6. Project ---
        {
            let state_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("project_state"),
                layout: &self.project_state_bgl,
                entries: &[
                    buf_entry(0, &self.velocity[0]),
                    buf_entry(1, &self.pressure[0]),
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("project"),
                ..Default::default()
            });
            pass.set_pipeline(&self.project_pipeline);
            pass.set_bind_group(0, &geo_bg, &[]);
            pass.set_bind_group(1, &state_bg, &[]);
            pass.dispatch_workgroups(wg, 1, 1);
        }

        // --- 7. To render ---
        {
            let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("to_render_bg0"),
                layout: &self.to_render_bgl_0,
                entries: &[buf_entry(0, &self.uniforms_buf)],
            });

            let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("to_render_bg1"),
                layout: &self.to_render_bgl_1,
                entries: &[
                    buf_entry(0, &self.density[0]),
                    buf_entry(1, &self.velocity[0]),
                ],
            });

            let bg2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("to_render_bg2"),
                layout: &self.to_render_bgl_2,
                entries: &[
                    buf_entry(0, densities_buf),
                    buf_entry(1, colors_buf),
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("to_render"),
                ..Default::default()
            });
            pass.set_pipeline(&self.to_render_pipeline);
            pass.set_bind_group(0, &bg0, &[]);
            pass.set_bind_group(1, &bg1, &[]);
            pass.set_bind_group(2, &bg2, &[]);
            pass.dispatch_workgroups(wg, 1, 1);
        }

        self.step_count += 1;
    }

    /// Read back simulation state and log statistics. Call after queue.submit().
    pub fn log_step_stats(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        densities_buf: &wgpu::Buffer,
        colors_buf: &wgpu::Buffer,
    ) {
        log::info!("[fluid] === Step {} stats ===", self.step_count);

        // Sim-internal buffers
        let sim_density = readback_f32(device, queue, &self.density[0]);
        let sim_velocity = readback_f32(device, queue, &self.velocity[0]);
        let sim_pressure = readback_f32(device, queue, &self.pressure[0]);
        let sim_divergence = readback_f32(device, queue, &self.divergence);

        log::info!("[fluid]   sim density:    {}", compute_stats(&sim_density));
        log::info!("[fluid]   sim divergence: {}", compute_stats(&sim_divergence));
        log::info!("[fluid]   sim pressure:   {}", compute_stats(&sim_pressure));

        // Velocity: report magnitude stats
        let vel_mag: Vec<f32> = sim_velocity
            .chunks_exact(4)
            .map(|c| (c[0]*c[0] + c[1]*c[1] + c[2]*c[2]).sqrt())
            .collect();
        log::info!("[fluid]   sim |velocity|: {}", compute_stats(&vel_mag));

        // Output render buffers
        let render_den = readback_f32(device, queue, densities_buf);
        let render_col = readback_f32(device, queue, colors_buf);

        log::info!("[fluid]   render density: {}", compute_stats(&render_den));
        log::info!("[fluid]   render colors:  {}", compute_stats(&render_col));
    }

    /// Reset simulation state (zero all velocity, density, pressure).
    pub fn reset(&mut self, queue: &wgpu::Queue) {
        self.step_count = 0;
        let m = self.tet_count as usize;
        let zeros_f32 = vec![0u8; m * 4];
        let zeros_vec4 = vec![0u8; m * 16];

        for buf in &self.velocity {
            queue.write_buffer(buf, 0, &zeros_vec4);
        }
        for buf in &self.density {
            queue.write_buffer(buf, 0, &zeros_f32);
        }
        for buf in &self.pressure {
            queue.write_buffer(buf, 0, &zeros_f32);
        }
        queue.write_buffer(&self.divergence, 0, &zeros_f32);
    }
}
