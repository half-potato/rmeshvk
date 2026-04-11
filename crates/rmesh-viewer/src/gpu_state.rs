//! GPU state: device, pipelines, buffers, bind groups, and profiling.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use rmesh_render::{
    BlitPipeline, ForwardPipelines, MaterialBuffers,
    MeshForwardPipelines, IntervalPipelines, ComputeIntervalPipelines, RenderTargets,
    SceneBuffers,
};
use rmesh_compositor::{PrimitiveGeometry, PrimitivePipeline, PrimitiveTargets};
use rmesh_sim::FluidSim;

#[derive(Clone, Copy, PartialEq)]
pub enum RenderMode {
    Regular,
    Quad,
    MeshShader,
    IntervalShader,
    RayTrace,
}

impl std::fmt::Display for RenderMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenderMode::Regular => write!(f, "Regular"),
            RenderMode::Quad => write!(f, "Quad"),
            RenderMode::MeshShader => write!(f, "Mesh Shader"),
            RenderMode::IntervalShader => write!(f, "Interval Shader"),
            RenderMode::RayTrace => write!(f, "Ray Trace"),
        }
    }
}

/// Per-frame GPU timing breakdown from timestamp queries.
#[derive(Clone, Default)]
pub struct GpuTimings {
    pub sh_eval_ms: f32,
    pub project_ms: f32,
    pub sort_ms: f32,
    pub prepass_ms: f32,
    pub render_ms: f32,
    pub total_ms: f32,
}

/// Per-frame CPU timing breakdown (rolling average).
#[derive(Clone, Default)]
pub struct CpuTimings {
    pub poll_readback_ms: f32,
    pub acquire_ms: f32,
    pub egui_ms: f32,
    pub encode_ms: f32,
    pub submit_ms: f32,
    pub present_ms: f32,
    pub total_ms: f32,
}

pub const TS_QUERY_COUNT: u32 = 8;

/// Pack f32 SH coefficients into f16 pairs stored as u32 (matching WGSL `unpack2x16float` layout).
pub fn pack_sh_coeffs_f16(coeffs: &[f32]) -> Vec<u32> {
    let mut packed = vec![0u32; (coeffs.len() + 1) / 2];
    for i in (0..coeffs.len()).step_by(2) {
        let lo = half::f16::from_f32(coeffs[i]);
        let hi = if i + 1 < coeffs.len() {
            half::f16::from_f32(coeffs[i + 1])
        } else {
            half::f16::ZERO
        };
        packed[i / 2] = (lo.to_bits() as u32) | ((hi.to_bits() as u32) << 16);
    }
    packed
}

pub struct GpuState {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub pipelines: ForwardPipelines,
    pub blit_pipeline: BlitPipeline,
    pub sort_pipelines: rmesh_sort::RadixSortPipelines,
    pub sort_state: rmesh_sort::RadixSortState,
    pub sort_state_16bit: rmesh_sort::RadixSortState,
    pub sort_backend: rmesh_sort::SortBackend,
    pub buffers: SceneBuffers,
    pub material_buffers: MaterialBuffers,
    pub targets: RenderTargets,
    pub compute_bg: wgpu::BindGroup,
    pub hw_compute_bg: wgpu::BindGroup,
    pub render_bg: wgpu::BindGroup,
    pub render_bg_b: wgpu::BindGroup,
    pub blit_bg: wgpu::BindGroup,
    pub tet_count: u32,
    pub sh_coeffs_buf: wgpu::Buffer,
    pub sh_degree: u32,
    pub pending_reconfigure: bool,
    // Mesh shader (optional)
    pub mesh_pipelines: Option<MeshForwardPipelines>,
    pub mesh_render_bg_a: Option<wgpu::BindGroup>,
    pub mesh_render_bg_b: Option<wgpu::BindGroup>,
    pub indirect_convert_bg: Option<wgpu::BindGroup>,
    // Interval shading (optional, requires mesh shader support)
    pub interval_pipelines: Option<IntervalPipelines>,
    pub interval_render_bg_a: Option<wgpu::BindGroup>,
    pub interval_render_bg_b: Option<wgpu::BindGroup>,
    pub interval_indirect_convert_bg: Option<wgpu::BindGroup>,
    // Compute-based interval shading (always available)
    pub compute_interval_pipelines: ComputeIntervalPipelines,
    pub compute_interval_gen_bg_a: wgpu::BindGroup,
    pub compute_interval_gen_bg_b: wgpu::BindGroup,
    pub compute_interval_gen_bg_a_16bit: wgpu::BindGroup,
    pub compute_interval_gen_bg_b_16bit: wgpu::BindGroup,
    pub compute_interval_render_bg: wgpu::BindGroup,
    pub compute_interval_convert_bg: wgpu::BindGroup,
    // Quad prepass bind groups (A/B for sort result location)
    pub prepass_bg_a: wgpu::BindGroup,
    pub prepass_bg_b: wgpu::BindGroup,
    pub quad_render_bg: wgpu::BindGroup,
    // Fluid simulation
    pub fluid_sim: Option<FluidSim>,
    pub tet_neighbors_buf: Option<wgpu::Buffer>,
    // Primitives
    pub primitive_geometry: PrimitiveGeometry,
    pub primitive_pipeline: PrimitivePipeline,
    pub primitive_targets: PrimitiveTargets,
    // egui
    pub egui_renderer: egui_wgpu::Renderer,
    pub egui_state: egui_winit::State,
    // Instance count readback
    pub instance_count_readback: wgpu::Buffer,
    pub visible_instance_count: u32,
    // GPU timestamp profiling
    pub ts_query_set: wgpu::QuerySet,
    pub ts_resolve_buf: wgpu::Buffer,
    pub ts_readback: [wgpu::Buffer; 2],
    pub ts_readback_ready: [Arc<AtomicBool>; 2],
    pub ts_readback_mapped: [Arc<AtomicBool>; 2],
    pub ts_frame: usize,
    pub ts_period_ns: f32,
    pub gpu_times_ms: GpuTimings,
    pub cpu_times_ms: CpuTimings,
    pub instance_count_ready: Arc<AtomicBool>,
    pub instance_count_mapped: Arc<AtomicBool>,
    // PBR aux data (per-tet material channels, uploaded when PBR data present)
    pub aux_data_buf: Option<wgpu::Buffer>,
    // Deferred PBR shading (only when PBR data is loaded)
    pub deferred_pipeline: Option<rmesh_render::DeferredShadePipeline>,
    pub deferred_bg: Option<wgpu::BindGroup>,
    /// Separate output texture for deferred pass (can't read+write color_view simultaneously)
    pub deferred_output: Option<wgpu::Texture>,
    pub deferred_output_view: Option<wgpu::TextureView>,
    /// Blit bind group pointing to deferred output (swapped when deferred is active)
    pub deferred_blit_bg: Option<wgpu::BindGroup>,
    pub has_pbr_data: bool,
    // Ray trace
    pub rt_pipeline: rmesh_render::RayTracePipeline,
    pub rt_buffers: rmesh_render::RayTraceBuffers,
    pub rt_bg: wgpu::BindGroup,
    /// Rgba32Float texture for copying raytrace buffer output for blitting
    pub rt_texture: wgpu::Texture,
    pub rt_texture_view: wgpu::TextureView,
    pub rt_blit_pipeline: rmesh_render::BlitPipelineNonFiltering,
    pub rt_blit_bg: wgpu::BindGroup,
}
