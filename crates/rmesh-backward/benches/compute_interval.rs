//! Compute-interval shading benchmark (no mesh shader — works on all GPUs).
//!
//! Measures compute-interval forward pipeline performance at 1920x1080
//! with ~10M tets. Requires SUBGROUP + TIMESTAMP_QUERY. Skips gracefully
//! if unavailable.
//!
//! Run: `cargo bench -p rmesh-backward --bench compute_interval`

use criterion::{criterion_group, criterion_main, Criterion};
use glam::{Mat4, Vec3};
use rmesh_util::camera::{look_at, perspective_matrix};
use rmesh_util::test_util::grid_tet_scene;

const W: u32 = 1920;
const H: u32 = 1080;
// Smaller grid than tiled_pipeline (126) because the fixed-slot vertex buffer
// is tet_count * 5 * 3 * 16 bytes, which exceeds max_storage_buffer_binding_size
// at ~10M tets. Grid 120 → ~8.4M tets → ~2.0GB vertex buffer.
const GRID_SIZE: u32 = 120;

fn setup_camera() -> (Mat4, glam::Mat3, [f32; 4], Vec3) {
    let fov_y = std::f32::consts::FRAC_PI_4;
    let aspect = W as f32 / H as f32;
    let proj = perspective_matrix(fov_y, aspect, 0.01, 100.0);
    let eye = Vec3::new(0.5, 0.5, 3.0);
    let target = Vec3::new(0.5, 0.5, 0.5);
    let up = Vec3::new(0.0, 1.0, 0.0);
    let view = look_at(eye, target, up);
    let vp = proj * view;
    let f = (target - eye).normalize();
    let r = f.cross(up).normalize();
    let u = r.cross(f);
    let c2w = glam::Mat3::from_cols(r, -u, f);
    let f_val = 1.0 / (fov_y / 2.0).tan();
    let intrinsics = [
        f_val * H as f32 / 2.0,
        f_val * H as f32 / 2.0,
        W as f32 / 2.0,
        H as f32 / 2.0,
    ];
    (vp, c2w, intrinsics, eye)
}

/// GPU state for the compute-interval shading benchmark.
#[allow(dead_code)]
struct ComputeIntervalBenchState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    tet_count: u32,
    buffers: rmesh_render::SceneBuffers,
    material: rmesh_render::MaterialBuffers,
    fwd_pipelines: rmesh_render::ForwardPipelines,
    targets: rmesh_render::RenderTargets,
    compute_bg: wgpu::BindGroup,
    ci_pipelines: rmesh_render::ComputeIntervalPipelines,
    sort_pipelines: rmesh_backward::RadixSortPipelines,
    sort_state: rmesh_backward::RadixSortState,
    ci_gen_bg_a: wgpu::BindGroup,
    ci_gen_bg_b: wgpu::BindGroup,
    ci_render_bg: wgpu::BindGroup,
    ci_convert_bg: wgpu::BindGroup,
    depth_view: wgpu::TextureView,
    use_16bit_sort: bool,
}

fn create_bench_state() -> Option<ComputeIntervalBenchState> {
    create_bench_state_with_bits(32)
}

fn create_bench_state_16bit() -> Option<ComputeIntervalBenchState> {
    create_bench_state_with_bits(16)
}

fn create_bench_state_with_bits(sorting_bits: u32) -> Option<ComputeIntervalBenchState> {
    let (device, queue) = pollster::block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok()?;

        let supported_limits = adapter.limits();
        let limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 16,
            ..supported_limits
        };

        adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::SUBGROUP
                    | wgpu::Features::SHADER_FLOAT32_ATOMIC
                    | wgpu::Features::TIMESTAMP_QUERY,
                required_limits: limits,
                ..Default::default()
            })
            .await
            .ok()
    })?;

    eprintln!("Generating compute-interval grid scene (grid_size={GRID_SIZE})...");
    let scene = grid_tet_scene(GRID_SIZE);
    eprintln!(
        "Scene: {} vertices, {} tets",
        scene.vertex_count, scene.tet_count
    );

    let (vp, c2w, intrinsics, eye) = setup_camera();

    let color_format = wgpu::TextureFormat::Rgba16Float;
    let base_colors = vec![0.5f32; scene.tet_count as usize * 3];
    let (buffers, material, fwd_pipelines, targets, compute_bg, _render_bg) =
        rmesh_render::setup_forward(
            &device,
            &queue,
            &scene,
            &base_colors,
            &scene.color_grads,
            W,
            H,
        );

    let uniforms = rmesh_render::make_uniforms(
        vp, c2w, intrinsics, eye,
        W as f32, H as f32,
        scene.tet_count, 0u32, 16, 0.0, 0,
        0.01, 100.0,
    );
    queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

    let ci_pipelines = rmesh_render::ComputeIntervalPipelines::new(&device, color_format);

    // Sort infrastructure (sorting_bits-bit keys, 1 payload — tet-level sort)
    let n_pow2 = scene.tet_count.next_power_of_two();
    let sort_pipelines =
        rmesh_backward::RadixSortPipelines::new(&device, 1, rmesh_backward::SortBackend::Drs);
    let sort_state = rmesh_backward::RadixSortState::new(
        &device, n_pow2, sorting_bits, 1, rmesh_backward::SortBackend::Drs,
    );
    sort_state.upload_configs(&queue);

    // Bind groups (A and B for sort buffer swapping)
    let ci_gen_bg_a = rmesh_render::create_compute_interval_gen_bind_group(
        &device, &ci_pipelines, &buffers, &material,
    );
    let ci_gen_bg_b = rmesh_render::create_compute_interval_gen_bind_group_with_sort_values(
        &device, &ci_pipelines, &buffers, &material, sort_state.values_b(),
    );
    let ci_render_bg = rmesh_render::create_compute_interval_render_bind_group(
        &device, &ci_pipelines, &buffers,
    );
    let ci_convert_bg = rmesh_render::create_compute_interval_indirect_convert_bind_group(
        &device, &ci_pipelines, &buffers,
    );

    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bench_ci_depth"),
        size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

    Some(ComputeIntervalBenchState {
        device,
        queue,
        tet_count: scene.tet_count,
        buffers,
        material,
        fwd_pipelines,
        targets,
        compute_bg,
        ci_pipelines,
        sort_pipelines,
        sort_state,
        ci_gen_bg_a,
        ci_gen_bg_b,
        ci_render_bg,
        ci_convert_bg,
        depth_view,
        use_16bit_sort: sorting_bits == 16,
    })
}

fn run_forward(s: &ComputeIntervalBenchState) {
    let mut encoder = s
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    // Clear color and depth
    {
        let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("bench_ci_clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &s.targets.color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &s.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
    }

    rmesh_render::record_sorted_compute_interval_forward_pass(
        &mut encoder,
        &s.device,
        &s.fwd_pipelines,
        &s.ci_pipelines,
        &s.sort_pipelines,
        &s.sort_state,
        &s.buffers,
        &s.targets,
        &s.compute_bg,
        &s.ci_gen_bg_a,
        &s.ci_gen_bg_b,
        &s.ci_render_bg,
        &s.ci_convert_bg,
        s.tet_count,
        &s.queue,
        &s.depth_view,
        None,
        None,
        s.use_16bit_sort,
    );

    s.queue.submit(std::iter::once(encoder.finish()));
    let _ = s.device.poll(wgpu::PollType::wait_indefinitely());
}

fn bench_compute_interval(c: &mut Criterion) {
    let state = match create_bench_state() {
        Some(s) => s,
        None => {
            eprintln!("Skipping bench (no GPU with SUBGROUP + TIMESTAMP_QUERY)");
            return;
        }
    };

    // Warmup
    run_forward(&state);

    c.bench_function("forward_compute_interval_2M", |b| {
        b.iter(|| run_forward(&state));
    });
}

fn bench_compute_interval_16bit(c: &mut Criterion) {
    let state = match create_bench_state_16bit() {
        Some(s) => s,
        None => {
            eprintln!("Skipping 16-bit bench (no GPU with SUBGROUP + TIMESTAMP_QUERY)");
            return;
        }
    };

    // Warmup
    run_forward(&state);

    c.bench_function("forward_compute_interval_16bit_2M", |b| {
        b.iter(|| run_forward(&state));
    });
}

criterion_group! {
    name = compute_interval;
    config = Criterion::default().sample_size(20);
    targets = bench_compute_interval, bench_compute_interval_16bit
}
criterion_main!(compute_interval);
