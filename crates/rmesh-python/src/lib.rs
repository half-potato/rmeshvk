//! PyO3 bindings for the wgpu-based tetrahedral volume renderer.
//!
//! Exposes `RMeshRenderer` to Python with:
//!   - `forward()`: run the forward render pipeline, return image as numpy array
//!   - `backward()`: upload dL/d(image), run backward pass, return gradients
//!   - `train_step()`: full forward+loss+backward+adam on GPU, return loss
//!   - `update_params()` / `get_params()`: parameter transfer for PyTorch optimizer path

use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray3, PyReadonlyArray1, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use glam::{Mat3, Mat4, Vec3, Vec4};
use rmesh_backward::{
    create_backward_tiled_bind_groups,
    create_prepare_dispatch_bind_group, create_rts_bind_group,
    create_tile_fill_bind_group, create_tile_gen_scan_bind_group,
    create_tile_ranges_bind_group_with_keys,
    BackwardTiledPipelines, GradientBuffers, MaterialGradBuffers, RadixSortPipelines, RadixSortState,
    ScanBuffers, ScanPipelines, SortBackend, TileBuffers, TilePipelines, TileUniforms,
};
use rmesh_render::{
    build_boundary_bvh, compute_tet_neighbors, create_compute_bind_group,
    create_locate_bind_group, create_rasterize_bind_group, create_raytrace_bind_group,
    create_render_bind_group, find_containing_tet_walk, make_uniforms, record_locate,
    record_project_compute, record_forward_pass, record_rasterize_compute, record_raytrace,
    record_tex_to_buffer, ForwardPipelines, LocatePipeline, LocateUniforms,
    RasterizeComputePipeline, MaterialBuffers, RayTraceBuffers, RayTracePipeline, RenderTargets,
    SceneBuffers, TexToBufferPipeline,
};
use rmesh_error::{
    ErrorPipeline, ErrorBuffers, ErrorInputBuffers,
    create_error_bg0, create_error_bg1, record_error_pass,
};
use rmesh_train::{
    create_adam_bind_group, create_loss_bind_group, AdamPipeline, AdamState, LossBuffers,
    LossPipeline, MaterialAdamState,
};
use rmesh_util::shared::{AdamUniforms, LossUniforms};

/// Helper: read back a GPU buffer to a Vec<f32>.
fn read_buffer_f32(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<f32> {
    let size = buffer.size();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback_staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    result
}

/// Helper: read back a GPU buffer to a Vec<u8>.
fn read_buffer_raw(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
) -> Vec<u8> {
    let size = buffer.size();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback_staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let result = data.to_vec();
    drop(data);
    staging.unmap();
    result
}

/// Helper: build a Mat4 from a flat [16] f32 column-major array.
fn mat4_from_flat(data: &[f32]) -> Mat4 {
    Mat4::from_cols(
        Vec4::new(data[0], data[1], data[2], data[3]),
        Vec4::new(data[4], data[5], data[6], data[7]),
        Vec4::new(data[8], data[9], data[10], data[11]),
        Vec4::new(data[12], data[13], data[14], data[15]),
    )
}

/// Helper: extract c2w rotation (Mat3) + intrinsics ([f32; 4]) from flat [16] f32.
/// Layout: [c2w_col0(xyz,0), c2w_col1(xyz,0), c2w_col2(xyz,0), intrinsics(fx,fy,cx,cy)]
fn c2w_from_flat(data: &[f32]) -> (Mat3, [f32; 4]) {
    let c2w = Mat3::from_cols(
        glam::Vec3::new(data[0], data[1], data[2]),
        glam::Vec3::new(data[4], data[5], data[6]),
        glam::Vec3::new(data[8], data[9], data[10]),
    );
    let intrinsics = [data[12], data[13], data[14], data[15]];
    (c2w, intrinsics)
}

#[pyclass]
struct RMeshRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    scene_buffers: SceneBuffers,
    material_buffers: MaterialBuffers,
    fwd_pipelines: ForwardPipelines,
    targets: RenderTargets,
    ttb: TexToBufferPipeline,
    // Backward infrastructure
    bwd_tiled_pipelines: BackwardTiledPipelines,
    grad_buffers: GradientBuffers,
    mat_grad_buffers: MaterialGradBuffers,
    loss_buffers: LossBuffers,
    loss_pipeline: LossPipeline,
    adam_pipeline: AdamPipeline,
    _adam_state: AdamState,
    _mat_adam_state: MaterialAdamState,
    // Tiled backward infrastructure
    tile_pipelines: TilePipelines,
    tile_buffers: TileBuffers,
    radix_pipelines: RadixSortPipelines,
    radix_state: RadixSortState,
    tile_fill_bg: wgpu::BindGroup,
    // A-buffer bind groups (sort result in primary buffers)
    tile_ranges_bg: wgpu::BindGroup,
    // B-buffer bind groups (sort result in alternate buffers)
    tile_ranges_bg_b: wgpu::BindGroup,
    // Forward tiled compute pipeline (subgroup-based, 4x4 tiles)
    rasterize: RasterizeComputePipeline,
    rasterize_bg: wgpu::BindGroup,
    rasterize_bg_b: wgpu::BindGroup,
    // Bind groups
    compute_bg: wgpu::BindGroup,
    render_bg: wgpu::BindGroup,
    // Loss bind group using rasterize.rendered_image (for train_step)
    loss_bg_tiled: wgpu::BindGroup,
    // Backward tiled bind groups using rasterize.rendered_image (A and B variants)
    bwd_tiled_bg0_tiled: wgpu::BindGroup,
    bwd_tiled_bg0_tiled_b: wgpu::BindGroup,
    bwd_tiled_bg1: wgpu::BindGroup,
    // Backward tiled bind groups using ttb.rendered_image (HW raster path, A and B variants)
    bwd_tiled_bg0_hw: wgpu::BindGroup,
    bwd_tiled_bg0_hw_b: wgpu::BindGroup,
    // Loss bind group using ttb.rendered_image (HW raster path)
    _loss_bg_hw: wgpu::BindGroup,
    adam_bgs: Vec<wgpu::BindGroup>,
    adam_uniforms_bufs: Vec<wgpu::Buffer>,
    // Scan-based tile pipeline (RTS prefix scan)
    scan_pipelines: ScanPipelines,
    scan_buffers: ScanBuffers,
    prepare_dispatch_bg: wgpu::BindGroup,
    rts_bg: wgpu::BindGroup,
    tile_gen_scan_bg: wgpu::BindGroup,
    // Ray trace infrastructure
    rt_pipeline: RayTracePipeline,
    rt_buffers: RayTraceBuffers,
    rt_bind_group: wgpu::BindGroup,
    // Point location pipeline
    locate_pipeline: LocatePipeline,
    locate_uniforms_buf: wgpu::Buffer,
    locate_query_buf: wgpu::Buffer,
    locate_hint_buf: wgpu::Buffer,
    locate_result_buf: wgpu::Buffer,
    locate_capacity: u32,
    // Error tiled pipeline
    error_pipeline: ErrorPipeline,
    error_buffers: ErrorBuffers,
    error_input_buffers: ErrorInputBuffers,
    error_bg0: wgpu::BindGroup,
    error_bg0_b: wgpu::BindGroup,
    error_bg1: wgpu::BindGroup,
    last_sort_in_b: bool,
    // Cached scene data for find_containing_tet and locate_tets
    cached_vertices: Vec<f32>,
    cached_indices: Vec<u32>,
    cached_neighbors: Vec<i32>,
    // Dummy depth view for forward pass (no primitives, all-pass)
    dummy_depth_view: wgpu::TextureView,
    // Scene metadata
    tet_count: u32,
    _vertex_count: u32,
    width: u32,
    height: u32,
    step: u32,
    tile_size: u32,
    param_counts: [u32; 3],
    min_t: f32,
}

#[pymethods]
impl RMeshRenderer {
    /// Create a new renderer.
    ///
    /// Args:
    ///     vertices: [N*3] f32 flat array of vertex positions
    ///     indices: [M*4] u32 flat array of tet indices
    ///     base_colors: [M*3] f32 per-tet base colors (pre-softplus)
    ///     densities: [M] f32 per-tet densities
    ///     color_grads: [M*3] f32 per-tet color gradients
    ///     circumdata: [M*4] f32 circumsphere data (cx, cy, cz, r^2)
    ///     width: render width
    ///     height: render height
    #[new]
    fn new(
        vertices: PyReadonlyArray1<f32>,
        indices: PyReadonlyArray1<u32>,
        base_colors: PyReadonlyArray1<f32>,
        densities: PyReadonlyArray1<f32>,
        color_grads: PyReadonlyArray1<f32>,
        circumdata: PyReadonlyArray1<f32>,
        width: u32,
        height: u32,
    ) -> PyResult<Self> {
        let vertices_slice = vertices.as_slice()?;
        let indices_slice = indices.as_slice()?;
        let base_colors_slice = base_colors.as_slice()?;
        let densities_slice = densities.as_slice()?;
        let color_grads_slice = color_grads.as_slice()?;
        let circumdata_slice = circumdata.as_slice()?;

        let tet_count = indices_slice.len() as u32 / 4;
        let vertex_count = vertices_slice.len() as u32 / 3;

        // Build SceneData
        let scene = rmesh_data::SceneData {
            vertices: vertices_slice.to_vec(),
            indices: indices_slice.to_vec(),
            densities: densities_slice.to_vec(),
            color_grads: color_grads_slice.to_vec(),
            circumdata: circumdata_slice.to_vec(),
            start_pose: [0.0; 7],
            vertex_count,
            tet_count,
        };

        // Initialize wgpu
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to find a suitable GPU adapter: {e}"))
        })?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("rmesh_renderer"),
                required_features: wgpu::Features::SUBGROUP | wgpu::Features::SHADER_FLOAT32_ATOMIC,
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 16,
                    max_storage_buffer_binding_size: 1 << 30, // 1 GiB
                    max_buffer_size: 1 << 30,
                    ..wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            },
        ))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Device error: {e}")))?;

        let color_format = wgpu::TextureFormat::Rgba16Float;

        // Create pipelines
        let fwd_pipelines = ForwardPipelines::new(&device, color_format);
        let bwd_tiled_pipelines = BackwardTiledPipelines::new(&device);
        let loss_pipeline = LossPipeline::new(&device);
        let adam_pipeline = AdamPipeline::new(&device);

        // Upload scene
        let scene_buffers = SceneBuffers::upload(&device, &queue, &scene);

        // Upload material data separately
        let material_buffers = MaterialBuffers::upload(&device, base_colors_slice, color_grads_slice, tet_count);

        // Render targets
        let targets = RenderTargets::new(&device, width, height);

        // Dummy depth texture (cleared to 1.0 = all-pass) for forward pass depth attachment
        let dummy_depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dummy_depth"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let dummy_depth_view = dummy_depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Dummy sh_coeffs buffer (Python training uses sh_degree=0 / base_colors path)
        let dummy_sh = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy_sh_coeffs"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        // Forward bind groups
        let compute_bg = create_compute_bind_group(&device, &fwd_pipelines, &scene_buffers, &material_buffers, &dummy_sh);
        let render_bg = create_render_bind_group(&device, &fwd_pipelines, &scene_buffers, &material_buffers);

        // Tex-to-buffer
        let ttb = TexToBufferPipeline::new(&device, &queue, &targets.color_view, width, height);

        // Gradient + optimizer state
        let grad_buffers =
            GradientBuffers::new(&device, vertex_count, tet_count);
        let mat_grad_buffers = MaterialGradBuffers::new(&device, tet_count);
        let adam_state = AdamState::new(&device, vertex_count, tet_count);
        let mat_adam_state = MaterialAdamState::new(&device, tet_count);

        // Loss buffers
        let loss_buffers = LossBuffers::new(&device, width, height);

        // Loss bind group (non-tiled path, unused but kept for API compat)
        let _loss_bg = create_loss_bind_group(
            &device,
            &loss_pipeline,
            &loss_buffers,
            &ttb.rendered_image,
        );

        // Adam bind groups + per-group uniform buffers (separate to avoid overwrite)
        let adam_uniforms_bufs: Vec<wgpu::Buffer> = (0..3)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("adam_uniforms_{i}")),
                    size: std::mem::size_of::<AdamUniforms>() as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        let adam_bgs = vec![
            create_adam_bind_group(
                &device,
                &adam_pipeline,
                &adam_uniforms_bufs[0],
                &scene_buffers.vertices,
                &grad_buffers.d_vertices,
                &adam_state.m_vertices,
                &adam_state.v_vertices,
            ),
            create_adam_bind_group(
                &device,
                &adam_pipeline,
                &adam_uniforms_bufs[1],
                &scene_buffers.densities,
                &grad_buffers.d_densities,
                &adam_state.m_densities,
                &adam_state.v_densities,
            ),
            create_adam_bind_group(
                &device,
                &adam_pipeline,
                &adam_uniforms_bufs[2],
                &material_buffers.color_grads,
                &mat_grad_buffers.d_color_grads,
                &mat_adam_state.m_color_grads,
                &mat_adam_state.v_color_grads,
            ),
        ];

        let param_counts = [
            vertex_count * 3,
            tet_count,
            tet_count * 3,
        ];

        // Tiled backward infrastructure
        let tile_size = 12u32;
        let tile_pipelines = TilePipelines::new(&device);
        let tile_buffers = TileBuffers::new(&device, tet_count, width, height, tile_size);

        // Radix sort — 64-bit keys: lo=depth(32 bits), hi=tile_id
        let sorting_bits = rmesh_backward::sorting_bits_for_tiles(tile_buffers.num_tiles, rmesh_backward::SortBackend::Drs);
        let radix_pipelines = RadixSortPipelines::new(&device, 2, SortBackend::Drs);
        let radix_state = RadixSortState::new(&device, tile_buffers.max_pairs_pow2, sorting_bits, 2, SortBackend::Drs);
        radix_state.upload_configs(&queue);

        let tile_fill_bg = create_tile_fill_bind_group(&device, &tile_pipelines, &tile_buffers);

        // Forward tiled compute pipeline (4x4 tiles with subgroups)
        let rasterize = RasterizeComputePipeline::new(&device, width, height, 0);

        // A-buffer bind groups (sort result in primary tile_sort_keys/values)
        // Use num_keys_buf as tile_pair_count since scan pipeline writes total_pairs there
        let tile_ranges_bg = create_tile_ranges_bind_group_with_keys(
            &device,
            &tile_pipelines,
            &tile_buffers.tile_sort_keys,
            &tile_buffers.tile_ranges,
            &tile_buffers.tile_uniforms,
            radix_state.num_keys_buf(),
        );
        // B-buffer bind groups (sort result in radix_state.keys_b/values_b)
        let tile_ranges_bg_b = create_tile_ranges_bind_group_with_keys(
            &device,
            &tile_pipelines,
            radix_state.keys_b(),
            &tile_buffers.tile_ranges,
            &tile_buffers.tile_uniforms,
            radix_state.num_keys_buf(),
        );

        // Forward tiled bind groups (A and B buffer variants)
        let rasterize_bg = create_rasterize_bind_group(
            &device,
            &rasterize,
            &scene_buffers.uniforms,
            &scene_buffers.vertices,
            &scene_buffers.indices,
            &material_buffers.colors,
            &scene_buffers.densities,
            &material_buffers.color_grads,
            &tile_buffers.tile_sort_values,
            &tile_buffers.tile_ranges,
            &tile_buffers.tile_uniforms,
        );
        let rasterize_bg_b = create_rasterize_bind_group(
            &device,
            &rasterize,
            &scene_buffers.uniforms,
            &scene_buffers.vertices,
            &scene_buffers.indices,
            &material_buffers.colors,
            &scene_buffers.densities,
            &material_buffers.color_grads,
            radix_state.values_b(),
            &tile_buffers.tile_ranges,
            &tile_buffers.tile_uniforms,
        );

        // Loss bind group using rasterize.rendered_image (for train_step tiled path)
        let loss_bg_tiled = create_loss_bind_group(
            &device,
            &loss_pipeline,
            &loss_buffers,
            &rasterize.rendered_image,
        );

        // Backward tiled bind groups using rasterize.rendered_image (for train_step)
        let (bwd_tiled_bg0_tiled, bwd_tiled_bg1) = create_backward_tiled_bind_groups(
            &device,
            &bwd_tiled_pipelines,
            &scene_buffers.uniforms,
            &loss_buffers.dl_d_image,
            &rasterize.rendered_image,
            &scene_buffers.vertices,
            &scene_buffers.indices,
            &scene_buffers.densities,
            &material_buffers.color_grads,
            &material_buffers.colors,
            &tile_buffers.tile_sort_values,
            &grad_buffers.d_vertices,
            &grad_buffers.d_densities,
            &mat_grad_buffers.d_color_grads,
            &mat_grad_buffers.d_base_colors,
            &tile_buffers.tile_ranges,
            &tile_buffers.tile_uniforms,
        );
        let (bwd_tiled_bg0_tiled_b, _) = create_backward_tiled_bind_groups(
            &device,
            &bwd_tiled_pipelines,
            &scene_buffers.uniforms,
            &loss_buffers.dl_d_image,
            &rasterize.rendered_image,
            &scene_buffers.vertices,
            &scene_buffers.indices,
            &scene_buffers.densities,
            &material_buffers.color_grads,
            &material_buffers.colors,
            radix_state.values_b(),
            &grad_buffers.d_vertices,
            &grad_buffers.d_densities,
            &mat_grad_buffers.d_color_grads,
            &mat_grad_buffers.d_base_colors,
            &tile_buffers.tile_ranges,
            &tile_buffers.tile_uniforms,
        );

        // HW raster backward bind groups using ttb.rendered_image
        let (bwd_tiled_bg0_hw, _) = create_backward_tiled_bind_groups(
            &device,
            &bwd_tiled_pipelines,
            &scene_buffers.uniforms,
            &loss_buffers.dl_d_image,
            &ttb.rendered_image,
            &scene_buffers.vertices,
            &scene_buffers.indices,
            &scene_buffers.densities,
            &material_buffers.color_grads,
            &material_buffers.colors,
            &tile_buffers.tile_sort_values,
            &grad_buffers.d_vertices,
            &grad_buffers.d_densities,
            &mat_grad_buffers.d_color_grads,
            &mat_grad_buffers.d_base_colors,
            &tile_buffers.tile_ranges,
            &tile_buffers.tile_uniforms,
        );
        let (bwd_tiled_bg0_hw_b, _) = create_backward_tiled_bind_groups(
            &device,
            &bwd_tiled_pipelines,
            &scene_buffers.uniforms,
            &loss_buffers.dl_d_image,
            &ttb.rendered_image,
            &scene_buffers.vertices,
            &scene_buffers.indices,
            &scene_buffers.densities,
            &material_buffers.color_grads,
            &material_buffers.colors,
            radix_state.values_b(),
            &grad_buffers.d_vertices,
            &grad_buffers.d_densities,
            &mat_grad_buffers.d_color_grads,
            &mat_grad_buffers.d_base_colors,
            &tile_buffers.tile_ranges,
            &tile_buffers.tile_uniforms,
        );

        // Loss bind group using ttb.rendered_image (HW raster path)
        let loss_bg_hw = create_loss_bind_group(
            &device,
            &loss_pipeline,
            &loss_buffers,
            &ttb.rendered_image,
        );

        // Scan-based tile pipeline infrastructure (RTS prefix scan)
        let scan_pipelines = ScanPipelines::new(&device);
        let scan_buffers = ScanBuffers::new(&device, tet_count);
        let prepare_dispatch_bg = create_prepare_dispatch_bind_group(
            &device,
            &scan_pipelines,
            &scene_buffers.indirect_args,
            &scan_buffers,
        );
        let rts_bg = create_rts_bind_group(
            &device,
            &scan_pipelines,
            &scene_buffers.tiles_touched,
            &scan_buffers,
        );
        let tile_gen_scan_bg = create_tile_gen_scan_bind_group(
            &device,
            &scan_pipelines,
            &tile_buffers,
            &scene_buffers.uniforms,
            &scene_buffers.vertices,
            &scene_buffers.indices,
            &scene_buffers.compact_tet_ids,
            &scene_buffers.circumdata,
            &scene_buffers.tiles_touched,
            &scan_buffers,
            radix_state.num_keys_buf(),
        );

        // Ray trace setup
        let neighbors = compute_tet_neighbors(&scene.indices, tet_count as usize);
        let bvh = build_boundary_bvh(&scene.vertices, &scene.indices, &neighbors, tet_count as usize);
        let rt_pipeline = RayTracePipeline::new(&device, width, height, 0);
        let rt_buffers = RayTraceBuffers::new(&device, &neighbors, &bvh);
        let rt_bind_group = create_raytrace_bind_group(&device, &rt_pipeline, &scene_buffers, &material_buffers, &rt_buffers);

        // Point location pipeline
        let locate_pipeline = LocatePipeline::new(&device);
        let locate_uniforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("locate_uniforms"),
            size: std::mem::size_of::<LocateUniforms>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let locate_query_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("locate_query"),
            size: 12, // 1 point * 3 floats * 4 bytes
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let locate_hint_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("locate_hint"),
            size: 4, // 1 i32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let locate_result_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("locate_result"),
            size: 4, // 1 i32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Error tiled pipeline
        let error_pipeline = ErrorPipeline::new(&device);
        let error_buffers = ErrorBuffers::new(&device, tet_count);
        let error_input_buffers = ErrorInputBuffers::new(&device, width, height);
        let error_bg0 = create_error_bg0(
            &device, &error_pipeline,
            &scene_buffers.uniforms, &scene_buffers.vertices, &scene_buffers.indices,
            &material_buffers.colors, &scene_buffers.densities, &material_buffers.color_grads,
            &tile_buffers.tile_sort_values, &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms,
        );
        let error_bg0_b = create_error_bg0(
            &device, &error_pipeline,
            &scene_buffers.uniforms, &scene_buffers.vertices, &scene_buffers.indices,
            &material_buffers.colors, &scene_buffers.densities, &material_buffers.color_grads,
            radix_state.values_b(), &tile_buffers.tile_ranges, &tile_buffers.tile_uniforms,
        );
        let error_bg1 = create_error_bg1(
            &device, &error_pipeline,
            &error_input_buffers, &error_buffers,
        );

        Ok(Self {
            device,
            queue,
            scene_buffers,
            material_buffers,
            fwd_pipelines,
            targets,
            ttb,
            bwd_tiled_pipelines,
            grad_buffers,
            mat_grad_buffers,
            loss_buffers,
            loss_pipeline,
            adam_pipeline,
            _adam_state: adam_state,
            _mat_adam_state: mat_adam_state,
            tile_pipelines,
            tile_buffers,
            radix_pipelines,
            radix_state,
            tile_fill_bg,
            tile_ranges_bg,
            tile_ranges_bg_b,
            rasterize,
            rasterize_bg,
            rasterize_bg_b,
            compute_bg,
            render_bg,
            loss_bg_tiled,
            bwd_tiled_bg0_tiled,
            bwd_tiled_bg0_tiled_b,
            bwd_tiled_bg1,
            bwd_tiled_bg0_hw,
            bwd_tiled_bg0_hw_b,
            _loss_bg_hw: loss_bg_hw,
            adam_bgs,
            adam_uniforms_bufs,
            scan_pipelines,
            scan_buffers,
            prepare_dispatch_bg,
            rts_bg,
            tile_gen_scan_bg,
            rt_pipeline,
            rt_buffers,
            rt_bind_group,
            locate_pipeline,
            locate_uniforms_buf,
            locate_query_buf,
            locate_hint_buf,
            locate_result_buf,
            locate_capacity: 1,
            error_pipeline,
            error_buffers,
            error_input_buffers,
            error_bg0,
            error_bg0_b,
            error_bg1,
            last_sort_in_b: false,
            cached_vertices: vertices_slice.to_vec(),
            cached_indices: indices_slice.to_vec(),
            cached_neighbors: neighbors,
            dummy_depth_view,
            tet_count,
            _vertex_count: vertex_count,
            width,
            height,
            step: 0,
            tile_size: 12,
            param_counts,
            min_t: 0.0,
        })
    }

    /// Set the minimum ray-origin offset along view direction (matches Slang camera.min_t).
    fn set_min_t(&mut self, min_t: f32) {
        self.min_t = min_t;
    }

    /// Run the forward rendering pipeline.
    ///
    /// Args:
    ///     cam_pos: [3] f32 camera position
    ///     vp: [16] f32 column-major view-projection matrix
    ///     c2w_intrinsics: [16] f32 (c2w_col0[4], c2w_col1[4], c2w_col2[4], intrinsics[4])
    ///
    /// Returns:
    ///     numpy array [H, W, 4] f32 (premultiplied RGBA)
    fn forward<'py>(
        &mut self,
        py: Python<'py>,
        cam_pos: PyReadonlyArray1<f32>,
        vp: PyReadonlyArray1<f32>,
        c2w_intrinsics: PyReadonlyArray1<f32>,
    ) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let cam_pos_slice = cam_pos.as_slice()?;
        let vp_slice = vp.as_slice()?;
        let ci_slice = c2w_intrinsics.as_slice()?;

        let cam = Vec3::new(cam_pos_slice[0], cam_pos_slice[1], cam_pos_slice[2]);
        let vp_mat = mat4_from_flat(vp_slice);
        let (c2w, intrinsics) = c2w_from_flat(ci_slice);

        let uniforms = make_uniforms(
            vp_mat,
            c2w,
            intrinsics,
            cam,
            self.width as f32,
            self.height as f32,
            self.tet_count,
            self.step,
            self.tile_size,
            self.min_t,
            0,
            0.01,
            1000.0,
        );

        // Upload uniforms
        self.queue.write_buffer(
            &self.scene_buffers.uniforms,
            0,
            bytemuck::bytes_of(&uniforms),
        );

        // Record forward pass + tex-to-buffer
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("forward"),
            });
        // Clear color target and depth before forward pass (which uses LoadOp::Load)
        {
            let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear_for_forward"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.targets.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.dummy_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
        }
        record_forward_pass(
            &mut encoder,
            &self.fwd_pipelines,
            &self.scene_buffers,
            &self.targets,
            &self.compute_bg,
            &self.render_bg,
            self.tet_count,
            &self.queue,
            &self.dummy_depth_view,
        );
        record_tex_to_buffer(&mut encoder, &self.ttb);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back rendered image
        let data = read_buffer_f32(&self.device, &self.queue, &self.ttb.rendered_image);

        // Convert to [H, W, 4] numpy array
        let array = Array3::from_shape_vec(
            (self.height as usize, self.width as usize, 4),
            data,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}")))?;

        Ok(array.into_pyarray(py))
    }

    /// Run the compute-based tiled forward rendering pipeline.
    ///
    /// Uses 4x4 tiles with subgroup shuffles (no hardware rasterization).
    /// Includes tile generation, radix sort, tile ranges, and compute forward.
    ///
    /// Args:
    ///     cam_pos: [3] f32 camera position
    ///     vp: [16] f32 column-major view-projection matrix
    ///     c2w_intrinsics: [16] f32 (c2w_col0[4], c2w_col1[4], c2w_col2[4], intrinsics[4])
    ///     render_aux: if True, also return aux [H,W,8] (normals, depth, entropy)
    ///
    /// Returns:
    ///     numpy array [H, W, 4] f32 (RGBA), or tuple (rgba, aux) if render_aux=True
    #[pyo3(signature = (cam_pos, vp, c2w_intrinsics, render_aux=false))]
    fn forward_tiled<'py>(
        &mut self,
        py: Python<'py>,
        cam_pos: PyReadonlyArray1<f32>,
        vp: PyReadonlyArray1<f32>,
        c2w_intrinsics: PyReadonlyArray1<f32>,
        render_aux: bool,
    ) -> PyResult<Py<PyAny>> {
        let cam_pos_slice = cam_pos.as_slice()?;
        let vp_slice = vp.as_slice()?;
        let ci_slice = c2w_intrinsics.as_slice()?;

        let cam = Vec3::new(cam_pos_slice[0], cam_pos_slice[1], cam_pos_slice[2]);
        let vp_mat = mat4_from_flat(vp_slice);
        let (c2w, intrinsics) = c2w_from_flat(ci_slice);

        let uniforms = make_uniforms(
            vp_mat,
            c2w,
            intrinsics,
            cam,
            self.width as f32,
            self.height as f32,
            self.tet_count,
            self.step,
            self.tile_size,
            self.min_t,
            0,
            0.01,
            1000.0,
        );

        // Upload uniforms
        self.queue.write_buffer(
            &self.scene_buffers.uniforms,
            0,
            bytemuck::bytes_of(&uniforms),
        );

        // Upload tile uniforms (visible_tet_count=0 placeholder — scan shaders read from GPU)
        let tile_uni = TileUniforms {
            screen_width: self.width,
            screen_height: self.height,
            tile_size: self.tile_size,
            tiles_x: self.tile_buffers.tiles_x,
            tiles_y: self.tile_buffers.tiles_y,
            num_tiles: self.tile_buffers.num_tiles,
            visible_tet_count: 0,
            _pad: [0; 5],
        };
        self.queue.write_buffer(
            &self.tile_buffers.tile_uniforms,
            0,
            bytemuck::bytes_of(&tile_uni),
        );

        // Single encoder: forward compute + scan tile pipeline + radix sort + forward tiled
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rasterize_compute"),
                });

            // Clear tile buffers + forward output + aux
            encoder.clear_buffer(&self.tile_buffers.tile_ranges, 0, None);
            encoder.clear_buffer(&self.rasterize.rendered_image, 0, None);
            if render_aux {
                encoder.clear_buffer(&self.rasterize.aux_image, 0, None);
            }

            // Forward compute (SH eval + cull + tiles_touched + compact_tet_ids, no sort)
            record_project_compute(
                &mut encoder,
                &self.fwd_pipelines,
                &self.scene_buffers,
                &self.compute_bg,
                self.tet_count,
                &self.queue,
            );

            // RTS scan-based tile pipeline
            rmesh_backward::record_scan_tile_pipeline(
                &mut encoder,
                &self.scan_pipelines,
                &self.tile_pipelines,
                &self.prepare_dispatch_bg,
                &self.rts_bg,
                &self.tile_fill_bg,
                &self.tile_gen_scan_bg,
                &self.scan_buffers,
                &self.tile_buffers,
            );

            // Radix sort
            let result_in_b = rmesh_backward::record_radix_sort(
                &mut encoder,
                &self.device,
                &self.radix_pipelines,
                &self.radix_state,
                &self.tile_buffers.tile_sort_keys,
                &self.tile_buffers.tile_sort_values,
            );
            self.last_sort_in_b = result_in_b;

            let (ranges_bg, fwd_bg) = if result_in_b {
                (&self.tile_ranges_bg_b, &self.rasterize_bg_b)
            } else {
                (&self.tile_ranges_bg, &self.rasterize_bg)
            };

            // Tile ranges
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("tile_ranges"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.tile_pipelines.tile_ranges_pipeline);
                pass.set_bind_group(0, ranges_bg, &[]);
                let total = (self.tile_buffers.max_pairs_pow2 + 255) / 256;
                if total <= 65535 {
                    pass.dispatch_workgroups(total, 1, 1);
                } else {
                    pass.dispatch_workgroups(65535, (total + 65534) / 65535, 1);
                }
            }

            // Forward tiled compute
            record_rasterize_compute(
                &mut encoder,
                &self.rasterize,
                fwd_bg,
                self.tile_buffers.num_tiles,
            );

            self.queue.submit(std::iter::once(encoder.finish()));
        }

        // Read back rendered image
        let data = read_buffer_f32(&self.device, &self.queue, &self.rasterize.rendered_image);

        let rgba = Array3::from_shape_vec(
            (self.height as usize, self.width as usize, 4),
            data,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}")))?;

        if render_aux {
            let aux_stride = self.rasterize.aux_stride as usize;
            let aux_data = read_buffer_f32(&self.device, &self.queue, &self.rasterize.aux_image);
            let aux = Array3::from_shape_vec(
                (self.height as usize, self.width as usize, aux_stride),
                aux_data,
            )
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}")))?;
            let tuple = pyo3::types::PyTuple::new(py, &[
                rgba.into_pyarray(py).into_any(),
                aux.into_pyarray(py).into_any(),
            ])?;
            Ok(tuple.into())
        } else {
            Ok(rgba.into_pyarray(py).into_any().unbind())
        }
    }

    /// Run the ray tracing forward pipeline (adjacency traversal, no sorting).
    ///
    /// Args:
    ///     cam_pos: [3] f32 camera position
    ///     vp: [16] f32 column-major view-projection matrix
    ///     c2w_intrinsics: [16] f32 (c2w_col0[4], c2w_col1[4], c2w_col2[4], intrinsics[4])
    ///     ray_origins: optional [N*3] f32 per-pixel ray origins (multi-origin support)
    ///     ray_dirs: optional [N*3] f32 per-pixel ray directions
    ///     start_tets: optional [N] i32 per-pixel start tet IDs
    ///     render_aux: if True, also return aux [H,W,8] (normals, depth, entropy)
    ///
    /// Returns:
    ///     numpy array [H, W, 4] f32 (RGBA), or tuple (rgba, aux) if render_aux=True
    #[pyo3(signature = (cam_pos, vp, c2w_intrinsics, ray_origins=None, ray_dirs=None, start_tets=None, render_aux=false))]
    fn forward_raytrace<'py>(
        &mut self,
        py: Python<'py>,
        cam_pos: PyReadonlyArray1<f32>,
        vp: PyReadonlyArray1<f32>,
        c2w_intrinsics: PyReadonlyArray1<f32>,
        ray_origins: Option<PyReadonlyArray1<f32>>,
        ray_dirs: Option<PyReadonlyArray1<f32>>,
        start_tets: Option<PyReadonlyArray1<i32>>,
        render_aux: bool,
    ) -> PyResult<Py<PyAny>> {
        let cam_pos_slice = cam_pos.as_slice()?;
        let vp_slice = vp.as_slice()?;
        let ci_slice = c2w_intrinsics.as_slice()?;

        let cam = Vec3::new(cam_pos_slice[0], cam_pos_slice[1], cam_pos_slice[2]);
        let vp_mat = mat4_from_flat(vp_slice);
        let (c2w, intrinsics) = c2w_from_flat(ci_slice);

        let mut uniforms = make_uniforms(
            vp_mat,
            c2w,
            intrinsics,
            cam,
            self.width as f32,
            self.height as f32,
            self.tet_count,
            self.step,
            self.tile_size,
            self.min_t,
            0,
            0.01,
            1000.0,
        );

        let has_custom_rays = ray_origins.is_some() && ray_dirs.is_some();

        if has_custom_rays {
            uniforms.ray_mode = 1;

            let origins_slice = ray_origins.as_ref().unwrap().as_slice()?;
            let dirs_slice = ray_dirs.as_ref().unwrap().as_slice()?;
            let n_pixels = (self.width * self.height) as usize;

            if origins_slice.len() != n_pixels * 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("ray_origins must be [W*H*3] = {} elements, got {}", n_pixels * 3, origins_slice.len())
                ));
            }
            if dirs_slice.len() != n_pixels * 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("ray_dirs must be [W*H*3] = {} elements, got {}", n_pixels * 3, dirs_slice.len())
                ));
            }

            // Resize and upload ray buffers
            let ray_buf_size = (n_pixels * 3 * 4) as u64;
            if self.rt_buffers.ray_origins.size() < ray_buf_size {
                self.rt_buffers.ray_origins = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("ray_origins"),
                    size: ray_buf_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.rt_buffers.ray_dirs = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("ray_dirs"),
                    size: ray_buf_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                // Recreate bind group with new buffers
                self.rt_bind_group = create_raytrace_bind_group(
                    &self.device, &self.rt_pipeline, &self.scene_buffers,
                    &self.material_buffers, &self.rt_buffers,
                );
            }
            self.queue.write_buffer(&self.rt_buffers.ray_origins, 0, bytemuck::cast_slice(origins_slice));
            self.queue.write_buffer(&self.rt_buffers.ray_dirs, 0, bytemuck::cast_slice(dirs_slice));

            // Upload per-pixel start tets
            if let Some(ref tets) = start_tets {
                let tets_slice = tets.as_slice()?;
                let start_tet_size = (n_pixels * 4) as u64;
                if self.rt_buffers.start_tet.size() < start_tet_size {
                    self.rt_buffers.start_tet = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("start_tet"),
                        size: start_tet_size,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    self.rt_bind_group = create_raytrace_bind_group(
                        &self.device, &self.rt_pipeline, &self.scene_buffers,
                        &self.material_buffers, &self.rt_buffers,
                    );
                }
                self.queue.write_buffer(&self.rt_buffers.start_tet, 0, bytemuck::cast_slice(tets_slice));
            } else {
                // All -1 (use BVH entry for all rays)
                let neg_ones = vec![-1i32; n_pixels];
                let start_tet_size = (n_pixels * 4) as u64;
                if self.rt_buffers.start_tet.size() < start_tet_size {
                    self.rt_buffers.start_tet = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("start_tet"),
                        size: start_tet_size,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    self.rt_bind_group = create_raytrace_bind_group(
                        &self.device, &self.rt_pipeline, &self.scene_buffers,
                        &self.material_buffers, &self.rt_buffers,
                    );
                }
                self.queue.write_buffer(&self.rt_buffers.start_tet, 0, bytemuck::cast_slice(&neg_ones));
            }
        } else {
            uniforms.ray_mode = 0;

            // Find containing tet (CPU walk with brute-force fallback)
            let start_tet = find_containing_tet_walk(
                &self.cached_vertices,
                &self.cached_indices,
                &self.cached_neighbors,
                self.tet_count as usize,
                cam,
                0,
            )
            .map(|t| t as i32)
            .unwrap_or(-1);
            self.queue.write_buffer(
                &self.rt_buffers.start_tet,
                0,
                bytemuck::cast_slice(&[start_tet]),
            );
        }

        // Upload uniforms
        self.queue.write_buffer(
            &self.scene_buffers.uniforms,
            0,
            bytemuck::bytes_of(&uniforms),
        );

        // Single encoder: forward compute (SH eval) + raytrace
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("forward_raytrace"),
                });

            encoder.clear_buffer(&self.rt_pipeline.rendered_image, 0, None);
            if render_aux {
                encoder.clear_buffer(&self.rt_pipeline.aux_image, 0, None);
            }

            // Forward compute (SH eval → colors_buf)
            record_project_compute(
                &mut encoder,
                &self.fwd_pipelines,
                &self.scene_buffers,
                &self.compute_bg,
                self.tet_count,
                &self.queue,
            );

            // Ray trace
            record_raytrace(
                &mut encoder,
                &self.rt_pipeline,
                &self.rt_bind_group,
                self.width,
                self.height,
            );

            self.queue.submit(std::iter::once(encoder.finish()));
        }

        // Read back rendered image
        let data = read_buffer_f32(&self.device, &self.queue, &self.rt_pipeline.rendered_image);

        let rgba = Array3::from_shape_vec(
            (self.height as usize, self.width as usize, 4),
            data,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}")))?;

        if render_aux {
            let aux_stride = self.rt_pipeline.aux_stride as usize;
            let aux_data = read_buffer_f32(&self.device, &self.queue, &self.rt_pipeline.aux_image);
            let aux = Array3::from_shape_vec(
                (self.height as usize, self.width as usize, aux_stride),
                aux_data,
            )
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}")))?;
            let tuple = pyo3::types::PyTuple::new(py, &[
                rgba.into_pyarray(py).into_any(),
                aux.into_pyarray(py).into_any(),
            ])?;
            Ok(tuple.into())
        } else {
            Ok(rgba.into_pyarray(py).into_any().unbind())
        }
    }

    /// Compute per-tet error statistics using the tiled pipeline.
    ///
    /// Must be called after `forward_tiled()` — reuses the sorted tile data from
    /// the most recent forward pass.
    ///
    /// Args:
    ///     pixel_err: [H*W] f32 per-pixel L1 error (flattened row-major)
    ///     ssim_err: [H*W] f32 per-pixel SSIM error (flattened row-major)
    ///
    /// Returns:
    ///     tuple (tet_err [T, 16] f32, tet_count [T, 2] i32)
    fn forward_error<'py>(
        &mut self,
        py: Python<'py>,
        pixel_err: PyReadonlyArray1<f32>,
        ssim_err: PyReadonlyArray1<f32>,
    ) -> PyResult<Py<PyAny>> {
        let pe = pixel_err.as_slice()?;
        let se = ssim_err.as_slice()?;

        let n_pixels = (self.width * self.height) as usize;
        if pe.len() != n_pixels {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("pixel_err must have {} elements, got {}", n_pixels, pe.len())
            ));
        }
        if se.len() != n_pixels {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("ssim_err must have {} elements, got {}", n_pixels, se.len())
            ));
        }

        // Upload pixel errors
        self.queue.write_buffer(
            &self.error_input_buffers.pixel_err,
            0,
            bytemuck::cast_slice(pe),
        );
        self.queue.write_buffer(
            &self.error_input_buffers.ssim_err,
            0,
            bytemuck::cast_slice(se),
        );

        // Build encoder
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("forward_error"),
            },
        );

        // Clear error output buffers
        encoder.clear_buffer(&self.error_buffers.tet_err, 0, None);
        encoder.clear_buffer(&self.error_buffers.tet_count, 0, None);

        // Select bind group based on last sort result
        let bg0 = if self.last_sort_in_b {
            &self.error_bg0_b
        } else {
            &self.error_bg0
        };

        // Dispatch error pass
        record_error_pass(
            &mut encoder,
            &self.error_pipeline,
            bg0,
            &self.error_bg1,
            self.tile_buffers.num_tiles,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back tet_err as [T, 16] f32
        let tet_err_data = read_buffer_f32(
            &self.device, &self.queue, &self.error_buffers.tet_err,
        );

        let tet_err_array = Array2::from_shape_vec(
            (self.tet_count as usize, 16),
            tet_err_data,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}")))?;

        // Read back tet_count as [T, 2] i32
        let tet_count_raw = read_buffer_raw(
            &self.device, &self.queue, &self.error_buffers.tet_count,
        );
        let tet_count_i32: &[i32] = bytemuck::cast_slice(&tet_count_raw);

        let tet_count_array = Array2::from_shape_vec(
            (self.tet_count as usize, 2),
            tet_count_i32.to_vec(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}")))?;

        let tuple = pyo3::types::PyTuple::new(py, &[
            tet_err_array.into_pyarray(py).into_any(),
            tet_count_array.into_pyarray(py).into_any(),
        ])?;
        Ok(tuple.into())
    }

    /// Locate the containing tet for each query point via GPU adjacency walking.
    ///
    /// Args:
    ///     query_points: [N*3] f32 flat array of 3D query points
    ///     hint_tet: global starting tet hint (default 0)
    ///
    /// Returns:
    ///     numpy array [N] i32 — tet index for each query, or -1 if outside mesh
    #[pyo3(signature = (query_points, hint_tet=0))]
    fn locate_tets<'py>(
        &mut self,
        py: Python<'py>,
        query_points: PyReadonlyArray1<f32>,
        hint_tet: i32,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let pts = query_points.as_slice()?;
        if pts.len() % 3 != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("query_points length must be divisible by 3, got {}", pts.len())
            ));
        }
        let n = (pts.len() / 3) as u32;
        if n == 0 {
            return Ok(Array1::<i32>::default(0).into_pyarray(py));
        }

        // Resize buffers if needed
        if n > self.locate_capacity {
            self.locate_query_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("locate_query"),
                size: (n as u64) * 3 * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.locate_hint_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("locate_hint"),
                size: (n as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.locate_result_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("locate_result"),
                size: (n as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            self.locate_capacity = n;
        }

        // Upload query points
        self.queue.write_buffer(&self.locate_query_buf, 0, bytemuck::cast_slice(pts));

        // Upload uniforms (use global hint_tet, so hint_tets buffer is unused)
        let uni = LocateUniforms {
            num_queries: n,
            hint_tet,
            tet_count: self.tet_count,
            _pad: 0,
        };
        self.queue.write_buffer(&self.locate_uniforms_buf, 0, bytemuck::bytes_of(&uni));

        // Create bind group
        let bg = create_locate_bind_group(
            &self.device,
            &self.locate_pipeline,
            &self.locate_uniforms_buf,
            &self.scene_buffers.vertices,
            &self.scene_buffers.indices,
            &self.rt_buffers.tet_neighbors,
            &self.locate_query_buf,
            &self.locate_hint_buf,
            &self.locate_result_buf,
        );

        // Record and submit
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("locate_tets"),
        });
        record_locate(&mut encoder, &self.locate_pipeline, &bg, n);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back results
        let raw = read_buffer_raw(&self.device, &self.queue, &self.locate_result_buf);
        let result_i32: &[i32] = bytemuck::cast_slice(&raw);
        let result = Array1::from_vec(result_i32[..n as usize].to_vec());

        Ok(result.into_pyarray(py))
    }

    /// Read back the auxiliary buffer from the last forward pass.
    ///
    /// Returns:
    ///     numpy array [H, W, 4] f32 with (t_min, t_max, optical_depth, dist)
    ///     for the last fragment rendered at each pixel.
    fn read_aux<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let bytes_per_pixel: u32 = 16; // Rgba32Float = 4 × f32
        let unpadded_bpr = self.width * bytes_per_pixel;
        let aligned_bpr = (unpadded_bpr + 255) & !255;
        let buf_size = (aligned_bpr * self.height) as u64;

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aux_readback"),
            size: buf_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("aux_copy"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.targets.aux0_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned_bpr),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        receiver
            .recv()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Map recv: {e}")))?
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Map error: {e}")))?;

        let data = slice.get_mapped_range();
        let mut result = Vec::with_capacity((self.width * self.height * 4) as usize);
        for row in 0..self.height {
            let start = (row * aligned_bpr) as usize;
            let end = start + (self.width * bytes_per_pixel) as usize;
            let row_f32: &[f32] = bytemuck::cast_slice(&data[start..end]);
            result.extend_from_slice(row_f32);
        }
        drop(data);
        staging.unmap();

        let array = Array3::from_shape_vec(
            (self.height as usize, self.width as usize, 4),
            result,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}")))?;

        Ok(array.into_pyarray(py))
    }

    /// Read back the instance_count from the indirect draw args buffer.
    /// This is the number of tets that passed frustum culling in the compute shader.
    fn read_instance_count(&self) -> PyResult<u32> {
        let raw = read_buffer_raw(&self.device, &self.queue, &self.scene_buffers.indirect_args);
        // DrawIndirectCommand: [vertex_count, instance_count, first_vertex, first_instance]
        let u32s: &[u32] = bytemuck::cast_slice(&raw);
        Ok(u32s[1])
    }

    /// Run the backward pass given upstream gradients.
    ///
    /// Args:
    ///     dl_d_image: [H, W, 4] f32 gradient of loss w.r.t. rendered image
    ///
    /// Returns:
    ///     dict with keys 'd_vertices', 'd_densities', 'd_color_grads',
    ///     each a 1D numpy f32 array.
    fn backward<'py>(
        &mut self,
        py: Python<'py>,
        dl_d_image: PyReadonlyArray3<f32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dl_data = dl_d_image.as_slice()?;

        // Upload gradient
        self.queue.write_buffer(
            &self.loss_buffers.dl_d_image,
            0,
            bytemuck::cast_slice(dl_data),
        );

        // Upload tile uniforms (visible_tet_count=0 placeholder — scan shaders read from GPU)
        let tile_uni = TileUniforms {
            screen_width: self.width,
            screen_height: self.height,
            tile_size: self.tile_size,
            tiles_x: self.tile_buffers.tiles_x,
            tiles_y: self.tile_buffers.tiles_y,
            num_tiles: self.tile_buffers.num_tiles,
            visible_tet_count: 0,
            _pad: [0; 5],
        };
        self.queue.write_buffer(
            &self.tile_buffers.tile_uniforms,
            0,
            bytemuck::bytes_of(&tile_uni),
        );

        // Clear gradient buffers + tile ranges
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("backward"),
            });
        encoder.clear_buffer(&self.grad_buffers.d_vertices, 0, None);
        encoder.clear_buffer(&self.grad_buffers.d_densities, 0, None);
        encoder.clear_buffer(&self.mat_grad_buffers.d_base_colors, 0, None);
        encoder.clear_buffer(&self.mat_grad_buffers.d_color_grads, 0, None);
        encoder.clear_buffer(&self.tile_buffers.tile_ranges, 0, None);

        // RTS scan-based tile pipeline (reads compact_tet_ids + tiles_touched from forward compute)
        rmesh_backward::record_scan_tile_pipeline(
            &mut encoder,
            &self.scan_pipelines,
            &self.tile_pipelines,
            &self.prepare_dispatch_bg,
            &self.rts_bg,
            &self.tile_fill_bg,
            &self.tile_gen_scan_bg,
            &self.scan_buffers,
            &self.tile_buffers,
        );

        // Radix sort
        let result_in_b = rmesh_backward::record_radix_sort(
            &mut encoder,
            &self.device,
            &self.radix_pipelines,
            &self.radix_state,
            &self.tile_buffers.tile_sort_keys,
            &self.tile_buffers.tile_sort_values,
        );

        // Use _tiled bind groups that reference rasterize.rendered_image
        // (backward() is called after forward_tiled() in the autograd path)
        let (ranges_bg, bwd_bg0) = if result_in_b {
            (&self.tile_ranges_bg_b, &self.bwd_tiled_bg0_tiled_b)
        } else {
            (&self.tile_ranges_bg, &self.bwd_tiled_bg0_tiled)
        };

        // Tile ranges
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tile_ranges"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.tile_pipelines.tile_ranges_pipeline);
            pass.set_bind_group(0, ranges_bg, &[]);
            let total = (self.tile_buffers.max_pairs_pow2 + 255) / 256;
            if total <= 65535 {
                pass.dispatch_workgroups(total, 1, 1);
            } else {
                pass.dispatch_workgroups(65535, (total + 65534) / 65535, 1);
            }
        }

        // Backward tiled compute
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("backward_tiled"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bwd_tiled_pipelines.pipeline);
            pass.set_bind_group(0, bwd_bg0, &[]);
            pass.set_bind_group(1, &self.bwd_tiled_bg1, &[]);
            let num_tiles = self.tile_buffers.num_tiles;
            if num_tiles <= 65535 {
                pass.dispatch_workgroups(num_tiles, 1, 1);
            } else {
                let x = 65535u32;
                let y = (num_tiles + x - 1) / x;
                pass.dispatch_workgroups(x, y, 1);
            }
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back gradients -- the grad buffers store atomic<u32> that are
        // bitcast f32. The raw bytes are already valid f32 on readback.
        let d_verts = read_buffer_f32(&self.device, &self.queue, &self.grad_buffers.d_vertices);
        let d_dens = read_buffer_f32(&self.device, &self.queue, &self.grad_buffers.d_densities);
        let d_base_cols =
            read_buffer_f32(&self.device, &self.queue, &self.mat_grad_buffers.d_base_colors);
        let d_grads =
            read_buffer_f32(&self.device, &self.queue, &self.mat_grad_buffers.d_color_grads);

        let dict = PyDict::new(py);
        dict.set_item(
            "d_vertices",
            Array1::from_vec(d_verts).into_pyarray(py),
        )?;
        dict.set_item(
            "d_densities",
            Array1::from_vec(d_dens).into_pyarray(py),
        )?;
        dict.set_item(
            "d_base_colors",
            Array1::from_vec(d_base_cols).into_pyarray(py),
        )?;
        dict.set_item(
            "d_color_grads",
            Array1::from_vec(d_grads).into_pyarray(py),
        )?;
        Ok(dict)
    }

    /// Run the backward pass for HW raster forward path.
    ///
    /// Same as backward() but uses ttb.rendered_image (from HW raster + tex-to-buffer)
    /// instead of rasterize.rendered_image (from tiled forward).
    ///
    /// Call after forward() (not forward_tiled()).
    ///
    /// Args:
    ///     dl_d_image: [H, W, 4] f32 gradient of loss w.r.t. rendered image
    ///
    /// Returns:
    ///     dict with keys 'd_vertices', 'd_densities', 'd_color_grads', 'd_base_colors',
    ///     each a 1D numpy f32 array.
    fn backward_hw<'py>(
        &mut self,
        py: Python<'py>,
        dl_d_image: PyReadonlyArray3<f32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dl_data = dl_d_image.as_slice()?;

        // Upload gradient
        self.queue.write_buffer(
            &self.loss_buffers.dl_d_image,
            0,
            bytemuck::cast_slice(dl_data),
        );

        // Upload tile uniforms
        let tile_uni = TileUniforms {
            screen_width: self.width,
            screen_height: self.height,
            tile_size: self.tile_size,
            tiles_x: self.tile_buffers.tiles_x,
            tiles_y: self.tile_buffers.tiles_y,
            num_tiles: self.tile_buffers.num_tiles,
            visible_tet_count: 0,
            _pad: [0; 5],
        };
        self.queue.write_buffer(
            &self.tile_buffers.tile_uniforms,
            0,
            bytemuck::bytes_of(&tile_uni),
        );

        // Clear gradient buffers + tile ranges
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("backward_hw"),
            });
        encoder.clear_buffer(&self.grad_buffers.d_vertices, 0, None);
        encoder.clear_buffer(&self.grad_buffers.d_densities, 0, None);
        encoder.clear_buffer(&self.mat_grad_buffers.d_base_colors, 0, None);
        encoder.clear_buffer(&self.mat_grad_buffers.d_color_grads, 0, None);
        encoder.clear_buffer(&self.tile_buffers.tile_ranges, 0, None);

        // RTS scan-based tile pipeline
        rmesh_backward::record_scan_tile_pipeline(
            &mut encoder,
            &self.scan_pipelines,
            &self.tile_pipelines,
            &self.prepare_dispatch_bg,
            &self.rts_bg,
            &self.tile_fill_bg,
            &self.tile_gen_scan_bg,
            &self.scan_buffers,
            &self.tile_buffers,
        );

        // Radix sort
        let result_in_b = rmesh_backward::record_radix_sort(
            &mut encoder,
            &self.device,
            &self.radix_pipelines,
            &self.radix_state,
            &self.tile_buffers.tile_sort_keys,
            &self.tile_buffers.tile_sort_values,
        );

        // Use HW bind groups that reference ttb.rendered_image
        let (ranges_bg, bwd_bg0) = if result_in_b {
            (&self.tile_ranges_bg_b, &self.bwd_tiled_bg0_hw_b)
        } else {
            (&self.tile_ranges_bg, &self.bwd_tiled_bg0_hw)
        };

        // Tile ranges
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tile_ranges"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.tile_pipelines.tile_ranges_pipeline);
            pass.set_bind_group(0, ranges_bg, &[]);
            let total = (self.tile_buffers.max_pairs_pow2 + 255) / 256;
            if total <= 65535 {
                pass.dispatch_workgroups(total, 1, 1);
            } else {
                pass.dispatch_workgroups(65535, (total + 65534) / 65535, 1);
            }
        }

        // Backward tiled compute
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("backward_tiled"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bwd_tiled_pipelines.pipeline);
            pass.set_bind_group(0, bwd_bg0, &[]);
            pass.set_bind_group(1, &self.bwd_tiled_bg1, &[]);
            let num_tiles = self.tile_buffers.num_tiles;
            if num_tiles <= 65535 {
                pass.dispatch_workgroups(num_tiles, 1, 1);
            } else {
                let x = 65535u32;
                let y = (num_tiles + x - 1) / x;
                pass.dispatch_workgroups(x, y, 1);
            }
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back gradients
        let d_verts = read_buffer_f32(&self.device, &self.queue, &self.grad_buffers.d_vertices);
        let d_dens = read_buffer_f32(&self.device, &self.queue, &self.grad_buffers.d_densities);
        let d_base_cols =
            read_buffer_f32(&self.device, &self.queue, &self.mat_grad_buffers.d_base_colors);
        let d_grads =
            read_buffer_f32(&self.device, &self.queue, &self.mat_grad_buffers.d_color_grads);

        let dict = PyDict::new(py);
        dict.set_item("d_vertices", Array1::from_vec(d_verts).into_pyarray(py))?;
        dict.set_item("d_densities", Array1::from_vec(d_dens).into_pyarray(py))?;
        dict.set_item("d_base_colors", Array1::from_vec(d_base_cols).into_pyarray(py))?;
        dict.set_item("d_color_grads", Array1::from_vec(d_grads).into_pyarray(py))?;
        Ok(dict)
    }

    /// Backward pass for raytrace forward (not yet implemented).
    fn backward_raytrace<'py>(
        &mut self,
        _py: Python<'py>,
        _dl_d_image: PyReadonlyArray3<f32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Raytrace backward not yet implemented"
        ))
    }

    /// Full forward+loss+backward+adam step on GPU.
    ///
    /// Args:
    ///     cam_pos: [3] f32 camera position
    ///     vp: [16] f32 column-major VP matrix
    ///     c2w_intrinsics: [16] f32 camera-to-world rotation (3×vec4 columns) + intrinsics (fx,fy,cx,cy)
    ///     gt_image: [H, W, 3] f32 ground truth image
    ///     lr_verts: learning rate for vertices
    ///     lr_dens: learning rate for densities
    ///     lr_grads: learning rate for color gradients
    ///     loss_type: 0 = L1, 1 = L2
    ///
    /// Returns:
    ///     scalar loss value (f32)
    fn train_step(
        &mut self,
        cam_pos: PyReadonlyArray1<f32>,
        vp: PyReadonlyArray1<f32>,
        c2w_intrinsics: PyReadonlyArray1<f32>,
        gt_image: PyReadonlyArray3<f32>,
        lr_verts: f32,
        lr_dens: f32,
        lr_grads: f32,
        loss_type: u32,
    ) -> PyResult<f32> {
        let cam_pos_slice = cam_pos.as_slice()?;
        let vp_slice = vp.as_slice()?;
        let c2w_int_slice = c2w_intrinsics.as_slice()?;
        let gt_data = gt_image.as_slice()?;

        let cam = Vec3::new(cam_pos_slice[0], cam_pos_slice[1], cam_pos_slice[2]);
        let vp_mat = mat4_from_flat(vp_slice);
        let (c2w, intrinsics) = c2w_from_flat(c2w_int_slice);

        let uniforms = make_uniforms(
            vp_mat,
            c2w,
            intrinsics,
            cam,
            self.width as f32,
            self.height as f32,
            self.tet_count,
            self.step,
            self.tile_size,
            self.min_t,
            0,
            0.01,
            1000.0,
        );

        // Upload uniforms + ground truth
        self.queue.write_buffer(
            &self.scene_buffers.uniforms,
            0,
            bytemuck::bytes_of(&uniforms),
        );
        self.queue.write_buffer(
            &self.loss_buffers.ground_truth,
            0,
            bytemuck::cast_slice(gt_data),
        );

        let loss_uni = LossUniforms {
            width: self.width,
            height: self.height,
            loss_type,
            lambda_ssim: 0.0,
        };
        self.queue.write_buffer(
            &self.loss_buffers.loss_uniforms,
            0,
            bytemuck::bytes_of(&loss_uni),
        );

        // Upload tile uniforms (visible_tet_count=0 placeholder — scan shaders read from GPU)
        let tile_uni = TileUniforms {
            screen_width: self.width,
            screen_height: self.height,
            tile_size: self.tile_size,
            tiles_x: self.tile_buffers.tiles_x,
            tiles_y: self.tile_buffers.tiles_y,
            num_tiles: self.tile_buffers.num_tiles,
            visible_tet_count: 0, // scan shaders read from GPU buffer
            _pad: [0; 5],
        };
        self.queue.write_buffer(
            &self.tile_buffers.tile_uniforms,
            0,
            bytemuck::bytes_of(&tile_uni),
        );

        // Single encoder: forward compute + scan tile pipeline + forward tiled + loss + backward + Adam
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("train_step_main"),
            });

        // Clear buffers
        encoder.clear_buffer(&self.grad_buffers.d_vertices, 0, None);
        encoder.clear_buffer(&self.grad_buffers.d_densities, 0, None);
        encoder.clear_buffer(&self.mat_grad_buffers.d_base_colors, 0, None);
        encoder.clear_buffer(&self.mat_grad_buffers.d_color_grads, 0, None);
        encoder.clear_buffer(&self.loss_buffers.loss_value, 0, None);
        encoder.clear_buffer(&self.tile_buffers.tile_ranges, 0, None);
        encoder.clear_buffer(&self.rasterize.rendered_image, 0, None);

        // Forward compute (SH eval + cull + tiles_touched + compact_tet_ids, no sort)
        record_project_compute(
            &mut encoder,
            &self.fwd_pipelines,
            &self.scene_buffers,
            &self.compute_bg,
            self.tet_count,
            &self.queue,
        );

        // RTS scan-based tile pipeline: prepare_dispatch → rts_reduce → rts_spine_scan
        // → rts_downsweep → tile_fill → tile_gen_scan (writes total_pairs to num_keys_buf)
        rmesh_backward::record_scan_tile_pipeline(
            &mut encoder,
            &self.scan_pipelines,
            &self.tile_pipelines,
            &self.prepare_dispatch_bg,
            &self.rts_bg,
            &self.tile_fill_bg,
            &self.tile_gen_scan_bg,
            &self.scan_buffers,
            &self.tile_buffers,
        );

        // Radix sort (ONCE — shared between forward and backward)
        let result_in_b = rmesh_backward::record_radix_sort(
            &mut encoder,
            &self.device,
            &self.radix_pipelines,
            &self.radix_state,
            &self.tile_buffers.tile_sort_keys,
            &self.tile_buffers.tile_sort_values,
        );

        let (ranges_bg, fwd_bg, bwd_bg0) = if result_in_b {
            (&self.tile_ranges_bg_b, &self.rasterize_bg_b, &self.bwd_tiled_bg0_tiled_b)
        } else {
            (&self.tile_ranges_bg, &self.rasterize_bg, &self.bwd_tiled_bg0_tiled)
        };

        // Tile ranges
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tile_ranges"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.tile_pipelines.tile_ranges_pipeline);
            pass.set_bind_group(0, ranges_bg, &[]);
            let total = (self.tile_buffers.max_pairs_pow2 + 255) / 256;
            if total <= 65535 {
                pass.dispatch_workgroups(total, 1, 1);
            } else {
                pass.dispatch_workgroups(65535, (total + 65534) / 65535, 1);
            }
        }

        // Forward tiled compute
        record_rasterize_compute(
            &mut encoder,
            &self.rasterize,
            fwd_bg,
            self.tile_buffers.num_tiles,
        );

        // Loss compute (reads from rasterize.rendered_image via loss_bg_tiled)
        rmesh_train::record_loss_pass(
            &mut encoder,
            &self.loss_pipeline,
            &self.loss_bg_tiled,
            self.width,
            self.height,
        );

        // Backward tiled compute (reuses same sorted pairs — no re-sort)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("backward_tiled"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bwd_tiled_pipelines.pipeline);
            pass.set_bind_group(0, bwd_bg0, &[]);
            pass.set_bind_group(1, &self.bwd_tiled_bg1, &[]);
            let num_tiles = self.tile_buffers.num_tiles;
            if num_tiles <= 65535 {
                pass.dispatch_workgroups(num_tiles, 1, 1);
            } else {
                let x = 65535u32;
                let y = (num_tiles + x - 1) / x;
                pass.dispatch_workgroups(x, y, 1);
            }
        }

        // Adam passes (all in same encoder, separate uniform buffers per group)
        self.step += 1; // Once per train_step, not per param group
        let learning_rates = [lr_verts, lr_dens, lr_grads];
        for (i, (bg, &count)) in self
            .adam_bgs
            .iter()
            .zip(self.param_counts.iter())
            .enumerate()
        {
            let adam_uni = AdamUniforms {
                param_count: count,
                step: self.step,
                lr: learning_rates[i],
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                _pad: [0; 2],
            };
            self.queue.write_buffer(
                &self.adam_uniforms_bufs[i],
                0,
                bytemuck::bytes_of(&adam_uni),
            );

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("adam"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.adam_pipeline.pipeline);
            pass.set_bind_group(0, bg, &[]);
            let wg_count = (count + 255) / 256;
            if wg_count <= 65535 {
                pass.dispatch_workgroups(wg_count, 1, 1);
            } else {
                pass.dispatch_workgroups(65535, (wg_count + 65534) / 65535, 1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back loss value
        let loss_data = read_buffer_raw(&self.device, &self.queue, &self.loss_buffers.loss_value);
        let loss_bits = u32::from_le_bytes(loss_data[0..4].try_into().unwrap());
        let loss = f32::from_bits(loss_bits);

        Ok(loss)
    }

    /// Re-upload trainable parameters from numpy arrays.
    ///
    /// Use this when PyTorch optimizer has updated the parameters.
    fn update_params(
        &mut self,
        vertices: PyReadonlyArray1<f32>,
        base_colors: PyReadonlyArray1<f32>,
        densities: PyReadonlyArray1<f32>,
        color_grads: PyReadonlyArray1<f32>,
    ) -> PyResult<()> {
        self.queue.write_buffer(
            &self.scene_buffers.vertices,
            0,
            bytemuck::cast_slice(vertices.as_slice()?),
        );
        self.queue.write_buffer(
            &self.material_buffers.base_colors,
            0,
            bytemuck::cast_slice(base_colors.as_slice()?),
        );
        self.queue.write_buffer(
            &self.scene_buffers.densities,
            0,
            bytemuck::cast_slice(densities.as_slice()?),
        );
        self.queue.write_buffer(
            &self.material_buffers.color_grads,
            0,
            bytemuck::cast_slice(color_grads.as_slice()?),
        );
        Ok(())
    }

    /// Read back current parameter values from GPU.
    ///
    /// Use this after wgpu-only training to get the optimized parameters.
    fn get_params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let verts = read_buffer_f32(&self.device, &self.queue, &self.scene_buffers.vertices);
        let base_cols = read_buffer_f32(&self.device, &self.queue, &self.material_buffers.base_colors);
        let dens = read_buffer_f32(&self.device, &self.queue, &self.scene_buffers.densities);
        let grads = read_buffer_f32(&self.device, &self.queue, &self.material_buffers.color_grads);

        let dict = PyDict::new(py);
        dict.set_item("vertices", Array1::from_vec(verts).into_pyarray(py))?;
        dict.set_item("base_colors", Array1::from_vec(base_cols).into_pyarray(py))?;
        dict.set_item("densities", Array1::from_vec(dens).into_pyarray(py))?;
        dict.set_item("color_grads", Array1::from_vec(grads).into_pyarray(py))?;
        Ok(dict)
    }

    /// Get the current training step counter.
    #[getter]
    fn step(&self) -> u32 {
        self.step
    }

    /// Get render width.
    #[getter]
    fn width(&self) -> u32 {
        self.width
    }

    /// Get render height.
    #[getter]
    fn height(&self) -> u32 {
        self.height
    }
}

/// Native module — re-exported as rmesh_wgpu._native by the Python package.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RMeshRenderer>()?;
    Ok(())
}
