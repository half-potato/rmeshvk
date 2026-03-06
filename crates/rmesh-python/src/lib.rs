//! PyO3 bindings for the wgpu-based tetrahedral volume renderer.
//!
//! Exposes `RMeshRenderer` to Python with:
//!   - `forward()`: run the forward render pipeline, return image as numpy array
//!   - `backward()`: upload dL/d(image), run backward pass, return gradients
//!   - `train_step()`: full forward+loss+backward+adam on GPU, return loss
//!   - `update_params()` / `get_params()`: parameter transfer for PyTorch optimizer path

use numpy::ndarray::{Array1, Array3};
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray1, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use glam::{Mat4, Vec3, Vec4};
use rmesh_backward::{
    create_adam_bind_group, create_backward_bind_groups, create_backward_tiled_bind_groups,
    create_loss_bind_group, create_prepare_dispatch_bind_group, create_prefix_scan_add_bind_group,
    create_prefix_scan_bind_group, create_block_scan_bind_group,
    create_tile_fill_bind_group, create_tile_gen_bind_group, create_tile_gen_scan_bind_group,
    create_tile_ranges_bind_group_with_keys, AdamState, AdamUniforms, BackwardPipelines,
    GradientBuffers, LossBuffers, LossUniforms, RadixSortPipelines, RadixSortState,
    ScanBuffers, ScanPipelines, TileBuffers, TilePipelines, TileUniforms,
};
use rmesh_render::{
    create_compute_bind_group, create_forward_tiled_bind_group, create_render_bind_group,
    make_uniforms, record_forward_compute, record_forward_pass, record_forward_tiled,
    record_tex_to_buffer, ForwardPipelines, ForwardTiledPipeline, RenderTargets, SceneBuffers,
    SortState, TexToBufferPipeline,
};

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

#[pyclass]
struct RMeshRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    scene_buffers: SceneBuffers,
    fwd_pipelines: ForwardPipelines,
    targets: RenderTargets,
    sort_state: SortState,
    ttb: TexToBufferPipeline,
    // Backward infrastructure
    bwd_pipelines: BackwardPipelines,
    grad_buffers: GradientBuffers,
    loss_buffers: LossBuffers,
    _adam_state: AdamState,
    // Tiled backward infrastructure
    tile_pipelines: TilePipelines,
    tile_buffers: TileBuffers,
    radix_pipelines: RadixSortPipelines,
    radix_state: RadixSortState,
    tile_fill_bg: wgpu::BindGroup,
    _tile_gen_bg: wgpu::BindGroup,
    // A-buffer bind groups (sort result in primary buffers)
    tile_ranges_bg: wgpu::BindGroup,
    bwd_tiled_bg0: wgpu::BindGroup,
    bwd_tiled_bg1: wgpu::BindGroup,
    // B-buffer bind groups (sort result in alternate buffers)
    tile_ranges_bg_b: wgpu::BindGroup,
    bwd_tiled_bg0_b: wgpu::BindGroup,
    // Forward tiled compute pipeline (subgroup-based, 4x4 tiles)
    fwd_tiled: ForwardTiledPipeline,
    fwd_tiled_bg: wgpu::BindGroup,
    fwd_tiled_bg_b: wgpu::BindGroup,
    // Debug image buffer for backward forward-replay verification
    debug_image: wgpu::Buffer,
    // Bind groups
    compute_bg: wgpu::BindGroup,
    render_bg: wgpu::BindGroup,
    _loss_bg: wgpu::BindGroup,
    // Loss bind group using fwd_tiled.rendered_image (for train_step)
    loss_bg_tiled: wgpu::BindGroup,
    // Backward tiled bind groups using fwd_tiled.rendered_image (A and B variants)
    bwd_tiled_bg0_tiled: wgpu::BindGroup,
    bwd_tiled_bg0_tiled_b: wgpu::BindGroup,
    _bwd_bg0: wgpu::BindGroup,
    _bwd_bg1: wgpu::BindGroup,
    adam_bgs: Vec<wgpu::BindGroup>,
    adam_uniforms_bufs: Vec<wgpu::Buffer>,
    // Scan-based tile pipeline (Changes 2-3)
    scan_pipelines: ScanPipelines,
    scan_buffers: ScanBuffers,
    prepare_dispatch_bg: wgpu::BindGroup,
    prefix_scan_bg: wgpu::BindGroup,
    block_scan_bg: wgpu::BindGroup,
    prefix_scan_add_bg: wgpu::BindGroup,
    tile_gen_scan_bg: wgpu::BindGroup,
    // Scene metadata
    tet_count: u32,
    _vertex_count: u32,
    sh_degree: u32,
    _sh_stride: u32,
    width: u32,
    height: u32,
    step: u32,
    param_counts: [u32; 4],
}

#[pymethods]
impl RMeshRenderer {
    /// Create a new renderer.
    ///
    /// Args:
    ///     vertices: [N*3] f32 flat array of vertex positions
    ///     indices: [M*4] u32 flat array of tet indices
    ///     sh_coeffs: [M*nc*3] f32 flat SH coefficients (nc = (deg+1)^2)
    ///     densities: [M] f32 per-tet densities
    ///     color_grads: [M*3] f32 per-tet color gradients
    ///     circumdata: [M*4] f32 circumsphere data (cx, cy, cz, r^2)
    ///     sh_degree: SH degree (0-3)
    ///     width: render width
    ///     height: render height
    #[new]
    fn new(
        vertices: PyReadonlyArray1<f32>,
        indices: PyReadonlyArray1<u32>,
        sh_coeffs: PyReadonlyArray1<f32>,
        densities: PyReadonlyArray1<f32>,
        color_grads: PyReadonlyArray1<f32>,
        circumdata: PyReadonlyArray1<f32>,
        sh_degree: u32,
        width: u32,
        height: u32,
    ) -> PyResult<Self> {
        let vertices_slice = vertices.as_slice()?;
        let indices_slice = indices.as_slice()?;
        let sh_coeffs_slice = sh_coeffs.as_slice()?;
        let densities_slice = densities.as_slice()?;
        let color_grads_slice = color_grads.as_slice()?;
        let circumdata_slice = circumdata.as_slice()?;

        let tet_count = indices_slice.len() as u32 / 4;
        let vertex_count = vertices_slice.len() as u32 / 3;
        let num_coeffs = (sh_degree + 1) * (sh_degree + 1);
        let sh_stride = num_coeffs * 3;

        // Build SceneData
        let scene = rmesh_data::SceneData {
            vertices: vertices_slice.to_vec(),
            indices: indices_slice.to_vec(),
            sh_coeffs: sh_coeffs_slice.to_vec(),
            densities: densities_slice.to_vec(),
            color_grads: color_grads_slice.to_vec(),
            circumdata: circumdata_slice.to_vec(),
            start_pose: [0.0; 7],
            vertex_count,
            tet_count,
            sh_degree,
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
                required_features: wgpu::Features::SUBGROUP,
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
        let aux_format = wgpu::TextureFormat::Rgba32Float;

        // Create pipelines
        let fwd_pipelines = ForwardPipelines::new(&device, color_format, aux_format);
        let bwd_pipelines = BackwardPipelines::new(&device);

        // Upload scene
        let scene_buffers = SceneBuffers::upload(&device, &queue, &scene);

        // Render targets
        let targets = RenderTargets::new(&device, width, height);

        // Sort state
        let sort_state = SortState::new(
            &device,
            &fwd_pipelines,
            &scene_buffers.sort_keys,
            &scene_buffers.sort_values,
            tet_count,
        );

        // Forward bind groups
        let compute_bg = create_compute_bind_group(&device, &fwd_pipelines, &scene_buffers);
        let render_bg = create_render_bind_group(&device, &fwd_pipelines, &scene_buffers);

        // Tex-to-buffer
        let ttb = TexToBufferPipeline::new(&device, &queue, &targets.color_view, width, height);

        // Gradient + optimizer state
        let grad_buffers =
            GradientBuffers::new(&device, vertex_count, tet_count, sh_stride);
        let adam_state = AdamState::new(&device, vertex_count, tet_count, sh_stride);

        // Loss buffers
        let loss_buffers = LossBuffers::new(&device, width, height);

        // Backward bind groups
        let loss_bg = create_loss_bind_group(
            &device,
            &bwd_pipelines,
            &loss_buffers,
            &ttb.rendered_image,
        );
        let (bwd_bg0, bwd_bg1) = create_backward_bind_groups(
            &device,
            &bwd_pipelines,
            &scene_buffers,
            &loss_buffers,
            &grad_buffers,
            &ttb.rendered_image,
        );

        // Adam bind groups + per-group uniform buffers (separate to avoid overwrite)
        let adam_uniforms_bufs: Vec<wgpu::Buffer> = (0..4)
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
                &bwd_pipelines,
                &adam_uniforms_bufs[0],
                &scene_buffers.sh_coeffs,
                &grad_buffers.d_sh_coeffs,
                &adam_state.m_sh,
                &adam_state.v_sh,
            ),
            create_adam_bind_group(
                &device,
                &bwd_pipelines,
                &adam_uniforms_bufs[1],
                &scene_buffers.vertices,
                &grad_buffers.d_vertices,
                &adam_state.m_vertices,
                &adam_state.v_vertices,
            ),
            create_adam_bind_group(
                &device,
                &bwd_pipelines,
                &adam_uniforms_bufs[2],
                &scene_buffers.densities,
                &grad_buffers.d_densities,
                &adam_state.m_densities,
                &adam_state.v_densities,
            ),
            create_adam_bind_group(
                &device,
                &bwd_pipelines,
                &adam_uniforms_bufs[3],
                &scene_buffers.color_grads,
                &grad_buffers.d_color_grads,
                &adam_state.m_color_grads,
                &adam_state.v_color_grads,
            ),
        ];

        let param_counts = [
            tet_count * sh_stride,
            vertex_count * 3,
            tet_count,
            tet_count * 3,
        ];

        // Tiled backward infrastructure
        let tile_size = 4u32;
        let tile_pipelines = TilePipelines::new(&device);
        let tile_buffers = TileBuffers::new(&device, tet_count, width, height, tile_size);

        // Radix sort (replaces bitonic TileSortState)
        let sorting_bits = 32u32; // full 32-bit key sort
        let radix_pipelines = RadixSortPipelines::new(&device);
        let radix_state = RadixSortState::new(&device, tile_buffers.max_pairs_pow2, sorting_bits);
        radix_state.upload_configs(&queue);

        let tile_fill_bg = create_tile_fill_bind_group(&device, &tile_pipelines, &tile_buffers);
        let tile_gen_bg = create_tile_gen_bind_group(
            &device,
            &tile_pipelines,
            &tile_buffers,
            &scene_buffers,
        );

        // Debug image buffer for backward forward-replay verification
        let n_pixels = (width as u64) * (height as u64);
        let debug_image = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("debug_image"),
            size: n_pixels * 4 * 4, // RGBA f32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Forward tiled compute pipeline (4x4 tiles with subgroups)
        let fwd_tiled = ForwardTiledPipeline::new(&device, width, height);

        // A-buffer bind groups (sort result in primary tile_sort_keys/values)
        // Use num_keys_buf as tile_pair_count since scan pipeline writes total_pairs there
        let tile_ranges_bg = create_tile_ranges_bind_group_with_keys(
            &device,
            &tile_pipelines,
            &tile_buffers.tile_sort_keys,
            &tile_buffers.tile_ranges,
            &tile_buffers.tile_uniforms,
            &radix_state.num_keys_buf,
        );
        let (bwd_tiled_bg0, bwd_tiled_bg1) = create_backward_tiled_bind_groups(
            &device,
            &tile_pipelines,
            &scene_buffers,
            &loss_buffers,
            &grad_buffers,
            &ttb.rendered_image,
            &tile_buffers.tile_sort_values,
            &tile_buffers.tile_ranges,
            &tile_buffers.tile_uniforms,
            &debug_image,
        );

        // B-buffer bind groups (sort result in radix_state.keys_b/values_b)
        let tile_ranges_bg_b = create_tile_ranges_bind_group_with_keys(
            &device,
            &tile_pipelines,
            &radix_state.keys_b,
            &tile_buffers.tile_ranges,
            &tile_buffers.tile_uniforms,
            &radix_state.num_keys_buf,
        );
        let (bwd_tiled_bg0_b, _) = create_backward_tiled_bind_groups(
            &device,
            &tile_pipelines,
            &scene_buffers,
            &loss_buffers,
            &grad_buffers,
            &ttb.rendered_image,
            &radix_state.values_b,
            &tile_buffers.tile_ranges,
            &tile_buffers.tile_uniforms,
            &debug_image,
        );

        // Forward tiled bind groups (A and B buffer variants)
        let fwd_tiled_bg = create_forward_tiled_bind_group(
            &device,
            &fwd_tiled,
            &scene_buffers,
            &tile_buffers.tile_sort_values,
            &tile_buffers.tile_ranges,
            &tile_buffers.tile_uniforms,
        );
        let fwd_tiled_bg_b = create_forward_tiled_bind_group(
            &device,
            &fwd_tiled,
            &scene_buffers,
            &radix_state.values_b,
            &tile_buffers.tile_ranges,
            &tile_buffers.tile_uniforms,
        );

        // Loss bind group using fwd_tiled.rendered_image (for train_step tiled path)
        let loss_bg_tiled = create_loss_bind_group(
            &device,
            &bwd_pipelines,
            &loss_buffers,
            &fwd_tiled.rendered_image,
        );

        // Backward tiled bind groups using fwd_tiled.rendered_image (for train_step)
        let (bwd_tiled_bg0_tiled, _) = create_backward_tiled_bind_groups(
            &device,
            &tile_pipelines,
            &scene_buffers,
            &loss_buffers,
            &grad_buffers,
            &fwd_tiled.rendered_image,
            &tile_buffers.tile_sort_values,
            &tile_buffers.tile_ranges,
            &tile_buffers.tile_uniforms,
            &debug_image,
        );
        let (bwd_tiled_bg0_tiled_b, _) = create_backward_tiled_bind_groups(
            &device,
            &tile_pipelines,
            &scene_buffers,
            &loss_buffers,
            &grad_buffers,
            &fwd_tiled.rendered_image,
            &radix_state.values_b,
            &tile_buffers.tile_ranges,
            &tile_buffers.tile_uniforms,
            &debug_image,
        );

        // Scan-based tile pipeline infrastructure
        let scan_pipelines = ScanPipelines::new(&device);
        let scan_buffers = ScanBuffers::new(&device, tet_count);
        let prepare_dispatch_bg = create_prepare_dispatch_bind_group(
            &device,
            &scan_pipelines,
            &scene_buffers.indirect_args,
            &scan_buffers,
        );
        let prefix_scan_bg = create_prefix_scan_bind_group(
            &device,
            &scan_pipelines,
            &scene_buffers.tiles_touched,
            &scan_buffers,
        );
        let block_scan_bg = create_block_scan_bind_group(
            &device,
            &scan_pipelines,
            &scan_buffers,
        );
        let prefix_scan_add_bg = create_prefix_scan_add_bind_group(
            &device,
            &scan_pipelines,
            &scan_buffers,
            &radix_state.num_keys_buf,
        );
        let tile_gen_scan_bg = create_tile_gen_scan_bind_group(
            &device,
            &scan_pipelines,
            &tile_buffers,
            &scene_buffers,
            &scan_buffers,
        );

        Ok(Self {
            device,
            queue,
            scene_buffers,
            fwd_pipelines,
            targets,
            sort_state,
            ttb,
            bwd_pipelines,
            grad_buffers,
            loss_buffers,
            _adam_state: adam_state,
            tile_pipelines,
            tile_buffers,
            radix_pipelines,
            radix_state,
            tile_fill_bg,
            _tile_gen_bg: tile_gen_bg,
            tile_ranges_bg,
            bwd_tiled_bg0,
            bwd_tiled_bg1,
            tile_ranges_bg_b,
            bwd_tiled_bg0_b,
            fwd_tiled,
            fwd_tiled_bg,
            fwd_tiled_bg_b,
            debug_image,
            compute_bg,
            render_bg,
            _loss_bg: loss_bg,
            loss_bg_tiled,
            bwd_tiled_bg0_tiled,
            bwd_tiled_bg0_tiled_b,
            _bwd_bg0: bwd_bg0,
            _bwd_bg1: bwd_bg1,
            adam_bgs,
            adam_uniforms_bufs,
            scan_pipelines,
            scan_buffers,
            prepare_dispatch_bg,
            prefix_scan_bg,
            block_scan_bg,
            prefix_scan_add_bg,
            tile_gen_scan_bg,
            tet_count,
            _vertex_count: vertex_count,
            sh_degree,
            _sh_stride: sh_stride,
            width,
            height,
            step: 0,
            param_counts,
        })
    }

    /// Run the forward rendering pipeline.
    ///
    /// Args:
    ///     cam_pos: [3] f32 camera position
    ///     vp: [16] f32 column-major view-projection matrix
    ///     inv_vp: [16] f32 column-major inverse view-projection matrix
    ///
    /// Returns:
    ///     numpy array [H, W, 4] f32 (premultiplied RGBA)
    fn forward<'py>(
        &mut self,
        py: Python<'py>,
        cam_pos: PyReadonlyArray1<f32>,
        vp: PyReadonlyArray1<f32>,
        inv_vp: PyReadonlyArray1<f32>,
    ) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let cam_pos_slice = cam_pos.as_slice()?;
        let vp_slice = vp.as_slice()?;
        let inv_vp_slice = inv_vp.as_slice()?;

        let cam = Vec3::new(cam_pos_slice[0], cam_pos_slice[1], cam_pos_slice[2]);
        let vp_mat = mat4_from_flat(vp_slice);
        let inv_vp_mat = mat4_from_flat(inv_vp_slice);

        let uniforms = make_uniforms(
            vp_mat,
            inv_vp_mat,
            cam,
            self.width as f32,
            self.height as f32,
            self.tet_count,
            self.sh_degree,
            self.step,
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
        record_forward_pass(
            &mut encoder,
            &self.fwd_pipelines,
            &self.scene_buffers,
            &self.targets,
            &self.compute_bg,
            &self.render_bg,
            &self.sort_state,
            self.tet_count,
            &self.queue,
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
    ///     inv_vp: [16] f32 column-major inverse view-projection matrix
    ///
    /// Returns:
    ///     numpy array [H, W, 4] f32 (RGBA)
    fn forward_tiled<'py>(
        &mut self,
        py: Python<'py>,
        cam_pos: PyReadonlyArray1<f32>,
        vp: PyReadonlyArray1<f32>,
        inv_vp: PyReadonlyArray1<f32>,
    ) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let cam_pos_slice = cam_pos.as_slice()?;
        let vp_slice = vp.as_slice()?;
        let inv_vp_slice = inv_vp.as_slice()?;

        let cam = Vec3::new(cam_pos_slice[0], cam_pos_slice[1], cam_pos_slice[2]);
        let vp_mat = mat4_from_flat(vp_slice);
        let inv_vp_mat = mat4_from_flat(inv_vp_slice);

        let uniforms = make_uniforms(
            vp_mat,
            inv_vp_mat,
            cam,
            self.width as f32,
            self.height as f32,
            self.tet_count,
            self.sh_degree,
            self.step,
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
            tile_size: 4,
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
                    label: Some("forward_tiled"),
                });

            // Clear tile buffers + forward output
            encoder.clear_buffer(&self.tile_buffers.tile_ranges, 0, None);
            encoder.clear_buffer(&self.fwd_tiled.rendered_image, 0, None);

            // Forward compute (SH eval + cull + tiles_touched + compact_tet_ids, no sort)
            record_forward_compute(
                &mut encoder,
                &self.fwd_pipelines,
                &self.scene_buffers,
                &self.compute_bg,
                self.tet_count,
                &self.queue,
            );

            // Scan-based tile pipeline
            rmesh_backward::record_scan_tile_pipeline(
                &mut encoder,
                &self.scan_pipelines,
                &self.tile_pipelines,
                &self.prepare_dispatch_bg,
                &self.prefix_scan_bg,
                &self.block_scan_bg,
                &self.prefix_scan_add_bg,
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

            let (ranges_bg, fwd_bg) = if result_in_b {
                (&self.tile_ranges_bg_b, &self.fwd_tiled_bg_b)
            } else {
                (&self.tile_ranges_bg, &self.fwd_tiled_bg)
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
            record_forward_tiled(
                &mut encoder,
                &self.fwd_tiled,
                fwd_bg,
                self.tile_buffers.num_tiles,
            );

            self.queue.submit(std::iter::once(encoder.finish()));
        }

        // Read back rendered image
        let data = read_buffer_f32(&self.device, &self.queue, &self.fwd_tiled.rendered_image);

        let array = Array3::from_shape_vec(
            (self.height as usize, self.width as usize, 4),
            data,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}")))?;

        Ok(array.into_pyarray(py))
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
    ///     dict with keys 'd_sh_coeffs', 'd_vertices', 'd_densities', 'd_color_grads',
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
            tile_size: 4,
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
        encoder.clear_buffer(&self.grad_buffers.d_sh_coeffs, 0, None);
        encoder.clear_buffer(&self.grad_buffers.d_vertices, 0, None);
        encoder.clear_buffer(&self.grad_buffers.d_densities, 0, None);
        encoder.clear_buffer(&self.grad_buffers.d_color_grads, 0, None);
        encoder.clear_buffer(&self.tile_buffers.tile_ranges, 0, None);
        encoder.clear_buffer(&self.debug_image, 0, None);

        // Scan-based tile pipeline (reads compact_tet_ids + tiles_touched from forward compute)
        rmesh_backward::record_scan_tile_pipeline(
            &mut encoder,
            &self.scan_pipelines,
            &self.tile_pipelines,
            &self.prepare_dispatch_bg,
            &self.prefix_scan_bg,
            &self.block_scan_bg,
            &self.prefix_scan_add_bg,
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

        let (ranges_bg, bwd_bg0) = if result_in_b {
            (&self.tile_ranges_bg_b, &self.bwd_tiled_bg0_b)
        } else {
            (&self.tile_ranges_bg, &self.bwd_tiled_bg0)
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
            pass.set_pipeline(&self.tile_pipelines.backward_tiled_pipeline);
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
        let d_sh = read_buffer_f32(&self.device, &self.queue, &self.grad_buffers.d_sh_coeffs);
        let d_verts = read_buffer_f32(&self.device, &self.queue, &self.grad_buffers.d_vertices);
        let d_dens = read_buffer_f32(&self.device, &self.queue, &self.grad_buffers.d_densities);
        let d_grads =
            read_buffer_f32(&self.device, &self.queue, &self.grad_buffers.d_color_grads);

        let dict = PyDict::new(py);
        dict.set_item(
            "d_sh_coeffs",
            Array1::from_vec(d_sh).into_pyarray(py),
        )?;
        dict.set_item(
            "d_vertices",
            Array1::from_vec(d_verts).into_pyarray(py),
        )?;
        dict.set_item(
            "d_densities",
            Array1::from_vec(d_dens).into_pyarray(py),
        )?;
        dict.set_item(
            "d_color_grads",
            Array1::from_vec(d_grads).into_pyarray(py),
        )?;
        Ok(dict)
    }

    /// Read the debug image written by the backward pass's forward replay.
    ///
    /// Returns:
    ///     numpy array [H, W, 4] f32 (composited RGBA from backward's forward replay)
    fn read_debug_image<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let data = read_buffer_f32(&self.device, &self.queue, &self.debug_image);
        let arr = Array3::from_shape_vec(
            (self.height as usize, self.width as usize, 4),
            data,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

    /// Full forward+loss+backward+adam step on GPU.
    ///
    /// Args:
    ///     cam_pos: [3] f32 camera position
    ///     vp: [16] f32 column-major VP matrix
    ///     inv_vp: [16] f32 column-major inverse VP matrix
    ///     gt_image: [H, W, 3] f32 ground truth image
    ///     lr_sh: learning rate for SH coefficients
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
        inv_vp: PyReadonlyArray1<f32>,
        gt_image: PyReadonlyArray3<f32>,
        lr_sh: f32,
        lr_verts: f32,
        lr_dens: f32,
        lr_grads: f32,
        loss_type: u32,
    ) -> PyResult<f32> {
        let cam_pos_slice = cam_pos.as_slice()?;
        let vp_slice = vp.as_slice()?;
        let inv_vp_slice = inv_vp.as_slice()?;
        let gt_data = gt_image.as_slice()?;

        let cam = Vec3::new(cam_pos_slice[0], cam_pos_slice[1], cam_pos_slice[2]);
        let vp_mat = mat4_from_flat(vp_slice);
        let inv_vp_mat = mat4_from_flat(inv_vp_slice);

        let uniforms = make_uniforms(
            vp_mat,
            inv_vp_mat,
            cam,
            self.width as f32,
            self.height as f32,
            self.tet_count,
            self.sh_degree,
            self.step,
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
            tile_size: 4,
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
        encoder.clear_buffer(&self.grad_buffers.d_sh_coeffs, 0, None);
        encoder.clear_buffer(&self.grad_buffers.d_vertices, 0, None);
        encoder.clear_buffer(&self.grad_buffers.d_densities, 0, None);
        encoder.clear_buffer(&self.grad_buffers.d_color_grads, 0, None);
        encoder.clear_buffer(&self.loss_buffers.loss_value, 0, None);
        encoder.clear_buffer(&self.tile_buffers.tile_ranges, 0, None);
        encoder.clear_buffer(&self.fwd_tiled.rendered_image, 0, None);
        encoder.clear_buffer(&self.debug_image, 0, None);

        // Forward compute (SH eval + cull + tiles_touched + compact_tet_ids, no sort)
        record_forward_compute(
            &mut encoder,
            &self.fwd_pipelines,
            &self.scene_buffers,
            &self.compute_bg,
            self.tet_count,
            &self.queue,
        );

        // Scan-based tile pipeline: prepare_dispatch → prefix_scan → block_scan
        // → prefix_scan_add (writes total_pairs to num_keys_buf) → tile_fill → tile_gen_scan
        rmesh_backward::record_scan_tile_pipeline(
            &mut encoder,
            &self.scan_pipelines,
            &self.tile_pipelines,
            &self.prepare_dispatch_bg,
            &self.prefix_scan_bg,
            &self.block_scan_bg,
            &self.prefix_scan_add_bg,
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
            (&self.tile_ranges_bg_b, &self.fwd_tiled_bg_b, &self.bwd_tiled_bg0_tiled_b)
        } else {
            (&self.tile_ranges_bg, &self.fwd_tiled_bg, &self.bwd_tiled_bg0_tiled)
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
        record_forward_tiled(
            &mut encoder,
            &self.fwd_tiled,
            fwd_bg,
            self.tile_buffers.num_tiles,
        );

        // Loss compute (reads from fwd_tiled.rendered_image via loss_bg_tiled)
        rmesh_backward::record_loss_pass(
            &mut encoder,
            &self.bwd_pipelines,
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
            pass.set_pipeline(&self.tile_pipelines.backward_tiled_pipeline);
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
        let learning_rates = [lr_sh, lr_verts, lr_dens, lr_grads];
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
            pass.set_pipeline(&self.bwd_pipelines.adam_pipeline);
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
        sh_coeffs: PyReadonlyArray1<f32>,
        densities: PyReadonlyArray1<f32>,
        color_grads: PyReadonlyArray1<f32>,
    ) -> PyResult<()> {
        self.queue.write_buffer(
            &self.scene_buffers.vertices,
            0,
            bytemuck::cast_slice(vertices.as_slice()?),
        );
        self.queue.write_buffer(
            &self.scene_buffers.sh_coeffs,
            0,
            bytemuck::cast_slice(sh_coeffs.as_slice()?),
        );
        self.queue.write_buffer(
            &self.scene_buffers.densities,
            0,
            bytemuck::cast_slice(densities.as_slice()?),
        );
        self.queue.write_buffer(
            &self.scene_buffers.color_grads,
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
        let sh = read_buffer_f32(&self.device, &self.queue, &self.scene_buffers.sh_coeffs);
        let dens = read_buffer_f32(&self.device, &self.queue, &self.scene_buffers.densities);
        let grads = read_buffer_f32(&self.device, &self.queue, &self.scene_buffers.color_grads);

        let dict = PyDict::new(py);
        dict.set_item("vertices", Array1::from_vec(verts).into_pyarray(py))?;
        dict.set_item("sh_coeffs", Array1::from_vec(sh).into_pyarray(py))?;
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
