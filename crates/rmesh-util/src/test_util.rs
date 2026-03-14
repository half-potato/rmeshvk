//! Shared test utilities for GPU kernel tests.
//!
//! Gated behind the `test-util` feature. Provides:
//!   - GPU device creation
//!   - Buffer readback helpers
//!   - Random scene generators
//!   - Circumsphere computation

use glam::Vec3;
use rand::Rng;
use rmesh_data::SceneData;

// ---------------------------------------------------------------------------
// GPU device helpers
// ---------------------------------------------------------------------------

/// Configuration for test device creation.
pub struct TestDeviceConfig {
    /// Which backends to use. `None` means wgpu default.
    pub backends: Option<wgpu::Backends>,
    /// Extra required features (SUBGROUP is always requested).
    pub extra_features: wgpu::Features,
    /// Base limits to start from (`Limits::default()` or `Limits::downlevel_defaults()`).
    pub base_limits: wgpu::Limits,
}

impl Default for TestDeviceConfig {
    fn default() -> Self {
        Self {
            backends: Some(wgpu::Backends::VULKAN | wgpu::Backends::METAL),
            extra_features: wgpu::Features::empty(),
            base_limits: wgpu::Limits::default(),
        }
    }
}

/// Create a GPU device with SUBGROUP feature. Returns None if no adapter is found.
pub fn create_test_device(config: TestDeviceConfig) -> Option<(wgpu::Device, wgpu::Queue)> {
    pollster::block_on(async {
        let instance_desc = match config.backends {
            Some(backends) => wgpu::InstanceDescriptor {
                backends,
                ..Default::default()
            },
            None => wgpu::InstanceDescriptor::default(),
        };
        let instance = wgpu::Instance::new(&instance_desc);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok()?;

        adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::SUBGROUP | config.extra_features,
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 16,
                    max_storage_buffer_binding_size: 1 << 30,
                    max_buffer_size: 1 << 30,
                    ..config.base_limits
                },
                ..Default::default()
            })
            .await
            .ok()
    })
}

/// Create a test device with default config (Vulkan|Metal, default limits).
pub fn create_test_device_default() -> Option<(wgpu::Device, wgpu::Queue)> {
    create_test_device(TestDeviceConfig::default())
}

// ---------------------------------------------------------------------------
// Buffer helpers
// ---------------------------------------------------------------------------

/// Read back a GPU buffer as a `Vec<T>`. Source buffer must have `COPY_SRC`.
pub fn read_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    count: usize,
) -> Vec<T> {
    let size = (count * std::mem::size_of::<T>()) as u64;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(source, 0, &readback, 0, size);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    receiver.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    readback.unmap();
    result
}

/// Create a read-write storage buffer with COPY_DST and COPY_SRC.
pub fn create_rw_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

// ---------------------------------------------------------------------------
// Scene helpers
// ---------------------------------------------------------------------------

/// Compute circumspheres from flat vertex/index arrays.
pub fn compute_circumspheres(vertices: &[f32], indices: &[u32]) -> Vec<f32> {
    let tet_count = indices.len() / 4;
    let mut circumdata = vec![0.0f32; tet_count * 4];
    for i in 0..tet_count {
        let i0 = indices[i * 4] as usize;
        let i1 = indices[i * 4 + 1] as usize;
        let i2 = indices[i * 4 + 2] as usize;
        let i3 = indices[i * 4 + 3] as usize;
        let v0 = Vec3::new(vertices[i0 * 3], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]);
        let v1 = Vec3::new(vertices[i1 * 3], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]);
        let v2 = Vec3::new(vertices[i2 * 3], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]);
        let v3 = Vec3::new(vertices[i3 * 3], vertices[i3 * 3 + 1], vertices[i3 * 3 + 2]);
        let a = v1 - v0;
        let b = v2 - v0;
        let c = v3 - v0;
        let (aa, bb, cc) = (a.dot(a), b.dot(b), c.dot(c));
        let cross_bc = b.cross(c);
        let cross_ca = c.cross(a);
        let cross_ab = a.cross(b);
        let mut denom = 2.0 * a.dot(cross_bc);
        if denom.abs() < 1e-12 {
            denom = 1.0;
        }
        let r = (aa * cross_bc + bb * cross_ca + cc * cross_ab) / denom;
        let center = v0 + r;
        circumdata[i * 4] = center.x;
        circumdata[i * 4 + 1] = center.y;
        circumdata[i * 4 + 2] = center.z;
        circumdata[i * 4 + 3] = r.dot(r);
    }
    circumdata
}

/// Generate a random tetrahedron centered roughly at origin.
/// Ensures positive orientation (det > 0) so face winding produces inward normals.
pub fn random_tet_vertices<R: Rng>(rng: &mut R, radius: f32) -> ([f32; 12], [u32; 4]) {
    let mut verts = [0.0f32; 12];
    for i in 0..4 {
        verts[i * 3] = (rng.random::<f32>() - 0.5) * 2.0 * radius;
        verts[i * 3 + 1] = (rng.random::<f32>() - 0.5) * 2.0 * radius;
        verts[i * 3 + 2] = (rng.random::<f32>() - 0.5) * 2.0 * radius;
    }
    // Ensure positive orientation: det(v1-v0, v2-v0, v3-v0) > 0
    let v0 = Vec3::new(verts[0], verts[1], verts[2]);
    let v1 = Vec3::new(verts[3], verts[4], verts[5]);
    let v2 = Vec3::new(verts[6], verts[7], verts[8]);
    let v3 = Vec3::new(verts[9], verts[10], verts[11]);
    let det = (v1 - v0).dot((v2 - v0).cross(v3 - v0));
    if det < 0.0 {
        // Swap vertices 2 and 3 to flip orientation
        return (verts, [0, 1, 3, 2]);
    }
    (verts, [0, 1, 2, 3])
}

/// Build a `SceneData` from raw vertices, indices, and per-tet parameters.
pub fn build_test_scene(
    vertices: Vec<f32>,
    indices: Vec<u32>,
    densities: Vec<f32>,
    color_grads: Vec<f32>,
) -> SceneData {
    let vertex_count = vertices.len() as u32 / 3;
    let tet_count = indices.len() as u32 / 4;
    let circumdata = compute_circumspheres(&vertices, &indices);

    SceneData {
        vertices,
        indices,
        densities,
        color_grads,
        circumdata,
        start_pose: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        vertex_count,
        tet_count,
    }
}

/// Build a single-tet test scene with random parameters.
pub fn random_single_tet_scene<R: Rng>(rng: &mut R, radius: f32) -> SceneData {
    let (verts, indices) = random_tet_vertices(rng, radius);
    let density = vec![rng.random::<f32>() * 5.0 + 0.5];
    let color_grads = vec![
        (rng.random::<f32>() - 0.5) * 0.2,
        (rng.random::<f32>() - 0.5) * 0.2,
        (rng.random::<f32>() - 0.5) * 0.2,
    ];

    build_test_scene(verts.to_vec(), indices.to_vec(), density, color_grads)
}

/// Generate a large grid-based tet scene for benchmarking.
///
/// Places `grid_size³` vertices on a regular grid in `[0, 1]³` with small
/// random jitter. Each cube cell is decomposed into 5 tetrahedra (all with
/// positive orientation).
///
/// `grid_size=126` → 2,000,376 vertices, 9,765,625 tets.
pub fn grid_tet_scene(grid_size: u32) -> SceneData {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(0xBE4C4);

    let n = grid_size;
    let vertex_count = n * n * n;
    let cell_count = (n - 1) * (n - 1) * (n - 1);
    let tet_count = cell_count * 5;

    // Vertex positions: regular grid with small jitter
    let step = 1.0 / (n - 1) as f32;
    let jitter = step * 0.1;
    let mut vertices = Vec::with_capacity(vertex_count as usize * 3);
    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                let x = ix as f32 * step + (rng.random::<f32>() - 0.5) * jitter;
                let y = iy as f32 * step + (rng.random::<f32>() - 0.5) * jitter;
                let z = iz as f32 * step + (rng.random::<f32>() - 0.5) * jitter;
                vertices.push(x);
                vertices.push(y);
                vertices.push(z);
            }
        }
    }

    // Index helper: vertex index for grid position (ix, iy, iz)
    let idx = |ix: u32, iy: u32, iz: u32| -> u32 { ix + iy * n + iz * n * n };

    // 5-tet decomposition of each cube cell.
    //
    // Cube vertices (named by binary xyz):
    //   A=v000  B=v100  C=v010  D=v110
    //   E=v001  F=v101  G=v011  H=v111
    //
    // Tets (all positive orientation):
    //   1: A B D F    2: A D C G    3: A E F G
    //   4: D F G H    5: A F D G   (central)
    let mut indices = Vec::with_capacity(tet_count as usize * 4);
    for iz in 0..(n - 1) {
        for iy in 0..(n - 1) {
            for ix in 0..(n - 1) {
                let a = idx(ix, iy, iz);
                let b = idx(ix + 1, iy, iz);
                let c = idx(ix, iy + 1, iz);
                let d = idx(ix + 1, iy + 1, iz);
                let e = idx(ix, iy, iz + 1);
                let f = idx(ix + 1, iy, iz + 1);
                let g = idx(ix, iy + 1, iz + 1);
                let h = idx(ix + 1, iy + 1, iz + 1);

                // Tet 1: A B D F
                indices.extend_from_slice(&[a, b, d, f]);
                // Tet 2: A D C G
                indices.extend_from_slice(&[a, d, c, g]);
                // Tet 3: A E F G
                indices.extend_from_slice(&[a, e, f, g]);
                // Tet 4: D F G H
                indices.extend_from_slice(&[d, f, g, h]);
                // Tet 5 (central): A F D G
                indices.extend_from_slice(&[a, f, d, g]);
            }
        }
    }

    // Per-tet parameters
    let mut densities = Vec::with_capacity(tet_count as usize);
    let mut color_grads = Vec::with_capacity(tet_count as usize * 3);
    for _ in 0..tet_count {
        densities.push(rng.random::<f32>() * 3.0 + 0.5);
        color_grads.push((rng.random::<f32>() - 0.5) * 0.2);
        color_grads.push((rng.random::<f32>() - 0.5) * 0.2);
        color_grads.push((rng.random::<f32>() - 0.5) * 0.2);
    }

    let circumdata = compute_circumspheres(&vertices, &indices);

    SceneData {
        vertices,
        indices,
        densities,
        color_grads,
        circumdata,
        start_pose: [0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0],
        vertex_count,
        tet_count,
    }
}

// ---------------------------------------------------------------------------
// GPU timestamp helpers
// ---------------------------------------------------------------------------

/// Create a GPU device with SUBGROUP + TIMESTAMP_QUERY features.
pub fn create_timestamp_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    create_test_device(TestDeviceConfig {
        extra_features: wgpu::Features::TIMESTAMP_QUERY,
        ..Default::default()
    })
}

/// Records GPU timestamps around compute passes for performance measurement.
///
/// Each call to [`allocate`] reserves a begin/end timestamp pair. After
/// recording passes and resolving, [`read_results`] returns per-pass
/// durations in milliseconds.
pub struct TimestampRecorder {
    query_set: wgpu::QuerySet,
    resolve_buf: wgpu::Buffer,
    names: Vec<String>,
    next_index: u32,
    capacity: u32,
    period_ns: f32,
}

impl TimestampRecorder {
    /// Create a new recorder with room for `capacity` named passes (2 queries each).
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, capacity: u32) -> Self {
        let query_count = capacity * 2;
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("timestamp_query_set"),
            ty: wgpu::QueryType::Timestamp,
            count: query_count,
        });
        let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp_resolve"),
            size: (query_count as u64) * 8, // u64 per query
            usage: wgpu::BufferUsages::QUERY_RESOLVE
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let period_ns = queue.get_timestamp_period();

        Self {
            query_set,
            resolve_buf,
            names: Vec::with_capacity(capacity as usize),
            next_index: 0,
            capacity,
            period_ns,
        }
    }

    /// Allocate a begin/end timestamp pair for a named pass.
    /// Returns (begin_index, end_index) for use with `ComputePassTimestampWrites`.
    pub fn allocate(&mut self, name: &str) -> (u32, u32) {
        assert!(
            self.next_index / 2 < self.capacity,
            "TimestampRecorder: exceeded capacity of {} passes",
            self.capacity
        );
        let begin = self.next_index;
        let end = self.next_index + 1;
        self.next_index += 2;
        self.names.push(name.to_string());
        (begin, end)
    }

    /// Reset for reuse (clears names and index counter, reuses GPU resources).
    pub fn reset(&mut self) {
        self.names.clear();
        self.next_index = 0;
    }

    /// Get a reference to the underlying query set (for building timestamp writes).
    pub fn query_set(&self) -> &wgpu::QuerySet {
        &self.query_set
    }

    /// Resolve all recorded timestamps to the readback buffer.
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.next_index > 0 {
            encoder.resolve_query_set(
                &self.query_set,
                0..self.next_index,
                &self.resolve_buf,
                0,
            );
        }
    }

    /// Read back resolved timestamps and compute per-pass durations in milliseconds.
    pub fn read_results(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Vec<(String, f64)> {
        if self.next_index == 0 {
            return Vec::new();
        }

        let byte_count = (self.next_index as u64) * 8;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp_readback"),
            size: byte_count,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&self.resolve_buf, 0, &readback, 0, byte_count);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        receiver.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let timestamps: &[u64] = bytemuck::cast_slice(&data);

        let mut results = Vec::with_capacity(self.names.len());
        for (i, name) in self.names.iter().enumerate() {
            let begin = timestamps[i * 2];
            let end = timestamps[i * 2 + 1];
            let ticks = end.wrapping_sub(begin);
            let ms = (ticks as f64) * (self.period_ns as f64) / 1_000_000.0;
            results.push((name.clone(), ms));
        }

        drop(data);
        readback.unmap();
        results
    }
}

/// Print a table of GPU timestamp results to stderr.
pub fn print_timestamp_table(results: &[(String, f64)]) {
    if results.is_empty() {
        return;
    }
    let max_name_len = results.iter().map(|(n, _)| n.len()).max().unwrap_or(10);
    eprintln!("\n{:─<width$}", "", width = max_name_len + 20);
    eprintln!("{:<width$}  {:>10}", "Pass", "Time (ms)", width = max_name_len);
    eprintln!("{:─<width$}", "", width = max_name_len + 20);
    let mut total = 0.0;
    for (name, ms) in results {
        eprintln!("{:<width$}  {:>10.3}", name, ms, width = max_name_len);
        total += ms;
    }
    eprintln!("{:─<width$}", "", width = max_name_len + 20);
    eprintln!("{:<width$}  {:>10.3}", "TOTAL", total, width = max_name_len);
    eprintln!("{:─<width$}", "", width = max_name_len + 20);
}
