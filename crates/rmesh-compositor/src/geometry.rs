//! Unit mesh generation for primitives (cube, sphere, plane, cylinder).
//! All meshes are centered at origin, fitting within a unit bounding box [-0.5, 0.5]^3.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// A single vertex with position and normal.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct PrimitiveVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

/// Offset and count into the shared vertex buffer for one primitive kind.
#[derive(Copy, Clone, Debug)]
pub struct MeshSlice {
    pub offset: u32,
    pub count: u32,
}

/// GPU vertex buffer containing all four unit meshes.
pub struct PrimitiveGeometry {
    pub vertex_buffer: wgpu::Buffer,
    pub kinds: [MeshSlice; 4],
}

impl PrimitiveGeometry {
    /// Generate all unit meshes and upload to the device.
    pub fn new(device: &wgpu::Device) -> Self {
        let cube = generate_cube();
        let sphere = generate_sphere(16, 8);
        let plane = generate_plane();
        let cylinder = generate_cylinder(24);

        let mut all_verts = Vec::new();
        let mut kinds = [MeshSlice { offset: 0, count: 0 }; 4];

        for (i, mesh) in [&cube, &sphere, &plane, &cylinder].iter().enumerate() {
            kinds[i] = MeshSlice {
                offset: all_verts.len() as u32,
                count: mesh.len() as u32,
            };
            all_verts.extend_from_slice(mesh);
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("primitive_geometry"),
            contents: bytemuck::cast_slice(&all_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            vertex_buffer,
            kinds,
        }
    }
}

fn generate_cube() -> Vec<PrimitiveVertex> {
    let mut verts = Vec::with_capacity(36);

    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        // +X
        ([1.0, 0.0, 0.0], [[0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5]]),
        // -X
        ([-1.0, 0.0, 0.0], [[-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5]]),
        // +Y
        ([0.0, 1.0, 0.0], [[-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]]),
        // -Y (fixed winding for back-face cull correctness)
        ([0.0, -1.0, 0.0], [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [-0.5, -0.5, -0.5]]),
        // +Z
        ([0.0, 0.0, 1.0], [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]]),
        // -Z
        ([0.0, 0.0, -1.0], [[0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5]]),
    ];

    for (normal, quad) in &faces {
        // Two triangles per quad (CCW winding)
        for &[a, b, c] in &[[0, 1, 2], [0, 2, 3]] {
            verts.push(PrimitiveVertex { position: quad[a], normal: *normal });
            verts.push(PrimitiveVertex { position: quad[b], normal: *normal });
            verts.push(PrimitiveVertex { position: quad[c], normal: *normal });
        }
    }

    verts
}

fn generate_sphere(slices: u32, stacks: u32) -> Vec<PrimitiveVertex> {
    let mut verts = Vec::new();
    let r = 0.5;

    for i in 0..stacks {
        let theta0 = std::f32::consts::PI * i as f32 / stacks as f32;
        let theta1 = std::f32::consts::PI * (i + 1) as f32 / stacks as f32;
        let (s0, c0) = (theta0.sin(), theta0.cos());
        let (s1, c1) = (theta1.sin(), theta1.cos());

        for j in 0..slices {
            let phi0 = 2.0 * std::f32::consts::PI * j as f32 / slices as f32;
            let phi1 = 2.0 * std::f32::consts::PI * (j + 1) as f32 / slices as f32;
            let (sp0, cp0) = (phi0.sin(), phi0.cos());
            let (sp1, cp1) = (phi1.sin(), phi1.cos());

            // Four corners of the quad
            let n00 = [s0 * cp0, c0, s0 * sp0];
            let n10 = [s1 * cp0, c1, s1 * sp0];
            let n01 = [s0 * cp1, c0, s0 * sp1];
            let n11 = [s1 * cp1, c1, s1 * sp1];

            let p00 = [n00[0] * r, n00[1] * r, n00[2] * r];
            let p10 = [n10[0] * r, n10[1] * r, n10[2] * r];
            let p01 = [n01[0] * r, n01[1] * r, n01[2] * r];
            let p11 = [n11[0] * r, n11[1] * r, n11[2] * r];

            // Top triangle (skip degenerate at north pole)
            if i > 0 {
                verts.push(PrimitiveVertex { position: p00, normal: n00 });
                verts.push(PrimitiveVertex { position: p10, normal: n10 });
                verts.push(PrimitiveVertex { position: p11, normal: n11 });
            }
            // Bottom triangle (skip degenerate at south pole)
            if i < stacks - 1 {
                verts.push(PrimitiveVertex { position: p00, normal: n00 });
                verts.push(PrimitiveVertex { position: p11, normal: n11 });
                verts.push(PrimitiveVertex { position: p01, normal: n01 });
            }
        }
    }

    verts
}

fn generate_plane() -> Vec<PrimitiveVertex> {
    let n = [0.0, 1.0, 0.0];
    let quad = [
        [-0.5, 0.0, -0.5],
        [0.5, 0.0, -0.5],
        [0.5, 0.0, 0.5],
        [-0.5, 0.0, 0.5],
    ];

    vec![
        PrimitiveVertex { position: quad[0], normal: n },
        PrimitiveVertex { position: quad[1], normal: n },
        PrimitiveVertex { position: quad[2], normal: n },
        PrimitiveVertex { position: quad[0], normal: n },
        PrimitiveVertex { position: quad[2], normal: n },
        PrimitiveVertex { position: quad[3], normal: n },
    ]
}

fn generate_cylinder(segments: u32) -> Vec<PrimitiveVertex> {
    let mut verts = Vec::new();
    let r = 0.5;
    let h = 0.5;

    for i in 0..segments {
        let a0 = 2.0 * std::f32::consts::PI * i as f32 / segments as f32;
        let a1 = 2.0 * std::f32::consts::PI * (i + 1) as f32 / segments as f32;
        let (s0, c0) = (a0.sin(), a0.cos());
        let (s1, c1) = (a1.sin(), a1.cos());

        // Side quad
        let n0 = [c0, 0.0, s0];
        let n1 = [c1, 0.0, s1];
        let p0b = [c0 * r, -h, s0 * r];
        let p1b = [c1 * r, -h, s1 * r];
        let p0t = [c0 * r, h, s0 * r];
        let p1t = [c1 * r, h, s1 * r];

        verts.push(PrimitiveVertex { position: p0b, normal: n0 });
        verts.push(PrimitiveVertex { position: p1b, normal: n1 });
        verts.push(PrimitiveVertex { position: p1t, normal: n1 });

        verts.push(PrimitiveVertex { position: p0b, normal: n0 });
        verts.push(PrimitiveVertex { position: p1t, normal: n1 });
        verts.push(PrimitiveVertex { position: p0t, normal: n0 });

        // Top cap
        let nt = [0.0, 1.0, 0.0];
        verts.push(PrimitiveVertex { position: [0.0, h, 0.0], normal: nt });
        verts.push(PrimitiveVertex { position: p0t, normal: nt });
        verts.push(PrimitiveVertex { position: p1t, normal: nt });

        // Bottom cap
        let nb = [0.0, -1.0, 0.0];
        verts.push(PrimitiveVertex { position: [0.0, -h, 0.0], normal: nb });
        verts.push(PrimitiveVertex { position: p1b, normal: nb });
        verts.push(PrimitiveVertex { position: p0b, normal: nb });
    }

    verts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_vertex_counts() {
        let cube = generate_cube();
        assert_eq!(cube.len(), 36);

        let sphere = generate_sphere(16, 8);
        assert!(sphere.len() > 400, "sphere should have many vertices, got {}", sphere.len());

        let plane = generate_plane();
        assert_eq!(plane.len(), 6);

        let cylinder = generate_cylinder(24);
        assert!(cylinder.len() > 100, "cylinder should have many vertices, got {}", cylinder.len());
    }

    #[test]
    fn all_normals_unit_length() {
        for mesh in [generate_cube(), generate_sphere(16, 8), generate_plane(), generate_cylinder(24)] {
            for v in &mesh {
                let len = (v.normal[0].powi(2) + v.normal[1].powi(2) + v.normal[2].powi(2)).sqrt();
                assert!((len - 1.0).abs() < 0.01, "normal not unit length: {:?} (len={})", v.normal, len);
            }
        }
    }
}
