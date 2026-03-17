// Precompute tet geometry: volumes, centroids, face area-weighted normals, Laplacian coefficients.
// Run once at load time.

struct FluidUniforms {
    dt: f32,
    viscosity: f32,
    tet_count: u32,
    jacobi_iter: u32,
    gravity: vec4<f32>,
    source_pos: vec4<f32>,
    source_strength: f32,
    buoyancy: f32,
    density_scale: f32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> uniforms: FluidUniforms;
@group(0) @binding(1) var<storage, read> vertices: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> tet_neighbors: array<i32>;

@group(1) @binding(0) var<storage, read_write> tet_volumes: array<f32>;
@group(1) @binding(1) var<storage, read_write> face_geo: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read_write> tet_centers: array<vec4<f32>>;

fn read_vertex(vi: u32) -> vec3<f32> {
    return vec3<f32>(vertices[vi * 3u], vertices[vi * 3u + 1u], vertices[vi * 3u + 2u]);
}

// Face table: each face has 3 vertex local indices and 1 opposite vertex index.
// Face 0: (0,2,1) opp 3
// Face 1: (1,2,3) opp 0
// Face 2: (0,3,2) opp 1
// Face 3: (3,0,1) opp 2
const FACE_A = array<u32, 4>(0u, 1u, 0u, 3u);
const FACE_B = array<u32, 4>(2u, 2u, 3u, 0u);
const FACE_C = array<u32, 4>(1u, 3u, 2u, 1u);
const FACE_OPP = array<u32, 4>(3u, 0u, 1u, 2u);

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.tet_count) {
        return;
    }

    // Read tet vertex indices
    let i0 = indices[tid * 4u + 0u];
    let i1 = indices[tid * 4u + 1u];
    let i2 = indices[tid * 4u + 2u];
    let i3 = indices[tid * 4u + 3u];

    // Read vertex positions
    let v0 = read_vertex(i0);
    let v1 = read_vertex(i1);
    let v2 = read_vertex(i2);
    let v3 = read_vertex(i3);
    let verts = array<vec3<f32>, 4>(v0, v1, v2, v3);

    // Centroid
    let center = (v0 + v1 + v2 + v3) * 0.25;
    tet_centers[tid] = vec4<f32>(center, 0.0);

    // Volume = |det([v1-v0, v2-v0, v3-v0])| / 6
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let e3 = v3 - v0;
    let vol = abs(dot(e1, cross(e2, e3))) / 6.0;
    tet_volumes[tid] = vol;

    // Compute neighbor centroids lazily per face
    for (var f = 0u; f < 4u; f++) {
        let va = verts[FACE_A[f]];
        let vb = verts[FACE_B[f]];
        let vc = verts[FACE_C[f]];
        let v_opp = verts[FACE_OPP[f]];

        // Face normal = cross(vc - va, vb - va)
        var normal = cross(vc - va, vb - va);
        let len = length(normal);
        let area = len * 0.5;

        if (len > 0.0) {
            normal = normal / len; // unit normal
        }

        // Orient outward: normal should point AWAY from opposite vertex
        if (dot(normal, v_opp - va) > 0.0) {
            normal = -normal;
        }

        let neighbor = tet_neighbors[tid * 4u + f];
        var coeff = 0.0;

        if (neighbor >= 0) {
            // Compute neighbor centroid
            let ni0 = indices[u32(neighbor) * 4u + 0u];
            let ni1 = indices[u32(neighbor) * 4u + 1u];
            let ni2 = indices[u32(neighbor) * 4u + 2u];
            let ni3 = indices[u32(neighbor) * 4u + 3u];
            let n_center = (read_vertex(ni0) + read_vertex(ni1) + read_vertex(ni2) + read_vertex(ni3)) * 0.25;
            let d_ij = length(n_center - center);
            if (d_ij > 0.0) {
                coeff = area / d_ij;
            }
        }

        // Store area-weighted normal in xyz, Laplacian coefficient in w
        face_geo[tid * 4u + f] = vec4<f32>(normal * area, coeff);
    }
}
