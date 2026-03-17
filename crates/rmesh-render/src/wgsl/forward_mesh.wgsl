// Mesh shader: replaces forward_vertex.wgsl for the sorted forward pass.
//
// 32 threads per workgroup → 8 tets per workgroup, 4 threads per tet.
// Each thread owns one tet vertex AND one face. Positions and normals
// are shared via subgroupShuffle (groups of 4 consecutive threads never
// straddle a subgroup boundary for subgroup sizes 4/8/16/32/64).
//
// Back-facing faces emit degenerate triangles (rasterizer discards them).
// The fragment shader (forward_fragment.wgsl) is reused unchanged.

enable wgpu_mesh_shader;
// enable subgroups;  // causes naga crash, unnecessary

struct Uniforms {
    vp_col0: vec4<f32>,
    vp_col1: vec4<f32>,
    vp_col2: vec4<f32>,
    vp_col3: vec4<f32>,
    c2w_col0: vec4<f32>,
    c2w_col1: vec4<f32>,
    c2w_col2: vec4<f32>,
    intrinsics: vec4<f32>,
    cam_pos_pad: vec4<f32>,
    screen_width: f32,
    screen_height: f32,
    tet_count: u32,
    step: u32,
    tile_size_u: u32,
    ray_mode: u32,
    min_t: f32,
    _pad1c: u32,
    _pad2: vec4<u32>,
};

struct DrawIndirectArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

// Storage bindings: same 7 as vertex shader + binding 7 for indirect_args
@group(0) @binding(0) var<storage, read> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> vertices: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> colors: array<f32>;
@group(0) @binding(4) var<storage, read> densities: array<f32>;
@group(0) @binding(5) var<storage, read> color_grads: array<f32>;
@group(0) @binding(6) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(7) var<storage, read> indirect_args: DrawIndirectArgs;

// Face table: (a, b, c, opposite_vertex) -- opposite used to flip normal inward
const TET_FACES: array<vec4<u32>, 4> = array<vec4<u32>, 4>(
    vec4<u32>(0u, 2u, 1u, 3u),
    vec4<u32>(1u, 2u, 3u, 0u),
    vec4<u32>(0u, 3u, 2u, 1u),
    vec4<u32>(3u, 0u, 1u, 2u),
);

// Vertex output: matches FragmentInput in forward_fragment.wgsl exactly
struct MeshVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) tet_density: f32,
    @location(1) @interpolate(flat) base_color: vec3<f32>,
    @location(2) plane_numerators: vec4<f32>,
    @location(3) plane_denominators: vec4<f32>,
    @location(4) ray_dir: vec3<f32>,
    @location(5) dc_dt: f32,
    @location(6) @interpolate(flat) face_n0: vec3<f32>,
    @location(7) @interpolate(flat) face_n1: vec3<f32>,
    @location(8) @interpolate(flat) face_n2: vec3<f32>,
    @location(9) @interpolate(flat) face_n3: vec3<f32>,
};

struct PrimitiveOutput {
    @builtin(triangle_indices) indices: vec3<u32>,
};

// 8 tets × 4 verts = 32 vertices, 8 tets × 4 faces = 32 primitives
struct MeshOutput {
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitive_count) primitive_count: u32,
    @builtin(vertices) verts: array<MeshVertexOutput, 32>,
    @builtin(primitives) prims: array<PrimitiveOutput, 32>,
};

var<workgroup> mesh_out: MeshOutput;

fn load_f32x3(buf_base: u32) -> vec3<f32> {
    return vec3<f32>(vertices[buf_base], vertices[buf_base + 1u], vertices[buf_base + 2u]);
}

fn load_color(idx: u32) -> vec3<f32> {
    return vec3<f32>(colors[idx * 3u], colors[idx * 3u + 1u], colors[idx * 3u + 2u]);
}

fn load_grad(idx: u32) -> vec3<f32> {
    return vec3<f32>(color_grads[idx * 3u], color_grads[idx * 3u + 1u], color_grads[idx * 3u + 2u]);
}

const TETS_PER_GROUP: u32 = 8u;

@mesh(mesh_out) @workgroup_size(32)
fn main(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(subgroup_invocation_id) subgroup_lane: u32,
) {
    let visible_count = indirect_args.instance_count;
    let tet_local = lid / 4u;   // 0..7 — which tet within the workgroup
    let vert_local = lid % 4u;  // 0..3 — which vertex/face within the tet
    let tet_global = wg_id.x * TETS_PER_GROUP + tet_local;

    // How many tets this workgroup processes
    let wg_first = wg_id.x * TETS_PER_GROUP;
    let wg_count = min(TETS_PER_GROUP, select(0u, visible_count - wg_first, visible_count > wg_first));

    // Thread 0 sets output counts
    if lid == 0u {
        mesh_out.vertex_count = wg_count * 4u;
        mesh_out.primitive_count = wg_count * 4u;
    }

    // Guard: only threads with valid tets do work
    if tet_global < visible_count {
        let tet_id = sorted_indices[tet_global];

        // Load 4 vertex indices
        let i0 = indices[tet_id * 4u];
        let i1 = indices[tet_id * 4u + 1u];
        let i2 = indices[tet_id * 4u + 2u];
        let i3 = indices[tet_id * 4u + 3u];

        // Each thread loads its own vertex position (thread k loads vertex k)
        var idx_arr = array<u32, 4>(i0, i1, i2, i3);
        let my_pos = load_f32x3(idx_arr[vert_local] * 3u);

        // Share 4 vertex positions via subgroupShuffle
        let base = (subgroup_lane / 4u) * 4u;
        let v0 = vec3<f32>(
            subgroupShuffle(my_pos.x, base),
            subgroupShuffle(my_pos.y, base),
            subgroupShuffle(my_pos.z, base),
        );
        let v1 = vec3<f32>(
            subgroupShuffle(my_pos.x, base + 1u),
            subgroupShuffle(my_pos.y, base + 1u),
            subgroupShuffle(my_pos.z, base + 1u),
        );
        let v2 = vec3<f32>(
            subgroupShuffle(my_pos.x, base + 2u),
            subgroupShuffle(my_pos.y, base + 2u),
            subgroupShuffle(my_pos.z, base + 2u),
        );
        let v3 = vec3<f32>(
            subgroupShuffle(my_pos.x, base + 3u),
            subgroupShuffle(my_pos.y, base + 3u),
            subgroupShuffle(my_pos.z, base + 3u),
        );

        // Each thread computes its face's inward normal (thread k → face k)
        let face = TET_FACES[vert_local];
        var varr = array<vec3<f32>, 4>(v0, v1, v2, v3);
        let fa = varr[face[0]];
        let fb = varr[face[1]];
        let fc = varr[face[2]];
        var my_n = cross(fc - fa, fb - fa);
        // Flip normal to point inward (toward opposite vertex)
        let v_opp = varr[face[3]];
        if dot(my_n, v_opp - fa) < 0.0 {
            my_n = -my_n;
        }

        // Share 4 face normals via subgroupShuffle
        let n0 = vec3<f32>(
            subgroupShuffle(my_n.x, base),
            subgroupShuffle(my_n.y, base),
            subgroupShuffle(my_n.z, base),
        );
        let n1 = vec3<f32>(
            subgroupShuffle(my_n.x, base + 1u),
            subgroupShuffle(my_n.y, base + 1u),
            subgroupShuffle(my_n.z, base + 1u),
        );
        let n2 = vec3<f32>(
            subgroupShuffle(my_n.x, base + 2u),
            subgroupShuffle(my_n.y, base + 2u),
            subgroupShuffle(my_n.z, base + 2u),
        );
        let n3 = vec3<f32>(
            subgroupShuffle(my_n.x, base + 3u),
            subgroupShuffle(my_n.y, base + 3u),
            subgroupShuffle(my_n.z, base + 3u),
        );

        let cam = uniforms.cam_pos_pad.xyz;

        // Share 4 plane numerators (one per face, per-tet constants)
        let my_num = dot(my_n, fa - cam);
        let num0 = subgroupShuffle(my_num, base);
        let num1 = subgroupShuffle(my_num, base + 1u);
        let num2 = subgroupShuffle(my_num, base + 2u);
        let num3 = subgroupShuffle(my_num, base + 3u);

        // Per-vertex: ray_dir and plane denominators
        let ray_dir = my_pos - cam;
        let denom0 = dot(n0, ray_dir);
        let denom1 = dot(n1, ray_dir);
        let denom2 = dot(n2, ray_dir);
        let denom3 = dot(n3, ray_dir);

        // Per-tet material data
        let tet_density = densities[tet_id];
        let color = load_color(tet_id);
        let grad = load_grad(tet_id);
        let dc_dt = dot(grad, ray_dir);
        let offset = dot(grad, cam - v0);
        let base_color = color + vec3<f32>(offset);

        // Project vertex to clip space
        let vp = mat4x4<f32>(uniforms.vp_col0, uniforms.vp_col1, uniforms.vp_col2, uniforms.vp_col3);
        let clip_pos = vp * vec4<f32>(my_pos, 1.0);

        // Write vertex output
        let vi = tet_local * 4u + vert_local;
        var out: MeshVertexOutput;
        out.position = clip_pos;
        out.tet_density = tet_density;
        out.base_color = base_color;
        out.plane_numerators = vec4<f32>(num0, num1, num2, num3);
        out.plane_denominators = vec4<f32>(denom0, denom1, denom2, denom3);
        out.ray_dir = ray_dir;
        out.dc_dt = dc_dt;
        out.face_n0 = n0;
        out.face_n1 = n1;
        out.face_n2 = n2;
        out.face_n3 = n3;
        mesh_out.verts[vi] = out;

        // Primitive: front-facing → real triangle, back-facing → degenerate
        let tb = tet_local * 4u;
        if my_num > 0.0 {
            mesh_out.prims[vi].indices = vec3<u32>(
                tb + face[0], tb + face[1], tb + face[2],
            );
        } else {
            mesh_out.prims[vi].indices = vec3<u32>(tb, tb, tb);
        }
    }
}
