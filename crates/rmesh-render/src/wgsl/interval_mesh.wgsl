// Interval shading mesh shader.
//
// Decomposes each tet into 3-4 non-overlapping screen-space triangles with
// interpolated front/back NDC depths. Each fragment is shaded exactly once.
//
// 16 threads per workgroup → 1 thread per tet (16 tets per workgroup).
// Max output: 16 × 15 = 240 vertices, 16 × 12 = 192 primitives.

enable wgpu_mesh_shader;

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
    sh_degree: u32,
    near_plane: f32,
    far_plane: f32,
    _pad1: vec2<u32>,
};

struct DrawIndirectArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0) var<storage, read> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> vertices: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> colors: array<f32>;
@group(0) @binding(4) var<storage, read> densities: array<f32>;
@group(0) @binding(5) var<storage, read> color_grads: array<f32>;
@group(0) @binding(6) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(7) var<storage, read> indirect_args: DrawIndirectArgs;

// Vertex output: interval shading attributes
struct IntervalVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) depths: vec2<f32>,                  // (z_front_ndc, z_back_ndc)
    @location(1) color_offsets: vec2<f32>,            // (offset_front, offset_back)
    @location(2) @interpolate(flat) density: f32,
    @location(3) @interpolate(flat) base_color: vec3<f32>,
};

struct PrimitiveOutput {
    @builtin(triangle_indices) indices: vec3<u32>,
};

// 16 tets × max 15 verts = 240, 16 tets × max 12 prims = 192
// (within the 256 vertex / 256 primitive hardware limits)
const MAX_VERTS: u32 = 240u;
const MAX_PRIMS: u32 = 192u;

struct MeshOutput {
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitive_count) primitive_count: u32,
    @builtin(vertices) verts: array<IntervalVertexOutput, 240>,
    @builtin(primitives) prims: array<PrimitiveOutput, 192>,
};

var<workgroup> mesh_out: MeshOutput;

// Shared prefix sums for compacting output
var<workgroup> vert_offsets: array<u32, 16>;
var<workgroup> prim_offsets: array<u32, 16>;

fn load_f32x3(buf_base: u32) -> vec3<f32> {
    return vec3<f32>(vertices[buf_base], vertices[buf_base + 1u], vertices[buf_base + 2u]);
}

fn load_color(idx: u32) -> vec3<f32> {
    return vec3<f32>(colors[idx * 3u], colors[idx * 3u + 1u], colors[idx * 3u + 2u]);
}

fn load_grad(idx: u32) -> vec3<f32> {
    return vec3<f32>(color_grads[idx * 3u], color_grads[idx * 3u + 1u], color_grads[idx * 3u + 2u]);
}

const TETS_PER_GROUP: u32 = 16u;

// Maps vertex index → TET_FACES index where that vertex is face[3] (opposite)
const OPPOSITE_FACE: array<u32, 4> = array<u32, 4>(1u, 2u, 3u, 0u);

// Face table: (a, b, c, opposite_vertex)
const TET_FACES: array<vec4<u32>, 4> = array<vec4<u32>, 4>(
    vec4<u32>(0u, 2u, 1u, 3u),
    vec4<u32>(1u, 2u, 3u, 0u),
    vec4<u32>(0u, 3u, 2u, 1u),
    vec4<u32>(3u, 0u, 1u, 2u),
);

// Edge table: 6 edges of a tet (a, b)
const TET_EDGES: array<vec2<u32>, 6> = array<vec2<u32>, 6>(
    vec2<u32>(0u, 1u),
    vec2<u32>(0u, 2u),
    vec2<u32>(0u, 3u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 3u),
    vec2<u32>(2u, 3u),
);

// Classify which vertex projects inside the opposite face's screen triangle.
// Returns the index (0-3) of the interior vertex, or 4 if none / ambiguous (Case 2).
fn classify_silhouette(p: array<vec2<f32>, 4>) -> u32 {
    // For each vertex, check if it lies inside the screen triangle of the opposite face
    for (var v = 0u; v < 4u; v++) {
        let face = TET_FACES[v]; // face opposite to vertex v has indices face.xyz, opposite=v
        // Actually face[3] == v (opposite vertex). The face vertices are face[0], face[1], face[2].
        let a = p[face[0]];
        let b = p[face[1]];
        let c = p[face[2]];
        let pt = p[face[3]]; // = p[v]

        // Barycentric sign test (2D cross products)
        let d0 = (b.x - a.x) * (pt.y - a.y) - (b.y - a.y) * (pt.x - a.x);
        let d1 = (c.x - b.x) * (pt.y - b.y) - (c.y - b.y) * (pt.x - b.x);
        let d2 = (a.x - c.x) * (pt.y - c.y) - (a.y - c.y) * (pt.x - c.x);

        let all_pos = d0 >= 0.0 && d1 >= 0.0 && d2 >= 0.0;
        let all_neg = d0 <= 0.0 && d1 <= 0.0 && d2 <= 0.0;

        if all_pos || all_neg {
            return face[3]; // The vertex that projects inside
        }
    }
    return 4u; // Case 2: no vertex inside any opposite face
}

// Compute barycentric coordinates of point pt in triangle (a, b, c) in 2D.
fn bary2d(pt: vec2<f32>, a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> vec3<f32> {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = pt - a;
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);
    let denom = d00 * d11 - d01 * d01;
    let inv_denom = select(0.0, 1.0 / denom, abs(denom) > 1e-10);
    let v = (d11 * d20 - d01 * d21) * inv_denom;
    let w = (d00 * d21 - d01 * d20) * inv_denom;
    return vec3<f32>(1.0 - v - w, v, w);
}

// Line-line intersection in 2D. Returns parameter t along line (a1,a2) where it crosses (b1,b2).
fn line_intersect_t(a1: vec2<f32>, a2: vec2<f32>, b1: vec2<f32>, b2: vec2<f32>) -> f32 {
    let d1 = a2 - a1;
    let d2 = b2 - b1;
    let denom = d1.x * d2.y - d1.y * d2.x;
    if abs(denom) < 1e-10 {
        return -1.0; // parallel — no valid intersection
    }
    let d = b1 - a1;
    return (d.x * d2.y - d.y * d2.x) / denom;
}

// Find which pair of opposite edges cross in screen space (Case 2).
// Opposite edge pairs: (01,23), (02,13), (03,12)
// Returns: (edge_a_idx, edge_b_idx) where the edges are from TET_EDGES
fn find_crossing_edges(p: array<vec2<f32>, 4>) -> vec2<u32> {
    // Opposite edge pairs by TET_EDGES index: (0,5)=(01,23), (1,4)=(02,13), (2,3)=(03,12)
    let pairs = array<vec2<u32>, 3>(
        vec2<u32>(0u, 5u),
        vec2<u32>(1u, 4u),
        vec2<u32>(2u, 3u),
    );

    for (var i = 0u; i < 3u; i++) {
        let ea = TET_EDGES[pairs[i].x];
        let eb = TET_EDGES[pairs[i].y];

        let a1 = p[ea.x]; let a2 = p[ea.y];
        let b1 = p[eb.x]; let b2 = p[eb.y];

        // Check if segments actually cross
        let ta = line_intersect_t(a1, a2, b1, b2);
        let tb = line_intersect_t(b1, b2, a1, a2);

        if ta > 0.0 && ta < 1.0 && tb > 0.0 && tb < 1.0 {
            return pairs[i];
        }
    }
    // Fallback: use first pair
    return pairs[0];
}

@mesh(mesh_out) @workgroup_size(16)
fn main(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let visible_count = indirect_args.instance_count;
    let tet_global = wg_id.x * TETS_PER_GROUP + lid;

    // Determine how many tets this workgroup processes
    let wg_first = wg_id.x * TETS_PER_GROUP;
    let wg_count = min(TETS_PER_GROUP, select(0u, visible_count - wg_first, visible_count > wg_first));

    // Per-tet output counts
    var my_vert_count = 0u;
    var my_prim_count = 0u;

    // Per-tet data (computed below)
    var ndc_xy: array<vec2<f32>, 4>;
    var ndc_z: array<f32, 4>;
    var tet_density: f32;
    var base_color: vec3<f32>;
    var grad: vec3<f32>;
    var v_world: array<vec3<f32>, 4>;
    var silhouette_case: u32 = 0u; // 1 = interior vertex, 2 = crossing edges
    var interior_vtx: u32 = 4u;
    var crossing_edges: vec2<u32>;

    if lid < wg_count && tet_global < visible_count {
        let tet_id = sorted_indices[tet_global];

        // Load 4 vertex positions
        let i0 = indices[tet_id * 4u];
        let i1 = indices[tet_id * 4u + 1u];
        let i2 = indices[tet_id * 4u + 2u];
        let i3 = indices[tet_id * 4u + 3u];

        v_world[0] = load_f32x3(i0 * 3u);
        v_world[1] = load_f32x3(i1 * 3u);
        v_world[2] = load_f32x3(i2 * 3u);
        v_world[3] = load_f32x3(i3 * 3u);

        // Transform to clip space
        let vp = mat4x4<f32>(uniforms.vp_col0, uniforms.vp_col1, uniforms.vp_col2, uniforms.vp_col3);

        var clip: array<vec4<f32>, 4>;
        for (var vi = 0u; vi < 4u; vi++) {
            clip[vi] = vp * vec4<f32>(v_world[vi], 1.0);
        }

        // Skip tets with any vertex behind the camera — NDC is unreliable.
        // (Matches tiled compute path behavior.)
        let any_behind = clip[0].w <= 0.0 || clip[1].w <= 0.0
                      || clip[2].w <= 0.0 || clip[3].w <= 0.0;

        if !any_behind {

        // Perspective divide → NDC
        for (var vi = 0u; vi < 4u; vi++) {
            let w = clip[vi].w;
            let inv_w = select(0.0, 1.0 / w, abs(w) > 1e-10);
            ndc_xy[vi] = clip[vi].xy * inv_w;
            ndc_z[vi] = clip[vi].z * inv_w;
        }

        // Material data
        tet_density = densities[tet_id];
        let color = load_color(tet_id);
        grad = load_grad(tet_id);

        // Base color at centroid
        let centroid = (v_world[0] + v_world[1] + v_world[2] + v_world[3]) * 0.25;
        let cam_pos = uniforms.cam_pos_pad.xyz;
        base_color = color + vec3<f32>(dot(grad, cam_pos - v_world[0]));

        // Classify silhouette
        interior_vtx = classify_silhouette(ndc_xy);

        if interior_vtx < 4u {
            silhouette_case = 1u;
            my_vert_count = 4u;
            my_prim_count = 3u;
        } else {
            silhouette_case = 2u;
            crossing_edges = find_crossing_edges(ndc_xy);
            my_vert_count = 5u;
            my_prim_count = 4u;
        }

        // Skip degenerate tets (all NDC Z very close)
        let z_range = max(max(ndc_z[0], ndc_z[1]), max(ndc_z[2], ndc_z[3]))
                    - min(min(ndc_z[0], ndc_z[1]), min(ndc_z[2], ndc_z[3]));
        if z_range < 1e-7 || tet_density < 1e-6 {
            my_vert_count = 0u;
            my_prim_count = 0u;
        }

        } // end if !any_behind

    }

    // Workgroup prefix sum for compact output allocation
    vert_offsets[lid] = my_vert_count;
    prim_offsets[lid] = my_prim_count;
    workgroupBarrier();

    // Inclusive prefix sum (simple serial — only 16 elements)
    if lid == 0u {
        var v_total = 0u;
        var p_total = 0u;
        for (var i = 0u; i < TETS_PER_GROUP; i++) {
            let vc = vert_offsets[i];
            let pc = prim_offsets[i];
            vert_offsets[i] = v_total;
            prim_offsets[i] = p_total;
            v_total += vc;
            p_total += pc;
        }
        mesh_out.vertex_count = v_total;
        mesh_out.primitive_count = p_total;
    }
    workgroupBarrier();

    // Guard: only threads with output proceed.
    // NOTE: Do NOT use `return` here — early return from divergent threads hangs mesh shaders.
    if my_vert_count > 0u {

    let v_base = vert_offsets[lid];
    let p_base = prim_offsets[lid];

    // Helper: compute front/back NDC depths for a silhouette vertex (on the tet boundary)
    // At a silhouette vertex, front = back = vertex NDC Z
    // For the center vertex (inside the silhouette), front = front-face Z, back = back-face Z

    let cam_pos = uniforms.cam_pos_pad.xyz;

    if silhouette_case == 1u {
        // Case 1: vertex `interior_vtx` projects inside opposite face
        // Look up the face where interior_vtx is the opposite vertex.
        let face = TET_FACES[OPPOSITE_FACE[interior_vtx]]; // face[3] == interior_vtx
        let si0 = face[0]; // silhouette vertex indices
        let si1 = face[1];
        let si2 = face[2];
        let center = face[3]; // = interior_vtx

        // Compute barycentrics of center projection in silhouette triangle
        let bary = bary2d(
            ndc_xy[center],
            ndc_xy[si0], ndc_xy[si1], ndc_xy[si2],
        );

        // Center vertex front/back depths:
        // The center vertex sees two faces: the opposite face (front or back) and
        // we need to determine which faces are front/back by checking face normals.
        // For simplicity: front = interpolated Z from silhouette face, back = center vertex Z
        // (or vice versa depending on depth ordering)
        let z_face = bary.x * ndc_z[si0] + bary.y * ndc_z[si1] + bary.z * ndc_z[si2];
        let z_center = ndc_z[center];

        // Front = closer to camera (smaller NDC Z in wgpu [0,1] convention)
        let z_front = min(z_face, z_center);
        let z_back = max(z_face, z_center);

        // Color offsets at entry/exit for center vertex
        // The ray through the center pixel enters at z_front and exits at z_back.
        // We compute the world-space points at entry/exit and evaluate grad·(pt - v0).
        // Approximate: offset_front/back from barycentric interp on the respective face.

        // For silhouette vertices: offset = dot(grad, v_world[i] - v_world[0])
        // (using v_world[0] as reference since base_color = color + dot(grad, cam - v0))
        // Actually base_color already includes cam offset. The color at a point p is:
        //   color(p) = base_colors[tet] + dot(grad, p - v0)
        // base_color = base_colors[tet] + dot(grad, cam - v0)
        // So: color(p) = base_color + dot(grad, p - cam)
        // At silhouette vertex i: offset = dot(grad, v_world[i] - cam)
        // At center: offset_front = bary interp of silhouette offsets (face side)
        //            offset_back = dot(grad, v_world[center] - cam) (vertex side)
        var sv_offset: array<f32, 3>;
        var sv_idx = array<u32, 3>(si0, si1, si2);
        for (var i = 0u; i < 3u; i++) {
            sv_offset[i] = dot(grad, v_world[sv_idx[i]] - cam_pos);
        }
        let center_offset = dot(grad, v_world[center] - cam_pos);

        let face_offset = bary.x * sv_offset[0] + bary.y * sv_offset[1] + bary.z * sv_offset[2];

        // Assign front/back offsets based on depth ordering
        var offset_front: f32;
        var offset_back: f32;
        if z_face <= z_center {
            offset_front = face_offset;
            offset_back = center_offset;
        } else {
            offset_front = center_offset;
            offset_back = face_offset;
        }

        // Emit 4 vertices: 3 silhouette + 1 center
        // Silhouette vertices: z_front = z_back = own Z, offset_front = offset_back = own offset
        for (var i = 0u; i < 3u; i++) {
            var out: IntervalVertexOutput;
            out.position = vec4<f32>(ndc_xy[sv_idx[i]], 0.0, 1.0);
            out.depths = vec2<f32>(ndc_z[sv_idx[i]], ndc_z[sv_idx[i]]);
            out.color_offsets = vec2<f32>(sv_offset[i], sv_offset[i]);
            out.density = tet_density;
            out.base_color = base_color;
            mesh_out.verts[v_base + i] = out;
        }

        // Center vertex
        var out_c: IntervalVertexOutput;
        out_c.position = vec4<f32>(ndc_xy[center], 0.0, 1.0);
        out_c.depths = vec2<f32>(z_front, z_back);
        out_c.color_offsets = vec2<f32>(offset_front, offset_back);
        out_c.density = tet_density;
        out_c.base_color = base_color;
        mesh_out.verts[v_base + 3u] = out_c;

        // 3 triangles forming a fan around center vertex (index 3 in local space)
        mesh_out.prims[p_base + 0u].indices = vec3<u32>(v_base + 0u, v_base + 1u, v_base + 3u);
        mesh_out.prims[p_base + 1u].indices = vec3<u32>(v_base + 1u, v_base + 2u, v_base + 3u);
        mesh_out.prims[p_base + 2u].indices = vec3<u32>(v_base + 2u, v_base + 0u, v_base + 3u);

    } else if silhouette_case == 2u {
        // Case 2: two opposite edges cross in screen space → 5 verts, 4 triangles
        let ea_idx = crossing_edges.x;
        let eb_idx = crossing_edges.y;
        let ea = TET_EDGES[ea_idx];
        let eb = TET_EDGES[eb_idx];

        // Compute the screen-space intersection point
        let ta = line_intersect_t(ndc_xy[ea.x], ndc_xy[ea.y], ndc_xy[eb.x], ndc_xy[eb.y]);
        let tb = line_intersect_t(ndc_xy[eb.x], ndc_xy[eb.y], ndc_xy[ea.x], ndc_xy[ea.y]);

        let center_xy = mix(ndc_xy[ea.x], ndc_xy[ea.y], ta);

        // Interpolate NDC Z along each edge
        let z_a = mix(ndc_z[ea.x], ndc_z[ea.y], ta);
        let z_b = mix(ndc_z[eb.x], ndc_z[eb.y], tb);

        let z_front = min(z_a, z_b);
        let z_back = max(z_a, z_b);

        // Color offsets at the intersection point along each edge
        let off_ea_x = dot(grad, v_world[ea.x] - cam_pos);
        let off_ea_y = dot(grad, v_world[ea.y] - cam_pos);
        let off_eb_x = dot(grad, v_world[eb.x] - cam_pos);
        let off_eb_y = dot(grad, v_world[eb.y] - cam_pos);

        let off_a = mix(off_ea_x, off_ea_y, ta);
        let off_b = mix(off_eb_x, off_eb_y, tb);

        var offset_front: f32;
        var offset_back: f32;
        if z_a <= z_b {
            offset_front = off_a;
            offset_back = off_b;
        } else {
            offset_front = off_b;
            offset_back = off_a;
        }

        // The 4 silhouette vertices are the endpoints of the crossing edges.
        // Order them around the silhouette: ea.x, eb.x, ea.y, eb.y
        // (since the edges cross, their endpoints alternate around the silhouette)
        let sv = array<u32, 4>(ea.x, eb.x, ea.y, eb.y);

        // Emit 4 silhouette vertices
        for (var i = 0u; i < 4u; i++) {
            let vi = sv[i];
            let off = dot(grad, v_world[vi] - cam_pos);
            var out: IntervalVertexOutput;
            out.position = vec4<f32>(ndc_xy[vi], 0.0, 1.0);
            out.depths = vec2<f32>(ndc_z[vi], ndc_z[vi]);
            out.color_offsets = vec2<f32>(off, off);
            out.density = tet_density;
            out.base_color = base_color;
            mesh_out.verts[v_base + i] = out;
        }

        // Center vertex (intersection point)
        var out_c: IntervalVertexOutput;
        out_c.position = vec4<f32>(center_xy, 0.0, 1.0);
        out_c.depths = vec2<f32>(z_front, z_back);
        out_c.color_offsets = vec2<f32>(offset_front, offset_back);
        out_c.density = tet_density;
        out_c.base_color = base_color;
        mesh_out.verts[v_base + 4u] = out_c;

        // 4 triangles forming a fan around center vertex (index 4 in local space)
        mesh_out.prims[p_base + 0u].indices = vec3<u32>(v_base + 0u, v_base + 1u, v_base + 4u);
        mesh_out.prims[p_base + 1u].indices = vec3<u32>(v_base + 1u, v_base + 2u, v_base + 4u);
        mesh_out.prims[p_base + 2u].indices = vec3<u32>(v_base + 2u, v_base + 3u, v_base + 4u);
        mesh_out.prims[p_base + 3u].indices = vec3<u32>(v_base + 3u, v_base + 0u, v_base + 4u);
    }

    } // end if my_vert_count > 0u
}
