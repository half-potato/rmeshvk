// Compute-based interval shading: generates screen-space triangles per tet.
//
// Port of interval_mesh.wgsl from mesh shader to compute shader.
// 1 thread per tet, writes 5 vertices (2 vec4 each) + 1 per-tet vec4 to fixed-slot buffers.
// Non-indexed draw: fan pattern computed in vertex shader.

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
@group(0) @binding(8) var<storage, read_write> out_vertices: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> out_tet_data: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read> vertex_normals: array<f32>; // [V * 3]

fn load_f32x3(buf_base: u32) -> vec3<f32> {
    return vec3<f32>(vertices[buf_base], vertices[buf_base + 1u], vertices[buf_base + 2u]);
}

fn load_color(idx: u32) -> vec3<f32> {
    return vec3<f32>(colors[idx * 3u], colors[idx * 3u + 1u], colors[idx * 3u + 2u]);
}

fn load_grad(idx: u32) -> vec3<f32> {
    return vec3<f32>(color_grads[idx * 3u], color_grads[idx * 3u + 1u], color_grads[idx * 3u + 2u]);
}

fn load_vnormal(vi: u32) -> vec3<f32> {
    return vec3<f32>(vertex_normals[vi * 3u], vertex_normals[vi * 3u + 1u], vertex_normals[vi * 3u + 2u]);
}

// Barycentric interpolation of field gradients on a face.
fn interp_gradient_3(n: array<vec3f, 4>, face: vec4u, bary: vec3f) -> vec3f {
    return bary.x * n[face[0]] + bary.y * n[face[1]] + bary.z * n[face[2]];
}

// Maps vertex index -> TET_FACES index where that vertex is face[3] (opposite)
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
    for (var v = 0u; v < 4u; v++) {
        let face = TET_FACES[v];
        let a = p[face[0]];
        let b = p[face[1]];
        let c = p[face[2]];
        let pt = p[face[3]];

        let d0 = (b.x - a.x) * (pt.y - a.y) - (b.y - a.y) * (pt.x - a.x);
        let d1 = (c.x - b.x) * (pt.y - b.y) - (c.y - b.y) * (pt.x - b.x);
        let d2 = (a.x - c.x) * (pt.y - c.y) - (a.y - c.y) * (pt.x - c.x);

        let all_pos = d0 >= 0.0 && d1 >= 0.0 && d2 >= 0.0;
        let all_neg = d0 <= 0.0 && d1 <= 0.0 && d2 <= 0.0;

        if all_pos || all_neg {
            return face[3];
        }
    }
    return 4u;
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
        return -1.0;
    }
    let d = b1 - a1;
    return (d.x * d2.y - d.y * d2.x) / denom;
}

// Find which pair of opposite edges cross in screen space (Case 2).
fn find_crossing_edges(p: array<vec2<f32>, 4>) -> vec2<u32> {
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

        let ta = line_intersect_t(a1, a2, b1, b2);
        let tb = line_intersect_t(b1, b2, a1, a2);

        if ta > 0.0 && ta < 1.0 && tb > 0.0 && tb < 1.0 {
            return pairs[i];
        }
    }
    return pairs[0];
}

// Write a vertex as 4 packed vec4s at the given slot.
// Layout: [i*4+0] = (ndc_xy, z_front, z_back)
//         [i*4+1] = (off_front, off_back, 0, 0)
//         [i*4+2] = (gradient.xyz, 0) — raw field gradient (n_front)
//         [i*4+3] = (gradient.xyz, 0) — n_back (same as n_front here)
fn write_vertex(slot: u32, ndc_xy: vec2<f32>, z_front: f32, z_back: f32,
                off_front: f32, off_back: f32, gradient: vec3f) {
    out_vertices[slot * 4u + 0u] = vec4<f32>(ndc_xy, z_front, z_back);
    out_vertices[slot * 4u + 1u] = vec4<f32>(off_front, off_back, 0.0, 0.0);
    out_vertices[slot * 4u + 2u] = vec4<f32>(gradient, 0.0);
    out_vertices[slot * 4u + 3u] = vec4<f32>(gradient, 0.0);
}

fn write_degenerate_vertex(slot: u32) {
    out_vertices[slot * 4u + 0u] = vec4<f32>(0.0);
    out_vertices[slot * 4u + 1u] = vec4<f32>(0.0);
    out_vertices[slot * 4u + 2u] = vec4<f32>(0.0);
    out_vertices[slot * 4u + 3u] = vec4<f32>(0.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    let visible_count = indirect_args.instance_count;

    // Base offset for this tet's fixed slots
    let v_base = tid * 5u;  // 5 vertices per tet

    if tid >= visible_count {
        return;
    }

    let tet_id = sorted_indices[tid];

    // Load 4 vertex positions
    let i0 = indices[tet_id * 4u];
    let i1 = indices[tet_id * 4u + 1u];
    let i2 = indices[tet_id * 4u + 2u];
    let i3 = indices[tet_id * 4u + 3u];

    var v_world: array<vec3<f32>, 4>;
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

    // Skip tets with any vertex behind the camera
    let any_behind = clip[0].w <= 0.0 || clip[1].w <= 0.0
                  || clip[2].w <= 0.0 || clip[3].w <= 0.0;

    if any_behind {
        for (var i = 0u; i < 5u; i++) { write_degenerate_vertex(v_base + i); }
        out_tet_data[tid * 2u] = vec4<f32>(0.0);
        out_tet_data[tid * 2u + 1u] = vec4<f32>(0.0);
        return;
    }

    // Perspective divide -> NDC
    var ndc_xy: array<vec2<f32>, 4>;
    var ndc_z: array<f32, 4>;
    for (var vi = 0u; vi < 4u; vi++) {
        let w = clip[vi].w;
        let inv_w = select(0.0, 1.0 / w, abs(w) > 1e-10);
        ndc_xy[vi] = clip[vi].xy * inv_w;
        ndc_z[vi] = clip[vi].z * inv_w;
    }

    // Material data
    let tet_density = densities[tet_id];
    let color = load_color(tet_id);
    let grad = load_grad(tet_id);

    let cam_pos = uniforms.cam_pos_pad.xyz;
    let base_color = color + vec3<f32>(dot(grad, cam_pos - v_world[0]));

    // Load per-vertex normals for this tet
    var vn: array<vec3f, 4>;
    vn[0] = load_vnormal(i0);
    vn[1] = load_vnormal(i1);
    vn[2] = load_vnormal(i2);
    vn[3] = load_vnormal(i3);

    // Write per-tet flat data: slot 0 = (base_color.rgb, density), slot 1 = (tet_id, 0, 0, 0)
    out_tet_data[tid * 2u] = vec4<f32>(base_color, tet_density);
    out_tet_data[tid * 2u + 1u] = vec4<f32>(bitcast<f32>(tet_id), 0.0, 0.0, 0.0);

    // Skip degenerate tets
    let z_range = max(max(ndc_z[0], ndc_z[1]), max(ndc_z[2], ndc_z[3]))
                - min(min(ndc_z[0], ndc_z[1]), min(ndc_z[2], ndc_z[3]));
    if z_range < 1e-7 || tet_density < 1e-6 {
        for (var i = 0u; i < 5u; i++) { write_degenerate_vertex(v_base + i); }
        return;
    }

    // Classify silhouette
    let interior_vtx = classify_silhouette(ndc_xy);

    if interior_vtx < 4u {
        // Case 1: vertex interior_vtx projects inside opposite face -> 3 silhouette verts + center
        let face = TET_FACES[OPPOSITE_FACE[interior_vtx]];
        let si0 = face[0];
        let si1 = face[1];
        let si2 = face[2];
        let center = face[3]; // = interior_vtx

        let bary = bary2d(ndc_xy[center], ndc_xy[si0], ndc_xy[si1], ndc_xy[si2]);

        let z_face = bary.x * ndc_z[si0] + bary.y * ndc_z[si1] + bary.z * ndc_z[si2];
        let z_center = ndc_z[center];
        let z_front = min(z_face, z_center);
        let z_back = max(z_face, z_center);

        var sv_offset: array<f32, 3>;
        let sv_idx = array<u32, 3>(si0, si1, si2);
        for (var i = 0u; i < 3u; i++) {
            sv_offset[i] = dot(grad, v_world[sv_idx[i]] - cam_pos);
        }
        let center_offset = dot(grad, v_world[center] - cam_pos);
        let face_offset = bary.x * sv_offset[0] + bary.y * sv_offset[1] + bary.z * sv_offset[2];

        var offset_front: f32;
        var offset_back: f32;
        if z_face <= z_center {
            offset_front = face_offset;
            offset_back = center_offset;
        } else {
            offset_front = center_offset;
            offset_back = face_offset;
        }

        // Compute gradient at center's front intersection point.
        // Front is on the face → interpolate with barycentrics. Back is the interior vertex.
        var grad_center: vec3f;
        if z_face <= z_center {
            grad_center = interp_gradient_3(vn, face, bary);
        } else {
            grad_center = vn[center];
        }

        // Slots 0-2: silhouette vertices (on tet boundary, raw vertex gradient)
        for (var i = 0u; i < 3u; i++) {
            write_vertex(v_base + u32(i), ndc_xy[sv_idx[i]],
                         ndc_z[sv_idx[i]], ndc_z[sv_idx[i]],
                         sv_offset[i], sv_offset[i],
                         vn[sv_idx[i]]);
        }
        // Slot 3: copy of vert 0 (wraps fan, makes tri 3 degenerate)
        write_vertex(v_base + 3u, ndc_xy[sv_idx[0]],
                     ndc_z[sv_idx[0]], ndc_z[sv_idx[0]],
                     sv_offset[0], sv_offset[0],
                     vn[sv_idx[0]]);
        // Slot 4: center vertex (gradient at entry point)
        write_vertex(v_base + 4u, ndc_xy[center], z_front, z_back,
                     offset_front, offset_back, grad_center);

    } else {
        // Case 2: two opposite edges cross -> 4 silhouette verts + center
        let crossing_edges = find_crossing_edges(ndc_xy);
        let ea_idx = crossing_edges.x;
        let eb_idx = crossing_edges.y;
        let ea = TET_EDGES[ea_idx];
        let eb = TET_EDGES[eb_idx];

        let ta = line_intersect_t(ndc_xy[ea.x], ndc_xy[ea.y], ndc_xy[eb.x], ndc_xy[eb.y]);
        let tb = line_intersect_t(ndc_xy[eb.x], ndc_xy[eb.y], ndc_xy[ea.x], ndc_xy[ea.y]);

        let center_xy = mix(ndc_xy[ea.x], ndc_xy[ea.y], ta);

        let z_a = mix(ndc_z[ea.x], ndc_z[ea.y], ta);
        let z_b = mix(ndc_z[eb.x], ndc_z[eb.y], tb);
        let z_front = min(z_a, z_b);
        let z_back = max(z_a, z_b);

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

        // Slots 0-3: 4 silhouette vertices (raw vertex gradients)
        let sv = array<u32, 4>(ea.x, eb.x, ea.y, eb.y);
        let sv_off = array<f32, 4>(off_ea_x, off_eb_x, off_ea_y, off_eb_y);
        for (var i = 0u; i < 4u; i++) {
            write_vertex(v_base + u32(i), ndc_xy[sv[i]],
                         ndc_z[sv[i]], ndc_z[sv[i]],
                         sv_off[i], sv_off[i],
                         vn[sv[i]]);
        }

        // Center: interpolate gradients along edges, use front intersection
        let g_a = mix(vn[ea.x], vn[ea.y], ta);
        let g_b = mix(vn[eb.x], vn[eb.y], tb);
        var grad_center: vec3f;
        if z_a <= z_b {
            grad_center = g_a;
        } else {
            grad_center = g_b;
        }

        // Slot 4: center vertex (intersection point)
        write_vertex(v_base + 4u, center_xy, z_front, z_back,
                     offset_front, offset_back, grad_center);
    }
}
