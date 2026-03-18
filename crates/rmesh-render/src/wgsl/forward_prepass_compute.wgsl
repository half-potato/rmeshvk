// Compute prepass for quad renderer: 1 thread per visible tet.
//
// Precomputes convex hull order, face normals, plane numerators, and material
// into a flat buffer. Stores WORLD-SPACE positions so the vertex shader can
// compute interpolated ray_dir (matching the face-based path).
//
// Output layout per tet: 10 × vec4<f32> = 160 bytes
//   [0..3] = world-space strip positions (convex hull order), w unused
//   [4..7] = (normal.xyz, plane_numerator) per face
//   [8]    = (base_color.rgb, density)
//   [9]    = (color_grad.rgb, unused)

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

@group(0) @binding(0) var<storage, read> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> vertices: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(4) var<storage, read> indirect_args: DrawIndirectArgs;
@group(0) @binding(5) var<storage, read_write> precomputed: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> colors: array<f32>;
@group(0) @binding(7) var<storage, read> densities: array<f32>;
@group(0) @binding(8) var<storage, read> color_grads: array<f32>;

const TET_FACES: array<vec4<u32>, 4> = array<vec4<u32>, 4>(
    vec4<u32>(0u, 2u, 1u, 3u),
    vec4<u32>(1u, 2u, 3u, 0u),
    vec4<u32>(0u, 3u, 2u, 1u),
    vec4<u32>(3u, 0u, 1u, 2u),
);

fn load_vertex(idx: u32) -> vec3<f32> {
    let i = idx * 3u;
    return vec3<f32>(vertices[i], vertices[i + 1u], vertices[i + 2u]);
}

fn load_color(idx: u32) -> vec3<f32> {
    return vec3<f32>(colors[idx * 3u], colors[idx * 3u + 1u], colors[idx * 3u + 2u]);
}

fn load_grad(idx: u32) -> vec3<f32> {
    return vec3<f32>(color_grads[idx * 3u], color_grads[idx * 3u + 1u], color_grads[idx * 3u + 2u]);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let vis_idx = global_id.x + global_id.y * num_workgroups.x * 64u;

    if (vis_idx >= indirect_args.instance_count) {
        return;
    }

    let tet_id = sorted_indices[vis_idx];

    // Load 4 vertex positions
    let i0 = indices[tet_id * 4u];
    let i1 = indices[tet_id * 4u + 1u];
    let i2 = indices[tet_id * 4u + 2u];
    let i3 = indices[tet_id * 4u + 3u];

    var verts: array<vec3<f32>, 4>;
    verts[0] = load_vertex(i0);
    verts[1] = load_vertex(i1);
    verts[2] = load_vertex(i2);
    verts[3] = load_vertex(i3);

    let cam = uniforms.cam_pos_pad.xyz;

    // Compute inward-pointing face normals and plane numerators
    var face_normals: array<vec3<f32>, 4>;
    var numerators: array<f32, 4>;

    for (var i = 0u; i < 4u; i++) {
        let f = TET_FACES[i];
        let va = verts[f[0]];
        let vb = verts[f[1]];
        let vc = verts[f[2]];
        var n = cross(vc - va, vb - va);
        let v_opp = verts[f[3]];
        if (dot(n, v_opp - va) < 0.0) {
            n = -n;
        }
        face_normals[i] = n;
        numerators[i] = dot(n, va - cam);
    }

    // Project all 4 verts to clip space (for hull computation only)
    let vp = mat4x4<f32>(uniforms.vp_col0, uniforms.vp_col1, uniforms.vp_col2, uniforms.vp_col3);
    var clips: array<vec4<f32>, 4>;
    clips[0] = vp * vec4<f32>(verts[0], 1.0);
    clips[1] = vp * vec4<f32>(verts[1], 1.0);
    clips[2] = vp * vec4<f32>(verts[2], 1.0);
    clips[3] = vp * vec4<f32>(verts[3], 1.0);

    let base = vis_idx * 10u;

    // If any vertex is behind the camera, skip this tet entirely (degenerate quad).
    var any_behind = false;
    for (var i = 0u; i < 4u; i++) {
        if (clips[i].w <= 0.0) {
            any_behind = true;
        }
    }
    if (any_behind) {
        precomputed[base + 0u] = vec4<f32>(0.0);
        precomputed[base + 1u] = vec4<f32>(0.0);
        precomputed[base + 2u] = vec4<f32>(0.0);
        precomputed[base + 3u] = vec4<f32>(0.0);
        precomputed[base + 4u] = vec4<f32>(0.0);
        precomputed[base + 5u] = vec4<f32>(0.0);
        precomputed[base + 6u] = vec4<f32>(0.0);
        precomputed[base + 7u] = vec4<f32>(0.0);
        precomputed[base + 8u] = vec4<f32>(0.0);
        precomputed[base + 9u] = vec4<f32>(0.0);
        return;
    }

    // All vertices in front of camera — compute tight convex hull in NDC
    var ndc: array<vec2<f32>, 4>;
    for (var i = 0u; i < 4u; i++) {
        ndc[i] = clips[i].xy / clips[i].w;
    }

    // Sort indices by angle from centroid → CCW convex hull order
    let centroid = (ndc[0] + ndc[1] + ndc[2] + ndc[3]) * 0.25;
    var angles: array<f32, 4>;
    for (var i = 0u; i < 4u; i++) {
        let d = ndc[i] - centroid;
        angles[i] = atan2(d.y, d.x);
    }

    // Sorting network for 4 elements (5 comparisons)
    var idx = array<u32, 4>(0u, 1u, 2u, 3u);
    if (angles[idx[0]] > angles[idx[1]]) { let t = idx[0]; idx[0] = idx[1]; idx[1] = t; }
    if (angles[idx[2]] > angles[idx[3]]) { let t = idx[2]; idx[2] = idx[3]; idx[3] = t; }
    if (angles[idx[0]] > angles[idx[2]]) { let t = idx[0]; idx[0] = idx[2]; idx[2] = t; }
    if (angles[idx[1]] > angles[idx[3]]) { let t = idx[1]; idx[1] = idx[3]; idx[3] = t; }
    if (angles[idx[1]] > angles[idx[2]]) { let t = idx[1]; idx[1] = idx[2]; idx[2] = t; }

    // Triangle strip order from CCW hull: [0, 1, 3, 2]
    // Write WORLD positions in strip order
    let strip_order = array<u32, 4>(0u, 1u, 3u, 2u);
    for (var i = 0u; i < 4u; i++) {
        let hull_i = idx[strip_order[i]];
        precomputed[base + i] = vec4<f32>(verts[hull_i], 0.0);
    }

    // Face normals with numerators packed in .w
    precomputed[base + 4u] = vec4<f32>(face_normals[0], numerators[0]);
    precomputed[base + 5u] = vec4<f32>(face_normals[1], numerators[1]);
    precomputed[base + 6u] = vec4<f32>(face_normals[2], numerators[2]);
    precomputed[base + 7u] = vec4<f32>(face_normals[3], numerators[3]);

    // Material: base_color with gradient offset + density
    let color = load_color(tet_id);
    let grad = load_grad(tet_id);
    let offset = dot(grad, cam - verts[0]);
    let base_color = color + vec3<f32>(offset);
    precomputed[base + 8u] = vec4<f32>(base_color, densities[tet_id]);
    precomputed[base + 9u] = vec4<f32>(grad, 0.0);
}
