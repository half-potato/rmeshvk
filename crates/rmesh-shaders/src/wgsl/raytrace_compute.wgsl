// Compute-based ray tracing renderer with adjacency traversal.
//
// Per-ray algorithm:
//   1. If camera is inside mesh (start_tet >= 0), begin at containing tet
//   2. Otherwise, BVH traverse to find nearest boundary face → entry tet
//   3. Walk through mesh via tet_neighbors adjacency until exit or threshold
//
// No sorting required — O(depth_complexity) per ray.

struct Uniforms {
    vp_col0: vec4<f32>,
    vp_col1: vec4<f32>,
    vp_col2: vec4<f32>,
    vp_col3: vec4<f32>,
    inv_vp_col0: vec4<f32>,
    inv_vp_col1: vec4<f32>,
    inv_vp_col2: vec4<f32>,
    inv_vp_col3: vec4<f32>,
    cam_pos_pad: vec4<f32>,
    screen_width: f32,
    screen_height: f32,
    tet_count: u32,
    sh_degree: u32,
    step: u32,
    _pad1: vec3<u32>,
};

struct BVHNode {
    aabb_min: vec3<f32>,
    left_or_face: i32,    // >= 0: left child index, < 0: -(leaf_start + 1)
    aabb_max: vec3<f32>,
    right_or_count: i32,  // internal: right child, leaf: face count
};

@group(0) @binding(0) var<storage, read> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> vertices: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> colors_buf: array<f32>;
@group(0) @binding(4) var<storage, read> densities: array<f32>;
@group(0) @binding(5) var<storage, read> color_grads_buf: array<f32>;
@group(0) @binding(6) var<storage, read> tet_neighbors: array<i32>;
@group(0) @binding(7) var<storage, read_write> rendered_image: array<f32>;
@group(0) @binding(8) var<storage, read> bvh_nodes: array<BVHNode>;
@group(0) @binding(9) var<storage, read> boundary_faces: array<u32>;
@group(0) @binding(10) var<storage, read> start_tet_buf: array<i32>;

// Face winding (inward normals) — must match forward_tiled_compute.wgsl
const FACES: array<vec3<u32>, 4> = array<vec3<u32>, 4>(
    vec3<u32>(0u, 2u, 1u),
    vec3<u32>(1u, 2u, 3u),
    vec3<u32>(0u, 3u, 2u),
    vec3<u32>(3u, 0u, 1u),
);

const LOG_T_THRESHOLD: f32 = -5.54;
const MAX_TRAVERSAL_ITERS: u32 = 512u;
const BVH_STACK_SIZE: u32 = 32u;

fn phi(x: f32) -> f32 {
    if (abs(x) < 1e-6) { return 1.0 - x * 0.5; }
    return (1.0 - exp(-x)) / x;
}

fn load_f32x3_v(idx: u32) -> vec3<f32> {
    return vec3<f32>(vertices[idx * 3u], vertices[idx * 3u + 1u], vertices[idx * 3u + 2u]);
}

// Ray-tet intersection with exit face tracking.
// Returns (c_premul.xyz, od, t_exit, exit_face).
// od < 0 means miss. exit_face = 4 means miss.
struct RayTetResult {
    c_premul: vec3<f32>,
    od: f32,
    t_exit: f32,
    exit_face: u32,
};

fn eval_tet_rt(
    tet_id: u32,
    cam: vec3<f32>,
    ray_dir: vec3<f32>,
    t_near: f32,
) -> RayTetResult {
    var result: RayTetResult;
    result.c_premul = vec3<f32>(0.0);
    result.od = -1.0;
    result.t_exit = t_near;
    result.exit_face = 4u;

    let vi0 = indices[tet_id * 4u];
    let vi1 = indices[tet_id * 4u + 1u];
    let vi2 = indices[tet_id * 4u + 2u];
    let vi3 = indices[tet_id * 4u + 3u];

    let v0 = load_f32x3_v(vi0);
    let v1 = load_f32x3_v(vi1);
    let v2 = load_f32x3_v(vi2);
    let v3 = load_f32x3_v(vi3);

    var verts = array<vec3<f32>, 4>(v0, v1, v2, v3);

    // Ray-tet intersection: find t_min (entry) and t_max (exit)
    var t_min_val: f32 = -3.402823e38;
    var t_max_val: f32 = 3.402823e38;
    var exit_face_idx = 4u;
    var valid = true;

    for (var fi = 0u; fi < 4u; fi++) {
        let f = FACES[fi];
        let va = verts[f[0]];
        let vb = verts[f[1]];
        let vc = verts[f[2]];
        let n = cross(vc - va, vb - va);

        let num = dot(n, va - cam);
        let den = dot(n, ray_dir);

        if (abs(den) < 1e-20) {
            if (num > 0.0) { valid = false; }
            continue;
        }

        let t = num / den;

        if (den > 0.0) {
            // Entering face
            if (t > t_min_val) { t_min_val = t; }
        } else {
            // Exiting face
            if (t < t_max_val) {
                t_max_val = t;
                exit_face_idx = fi;
            }
        }
    }

    // Clamp entry to t_near to avoid re-processing the entry face
    t_min_val = max(t_min_val, t_near);

    if (!valid || t_min_val >= t_max_val) {
        return result;
    }

    // Volume integral (same as eval_tet in forward_tiled_compute.wgsl)
    let density_raw = densities[tet_id];
    let colors_tet = vec3<f32>(colors_buf[tet_id * 3u], colors_buf[tet_id * 3u + 1u], colors_buf[tet_id * 3u + 2u]);
    let grad = vec3<f32>(color_grads_buf[tet_id * 3u], color_grads_buf[tet_id * 3u + 1u], color_grads_buf[tet_id * 3u + 2u]);

    let base_offset = dot(grad, cam - verts[0]);
    let base_color = colors_tet + vec3<f32>(base_offset);
    let dc_dt = dot(grad, ray_dir);

    let c_start = max(base_color + vec3<f32>(dc_dt * t_min_val), vec3<f32>(0.0));
    let c_end = max(base_color + vec3<f32>(dc_dt * t_max_val), vec3<f32>(0.0));

    let dist = t_max_val - t_min_val;
    let od = max(density_raw * dist, 1e-8);

    let alpha_t = exp(-od);
    let phi_val = phi(od);
    let w0 = phi_val - alpha_t;
    let w1 = 1.0 - phi_val;
    let c_premul = c_end * w0 + c_start * w1;

    result.c_premul = c_premul;
    result.od = od;
    result.t_exit = t_max_val;
    result.exit_face = exit_face_idx;
    return result;
}

// --- BVH traversal ---

fn ray_aabb_intersect(origin: vec3<f32>, inv_dir: vec3<f32>, aabb_min: vec3<f32>, aabb_max: vec3<f32>, t_max: f32) -> bool {
    let t1 = (aabb_min - origin) * inv_dir;
    let t2 = (aabb_max - origin) * inv_dir;
    let t_lo = min(t1, t2);
    let t_hi = max(t1, t2);
    let t_enter = max(max(t_lo.x, t_lo.y), t_lo.z);
    let t_exit = min(min(t_hi.x, t_hi.y), t_hi.z);
    return t_enter <= t_exit && t_exit >= 0.0 && t_enter < t_max;
}

fn ray_triangle_intersect(origin: vec3<f32>, dir: vec3<f32>, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> f32 {
    // Moller-Trumbore
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let h = cross(dir, e2);
    let a = dot(e1, h);
    if (abs(a) < 1e-10) { return -1.0; }
    let f = 1.0 / a;
    let s = origin - v0;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) { return -1.0; }
    let q = cross(s, e1);
    let v = f * dot(dir, q);
    if (v < 0.0 || u + v > 1.0) { return -1.0; }
    let t = f * dot(e2, q);
    if (t < 0.0) { return -1.0; }
    return t;
}

struct BVHHit {
    t_hit: f32,
    tet_id: i32,
};

fn bvh_trace(cam: vec3<f32>, ray_dir: vec3<f32>, num_bvh_nodes: u32) -> BVHHit {
    var result: BVHHit;
    result.t_hit = 3.402823e38;
    result.tet_id = -1;

    if (num_bvh_nodes == 0u) {
        return result;
    }

    let inv_dir = vec3<f32>(1.0 / ray_dir.x, 1.0 / ray_dir.y, 1.0 / ray_dir.z);

    var stack: array<u32, 32>;
    var stack_ptr = 0u;
    stack[0] = 0u;
    stack_ptr = 1u;

    while (stack_ptr > 0u) {
        stack_ptr -= 1u;
        let node_idx = stack[stack_ptr];
        let node = bvh_nodes[node_idx];

        if (!ray_aabb_intersect(cam, inv_dir, node.aabb_min, node.aabb_max, result.t_hit)) {
            continue;
        }

        if (node.left_or_face < 0) {
            // Leaf node
            let face_start = u32(-(node.left_or_face + 1));
            let face_count = u32(node.right_or_count);
            for (var i = 0u; i < face_count; i++) {
                let packed = boundary_faces[face_start + i];
                let tet_id = packed >> 2u;
                let face_idx = packed & 3u;

                let f = FACES[face_idx];
                let fv0 = load_f32x3_v(indices[tet_id * 4u + f[0]]);
                let fv1 = load_f32x3_v(indices[tet_id * 4u + f[1]]);
                let fv2 = load_f32x3_v(indices[tet_id * 4u + f[2]]);

                let t = ray_triangle_intersect(cam, ray_dir, fv0, fv1, fv2);
                if (t >= 0.0 && t < result.t_hit) {
                    result.t_hit = t;
                    result.tet_id = i32(tet_id);
                }
            }
        } else {
            // Internal node — push children
            if (stack_ptr < BVH_STACK_SIZE - 1u) {
                stack[stack_ptr] = u32(node.left_or_face);
                stack_ptr += 1u;
                stack[stack_ptr] = u32(node.right_or_count);
                stack_ptr += 1u;
            }
        }
    }

    return result;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let px = global_id.x;
    let py = global_id.y;
    let w = u32(uniforms.screen_width);
    let h = u32(uniforms.screen_height);

    if (px >= w || py >= h) {
        return;
    }

    // Compute ray from pixel coordinates via inverse VP
    let ndc_x = (2.0 * (f32(px) + 0.5) / f32(w)) - 1.0;
    let ndc_y = 1.0 - (2.0 * (f32(py) + 0.5) / f32(h));

    let inv_vp = mat4x4<f32>(uniforms.inv_vp_col0, uniforms.inv_vp_col1, uniforms.inv_vp_col2, uniforms.inv_vp_col3);
    let near_clip = inv_vp * vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    let far_clip = inv_vp * vec4<f32>(ndc_x, ndc_y, 1.0, 1.0);
    let near_world = near_clip.xyz / near_clip.w;
    let far_world = far_clip.xyz / far_clip.w;

    let cam = uniforms.cam_pos_pad.xyz;
    let ray_dir = normalize(far_world - near_world);

    // Determine start tet and t_cursor
    let start_tet_id = start_tet_buf[0];
    var current_tet: i32;
    var t_cursor: f32;

    if (start_tet_id >= 0) {
        // Camera inside mesh
        current_tet = start_tet_id;
        t_cursor = 0.0;
    } else {
        // Camera outside mesh — BVH trace to find entry
        let num_nodes = arrayLength(&bvh_nodes);
        let hit = bvh_trace(cam, ray_dir, num_nodes);
        if (hit.tet_id < 0) {
            // No hit — write background (transparent black)
            let pixel_idx = py * w + px;
            rendered_image[pixel_idx * 4u] = 0.0;
            rendered_image[pixel_idx * 4u + 1u] = 0.0;
            rendered_image[pixel_idx * 4u + 2u] = 0.0;
            rendered_image[pixel_idx * 4u + 3u] = 0.0;
            return;
        }
        current_tet = hit.tet_id;
        t_cursor = hit.t_hit;
    }

    // Adjacency traversal loop
    var color_accum = vec3<f32>(0.0);
    var log_t: f32 = 0.0;
    var iter = 0u;

    while (current_tet >= 0 && log_t > LOG_T_THRESHOLD && iter < MAX_TRAVERSAL_ITERS) {
        let result = eval_tet_rt(u32(current_tet), cam, ray_dir, t_cursor);

        if (result.od > 0.0) {
            let T_j = exp(log_t);
            color_accum += result.c_premul * T_j;
            log_t -= result.od;
        }

        if (result.exit_face >= 4u) {
            // Miss or degenerate — stop
            break;
        }

        // Move to neighbor through exit face
        let neighbor = tet_neighbors[u32(current_tet) * 4u + result.exit_face];
        current_tet = neighbor;
        t_cursor = result.t_exit;
        iter += 1u;
    }

    // Write output
    let pixel_idx = py * w + px;
    let T_final = exp(log_t);
    rendered_image[pixel_idx * 4u] = color_accum.x;
    rendered_image[pixel_idx * 4u + 1u] = color_accum.y;
    rendered_image[pixel_idx * 4u + 2u] = color_accum.z;
    rendered_image[pixel_idx * 4u + 3u] = 1.0 - T_final;
}
