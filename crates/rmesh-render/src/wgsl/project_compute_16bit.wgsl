// Project compute shader with 16-bit linear sort keys.
//
// Identical to project_compute.wgsl except for depth key generation:
// uses linear quantization into [0, far_plane] for 16-bit keys (2 DRS passes
// instead of 4), giving ~2x sort speedup with negligible visual difference.

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
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0) var<storage, read> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> vertices: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> densities: array<f32>;
@group(0) @binding(4) var<storage, read> color_grads: array<f32>;
@group(0) @binding(5) var<storage, read> circumdata: array<f32>;
@group(0) @binding(6) var<storage, read_write> colors: array<f32>;
@group(0) @binding(7) var<storage, read_write> sort_keys: array<u32>;
@group(0) @binding(8) var<storage, read_write> sort_values: array<u32>;
@group(0) @binding(9) var<storage, read_write> indirect_args: DrawIndirectArgs;
// Compact outputs for tiled path (indexed by vis_idx from atomicAdd)
@group(0) @binding(10) var<storage, read_write> tiles_touched: array<u32>;
@group(0) @binding(11) var<storage, read_write> compact_tet_ids: array<u32>;
@group(0) @binding(12) var<storage, read> base_colors_buf: array<f32>;
@group(0) @binding(13) var<storage, read> sh_coeffs: array<u32>;

// --- SH constants ---
const C0: f32 = 0.28209479177387814;
const C1: f32 = 0.4886025119029199;
const C2_0: f32 = 1.0925484305920792;
const C2_1: f32 = -1.0925484305920792;
const C2_2: f32 = 0.31539156525252005;
const C2_3: f32 = -1.0925484305920792;
const C2_4: f32 = 0.5462742152960396;
const C3_0: f32 = -0.5900435899266435;
const C3_1: f32 = 2.890611442640554;
const C3_2: f32 = -0.4570457994644658;
const C3_3: f32 = 0.3731763325901154;
const C3_4: f32 = -0.4570457994644658;
const C3_5: f32 = 1.445305721320277;
const C3_6: f32 = -0.5900435899266435;

fn softplus(x: f32) -> f32 {
    if (x * 10.0 > 20.0) { return x; }
    return 0.1 * log(1.0 + exp(10.0 * x));
}

fn read_sh_f16(base: u32, i: u32) -> f32 {
    let word = sh_coeffs[base + i / 2u];
    let unpacked = unpack2x16float(word);
    return select(unpacked.x, unpacked.y, (i & 1u) != 0u);
}

fn project_to_ndc(pos: vec3<f32>, vp: mat4x4<f32>) -> vec4<f32> {
    let clip = vp * vec4<f32>(pos, 1.0);
    let inv_w = 1.0 / (clip.w + 1e-6);
    return vec4<f32>(clip.xyz * inv_w, clip.w);
}

fn load_vertex(idx: u32) -> vec3<f32> {
    let i = idx * 3u;
    return vec3<f32>(vertices[i], vertices[i + 1u], vertices[i + 2u]);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let tet_id = global_id.x + global_id.y * num_workgroups.x * 64u;
    // Padding threads (beyond tet_count): initialize sort buffers.
    // Use 0xFFFF (16-bit max) so DRS with sorting_bits=16 sorts these to end.
    if (tet_id >= uniforms.tet_count) {
        if (tet_id < arrayLength(&sort_keys)) {
            sort_keys[tet_id] = 0xFFFFu;
            sort_values[tet_id] = tet_id;
        }
        return;
    }

    // --- 1. Load geometry ---
    let i0 = indices[tet_id * 4u];
    let i1 = indices[tet_id * 4u + 1u];
    let i2 = indices[tet_id * 4u + 2u];
    let i3 = indices[tet_id * 4u + 3u];

    let v0 = load_vertex(i0);
    let v1 = load_vertex(i1);
    let v2 = load_vertex(i2);
    let v3 = load_vertex(i3);

    // --- 2. Frustum culling ---
    let vp = mat4x4<f32>(uniforms.vp_col0, uniforms.vp_col1, uniforms.vp_col2, uniforms.vp_col3);
    let p0 = project_to_ndc(v0, vp);
    let p1 = project_to_ndc(v1, vp);
    let p2 = project_to_ndc(v2, vp);
    let p3 = project_to_ndc(v3, vp);

    // If any vertex is behind the camera (clip.w <= 0), NDC coords are unreliable.
    // Mark as visible and let the hardware rasterizer handle near-plane clipping.
    let any_behind = (p0.w <= 0.0) || (p1.w <= 0.0) || (p2.w <= 0.0) || (p3.w <= 0.0);

    var visible = true;
    if (!any_behind) {
        let min_x = min(min(p0.x, p1.x), min(p2.x, p3.x));
        let max_x = max(max(p0.x, p1.x), max(p2.x, p3.x));
        let min_y = min(min(p0.y, p1.y), min(p2.y, p3.y));
        let max_y = max(max(p0.y, p1.y), max(p2.y, p3.y));
        let min_z = min(min(p0.z, p1.z), min(p2.z, p3.z));

        if (max_x < -1.0 || min_x > 1.0 || max_y < -1.0 || min_y > 1.0 || min_z > 1.0) {
            visible = false;
        }
        let ext_x = (max_x - min_x) * uniforms.screen_width;
        let ext_y = (max_y - min_y) * uniforms.screen_height;
        if (ext_x * ext_y < 1.0) {
            visible = false;
        }
    }

    if (!visible) {
        sort_keys[tet_id] = 0xFFFFu;
        sort_values[tet_id] = tet_id;
        return;
    }

    // --- 2b. Max optical depth cull ---
    // Maximum ray path through tet <= longest edge. If density * longest_edge
    // is below threshold, no ray can produce meaningful alpha — skip entirely.
    {
        let e01 = v1 - v0; let e02 = v2 - v0; let e03 = v3 - v0;
        let e12 = v2 - v1; let e13 = v3 - v1; let e23 = v3 - v2;
        let max_edge_sq = max(
            max(dot(e01, e01), max(dot(e02, e02), dot(e03, e03))),
            max(dot(e12, e12), max(dot(e13, e13), dot(e23, e23)))
        );
        let max_od = densities[tet_id] * sqrt(max_edge_sq);
        if (max_od < 0.01) {
            sort_keys[tet_id] = 0xFFFFu;
            sort_values[tet_id] = tet_id;
            return;
        }
    }

    // --- 3. Linear 16-bit depth key from circumsphere distance ---
    let cx = circumdata[tet_id * 4u];
    let cy = circumdata[tet_id * 4u + 1u];
    let cz = circumdata[tet_id * 4u + 2u];
    let r2 = circumdata[tet_id * 4u + 3u];

    let cam = uniforms.cam_pos_pad.xyz;
    let diff = vec3<f32>(cx, cy, cz) - cam;
    let depth_raw = dot(diff, diff) - r2;
    // Linear distance from squared circumsphere distance.
    // Negative values (camera inside circumsphere) clamp to 0 = closest = sorted last (back-to-front).
    let dist = sqrt(max(depth_raw, 0.0));
    let t = clamp(dist / uniforms.far_plane, 0.0, 1.0);
    let key16 = u32(t * 65535.0);
    sort_keys[tet_id] = 65535u - key16; // Invert for back-to-front
    sort_values[tet_id] = tet_id;

    // Atomic increment instance count, get compact vis_idx
    let vis_idx = atomicAdd(&indirect_args.instance_count, 1u);

    // Write compact tet id list for tiled path
    compact_tet_ids[vis_idx] = tet_id;

    // Compute tiles_touched for prefix scan (conservative scanline)
    let W = uniforms.screen_width;
    let H = uniforms.screen_height;
    let tile_size = f32(uniforms.tile_size_u);

    if (any_behind) {
        // Cull entirely — behind-camera tets would overflow pair buffers
        tiles_touched[vis_idx] = 0u;
    } else {
        // NDC to pixel coords for all 4 vertices
        var proj: array<vec2<f32>, 4>;
        proj[0] = vec2<f32>((p0.x + 1.0) * 0.5 * W, (1.0 - p0.y) * 0.5 * H);
        proj[1] = vec2<f32>((p1.x + 1.0) * 0.5 * W, (1.0 - p1.y) * 0.5 * H);
        proj[2] = vec2<f32>((p2.x + 1.0) * 0.5 * W, (1.0 - p2.y) * 0.5 * H);
        proj[3] = vec2<f32>((p3.x + 1.0) * 0.5 * W, (1.0 - p3.y) * 0.5 * H);

        // AABB for tile bounds
        let pix_min_x = max(min(min(proj[0].x, proj[1].x), min(proj[2].x, proj[3].x)), 0.0);
        let pix_max_x = min(max(max(proj[0].x, proj[1].x), max(proj[2].x, proj[3].x)), W - 1.0);
        let pix_min_y = max(min(min(proj[0].y, proj[1].y), min(proj[2].y, proj[3].y)), 0.0);
        let pix_max_y = min(max(max(proj[0].y, proj[1].y), max(proj[2].y, proj[3].y)), H - 1.0);

        if (pix_min_x > pix_max_x || pix_min_y > pix_max_y) {
            tiles_touched[vis_idx] = 0u;
        } else {
            let T = tile_size;
            let tiles_x_total = u32(ceil(W / T));
            let tile_min_x = u32(pix_min_x / T);
            let tile_max_x = min(u32(pix_max_x / T), tiles_x_total - 1u);
            let tile_min_y = u32(pix_min_y / T);
            let tile_max_y = min(u32(pix_max_y / T), u32(ceil(H / T)) - 1u);

            // Conservative scanline tile counting (Variant B)
            var tile_count = 0u;
            for (var ty = tile_min_y; ty <= tile_max_y; ty++) {
                var xl = 1e10f;
                var xr = -1e10f;

                // Edge intersections at both tile-row boundaries
                let ei_arr = array<u32, 6>(0u, 0u, 0u, 1u, 1u, 2u);
                let ej_arr = array<u32, 6>(1u, 2u, 3u, 2u, 3u, 3u);
                for (var e = 0u; e < 6u; e++) {
                    let ei = ei_arr[e];
                    let ej = ej_arr[e];
                    let yi = proj[ei].y;
                    let yj = proj[ej].y;
                    let xi = proj[ei].x;
                    let xj = proj[ej].x;
                    for (var b = 0u; b < 2u; b++) {
                        let ytest = f32(ty + b) * T;
                        if ((yi <= ytest && yj > ytest) || (yj <= ytest && yi > ytest)) {
                            let t_edge = (ytest - yi) / (yj - yi);
                            let x = xi + t_edge * (xj - xi);
                            xl = min(xl, x);
                            xr = max(xr, x);
                        }
                    }
                }

                // Vertices within tile-row band
                for (var v = 0u; v < 4u; v++) {
                    let vy = proj[v].y;
                    if (vy >= f32(ty) * T && vy <= f32(ty + 1u) * T) {
                        xl = min(xl, proj[v].x);
                        xr = max(xr, proj[v].x);
                    }
                }

                if (xl <= xr) {
                    let tx0 = clamp(u32(max(floor(xl / T), 0.0)), tile_min_x, tile_max_x);
                    let tx1 = clamp(u32(max(floor(xr / T), 0.0)), tile_min_x, tile_max_x);
                    tile_count += tx1 - tx0 + 1u;
                }
            }
            tiles_touched[vis_idx] = tile_count;
        }
    }

    // --- 4. Color evaluation ---
    if (uniforms.sh_degree > 0u) {
        // Inline SH evaluation (only for visible tets, reuses already-loaded vertices)
        let centroid = (v0 + v1 + v2 + v3) * 0.25;
        let cam = uniforms.cam_pos_pad.xyz;
        let raw_dir = centroid - cam;
        let len = length(raw_dir);
        var dir = vec3<f32>(0.0);
        if (len > 0.0) { dir = raw_dir / len; }
        let x = dir.x; let y = dir.y; let z = dir.z;

        let nc = (uniforms.sh_degree + 1u) * (uniforms.sh_degree + 1u);
        let stride = nc * 3u;
        let sh_base = tet_id * ((stride + 1u) / 2u);

        var rgb = vec3<f32>(0.0);
        for (var c = 0u; c < 3u; c++) {
            let ch_offset = c * nc;
            var val = C0 * read_sh_f16(sh_base, ch_offset);

            if (uniforms.sh_degree >= 1u) {
                val -= C1 * y * read_sh_f16(sh_base, ch_offset + 1u);
                val += C1 * z * read_sh_f16(sh_base, ch_offset + 2u);
                val -= C1 * x * read_sh_f16(sh_base, ch_offset + 3u);
            }
            if (uniforms.sh_degree >= 2u) {
                let xx = x * x; let yy = y * y; let zz = z * z;
                let xy = x * y; let yz = y * z; let xz = x * z;
                val += C2_0 * xy * read_sh_f16(sh_base, ch_offset + 4u);
                val += C2_1 * yz * read_sh_f16(sh_base, ch_offset + 5u);
                val += C2_2 * (2.0 * zz - xx - yy) * read_sh_f16(sh_base, ch_offset + 6u);
                val += C2_3 * xz * read_sh_f16(sh_base, ch_offset + 7u);
                val += C2_4 * (xx - yy) * read_sh_f16(sh_base, ch_offset + 8u);
            }
            if (uniforms.sh_degree >= 3u) {
                let xx = x * x; let yy = y * y; let zz = z * z;
                val += C3_0 * y * (3.0 * xx - yy) * read_sh_f16(sh_base, ch_offset + 9u);
                val += C3_1 * x * y * z * read_sh_f16(sh_base, ch_offset + 10u);
                val += C3_2 * y * (4.0 * zz - xx - yy) * read_sh_f16(sh_base, ch_offset + 11u);
                val += C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * read_sh_f16(sh_base, ch_offset + 12u);
                val += C3_4 * x * (4.0 * zz - xx - yy) * read_sh_f16(sh_base, ch_offset + 13u);
                val += C3_5 * z * (xx - yy) * read_sh_f16(sh_base, ch_offset + 14u);
                val += C3_6 * x * (xx - 3.0 * yy) * read_sh_f16(sh_base, ch_offset + 15u);
            }
            rgb[c] = val + 0.5;
        }

        let grad = vec3<f32>(color_grads[tet_id * 3u], color_grads[tet_id * 3u + 1u], color_grads[tet_id * 3u + 2u]);
        let offset = dot(grad, v0 - centroid);
        rgb += vec3<f32>(offset);

        colors[tet_id * 3u]     = softplus(rgb.x);
        colors[tet_id * 3u + 1u] = softplus(rgb.y);
        colors[tet_id * 3u + 2u] = softplus(rgb.z);
    } else {
        // Training path: copy base_colors
        colors[tet_id * 3u] = base_colors_buf[tet_id * 3u];
        colors[tet_id * 3u + 1u] = base_colors_buf[tet_id * 3u + 1u];
        colors[tet_id * 3u + 2u] = base_colors_buf[tet_id * 3u + 2u];
    }
}
