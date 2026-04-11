// Interval shading fragment shader.
//
// Linearizes interpolated NDC depths to view-space Z, computes ray segment
// distance, and evaluates the volume rendering integral.
//
// MRT outputs (all Rgba16Float, .a = alpha for correct hardware blend):
//   location(0): plaster.rgb * a, a
//   location(1): roughness * a, env_f0 * a, env_f1 * a, a
//   location(2): oct_normal.xy * a, pack(env_f2,env_f3) * a, a
//   location(3): albedo.rgb * a, a

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

@group(0) @binding(0) var<storage, read> uniforms: Uniforms;
@group(0) @binding(3) var<storage, read> aux_data: array<f32>;        // [M * AUX_DIM]
@group(0) @binding(4) var<storage, read> vertex_normals: array<f32>;  // [V * 3]
@group(0) @binding(5) var<storage, read> tet_indices: array<u32>;     // [M * 4]

const AUX_DIM: u32 = 8u;

struct FragmentInput {
    @location(0) depths: vec2<f32>,                  // (z_front_ndc, z_back_ndc)
    @location(1) color_offsets: vec2<f32>,            // (offset_front, offset_back)
    @location(2) @interpolate(flat) density: f32,
    @location(3) @interpolate(flat) base_color: vec3<f32>,
    @location(4) @interpolate(flat) tet_id: u32,
};

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) aux0: vec4<f32>,
    @location(2) normals: vec4<f32>,
    @location(3) albedo: vec4<f32>,
};

// Octahedral normal encoding: unit sphere → [0,1]^2
fn oct_encode(n: vec3f) -> vec2f {
    let s = abs(n.x) + abs(n.y) + abs(n.z);
    var p = n.xy / s;
    if (n.z < 0.0) {
        p = (1.0 - abs(p.yx)) * sign(p);
    }
    return p * 0.5 + 0.5;
}

// Pack two [0,1] values into one float: a gets 8-bit integer part, b gets fractional part
fn pack_2f(a: f32, b: f32) -> f32 {
    return floor(clamp(a, 0.0, 1.0) * 255.0) + clamp(b, 0.0, 1.0);
}

// phi(x) = (1 - exp(-x)) / x
// Taylor with 4 terms for |x| < 0.02 avoids catastrophic cancellation.
fn phi(x: f32) -> f32 {
    if (abs(x) < 0.02) {
        return 1.0 + x * (-0.5 + x * (1.0 / 6.0 + x * (-1.0 / 24.0)));
    }
    return (1.0 - exp(-x)) / x;
}

// Volume rendering integral for a ray segment through a tet.
fn compute_integral(c0: vec3<f32>, c1: vec3<f32>, optical_depth: f32) -> vec4<f32> {
    let alpha = exp(-optical_depth);
    let phi_val = phi(optical_depth);
    let w0 = phi_val - alpha;
    let w1 = 1.0 - phi_val;
    let c = c0 * w0 + c1 * w1;
    return vec4<f32>(c, 1.0 - alpha);
}

@fragment
fn main(@builtin(position) frag_coord: vec4<f32>, in: FragmentInput) -> FragmentOutput {
    var out: FragmentOutput;

    let near = uniforms.near_plane;
    let far = uniforms.far_plane;

    // Linearize interpolated NDC depths -> view-space Z (positive = into screen)
    let range = far - near;
    let z_front_clamped = clamp(in.depths.x, 0.0, 1.0);
    let z_back_clamped = clamp(in.depths.y, 0.0, 1.0);
    let z_f = near * far / (far - z_front_clamped * range);
    let z_b = near * far / (far - z_back_clamped * range);

    // Reconstruct ray direction scale factor from pixel position
    let fx = uniforms.intrinsics.x;
    let fy = uniforms.intrinsics.y;
    let cx = uniforms.intrinsics.z;
    let cy = uniforms.intrinsics.w;

    let x_cam = (frag_coord.x - cx) / fx;
    let y_cam = (frag_coord.y - cy) / fy;
    let ray_scale = length(vec3<f32>(x_cam, y_cam, 1.0));

    // Distance through tet = |z_b - z_f| * ray_scale
    let dist = abs(z_b - z_f) * ray_scale;

    // Colors at entry/exit from interpolated offsets
    let c_front = max(in.base_color + vec3<f32>(in.color_offsets.x), vec3<f32>(0.0));
    let c_back = max(in.base_color + vec3<f32>(in.color_offsets.y), vec3<f32>(0.0));

    // Volume rendering integral
    let od = clamp(in.density * dist, 0.0, 88.0);
    out.color = compute_integral(c_back, c_front, od);

    let alpha = out.color.a;

    // --- MRT: G-buffer outputs ---

    // Per-tet aux channels lookup
    let aux_base = in.tet_id * AUX_DIM;
    let roughness = aux_data[aux_base + 0u];
    let env_f0 = aux_data[aux_base + 1u];
    let env_f1 = aux_data[aux_base + 2u];
    let env_f2 = aux_data[aux_base + 3u];
    let env_f3 = aux_data[aux_base + 4u];
    let alb_r = aux_data[aux_base + 5u];
    let alb_g = aux_data[aux_base + 6u];
    let alb_b = aux_data[aux_base + 7u];

    // Average vertex normals for this tet
    let i0 = tet_indices[in.tet_id * 4u];
    let i1 = tet_indices[in.tet_id * 4u + 1u];
    let i2 = tet_indices[in.tet_id * 4u + 2u];
    let i3 = tet_indices[in.tet_id * 4u + 3u];
    let n0 = vec3f(vertex_normals[i0*3u], vertex_normals[i0*3u+1u], vertex_normals[i0*3u+2u]);
    let n1 = vec3f(vertex_normals[i1*3u], vertex_normals[i1*3u+1u], vertex_normals[i1*3u+2u]);
    let n2 = vec3f(vertex_normals[i2*3u], vertex_normals[i2*3u+1u], vertex_normals[i2*3u+2u]);
    let n3 = vec3f(vertex_normals[i3*3u], vertex_normals[i3*3u+1u], vertex_normals[i3*3u+2u]);
    let normal = normalize(n0 + n1 + n2 + n3);

    // Octahedral-encode normal, pack env_f2+f3 into single channel
    let oct_n = oct_encode(normal);
    let env_packed = pack_2f(env_f2, env_f3);

    // Premultiply by alpha — .a = alpha for correct hardware OneMinusSrcAlpha blend
    out.aux0 = vec4f(roughness * alpha, env_f0 * alpha, env_f1 * alpha, alpha);
    out.normals = vec4f(oct_n * alpha, env_packed * alpha, alpha);
    out.albedo = vec4f(alb_r * alpha, alb_g * alpha, alb_b * alpha, alpha);

    return out;
}
