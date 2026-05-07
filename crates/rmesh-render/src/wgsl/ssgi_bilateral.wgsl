// Bilateral SSGI denoise — depth + normal aware, separable (H + V passes).
//
// Same structure as ao_bilateral.wgsl but operates on Rgba16Float radiance
// instead of R8Unorm AO. Sky / no-surface pixels pass through unchanged.

const RADIUS: i32 = 5;
const SIGMA_S: f32 = 3.0;
const Z_SKY_HALF: f32 = 0.5e20;

struct BlurUniforms {
    dir_x: i32,
    dir_y: i32,
    sigma_z: f32,
    sigma_n: f32,
}

@group(0) @binding(0) var<uniform> u: BlurUniforms;
@group(0) @binding(1) var ssgi_tex: texture_2d<f32>;
@group(0) @binding(2) var depth_tex: texture_2d<f32>;
@group(0) @binding(3) var normals_tex: texture_2d<f32>;

struct VsOut { @builtin(position) pos: vec4f }

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    var out: VsOut;
    out.pos = vec4f(x, y, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
    let coords = vec2i(i32(frag_coord.x), i32(frag_coord.y));
    let dim = vec2i(textureDimensions(ssgi_tex));
    let dir = vec2i(u.dir_x, u.dir_y);

    let center = textureLoad(ssgi_tex, coords, 0);
    let z_c  = textureLoad(depth_tex, coords, 0).r;
    let n_raw_c = textureLoad(normals_tex, coords, 0);

    if (n_raw_c.a < 0.01 || z_c >= Z_SKY_HALF) {
        return center;
    }
    let n_c = normalize(n_raw_c.rgb / n_raw_c.a);

    let inv_2sz2 = 1.0 / max(2.0 * u.sigma_z * u.sigma_z, 1e-8);
    let inv_2ss2 = 1.0 / (2.0 * SIGMA_S * SIGMA_S);

    var sum_w: f32 = 0.0;
    var sum_v: vec3f = vec3f(0.0);

    for (var i: i32 = -RADIUS; i <= RADIUS; i = i + 1) {
        let p = clamp(coords + dir * i, vec2i(0, 0), dim - vec2i(1, 1));

        let v_p = textureLoad(ssgi_tex, p, 0).rgb;
        let z_p = textureLoad(depth_tex, p, 0).r;
        let n_raw_p = textureLoad(normals_tex, p, 0);
        if (n_raw_p.a < 0.01 || z_p >= Z_SKY_HALF) { continue; }
        let n_p = normalize(n_raw_p.rgb / n_raw_p.a);

        let w_s = exp(-f32(i * i) * inv_2ss2);
        let dz = z_p - z_c;
        let w_z = exp(-dz * dz * inv_2sz2);
        let w_n = pow(max(dot(n_p, n_c), 0.0), u.sigma_n);
        let w = w_s * w_z * w_n;

        sum_w = sum_w + w;
        sum_v = sum_v + v_p * w;
    }

    let out_rgb = select(center.rgb, sum_v / sum_w, sum_w > 1e-6);
    return vec4f(out_rgb, center.a);
}
