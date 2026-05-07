// Bilateral AO filter — depth + normal aware, separable (H + V passes).
//
// One pipeline used twice with different bind groups: the `direction` uniform
// is (1,0) for the horizontal pass and (0,1) for the vertical pass. Reads:
//   - ao_tex     : current AO (R8Unorm or R16Float)
//   - depth_tex  : Hi-Z mip 0, linear view-space Z (R32Float)
//   - normals_tex: volume normals MRT (gradient·α, α)
// Writes filtered AO to the bound color attachment.
//
// Weights per tap: spatial Gaussian × depth Gaussian × normal cosine-power.
// Pixels with no surface (n_raw.a < 0.01) pass through unchanged.

const RADIUS: i32 = 5;
const SIGMA_S: f32 = 3.0;
const Z_SKY_HALF: f32 = 0.5e20;

struct BlurUniforms {
    // (1,0) for H pass, (0,1) for V pass. i32 so we can multiply by signed i.
    dir_x: i32,
    dir_y: i32,
    sigma_z: f32,        // depth tolerance, world units
    sigma_n: f32,        // normal cosine power (e.g. 8 = cos^8)
}

@group(0) @binding(0) var<uniform> u: BlurUniforms;
@group(0) @binding(1) var ao_tex: texture_2d<f32>;
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
    let dim = vec2i(textureDimensions(ao_tex));
    let dir = vec2i(u.dir_x, u.dir_y);

    let ao_c = textureLoad(ao_tex, coords, 0).r;
    let z_c  = textureLoad(depth_tex, coords, 0).r;
    let n_raw_c = textureLoad(normals_tex, coords, 0);

    // No surface (sky) or sky depth → pass through; nothing meaningful to blur.
    if (n_raw_c.a < 0.01 || z_c >= Z_SKY_HALF) {
        return vec4f(ao_c);
    }
    let n_c = normalize(n_raw_c.rgb / n_raw_c.a);

    let inv_2sz2 = 1.0 / max(2.0 * u.sigma_z * u.sigma_z, 1e-8);
    let inv_2ss2 = 1.0 / (2.0 * SIGMA_S * SIGMA_S);

    var sum_w: f32 = 0.0;
    var sum_v: f32 = 0.0;

    for (var i: i32 = -RADIUS; i <= RADIUS; i = i + 1) {
        let p = clamp(coords + dir * i, vec2i(0, 0), dim - vec2i(1, 1));

        let ao_p = textureLoad(ao_tex, p, 0).r;
        let z_p  = textureLoad(depth_tex, p, 0).r;
        let n_raw_p = textureLoad(normals_tex, p, 0);
        if (n_raw_p.a < 0.01 || z_p >= Z_SKY_HALF) { continue; }
        let n_p = normalize(n_raw_p.rgb / n_raw_p.a);

        let w_s = exp(-f32(i * i) * inv_2ss2);
        let dz = z_p - z_c;
        let w_z = exp(-dz * dz * inv_2sz2);
        let w_n = pow(max(dot(n_p, n_c), 0.0), u.sigma_n);
        let w = w_s * w_z * w_n;

        sum_w = sum_w + w;
        sum_v = sum_v + ao_p * w;
    }

    let ao_out = select(ao_c, sum_v / sum_w, sum_w > 1e-6);
    return vec4f(ao_out);
}
