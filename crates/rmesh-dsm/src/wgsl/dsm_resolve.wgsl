// Fourier DSM resolve: reconstructs transmittance T(z_query) from stored
// Fourier coefficients. See TMSM.md for derivation.
//
// Reads 3 textures (9 coefficients packed as in dsm_fourier_fragment.wgsl)
// and a uniform z_query depth, outputs grayscale T value.

struct ResolveUniforms {
    z_query: f32,  // metric (linear) depth
    near: f32,
    far: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> params: ResolveUniforms;
@group(0) @binding(1) var rt0_tex: texture_2d<f32>;
@group(0) @binding(2) var rt1_tex: texture_2d<f32>;
@group(0) @binding(3) var rt2_tex: texture_2d<f32>;

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let coords = vec2<u32>(u32(frag_coord.x), u32(frag_coord.y));

    let c0 = textureLoad(rt0_tex, coords, 0);
    let c1 = textureLoad(rt1_tex, coords, 0);
    let c2 = textureLoad(rt2_tex, coords, 0);

    // Unpack coefficients
    let a0 = c0.x;
    let a1 = c0.y;  let b1 = c0.z;
    let a2 = c0.w;  let b2 = c1.x;
    let a3 = c1.y;  let b3 = c1.z;
    let a4 = c1.w;  let b4 = c2.x;

    // Normalize metric depth to [0,1] (Fourier basis is in linear depth space)
    let z = clamp((params.z_query - params.near) / (params.far - params.near), 0.0, 1.0);

    // Reconstruct absorbance A(z) = integral_0^z sigma(t) dt
    // A(z) = (a0/2)*z + sum_k [ a_k*sin(2*PI*k*z)/(2*PI*k) + b_k*(1-cos(2*PI*k*z))/(2*PI*k) ]
    var A = (a0 / 2.0) * z;

    // k=1
    let w1 = TWO_PI * 1.0;
    A += a1 * sin(w1 * z) / w1 + b1 * (1.0 - cos(w1 * z)) / w1;

    // k=2
    let w2 = TWO_PI * 2.0;
    A += a2 * sin(w2 * z) / w2 + b2 * (1.0 - cos(w2 * z)) / w2;

    // k=3
    let w3 = TWO_PI * 3.0;
    A += a3 * sin(w3 * z) / w3 + b3 * (1.0 - cos(w3 * z)) / w3;

    // k=4
    let w4 = TWO_PI * 4.0;
    A += a4 * sin(w4 * z) / w4 + b4 * (1.0 - cos(w4 * z)) / w4;

    // Clamp absorbance (Gibbs oscillation can make it negative)
    A = max(A, 0.0);

    let T = exp(-A);

    return vec4<f32>(T, T, T, 1.0);
}
