// Hi-Z mip 0 generation. Fullscreen pass that fuses the hardware depth buffer
// (opaque primitives) with the volume's expected-termination depth MRT into a
// single linear view-space Z. Sky pixels (no surface) write Z_SKY (a large
// finite value) so the downsample chain's min() leaves them as "no occluder".
//
// Output format: R32Float, value = positive view-space Z (linear).

struct HizUniforms {
    near: f32,
    far: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> g: HizUniforms;
@group(0) @binding(1) var hw_depth_tex: texture_depth_2d;
@group(0) @binding(2) var volume_depth_tex: texture_2d<f32>;

struct VsOut { @builtin(position) pos: vec4f }

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    var out: VsOut;
    out.pos = vec4f(x, y, 0.0, 1.0);
    return out;
}

const Z_SKY: f32 = 1.0e20;

fn ndc_to_linear_z(ndc_z: f32) -> f32 {
    return g.near * g.far / (g.far - ndc_z * (g.far - g.near));
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
    let coords = vec2u(u32(frag_coord.x), u32(frag_coord.y));
    let hw_d = textureLoad(hw_depth_tex, coords, 0);
    let vol = textureLoad(volume_depth_tex, coords, 0);
    let has_hw = hw_d < 0.999;
    let has_vol = vol.a > 0.01;
    var z: f32 = Z_SKY;
    if (has_hw) { z = ndc_to_linear_z(hw_d); }
    if (has_vol) { z = min(z, vol.r / vol.a); }
    return vec4f(z, 0.0, 0.0, 0.0);
}
