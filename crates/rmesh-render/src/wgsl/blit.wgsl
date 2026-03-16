// Fullscreen triangle blit: samples an Rgba16Float texture and outputs to the swapchain.

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    // Fullscreen triangle: 3 vertices cover clip space [-1,1]^2.
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    // Map clip [-1,1] to UV [0,1], flip Y so top-left is (0,0).
    out.uv = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let color = textureSample(src_tex, src_sampler, in.uv);
    // Premultiplied alpha on black background — RGB is already correct.
    // Set alpha=1 for the swapchain (opaque window).
    return vec4<f32>(color.rgb, 1.0);
}
