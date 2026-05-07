// Hi-Z mip downsample. Each pass writes one descendant mip from its parent.
// Bound texture is a single-mip view of the parent — `textureLoad(_, _, 0)`
// reads from that mip directly. Out-of-bounds samples (odd parent extents)
// are clamped to the texture edge so they don't pollute min() with zeros.

@group(0) @binding(0) var src_tex: texture_2d<f32>;

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
    let dst = vec2u(u32(frag_coord.x), u32(frag_coord.y));
    let src_dim = textureDimensions(src_tex);
    let src = dst * 2u;
    let max_xy = src_dim - vec2u(1u, 1u);
    let z00 = textureLoad(src_tex, min(src + vec2u(0u, 0u), max_xy), 0).r;
    let z10 = textureLoad(src_tex, min(src + vec2u(1u, 0u), max_xy), 0).r;
    let z01 = textureLoad(src_tex, min(src + vec2u(0u, 1u), max_xy), 0).r;
    let z11 = textureLoad(src_tex, min(src + vec2u(1u, 1u), max_xy), 0).r;
    return vec4f(min(min(z00, z10), min(z01, z11)), 0.0, 0.0, 0.0);
}
