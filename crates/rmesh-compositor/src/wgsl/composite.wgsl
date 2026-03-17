// Compositor: fullscreen triangle that composites tet volume rendering with
// opaque primitive rendering using depth comparison.

struct CompositorUniforms {
    near: f32,
    far: f32,
}

@group(0) @binding(0) var tet_color_tex: texture_2d<f32>;
@group(0) @binding(1) var tet_depth_tex: texture_2d<f32>;
@group(0) @binding(2) var prim_color_tex: texture_2d<f32>;
@group(0) @binding(3) var prim_depth_tex: texture_depth_2d;
@group(0) @binding(4) var<uniform> params: CompositorUniforms;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let coords = vec2<i32>(in.pos.xy);

    // Tet volume: premultiplied-alpha color, depth stored as (depth*alpha, _, _, alpha) in Rgba16Float
    let tet_color = textureLoad(tet_color_tex, coords, 0);
    let tet_depth_raw = textureLoad(tet_depth_tex, coords, 0);

    let tet_alpha = tet_color.a;
    // Tet depth is premultiplied: depth_val = depth * alpha. Recover linear view-space depth.
    let tet_view_z = select(1e20, tet_depth_raw.x / tet_alpha, tet_alpha > 0.001);

    // Primitive: hardware depth in [0,1] clip space
    let prim_clip_d = textureLoad(prim_depth_tex, coords, 0);
    let near = params.near;
    let far = params.far;
    // Reverse the perspective projection: clip_d = (far * (z - near)) / (z * (far - near))
    // => z = near * far / (far - clip_d * (far - near))
    let has_prim = prim_clip_d < 1.0;
    let prim_view_z = select(1e20, near * far / (far - prim_clip_d * (far - near)), has_prim);

    // Primitive color (opaque)
    let prim_color = textureLoad(prim_color_tex, coords, 0);

    // Composite
    if !has_prim {
        // No primitive — pass through tet color unchanged
        return tet_color;
    }
    if prim_view_z < tet_view_z {
        // Primitive is closer — fully opaque override
        return vec4<f32>(prim_color.rgb, 1.0);
    }
    // Tet is closer — tet premultiplied color in front, primitive visible through transparency
    return vec4<f32>(tet_color.rgb + (1.0 - tet_alpha) * prim_color.rgb, 1.0);
}
