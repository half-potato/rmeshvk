// Texture-to-buffer conversion shader.
//
// Reads an Rgba16Float texture (hardware f16->f32 conversion) and writes
// to a flat f32 storage buffer for consumption by loss/backward shaders.

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> params: vec2<u32>; // width, height

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let width = params.x;
    let height = params.y;

    if (id.x >= width || id.y >= height) {
        return;
    }

    let pixel = textureLoad(input_tex, vec2<i32>(id.xy), 0);
    let idx = (id.y * width + id.x) * 4u;
    output[idx] = pixel.r;
    output[idx + 1u] = pixel.g;
    output[idx + 2u] = pixel.b;
    output[idx + 3u] = pixel.a;
}
