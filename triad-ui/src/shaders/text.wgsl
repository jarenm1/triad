// Text rendering shader
// Uses a single-channel (R8) glyph atlas

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
}

@group(0) @binding(0) var glyph_texture: texture_2d<f32>;
@group(0) @binding(1) var glyph_sampler: sampler;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(input.position, 0.0, 1.0);
    output.uv = input.uv;
    output.color = input.color;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sample the single-channel glyph texture
    let alpha = textureSample(glyph_texture, glyph_sampler, input.uv).r;
    
    // Discard fully transparent pixels
    if (alpha < 0.01) {
        discard;
    }
    
    // Apply text color with sampled alpha
    return vec4<f32>(input.color.rgb, input.color.a * alpha);
}

