// Layer blending shader - composites multiple layer textures with opacity

struct LayerUniforms {
    opacity: vec3<f32>,  // Opacity for each layer (Points, Gaussians, Triangles)
    _padding: f32,
}

@group(0) @binding(0) var layer0: texture_2d<f32>;  // Points layer
@group(0) @binding(1) var layer1: texture_2d<f32>;  // Gaussians layer
@group(0) @binding(2) var layer2: texture_2d<f32>;  // Triangles layer
@group(0) @binding(3) var layer_sampler: sampler;
@group(0) @binding(4) var<uniform> uniforms: LayerUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Fullscreen quad
    let x = f32((vertex_index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vertex_index & 2u) * 2.0 - 1.0;
    
    var out: VertexOutput;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;
    
    // Sample all layers
    let color0 = textureSample(layer0, layer_sampler, uv);
    let color1 = textureSample(layer1, layer_sampler, uv);
    let color2 = textureSample(layer2, layer_sampler, uv);
    
    // Blend layers with opacity: result = sum(layer[i] * opacity[i] * alpha[i])
    // Multiply by texture alpha so transparent pixels don't contribute
    var result = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    
    result += color0 * uniforms.opacity.x * color0.a;
    result += color1 * uniforms.opacity.y * color1.a;
    result += color2 * uniforms.opacity.z * color2.a;
    
    // Clamp alpha to [0, 1] for proper blending
    result.a = clamp(result.a, 0.0, 1.0);
    
    return result;
}
