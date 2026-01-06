// Triangle fragment shader
// Simple solid triangle rendering with premultiplied alpha

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) opacity: f32,
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Premultiplied alpha output for proper blending
    return vec4<f32>(input.color * input.opacity, input.opacity);
}
