// Simple debug shader - just outputs a solid color to verify rendering works

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    // Generate a full-screen quad
    let x = f32((vertex_index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vertex_index & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    // Output bright red to verify fragment shader works
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}

