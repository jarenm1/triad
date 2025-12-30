// Point cloud fragment shader
// Renders points as circular discs with smooth edges

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) opacity: f32,
    @location(2) local_pos: vec2<f32>,  // Position within the quad
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Compute distance from center (0,0) in local space
    let dist_sq = dot(input.local_pos, input.local_pos);
    
    // Circular disc with smooth edge
    // At distance 1.0, we're at the edge of the point
    let dist = sqrt(dist_sq);
    
    // Smooth falloff at the edge
    let edge_smoothness = 0.1;
    let alpha_factor = 1.0 - smoothstep(1.0 - edge_smoothness, 1.0 + edge_smoothness, dist);
    
    // Discard pixels outside the circle
    if (alpha_factor < 0.01) {
        discard;
    }
    
    // Apply opacity
    let alpha = input.opacity * alpha_factor;
    
    return vec4<f32>(input.color, alpha);
}

