// Gaussian Splatting Fragment Shader - Simplified version
// Simple circular Gaussian falloff

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
    
    // Gaussian falloff: exp(-0.5 * (r/sigma)^2)
    // Since we used 3-sigma for the quad size, and local_pos is in [-1, 3] range,
    // we need to normalize. At the edge (local_pos = 1), we want ~3 sigma.
    // So: gaussian_power = dist_sq * 0.5 (since dist=1 should be ~1 sigma)
    let gaussian_weight = exp(-0.5 * dist_sq);
    
    // Discard pixels that are too far from center (optimization)
    if (gaussian_weight < 0.01) {
        discard;
    }
    
    // Apply opacity
    let alpha = input.opacity * gaussian_weight;
    
    return vec4<f32>(input.color, alpha);
}
