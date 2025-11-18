// Gaussian Splatting Fragment Shader
// Implements alpha blending with 2D Gaussian falloff

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) opacity: f32,
    @location(2) center: vec3<f32>,
    @location(3) cov2d_a: vec2<f32>,
    @location(4) cov2d_b: vec2<f32>,
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
}

@fragment
fn fs_main(input: VertexOutput) -> FragmentOutput {
    // Compute pixel position relative to Gaussian center
    let pixel_pos = input.position.xy;
    let center_pos = input.center.xy;
    let delta = pixel_pos - center_pos;
    
    // Compute 2D Gaussian falloff
    // Using simplified covariance computation
    let cov_a = input.cov2d_a;
    let cov_b = input.cov2d_b;
    
    // Simplified Gaussian: exp(-0.5 * (delta^T * cov^-1 * delta))
    // For performance, we use a simpler approximation
    let dist_sq = dot(delta, delta);
    let scale_factor = length(cov_a) + length(cov_b);
    let gaussian_weight = exp(-0.5 * dist_sq / (scale_factor * scale_factor + 0.0001));
    
    // Apply opacity and Gaussian falloff
    let alpha = input.opacity * gaussian_weight;
    
    // Alpha blending
    let final_color = vec4<f32>(input.color, alpha);
    
    return FragmentOutput(final_color);
}

