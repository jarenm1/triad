// Gaussian Splatting Fragment Shader
// Implements alpha blending with 2D Gaussian falloff

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) opacity: f32,
    @location(2) center_ndc: vec2<f32>, // Center in NDC space
    @location(3) pixel_ndc: vec2<f32>,  // Interpolated NDC position of this fragment
    @location(4) cov2d_a: vec2<f32>,
    @location(5) cov2d_b: vec2<f32>,
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
}

@fragment
fn fs_main(input: VertexOutput) -> FragmentOutput {
    // Both center and pixel are now in NDC space (-1 to 1)
    // No coordinate conversion needed!
    let delta = input.pixel_ndc - input.center_ndc;
    
    // Compute 2D Gaussian falloff using the covariance matrix
    // The covariance matrix is stored as:
    // cov2d_a = (cov_00, cov_01)
    // cov2d_b = (cov_01, cov_11)
    // This represents the 2x2 symmetric matrix: [[cov_00, cov_01], [cov_01, cov_11]]
    
    let cov_00 = input.cov2d_a.x;
    let cov_01 = input.cov2d_a.y; // Same as cov2d_b.x
    let cov_11 = input.cov2d_b.y;
    
    // Compute determinant of covariance matrix
    let det = cov_00 * cov_11 - cov_01 * cov_01;
    
    // Avoid division by zero - if determinant is too small, use simple distance falloff
    if (abs(det) < 0.0001) {
        // Fallback to circular Gaussian
        let dist_sq = dot(delta, delta);
        let gaussian_weight = exp(-0.5 * dist_sq);
        let alpha = input.opacity * gaussian_weight;
        return FragmentOutput(vec4<f32>(input.color, clamp(alpha, 0.0, 1.0)));
    }
    
    // Compute inverse covariance matrix
    // For 2x2 matrix [[a, b], [b, c]], inverse is (1/det) * [[c, -b], [-b, a]]
    let inv_det = 1.0 / det;
    let inv_cov_00 = inv_det * cov_11;
    let inv_cov_01 = -inv_det * cov_01;
    let inv_cov_11 = inv_det * cov_00;
    
    // Compute delta^T * inv_cov * delta
    // This is: delta.x * (inv_cov_00 * delta.x + inv_cov_01 * delta.y) +
    //          delta.y * (inv_cov_01 * delta.x + inv_cov_11 * delta.y)
    let power = delta.x * (inv_cov_00 * delta.x + inv_cov_01 * delta.y) +
                delta.y * (inv_cov_01 * delta.x + inv_cov_11 * delta.y);
    
    // Clamp power to avoid overflow
    if (power > 10.0) {
        return FragmentOutput(vec4<f32>(input.color, 0.0));
    }
    
    // Compute Gaussian: exp(-0.5 * power)
    let gaussian_weight = exp(-0.5 * power);
    
    // Apply opacity
    let alpha = input.opacity * gaussian_weight;
    
    // Clamp to valid range
    let final_alpha = clamp(alpha, 0.0, 1.0);
    
    return FragmentOutput(vec4<f32>(input.color, final_alpha));
}

