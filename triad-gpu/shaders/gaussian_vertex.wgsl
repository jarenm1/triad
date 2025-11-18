// Gaussian Splatting Vertex Shader
// Processes simplified Gaussian points with position, radius, color, and opacity.

struct GaussianPoint {
    position_radius: vec4<f32>, // xyz + radius
    color_opacity: vec4<f32>,   // rgb + opacity
}

struct CameraUniforms {
    view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    view_pos: vec3<f32>,
    _padding: f32,
}

@group(0) @binding(0) var<storage, read> gaussians: array<GaussianPoint>;
@group(0) @binding(1) var<uniform> camera: CameraUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) opacity: f32,
    @location(2) center_ndc: vec2<f32>, // Center in NDC space
    @location(3) pixel_ndc: vec2<f32>,  // This vertex's position in NDC space (for interpolation)
    @location(4) cov2d_a: vec2<f32>,
    @location(5) cov2d_b: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Each Gaussian is rendered as a triangle (3 vertices)
    let gaussian_idx = vertex_index / 3u;
    if (gaussian_idx >= arrayLength(&gaussians)) {
        // Return a degenerate vertex outside clip space
        return VertexOutput(
            vec4<f32>(0.0, 0.0, 0.0, 0.0),
            vec3<f32>(0.0),
            0.0,
            vec2<f32>(0.0),
            vec2<f32>(0.0),
            vec2<f32>(0.0),
            vec2<f32>(0.0)
        );
    }
    let tri_vertex = vertex_index % 3u;
    
    let gaussian = gaussians[gaussian_idx];
    let position = gaussian.position_radius.xyz;
    let radius = max(gaussian.position_radius.w, 1e-4);
    let rgb = gaussian.color_opacity.xyz;
    let opacity = gaussian.color_opacity.w;
    
    // Triangle vertices in normalized device space.
    // These offsets get scaled per-Gaussian so the rasterized triangle always covers
    // enough of the Gaussian footprint for the fragment shader to evaluate.
    let tri_offsets = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.1547),      // Top vertex (2/√3 * scale)
        vec2<f32>(-1.0, -0.57735),  // Bottom left
        vec2<f32>(1.0, -0.57735)    // Bottom right
    );
    // Compute projection Jacobian and project the isotropic 3D Gaussian into screen space.
    // For perspective projection: p_ndc = (x/w, y/w) where (x,y,z,w) = proj_matrix * p_cam
    // The Jacobian J accounts for perspective division
    // IMPORTANT: Use the center position, not the vertex position, for covariance calculation
    let center_view = camera.view_matrix * vec4<f32>(position, 1.0);
    let center_clip = camera.proj_matrix * center_view;
    let w_mask = abs(center_clip.w) < 1e-6;
    let safe_w = select(center_clip.w, 1.0, w_mask);
    
    // Extract projection matrix elements (assuming standard perspective projection)
    // For p_clip = P * p_cam, we need ∂(p_clip.xy / p_clip.w) / ∂p_cam
    // J = (1/w) * [P[0][0] - x*P[3][0], P[0][1] - x*P[3][1], P[0][2] - x*P[3][2]]
    //            [P[1][0] - y*P[3][0], P[1][1] - y*P[3][1], P[1][2] - y*P[3][2]]
    let proj = camera.proj_matrix;
    let x_clip = center_clip.x;
    let y_clip = center_clip.y;
    
    // Build 2x3 Jacobian matrix
    let j00 = (proj[0][0] - x_clip * proj[3][0]) / safe_w;
    let j01 = (proj[0][1] - x_clip * proj[3][1]) / safe_w;
    let j02 = (proj[0][2] - x_clip * proj[3][2]) / safe_w;
    let j10 = (proj[1][0] - y_clip * proj[3][0]) / safe_w;
    let j11 = (proj[1][1] - y_clip * proj[3][1]) / safe_w;
    let j12 = (proj[1][2] - y_clip * proj[3][2]) / safe_w;
    
    // Project isotropic covariance: Σ_screen = radius^2 * (J * J^T)
    let j0 = vec3<f32>(j00, j01, j02);
    let j1 = vec3<f32>(j10, j11, j12);
    let radius_sq = radius * radius;
    let cov2d_00 = radius_sq * dot(j0, j0);
    let cov2d_01 = radius_sq * dot(j0, j1);
    let cov2d_11 = radius_sq * dot(j1, j1);
    
    // Pack 2x2 symmetric covariance matrix into two vec2s
    // cov2d_a = (cov2d_00, cov2d_01)
    // cov2d_b = (cov2d_01, cov2d_11)
    let cov2d_a = vec2<f32>(cov2d_00, cov2d_01);
    let cov2d_b = vec2<f32>(cov2d_01, cov2d_11);
    
    // Compute center position in NDC space for fragment shader
    // The center is the Gaussian's world position (not the quad vertex position)
    // Reuse center_view and center_clip computed earlier for covariance calculation
    let center_ndc = select(
        center_clip.xy / safe_w,
        vec2<f32>(0.0, 0.0),
        w_mask
    );
    
    // Use the projected covariance to determine how large the Gaussian is on screen.
    let trace = cov2d_00 + cov2d_11;
    let det = cov2d_00 * cov2d_11 - cov2d_01 * cov2d_01;
    let discriminant = max(trace * trace - 4.0 * det, 0.0);
    let lambda_max = 0.5 * (trace + sqrt(discriminant));
    let sigma_max = sqrt(max(lambda_max, 1e-6));
    let coverage_sigma = 3.0; // Cover ~99% of distribution
    let radius_ndc = coverage_sigma * sigma_max;
    
    // Build a triangle in clip space centered on the Gaussian. This keeps the primitive
    // axis-aligned in screen space, which is fine because the fragment shader applies the
    // anisotropic falloff using the full covariance.
    let offset_ndc = tri_offsets[tri_vertex] * radius_ndc;
    let pixel_ndc = center_ndc + offset_ndc;
    let clip_pos = vec4<f32>(
        pixel_ndc.x * safe_w,
        pixel_ndc.y * safe_w,
        center_clip.z,
        safe_w
    );
    
    return VertexOutput(
        clip_pos,
        rgb,
        opacity,
        center_ndc,        // Center in NDC space
        pixel_ndc,        // This vertex's NDC position (will be interpolated)
        cov2d_a,
        cov2d_b
    );
}

