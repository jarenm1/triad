// Gaussian Splatting Vertex Shader
// Processes Gaussian points with position, scale, rotation, color, and opacity

struct GaussianPoint {
    position: vec3<f32>,
    scale: vec3<f32>,
    rotation: vec4<f32>, // quaternion
    color: vec3<f32>,
    opacity: f32,
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
    // Convert quaternion to rotation matrix
    let q = gaussian.rotation;
    
    // Normalize quaternion to avoid introducing scale/shear
    // If quaternion has zero length, use identity quaternion (0, 0, 0, 1) instead
    let q_len = sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    let qn = select(
        vec4<f32>(q.x / q_len, q.y / q_len, q.z / q_len, q.w / q_len),
        vec4<f32>(0.0, 0.0, 0.0, 1.0), // identity quaternion (no rotation)
        q_len == 0.0
    );
    
    let rot_matrix = mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (qn.y * qn.y + qn.z * qn.z), 2.0 * (qn.x * qn.y - qn.w * qn.z), 2.0 * (qn.x * qn.z + qn.w * qn.y)),
        vec3<f32>(2.0 * (qn.x * qn.y + qn.w * qn.z), 1.0 - 2.0 * (qn.x * qn.x + qn.z * qn.z), 2.0 * (qn.y * qn.z - qn.w * qn.x)),
        vec3<f32>(2.0 * (qn.x * qn.z - qn.w * qn.y), 2.0 * (qn.y * qn.z + qn.w * qn.x), 1.0 - 2.0 * (qn.x * qn.x + qn.y * qn.y))
    );
    
    // Triangle vertices in local space - form an equilateral triangle
    // Centered at origin, rotated by the quaternion
    let tri_offsets = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.1547),      // Top vertex (2/√3 * scale)
        vec2<f32>(-1.0, -0.57735),  // Bottom left
        vec2<f32>(1.0, -0.57735)    // Bottom right
    );
    
    let offset = tri_offsets[tri_vertex];
    
    // Transform triangle vertex by rotation only (no scaling)
    let local_pos = rot_matrix * vec3<f32>(offset.x, offset.y, 0.0);
    let world_pos = gaussian.position + local_pos;
    
    // Transform to clip space
    let view_pos = camera.view_matrix * vec4<f32>(world_pos, 1.0);
    let clip_pos = camera.proj_matrix * view_pos;
    
    // Compute NDC position for this vertex (for fragment shader interpolation)
    let ndc_pos = clip_pos.xy / clip_pos.w;
    
    // Compute 2D covariance using Jacobian-based projection
    // Step 1: Construct 3D covariance Σ_w from rot_matrix and scale
    // Σ_w = R * diag(s²) * R^T
    // More efficient construction: Σ_w[i][j] = sum_k(R[i][k] * s[k]² * R[j][k])
    let scale_sq = vec3<f32>(gaussian.scale.x * gaussian.scale.x, gaussian.scale.y * gaussian.scale.y, gaussian.scale.z * gaussian.scale.z);
    let sigma_w_00 = rot_matrix[0].x * scale_sq.x * rot_matrix[0].x + rot_matrix[0].y * scale_sq.y * rot_matrix[0].y + rot_matrix[0].z * scale_sq.z * rot_matrix[0].z;
    let sigma_w_01 = rot_matrix[0].x * scale_sq.x * rot_matrix[1].x + rot_matrix[0].y * scale_sq.y * rot_matrix[1].y + rot_matrix[0].z * scale_sq.z * rot_matrix[1].z;
    let sigma_w_02 = rot_matrix[0].x * scale_sq.x * rot_matrix[2].x + rot_matrix[0].y * scale_sq.y * rot_matrix[2].y + rot_matrix[0].z * scale_sq.z * rot_matrix[2].z;
    let sigma_w_11 = rot_matrix[1].x * scale_sq.x * rot_matrix[1].x + rot_matrix[1].y * scale_sq.y * rot_matrix[1].y + rot_matrix[1].z * scale_sq.z * rot_matrix[1].z;
    let sigma_w_12 = rot_matrix[1].x * scale_sq.x * rot_matrix[2].x + rot_matrix[1].y * scale_sq.y * rot_matrix[2].y + rot_matrix[1].z * scale_sq.z * rot_matrix[2].z;
    let sigma_w_22 = rot_matrix[2].x * scale_sq.x * rot_matrix[2].x + rot_matrix[2].y * scale_sq.y * rot_matrix[2].y + rot_matrix[2].z * scale_sq.z * rot_matrix[2].z;
    
    // Step 2: Transform to camera space: Σ_c = R_view * Σ_w * R_view^T
    // Extract rotation part from view matrix (upper-left 3x3)
    let view_rot = mat3x3<f32>(
        camera.view_matrix[0].xyz,
        camera.view_matrix[1].xyz,
        camera.view_matrix[2].xyz
    );
    // Compute Σ_c = R_view * Σ_w * R_view^T
    let sigma_w_mat = mat3x3<f32>(
        vec3<f32>(sigma_w_00, sigma_w_01, sigma_w_02),
        vec3<f32>(sigma_w_01, sigma_w_11, sigma_w_12),
        vec3<f32>(sigma_w_02, sigma_w_12, sigma_w_22)
    );
    let sigma_w_rotated = view_rot * sigma_w_mat;
    let sigma_c = sigma_w_rotated * transpose(view_rot);
    
    // Step 3: Compute projection Jacobian and project to screen space
    // For perspective projection: p_ndc = (x/w, y/w) where (x,y,z,w) = proj_matrix * p_cam
    // The Jacobian J accounts for perspective division
    let p_cam = view_pos.xyz;
    let w = clip_pos.w;
    
    // Extract projection matrix elements (assuming standard perspective projection)
    // For p_clip = P * p_cam, we need ∂(p_clip.xy / p_clip.w) / ∂p_cam
    // J = (1/w) * [P[0][0] - x*P[3][0], P[0][1] - x*P[3][1], P[0][2] - x*P[3][2]]
    //            [P[1][0] - y*P[3][0], P[1][1] - y*P[3][1], P[1][2] - y*P[3][2]]
    let proj = camera.proj_matrix;
    let x_clip = clip_pos.x;
    let y_clip = clip_pos.y;
    
    // Build 2x3 Jacobian matrix
    let j00 = (proj[0][0] - x_clip * proj[3][0]) / w;
    let j01 = (proj[0][1] - x_clip * proj[3][1]) / w;
    let j02 = (proj[0][2] - x_clip * proj[3][2]) / w;
    let j10 = (proj[1][0] - y_clip * proj[3][0]) / w;
    let j11 = (proj[1][1] - y_clip * proj[3][1]) / w;
    let j12 = (proj[1][2] - y_clip * proj[3][2]) / w;
    
    // Step 4: Project covariance: Σ_screen = J * Σ_c * J^T
    // J is 2x3, Σ_c is 3x3, result is 2x2
    // Σ_screen[0][0] = J[0] * Σ_c * J[0]^T
    // Σ_screen[0][1] = J[0] * Σ_c * J[1]^T
    // Σ_screen[1][1] = J[1] * Σ_c * J[1]^T
    let j0 = vec3<f32>(j00, j01, j02);
    let j1 = vec3<f32>(j10, j11, j12);
    let sigma_c_j0 = sigma_c * j0;
    let sigma_c_j1 = sigma_c * j1;
    let cov2d_00 = dot(j0, sigma_c_j0);
    let cov2d_01 = dot(j0, sigma_c_j1);
    let cov2d_11 = dot(j1, sigma_c_j1);
    
    // Pack 2x2 symmetric covariance matrix into two vec2s
    // cov2d_a = (cov2d_00, cov2d_01)
    // cov2d_b = (cov2d_01, cov2d_11)
    let cov2d_a = vec2<f32>(cov2d_00, cov2d_01);
    let cov2d_b = vec2<f32>(cov2d_01, cov2d_11);
    
    // Compute center position in NDC space for fragment shader
    // The center is the Gaussian's world position (not the quad vertex position)
    let center_view = camera.view_matrix * vec4<f32>(gaussian.position, 1.0);
    let center_clip = camera.proj_matrix * center_view;
    let center_ndc = center_clip.xy / center_clip.w;
    
    return VertexOutput(
        clip_pos,
        gaussian.color,
        gaussian.opacity,
        center_ndc,        // Center in NDC space
        ndc_pos,          // This vertex's NDC position (will be interpolated)
        cov2d_a,
        cov2d_b
    );
}

