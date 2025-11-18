// Gaussian Splatting Vertex Shader
// Processes Gaussian points with position, orientation, per-axis scale, color, and opacity.

struct GaussianPoint {
    position_radius: vec4<f32>, // xyz + radius hint
    color_opacity: vec4<f32>,   // rgb + opacity
    rotation: vec4<f32>,        // quaternion
    scale: vec4<f32>,           // xyz scale (+ padding)
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
    @location(6) falloff_scale: f32,
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
            vec2<f32>(0.0),
            1.0
        );
    }
    let tri_vertex = vertex_index % 3u;
    
    let gaussian = gaussians[gaussian_idx];
    let position = gaussian.position_radius.xyz;
    let radius_hint = max(gaussian.position_radius.w, 1e-4);
    let rgb = gaussian.color_opacity.xyz;
    let opacity = gaussian.color_opacity.w;
    let softness_base = clamp(1.0 - opacity, 0.0, 1.0);
    let softness = max(softness_base, 0.08); // ensure some blur even if opacity≈1
    let radius_scale = mix(1.2, 5.0, softness);
    let falloff_scale = mix(1.0, 0.05, softness); // lower scale -> slower falloff (blurrier)
    
    let scale_vec = vec3<f32>(
        max(gaussian.scale.x, 1e-4),
        max(gaussian.scale.y, 1e-4),
        max(gaussian.scale.z, 1e-4)
    );
    let q = gaussian.rotation;
    let q_len = sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    let epsilon = 1e-8;
    let qn = select(
        vec4<f32>(q.x / q_len, q.y / q_len, q.z / q_len, q.w / q_len),
        vec4<f32>(0.0, 0.0, 0.0, 1.0), // identity quaternion (no rotation)
        q_len < epsilon
    );
    
    let rot_matrix = mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (qn.y * qn.y + qn.z * qn.z), 2.0 * (qn.x * qn.y - qn.w * qn.z), 2.0 * (qn.x * qn.z + qn.w * qn.y)),
        vec3<f32>(2.0 * (qn.x * qn.y + qn.w * qn.z), 1.0 - 2.0 * (qn.x * qn.x + qn.z * qn.z), 2.0 * (qn.y * qn.z - qn.w * qn.x)),
        vec3<f32>(2.0 * (qn.x * qn.z - qn.w * qn.y), 2.0 * (qn.y * qn.z + qn.w * qn.x), 1.0 - 2.0 * (qn.x * qn.x + qn.y * qn.y))
    );
    
    let tri_offsets = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.1547),      // Top vertex (2/√3 * scale)
        vec2<f32>(-1.0, -0.57735),  // Bottom left
        vec2<f32>(1.0, -0.57735)    // Bottom right
    );
    
    let geo_radius = radius_hint * radius_scale;
    let offset = tri_offsets[tri_vertex] * geo_radius;
    let local_pos = rot_matrix * vec3<f32>(offset.x, offset.y, 0.0);
    let world_pos = position + local_pos;
    
    let view_pos = camera.view_matrix * vec4<f32>(world_pos, 1.0);
    let clip_pos = camera.proj_matrix * view_pos;
    let ndc_pos = select(
        clip_pos.xy / clip_pos.w,
        vec2<f32>(0.0, 0.0),
        abs(clip_pos.w) < 1e-6
    );
    
    // Covariance computation using anisotropic scales.
    let center_view = camera.view_matrix * vec4<f32>(position, 1.0);
    let center_clip = camera.proj_matrix * center_view;
    let w_mask = abs(center_clip.w) < 1e-6;
    let safe_w = select(center_clip.w, 1.0, w_mask);
    let center_ndc = select(
        center_clip.xy / safe_w,
        vec2<f32>(0.0, 0.0),
        w_mask
    );
    
    let scale_sq = vec3<f32>(
        scale_vec.x * scale_vec.x,
        scale_vec.y * scale_vec.y,
        scale_vec.z * scale_vec.z
    );
    let sigma_w_00 = rot_matrix[0].x * scale_sq.x * rot_matrix[0].x + rot_matrix[0].y * scale_sq.y * rot_matrix[0].y + rot_matrix[0].z * scale_sq.z * rot_matrix[0].z;
    let sigma_w_01 = rot_matrix[0].x * scale_sq.x * rot_matrix[1].x + rot_matrix[0].y * scale_sq.y * rot_matrix[1].y + rot_matrix[0].z * scale_sq.z * rot_matrix[1].z;
    let sigma_w_02 = rot_matrix[0].x * scale_sq.x * rot_matrix[2].x + rot_matrix[0].y * scale_sq.y * rot_matrix[2].y + rot_matrix[0].z * scale_sq.z * rot_matrix[2].z;
    let sigma_w_11 = rot_matrix[1].x * scale_sq.x * rot_matrix[1].x + rot_matrix[1].y * scale_sq.y * rot_matrix[1].y + rot_matrix[1].z * scale_sq.z * rot_matrix[1].z;
    let sigma_w_12 = rot_matrix[1].x * scale_sq.x * rot_matrix[2].x + rot_matrix[1].y * scale_sq.y * rot_matrix[2].y + rot_matrix[1].z * scale_sq.z * rot_matrix[2].z;
    let sigma_w_22 = rot_matrix[2].x * scale_sq.x * rot_matrix[2].x + rot_matrix[2].y * scale_sq.y * rot_matrix[2].y + rot_matrix[2].z * scale_sq.z * rot_matrix[2].z;
    
    let view_rot = mat3x3<f32>(
        camera.view_matrix[0].xyz,
        camera.view_matrix[1].xyz,
        camera.view_matrix[2].xyz
    );
    let sigma_w_mat = mat3x3<f32>(
        vec3<f32>(sigma_w_00, sigma_w_01, sigma_w_02),
        vec3<f32>(sigma_w_01, sigma_w_11, sigma_w_12),
        vec3<f32>(sigma_w_02, sigma_w_12, sigma_w_22)
    );
    let sigma_c = view_rot * sigma_w_mat * transpose(view_rot);
    
    let proj = camera.proj_matrix;
    let x_clip = center_clip.x;
    let y_clip = center_clip.y;
    
    let j00 = (proj[0][0] - x_clip * proj[3][0]) / safe_w;
    let j01 = (proj[0][1] - x_clip * proj[3][1]) / safe_w;
    let j02 = (proj[0][2] - x_clip * proj[3][2]) / safe_w;
    let j10 = (proj[1][0] - y_clip * proj[3][0]) / safe_w;
    let j11 = (proj[1][1] - y_clip * proj[3][1]) / safe_w;
    let j12 = (proj[1][2] - y_clip * proj[3][2]) / safe_w;
    
    let j0 = vec3<f32>(j00, j01, j02);
    let j1 = vec3<f32>(j10, j11, j12);
    let sigma_c_j0 = sigma_c * j0;
    let sigma_c_j1 = sigma_c * j1;
    var cov2d_00 = dot(j0, sigma_c_j0);
    var cov2d_01 = dot(j0, sigma_c_j1);
    var cov2d_11 = dot(j1, sigma_c_j1);
    
    let blur_factor = radius_scale * radius_scale;
    cov2d_00 = max(cov2d_00 * blur_factor, 1e-6);
    cov2d_01 = cov2d_01 * blur_factor;
    cov2d_11 = max(cov2d_11 * blur_factor, 1e-6);
    
    let cov2d_a = vec2<f32>(cov2d_00, cov2d_01);
    let cov2d_b = vec2<f32>(cov2d_01, cov2d_11);
    
    return VertexOutput(
        clip_pos,
        rgb,
        opacity,
        center_ndc,
        ndc_pos,
        cov2d_a,
        cov2d_b,
        falloff_scale
    );
}

