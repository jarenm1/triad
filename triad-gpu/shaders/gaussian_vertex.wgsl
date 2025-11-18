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
    @location(2) center: vec3<f32>,
    @location(3) cov2d_a: vec2<f32>,
    @location(4) cov2d_b: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Each Gaussian is rendered as a quad (4 vertices)
    let gaussian_idx = vertex_index / 4u;
    let quad_vertex = vertex_index % 4u;
    
    let gaussian = gaussians[gaussian_idx];
    
    // Convert quaternion to rotation matrix
    let q = gaussian.rotation;
    let rot_matrix = mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (q.y * q.y + q.z * q.z), 2.0 * (q.x * q.y - q.w * q.z), 2.0 * (q.x * q.z + q.w * q.y)),
        vec3<f32>(2.0 * (q.x * q.y + q.w * q.z), 1.0 - 2.0 * (q.x * q.x + q.z * q.z), 2.0 * (q.y * q.z - q.w * q.x)),
        vec3<f32>(2.0 * (q.x * q.z - q.w * q.y), 2.0 * (q.y * q.z + q.w * q.x), 1.0 - 2.0 * (q.x * q.x + q.y * q.y))
    );
    
    // Quad vertices in local space (-1 to 1)
    let quad_offsets = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0)
    );
    
    let offset = quad_offsets[quad_vertex];
    
    // Transform quad vertex by scale and rotation
    let local_pos = rot_matrix * vec3<f32>(offset.x * gaussian.scale.x, offset.y * gaussian.scale.y, 0.0);
    let world_pos = gaussian.position + local_pos;
    
    // Transform to clip space
    let view_pos = camera.view_matrix * vec4<f32>(world_pos, 1.0);
    let clip_pos = camera.proj_matrix * view_pos;
    
    // Compute 2D covariance for fragment shader
    let view_dir = normalize(camera.view_pos - gaussian.position);
    let right = normalize(cross(view_dir, vec3<f32>(0.0, 1.0, 0.0)));
    let up = cross(right, view_dir);
    
    let scale_2d = vec2<f32>(gaussian.scale.x, gaussian.scale.y);
    let cov2d_a = right.xy * scale_2d.x;
    let cov2d_b = up.xy * scale_2d.y;
    
    return VertexOutput(
        clip_pos,
        gaussian.color,
        gaussian.opacity,
        gaussian.position,
        cov2d_a,
        cov2d_b
    );
}

