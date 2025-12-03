// Gaussian splatting vertex shader
// With rotation support for anisotropic splats

struct GaussianPoint {
    position: vec3<f32>,      // xyz center
    _pad0: f32,
    color_opacity: vec4<f32>, // rgb + opacity (a)
    rotation: vec4<f32>,      // quaternion (x, y, z, w)
    scale: vec4<f32>,         // xyz scale + padding
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
    @location(2) local_pos: vec2<f32>,  // Position within the quad (-1 to 1)
}

// Build rotation matrix from quaternion (x, y, z, w)
fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    let x = q.x;
    let y = q.y;
    let z = q.z;
    let w = q.w;
    
    let x2 = x + x;
    let y2 = y + y;
    let z2 = z + z;
    
    let xx = x * x2;
    let xy = x * y2;
    let xz = x * z2;
    let yy = y * y2;
    let yz = y * z2;
    let zz = z * z2;
    let wx = w * x2;
    let wy = w * y2;
    let wz = w * z2;
    
    // Column-major rotation matrix
    return mat3x3<f32>(
        vec3<f32>(1.0 - (yy + zz), xy + wz, xz - wy),
        vec3<f32>(xy - wz, 1.0 - (xx + zz), yz + wx),
        vec3<f32>(xz + wy, yz - wx, 1.0 - (xx + yy))
    );
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Each Gaussian is rendered as a triangle (3 vertices)
    let gaussian_idx = vertex_index / 3u;
    let corner_idx = vertex_index % 3u;
    
    // Bounds check
    if (gaussian_idx >= arrayLength(&gaussians)) {
        out.position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.opacity = 0.0;
        return out;
    }
    
    let gaussian = gaussians[gaussian_idx];
    let center = gaussian.position;
    let scale = gaussian.scale.xyz;
    let rotation = gaussian.rotation;
    
    // Transform center to view space
    let center_view = (camera.view_matrix * vec4<f32>(center, 1.0)).xyz;
    
    // Skip if behind camera
    if (center_view.z >= -0.1) {
        out.position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.opacity = 0.0;
        return out;
    }
    
    // Triangle corners that cover a quad (using one big triangle)
    var local_pos: vec2<f32>;
    switch (corner_idx) {
        case 0u: { local_pos = vec2<f32>(-1.0, -1.0); }
        case 1u: { local_pos = vec2<f32>(3.0, -1.0); }
        case 2u: { local_pos = vec2<f32>(-1.0, 3.0); }
        default: { local_pos = vec2<f32>(0.0, 0.0); }
    }
    
    // Build rotation matrix from quaternion
    let R = quat_to_mat3(rotation);
    
    // Get the rotated axes of the Gaussian ellipsoid
    // These are the columns of the rotation matrix, scaled
    let axis_x = R[0] * scale.x * 3.0;  // 3-sigma coverage
    let axis_y = R[1] * scale.y * 3.0;
    
    // Project these 3D axes to view space to get 2D ellipse axes
    let view_rot = mat3x3<f32>(
        camera.view_matrix[0].xyz,
        camera.view_matrix[1].xyz,
        camera.view_matrix[2].xyz
    );
    
    let axis_x_view = view_rot * axis_x;
    let axis_y_view = view_rot * axis_y;
    
    // Use only the x,y components for the screen-space offset
    // (billboard in view space)
    let screen_axis_x = vec2<f32>(axis_x_view.x, axis_x_view.y);
    let screen_axis_y = vec2<f32>(axis_y_view.x, axis_y_view.y);
    
    // Compute the offset in view space
    let offset_view = vec3<f32>(
        local_pos.x * screen_axis_x.x + local_pos.y * screen_axis_y.x,
        local_pos.x * screen_axis_x.y + local_pos.y * screen_axis_y.y,
        0.0  // Keep in the view plane
    );
    
    // Add offset to center in view space
    let vertex_view = center_view + offset_view;
    
    // Project to clip space
    let clip_pos = camera.proj_matrix * vec4<f32>(vertex_view, 1.0);
    
    // Output
    out.position = clip_pos;
    out.color = gaussian.color_opacity.rgb;
    out.opacity = gaussian.color_opacity.a;
    out.local_pos = local_pos;
    
    return out;
}
