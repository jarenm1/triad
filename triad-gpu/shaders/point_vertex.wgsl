// Point cloud vertex shader
// Renders points as screen-aligned quads (point sprites)

struct PointPrimitive {
    position: vec3<f32>,  // xyz position
    size: f32,            // point size (world units)
    color: vec4<f32>,     // rgba color
}

struct CameraUniforms {
    view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    view_pos: vec3<f32>,
    _padding: f32,
}

@group(0) @binding(0) var<storage, read> points: array<PointPrimitive>;
@group(0) @binding(1) var<uniform> camera: CameraUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) opacity: f32,
    @location(2) local_pos: vec2<f32>,  // Position within the quad (-1 to 1)
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Each point is rendered as a triangle (3 vertices) covering a quad
    let point_idx = vertex_index / 3u;
    let corner_idx = vertex_index % 3u;
    
    // Bounds check
    if (point_idx >= arrayLength(&points)) {
        out.position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.opacity = 0.0;
        return out;
    }
    
    let point = points[point_idx];
    let center = point.position;
    let size = point.size;
    
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
    
    // Compute screen-space offset (billboard)
    // Scale the offset based on point size
    let offset_view = vec3<f32>(
        local_pos.x * size,
        local_pos.y * size,
        0.0  // Keep in the view plane
    );
    
    // Add offset to center in view space
    let vertex_view = center_view + offset_view;
    
    // Project to clip space
    let clip_pos = camera.proj_matrix * vec4<f32>(vertex_view, 1.0);
    
    // Output
    out.position = clip_pos;
    out.color = point.color.rgb;
    out.opacity = point.color.a;
    out.local_pos = local_pos;
    
    return out;
}

