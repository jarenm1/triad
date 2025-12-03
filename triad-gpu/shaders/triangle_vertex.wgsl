// Triangle Splatting+ vertex shader
// Renders triangles with barycentric coordinates for edge falloff

struct TrianglePrimitive {
    v0: vec4<f32>,      // xyz position + padding
    v1: vec4<f32>,      // xyz position + padding
    v2: vec4<f32>,      // xyz position + padding
    color: vec4<f32>,   // rgb color + opacity
}

struct CameraUniforms {
    view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    view_pos: vec3<f32>,
    _padding: f32,
}

@group(0) @binding(0) var<storage, read> triangles: array<TrianglePrimitive>;
@group(0) @binding(1) var<uniform> camera: CameraUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) opacity: f32,
    @location(2) barycentric: vec3<f32>,  // Barycentric coordinates for edge falloff
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Each triangle has 3 vertices
    let triangle_idx = vertex_index / 3u;
    let corner_idx = vertex_index % 3u;
    
    // Bounds check
    if (triangle_idx >= arrayLength(&triangles)) {
        out.position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.opacity = 0.0;
        out.barycentric = vec3<f32>(0.0, 0.0, 0.0);
        return out;
    }
    
    let tri = triangles[triangle_idx];
    
    // Select vertex position and barycentric coordinate based on corner
    var world_pos: vec3<f32>;
    var bary: vec3<f32>;
    
    switch (corner_idx) {
        case 0u: {
            world_pos = tri.v0.xyz;
            bary = vec3<f32>(1.0, 0.0, 0.0);
        }
        case 1u: {
            world_pos = tri.v1.xyz;
            bary = vec3<f32>(0.0, 1.0, 0.0);
        }
        case 2u: {
            world_pos = tri.v2.xyz;
            bary = vec3<f32>(0.0, 0.0, 1.0);
        }
        default: {
            world_pos = vec3<f32>(0.0, 0.0, 0.0);
            bary = vec3<f32>(0.0, 0.0, 0.0);
        }
    }
    
    // Transform to clip space
    let view_pos = camera.view_matrix * vec4<f32>(world_pos, 1.0);
    let clip_pos = camera.proj_matrix * view_pos;
    
    // Output
    out.position = clip_pos;
    out.color = tri.color.rgb;
    out.opacity = tri.color.a;
    out.barycentric = bary;
    
    return out;
}

