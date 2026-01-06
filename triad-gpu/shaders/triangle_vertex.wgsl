// Triangle vertex shader
// Simple triangle rendering without edge falloff

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
        out.color = vec3<f32>(0.0, 0.0, 0.0);
        out.opacity = 0.0;
        return out;
    }

    let tri = triangles[triangle_idx];

    // Select vertex position based on corner
    var world_pos: vec3<f32>;

    switch (corner_idx) {
        case 0u: {
            world_pos = tri.v0.xyz;
        }
        case 1u: {
            world_pos = tri.v1.xyz;
        }
        case 2u: {
            world_pos = tri.v2.xyz;
        }
        default: {
            world_pos = vec3<f32>(0.0, 0.0, 0.0);
        }
    }

    // Transform to clip space
    let view_pos = camera.view_matrix * vec4<f32>(world_pos, 1.0);
    let clip_pos = camera.proj_matrix * view_pos;

    // Output
    out.position = clip_pos;
    out.color = tri.color.rgb;
    out.opacity = tri.color.a;

    return out;
}

