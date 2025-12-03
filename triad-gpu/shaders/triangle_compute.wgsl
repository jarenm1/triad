// Triangle Splatting+ compute shader
// Computes depth values for triangle sorting

struct TrianglePrimitive {
    v0: vec4<f32>,      // xyz position + padding
    v1: vec4<f32>,      // xyz position + padding
    v2: vec4<f32>,      // xyz position + padding
    color: vec4<f32>,   // rgb color + opacity
}

struct SortData {
    depth: f32,
    index: u32,
}

struct CameraUniforms {
    view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    view_pos: vec3<f32>,
    _padding: f32,
}

@group(0) @binding(0) var<storage, read> triangles: array<TrianglePrimitive>;
@group(0) @binding(1) var<storage, read_write> sort_data: array<SortData>;
@group(0) @binding(2) var<uniform> camera: CameraUniforms;

// Compute triangle center in world space
fn triangle_center(tri: TrianglePrimitive) -> vec3<f32> {
    return (tri.v0.xyz + tri.v1.xyz + tri.v2.xyz) / 3.0;
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Bounds check
    if (idx >= arrayLength(&triangles) || idx >= arrayLength(&sort_data)) {
        return;
    }
    
    let tri = triangles[idx];
    
    // Compute triangle center
    let center = triangle_center(tri);
    
    // Transform to view space and extract depth
    let view_pos = camera.view_matrix * vec4<f32>(center, 1.0);
    let depth = view_pos.z; // Negative Z is forward in view space
    
    // Store depth and index for sorting
    sort_data[idx] = SortData(depth, idx);
}

