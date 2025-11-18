// Gaussian Splatting Compute Shader for Depth Sorting
// Sorts Gaussians by depth for correct alpha blending order

struct GaussianPoint {
    position: vec3<f32>,
    scale: vec3<f32>,
    rotation: vec4<f32>,
    color: vec3<f32>,
    opacity: f32,
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

@group(0) @binding(0) var<storage, read> gaussians: array<GaussianPoint>;
@group(0) @binding(1) var<storage, read_write> sort_data: array<SortData>;
@group(0) @binding(2) var<uniform> camera: CameraUniforms;

// Simple depth computation for sorting
// In production, you'd use a more sophisticated sorting algorithm
@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= arrayLength(&gaussians)) {
        return;
    }
    
    let gaussian = gaussians[idx];
    
    // Transform position to view space and compute depth
    let view_pos = camera.view_matrix * vec4<f32>(gaussian.position, 1.0);
    let depth = view_pos.z; // Negative Z is forward in view space
    
    sort_data[idx] = SortData(depth, idx);
}

