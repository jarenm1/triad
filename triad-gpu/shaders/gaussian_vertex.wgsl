// Gaussian spatting render vertex shader.
struct GaussianPoint {
    position: vec3<f32>,      // xyz
    color_opacity: vec4<f32>, // rgb + opacity (a)
    rotation: vec4<f32>,      // quaternion (xyzw)
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
    @location(2) center_ndc: vec2<f32>,   // Gaussian center in NDC
    @location(3) pixel_ndc: vec2<f32>,    // This vertex corner in NDC (for frag shader)
    @location(4) cov2d_a: vec2<f32>,      // cov2d row 0 (xx, xy)
    @location(5) cov2d_b: vec2<f32>,      // cov2d row 1 (xy, yy)
    @location(6) falloff_scale: f32,      // Controls gaussian falloff steepness
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {

    return out;
}

@fragment
fn fs_main()
