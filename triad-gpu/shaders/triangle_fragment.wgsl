// Triangle Splatting+ fragment shader
// Soft edge falloff for smooth blending between overlapping triangles
// Based on the Triangle Splatting paper's approach

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) opacity: f32,
    @location(2) barycentric: vec3<f32>,
}

// Edge falloff parameters
// Wider falloff creates smoother blending between triangles
const EDGE_FALLOFF_START: f32 = 0.0;   // Start fading at edge
const EDGE_FALLOFF_END: f32 = 0.0;    // Fully opaque at this distance from edge

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Compute distance to nearest edge using barycentric coordinates
    // Each barycentric coord represents distance to the opposite edge
    let edge_dist = min(min(input.barycentric.x, input.barycentric.y), input.barycentric.z);
    
    // Smooth falloff using smoothstep - creates C1 continuous blending
    // At edge (edge_dist=0): alpha=0
    // At EDGE_FALLOFF_END: alpha=1 (fully opaque)
    let edge_alpha = smoothstep(EDGE_FALLOFF_START, EDGE_FALLOFF_END, edge_dist);
    
    // Combine with base opacity
    let final_alpha = input.opacity * edge_alpha;
    
    // Premultiplied alpha output for better blending
    return vec4<f32>(input.color * final_alpha, final_alpha);
}
