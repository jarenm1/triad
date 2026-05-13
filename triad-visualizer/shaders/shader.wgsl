struct CameraUniforms {
    view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    view_pos: vec3<f32>,
    _pad: f32,
};

struct RenderInstance {
    center: vec4<f32>,
    axis_x: vec4<f32>,
    axis_y: vec4<f32>,
    axis_z: vec4<f32>,
    half_extents: vec4<f32>,
    color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) light: f32,
};

@group(0) @binding(0) var<uniform> camera_u: CameraUniforms;
@group(0) @binding(1) var<storage, read> instances: array<RenderInstance>;

const POSITIONS: array<vec3<f32>, 36> = array<vec3<f32>, 36>(
    vec3<f32>(-1.0, -1.0,  1.0), vec3<f32>( 1.0, -1.0,  1.0), vec3<f32>( 1.0,  1.0,  1.0),
    vec3<f32>(-1.0, -1.0,  1.0), vec3<f32>( 1.0,  1.0,  1.0), vec3<f32>(-1.0,  1.0,  1.0),
    vec3<f32>( 1.0, -1.0, -1.0), vec3<f32>(-1.0, -1.0, -1.0), vec3<f32>(-1.0,  1.0, -1.0),
    vec3<f32>( 1.0, -1.0, -1.0), vec3<f32>(-1.0,  1.0, -1.0), vec3<f32>( 1.0,  1.0, -1.0),
    vec3<f32>(-1.0, -1.0, -1.0), vec3<f32>(-1.0, -1.0,  1.0), vec3<f32>(-1.0,  1.0,  1.0),
    vec3<f32>(-1.0, -1.0, -1.0), vec3<f32>(-1.0,  1.0,  1.0), vec3<f32>(-1.0,  1.0, -1.0),
    vec3<f32>( 1.0, -1.0,  1.0), vec3<f32>( 1.0, -1.0, -1.0), vec3<f32>( 1.0,  1.0, -1.0),
    vec3<f32>( 1.0, -1.0,  1.0), vec3<f32>( 1.0,  1.0, -1.0), vec3<f32>( 1.0,  1.0,  1.0),
    vec3<f32>(-1.0,  1.0,  1.0), vec3<f32>( 1.0,  1.0,  1.0), vec3<f32>( 1.0,  1.0, -1.0),
    vec3<f32>(-1.0,  1.0,  1.0), vec3<f32>( 1.0,  1.0, -1.0), vec3<f32>(-1.0,  1.0, -1.0),
    vec3<f32>(-1.0, -1.0, -1.0), vec3<f32>( 1.0, -1.0, -1.0), vec3<f32>( 1.0, -1.0,  1.0),
    vec3<f32>(-1.0, -1.0, -1.0), vec3<f32>( 1.0, -1.0,  1.0), vec3<f32>(-1.0, -1.0,  1.0)
);

const NORMALS: array<vec3<f32>, 36> = array<vec3<f32>, 36>(
    vec3<f32>( 0.0,  0.0,  1.0), vec3<f32>( 0.0,  0.0,  1.0), vec3<f32>( 0.0,  0.0,  1.0),
    vec3<f32>( 0.0,  0.0,  1.0), vec3<f32>( 0.0,  0.0,  1.0), vec3<f32>( 0.0,  0.0,  1.0),
    vec3<f32>( 0.0,  0.0, -1.0), vec3<f32>( 0.0,  0.0, -1.0), vec3<f32>( 0.0,  0.0, -1.0),
    vec3<f32>( 0.0,  0.0, -1.0), vec3<f32>( 0.0,  0.0, -1.0), vec3<f32>( 0.0,  0.0, -1.0),
    vec3<f32>(-1.0,  0.0,  0.0), vec3<f32>(-1.0,  0.0,  0.0), vec3<f32>(-1.0,  0.0,  0.0),
    vec3<f32>(-1.0,  0.0,  0.0), vec3<f32>(-1.0,  0.0,  0.0), vec3<f32>(-1.0,  0.0,  0.0),
    vec3<f32>( 1.0,  0.0,  0.0), vec3<f32>( 1.0,  0.0,  0.0), vec3<f32>( 1.0,  0.0,  0.0),
    vec3<f32>( 1.0,  0.0,  0.0), vec3<f32>( 1.0,  0.0,  0.0), vec3<f32>( 1.0,  0.0,  0.0),
    vec3<f32>( 0.0,  1.0,  0.0), vec3<f32>( 0.0,  1.0,  0.0), vec3<f32>( 0.0,  1.0,  0.0),
    vec3<f32>( 0.0,  1.0,  0.0), vec3<f32>( 0.0,  1.0,  0.0), vec3<f32>( 0.0,  1.0,  0.0),
    vec3<f32>( 0.0, -1.0,  0.0), vec3<f32>( 0.0, -1.0,  0.0), vec3<f32>( 0.0, -1.0,  0.0),
    vec3<f32>( 0.0, -1.0,  0.0), vec3<f32>( 0.0, -1.0,  0.0), vec3<f32>( 0.0, -1.0,  0.0)
);

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VsOut {
    let instance = instances[instance_index];
    let local_position = POSITIONS[vertex_index];
    let local_normal = NORMALS[vertex_index];
    let world_position =
        instance.center.xyz
        + instance.axis_x.xyz * (local_position.x * instance.half_extents.x)
        + instance.axis_y.xyz * (local_position.y * instance.half_extents.y)
        + instance.axis_z.xyz * (local_position.z * instance.half_extents.z);
    let world_normal = normalize(
        instance.axis_x.xyz * local_normal.x
        + instance.axis_y.xyz * local_normal.y
        + instance.axis_z.xyz * local_normal.z
    );
    let light_dir = normalize(vec3<f32>(0.35, 0.55, 0.75));

    var out: VsOut;
    out.clip_pos =
        camera_u.proj_matrix * camera_u.view_matrix * vec4<f32>(world_position, 1.0);
    out.color = instance.color;
    out.light = max(dot(world_normal, light_dir), 0.0);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let lit = 0.3 + 0.7 * in.light;
    return vec4<f32>(in.color.rgb * lit, in.color.a);
}
