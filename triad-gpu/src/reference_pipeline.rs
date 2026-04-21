use crate::{
    BindingType, BufferUsage, ColorLoadOp, DispatchIndirectArgs, DrawIndirectArgs, FrameGraph,
    Renderer, ResourceRegistry,
};
use pollster::FutureExt;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ParticleState {
    position: [f32; 2],
    velocity: [f32; 2],
    alive: u32,
    _pad: [u32; 3],
}

const SIMULATE_SHADER: &str = r#"
struct ParticleState {
    position: vec2<f32>,
    velocity: vec2<f32>,
    alive: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct ParticleBuffer {
    particles: array<ParticleState>,
};

@group(0) @binding(0) var<storage, read_write> particles: ParticleBuffer;

@compute @workgroup_size(1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= arrayLength(&particles.particles)) {
        return;
    }

    particles.particles[index].position += particles.particles[index].velocity;
}
"#;

const CULL_SHADER: &str = r#"
struct ParticleState {
    position: vec2<f32>,
    velocity: vec2<f32>,
    alive: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct ParticleBuffer {
    particles: array<ParticleState>,
};

struct VisibleIds {
    ids: array<u32>,
};

struct DrawArgs {
    vertex_count: u32,
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0) var<storage, read> particles: ParticleBuffer;
@group(0) @binding(1) var<storage, read_write> visible: VisibleIds;
@group(0) @binding(2) var<storage, read_write> draw_args: DrawArgs;

@compute @workgroup_size(1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= arrayLength(&particles.particles)) {
        return;
    }

    if (particles.particles[index].alive == 0u) {
        return;
    }

    let write_index = atomicAdd(&draw_args.instance_count, 1u);
    visible.ids[write_index] = index;
}
"#;

const RENDER_SHADER: &str = r#"
struct ParticleState {
    position: vec2<f32>,
    velocity: vec2<f32>,
    alive: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct ParticleBuffer {
    particles: array<ParticleState>,
};

struct VisibleIds {
    ids: array<u32>,
};

@group(0) @binding(0) var<storage, read> particles: ParticleBuffer;
@group(0) @binding(1) var<storage, read> visible: VisibleIds;

struct VsOut {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(instance_index) instance_index: u32) -> VsOut {
    let particle_id = visible.ids[instance_index];
    let particle = particles.particles[particle_id];

    var out: VsOut;
    out.position = vec4<f32>(particle.position, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.95, 0.75, 0.2, 1.0);
}
"#;

fn create_target_view(
    renderer: &Renderer,
    registry: &mut ResourceRegistry,
) -> crate::Handle<wgpu::TextureView> {
    let texture = renderer.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("reference target"),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    registry.insert(texture);
    registry.insert(view)
}

#[test]
fn test_reference_simulate_compact_render_pipeline() {
    let renderer = match Renderer::new().block_on() {
        Ok(renderer) => renderer,
        Err(err) => {
            eprintln!("skipping reference pipeline test: {err}");
            return;
        }
    };
    let mut registry = ResourceRegistry::default();

    let particles = [
        ParticleState {
            position: [-0.75, -0.75],
            velocity: [0.1, 0.05],
            alive: 1,
            _pad: [0; 3],
        },
        ParticleState {
            position: [-0.25, 0.0],
            velocity: [0.05, -0.05],
            alive: 1,
            _pad: [0; 3],
        },
        ParticleState {
            position: [0.25, 0.25],
            velocity: [-0.03, 0.02],
            alive: 1,
            _pad: [0; 3],
        },
        ParticleState {
            position: [0.7, -0.2],
            velocity: [-0.04, 0.03],
            alive: 0,
            _pad: [0; 3],
        },
    ];

    let particle_buffer = renderer
        .create_gpu_buffer::<ParticleState>()
        .label("particles")
        .with_data(&particles)
        .build(&mut registry)
        .expect("particle buffer");

    let visible_ids = renderer
        .create_gpu_buffer::<u32>()
        .label("visible ids")
        .capacity(particles.len())
        .build(&mut registry)
        .expect("visible ids buffer");

    let dispatch_args = renderer
        .create_gpu_buffer::<DispatchIndirectArgs>()
        .label("dispatch args")
        .with_data(&[DispatchIndirectArgs::new(particles.len() as u32, 1, 1)])
        .usage(BufferUsage::Indirect)
        .build(&mut registry)
        .expect("dispatch args");

    let draw_args = renderer
        .create_gpu_buffer::<DrawIndirectArgs>()
        .label("draw args")
        .with_data(&[DrawIndirectArgs::new(1, 0, 0, 0)])
        .usage(BufferUsage::Indirect)
        .add_usage(wgpu::BufferUsages::STORAGE)
        .build(&mut registry)
        .expect("draw args");

    let simulate_shader = renderer
        .create_shader_module()
        .label("simulate shader")
        .with_wgsl_source(SIMULATE_SHADER)
        .build(&mut registry)
        .expect("simulate shader");

    let cull_shader = renderer
        .create_shader_module()
        .label("cull shader")
        .with_wgsl_source(CULL_SHADER)
        .build(&mut registry)
        .expect("cull shader");

    let render_shader = renderer
        .create_shader_module()
        .label("render shader")
        .with_wgsl_source(RENDER_SHADER)
        .build(&mut registry)
        .expect("render shader");

    let (simulate_layout, simulate_bind_group) = renderer
        .create_bind_group()
        .label("simulate")
        .buffer(0, particle_buffer.handle(), BindingType::StorageWrite)
        .build(&mut registry)
        .expect("simulate bind group");

    let (cull_layout, cull_bind_group) = renderer
        .create_bind_group()
        .label("cull")
        .buffer(0, particle_buffer.handle(), BindingType::StorageRead)
        .buffer(1, visible_ids.handle(), BindingType::StorageWrite)
        .buffer(2, draw_args.handle(), BindingType::StorageWrite)
        .build(&mut registry)
        .expect("cull bind group");

    let (render_layout, render_bind_group) = renderer
        .create_bind_group()
        .label("render")
        .buffer(0, particle_buffer.handle(), BindingType::StorageRead)
        .buffer(1, visible_ids.handle(), BindingType::StorageRead)
        .build(&mut registry)
        .expect("render bind group");

    let simulate_pipeline_layout =
        renderer
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("simulate pipeline layout"),
                bind_group_layouts: &[registry
                    .get(simulate_layout)
                    .expect("simulate layout in registry")],
                push_constant_ranges: &[],
            });

    let cull_pipeline_layout =
        renderer
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("cull pipeline layout"),
                bind_group_layouts: &[registry.get(cull_layout).expect("cull layout in registry")],
                push_constant_ranges: &[],
            });

    let render_pipeline_layout =
        renderer
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("render pipeline layout"),
                bind_group_layouts: &[registry
                    .get(render_layout)
                    .expect("render layout in registry")],
                push_constant_ranges: &[],
            });

    let simulate_pipeline = renderer
        .create_compute_pipeline()
        .with_label("simulate pipeline")
        .with_compute_shader(simulate_shader)
        .with_layout(simulate_pipeline_layout)
        .build(&mut registry)
        .expect("simulate pipeline");

    let cull_pipeline = renderer
        .create_compute_pipeline()
        .with_label("cull pipeline")
        .with_compute_shader(cull_shader)
        .with_layout(cull_pipeline_layout)
        .build(&mut registry)
        .expect("cull pipeline");

    let render_pipeline = renderer
        .create_render_pipeline()
        .with_label("render pipeline")
        .with_vertex_shader(render_shader)
        .with_fragment_shader(render_shader)
        .with_layout(render_pipeline_layout)
        .with_primitive(wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::PointList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        })
        .with_fragment_target(Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba8Unorm,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        }))
        .build(&mut registry)
        .expect("render pipeline");

    let target_view = create_target_view(&renderer, &mut registry);

    let simulate_pass = renderer
        .create_compute_pass("simulate")
        .read(dispatch_args.handle())
        .read_write(particle_buffer.handle())
        .with_pipeline(simulate_pipeline)
        .with_bind_group(0, simulate_bind_group)
        .dispatch_indirect(dispatch_args.handle(), 0)
        .build()
        .expect("simulate pass");

    let cull_pass = renderer
        .create_compute_pass("compact")
        .read(dispatch_args.handle())
        .read(particle_buffer.handle())
        .write(visible_ids.handle())
        .read_write(draw_args.handle())
        .with_pipeline(cull_pipeline)
        .with_bind_group(0, cull_bind_group)
        .dispatch_indirect(dispatch_args.handle(), 0)
        .build()
        .expect("cull pass");

    let render_pass = renderer
        .create_render_pass("render")
        .read(particle_buffer.handle())
        .read(visible_ids.handle())
        .with_pipeline(render_pipeline)
        .with_bind_group(0, render_bind_group)
        .with_color_attachment(target_view, ColorLoadOp::Clear(wgpu::Color::BLACK))
        .draw_indirect(draw_args.handle(), 0)
        .build()
        .expect("render pass");

    let mut graph = FrameGraph::new();
    graph.add_pass(simulate_pass);
    graph.add_pass(cull_pass);
    graph.add_pass(render_pass);

    let mut executable = graph.build().expect("frame graph");
    let command_buffers =
        executable.execute_no_submit(renderer.device(), renderer.queue(), &registry);
    assert_eq!(command_buffers.len(), 3);

    renderer.queue().submit(command_buffers);
}
