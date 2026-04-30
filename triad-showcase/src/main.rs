use std::collections::VecDeque;
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use glam::{Vec3, vec3};
use tracing::info;
use triad_gpu::{
    BindingType, BufferUsage, ColorLoadOp, DepthLoadOp, ExecutableFrameGraph, FrameGraphError,
    FrameTextureView, RenderPassBuilder, Renderer, ResourceRegistry, ShaderStage, wgpu,
};
use triad_window::{CameraPose, CameraUniforms, RendererManager, WindowConfig, egui, run_with_renderer_config};
#[cfg(target_arch = "wasm32")]
use triad_window::{WebWindowConfig, run_with_renderer_config_web};

const WINDOW_TITLE: &str = "Triad Showcase";
const FLOOR_ALTITUDE: f32 = 0.1;
const FLOOR_HALF_THICKNESS: f32 = 0.02;
const GATE_DEPTH_HALF: f32 = 0.04;
const GATE_FRAME_THICKNESS: f32 = 0.08;
const DRONE_CORE_HALF_EXTENTS: [f32; 3] = [0.06, 0.025, 0.035];
const DRONE_ARM_HALF_EXTENTS: [f32; 3] = [0.14, 0.015, 0.015];
const DRONE_MOTOR_HALF_EXTENTS: [f32; 3] = [0.028, 0.018, 0.028];
const DRONE_ARM_OFFSET: f32 = 0.11;
const DRONE_MOTOR_OFFSET: f32 = 0.22;
const DRONE_MODEL_INSTANCE_COUNT: usize = 9;
const TARGET_HALF_EXTENTS: [f32; 3] = [0.05, 0.05, 0.05];
const TRAIL_MAX_POINTS: usize = 96;
const TRAIL_INSTANCE_COUNT: usize = TRAIL_MAX_POINTS - 1;
const TRAIL_WINDOW_SECONDS: f32 = 5.5;
const TRAIL_HALF_WIDTH: f32 = 0.012;
const TRAIL_HALF_HEIGHT: f32 = 0.006;
const DEBUG_VECTOR_INSTANCE_COUNT: usize = 3;
const DEBUG_VECTOR_HALF_WIDTH: f32 = 0.014;
const DEBUG_VECTOR_HALF_HEIGHT: f32 = 0.007;
const FORWARD_VECTOR_LENGTH: f32 = 0.55;
const THRUST_VECTOR_LENGTH: f32 = 0.55;
const VELOCITY_VECTOR_SCALE: f32 = 0.22;
const VELOCITY_VECTOR_MAX_LENGTH: f32 = 0.9;

const SHOWCASE_SHADER: &str = r#"
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
"#;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RenderInstance {
    center: [f32; 4],
    axis_x: [f32; 4],
    axis_y: [f32; 4],
    axis_z: [f32; 4],
    half_extents: [f32; 4],
    color: [f32; 4],
}

impl RenderInstance {
    fn hidden() -> Self {
        Self::oriented_box(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        )
    }

    fn oriented_box(
        center: [f32; 3],
        axis_x: [f32; 3],
        axis_y: [f32; 3],
        axis_z: [f32; 3],
        half_extents: [f32; 3],
        color: [f32; 4],
    ) -> Self {
        Self {
            center: [center[0], center[1], center[2], 0.0],
            axis_x: [axis_x[0], axis_x[1], axis_x[2], 0.0],
            axis_y: [axis_y[0], axis_y[1], axis_y[2], 0.0],
            axis_z: [axis_z[0], axis_z[1], axis_z[2], 0.0],
            half_extents: [half_extents[0], half_extents[1], half_extents[2], 0.0],
            color,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ShowcaseGate {
    center: [f32; 3],
    half_extents: [f32; 3],
    forward: [f32; 3],
}

#[derive(Clone, Copy, Debug)]
struct DronePose {
    position: [f32; 3],
    velocity: [f32; 3],
    attitude: [f32; 3],
}

#[derive(Clone, Copy, Debug)]
struct Keyframe {
    time_seconds: f32,
    pose: DronePose,
}

#[derive(Debug)]
struct ShowcaseClip {
    name: &'static str,
    duration_seconds: f32,
    bounds: f32,
    gates: Vec<ShowcaseGate>,
    gate_focus_times: Vec<f32>,
    keyframes: Vec<Keyframe>,
}

impl ShowcaseClip {
    fn sample_pose(&self, time_seconds: f32) -> DronePose {
        let time_seconds = time_seconds.clamp(0.0, self.duration_seconds);
        let Some(first) = self.keyframes.first().copied() else {
            return DronePose::default();
        };
        if time_seconds <= first.time_seconds {
            return first.pose;
        }

        for pair in self.keyframes.windows(2) {
            let a = pair[0];
            let b = pair[1];
            if time_seconds <= b.time_seconds {
                let span = (b.time_seconds - a.time_seconds).max(1.0e-6);
                let t = ((time_seconds - a.time_seconds) / span).clamp(0.0, 1.0);
                return DronePose {
                    position: lerp3(a.pose.position, b.pose.position, t),
                    velocity: lerp3(a.pose.velocity, b.pose.velocity, t),
                    attitude: [
                        lerp_angle(a.pose.attitude[0], b.pose.attitude[0], t),
                        lerp_angle(a.pose.attitude[1], b.pose.attitude[1], t),
                        lerp_angle(a.pose.attitude[2], b.pose.attitude[2], t),
                    ],
                };
            }
        }

        self.keyframes.last().copied().unwrap_or(first).pose
    }

    fn active_gate_index(&self, time_seconds: f32) -> usize {
        self.gate_focus_times
            .iter()
            .position(|&focus_time| time_seconds <= focus_time)
            .unwrap_or_else(|| self.gates.len().saturating_sub(1))
    }

    fn trail_points(&self, time_seconds: f32) -> VecDeque<[f32; 3]> {
        let mut points = VecDeque::with_capacity(TRAIL_MAX_POINTS);
        if self.duration_seconds <= 0.0 {
            return points;
        }
        for index in 0..TRAIL_MAX_POINTS {
            let fraction = index as f32 / TRAIL_INSTANCE_COUNT.max(1) as f32;
            let sample_time = (time_seconds - (1.0 - fraction) * TRAIL_WINDOW_SECONDS)
                .rem_euclid(self.duration_seconds);
            points.push_back(self.sample_pose(sample_time).position);
        }
        points
    }
}

impl Default for DronePose {
    fn default() -> Self {
        Self {
            position: [0.0, 1.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
            attitude: [0.0, 0.0, 0.0],
        }
    }
}

#[derive(Debug)]
struct UiState {
    playing: bool,
    speed: f32,
    seek_fraction: f32,
    request_seek: bool,
    request_restart: bool,
    clip_name: String,
    current_gate: usize,
    gate_count: usize,
    time_seconds: f32,
    duration_seconds: f32,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            playing: true,
            speed: 1.0,
            seek_fraction: 0.0,
            request_seek: false,
            request_restart: false,
            clip_name: "Dock Sweep".to_string(),
            current_gate: 1,
            gate_count: 0,
            time_seconds: 0.0,
            duration_seconds: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
struct UiSnapshot {
    playing: bool,
    speed: f32,
    seek_fraction: f32,
    request_seek: bool,
    request_restart: bool,
}

struct ShowcaseManager {
    ui_state: Arc<Mutex<UiState>>,
    camera_buffer: triad_gpu::Handle<wgpu::Buffer>,
    instance_buffer: triad_gpu::Handle<wgpu::Buffer>,
    render_bind_group: triad_gpu::Handle<wgpu::BindGroup>,
    render_pipeline: triad_gpu::Handle<wgpu::RenderPipeline>,
    frame_target: triad_gpu::Handle<FrameTextureView>,
    depth_frame: triad_gpu::Handle<FrameTextureView>,
    clip: ShowcaseClip,
    instances: Vec<RenderInstance>,
    time_seconds: f32,
    last_update: Instant,
}

impl ShowcaseManager {
    fn new(
        renderer: &Renderer,
        registry: &mut ResourceRegistry,
        surface_format: wgpu::TextureFormat,
        ui_state: Arc<Mutex<UiState>>,
    ) -> Result<Self, Box<dyn Error>> {
        let clip = build_showcase_clip();
        let max_instances = visible_instance_capacity(clip.gates.len());
        let hidden_instances = vec![RenderInstance::hidden(); max_instances];

        let camera_buffer = renderer
            .create_gpu_buffer::<CameraUniforms>()
            .label("showcase camera")
            .with_data(&[CameraUniforms::from_matrices(
                glam::Mat4::IDENTITY,
                glam::Mat4::IDENTITY,
                Vec3::ZERO,
            )])
            .usage(BufferUsage::Uniform)
            .build(registry)?;

        let instance_buffer = renderer
            .create_gpu_buffer::<RenderInstance>()
            .label("showcase instances")
            .with_data(&hidden_instances)
            .build(registry)?;

        let shader = renderer
            .create_shader_module()
            .label("showcase boxes")
            .with_wgsl_source(SHOWCASE_SHADER)
            .build(registry)?;

        let (render_layout, render_bind_group) = renderer
            .create_bind_group()
            .label("showcase render")
            .buffer_stage(
                0,
                ShaderStage::Vertex,
                camera_buffer.handle(),
                BindingType::Uniform,
            )
            .buffer_stage(
                1,
                ShaderStage::Vertex,
                instance_buffer.handle(),
                BindingType::StorageRead,
            )
            .build(registry)?;

        let render_pipeline_layout =
            renderer
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("showcase render layout"),
                    bind_group_layouts: &[registry
                        .get(render_layout)
                        .expect("showcase render layout should exist")],
                    push_constant_ranges: &[],
                });

        let render_pipeline = renderer
            .create_render_pipeline()
            .with_label("showcase render pipeline")
            .with_vertex_shader(shader)
            .with_fragment_shader(shader)
            .with_layout(render_pipeline_layout)
            .with_primitive(wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            })
            .with_fragment_target(Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            }))
            .with_depth_stencil(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            })
            .build(registry)?;

        let manager = Self {
            ui_state,
            camera_buffer: camera_buffer.handle(),
            instance_buffer: instance_buffer.handle(),
            render_bind_group,
            render_pipeline,
            frame_target: registry.insert(FrameTextureView::new()),
            depth_frame: registry.insert(FrameTextureView::new()),
            clip,
            instances: hidden_instances,
            time_seconds: 0.0,
            last_update: Instant::now(),
        };
        manager.publish_ui(0.0);
        Ok(manager)
    }

    fn snapshot_ui(&self) -> UiSnapshot {
        let mut ui = self.ui_state.lock().expect("ui state poisoned");
        let snapshot = UiSnapshot {
            playing: ui.playing,
            speed: ui.speed,
            seek_fraction: ui.seek_fraction,
            request_seek: ui.request_seek,
            request_restart: ui.request_restart,
        };
        ui.request_seek = false;
        ui.request_restart = false;
        snapshot
    }

    fn publish_ui(&self, time_seconds: f32) {
        let mut ui = self.ui_state.lock().expect("ui state poisoned");
        ui.clip_name = self.clip.name.to_string();
        ui.current_gate = self.clip.active_gate_index(time_seconds) + 1;
        ui.gate_count = self.clip.gates.len();
        ui.time_seconds = time_seconds;
        ui.duration_seconds = self.clip.duration_seconds;
        ui.seek_fraction = if self.clip.duration_seconds > 0.0 {
            (time_seconds / self.clip.duration_seconds).clamp(0.0, 1.0)
        } else {
            0.0
        };
    }

    fn rebuild_instances(&mut self, pose: DronePose) {
        self.instances.clear();
        self.instances.push(floor_instance(self.clip.bounds));
        for gate in &self.clip.gates {
            self.instances.extend(gate_bar_instances(*gate));
        }
        let active_gate = self.clip.active_gate_index(self.time_seconds);
        if let Some(gate) = self.clip.gates.get(active_gate).copied() {
            self.instances.push(target_instance(gate));
        }
        self.instances.extend(drone_instances(pose));
        let trail = self.clip.trail_points(self.time_seconds);
        self.instances.extend(trail_instances(&trail));
        self.instances.extend(debug_vector_instances(pose));
    }
}

impl RendererManager for ShowcaseManager {
    fn update(
        &mut self,
        renderer: &Renderer,
        registry: &mut ResourceRegistry,
        camera: &CameraUniforms,
    ) -> Result<(), Box<dyn Error>> {
        renderer.write_buffer(self.camera_buffer, std::slice::from_ref(camera), registry)?;

        let snapshot = self.snapshot_ui();
        let now = Instant::now();
        let dt = (now - self.last_update).as_secs_f32().clamp(1.0 / 240.0, 0.1);
        self.last_update = now;

        if snapshot.request_restart {
            self.time_seconds = 0.0;
        } else if snapshot.request_seek {
            self.time_seconds =
                snapshot.seek_fraction.clamp(0.0, 1.0) * self.clip.duration_seconds;
        } else if snapshot.playing {
            self.time_seconds = (self.time_seconds + dt * snapshot.speed.max(0.05))
                .rem_euclid(self.clip.duration_seconds.max(1.0e-6));
        }

        let pose = self.clip.sample_pose(self.time_seconds);
        self.rebuild_instances(pose);
        renderer.write_buffer(self.instance_buffer, &self.instances, registry)?;
        self.publish_ui(self.time_seconds);
        Ok(())
    }

    fn prepare_frame(
        &mut self,
        registry: &mut ResourceRegistry,
        final_view: Arc<wgpu::TextureView>,
        depth_view: Option<Arc<wgpu::TextureView>>,
    ) -> Result<bool, Box<dyn Error>> {
        registry
            .get(self.frame_target)
            .expect("showcase frame target should exist")
            .set(final_view);
        if let Some(depth) = depth_view {
            registry
                .get(self.depth_frame)
                .expect("showcase depth target should exist")
                .set(depth);
        }
        Ok(false)
    }

    fn build_frame_graph(&mut self) -> Result<ExecutableFrameGraph, FrameGraphError> {
        let render_pass = RenderPassBuilder::new("ShowcaseRender")
            .with_pipeline(self.render_pipeline)
            .with_bind_group(0, self.render_bind_group)
            .with_frame_color_attachment(
                self.frame_target,
                ColorLoadOp::Clear(wgpu::Color {
                    r: 0.07,
                    g: 0.08,
                    b: 0.11,
                    a: 1.0,
                }),
            )
            .with_frame_depth_stencil_attachment(
                self.depth_frame,
                DepthLoadOp::Clear(1.0),
                wgpu::StoreOp::Store,
                None,
            )
            .draw(36, self.instances.len() as u32)
            .build()
            .expect("showcase render pass should build");

        let mut graph = triad_gpu::FrameGraph::new();
        graph.add_pass(render_pass);
        graph.build()
    }

    fn resize(
        &mut self,
        _device: &wgpu::Device,
        _registry: &mut ResourceRegistry,
        _width: u32,
        _height: u32,
    ) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
}

fn configure_showcase_controls(
    controls: &mut triad_window::Controls,
    ui_state_for_controls: Arc<Mutex<UiState>>,
) {
    controls.request_reset(CameraPose::new(
        Vec3::new(8.5, 5.5, 10.5),
        Vec3::new(4.5, 1.5, -0.8),
    ));

    controls.on_ui(move |ctx| {
        let mut ui = ui_state_for_controls.lock().expect("ui state poisoned");
        egui::Window::new("Showcase")
            .default_pos(egui::pos2(12.0, 84.0))
            .resizable(false)
            .show(ctx, |panel| {
                panel.label(format!("Clip: {}", ui.clip_name));
                panel.label("Mouse: orbit | Shift+drag: pan | Wheel: zoom");
                panel.separator();
                panel.checkbox(&mut ui.playing, "Play");
                panel.add(egui::Slider::new(&mut ui.speed, 0.25..=2.0).text("Speed"));
                let response =
                    panel.add(egui::Slider::new(&mut ui.seek_fraction, 0.0..=1.0).text("Progress"));
                if response.changed() {
                    ui.request_seek = true;
                }
                if panel.button("Restart Clip").clicked() {
                    ui.request_restart = true;
                }
                panel.separator();
                panel.label(format!(
                    "Time: {:.1}s / {:.1}s",
                    ui.time_seconds, ui.duration_seconds
                ));
                panel.label(format!("Target Gate: {} / {}", ui.current_gate, ui.gate_count));
                panel.label("Flow: launch -> weave gates -> low sweep -> reset loop");
            });
    });
}

pub fn run_showcase_native() -> Result<(), Box<dyn Error>> {
    init_logging();
    let ui_state = Arc::new(Mutex::new(UiState::default()));
    let ui_state_for_controls = Arc::clone(&ui_state);
    let ui_state_for_manager = Arc::clone(&ui_state);

    run_with_renderer_config(
        WINDOW_TITLE,
        WindowConfig::default(),
        move |controls| configure_showcase_controls(controls, ui_state_for_controls),
        move |renderer, registry, surface_format, _width, _height| {
            Ok(Box::new(ShowcaseManager::new(
                renderer,
                registry,
                surface_format,
                ui_state_for_manager,
            )?))
        },
    )
}

#[cfg(target_arch = "wasm32")]
pub fn run_showcase_web(canvas_id: impl Into<String>) -> Result<(), Box<dyn Error>> {
    init_logging();
    let ui_state = Arc::new(Mutex::new(UiState::default()));
    let ui_state_for_controls = Arc::clone(&ui_state);
    let ui_state_for_manager = Arc::clone(&ui_state);

    run_with_renderer_config_web(
        WINDOW_TITLE,
        WindowConfig::default(),
        WebWindowConfig::new(canvas_id),
        move |controls| configure_showcase_controls(controls, ui_state_for_controls),
        move |renderer, registry, surface_format, _width, _height| {
            Ok(Box::new(ShowcaseManager::new(
                renderer,
                registry,
                surface_format,
                ui_state_for_manager,
            )?))
        },
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn Error>> {
    run_showcase_native()
}

#[cfg(target_arch = "wasm32")]
fn main() {}

fn init_logging() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "triad_showcase=info,triad_window=info".into()),
        )
        .with_target(false)
        .compact()
        .try_init();
    info!("starting showcase");
}

fn visible_instance_capacity(gate_count: usize) -> usize {
    1 + gate_count * 4 + 1 + DRONE_MODEL_INSTANCE_COUNT + TRAIL_INSTANCE_COUNT + DEBUG_VECTOR_INSTANCE_COUNT
}

fn build_showcase_clip() -> ShowcaseClip {
    let gates = vec![
        ShowcaseGate {
            center: [0.0, 1.35, 0.0],
            half_extents: [0.55, 0.42, 0.10],
            forward: [1.0, 0.0, 0.0],
        },
        ShowcaseGate {
            center: [4.0, 1.75, 3.0],
            half_extents: [0.55, 0.45, 0.10],
            forward: [0.8, 0.0, 0.6],
        },
        ShowcaseGate {
            center: [8.1, 2.15, 0.0],
            half_extents: [0.55, 0.48, 0.10],
            forward: [1.0, 0.0, -0.1],
        },
        ShowcaseGate {
            center: [11.6, 1.65, -2.8],
            half_extents: [0.58, 0.46, 0.10],
            forward: [0.82, 0.0, -0.57],
        },
    ];

    let path: [(f32, Vec3); 10] = [
        (0.0, vec3(-4.0, 1.15, 0.0)),
        (1.7, vec3(-1.4, 1.22, -0.2)),
        (3.2, vec3(0.4, 1.38, 0.1)),
        (5.4, vec3(4.0, 1.78, 3.0)),
        (7.6, vec3(6.8, 2.02, 2.3)),
        (9.7, vec3(8.1, 2.16, 0.1)),
        (12.1, vec3(11.6, 1.67, -2.8)),
        (14.8, vec3(8.4, 1.88, -5.7)),
        (17.0, vec3(2.0, 1.55, -3.3)),
        (18.0, vec3(-4.0, 1.15, 0.0)),
    ];

    let mut keyframes = Vec::with_capacity(path.len());
    for (index, (time_seconds, position)) in path.iter().copied().enumerate() {
        let prev_position = if index == 0 {
            path[index + 1].1
        } else {
            path[index - 1].1
        };
        let next_position = if index + 1 >= path.len() {
            path[index - 1].1
        } else {
            path[index + 1].1
        };

        let tangent = safe_normalize(next_position - prev_position, Vec3::X);
        let horizontal = Vec3::new(tangent.x, 0.0, tangent.z).normalize_or_zero();
        let prev_horizontal = Vec3::new(
            (position - prev_position).x,
            0.0,
            (position - prev_position).z,
        )
        .normalize_or_zero();
        let next_horizontal = Vec3::new(
            (next_position - position).x,
            0.0,
            (next_position - position).z,
        )
        .normalize_or_zero();
        let signed_turn =
            prev_horizontal.x * next_horizontal.z - prev_horizontal.z * next_horizontal.x;
        let yaw = horizontal.z.atan2(horizontal.x);
        let pitch = tangent.y.clamp(-1.0, 1.0).asin();
        let roll = (-signed_turn * 0.75).clamp(-0.45, 0.45);
        let velocity = if index + 1 >= path.len() {
            (position - prev_position) / (time_seconds - path[index - 1].0).max(1.0e-6)
        } else if index == 0 {
            (next_position - position) / (path[index + 1].0 - time_seconds).max(1.0e-6)
        } else {
            (next_position - prev_position) / (path[index + 1].0 - path[index - 1].0).max(1.0e-6)
        };

        keyframes.push(Keyframe {
            time_seconds,
            pose: DronePose {
                position: vec3_to_array(position),
                velocity: vec3_to_array(velocity),
                attitude: [roll, pitch, yaw],
            },
        });
    }

    ShowcaseClip {
        name: "Dock Sweep",
        duration_seconds: 18.0,
        bounds: 16.0,
        gates,
        gate_focus_times: vec![3.8, 6.4, 10.6, 18.0],
        keyframes,
    }
}

fn gate_bar_instances(gate: ShowcaseGate) -> [RenderInstance; 4] {
    let forward = normalized_xz(gate.forward).unwrap_or([0.0, 0.0, 1.0]);
    let right = [forward[2], 0.0, -forward[0]];
    let up = [0.0, 1.0, 0.0];
    let hole_half_width = gate.half_extents[0].max(0.1);
    let hole_half_height = gate.half_extents[1].max(0.1);
    let center = gate.center;
    let gate_color = [0.96, 0.57, 0.14, 1.0];

    [
        RenderInstance::oriented_box(
            [
                center[0] - right[0] * (hole_half_width + GATE_FRAME_THICKNESS),
                center[1],
                center[2] - right[2] * (hole_half_width + GATE_FRAME_THICKNESS),
            ],
            right,
            forward,
            up,
            [
                GATE_FRAME_THICKNESS,
                GATE_DEPTH_HALF,
                hole_half_height + 2.0 * GATE_FRAME_THICKNESS,
            ],
            gate_color,
        ),
        RenderInstance::oriented_box(
            [
                center[0] + right[0] * (hole_half_width + GATE_FRAME_THICKNESS),
                center[1],
                center[2] + right[2] * (hole_half_width + GATE_FRAME_THICKNESS),
            ],
            right,
            forward,
            up,
            [
                GATE_FRAME_THICKNESS,
                GATE_DEPTH_HALF,
                hole_half_height + 2.0 * GATE_FRAME_THICKNESS,
            ],
            gate_color,
        ),
        RenderInstance::oriented_box(
            [
                center[0],
                center[1] + hole_half_height + GATE_FRAME_THICKNESS,
                center[2],
            ],
            right,
            forward,
            up,
            [
                hole_half_width + 2.0 * GATE_FRAME_THICKNESS,
                GATE_DEPTH_HALF,
                GATE_FRAME_THICKNESS,
            ],
            gate_color,
        ),
        RenderInstance::oriented_box(
            [
                center[0],
                center[1] - hole_half_height - GATE_FRAME_THICKNESS,
                center[2],
            ],
            right,
            forward,
            up,
            [
                hole_half_width + 2.0 * GATE_FRAME_THICKNESS,
                GATE_DEPTH_HALF,
                GATE_FRAME_THICKNESS,
            ],
            gate_color,
        ),
    ]
}

fn floor_instance(bounds: f32) -> RenderInstance {
    RenderInstance::oriented_box(
        [0.0, FLOOR_ALTITUDE - FLOOR_HALF_THICKNESS, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [bounds, FLOOR_HALF_THICKNESS, bounds],
        [0.18, 0.19, 0.23, 1.0],
    )
}

fn drone_instances(pose: DronePose) -> Vec<RenderInstance> {
    let (forward, right, up) = drone_basis(pose);
    let center = pose.position;
    let mut parts = Vec::with_capacity(DRONE_MODEL_INSTANCE_COUNT);
    parts.push(RenderInstance::oriented_box(
        center,
        right,
        forward,
        up,
        DRONE_CORE_HALF_EXTENTS,
        [0.20, 0.77, 0.95, 1.0],
    ));

    let arm_dirs: [[f32; 3]; 4] = [
        [1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 0.0, -1.0],
    ];
    for axis in arm_dirs {
        let inv_len = (axis[0] * axis[0] + axis[2] * axis[2]).sqrt().recip();
        let dir = [axis[0] * inv_len, 0.0, axis[2] * inv_len];
        let arm_right = [
            right[0] * dir[0] + forward[0] * dir[2],
            right[1] * dir[0] + forward[1] * dir[2],
            right[2] * dir[0] + forward[2] * dir[2],
        ];
        let arm_offset = [
            arm_right[0] * DRONE_ARM_OFFSET,
            arm_right[1] * DRONE_ARM_OFFSET,
            arm_right[2] * DRONE_ARM_OFFSET,
        ];
        let motor_offset = [
            arm_right[0] * DRONE_MOTOR_OFFSET,
            arm_right[1] * DRONE_MOTOR_OFFSET,
            arm_right[2] * DRONE_MOTOR_OFFSET,
        ];
        let arm_center = [
            center[0] + arm_offset[0],
            center[1] + arm_offset[1],
            center[2] + arm_offset[2],
        ];
        let motor_center = [
            center[0] + motor_offset[0],
            center[1] + motor_offset[1],
            center[2] + motor_offset[2],
        ];

        parts.push(RenderInstance::oriented_box(
            arm_center,
            arm_right,
            up,
            forward,
            DRONE_ARM_HALF_EXTENTS,
            [0.24, 0.84, 0.97, 1.0],
        ));
        parts.push(RenderInstance::oriented_box(
            motor_center,
            right,
            up,
            forward,
            DRONE_MOTOR_HALF_EXTENTS,
            [0.92, 0.35, 0.30, 1.0],
        ));
    }

    parts
}

fn trail_instances(points: &VecDeque<[f32; 3]>) -> Vec<RenderInstance> {
    let segment_count = points.len().saturating_sub(1);
    let mut instances = Vec::with_capacity(segment_count.min(TRAIL_INSTANCE_COUNT));
    for (index, (start, end)) in points.iter().zip(points.iter().skip(1)).enumerate() {
        let progress = (index + 1) as f32 / segment_count.max(1) as f32;
        let color = [0.18 + progress * 0.30, 0.72 + progress * 0.18, 0.98, 1.0];
        if let Some(instance) = segment_instance(
            vec3_from_array(*start),
            vec3_from_array(*end),
            TRAIL_HALF_WIDTH,
            TRAIL_HALF_HEIGHT,
            color,
        ) {
            instances.push(instance);
        }
    }
    instances
}

fn debug_vector_instances(pose: DronePose) -> Vec<RenderInstance> {
    let (forward, _, up) = drone_basis(pose);
    let center = vec3_from_array(pose.position);
    let velocity = vec3_from_array(pose.velocity);
    let mut instances = Vec::with_capacity(DEBUG_VECTOR_INSTANCE_COUNT);

    if let Some(instance) = segment_instance(
        center,
        center + vec3_from_array(forward) * FORWARD_VECTOR_LENGTH,
        DEBUG_VECTOR_HALF_WIDTH,
        DEBUG_VECTOR_HALF_HEIGHT,
        [0.99, 0.92, 0.24, 1.0],
    ) {
        instances.push(instance);
    }
    if let Some(instance) = segment_instance(
        center,
        center + vec3_from_array(up) * THRUST_VECTOR_LENGTH,
        DEBUG_VECTOR_HALF_WIDTH,
        DEBUG_VECTOR_HALF_HEIGHT,
        [0.34, 0.95, 0.46, 1.0],
    ) {
        instances.push(instance);
    }
    let velocity_length = velocity.length();
    if velocity_length > 1.0e-5 {
        let scaled_velocity =
            velocity * VELOCITY_VECTOR_SCALE.min(VELOCITY_VECTOR_MAX_LENGTH / velocity_length);
        if let Some(instance) = segment_instance(
            center,
            center + scaled_velocity,
            DEBUG_VECTOR_HALF_WIDTH,
            DEBUG_VECTOR_HALF_HEIGHT,
            [0.96, 0.32, 0.28, 1.0],
        ) {
            instances.push(instance);
        }
    }

    instances
}

fn target_instance(gate: ShowcaseGate) -> RenderInstance {
    RenderInstance::oriented_box(
        gate.center,
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        TARGET_HALF_EXTENTS,
        [0.95, 0.2, 0.3, 1.0],
    )
}

fn normalized_xz(value: [f32; 3]) -> Option<[f32; 3]> {
    let length_sq = value[0] * value[0] + value[2] * value[2];
    if length_sq <= 1.0e-6 {
        return None;
    }
    let inv_len = length_sq.sqrt().recip();
    Some([value[0] * inv_len, 0.0, value[2] * inv_len])
}

fn drone_basis(pose: DronePose) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let yaw = pose.attitude[2];
    let (roll, pitch) = (pose.attitude[0], pose.attitude[1]);
    let (cy, sy) = (yaw.cos(), yaw.sin());
    let (cr, sr) = (roll.cos(), roll.sin());
    let (cp, sp) = (pitch.cos(), pitch.sin());
    let forward = [cy * cp, sp, sy * cp];
    let right = [cy * sp * sr + sy * cr, cp * sr, sy * sp * sr - cy * cr];
    let up = [cy * sp * cr - sy * sr, cp * cr, sy * sp * cr + cy * sr];
    (forward, right, up)
}

fn segment_instance(
    start: Vec3,
    end: Vec3,
    half_width: f32,
    half_height: f32,
    color: [f32; 4],
) -> Option<RenderInstance> {
    let delta = end - start;
    let length = delta.length();
    if length <= 1.0e-5 {
        return None;
    }

    let forward = delta / length;
    let right = safe_normalize(
        forward.cross(Vec3::Y),
        if forward.y.abs() < 0.95 {
            Vec3::Z
        } else {
            Vec3::X
        },
    );
    let up = safe_normalize(right.cross(forward), Vec3::Y);
    Some(RenderInstance::oriented_box(
        vec3_to_array((start + end) * 0.5),
        vec3_to_array(right),
        vec3_to_array(forward),
        vec3_to_array(up),
        [half_width, length * 0.5, half_height],
        color,
    ))
}

fn safe_normalize(value: Vec3, fallback: Vec3) -> Vec3 {
    if value.length_squared() <= 1.0e-6 {
        fallback.normalize_or_zero()
    } else {
        value.normalize()
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
    ]
}

fn lerp_angle(a: f32, b: f32, t: f32) -> f32 {
    let delta = (b - a + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
        - std::f32::consts::PI;
    a + delta * t
}

fn vec3_from_array(value: [f32; 3]) -> Vec3 {
    Vec3::new(value[0], value[1], value[2])
}

fn vec3_to_array(value: Vec3) -> [f32; 3] {
    [value.x, value.y, value.z]
}
