use std::error::Error;
use std::sync::{Arc, Mutex};

use glam::Vec3;
use tracing::info;
use triad_gpu::{
    BindingType, BufferUsage, ColorLoadOp, DepthLoadOp, ExecutableFrameGraph, FrameGraphError,
    FrameTextureView, RenderPassBuilder, Renderer, ResourceRegistry, ShaderStage, wgpu,
};
use triad_sim::{
    Action, CourseSpec, EnvLayoutHeader, EnvState, Gate, GpuSimulation, GpuSimulationConfig,
    ResetParams,
};
use triad_window::{
    CameraPose, CameraUniforms, RendererManager, WindowConfig, egui, run_with_renderer_config,
};

const WINDOW_TITLE: &str = "Triad Visualizer";
const GATE_DEPTH_HALF: f32 = 0.015;
const GATE_THICKNESS_MIN: f32 = 0.0125;
const DRONE_HALF_EXTENTS: [f32; 3] = [0.08, 0.08, 0.05];
const TARGET_HALF_EXTENTS: [f32; 3] = [0.05, 0.05, 0.05];
const VISUALIZER_ENV_COUNT: usize = 128;

const VISUALIZER_SHADER: &str = r#"
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

#[derive(Debug)]
struct UiState {
    running: bool,
    selected_env: usize,
    difficulty: f32,
    curriculum_stage: u32,
    seed_base: u32,
    request_reset_all: bool,
    request_randomize: bool,
    request_step: bool,
    gate_count: u32,
    current_gate: u32,
    done: bool,
    position: [f32; 3],
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            running: false,
            selected_env: 0,
            difficulty: 0.35,
            curriculum_stage: 1,
            seed_base: 1,
            request_reset_all: false,
            request_randomize: false,
            request_step: false,
            gate_count: 0,
            current_gate: 0,
            done: false,
            position: [0.0, 0.0, 0.0],
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct UiSnapshot {
    running: bool,
    selected_env: usize,
    difficulty: f32,
    curriculum_stage: u32,
    seed_base: u32,
    reset_all: bool,
    randomize: bool,
    step_once: bool,
}

struct VisualizerManager {
    sim: GpuSimulation,
    ui_state: Arc<Mutex<UiState>>,
    camera_buffer: triad_gpu::Handle<wgpu::Buffer>,
    instance_buffer: triad_gpu::Handle<wgpu::Buffer>,
    render_bind_group: triad_gpu::Handle<wgpu::BindGroup>,
    render_pipeline: triad_gpu::Handle<wgpu::RenderPipeline>,
    frame_target: triad_gpu::Handle<FrameTextureView>,
    depth_frame: triad_gpu::Handle<FrameTextureView>,
    cached_layouts: Vec<EnvLayoutHeader>,
    cached_gates: Vec<Gate>,
    cached_states: Vec<EnvState>,
    actions: Vec<Action>,
    instances: Vec<RenderInstance>,
    selected_env: usize,
    layouts_dirty: bool,
}

impl VisualizerManager {
    fn new(
        renderer: &Renderer,
        registry: &mut ResourceRegistry,
        surface_format: wgpu::TextureFormat,
        ui_state: Arc<Mutex<UiState>>,
    ) -> Result<Self, Box<dyn Error>> {
        let course = visualizer_course();
        let sim = GpuSimulation::new(
            renderer,
            registry,
            GpuSimulationConfig {
                env_count: VISUALIZER_ENV_COUNT,
                max_gates_per_env: required_gate_capacity(&course),
                ..GpuSimulationConfig::default()
            },
        )?;
        sim.set_course(renderer, registry, &course)?;

        let camera_buffer = renderer
            .create_gpu_buffer::<CameraUniforms>()
            .label("visualizer camera")
            .with_data(&[CameraUniforms::from_matrices(
                glam::Mat4::IDENTITY,
                glam::Mat4::IDENTITY,
                Vec3::ZERO,
            )])
            .usage(BufferUsage::Uniform)
            .build(registry)?;

        let max_instances = visible_instance_capacity(sim.config().max_gates_per_env);
        let hidden_instances = vec![RenderInstance::hidden(); max_instances];
        let instance_buffer = renderer
            .create_gpu_buffer::<RenderInstance>()
            .label("visualizer instances")
            .with_data(&hidden_instances)
            .build(registry)?;

        let shader = renderer
            .create_shader_module()
            .label("visualizer boxes")
            .with_wgsl_source(VISUALIZER_SHADER)
            .build(registry)?;

        let (render_layout, render_bind_group) = renderer
            .create_bind_group()
            .label("visualizer render")
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
                    label: Some("visualizer render layout"),
                    bind_group_layouts: &[registry
                        .get(render_layout)
                        .expect("visualizer render layout should exist")],
                    push_constant_ranges: &[],
                });

        let render_pipeline = renderer
            .create_render_pipeline()
            .with_label("visualizer render pipeline")
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

        let zero_actions = vec![Action::new([0.0; 4]); sim.env_count()];
        sim.set_actions(renderer, registry, &zero_actions)?;

        Ok(Self {
            sim,
            ui_state,
            camera_buffer: camera_buffer.handle(),
            instance_buffer: instance_buffer.handle(),
            render_bind_group,
            render_pipeline,
            frame_target: registry.insert(FrameTextureView::new()),
            depth_frame: registry.insert(FrameTextureView::new()),
            cached_layouts: Vec::new(),
            cached_gates: Vec::new(),
            cached_states: Vec::new(),
            actions: zero_actions,
            instances: hidden_instances,
            selected_env: 0,
            layouts_dirty: true,
        })
    }

    fn snapshot_ui(&self) -> UiSnapshot {
        let mut state = self.ui_state.lock().expect("ui state poisoned");
        let snapshot = UiSnapshot {
            running: state.running,
            selected_env: state
                .selected_env
                .min(self.sim.env_count().saturating_sub(1)),
            difficulty: state.difficulty,
            curriculum_stage: state.curriculum_stage.min(3),
            seed_base: state.seed_base,
            reset_all: state.request_reset_all,
            randomize: state.request_randomize,
            step_once: state.request_step,
        };
        state.request_reset_all = false;
        state.request_randomize = false;
        state.request_step = false;
        snapshot
    }

    fn randomize_reset_params(
        &self,
        base_seed: u32,
        difficulty: f32,
        curriculum_stage: u32,
    ) -> Vec<ResetParams> {
        (0..self.sim.env_count())
            .map(|env_index| {
                let env_seed = hash_u32(base_seed ^ (env_index as u32).wrapping_mul(0x9e37_79b9));
                let grammar_id = env_seed % 4;
                let difficulty_jitter = hash_to_unit(env_seed ^ 0x85eb_ca6b) * 0.2 - 0.1;
                let env_difficulty = (difficulty + difficulty_jitter).clamp(0.0, 1.0);
                ResetParams::new(env_seed, grammar_id, env_difficulty, curriculum_stage)
            })
            .collect()
    }

    fn refresh_layout_cache(
        &mut self,
        renderer: &Renderer,
        registry: &ResourceRegistry,
    ) -> Result<(), Box<dyn Error>> {
        self.cached_layouts = self.sim.readback_layout_headers(renderer, registry)?;
        self.cached_gates = self.sim.readback_gates(renderer, registry)?;
        self.layouts_dirty = false;
        Ok(())
    }

    fn refresh_state_cache(
        &mut self,
        renderer: &Renderer,
        registry: &ResourceRegistry,
    ) -> Result<(), Box<dyn Error>> {
        self.cached_states = self.sim.readback_state(renderer, registry)?;
        Ok(())
    }

    fn selected_gate_count(&self) -> u32 {
        self.cached_layouts
            .get(self.selected_env)
            .map(|layout| layout.gate_count)
            .unwrap_or(0)
    }

    fn target_gate(&self, state: &EnvState) -> Option<Gate> {
        let layout = *self.cached_layouts.get(self.selected_env)?;
        if layout.gate_count == 0 {
            return None;
        }
        let gate_index = state.current_gate.min(layout.gate_count.saturating_sub(1));
        self.cached_gates
            .get((layout.gate_offset + gate_index) as usize)
            .copied()
    }

    fn rebuild_instances(&mut self, selected_state: Option<EnvState>) {
        self.instances.fill(RenderInstance::hidden());

        let Some(layout) = self.cached_layouts.get(self.selected_env).copied() else {
            return;
        };

        let mut write_index = 0usize;
        for gate_index in 0..layout.gate_count as usize {
            if let Some(gate) = self
                .cached_gates
                .get((layout.gate_offset as usize) + gate_index)
                .copied()
            {
                for bar in gate_bar_instances(gate) {
                    if let Some(slot) = self.instances.get_mut(write_index) {
                        *slot = bar;
                        write_index += 1;
                    }
                }
            }
        }

        for obstacle_index in 0..layout.obstacle_count as usize {
            if let Some(obstacle) = self
                .cached_gates
                .get((layout.obstacle_offset as usize) + obstacle_index)
                .copied()
            {
                if let Some(slot) = self.instances.get_mut(write_index) {
                    *slot = obstacle_instance(obstacle);
                    write_index += 1;
                }
            }
        }

        if let Some(state) = selected_state {
            let target_gate = self.target_gate(&state);
            if let Some(slot) = self.instances.get_mut(write_index) {
                *slot = drone_instance(state, target_gate);
                write_index += 1;
            }
            if let Some(target_gate) = target_gate {
                if let Some(slot) = self.instances.get_mut(write_index) {
                    *slot = target_instance(target_gate);
                }
            }
        }
    }

    fn update_ui_snapshot(&self, selected_state: Option<EnvState>) {
        let mut ui = self.ui_state.lock().expect("ui state poisoned");
        if let Some(state) = selected_state {
            ui.gate_count = self.selected_gate_count();
            ui.current_gate = state.current_gate;
            ui.done = state.done != 0;
            ui.position = state.position;
        } else {
            ui.gate_count = 0;
            ui.current_gate = 0;
            ui.done = false;
            ui.position = [0.0, 0.0, 0.0];
        }
    }
}

impl RendererManager for VisualizerManager {
    fn update(
        &mut self,
        renderer: &Renderer,
        registry: &mut ResourceRegistry,
        camera: &CameraUniforms,
    ) -> Result<(), Box<dyn Error>> {
        renderer.write_buffer(self.camera_buffer, std::slice::from_ref(camera), registry)?;

        let snapshot = self.snapshot_ui();
        self.selected_env = snapshot.selected_env;

        let mut forced_reset_step = false;
        if self.layouts_dirty {
            self.sim.step(renderer, registry);
            forced_reset_step = true;
        }

        if snapshot.randomize {
            let params = self.randomize_reset_params(
                snapshot.seed_base,
                snapshot.difficulty,
                snapshot.curriculum_stage,
            );
            self.sim.set_reset_params(renderer, registry, &params)?;
            self.sim.reset_all(renderer, registry)?;
            self.sim.step(renderer, registry);
            self.layouts_dirty = true;
            forced_reset_step = true;
        } else if snapshot.reset_all {
            self.sim.reset_all(renderer, registry)?;
            self.sim.step(renderer, registry);
            self.layouts_dirty = true;
            forced_reset_step = true;
        }

        if self.layouts_dirty || forced_reset_step {
            self.refresh_layout_cache(renderer, registry)?;
        }

        self.refresh_state_cache(renderer, registry)?;

        let selected_state_before_step = self.cached_states.get(self.selected_env).copied();
        self.actions.fill(Action::new([0.0; 4]));
        if snapshot.running || snapshot.step_once {
            if let Some(state) = selected_state_before_step {
                if let Some(target_gate) = self.target_gate(&state) {
                    self.actions[self.selected_env] = autopilot_action(state, target_gate);
                }
            }
            self.sim.set_actions(renderer, registry, &self.actions)?;
            self.sim.step(renderer, registry);
            self.refresh_state_cache(renderer, registry)?;
        }

        let selected_state = self.cached_states.get(self.selected_env).copied();
        self.rebuild_instances(selected_state);
        renderer.write_buffer(self.instance_buffer, &self.instances, registry)?;
        self.update_ui_snapshot(selected_state);

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
            .expect("visualizer frame target should exist")
            .set(final_view);
        if let Some(depth) = depth_view {
            registry
                .get(self.depth_frame)
                .expect("visualizer depth target should exist")
                .set(depth);
        }
        Ok(false)
    }

    fn build_frame_graph(&mut self) -> Result<ExecutableFrameGraph, FrameGraphError> {
        let render_pass = RenderPassBuilder::new("VisualizerRender")
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
            .expect("visualizer render pass should build");

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

fn main() -> Result<(), Box<dyn Error>> {
    init_logging();

    let ui_state = Arc::new(Mutex::new(UiState::default()));
    let ui_state_for_controls = Arc::clone(&ui_state);
    let ui_state_for_manager = Arc::clone(&ui_state);

    run_with_renderer_config(
        WINDOW_TITLE,
        WindowConfig::default(),
        move |controls| {
            controls.request_reset(CameraPose::new(
                Vec3::new(8.5, 5.5, 8.5),
                Vec3::new(0.0, 0.75, 0.0),
            ));

            controls.on_ui(move |ctx| {
                let mut ui = ui_state_for_controls.lock().expect("ui state poisoned");
                egui::Window::new("Visualizer")
                    .default_pos(egui::pos2(12.0, 84.0))
                    .show(ctx, |panel| {
                        if panel
                            .button(if ui.running { "Pause" } else { "Run" })
                            .clicked()
                        {
                            ui.running = !ui.running;
                        }
                        if panel.button("Step").clicked() {
                            ui.request_step = true;
                        }
                        if panel.button("Reset All").clicked() {
                            ui.request_reset_all = true;
                        }
                        if panel.button("Randomize").clicked() {
                            ui.request_randomize = true;
                            ui.seed_base = ui.seed_base.wrapping_add(1);
                        }

                        panel.separator();
                        panel.add(
                            egui::Slider::new(
                                &mut ui.selected_env,
                                0..=(VISUALIZER_ENV_COUNT.saturating_sub(1)),
                            )
                            .text("Selected Env"),
                        );
                        panel.add(
                            egui::Slider::new(&mut ui.difficulty, 0.0..=1.0).text("Difficulty"),
                        );
                        panel.add(
                            egui::Slider::new(&mut ui.curriculum_stage, 0..=3).text("Curriculum"),
                        );

                        panel.label(format!("Gate Count: {}", ui.gate_count));
                        panel.label(format!("Current Gate: {}", ui.current_gate));
                        panel.label(format!("Done: {}", ui.done));
                        panel.label(format!(
                            "Position: {:.2}, {:.2}, {:.2}",
                            ui.position[0], ui.position[1], ui.position[2]
                        ));
                    });
            });
        },
        move |renderer, registry, surface_format, _width, _height| {
            let manager =
                VisualizerManager::new(renderer, registry, surface_format, ui_state_for_manager)?;
            Ok(Box::new(manager))
        },
    )
}

fn init_logging() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "triad_visualizer=info,triad_window=info".into()),
        )
        .with_target(false)
        .compact()
        .try_init();
    info!("starting visualizer");
}

fn visible_instance_capacity(max_gates_per_env: usize) -> usize {
    max_gates_per_env * 4 + max_obstacles_per_env(max_gates_per_env) + 2
}

fn required_gate_capacity(course: &CourseSpec) -> usize {
    course.total_gate_count().max(1) as usize
}

fn visualizer_course() -> CourseSpec {
    CourseSpec::default_drone_course()
}

fn gate_bar_instances(gate: Gate) -> [RenderInstance; 4] {
    let forward = normalized_xz(gate.forward).unwrap_or([0.0, 0.0, 1.0]);
    let right = [forward[2], 0.0, -forward[0]];
    let up = [0.0, 1.0, 0.0];
    let hole_half_width = gate.half_extents[0].max(0.1);
    let hole_half_height = gate.half_extents[1].max(0.1);
    let depth_half = gate.half_extents[2].max(GATE_DEPTH_HALF);
    let thickness = (hole_half_width.min(hole_half_height) * 0.08).max(GATE_THICKNESS_MIN);
    let center = gate.center;
    let gate_color = [0.96, 0.57, 0.14, 1.0];

    [
        RenderInstance::oriented_box(
            [
                center[0] - right[0] * (hole_half_width + thickness),
                center[1],
                center[2] - right[2] * (hole_half_width + thickness),
            ],
            right,
            forward,
            up,
            [thickness, depth_half, hole_half_height + 2.0 * thickness],
            gate_color,
        ),
        RenderInstance::oriented_box(
            [
                center[0] + right[0] * (hole_half_width + thickness),
                center[1],
                center[2] + right[2] * (hole_half_width + thickness),
            ],
            right,
            forward,
            up,
            [thickness, depth_half, hole_half_height + 2.0 * thickness],
            gate_color,
        ),
        RenderInstance::oriented_box(
            [
                center[0],
                center[1] + hole_half_height + thickness,
                center[2],
            ],
            right,
            forward,
            up,
            [hole_half_width + 2.0 * thickness, depth_half, thickness],
            gate_color,
        ),
        RenderInstance::oriented_box(
            [
                center[0],
                center[1] - hole_half_height - thickness,
                center[2],
            ],
            right,
            forward,
            up,
            [hole_half_width + 2.0 * thickness, depth_half, thickness],
            gate_color,
        ),
    ]
}

fn obstacle_instance(obstacle: Gate) -> RenderInstance {
    let forward = normalized_xz(obstacle.forward).unwrap_or([0.0, 0.0, 1.0]);
    let right = [forward[2], 0.0, -forward[0]];
    let up = [0.0, 1.0, 0.0];
    RenderInstance::oriented_box(
        obstacle.center,
        right,
        forward,
        up,
        [
            obstacle.half_extents[0].max(0.05),
            obstacle.half_extents[2].max(0.05),
            obstacle.half_extents[1].max(0.05),
        ],
        [0.18, 0.74, 0.68, 1.0],
    )
}

fn drone_instance(state: EnvState, _target_gate: Option<Gate>) -> RenderInstance {
    let yaw = state.attitude[2];
    let forward = [yaw.cos(), 0.0, yaw.sin()];
    let right = [forward[2], 0.0, -forward[0]];
    let up = [0.0, 1.0, 0.0];
    RenderInstance::oriented_box(
        [state.position[0], state.position[1], state.position[2]],
        right,
        forward,
        up,
        DRONE_HALF_EXTENTS,
        [0.25, 0.8, 0.95, 1.0],
    )
}

fn target_instance(gate: Gate) -> RenderInstance {
    RenderInstance::oriented_box(
        [gate.center[0], gate.center[1], gate.center[2]],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        TARGET_HALF_EXTENTS,
        [0.95, 0.2, 0.3, 1.0],
    )
}

fn autopilot_action(state: EnvState, target_gate: Gate) -> Action {
    let delta_x = target_gate.center[0] - state.position[0];
    let delta_y = target_gate.center[1] - state.position[1];
    let delta_z = target_gate.center[2] - state.position[2];
    let yaw = state.attitude[2];
    let heading = [yaw.cos(), yaw.sin()];
    let right = [heading[1], -heading[0]];
    let horizontal_delta = [delta_x, delta_z];
    let horizontal_velocity = [state.velocity[0], state.velocity[2]];
    let forward_error = horizontal_delta[0] * heading[0] + horizontal_delta[1] * heading[1];
    let lateral_error = horizontal_delta[0] * right[0] + horizontal_delta[1] * right[1];
    let forward_velocity =
        horizontal_velocity[0] * heading[0] + horizontal_velocity[1] * heading[1];
    let lateral_velocity = horizontal_velocity[0] * right[0] + horizontal_velocity[1] * right[1];
    let desired_yaw = target_gate.forward[2].atan2(target_gate.forward[0]);
    let yaw_error = ((desired_yaw - yaw) + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
        - std::f32::consts::PI;

    let collective = (0.58 + delta_y * 0.22 - state.velocity[1] * 0.08).clamp(0.2, 0.9);
    let roll_cmd = (-lateral_error * 0.07 + lateral_velocity * 0.04).clamp(-0.25, 0.25);
    let pitch_cmd = (forward_error * 0.07 - forward_velocity * 0.04).clamp(-0.25, 0.25);
    let yaw_cmd = (yaw_error * 0.18 - state.angular_velocity[2] * 0.05).clamp(-0.16, 0.16);

    Action::new([
        (collective - roll_cmd - pitch_cmd + yaw_cmd).clamp(0.0, 1.0),
        (collective + roll_cmd - pitch_cmd - yaw_cmd).clamp(0.0, 1.0),
        (collective + roll_cmd + pitch_cmd + yaw_cmd).clamp(0.0, 1.0),
        (collective - roll_cmd + pitch_cmd - yaw_cmd).clamp(0.0, 1.0),
    ])
}

fn normalized_xz(value: [f32; 3]) -> Option<[f32; 3]> {
    let length_sq = value[0] * value[0] + value[2] * value[2];
    if length_sq <= 1e-6 {
        return None;
    }
    let inv_len = length_sq.sqrt().recip();
    Some([value[0] * inv_len, 0.0, value[2] * inv_len])
}

fn hash_u32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^= value >> 16;
    value
}

fn hash_to_unit(value: u32) -> f32 {
    (hash_u32(value) & 0x00ff_ffff) as f32 / 16_777_215.0
}

fn max_obstacles_per_env(max_gates_per_env: usize) -> usize {
    max_gates_per_env.saturating_div(2).max(1)
}
