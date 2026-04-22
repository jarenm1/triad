use std::error::Error;
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};

use glam::Vec3;
use serde::{Deserialize, Serialize};
use tracing::info;
use triad_gpu::{
    BindingType, BufferUsage, ColorLoadOp, DepthLoadOp, ExecutableFrameGraph, FrameGraphError,
    FrameTextureView, RenderPassBuilder, Renderer, ResourceRegistry, ShaderStage, wgpu,
};
use triad_sim::{
    Action, CourseSpec, EnvLayoutHeader, EnvState, Gate, GpuSimulation, GpuSimulationConfig,
    Observation, ResetParams, RewardDone,
};
use triad_window::{
    CameraPose, CameraUniforms, RendererManager, WindowConfig, egui, run_with_renderer_config,
};

const WINDOW_TITLE: &str = "Triad Visualizer";
const GATE_DEPTH_HALF: f32 = 0.015;
const GATE_THICKNESS_MIN: f32 = 0.035;
const DRONE_CORE_HALF_EXTENTS: [f32; 3] = [0.06, 0.025, 0.035];
const DRONE_ARM_HALF_EXTENTS: [f32; 3] = [0.14, 0.015, 0.015];
const DRONE_MOTOR_HALF_EXTENTS: [f32; 3] = [0.028, 0.018, 0.028];
const DRONE_ARM_OFFSET: f32 = 0.11;
const DRONE_MOTOR_OFFSET: f32 = 0.22;
const DRONE_MODEL_INSTANCE_COUNT: usize = 9;
const TARGET_HALF_EXTENTS: [f32; 3] = [0.05, 0.05, 0.05];
const VISUALIZER_ENV_COUNT: usize = 128;
const DONE_REASON_COMPLETE: u32 = 1 << 0;
const DONE_REASON_GATE_COLLISION: u32 = 1 << 1;
const DONE_REASON_OBSTACLE_COLLISION: u32 = 1 << 2;
const DONE_REASON_FLOOR_COLLISION: u32 = 1 << 3;
const DONE_REASON_OUT_OF_BOUNDS: u32 = 1 << 4;
const DONE_REASON_STEP_LIMIT: u32 = 1 << 5;
const DONE_REASON_EXCESSIVE_TILT: u32 = 1 << 6;

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
    replay_active: bool,
    use_checkpoint_policy: bool,
    selected_env: usize,
    difficulty: f32,
    curriculum_stage: u32,
    seed_base: u32,
    checkpoint_path: String,
    checkpoint_status: String,
    request_load_checkpoint: bool,
    request_reset_selected: bool,
    request_reset_all: bool,
    request_randomize: bool,
    gate_count: u32,
    current_gate: u32,
    done: bool,
    position: [f32; 3],
    reward: f32,
    done_reason_bits: u32,
    progress: f32,
    distance_to_gate: f32,
    gate_alignment: f32,
    mean_motor_thrust: f32,
    progress_reward: f32,
    distance_penalty: f32,
    alignment_reward: f32,
    tilt_penalty: f32,
    completion_bonus: f32,
    collision_penalty: f32,
    last_value_estimate: f32,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            replay_active: false,
            use_checkpoint_policy: false,
            selected_env: 0,
            difficulty: 0.35,
            curriculum_stage: 1,
            seed_base: 1,
            checkpoint_path: "checkpoints/ppo.pt".to_string(),
            checkpoint_status: "Heuristic replay ready".to_string(),
            request_load_checkpoint: false,
            request_reset_selected: false,
            request_reset_all: false,
            request_randomize: false,
            gate_count: 0,
            current_gate: 0,
            done: false,
            position: [0.0, 0.0, 0.0],
            reward: 0.0,
            done_reason_bits: 0,
            progress: 0.0,
            distance_to_gate: 0.0,
            gate_alignment: 0.0,
            mean_motor_thrust: 0.0,
            progress_reward: 0.0,
            distance_penalty: 0.0,
            alignment_reward: 0.0,
            tilt_penalty: 0.0,
            completion_bonus: 0.0,
            collision_penalty: 0.0,
            last_value_estimate: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
struct UiSnapshot {
    replay_active: bool,
    use_checkpoint_policy: bool,
    selected_env: usize,
    difficulty: f32,
    curriculum_stage: u32,
    seed_base: u32,
    checkpoint_path: String,
    load_checkpoint: bool,
    reset_selected: bool,
    reset_all: bool,
    randomize: bool,
}

#[derive(Serialize)]
struct PpoPolicyRequest {
    observations: Vec<Vec<f32>>,
    deterministic: bool,
}

#[derive(Deserialize)]
struct PpoPolicyResponse {
    actions: Option<Vec<Vec<f32>>>,
    values: Option<Vec<f32>>,
    error: Option<String>,
}

struct PpoPolicyClient {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl PpoPolicyClient {
    fn spawn(checkpoint_path: &str) -> Result<Self, Box<dyn Error>> {
        if checkpoint_path.trim().is_empty() {
            return Err(io::Error::other("checkpoint path is empty").into());
        }

        let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .ok_or_else(|| io::Error::other("workspace root missing"))?
            .to_path_buf();
        let mut child = Command::new("uv")
            .args([
                "run",
                "--extra",
                "training",
                "python",
                "-m",
                "triad_py",
                "ppo-policy-server",
                "--checkpoint",
                checkpoint_path,
                "--device",
                "cpu",
            ])
            .current_dir(workspace_root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| io::Error::other("policy server stdin unavailable"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("policy server stdout unavailable"))?;
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        })
    }

    fn predict(
        &mut self,
        observations: &[Observation],
    ) -> Result<(Vec<[f32; 4]>, Vec<f32>), Box<dyn Error>> {
        let request = PpoPolicyRequest {
            observations: observations.iter().map(flatten_observation).collect(),
            deterministic: true,
        };
        writeln!(self.stdin, "{}", serde_json::to_string(&request)?)?;
        self.stdin.flush()?;

        let mut line = String::new();
        if self.stdout.read_line(&mut line)? == 0 {
            return Err(io::Error::other("policy server exited unexpectedly").into());
        }

        let response: PpoPolicyResponse = serde_json::from_str(line.trim())?;
        if let Some(error) = response.error {
            return Err(io::Error::other(error).into());
        }

        let action_rows = response
            .actions
            .ok_or_else(|| io::Error::other("policy response missing actions"))?;
        let mut actions = Vec::with_capacity(action_rows.len());
        for row in action_rows {
            if row.len() != 4 {
                return Err(
                    io::Error::other(format!(
                        "expected 4 action values, got {}",
                        row.len()
                    ))
                    .into(),
                );
            }
            actions.push([row[0], row[1], row[2], row[3]]);
        }
        Ok((actions, response.values.unwrap_or_default()))
    }
}

impl Drop for PpoPolicyClient {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "{{\"kind\":\"shutdown\"}}");
        let _ = self.stdin.flush();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
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
    cached_observations: Vec<Observation>,
    cached_reward_done: Vec<RewardDone>,
    actions: Vec<Action>,
    instances: Vec<RenderInstance>,
    checkpoint_client: Option<PpoPolicyClient>,
    last_value_estimates: Vec<f32>,
    selected_env: usize,
    layouts_dirty: bool,
    applied_difficulty: f32,
    applied_curriculum_stage: u32,
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
            cached_observations: Vec::new(),
            cached_reward_done: Vec::new(),
            actions: zero_actions,
            instances: hidden_instances,
            checkpoint_client: None,
            last_value_estimates: vec![0.0; VISUALIZER_ENV_COUNT],
            selected_env: 0,
            layouts_dirty: true,
            applied_difficulty: 0.35,
            applied_curriculum_stage: 1,
        })
    }

    fn snapshot_ui(&self) -> UiSnapshot {
        let mut state = self.ui_state.lock().expect("ui state poisoned");
        let snapshot = UiSnapshot {
            replay_active: state.replay_active,
            use_checkpoint_policy: state.use_checkpoint_policy,
            selected_env: state
                .selected_env
                .min(self.sim.env_count().saturating_sub(1)),
            difficulty: state.difficulty,
            curriculum_stage: state.curriculum_stage.min(3),
            seed_base: state.seed_base,
            checkpoint_path: state.checkpoint_path.clone(),
            load_checkpoint: state.request_load_checkpoint,
            reset_selected: state.request_reset_selected,
            reset_all: state.request_reset_all,
            randomize: state.request_randomize,
        };
        state.request_load_checkpoint = false;
        state.request_reset_selected = false;
        state.request_reset_all = false;
        state.request_randomize = false;
        snapshot
    }

    fn set_checkpoint_status(&self, status: impl Into<String>) {
        self.ui_state
            .lock()
            .expect("ui state poisoned")
            .checkpoint_status = status.into();
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

    fn refresh_observation_cache(
        &mut self,
        renderer: &Renderer,
        registry: &ResourceRegistry,
    ) -> Result<(), Box<dyn Error>> {
        self.cached_observations = self.sim.readback_observations(renderer, registry)?;
        Ok(())
    }

    fn refresh_reward_done_cache(
        &mut self,
        renderer: &Renderer,
        registry: &ResourceRegistry,
    ) -> Result<(), Box<dyn Error>> {
        self.cached_reward_done = self.sim.readback_reward_done(renderer, registry)?;
        Ok(())
    }

    fn selected_gate_count(&self) -> u32 {
        self.cached_layouts
            .get(self.selected_env)
            .map(|layout| layout.gate_count)
            .unwrap_or(0)
    }

    fn target_gate_for_env(&self, env_index: usize, state: &EnvState) -> Option<Gate> {
        let layout = *self.cached_layouts.get(env_index)?;
        if layout.gate_count == 0 {
            return None;
        }
        let gate_index = state.current_gate.min(layout.gate_count.saturating_sub(1));
        self.cached_gates
            .get((layout.gate_offset + gate_index) as usize)
            .copied()
    }

    fn apply_heuristic_actions(&mut self) {
        self.actions.fill(Action::new([0.0; 4]));
        self.last_value_estimates.fill(0.0);
        for (env_index, state) in self.cached_states.iter().copied().enumerate() {
            if let Some(target_gate) = self.target_gate_for_env(env_index, &state) {
                self.actions[env_index] = autopilot_action(state, target_gate);
            }
        }
        self.set_checkpoint_status("Replay active with heuristic policy");
    }

    fn apply_checkpoint_actions(&mut self) -> Result<(), Box<dyn Error>> {
        let Some(client) = self.checkpoint_client.as_mut() else {
            return Err(io::Error::other("checkpoint policy is not loaded").into());
        };

        let (action_rows, values) = client.predict(&self.cached_observations)?;
        if action_rows.len() != self.actions.len() {
            return Err(
                io::Error::other(format!(
                    "policy returned {} action rows for {} envs",
                    action_rows.len(),
                    self.actions.len()
                ))
                .into(),
            );
        }

        for (index, action) in action_rows.into_iter().enumerate() {
            self.actions[index] = Action::new(action);
        }
        self.last_value_estimates.fill(0.0);
        for (index, value) in values.into_iter().enumerate() {
            if let Some(slot) = self.last_value_estimates.get_mut(index) {
                *slot = value;
            }
        }
        self.set_checkpoint_status("Replay active with PPO checkpoint");
        Ok(())
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
            let target_gate = self.target_gate_for_env(self.selected_env, &state);
            for instance in drone_instances(state) {
                if let Some(slot) = self.instances.get_mut(write_index) {
                    *slot = instance;
                    write_index += 1;
                }
            }
            if let Some(target_gate) = target_gate {
                if let Some(slot) = self.instances.get_mut(write_index) {
                    *slot = target_instance(target_gate);
                }
            }
        }
    }

    fn update_ui_snapshot(
        &self,
        selected_state: Option<EnvState>,
        selected_observation: Option<Observation>,
        selected_reward_done: Option<RewardDone>,
    ) {
        let mut ui = self.ui_state.lock().expect("ui state poisoned");
        if let Some(state) = selected_state {
            ui.gate_count = self.selected_gate_count();
            ui.current_gate = state.current_gate;
            ui.done = state.done != 0;
            ui.position = state.position;
            ui.last_value_estimate = self
                .last_value_estimates
                .get(self.selected_env)
                .copied()
                .unwrap_or(0.0);
        } else {
            ui.gate_count = 0;
            ui.current_gate = 0;
            ui.done = false;
            ui.position = [0.0, 0.0, 0.0];
            ui.last_value_estimate = 0.0;
        }

        if let Some(observation) = selected_observation {
            ui.progress = observation.progress;
            ui.distance_to_gate = observation.distance_to_gate;
            ui.gate_alignment = observation.gate_alignment;
            ui.mean_motor_thrust = observation.mean_motor_thrust;
        } else {
            ui.progress = 0.0;
            ui.distance_to_gate = 0.0;
            ui.gate_alignment = 0.0;
            ui.mean_motor_thrust = 0.0;
        }

        if let Some(reward_done) = selected_reward_done {
            ui.reward = reward_done.reward;
            ui.done_reason_bits = reward_done.done_reason;
            ui.progress_reward = reward_done.progress_reward;
            ui.distance_penalty = reward_done.distance_penalty;
            ui.alignment_reward = reward_done.alignment_reward;
            ui.tilt_penalty = reward_done.tilt_penalty;
            ui.completion_bonus = reward_done.completion_bonus;
            ui.collision_penalty = reward_done.collision_penalty;
        } else {
            ui.reward = 0.0;
            ui.done_reason_bits = 0;
            ui.progress_reward = 0.0;
            ui.distance_penalty = 0.0;
            ui.alignment_reward = 0.0;
            ui.tilt_penalty = 0.0;
            ui.completion_bonus = 0.0;
            ui.collision_penalty = 0.0;
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
        let generation_changed = (snapshot.difficulty - self.applied_difficulty).abs() > 1e-5
            || snapshot.curriculum_stage != self.applied_curriculum_stage;

        if snapshot.load_checkpoint {
            match PpoPolicyClient::spawn(&snapshot.checkpoint_path) {
                Ok(client) => {
                    self.checkpoint_client = Some(client);
                    self.set_checkpoint_status(format!(
                        "Loaded checkpoint {}",
                        snapshot.checkpoint_path
                    ));
                }
                Err(error) => {
                    self.checkpoint_client = None;
                    self.set_checkpoint_status(format!("Checkpoint load failed: {error}"));
                }
            }
        }

        let mut forced_reset_step = false;
        if self.layouts_dirty {
            self.sim.step(renderer, registry);
            forced_reset_step = true;
        }

        if snapshot.randomize || generation_changed {
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
            self.applied_difficulty = snapshot.difficulty;
            self.applied_curriculum_stage = snapshot.curriculum_stage;
        } else if snapshot.reset_selected {
            self.sim
                .request_resets(renderer, registry, &[self.selected_env])?;
            self.sim.step(renderer, registry);
            forced_reset_step = true;
        } else if snapshot.reset_all {
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
        }

        if self.layouts_dirty || forced_reset_step {
            self.refresh_layout_cache(renderer, registry)?;
        }

        self.refresh_state_cache(renderer, registry)?;
        self.refresh_observation_cache(renderer, registry)?;
        self.refresh_reward_done_cache(renderer, registry)?;

        if snapshot.replay_active {
            let apply_result = if snapshot.use_checkpoint_policy {
                self.apply_checkpoint_actions()
            } else {
                self.apply_heuristic_actions();
                Ok(())
            };

            match apply_result {
                Ok(()) => {
                    self.sim.set_actions(renderer, registry, &self.actions)?;
                    self.sim.step(renderer, registry);
                    self.refresh_state_cache(renderer, registry)?;
                    self.refresh_observation_cache(renderer, registry)?;
                    self.refresh_reward_done_cache(renderer, registry)?;
                }
                Err(error) => {
                    {
                        let mut ui = self.ui_state.lock().expect("ui state poisoned");
                        ui.replay_active = false;
                    }
                    self.set_checkpoint_status(format!("Replay paused: {error}"));
                }
            }
        }

        let selected_state = self.cached_states.get(self.selected_env).copied();
        let selected_observation = self.cached_observations.get(self.selected_env).copied();
        let selected_reward_done = self.cached_reward_done.get(self.selected_env).copied();
        if snapshot.replay_active
            && selected_reward_done.map(|value| value.done != 0).unwrap_or(false)
        {
            {
                let mut ui = self.ui_state.lock().expect("ui state poisoned");
                ui.replay_active = false;
            }
            if let Some(reward_done) = selected_reward_done {
                self.set_checkpoint_status(format!(
                    "Replay paused at terminal state: {}",
                    format_done_reasons(reward_done.done_reason)
                ));
            }
        }
        self.rebuild_instances(selected_state);
        renderer.write_buffer(self.instance_buffer, &self.instances, registry)?;
        self.update_ui_snapshot(selected_state, selected_observation, selected_reward_done);

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
                        panel.checkbox(&mut ui.replay_active, "Replay Active");
                        panel.checkbox(&mut ui.use_checkpoint_policy, "Use Checkpoint Policy");
                        panel.horizontal(|row| {
                            row.label("Checkpoint");
                            row.text_edit_singleline(&mut ui.checkpoint_path);
                        });
                        if panel.button("Load Checkpoint").clicked() {
                            ui.request_load_checkpoint = true;
                        }
                        panel.label(format!("Policy Status: {}", ui.checkpoint_status));
                        if panel.button("Reset Selected").clicked() {
                            ui.request_reset_selected = true;
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

                        panel.separator();
                        panel.label(format!("Gate Count: {}", ui.gate_count));
                        panel.label(format!("Current Gate: {}", ui.current_gate));
                        panel.label(format!("Done: {}", ui.done));
                        panel.label(format!(
                            "Done Reason: {}",
                            format_done_reasons(ui.done_reason_bits)
                        ));
                        panel.label(format!("Reward: {:.3}", ui.reward));
                        panel.label(format!("Value Estimate: {:.3}", ui.last_value_estimate));
                        panel.label(format!("Progress: {:.3}", ui.progress));
                        panel.label(format!("Distance To Gate: {:.3}", ui.distance_to_gate));
                        panel.label(format!("Gate Alignment: {:.3}", ui.gate_alignment));
                        panel.label(format!(
                            "Mean Motor Thrust: {:.3}",
                            ui.mean_motor_thrust
                        ));
                        panel.label(format!(
                            "Position: {:.2}, {:.2}, {:.2}",
                            ui.position[0], ui.position[1], ui.position[2]
                        ));

                        panel.separator();
                        panel.label("Reward Breakdown");
                        panel.label(format!("  progress: +{:.3}", ui.progress_reward));
                        panel.label(format!("  distance: -{:.3}", ui.distance_penalty));
                        panel.label(format!("  alignment: +{:.3}", ui.alignment_reward));
                        panel.label(format!("  tilt: -{:.3}", ui.tilt_penalty));
                        panel.label(format!("  completion: +{:.3}", ui.completion_bonus));
                        panel.label(format!("  collision: -{:.3}", ui.collision_penalty));
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
    max_gates_per_env * 4
        + max_obstacles_per_env(max_gates_per_env)
        + DRONE_MODEL_INSTANCE_COUNT
        + 1
}

fn required_gate_capacity(course: &CourseSpec) -> usize {
    course.total_gate_count().max(1) as usize
}

fn visualizer_course() -> CourseSpec {
    CourseSpec::default_drone_course()
}

fn flatten_observation(observation: &Observation) -> Vec<f32> {
    vec![
        observation.position[0],
        observation.position[1],
        observation.position[2],
        observation.velocity[0],
        observation.velocity[1],
        observation.velocity[2],
        observation.attitude[0],
        observation.attitude[1],
        observation.attitude[2],
        observation.angular_velocity[0],
        observation.angular_velocity[1],
        observation.angular_velocity[2],
        observation.target_gate_position[0],
        observation.target_gate_position[1],
        observation.target_gate_position[2],
        observation.target_gate_forward[0],
        observation.target_gate_forward[1],
        observation.target_gate_forward[2],
        observation.progress,
        observation.distance_to_gate,
        observation.gate_alignment,
        observation.mean_motor_thrust,
    ]
}

fn format_done_reasons(done_reason_bits: u32) -> String {
    if done_reason_bits == 0 {
        return "none".to_string();
    }

    let mut labels = Vec::new();
    if done_reason_bits & DONE_REASON_COMPLETE != 0 {
        labels.push("complete");
    }
    if done_reason_bits & DONE_REASON_GATE_COLLISION != 0 {
        labels.push("gate_collision");
    }
    if done_reason_bits & DONE_REASON_OBSTACLE_COLLISION != 0 {
        labels.push("obstacle_collision");
    }
    if done_reason_bits & DONE_REASON_FLOOR_COLLISION != 0 {
        labels.push("floor_collision");
    }
    if done_reason_bits & DONE_REASON_OUT_OF_BOUNDS != 0 {
        labels.push("out_of_bounds");
    }
    if done_reason_bits & DONE_REASON_STEP_LIMIT != 0 {
        labels.push("step_limit");
    }
    if done_reason_bits & DONE_REASON_EXCESSIVE_TILT != 0 {
        labels.push("excessive_tilt");
    }
    labels.join(", ")
}

fn gate_bar_instances(gate: Gate) -> [RenderInstance; 4] {
    let forward = normalized_xz(gate.forward).unwrap_or([0.0, 0.0, 1.0]);
    let right = [forward[2], 0.0, -forward[0]];
    let up = [0.0, 1.0, 0.0];
    let hole_half_width = gate.half_extents[0].max(0.1);
    let hole_half_height = gate.half_extents[1].max(0.1);
    let depth_half = gate.half_extents[2].max(GATE_DEPTH_HALF);
    let thickness = (hole_half_width.min(hole_half_height) * 0.16).max(GATE_THICKNESS_MIN);
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

fn drone_instances(state: EnvState) -> Vec<RenderInstance> {
    let yaw = state.attitude[2];
    let (roll, pitch) = (state.attitude[0], state.attitude[1]);
    let (cy, sy) = (yaw.cos(), yaw.sin());
    let (cr, sr) = (roll.cos(), roll.sin());
    let (cp, sp) = (pitch.cos(), pitch.sin());
    let forward = [cy * cp, sp, sy * cp];
    let right = [cy * sp * sr + sy * cr, cp * sr, sy * sp * sr - cy * cr];
    let up = [cy * sp * cr - sy * sr, cp * cr, sy * sp * cr + cy * sr];

    let center = [state.position[0], state.position[1], state.position[2]];
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
