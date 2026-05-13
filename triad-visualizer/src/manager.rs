use std::collections::VecDeque;
use std::error::Error;
use std::sync::{Arc, Mutex};

use glam::Vec3;
use triad_gpu::{
    BindingType, BufferUsage, ColorLoadOp, DepthLoadOp, ExecutableFrameGraph, FrameGraphError,
    FrameTextureView, RenderPassBuilder, Renderer, ResourceRegistry, ShaderStage, wgpu,
};
use triad_sim::{
    Action, CourseSpec, EnvLayoutHeader, EnvState, Gate, GpuSimulation, GpuSimulationConfig,
    Observation, ResetParams, RewardDone,
};
use triad_window::{CameraUniforms, RendererManager};

use crate::constants::VISUALIZER_ENV_COUNT;
use crate::render::{
    RenderInstance, TRAIL_MAX_POINTS, TRAIL_MIN_POINT_DISTANCE, TRAIL_RESET_DISTANCE,
    VISUALIZER_SHADER, debug_vector_instances, drone_instances, floor_instance, gate_bar_instances,
    obstacle_instance, required_gate_capacity, target_instance, trail_instances,
    visible_instance_capacity,
};
use crate::replay::{ReplayPlayback, apply_replay_controller, terminal_pause_status};
use crate::ui::{UiSnapshot, UiState, curriculum_phase_profile};

const PLAYING_READBACK_INTERVAL_FRAMES: u32 = 4;

pub(crate) struct VisualizerManager {
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
    trail_points: Vec<VecDeque<[f32; 3]>>,
    selected_env: usize,
    layouts_dirty: bool,
    frame_index: u64,
    applied_difficulty: f32,
    applied_curriculum_phase: usize,
    applied_curriculum_stage: u32,
}

impl VisualizerManager {
    pub(crate) fn new(
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

        let zero_actions = vec![Action::idle(); sim.env_count()];
        let trail_points = (0..sim.env_count())
            .map(|_| VecDeque::with_capacity(TRAIL_MAX_POINTS))
            .collect();
        sim.set_actions(renderer, registry, &zero_actions)?;

        let mut manager = Self {
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
            trail_points,
            selected_env: 0,
            layouts_dirty: true,
            frame_index: 0,
            applied_difficulty: 0.35,
            applied_curriculum_phase: 0,
            applied_curriculum_stage: 0,
        };

        let (seed_base, difficulty, curriculum_phase, curriculum_stage) = {
            let state = manager.ui_state.lock().expect("ui state poisoned");
            let curriculum_phase = state.curriculum_phase;
            let curriculum_stage = curriculum_phase_profile(curriculum_phase).curriculum_stage;
            (
                state.seed_base,
                state.difficulty,
                curriculum_phase,
                curriculum_stage,
            )
        };
        let params = manager.randomize_reset_params(seed_base, difficulty, curriculum_phase);
        manager.sim.set_reset_params(renderer, registry, &params)?;
        manager.sim.reset_all(renderer, registry)?;
        manager.applied_difficulty = difficulty;
        manager.applied_curriculum_phase = curriculum_phase;
        manager.applied_curriculum_stage = curriculum_stage;

        Ok(manager)
    }

    fn snapshot_ui(&self) -> UiSnapshot {
        let mut state = self.ui_state.lock().expect("ui state poisoned");
        state.snapshot(self.sim.env_count())
    }

    fn set_replay_status(&self, status: impl Into<String>) {
        self.ui_state
            .lock()
            .expect("ui state poisoned")
            .replay
            .status = status.into();
    }

    fn randomize_reset_params(
        &self,
        base_seed: u32,
        difficulty: f32,
        curriculum_phase: usize,
    ) -> Vec<ResetParams> {
        let profile = curriculum_phase_profile(curriculum_phase);
        let difficulty_span = (profile.difficulty_max - profile.difficulty_min).max(0.0);
        let target_difficulty =
            profile.difficulty_min + difficulty.clamp(0.0, 1.0) * difficulty_span;
        let jitter_span = difficulty_span * 0.2;
        (0..self.sim.env_count())
            .map(|env_index| {
                let env_seed = hash_u32(base_seed ^ (env_index as u32).wrapping_mul(0x9e37_79b9));
                let grammar_index = (env_seed as usize) % profile.grammar_ids.len();
                let grammar_id = profile.grammar_ids[grammar_index];
                let difficulty_jitter =
                    (hash_to_unit(env_seed ^ 0x85eb_ca6b) * 2.0 - 1.0) * jitter_span;
                let env_difficulty = (target_difficulty + difficulty_jitter)
                    .clamp(profile.difficulty_min, profile.difficulty_max);
                ResetParams::new(
                    env_seed,
                    grammar_id,
                    env_difficulty,
                    profile.curriculum_stage,
                )
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

    fn refresh_dynamic_caches(
        &mut self,
        renderer: &Renderer,
        registry: &ResourceRegistry,
    ) -> Result<(), Box<dyn Error>> {
        self.refresh_state_cache(renderer, registry)?;
        self.refresh_observation_cache(renderer, registry)?;
        self.refresh_reward_done_cache(renderer, registry)?;
        self.update_trails();
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

    fn apply_replay_actions(&mut self, snapshot: UiSnapshot) {
        let layouts = &self.cached_layouts;
        let gates = &self.cached_gates;
        let status = apply_replay_controller(
            snapshot.replay.controller,
            &mut self.actions,
            &self.cached_states,
            |env_index, state| target_gate_for_env(layouts, gates, env_index, state),
        );
        self.set_replay_status(status);
    }

    fn rebuild_instances(&mut self, selected_state: Option<EnvState>) {
        self.instances.fill(RenderInstance::hidden());

        let Some(layout) = self.cached_layouts.get(self.selected_env).copied() else {
            return;
        };

        let mut write_index = 0usize;
        if let Some(slot) = self.instances.get_mut(write_index) {
            *slot = floor_instance(self.sim.config().bounds);
            write_index += 1;
        }
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
            if let Some(trail) = self.trail_points.get(self.selected_env) {
                for instance in trail_instances(trail) {
                    if let Some(slot) = self.instances.get_mut(write_index) {
                        *slot = instance;
                        write_index += 1;
                    }
                }
            }
            for instance in drone_instances(state) {
                if let Some(slot) = self.instances.get_mut(write_index) {
                    *slot = instance;
                    write_index += 1;
                }
            }
            for instance in debug_vector_instances(state) {
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
        } else {
            ui.gate_count = 0;
            ui.current_gate = 0;
            ui.done = false;
            ui.position = [0.0, 0.0, 0.0];
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
            ui.shaping_reward = reward_done.shaping_reward;
            ui.time_penalty = reward_done.time_penalty;
            ui.sparse_objective_reward = reward_done.sparse_objective_reward;
            ui.collision_penalty = reward_done.collision_penalty;
        } else {
            ui.reward = 0.0;
            ui.done_reason_bits = 0;
            ui.shaping_reward = 0.0;
            ui.time_penalty = 0.0;
            ui.sparse_objective_reward = 0.0;
            ui.collision_penalty = 0.0;
        }
    }

    fn clear_trail(&mut self, env_index: usize) {
        if let Some(trail) = self.trail_points.get_mut(env_index) {
            trail.clear();
        }
    }

    fn clear_all_trails(&mut self) {
        for trail in &mut self.trail_points {
            trail.clear();
        }
    }

    fn update_trails(&mut self) {
        for (env_index, state) in self.cached_states.iter().enumerate() {
            let Some(trail) = self.trail_points.get_mut(env_index) else {
                continue;
            };
            let position = state.position;
            let point = vec3_from_array(position);

            match trail.back().copied() {
                None => {
                    trail.push_back(position);
                }
                Some(last_position) => {
                    if state.done != 0 {
                        continue;
                    }

                    let distance = (point - vec3_from_array(last_position)).length();
                    if distance <= TRAIL_MIN_POINT_DISTANCE {
                        continue;
                    }
                    if distance >= TRAIL_RESET_DISTANCE {
                        trail.clear();
                    }
                    if trail.len() == TRAIL_MAX_POINTS {
                        trail.pop_front();
                    }
                    trail.push_back(position);
                }
            }
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
        self.frame_index = self.frame_index.wrapping_add(1);

        let snapshot = self.snapshot_ui();
        self.selected_env = snapshot.selected_env;
        if snapshot.replay.load_model_requested {
            if snapshot.replay.model_path.trim().is_empty() {
                self.set_replay_status("Model load skipped: no model path selected");
            } else {
                self.set_replay_status(format!("Model selected: {}", snapshot.replay.model_path));
            }
        }
        let replay_advancing = snapshot.replay.should_advance();
        let continuous_replay = snapshot.replay.playback == ReplayPlayback::Playing;
        let throttle_readback = continuous_replay && !snapshot.replay.step_requested;
        let readback_due = !throttle_readback
            || self.frame_index % u64::from(PLAYING_READBACK_INTERVAL_FRAMES) == 0;
        let generation_changed = (snapshot.difficulty - self.applied_difficulty).abs() > 1e-5
            || snapshot.curriculum_phase != self.applied_curriculum_phase;

        let mut forced_reset_step = false;
        if self.layouts_dirty {
            self.sim.step(renderer, registry);
            forced_reset_step = true;
        }

        if snapshot.randomize || generation_changed {
            let params = self.randomize_reset_params(
                snapshot.seed_base,
                snapshot.difficulty,
                snapshot.curriculum_phase,
            );
            self.sim.set_reset_params(renderer, registry, &params)?;
            self.sim.reset_all(renderer, registry)?;
            self.clear_all_trails();
            self.sim.step(renderer, registry);
            self.layouts_dirty = true;
            forced_reset_step = true;
            self.applied_difficulty = snapshot.difficulty;
            self.applied_curriculum_phase = snapshot.curriculum_phase;
            self.applied_curriculum_stage = snapshot.curriculum_stage;
        } else if snapshot.reset_selected {
            self.sim
                .request_resets(renderer, registry, &[self.selected_env])?;
            self.clear_trail(self.selected_env);
            self.sim.step(renderer, registry);
            forced_reset_step = true;
        } else if snapshot.reset_all {
            let params = self.randomize_reset_params(
                snapshot.seed_base,
                snapshot.difficulty,
                snapshot.curriculum_phase,
            );
            self.sim.set_reset_params(renderer, registry, &params)?;
            self.sim.reset_all(renderer, registry)?;
            self.clear_all_trails();
            self.sim.step(renderer, registry);
            self.layouts_dirty = true;
            forced_reset_step = true;
        }

        if self.layouts_dirty || forced_reset_step {
            self.refresh_layout_cache(renderer, registry)?;
        }

        if !replay_advancing || readback_due {
            self.refresh_dynamic_caches(renderer, registry)?;
        }

        if replay_advancing {
            self.apply_replay_actions(snapshot.clone());
            self.sim.set_actions(renderer, registry, &self.actions)?;
            self.sim.step(renderer, registry);
            if readback_due {
                self.refresh_dynamic_caches(renderer, registry)?;
            }
        }

        let selected_state = self.cached_states.get(self.selected_env).copied();
        let selected_observation = self.cached_observations.get(self.selected_env).copied();
        let selected_reward_done = self.cached_reward_done.get(self.selected_env).copied();
        if snapshot.replay.auto_pause_on_terminal
            && snapshot.replay.playback == ReplayPlayback::Playing
            && selected_reward_done
                .map(|value| value.done != 0)
                .unwrap_or(false)
        {
            {
                let mut ui = self.ui_state.lock().expect("ui state poisoned");
                ui.replay.playback = ReplayPlayback::Paused;
            }
            if let Some(reward_done) = selected_reward_done {
                self.set_replay_status(terminal_pause_status(reward_done));
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

fn visualizer_course() -> CourseSpec {
    CourseSpec::default_drone_course()
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

fn vec3_from_array(value: [f32; 3]) -> Vec3 {
    Vec3::new(value[0], value[1], value[2])
}

fn target_gate_for_env(
    layouts: &[EnvLayoutHeader],
    gates: &[Gate],
    env_index: usize,
    state: &EnvState,
) -> Option<Gate> {
    let layout = *layouts.get(env_index)?;
    if layout.gate_count == 0 {
        return None;
    }
    let gate_index = state.current_gate.min(layout.gate_count.saturating_sub(1));
    gates
        .get((layout.gate_offset + gate_index) as usize)
        .copied()
}
