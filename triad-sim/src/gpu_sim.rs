use triad_gpu::{
    BindGroupError, BindingType, BufferCopy, BufferError, BufferUsage, ComputePassBuilder,
    CopyPassBuilder, ExecutableFrameGraph, FrameGraph, Renderer, ResourceRegistry, Result,
    ShaderStage, wgpu,
};

const SIM_WORKGROUP_SIZE: u32 = 64;

const STEP_SHADER: &str = r#"
struct EnvState {
    position: vec2<f32>,
    velocity: vec2<f32>,
    step_count: u32,
    done: u32,
    current_gate: u32,
    _pad0: u32,
};

struct Action {
    accel: vec2<f32>,
    _pad: vec2<f32>,
};

struct Observation {
    position: vec2<f32>,
    velocity: vec2<f32>,
    target_gate_position: vec2<f32>,
    progress: f32,
    _pad0: f32,
};

struct RewardDone {
    reward: f32,
    done: u32,
    _pad0: u32,
    _pad1: u32,
};

struct SimParams {
    dt_seconds: f32,
    bounds: f32,
    env_count: u32,
    max_steps: u32,
    max_gates_per_env: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct ResetParams {
    seed: u32,
    grammar_id: u32,
    difficulty: f32,
    _pad0: u32,
};

struct EnvLayoutHeader {
    gate_offset: u32,
    gate_count: u32,
    obstacle_offset: u32,
    obstacle_count: u32,
};

struct Gate {
    center: vec2<f32>,
    half_extents: vec2<f32>,
    forward: vec2<f32>,
    _pad0: vec2<f32>,
};

struct EnvStateBuffer {
    values: array<EnvState>,
};

struct ActionBuffer {
    values: array<Action>,
};

struct ObservationBuffer {
    values: array<Observation>,
};

struct RewardDoneBuffer {
    values: array<RewardDone>,
};

struct ResetMaskBuffer {
    values: array<u32>,
};

struct ResetParamsBuffer {
    values: array<ResetParams>,
};

struct EnvLayoutBuffer {
    values: array<EnvLayoutHeader>,
};

struct GateBuffer {
    values: array<Gate>,
};

@group(0) @binding(0) var<storage, read_write> states: EnvStateBuffer;
@group(0) @binding(1) var<storage, read> actions: ActionBuffer;
@group(0) @binding(2) var<storage, read_write> observations: ObservationBuffer;
@group(0) @binding(3) var<storage, read_write> reward_done: RewardDoneBuffer;
@group(0) @binding(4) var<storage, read_write> reset_mask: ResetMaskBuffer;
@group(0) @binding(5) var<uniform> params: SimParams;
@group(0) @binding(6) var<storage, read> reset_params: ResetParamsBuffer;
@group(0) @binding(7) var<storage, read_write> layouts: EnvLayoutBuffer;
@group(0) @binding(8) var<storage, read_write> gates: GateBuffer;

fn hash_to_unit(seed: u32) -> f32 {
    let mixed = seed * 747796405u + 2891336453u;
    let masked = mixed & 0x00ffffffu;
    return f32(masked) / 16777215.0;
}

fn gate_slot(env_index: u32, gate_index: u32) -> u32 {
    return env_index * params.max_gates_per_env + gate_index;
}

fn gate_count_for_difficulty(difficulty: f32) -> u32 {
    let max_gates = max(params.max_gates_per_env, 1u);
    let scaled = u32(round(clamp(difficulty, 0.0, 1.0) * f32(max(max_gates, 2u) - 2u)));
    return min(max_gates, max(2u, 2u + scaled));
}

fn reset_env(index: u32) {
    let reset = reset_params.values[index];
    let difficulty = clamp(reset.difficulty, 0.0, 1.0);
    let env_fraction = f32(index) / max(f32(params.env_count), 1.0);
    let base_angle =
        env_fraction * 6.283185307179586 + hash_to_unit(reset.seed ^ 0x12345u) * 6.283185307179586;
    let forward_axis = normalize(vec2<f32>(cos(base_angle), sin(base_angle)));
    let lateral_axis = vec2<f32>(-forward_axis.y, forward_axis.x);
    let radius = 0.15 + 0.2 * hash_to_unit(reset.seed ^ 0xabcdefu);
    let start_position = forward_axis * radius;
    let count = gate_count_for_difficulty(difficulty);
    let base_slot = gate_slot(index, 0u);

    layouts.values[index].gate_offset = base_slot;
    layouts.values[index].gate_count = count;
    layouts.values[index].obstacle_offset = 0u;
    layouts.values[index].obstacle_count = 0u;

    var gate_index = 0u;
    loop {
        if (gate_index >= params.max_gates_per_env) {
            break;
        }

        let slot = gate_slot(index, gate_index);
        if (gate_index < count) {
            let t = f32(gate_index) + 1.0;
            let spacing = 0.18 + difficulty * 0.12;
            let wobble = sin(
                t * 0.8
                    + hash_to_unit(reset.seed + gate_index * 17u + reset.grammar_id * 31u)
                        * 6.283185307179586,
            ) * (0.04 + difficulty * 0.22);
            gates.values[slot].center =
                start_position + forward_axis * (spacing * (t + 0.5)) + lateral_axis * wobble;
            gates.values[slot].half_extents = vec2<f32>(
                0.09 - difficulty * 0.025,
                0.09 - difficulty * 0.025,
            );
            gates.values[slot].forward = forward_axis;
            gates.values[slot]._pad0 = vec2<f32>(0.0, 0.0);
        } else {
            gates.values[slot].center = vec2<f32>(0.0, 0.0);
            gates.values[slot].half_extents = vec2<f32>(0.0, 0.0);
            gates.values[slot].forward = vec2<f32>(0.0, 0.0);
            gates.values[slot]._pad0 = vec2<f32>(0.0, 0.0);
        }

        gate_index = gate_index + 1u;
    }

    states.values[index].position = start_position;
    states.values[index].velocity = vec2<f32>(0.0, 0.0);
    states.values[index].step_count = 0u;
    states.values[index].done = 0u;
    states.values[index].current_gate = 0u;
    states.values[index]._pad0 = 0u;

    observations.values[index].position = start_position;
    observations.values[index].velocity = vec2<f32>(0.0, 0.0);
    observations.values[index].target_gate_position = gates.values[base_slot].center;
    observations.values[index].progress = 0.0;
    observations.values[index]._pad0 = 0.0;

    reward_done.values[index].reward = 0.0;
    reward_done.values[index].done = 0u;
    reward_done.values[index]._pad0 = 0u;
    reward_done.values[index]._pad1 = 0u;
    reset_mask.values[index] = 0u;
}

fn current_target_gate(index: u32, current_gate: u32) -> Gate {
    let header = layouts.values[index];
    let safe_count = max(header.gate_count, 1u);
    let safe_gate = min(current_gate, safe_count - 1u);
    return gates.values[header.gate_offset + safe_gate];
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= params.env_count) {
        return;
    }

    if (reset_mask.values[index] != 0u) {
        reset_env(index);
        return;
    }

    var state = states.values[index];
    var target_gate = current_target_gate(index, state.current_gate);

    if (state.done != 0u) {
        observations.values[index].position = state.position;
        observations.values[index].velocity = state.velocity;
        observations.values[index].target_gate_position = target_gate.center;
        observations.values[index].progress = observations.values[index].progress;
        reward_done.values[index].reward = 0.0;
        reward_done.values[index].done = 1u;
        return;
    }

    let accel = actions.values[index].accel;
    state.velocity = state.velocity + accel * params.dt_seconds;
    state.position = state.position + state.velocity * params.dt_seconds;
    state.step_count = state.step_count + 1u;

    let header = layouts.values[index];
    var delta = target_gate.center - state.position;
    var distance_to_gate = length(delta);
    let gate_radius = max(target_gate.half_extents.x, target_gate.half_extents.y) * 1.5;
    if (distance_to_gate <= gate_radius) {
        if (state.current_gate + 1u < header.gate_count) {
            state.current_gate = state.current_gate + 1u;
            target_gate = current_target_gate(index, state.current_gate);
            delta = target_gate.center - state.position;
            distance_to_gate = length(delta);
        } else {
            state.done = 1u;
        }
    }

    let out_of_bounds = abs(state.position.x) > params.bounds
        || abs(state.position.y) > params.bounds;
    let reached_step_limit = state.step_count >= params.max_steps;
    if (out_of_bounds || reached_step_limit) {
        state.done = 1u;
    }

    states.values[index] = state;

    observations.values[index].position = state.position;
    observations.values[index].velocity = state.velocity;
    observations.values[index].target_gate_position = target_gate.center;
    observations.values[index].progress = f32(state.current_gate) / max(f32(header.gate_count), 1.0);
    observations.values[index]._pad0 = 0.0;

    reward_done.values[index].reward =
        observations.values[index].progress * 2.0 - distance_to_gate;
    reward_done.values[index].done = state.done;
    reward_done.values[index]._pad0 = 0u;
    reward_done.values[index]._pad1 = 0u;
}
"#;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EnvState {
    pub position: [f32; 2],
    pub velocity: [f32; 2],
    pub step_count: u32,
    pub done: u32,
    pub current_gate: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Action {
    pub accel: [f32; 2],
    pub _pad: [f32; 2],
}

impl Action {
    #[must_use]
    pub fn new(accel: [f32; 2]) -> Self {
        Self {
            accel,
            _pad: [0.0; 2],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Observation {
    pub position: [f32; 2],
    pub velocity: [f32; 2],
    pub target_gate_position: [f32; 2],
    pub progress: f32,
    pub _pad: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RewardDone {
    pub reward: f32,
    pub done: u32,
    pub _pad: [u32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ResetParams {
    pub seed: u32,
    pub grammar_id: u32,
    pub difficulty: f32,
    pub _pad: u32,
}

impl ResetParams {
    #[must_use]
    pub fn new(seed: u32, grammar_id: u32, difficulty: f32) -> Self {
        Self {
            seed,
            grammar_id,
            difficulty,
            _pad: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EnvLayoutHeader {
    pub gate_offset: u32,
    pub gate_count: u32,
    pub obstacle_offset: u32,
    pub obstacle_count: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Gate {
    pub center: [f32; 2],
    pub half_extents: [f32; 2],
    pub forward: [f32; 2],
    pub _pad: [f32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SimParams {
    dt_seconds: f32,
    bounds: f32,
    env_count: u32,
    max_steps: u32,
    max_gates_per_env: u32,
    _pad: [u32; 3],
}

#[derive(Debug, Clone, Copy)]
pub struct GpuSimulationConfig {
    pub env_count: usize,
    pub dt_seconds: f32,
    pub bounds: f32,
    pub max_steps: u32,
    pub max_gates_per_env: usize,
}

impl Default for GpuSimulationConfig {
    fn default() -> Self {
        Self {
            env_count: 1024,
            dt_seconds: 1.0 / 60.0,
            bounds: 1.0,
            max_steps: 512,
            max_gates_per_env: 8,
        }
    }
}

/// Minimal GPU-first simulation core for future headless training and visualization runners.
///
/// The authoritative environment state lives in GPU buffers. Callers step the simulation via
/// a cached compute graph and only trigger CPU readback explicitly.
pub struct GpuSimulation {
    config: GpuSimulationConfig,
    state: triad_gpu::GpuBuffer<EnvState>,
    actions: triad_gpu::GpuBuffer<Action>,
    observations: triad_gpu::GpuBuffer<Observation>,
    reward_done: triad_gpu::GpuBuffer<RewardDone>,
    reset_mask: triad_gpu::GpuBuffer<u32>,
    reset_params: triad_gpu::GpuBuffer<ResetParams>,
    layout_headers: triad_gpu::GpuBuffer<EnvLayoutHeader>,
    gates: triad_gpu::GpuBuffer<Gate>,
    params: triad_gpu::GpuBuffer<SimParams>,
    state_readback: triad_gpu::GpuBuffer<EnvState>,
    observation_readback: triad_gpu::GpuBuffer<Observation>,
    reward_done_readback: triad_gpu::GpuBuffer<RewardDone>,
    layout_headers_readback: triad_gpu::GpuBuffer<EnvLayoutHeader>,
    gates_readback: triad_gpu::GpuBuffer<Gate>,
    step_pipeline: triad_gpu::Handle<wgpu::ComputePipeline>,
    step_bind_group: triad_gpu::Handle<wgpu::BindGroup>,
    step_graph: ExecutableFrameGraph,
    state_readback_graph: ExecutableFrameGraph,
    observation_readback_graph: ExecutableFrameGraph,
    reward_done_readback_graph: ExecutableFrameGraph,
    layout_headers_readback_graph: ExecutableFrameGraph,
    gates_readback_graph: ExecutableFrameGraph,
}

impl GpuSimulation {
    pub fn new(
        renderer: &Renderer,
        registry: &mut ResourceRegistry,
        config: GpuSimulationConfig,
    ) -> Result<Self> {
        if config.env_count == 0 {
            return Err(BufferError::CapacityExceeded {
                requested: 1,
                capacity: 0,
            }
            .into());
        }
        if config.max_gates_per_env == 0 {
            return Err(BufferError::CapacityExceeded {
                requested: 1,
                capacity: 0,
            }
            .into());
        }

        let reset_values = vec![1u32; config.env_count];
        let reset_params_values: Vec<ResetParams> = (0..config.env_count)
            .map(|env_index| ResetParams::new(env_index as u32, 0, 0.35))
            .collect();
        let params = SimParams {
            dt_seconds: config.dt_seconds,
            bounds: config.bounds,
            env_count: config.env_count as u32,
            max_steps: config.max_steps,
            max_gates_per_env: config.max_gates_per_env as u32,
            _pad: [0; 3],
        };
        let gate_capacity = config.env_count * config.max_gates_per_env;

        let state = renderer
            .create_gpu_buffer::<EnvState>()
            .label("sim state")
            .capacity(config.env_count)
            .add_usage(wgpu::BufferUsages::COPY_SRC)
            .build(registry)?;

        let actions = renderer
            .create_gpu_buffer::<Action>()
            .label("sim actions")
            .capacity(config.env_count)
            .build(registry)?;

        let observations = renderer
            .create_gpu_buffer::<Observation>()
            .label("sim observations")
            .capacity(config.env_count)
            .add_usage(wgpu::BufferUsages::COPY_SRC)
            .build(registry)?;

        let reward_done = renderer
            .create_gpu_buffer::<RewardDone>()
            .label("sim reward/done")
            .capacity(config.env_count)
            .add_usage(wgpu::BufferUsages::COPY_SRC)
            .build(registry)?;

        let reset_mask = renderer
            .create_gpu_buffer::<u32>()
            .label("sim reset mask")
            .with_data(&reset_values)
            .build(registry)?;

        let reset_params = renderer
            .create_gpu_buffer::<ResetParams>()
            .label("sim reset params")
            .with_data(&reset_params_values)
            .build(registry)?;

        let layout_headers = renderer
            .create_gpu_buffer::<EnvLayoutHeader>()
            .label("sim env layout headers")
            .capacity(config.env_count)
            .add_usage(wgpu::BufferUsages::COPY_SRC)
            .build(registry)?;

        let gates = renderer
            .create_gpu_buffer::<Gate>()
            .label("sim packed gates")
            .capacity(gate_capacity)
            .add_usage(wgpu::BufferUsages::COPY_SRC)
            .build(registry)?;

        let params = renderer
            .create_gpu_buffer::<SimParams>()
            .label("sim params")
            .with_data(&[params])
            .usage(BufferUsage::Uniform)
            .build(registry)?;

        let state_readback = renderer
            .create_gpu_buffer::<EnvState>()
            .label("sim state readback")
            .capacity(config.env_count)
            .usage(BufferUsage::Readback)
            .build(registry)?;

        let observation_readback = renderer
            .create_gpu_buffer::<Observation>()
            .label("sim observation readback")
            .capacity(config.env_count)
            .usage(BufferUsage::Readback)
            .build(registry)?;

        let reward_done_readback = renderer
            .create_gpu_buffer::<RewardDone>()
            .label("sim reward/done readback")
            .capacity(config.env_count)
            .usage(BufferUsage::Readback)
            .build(registry)?;

        let layout_headers_readback = renderer
            .create_gpu_buffer::<EnvLayoutHeader>()
            .label("sim layout headers readback")
            .capacity(config.env_count)
            .usage(BufferUsage::Readback)
            .build(registry)?;

        let gates_readback = renderer
            .create_gpu_buffer::<Gate>()
            .label("sim gates readback")
            .capacity(gate_capacity)
            .usage(BufferUsage::Readback)
            .build(registry)?;

        let shader = renderer
            .create_shader_module()
            .label("gpu simulation step")
            .with_wgsl_source(STEP_SHADER)
            .build(registry)?;

        let (bind_group_layout, step_bind_group) = renderer
            .create_bind_group()
            .label("gpu simulation step")
            .buffer_stage(
                0,
                ShaderStage::Compute,
                state.handle(),
                BindingType::StorageWrite,
            )
            .buffer_stage(
                1,
                ShaderStage::Compute,
                actions.handle(),
                BindingType::StorageRead,
            )
            .buffer_stage(
                2,
                ShaderStage::Compute,
                observations.handle(),
                BindingType::StorageWrite,
            )
            .buffer_stage(
                3,
                ShaderStage::Compute,
                reward_done.handle(),
                BindingType::StorageWrite,
            )
            .buffer_stage(
                4,
                ShaderStage::Compute,
                reset_mask.handle(),
                BindingType::StorageWrite,
            )
            .buffer_stage(
                5,
                ShaderStage::Compute,
                params.handle(),
                BindingType::Uniform,
            )
            .buffer_stage(
                6,
                ShaderStage::Compute,
                reset_params.handle(),
                BindingType::StorageRead,
            )
            .buffer_stage(
                7,
                ShaderStage::Compute,
                layout_headers.handle(),
                BindingType::StorageWrite,
            )
            .buffer_stage(
                8,
                ShaderStage::Compute,
                gates.handle(),
                BindingType::StorageWrite,
            )
            .build(registry)?;

        let bind_group_layout_ref = registry
            .get(bind_group_layout)
            .ok_or(BindGroupError::LayoutNotFound)?;
        let pipeline_layout =
            renderer
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("gpu simulation step layout"),
                    bind_group_layouts: std::slice::from_ref(&bind_group_layout_ref),
                    push_constant_ranges: &[],
                });

        let step_pipeline = renderer
            .create_compute_pipeline()
            .with_label("gpu simulation step pipeline")
            .with_compute_shader(shader)
            .with_layout(pipeline_layout)
            .build(registry)?;

        let dispatch_count = dispatch_count(config.env_count);
        let step_graph = build_step_graph(
            step_pipeline,
            step_bind_group,
            state.handle(),
            actions.handle(),
            observations.handle(),
            reward_done.handle(),
            reset_mask.handle(),
            params.handle(),
            reset_params.handle(),
            layout_headers.handle(),
            gates.handle(),
            dispatch_count,
        )?;

        let state_readback_graph = build_readback_graph(
            "GpuSimulationReadbackState",
            state.handle(),
            state.size_bytes(),
            state_readback.handle(),
        )?;
        let observation_readback_graph = build_readback_graph(
            "GpuSimulationReadbackObservation",
            observations.handle(),
            observations.size_bytes(),
            observation_readback.handle(),
        )?;
        let reward_done_readback_graph = build_readback_graph(
            "GpuSimulationReadbackRewardDone",
            reward_done.handle(),
            reward_done.size_bytes(),
            reward_done_readback.handle(),
        )?;
        let layout_headers_readback_graph = build_readback_graph(
            "GpuSimulationReadbackLayoutHeaders",
            layout_headers.handle(),
            layout_headers.size_bytes(),
            layout_headers_readback.handle(),
        )?;
        let gates_readback_graph = build_readback_graph(
            "GpuSimulationReadbackGates",
            gates.handle(),
            gates.size_bytes(),
            gates_readback.handle(),
        )?;

        Ok(Self {
            config,
            state,
            actions,
            observations,
            reward_done,
            reset_mask,
            reset_params,
            layout_headers,
            gates,
            params,
            state_readback,
            observation_readback,
            reward_done_readback,
            layout_headers_readback,
            gates_readback,
            step_pipeline,
            step_bind_group,
            step_graph,
            state_readback_graph,
            observation_readback_graph,
            reward_done_readback_graph,
            layout_headers_readback_graph,
            gates_readback_graph,
        })
    }

    #[must_use]
    pub fn config(&self) -> GpuSimulationConfig {
        self.config
    }

    #[must_use]
    pub fn env_count(&self) -> usize {
        self.config.env_count
    }

    #[must_use]
    pub fn gate_capacity(&self) -> usize {
        self.config.env_count * self.config.max_gates_per_env
    }

    #[must_use]
    pub fn state_buffer(&self) -> triad_gpu::Handle<wgpu::Buffer> {
        self.state.handle()
    }

    #[must_use]
    pub fn action_buffer(&self) -> triad_gpu::Handle<wgpu::Buffer> {
        self.actions.handle()
    }

    #[must_use]
    pub fn observation_buffer(&self) -> triad_gpu::Handle<wgpu::Buffer> {
        self.observations.handle()
    }

    #[must_use]
    pub fn reward_done_buffer(&self) -> triad_gpu::Handle<wgpu::Buffer> {
        self.reward_done.handle()
    }

    #[must_use]
    pub fn reset_mask_buffer(&self) -> triad_gpu::Handle<wgpu::Buffer> {
        self.reset_mask.handle()
    }

    #[must_use]
    pub fn reset_params_buffer(&self) -> triad_gpu::Handle<wgpu::Buffer> {
        self.reset_params.handle()
    }

    #[must_use]
    pub fn layout_headers_buffer(&self) -> triad_gpu::Handle<wgpu::Buffer> {
        self.layout_headers.handle()
    }

    #[must_use]
    pub fn gates_buffer(&self) -> triad_gpu::Handle<wgpu::Buffer> {
        self.gates.handle()
    }

    #[must_use]
    pub fn params_buffer(&self) -> triad_gpu::Handle<wgpu::Buffer> {
        self.params.handle()
    }

    #[must_use]
    pub fn step_pipeline(&self) -> triad_gpu::Handle<wgpu::ComputePipeline> {
        self.step_pipeline
    }

    #[must_use]
    pub fn step_bind_group(&self) -> triad_gpu::Handle<wgpu::BindGroup> {
        self.step_bind_group
    }

    pub fn set_actions(
        &self,
        renderer: &Renderer,
        registry: &ResourceRegistry,
        actions: &[Action],
    ) -> Result<()> {
        if actions.len() > self.config.env_count {
            return Err(BufferError::CapacityExceeded {
                requested: actions.len(),
                capacity: self.config.env_count,
            }
            .into());
        }

        renderer.write_buffer(self.actions.handle(), actions, registry)?;
        Ok(())
    }

    pub fn set_reset_params(
        &self,
        renderer: &Renderer,
        registry: &ResourceRegistry,
        reset_params: &[ResetParams],
    ) -> Result<()> {
        if reset_params.len() > self.config.env_count {
            return Err(BufferError::CapacityExceeded {
                requested: reset_params.len(),
                capacity: self.config.env_count,
            }
            .into());
        }

        renderer.write_buffer(self.reset_params.handle(), reset_params, registry)?;
        Ok(())
    }

    pub fn reset_all(&self, renderer: &Renderer, registry: &ResourceRegistry) -> Result<()> {
        let reset_values = vec![1u32; self.config.env_count];
        renderer.write_buffer(self.reset_mask.handle(), &reset_values, registry)?;
        Ok(())
    }

    pub fn request_resets(
        &self,
        renderer: &Renderer,
        registry: &ResourceRegistry,
        env_indices: &[usize],
    ) -> Result<()> {
        for &index in env_indices {
            if index >= self.config.env_count {
                return Err(BufferError::CapacityExceeded {
                    requested: index + 1,
                    capacity: self.config.env_count,
                }
                .into());
            }
            renderer.write_buffer_offset(
                self.reset_mask.handle(),
                (index * std::mem::size_of::<u32>()) as u64,
                &[1u32],
                registry,
            )?;
        }
        Ok(())
    }

    pub fn step(&mut self, renderer: &Renderer, registry: &ResourceRegistry) {
        self.step_graph
            .execute(renderer.device(), renderer.queue(), registry);
    }

    pub fn step_n(&mut self, renderer: &Renderer, registry: &ResourceRegistry, steps: usize) {
        for _ in 0..steps {
            self.step(renderer, registry);
        }
    }

    pub fn readback_state(
        &mut self,
        renderer: &Renderer,
        registry: &ResourceRegistry,
    ) -> Result<Vec<EnvState>> {
        self.state_readback_graph
            .execute(renderer.device(), renderer.queue(), registry);
        Ok(renderer.read_buffer(self.state_readback.handle(), registry)?)
    }

    pub fn readback_observations(
        &mut self,
        renderer: &Renderer,
        registry: &ResourceRegistry,
    ) -> Result<Vec<Observation>> {
        self.observation_readback_graph
            .execute(renderer.device(), renderer.queue(), registry);
        Ok(renderer.read_buffer(self.observation_readback.handle(), registry)?)
    }

    pub fn readback_reward_done(
        &mut self,
        renderer: &Renderer,
        registry: &ResourceRegistry,
    ) -> Result<Vec<RewardDone>> {
        self.reward_done_readback_graph
            .execute(renderer.device(), renderer.queue(), registry);
        Ok(renderer.read_buffer(self.reward_done_readback.handle(), registry)?)
    }

    pub fn readback_layout_headers(
        &mut self,
        renderer: &Renderer,
        registry: &ResourceRegistry,
    ) -> Result<Vec<EnvLayoutHeader>> {
        self.layout_headers_readback_graph
            .execute(renderer.device(), renderer.queue(), registry);
        Ok(renderer.read_buffer(self.layout_headers_readback.handle(), registry)?)
    }

    pub fn readback_gates(
        &mut self,
        renderer: &Renderer,
        registry: &ResourceRegistry,
    ) -> Result<Vec<Gate>> {
        self.gates_readback_graph
            .execute(renderer.device(), renderer.queue(), registry);
        Ok(renderer.read_buffer(self.gates_readback.handle(), registry)?)
    }
}

fn dispatch_count(env_count: usize) -> u32 {
    ((env_count as u32) + SIM_WORKGROUP_SIZE - 1) / SIM_WORKGROUP_SIZE
}

fn build_step_graph(
    pipeline: triad_gpu::Handle<wgpu::ComputePipeline>,
    bind_group: triad_gpu::Handle<wgpu::BindGroup>,
    state: triad_gpu::Handle<wgpu::Buffer>,
    actions: triad_gpu::Handle<wgpu::Buffer>,
    observations: triad_gpu::Handle<wgpu::Buffer>,
    reward_done: triad_gpu::Handle<wgpu::Buffer>,
    reset_mask: triad_gpu::Handle<wgpu::Buffer>,
    params: triad_gpu::Handle<wgpu::Buffer>,
    reset_params: triad_gpu::Handle<wgpu::Buffer>,
    layout_headers: triad_gpu::Handle<wgpu::Buffer>,
    gates: triad_gpu::Handle<wgpu::Buffer>,
    dispatch_x: u32,
) -> Result<ExecutableFrameGraph> {
    let pass = ComputePassBuilder::new("GpuSimulationStep")
        .read_write(state)
        .read(actions)
        .write(observations)
        .write(reward_done)
        .read_write(reset_mask)
        .read(params)
        .read(reset_params)
        .write(layout_headers)
        .write(gates)
        .with_pipeline(pipeline)
        .with_bind_group(0, bind_group)
        .dispatch(dispatch_x, 1, 1)
        .build()?;

    let mut graph = FrameGraph::new();
    graph.add_pass(pass);
    Ok(graph.build()?)
}

fn build_readback_graph(
    name: &str,
    src: triad_gpu::Handle<wgpu::Buffer>,
    size: u64,
    dst: triad_gpu::Handle<wgpu::Buffer>,
) -> Result<ExecutableFrameGraph> {
    let pass = CopyPassBuilder::new(name)
        .copy_buffer_region(BufferCopy::new(src, dst, size))
        .build()?;

    let mut graph = FrameGraph::new();
    graph.add_pass(pass);
    Ok(graph.build()?)
}
