use triad_gpu::{
    BindGroupError, BindingType, BufferCopy, BufferError, BufferUsage, ComputePassBuilder,
    CopyPassBuilder, ExecutableFrameGraph, FrameGraph, Renderer, ResourceRegistry, Result,
    ShaderStage, wgpu,
};

use crate::course::{CompiledCourse, CourseSpec, GpuCourseHeader, GpuStageSpec};

const SIM_WORKGROUP_SIZE: u32 = 64;
const COURSE_STAGE_CAPACITY: usize = 32;

const STEP_SHADER: &str = r#"
struct EnvState {
    position: vec2<f32>,
    velocity: vec2<f32>,
    step_count: u32,
    done: u32,
    current_gate: u32,
    current_lap: u32,
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

struct CourseHeader {
    stage_count: u32,
    total_gate_count: u32,
    loop_enabled: u32,
    laps_required: u32,
};

struct StageSpec {
    kind: u32,
    gate_count: u32,
    flags: u32,
    _pad0: u32,
    spacing: f32,
    lateral_amp: f32,
    turn_radians: f32,
    radius: f32,
    vertical_amp: f32,
    hole_half_width: f32,
    hole_half_height: f32,
    _pad1: f32,
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

struct GateBuffer {
    values: array<Gate>,
};

struct StageBuffer {
    values: array<StageSpec>,
};

@group(0) @binding(0) var<storage, read_write> states: EnvStateBuffer;
@group(0) @binding(1) var<storage, read> actions: ActionBuffer;
@group(0) @binding(2) var<storage, read_write> observations: ObservationBuffer;
@group(0) @binding(3) var<storage, read_write> reward_done: RewardDoneBuffer;
@group(0) @binding(4) var<storage, read_write> reset_mask: ResetMaskBuffer;
@group(0) @binding(5) var<uniform> params: SimParams;
@group(0) @binding(6) var<storage, read> reset_params: ResetParamsBuffer;
@group(0) @binding(7) var<storage, read_write> gates: GateBuffer;
@group(0) @binding(8) var<uniform> course_header: CourseHeader;
@group(0) @binding(9) var<storage, read> course_stages: StageBuffer;

fn hash_to_unit(seed: u32) -> f32 {
    let mixed = seed * 747796405u + 2891336453u;
    let masked = mixed & 0x00ffffffu;
    return f32(masked) / 16777215.0;
}

fn rotate_vec2(v: vec2<f32>, angle: f32) -> vec2<f32> {
    let s = sin(angle);
    let c = cos(angle);
    return vec2<f32>(c * v.x - s * v.y, s * v.x + c * v.y);
}

fn gate_slot(env_index: u32, gate_index: u32) -> u32 {
    return env_index * params.max_gates_per_env + gate_index;
}

fn stage_direction(stage: StageSpec) -> f32 {
    return select(-1.0, 1.0, stage.flags != 0u);
}

fn generated_gate_count() -> u32 {
    return min(course_header.total_gate_count, params.max_gates_per_env);
}

fn env_gate_offset(index: u32) -> u32 {
    return index * params.max_gates_per_env;
}

fn gate_from_stage(
    stage: StageSpec,
    cursor_position: vec2<f32>,
    cursor_forward: vec2<f32>,
    local_gate_index: u32,
    total_gate_index: u32,
    total_gate_count: u32,
    seed: u32,
) -> Gate {
    let lateral_axis = vec2<f32>(-cursor_forward.y, cursor_forward.x);
    let progress = (f32(total_gate_index) + 0.5) / max(f32(total_gate_count), 1.0);
    let elevation =
        sin(progress * 6.283185307179586 + hash_to_unit(seed ^ 0x44f1u) * 6.283185307179586)
            * stage.vertical_amp
        + (hash_to_unit(seed ^ (total_gate_index * 97u + 0x0f0fu)) - 0.5) * 0.12;

    var gate: Gate;
    if (stage.kind == 2u) {
        let t = (f32(local_gate_index) + 1.0) / max(f32(stage.gate_count), 1.0);
        let lateral = sin(t * 6.283185307179586) * stage.lateral_amp;
        gate.center = cursor_position + cursor_forward * stage.spacing + lateral_axis * lateral;
        gate.forward = cursor_forward;
    } else if (stage.kind == 3u) {
        let turn_sign = stage_direction(stage);
        let arc_center = cursor_position + lateral_axis * (turn_sign * stage.radius);
        let start_offset = cursor_position - arc_center;
        let angle =
            turn_sign * stage.turn_radians
            * ((f32(local_gate_index) + 1.0) / max(f32(stage.gate_count), 1.0));
        gate.center = arc_center + rotate_vec2(start_offset, angle);
        gate.forward = normalize(rotate_vec2(cursor_forward, angle));
    } else {
        gate.center = cursor_position + cursor_forward * stage.spacing;
        gate.forward = cursor_forward;
    }
    gate.half_extents = vec2<f32>(stage.hole_half_width, stage.hole_half_height);
    gate._pad0 = vec2<f32>(elevation, 0.0);
    return gate;
}

fn next_cursor_forward(stage: StageSpec, cursor_forward: vec2<f32>) -> vec2<f32> {
    if (stage.kind == 3u) {
        return normalize(rotate_vec2(
            cursor_forward,
            stage_direction(stage) * stage.turn_radians,
        ));
    }
    return cursor_forward;
}

fn progress_value(current_lap: u32, current_gate: u32, gate_count: u32) -> f32 {
    let laps_required = max(course_header.laps_required, 1u);
    let total_targets = max(gate_count * laps_required, 1u);
    let completed = current_lap * gate_count + current_gate;
    return f32(completed) / f32(total_targets);
}

fn reset_env(index: u32) {
    let reset = reset_params.values[index];
    let count = generated_gate_count();
    let base_slot = env_gate_offset(index);

    let course_angle =
        hash_to_unit(reset.seed ^ 0x12345u) * 6.283185307179586
        + f32(reset.grammar_id % 4u) * 1.5707963267948966;
    var cursor_position = vec2<f32>(0.0, 0.0);
    var cursor_forward = rotate_vec2(vec2<f32>(1.0, 0.0), course_angle);
    var gate_index = 0u;
    var stage_index = 0u;

    loop {
        if (stage_index >= course_header.stage_count || gate_index >= count) {
            break;
        }

        let stage = course_stages.values[stage_index];
        var local_gate_index = 0u;
        loop {
            if (local_gate_index >= stage.gate_count || gate_index >= count) {
                break;
            }

            let gate = gate_from_stage(
                stage,
                cursor_position,
                cursor_forward,
                local_gate_index,
                gate_index,
                count,
                reset.seed,
            );
            gates.values[gate_slot(index, gate_index)] = gate;
            cursor_position = gate.center;
            gate_index = gate_index + 1u;
            local_gate_index = local_gate_index + 1u;
        }

        cursor_forward = next_cursor_forward(stage, cursor_forward);
        stage_index = stage_index + 1u;
    }

    loop {
        if (gate_index >= params.max_gates_per_env) {
            break;
        }
        let slot = gate_slot(index, gate_index);
        gates.values[slot].center = vec2<f32>(0.0, 0.0);
        gates.values[slot].half_extents = vec2<f32>(0.0, 0.0);
        gates.values[slot].forward = vec2<f32>(0.0, 0.0);
        gates.values[slot]._pad0 = vec2<f32>(0.0, 0.0);
        gate_index = gate_index + 1u;
    }

    let first_gate = gates.values[base_slot];
    let start_position = first_gate.center - first_gate.forward * 1.0;
    states.values[index].position = start_position;
    states.values[index].velocity = vec2<f32>(0.0, 0.0);
    states.values[index].step_count = 0u;
    states.values[index].done = 0u;
    states.values[index].current_gate = 0u;
    states.values[index].current_lap = 0u;

    observations.values[index].position = start_position;
    observations.values[index].velocity = vec2<f32>(0.0, 0.0);
    observations.values[index].target_gate_position = first_gate.center;
    observations.values[index].progress = 0.0;
    observations.values[index]._pad0 = 0.0;

    reward_done.values[index].reward = 0.0;
    reward_done.values[index].done = 0u;
    reward_done.values[index]._pad0 = 0u;
    reward_done.values[index]._pad1 = 0u;
    reset_mask.values[index] = 0u;
}

fn current_target_gate(index: u32, current_gate: u32) -> Gate {
    let safe_count = max(generated_gate_count(), 1u);
    let safe_gate = min(current_gate, safe_count - 1u);
    return gates.values[env_gate_offset(index) + safe_gate];
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
        observations.values[index].progress =
            progress_value(state.current_lap, state.current_gate, generated_gate_count());
        reward_done.values[index].reward = 0.0;
        reward_done.values[index].done = 1u;
        return;
    }

    let accel = actions.values[index].accel;
    state.velocity = state.velocity + accel * params.dt_seconds;
    state.position = state.position + state.velocity * params.dt_seconds;
    state.step_count = state.step_count + 1u;

    let gate_count = generated_gate_count();
    var delta = target_gate.center - state.position;
    var distance_to_gate = length(delta);
    let gate_radius = max(target_gate.half_extents.x, target_gate.half_extents.y) * 1.5;
    if (distance_to_gate <= gate_radius) {
        if (state.current_gate + 1u < gate_count) {
            state.current_gate = state.current_gate + 1u;
            target_gate = current_target_gate(index, state.current_gate);
            delta = target_gate.center - state.position;
            distance_to_gate = length(delta);
        } else if (course_header.loop_enabled != 0u
            && state.current_lap + 1u < max(course_header.laps_required, 1u))
        {
            state.current_lap = state.current_lap + 1u;
            state.current_gate = 0u;
            target_gate = current_target_gate(index, 0u);
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
    observations.values[index].progress =
        progress_value(state.current_lap, state.current_gate, gate_count);
    observations.values[index]._pad0 = 0.0;

    reward_done.values[index].reward = observations.values[index].progress * 2.0 - distance_to_gate;
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
    pub current_lap: u32,
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
    pub laps_required: u32,
}

impl Default for GpuSimulationConfig {
    fn default() -> Self {
        Self {
            env_count: 1024,
            dt_seconds: 1.0 / 60.0,
            bounds: 8.0,
            max_steps: 1024,
            max_gates_per_env: 24,
            laps_required: 1,
        }
    }
}

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
    course_header: triad_gpu::GpuBuffer<GpuCourseHeader>,
    course_stages: triad_gpu::GpuBuffer<GpuStageSpec>,
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

        let mut default_course = CourseSpec::default_drone_course();
        default_course.laps_required = config.laps_required.max(1);
        let compiled_course = default_course.compile();
        let gate_capacity = config.env_count * config.max_gates_per_env;
        let layout_values = build_layout_headers(
            config.env_count,
            config.max_gates_per_env,
            compiled_course.header.total_gate_count,
        );

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
            .with_data(&layout_values)
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

        let course_header = renderer
            .create_gpu_buffer::<GpuCourseHeader>()
            .label("sim course header")
            .with_data(&[compiled_course.header])
            .usage(BufferUsage::Uniform)
            .build(registry)?;

        let course_stages = renderer
            .create_gpu_buffer::<GpuStageSpec>()
            .label("sim course stages")
            .capacity(COURSE_STAGE_CAPACITY)
            .with_data(&compiled_course.stages)
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
                gates.handle(),
                BindingType::StorageWrite,
            )
            .buffer_stage(
                8,
                ShaderStage::Compute,
                course_header.handle(),
                BindingType::Uniform,
            )
            .buffer_stage(
                9,
                ShaderStage::Compute,
                course_stages.handle(),
                BindingType::StorageRead,
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
            gates.handle(),
            course_header.handle(),
            course_stages.handle(),
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
            course_header,
            course_stages,
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

    pub fn set_course(
        &self,
        renderer: &Renderer,
        registry: &ResourceRegistry,
        course: &CourseSpec,
    ) -> Result<()> {
        let CompiledCourse { header, stages } = course.compile();
        if stages.len() > COURSE_STAGE_CAPACITY {
            return Err(BufferError::CapacityExceeded {
                requested: stages.len(),
                capacity: COURSE_STAGE_CAPACITY,
            }
            .into());
        }
        if header.total_gate_count as usize > self.config.max_gates_per_env {
            return Err(BufferError::CapacityExceeded {
                requested: header.total_gate_count as usize,
                capacity: self.config.max_gates_per_env,
            }
            .into());
        }

        let layout_values = build_layout_headers(
            self.config.env_count,
            self.config.max_gates_per_env,
            header.total_gate_count,
        );
        renderer.write_buffer(self.course_header.handle(), &[header], registry)?;
        renderer.write_buffer(self.course_stages.handle(), &stages, registry)?;
        renderer.write_buffer(self.layout_headers.handle(), &layout_values, registry)?;
        Ok(())
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
    gates: triad_gpu::Handle<wgpu::Buffer>,
    course_header: triad_gpu::Handle<wgpu::Buffer>,
    course_stages: triad_gpu::Handle<wgpu::Buffer>,
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
        .write(gates)
        .read(course_header)
        .read(course_stages)
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

fn build_layout_headers(
    env_count: usize,
    max_gates_per_env: usize,
    gate_count: u32,
) -> Vec<EnvLayoutHeader> {
    (0..env_count)
        .map(|env_index| EnvLayoutHeader {
            gate_offset: (env_index * max_gates_per_env) as u32,
            gate_count,
            obstacle_offset: 0,
            obstacle_count: 0,
        })
        .collect()
}
