use std::ffi::{CStr, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;
use std::sync::{Mutex, OnceLock};

use pollster::FutureExt;
use triad_gpu::{Renderer, ResourceRegistry};
use triad_sim::{
    Action, CourseSpec, EnvState, GpuSimulation, GpuSimulationConfig, Observation, ResetParams,
    RewardDone, StageKind, StageSpec, TurnDirection,
};

const TURN_DIRECTION_LEFT: u32 = 0;
const TURN_DIRECTION_RIGHT: u32 = 1;
const STAGE_KIND_INTRO: u32 = 0;
const STAGE_KIND_STRAIGHT: u32 = 1;
const STAGE_KIND_OFFSET: u32 = 2;
const STAGE_KIND_TURN90: u32 = 3;
const CURRICULUM_STAGE_INTRO: u32 = 0;
const CURRICULUM_STAGE_ARENA: u32 = 1;
const CURRICULUM_STAGE_TECHNICAL: u32 = 2;
const CURRICULUM_STAGE_ELEVATED: u32 = 3;
const ACTION_STRIDE: usize = 4;
const OBSERVATION_STRIDE: usize = 22;

static LAST_ERROR: OnceLock<Mutex<String>> = OnceLock::new();

fn last_error() -> &'static Mutex<String> {
    LAST_ERROR.get_or_init(|| Mutex::new(String::new()))
}

fn set_last_error(message: impl Into<String>) {
    if let Ok(mut last_error) = last_error().lock() {
        *last_error = message.into();
    }
}

fn clear_last_error() {
    set_last_error("");
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic".to_owned()
    }
}

fn ffi_guard<T>(default: T, f: impl FnOnce() -> T) -> T {
    clear_last_error();
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(value) => value,
        Err(payload) => {
            set_last_error(format!("panic in triad-py FFI: {}", panic_message(payload)));
            default
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TriadStageDesc {
    pub kind: u32,
    pub gate_count: u32,
    pub spacing: f32,
    pub lateral_amp: f32,
    pub turn_degrees: f32,
    pub radius: f32,
    pub vertical_amp: f32,
    pub hole_half_width: f32,
    pub hole_half_height: f32,
    pub direction: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TriadCourseStats {
    pub stage_count: u32,
    pub total_gate_count: u32,
    pub loop_enabled: u32,
    pub laps_required: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TriadSimConfig {
    pub env_count: u32,
    pub dt_seconds: f32,
    pub bounds: f32,
    pub max_steps: u32,
    pub max_gates_per_env: u32,
    pub laps_required: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TriadAction {
    pub motor_0: f32,
    pub motor_1: f32,
    pub motor_2: f32,
    pub motor_3: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TriadResetParams {
    pub seed: u32,
    pub grammar_id: u32,
    pub difficulty: f32,
    pub curriculum_stage: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TriadEnvState {
    pub position_x: f32,
    pub position_y: f32,
    pub position_z: f32,
    pub velocity_x: f32,
    pub velocity_y: f32,
    pub velocity_z: f32,
    pub roll: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub angular_velocity_x: f32,
    pub angular_velocity_y: f32,
    pub angular_velocity_z: f32,
    pub motor_0: f32,
    pub motor_1: f32,
    pub motor_2: f32,
    pub motor_3: f32,
    pub step_count: u32,
    pub done: u32,
    pub current_gate: u32,
    pub current_lap: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TriadObservation {
    pub position_x: f32,
    pub position_y: f32,
    pub position_z: f32,
    pub velocity_x: f32,
    pub velocity_y: f32,
    pub velocity_z: f32,
    pub roll: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub angular_velocity_x: f32,
    pub angular_velocity_y: f32,
    pub angular_velocity_z: f32,
    pub target_gate_x: f32,
    pub target_gate_y: f32,
    pub target_gate_z: f32,
    pub target_gate_forward_x: f32,
    pub target_gate_forward_y: f32,
    pub target_gate_forward_z: f32,
    pub progress: f32,
    pub distance_to_gate: f32,
    pub gate_alignment: f32,
    pub mean_motor_thrust: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TriadRewardDone {
    pub reward: f32,
    pub done: u32,
    pub done_reason: u32,
    pub _pad0: u32,
    pub progress_reward: f32,
    pub distance_penalty: f32,
    pub alignment_reward: f32,
    pub tilt_penalty: f32,
    pub completion_bonus: f32,
    pub collision_penalty: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

impl Default for TriadStageDesc {
    fn default() -> Self {
        Self {
            kind: STAGE_KIND_STRAIGHT,
            gate_count: 1,
            spacing: 1.5,
            lateral_amp: 0.0,
            turn_degrees: 0.0,
            radius: 0.0,
            vertical_amp: 0.0,
            hole_half_width: 0.14,
            hole_half_height: 0.14,
            direction: TURN_DIRECTION_LEFT,
        }
    }
}

impl From<GpuSimulationConfig> for TriadSimConfig {
    fn from(config: GpuSimulationConfig) -> Self {
        Self {
            env_count: config.env_count as u32,
            dt_seconds: config.dt_seconds,
            bounds: config.bounds,
            max_steps: config.max_steps,
            max_gates_per_env: config.max_gates_per_env as u32,
            laps_required: config.laps_required,
        }
    }
}

impl From<TriadSimConfig> for GpuSimulationConfig {
    fn from(config: TriadSimConfig) -> Self {
        Self {
            env_count: config.env_count as usize,
            dt_seconds: config.dt_seconds,
            bounds: config.bounds,
            max_steps: config.max_steps,
            max_gates_per_env: config.max_gates_per_env as usize,
            laps_required: config.laps_required,
        }
    }
}

pub struct TriadSimulation {
    renderer: Renderer,
    registry: ResourceRegistry,
    simulation: GpuSimulation,
    action_scratch: Vec<Action>,
}

impl TriadSimulation {
    fn new(config: TriadSimConfig) -> Result<Self, String> {
        let renderer = Renderer::new()
            .block_on()
            .map_err(|error| format!("renderer init failed: {error}"))?;
        let mut registry = ResourceRegistry::default();
        let simulation = GpuSimulation::new(&renderer, &mut registry, config.into())
            .map_err(|error| format!("simulation init failed: {error}"))?;
        let zero_actions = vec![Action::new([0.0; 4]); simulation.env_count()];
        simulation
            .set_actions(&renderer, &registry, &zero_actions)
            .map_err(|error| format!("failed to zero actions: {error}"))?;
        Ok(Self {
            renderer,
            registry,
            simulation,
            action_scratch: zero_actions,
        })
    }
}

fn direction_from_u32(value: u32) -> Option<TurnDirection> {
    match value {
        TURN_DIRECTION_LEFT => Some(TurnDirection::Left),
        TURN_DIRECTION_RIGHT => Some(TurnDirection::Right),
        _ => None,
    }
}

fn stage_kind_from_u32(value: u32) -> Option<StageKind> {
    match value {
        STAGE_KIND_INTRO => Some(StageKind::Intro),
        STAGE_KIND_STRAIGHT => Some(StageKind::Straight),
        STAGE_KIND_OFFSET => Some(StageKind::Offset),
        STAGE_KIND_TURN90 => Some(StageKind::Turn90),
        _ => None,
    }
}

fn stage_from_desc(desc: &TriadStageDesc) -> Option<StageSpec> {
    Some(StageSpec {
        kind: stage_kind_from_u32(desc.kind)?,
        gate_count: desc.gate_count,
        spacing: desc.spacing,
        lateral_amp: desc.lateral_amp,
        turn_degrees: desc.turn_degrees,
        radius: desc.radius,
        vertical_amp: desc.vertical_amp,
        hole_half_width: desc.hole_half_width,
        hole_half_height: desc.hole_half_height,
        direction: direction_from_u32(desc.direction)?,
    })
}

fn course_from_ptr<'a>(course: *mut CourseSpec) -> Option<&'a mut CourseSpec> {
    if course.is_null() {
        return None;
    }
    Some(unsafe { &mut *course })
}

fn simulation_from_ptr<'a>(simulation: *mut TriadSimulation) -> Option<&'a mut TriadSimulation> {
    if simulation.is_null() {
        return None;
    }
    Some(unsafe { &mut *simulation })
}

fn cstr_to_string(value: *const c_char) -> Option<String> {
    if value.is_null() {
        return None;
    }
    let text = unsafe { CStr::from_ptr(value) };
    text.to_str().ok().map(ToOwned::to_owned)
}

fn copy_vec_to_out<T: Copy>(values: &[T], out: *mut T, out_len: usize) -> Result<(), String> {
    if out.is_null() {
        return Err("output pointer was null".to_string());
    }
    if out_len < values.len() {
        return Err(format!(
            "output capacity too small: need {}, got {}",
            values.len(),
            out_len
        ));
    }
    unsafe {
        ptr::copy_nonoverlapping(values.as_ptr(), out, values.len());
    }
    Ok(())
}

fn convert_actions(actions: &[TriadAction]) -> Vec<Action> {
    actions
        .iter()
        .map(|action| {
            Action::new([
                action.motor_0,
                action.motor_1,
                action.motor_2,
                action.motor_3,
            ])
        })
        .collect()
}

fn write_flat_actions_into_scratch(
    scratch: &mut [Action],
    action_values: &[f32],
) -> Result<(), String> {
    if !action_values.len().is_multiple_of(ACTION_STRIDE) {
        return Err(format!(
            "flat action buffer length must be a multiple of {}, got {}",
            ACTION_STRIDE,
            action_values.len()
        ));
    }
    let env_count = action_values.len() / ACTION_STRIDE;
    if env_count != scratch.len() {
        return Err(format!(
            "flat action buffer env count mismatch: expected {}, got {}",
            scratch.len(),
            env_count
        ));
    }

    for (index, chunk) in action_values.chunks_exact(ACTION_STRIDE).enumerate() {
        scratch[index] = Action::new([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(())
}

fn convert_reset_params(reset_params: &[TriadResetParams]) -> Vec<ResetParams> {
    reset_params
        .iter()
        .map(|params| {
            ResetParams::new(
                params.seed,
                params.grammar_id,
                params.difficulty,
                params.curriculum_stage,
            )
        })
        .collect()
}

fn convert_state(state: &EnvState) -> TriadEnvState {
    TriadEnvState {
        position_x: state.position[0],
        position_y: state.position[1],
        position_z: state.position[2],
        velocity_x: state.velocity[0],
        velocity_y: state.velocity[1],
        velocity_z: state.velocity[2],
        roll: state.attitude[0],
        pitch: state.attitude[1],
        yaw: state.attitude[2],
        angular_velocity_x: state.angular_velocity[0],
        angular_velocity_y: state.angular_velocity[1],
        angular_velocity_z: state.angular_velocity[2],
        motor_0: state.motor_thrust[0],
        motor_1: state.motor_thrust[1],
        motor_2: state.motor_thrust[2],
        motor_3: state.motor_thrust[3],
        step_count: state.step_count,
        done: state.done,
        current_gate: state.current_gate,
        current_lap: state.current_lap,
    }
}

fn convert_observation(observation: &Observation) -> TriadObservation {
    TriadObservation {
        position_x: observation.position[0],
        position_y: observation.position[1],
        position_z: observation.position[2],
        velocity_x: observation.velocity[0],
        velocity_y: observation.velocity[1],
        velocity_z: observation.velocity[2],
        roll: observation.attitude[0],
        pitch: observation.attitude[1],
        yaw: observation.attitude[2],
        angular_velocity_x: observation.angular_velocity[0],
        angular_velocity_y: observation.angular_velocity[1],
        angular_velocity_z: observation.angular_velocity[2],
        target_gate_x: observation.target_gate_position[0],
        target_gate_y: observation.target_gate_position[1],
        target_gate_z: observation.target_gate_position[2],
        target_gate_forward_x: observation.target_gate_forward[0],
        target_gate_forward_y: observation.target_gate_forward[1],
        target_gate_forward_z: observation.target_gate_forward[2],
        progress: observation.progress,
        distance_to_gate: observation.distance_to_gate,
        gate_alignment: observation.gate_alignment,
        mean_motor_thrust: observation.mean_motor_thrust,
    }
}

fn convert_reward_done(reward_done: &RewardDone) -> TriadRewardDone {
    TriadRewardDone {
        reward: reward_done.reward,
        done: reward_done.done,
        done_reason: reward_done.done_reason,
        _pad0: reward_done._pad,
        progress_reward: reward_done.progress_reward,
        distance_penalty: reward_done.distance_penalty,
        alignment_reward: reward_done.alignment_reward,
        tilt_penalty: reward_done.tilt_penalty,
        completion_bonus: reward_done.completion_bonus,
        collision_penalty: reward_done.collision_penalty,
        _pad1: reward_done._pad1,
        _pad2: reward_done._pad2,
    }
}

fn flatten_observations(observations: &[Observation]) -> Vec<f32> {
    let mut flat = Vec::with_capacity(observations.len() * OBSERVATION_STRIDE);
    for observation in observations {
        flat.extend_from_slice(&[
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
        ]);
    }
    flat
}

fn flatten_reward_done(reward_done: &[RewardDone]) -> (Vec<f32>, Vec<u8>, Vec<u32>) {
    let mut rewards = Vec::with_capacity(reward_done.len());
    let mut dones = Vec::with_capacity(reward_done.len());
    let mut done_reasons = Vec::with_capacity(reward_done.len());
    for item in reward_done {
        rewards.push(item.reward);
        dones.push(u8::from(item.done != 0));
        done_reasons.push(item.done_reason);
    }
    (rewards, dones, done_reasons)
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_last_error_message(out_buffer: *mut c_char, out_capacity: usize) -> usize {
    let message = last_error()
        .lock()
        .map(|guard| guard.clone())
        .unwrap_or_else(|_| "failed to acquire last error".to_string());
    let required = message.len() + 1;
    if out_buffer.is_null() || out_capacity == 0 {
        return required;
    }

    let bytes = message.as_bytes();
    let copy_len = bytes.len().min(out_capacity.saturating_sub(1));
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), out_buffer.cast::<u8>(), copy_len);
        *out_buffer.add(copy_len) = 0;
    }
    required
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_course_create() -> *mut CourseSpec {
    clear_last_error();
    Box::into_raw(Box::new(CourseSpec::new("course")))
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_course_create_default_drone() -> *mut CourseSpec {
    clear_last_error();
    Box::into_raw(Box::new(CourseSpec::default_drone_course()))
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_course_destroy(course: *mut CourseSpec) {
    clear_last_error();
    if course.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(course));
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_course_set_name(course: *mut CourseSpec, name: *const c_char) -> bool {
    clear_last_error();
    let Some(course) = course_from_ptr(course) else {
        set_last_error("course pointer was null");
        return false;
    };
    let Some(name) = cstr_to_string(name) else {
        set_last_error("course name was invalid");
        return false;
    };
    course.name = name;
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_course_set_loop_enabled(course: *mut CourseSpec, enabled: bool) -> bool {
    clear_last_error();
    let Some(course) = course_from_ptr(course) else {
        set_last_error("course pointer was null");
        return false;
    };
    course.loop_enabled = enabled;
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_course_set_laps_required(course: *mut CourseSpec, laps: u32) -> bool {
    clear_last_error();
    let Some(course) = course_from_ptr(course) else {
        set_last_error("course pointer was null");
        return false;
    };
    course.laps_required = laps.max(1);
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_course_clear_stages(course: *mut CourseSpec) -> bool {
    clear_last_error();
    let Some(course) = course_from_ptr(course) else {
        set_last_error("course pointer was null");
        return false;
    };
    course.clear_stages();
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_course_add_stage(
    course: *mut CourseSpec,
    stage_desc: TriadStageDesc,
) -> bool {
    clear_last_error();
    let Some(course) = course_from_ptr(course) else {
        set_last_error("course pointer was null");
        return false;
    };
    let Some(stage) = stage_from_desc(&stage_desc) else {
        set_last_error("invalid stage descriptor");
        return false;
    };
    course.push_stage(stage);
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_course_get_stats(
    course: *mut CourseSpec,
    out_stats: *mut TriadCourseStats,
) -> bool {
    clear_last_error();
    let Some(course) = course_from_ptr(course) else {
        set_last_error("course pointer was null");
        return false;
    };
    if out_stats.is_null() {
        set_last_error("course stats output pointer was null");
        return false;
    }

    unsafe {
        *out_stats = TriadCourseStats {
            stage_count: course.stages.len() as u32,
            total_gate_count: course.total_gate_count(),
            loop_enabled: u32::from(course.loop_enabled),
            laps_required: course.laps_required.max(1),
        };
    }
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_sim_config_default() -> TriadSimConfig {
    clear_last_error();
    GpuSimulationConfig::default().into()
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_action_stride() -> usize {
    clear_last_error();
    ACTION_STRIDE
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_observation_stride() -> usize {
    clear_last_error();
    OBSERVATION_STRIDE
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_simulation_create(config: TriadSimConfig) -> *mut TriadSimulation {
    ffi_guard(ptr::null_mut(), || match TriadSimulation::new(config) {
        Ok(simulation) => Box::into_raw(Box::new(simulation)),
        Err(error) => {
            set_last_error(error);
            ptr::null_mut()
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_simulation_destroy(simulation: *mut TriadSimulation) {
    ffi_guard((), || {
        if simulation.is_null() {
            return;
        }
        unsafe {
            drop(Box::from_raw(simulation));
        }
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_simulation_env_count(simulation: *mut TriadSimulation) -> u32 {
    ffi_guard(0, || {
        let Some(simulation) = simulation_from_ptr(simulation) else {
            set_last_error("simulation pointer was null");
            return 0;
        };
        simulation.simulation.env_count() as u32
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_simulation_set_course(
    simulation: *mut TriadSimulation,
    course: *mut CourseSpec,
) -> bool {
    ffi_guard(false, || {
        let Some(simulation) = simulation_from_ptr(simulation) else {
            set_last_error("simulation pointer was null");
            return false;
        };
        let Some(course) = course_from_ptr(course) else {
            set_last_error("course pointer was null");
            return false;
        };
        match simulation
            .simulation
            .set_course(&simulation.renderer, &simulation.registry, course)
        {
            Ok(()) => true,
            Err(error) => {
                set_last_error(format!("failed to set course: {error}"));
                false
            }
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_simulation_set_actions(
    simulation: *mut TriadSimulation,
    actions: *const TriadAction,
    action_count: usize,
) -> bool {
    ffi_guard(false, || {
        let Some(simulation) = simulation_from_ptr(simulation) else {
            set_last_error("simulation pointer was null");
            return false;
        };
        if actions.is_null() {
            set_last_error("actions pointer was null");
            return false;
        }
        let actions = unsafe { std::slice::from_raw_parts(actions, action_count) };
        let converted = convert_actions(actions);
        match simulation.simulation.set_actions(
            &simulation.renderer,
            &simulation.registry,
            &converted,
        ) {
            Ok(()) => true,
            Err(error) => {
                set_last_error(format!("failed to set actions: {error}"));
                false
            }
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_simulation_set_reset_params(
    simulation: *mut TriadSimulation,
    reset_params: *const TriadResetParams,
    reset_param_count: usize,
) -> bool {
    ffi_guard(false, || {
        let Some(simulation) = simulation_from_ptr(simulation) else {
            set_last_error("simulation pointer was null");
            return false;
        };
        if reset_params.is_null() {
            set_last_error("reset params pointer was null");
            return false;
        }
        let reset_params = unsafe { std::slice::from_raw_parts(reset_params, reset_param_count) };
        let converted = convert_reset_params(reset_params);
        match simulation.simulation.set_reset_params(
            &simulation.renderer,
            &simulation.registry,
            &converted,
        ) {
            Ok(()) => true,
            Err(error) => {
                set_last_error(format!("failed to set reset params: {error}"));
                false
            }
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_simulation_reset_all(simulation: *mut TriadSimulation) -> bool {
    ffi_guard(false, || {
        let Some(simulation) = simulation_from_ptr(simulation) else {
            set_last_error("simulation pointer was null");
            return false;
        };
        match simulation
            .simulation
            .reset_all(&simulation.renderer, &simulation.registry)
        {
            Ok(()) => true,
            Err(error) => {
                set_last_error(format!("failed to reset simulation: {error}"));
                false
            }
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_simulation_request_resets(
    simulation: *mut TriadSimulation,
    env_indices: *const u32,
    env_index_count: usize,
) -> bool {
    ffi_guard(false, || {
        let Some(simulation) = simulation_from_ptr(simulation) else {
            set_last_error("simulation pointer was null");
            return false;
        };
        if env_indices.is_null() {
            set_last_error("env indices pointer was null");
            return false;
        }
        let env_indices = unsafe { std::slice::from_raw_parts(env_indices, env_index_count) };
        let env_indices = env_indices
            .iter()
            .map(|value| *value as usize)
            .collect::<Vec<_>>();
        match simulation.simulation.request_resets(
            &simulation.renderer,
            &simulation.registry,
            &env_indices,
        ) {
            Ok(()) => true,
            Err(error) => {
                set_last_error(format!("failed to request resets: {error}"));
                false
            }
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_simulation_step(simulation: *mut TriadSimulation, steps: usize) -> bool {
    ffi_guard(false, || {
        let Some(simulation) = simulation_from_ptr(simulation) else {
            set_last_error("simulation pointer was null");
            return false;
        };
        simulation
            .simulation
            .step_n(&simulation.renderer, &simulation.registry, steps.max(1));
        true
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_simulation_readback_state(
    simulation: *mut TriadSimulation,
    out_states: *mut TriadEnvState,
    out_count: usize,
) -> bool {
    ffi_guard(false, || {
        let Some(simulation) = simulation_from_ptr(simulation) else {
            set_last_error("simulation pointer was null");
            return false;
        };
        let values = match simulation
            .simulation
            .readback_state(&simulation.renderer, &simulation.registry)
        {
            Ok(values) => values,
            Err(error) => {
                set_last_error(format!("failed to read back state: {error}"));
                return false;
            }
        };
        let converted = values.iter().map(convert_state).collect::<Vec<_>>();
        match copy_vec_to_out(&converted, out_states, out_count) {
            Ok(()) => true,
            Err(error) => {
                set_last_error(error);
                false
            }
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_simulation_readback_observations(
    simulation: *mut TriadSimulation,
    out_observations: *mut TriadObservation,
    out_count: usize,
) -> bool {
    ffi_guard(false, || {
        let Some(simulation) = simulation_from_ptr(simulation) else {
            set_last_error("simulation pointer was null");
            return false;
        };
        let values = match simulation
            .simulation
            .readback_observations(&simulation.renderer, &simulation.registry)
        {
            Ok(values) => values,
            Err(error) => {
                set_last_error(format!("failed to read back observations: {error}"));
                return false;
            }
        };
        let converted = values.iter().map(convert_observation).collect::<Vec<_>>();
        match copy_vec_to_out(&converted, out_observations, out_count) {
            Ok(()) => true,
            Err(error) => {
                set_last_error(error);
                false
            }
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_simulation_readback_observations_flat(
    simulation: *mut TriadSimulation,
    out_observations: *mut f32,
    out_count: usize,
) -> bool {
    ffi_guard(false, || {
        let Some(simulation) = simulation_from_ptr(simulation) else {
            set_last_error("simulation pointer was null");
            return false;
        };
        let values = match simulation
            .simulation
            .readback_observations(&simulation.renderer, &simulation.registry)
        {
            Ok(values) => values,
            Err(error) => {
                set_last_error(format!("failed to read back observations: {error}"));
                return false;
            }
        };
        let flat = flatten_observations(&values);
        match copy_vec_to_out(&flat, out_observations, out_count) {
            Ok(()) => true,
            Err(error) => {
                set_last_error(error);
                false
            }
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_simulation_readback_reward_done(
    simulation: *mut TriadSimulation,
    out_reward_done: *mut TriadRewardDone,
    out_count: usize,
) -> bool {
    ffi_guard(false, || {
        let Some(simulation) = simulation_from_ptr(simulation) else {
            set_last_error("simulation pointer was null");
            return false;
        };
        let values = match simulation
            .simulation
            .readback_reward_done(&simulation.renderer, &simulation.registry)
        {
            Ok(values) => values,
            Err(error) => {
                set_last_error(format!("failed to read back reward_done: {error}"));
                return false;
            }
        };
        let converted = values.iter().map(convert_reward_done).collect::<Vec<_>>();
        match copy_vec_to_out(&converted, out_reward_done, out_count) {
            Ok(()) => true,
            Err(error) => {
                set_last_error(error);
                false
            }
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_simulation_readback_reward_done_flat(
    simulation: *mut TriadSimulation,
    out_rewards: *mut f32,
    reward_count: usize,
    out_dones: *mut u8,
    done_count: usize,
) -> bool {
    ffi_guard(false, || {
        let Some(simulation) = simulation_from_ptr(simulation) else {
            set_last_error("simulation pointer was null");
            return false;
        };
        let values = match simulation
            .simulation
            .readback_reward_done(&simulation.renderer, &simulation.registry)
        {
            Ok(values) => values,
            Err(error) => {
                set_last_error(format!("failed to read back reward_done: {error}"));
                return false;
            }
        };
        let (rewards, dones, _) = flatten_reward_done(&values);
        if let Err(error) = copy_vec_to_out(&rewards, out_rewards, reward_count) {
            set_last_error(error);
            return false;
        }
        match copy_vec_to_out(&dones, out_dones, done_count) {
            Ok(()) => true,
            Err(error) => {
                set_last_error(error);
                false
            }
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_simulation_step_actions_readback(
    simulation: *mut TriadSimulation,
    actions: *const TriadAction,
    action_count: usize,
    steps: usize,
    out_observations: *mut f32,
    observation_count: usize,
    out_rewards: *mut f32,
    reward_count: usize,
    out_dones: *mut u8,
    done_count: usize,
    out_done_reasons: *mut u32,
    done_reason_count: usize,
) -> bool {
    ffi_guard(false, || {
        let Some(simulation) = simulation_from_ptr(simulation) else {
            set_last_error("simulation pointer was null");
            return false;
        };
        if actions.is_null() {
            set_last_error("actions pointer was null");
            return false;
        }
        let actions = unsafe { std::slice::from_raw_parts(actions, action_count) };
        let converted = convert_actions(actions);
        if let Err(error) = simulation.simulation.set_actions(
            &simulation.renderer,
            &simulation.registry,
            &converted,
        ) {
            set_last_error(format!("failed to set actions: {error}"));
            return false;
        }
        simulation
            .simulation
            .step_n(&simulation.renderer, &simulation.registry, steps.max(1));

        let observations = match simulation
            .simulation
            .readback_observations(&simulation.renderer, &simulation.registry)
        {
            Ok(values) => values,
            Err(error) => {
                set_last_error(format!("failed to read back observations: {error}"));
                return false;
            }
        };
        let flat_observations = flatten_observations(&observations);
        if let Err(error) = copy_vec_to_out(&flat_observations, out_observations, observation_count)
        {
            set_last_error(error);
            return false;
        }

        let reward_done = match simulation
            .simulation
            .readback_reward_done(&simulation.renderer, &simulation.registry)
        {
            Ok(values) => values,
            Err(error) => {
                set_last_error(format!("failed to read back reward_done: {error}"));
                return false;
            }
        };
        let (rewards, dones, done_reasons) = flatten_reward_done(&reward_done);
        if let Err(error) = copy_vec_to_out(&rewards, out_rewards, reward_count) {
            set_last_error(error);
            return false;
        }
        if let Err(error) = copy_vec_to_out(&dones, out_dones, done_count) {
            set_last_error(error);
            return false;
        }
        match copy_vec_to_out(&done_reasons, out_done_reasons, done_reason_count) {
            Ok(()) => true,
            Err(error) => {
                set_last_error(error);
                false
            }
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_simulation_step_flat_actions_readback(
    simulation: *mut TriadSimulation,
    action_values: *const f32,
    action_value_count: usize,
    steps: usize,
    out_observations: *mut f32,
    observation_count: usize,
    out_rewards: *mut f32,
    reward_count: usize,
    out_dones: *mut u8,
    done_count: usize,
    out_done_reasons: *mut u32,
    done_reason_count: usize,
) -> bool {
    ffi_guard(false, || {
        let Some(simulation) = simulation_from_ptr(simulation) else {
            set_last_error("simulation pointer was null");
            return false;
        };
        if action_values.is_null() {
            set_last_error("action values pointer was null");
            return false;
        }
        let action_values =
            unsafe { std::slice::from_raw_parts(action_values, action_value_count) };
        if let Err(error) =
            write_flat_actions_into_scratch(&mut simulation.action_scratch, action_values)
        {
            set_last_error(error);
            return false;
        }
        if let Err(error) = simulation.simulation.set_actions(
            &simulation.renderer,
            &simulation.registry,
            &simulation.action_scratch,
        ) {
            set_last_error(format!("failed to set actions: {error}"));
            return false;
        }
        simulation
            .simulation
            .step_n(&simulation.renderer, &simulation.registry, steps.max(1));

        let observations = match simulation
            .simulation
            .readback_observations(&simulation.renderer, &simulation.registry)
        {
            Ok(values) => values,
            Err(error) => {
                set_last_error(format!("failed to read back observations: {error}"));
                return false;
            }
        };
        let flat_observations = flatten_observations(&observations);
        if let Err(error) = copy_vec_to_out(&flat_observations, out_observations, observation_count)
        {
            set_last_error(error);
            return false;
        }

        let reward_done = match simulation
            .simulation
            .readback_reward_done(&simulation.renderer, &simulation.registry)
        {
            Ok(values) => values,
            Err(error) => {
                set_last_error(format!("failed to read back reward_done: {error}"));
                return false;
            }
        };
        let (rewards, dones, done_reasons) = flatten_reward_done(&reward_done);
        if let Err(error) = copy_vec_to_out(&rewards, out_rewards, reward_count) {
            set_last_error(error);
            return false;
        }
        if let Err(error) = copy_vec_to_out(&dones, out_dones, done_count) {
            set_last_error(error);
            return false;
        }
        match copy_vec_to_out(&done_reasons, out_done_reasons, done_reason_count) {
            Ok(()) => true,
            Err(error) => {
                set_last_error(error);
                false
            }
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_stage_kind_intro() -> u32 {
    clear_last_error();
    STAGE_KIND_INTRO
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_stage_kind_straight() -> u32 {
    clear_last_error();
    STAGE_KIND_STRAIGHT
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_stage_kind_offset() -> u32 {
    clear_last_error();
    STAGE_KIND_OFFSET
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_stage_kind_turn90() -> u32 {
    clear_last_error();
    STAGE_KIND_TURN90
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_turn_direction_left() -> u32 {
    clear_last_error();
    TURN_DIRECTION_LEFT
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_turn_direction_right() -> u32 {
    clear_last_error();
    TURN_DIRECTION_RIGHT
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_curriculum_stage_intro() -> u32 {
    clear_last_error();
    CURRICULUM_STAGE_INTRO
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_curriculum_stage_arena() -> u32 {
    clear_last_error();
    CURRICULUM_STAGE_ARENA
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_curriculum_stage_technical() -> u32 {
    clear_last_error();
    CURRICULUM_STAGE_TECHNICAL
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_curriculum_stage_elevated() -> u32 {
    clear_last_error();
    CURRICULUM_STAGE_ELEVATED
}
