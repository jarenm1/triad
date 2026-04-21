use std::ffi::{CStr, c_char};
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
    pub accel_x: f32,
    pub accel_y: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TriadResetParams {
    pub seed: u32,
    pub grammar_id: u32,
    pub difficulty: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TriadEnvState {
    pub position_x: f32,
    pub position_y: f32,
    pub velocity_x: f32,
    pub velocity_y: f32,
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
    pub velocity_x: f32,
    pub velocity_y: f32,
    pub target_gate_x: f32,
    pub target_gate_y: f32,
    pub progress: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TriadRewardDone {
    pub reward: f32,
    pub done: u32,
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
}

impl TriadSimulation {
    fn new(config: TriadSimConfig) -> Result<Self, String> {
        let renderer = Renderer::new()
            .block_on()
            .map_err(|error| format!("renderer init failed: {error}"))?;
        let mut registry = ResourceRegistry::default();
        let simulation = GpuSimulation::new(&renderer, &mut registry, config.into())
            .map_err(|error| format!("simulation init failed: {error}"))?;
        let zero_actions = vec![Action::new([0.0, 0.0]); simulation.env_count()];
        simulation
            .set_actions(&renderer, &registry, &zero_actions)
            .map_err(|error| format!("failed to zero actions: {error}"))?;
        Ok(Self {
            renderer,
            registry,
            simulation,
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
        .map(|action| Action::new([action.accel_x, action.accel_y]))
        .collect()
}

fn convert_reset_params(reset_params: &[TriadResetParams]) -> Vec<ResetParams> {
    reset_params
        .iter()
        .map(|params| ResetParams::new(params.seed, params.grammar_id, params.difficulty))
        .collect()
}

fn convert_state(state: &EnvState) -> TriadEnvState {
    TriadEnvState {
        position_x: state.position[0],
        position_y: state.position[1],
        velocity_x: state.velocity[0],
        velocity_y: state.velocity[1],
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
        velocity_x: observation.velocity[0],
        velocity_y: observation.velocity[1],
        target_gate_x: observation.target_gate_position[0],
        target_gate_y: observation.target_gate_position[1],
        progress: observation.progress,
    }
}

fn convert_reward_done(reward_done: &RewardDone) -> TriadRewardDone {
    TriadRewardDone {
        reward: reward_done.reward,
        done: reward_done.done,
    }
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
pub extern "C" fn triad_simulation_create(config: TriadSimConfig) -> *mut TriadSimulation {
    clear_last_error();
    match TriadSimulation::new(config) {
        Ok(simulation) => Box::into_raw(Box::new(simulation)),
        Err(error) => {
            set_last_error(error);
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_simulation_destroy(simulation: *mut TriadSimulation) {
    clear_last_error();
    if simulation.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(simulation));
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_simulation_env_count(simulation: *mut TriadSimulation) -> u32 {
    clear_last_error();
    let Some(simulation) = simulation_from_ptr(simulation) else {
        set_last_error("simulation pointer was null");
        return 0;
    };
    simulation.simulation.env_count() as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_simulation_set_course(
    simulation: *mut TriadSimulation,
    course: *mut CourseSpec,
) -> bool {
    clear_last_error();
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
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_simulation_set_actions(
    simulation: *mut TriadSimulation,
    actions: *const TriadAction,
    action_count: usize,
) -> bool {
    clear_last_error();
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
    match simulation
        .simulation
        .set_actions(&simulation.renderer, &simulation.registry, &converted)
    {
        Ok(()) => true,
        Err(error) => {
            set_last_error(format!("failed to set actions: {error}"));
            false
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_simulation_set_reset_params(
    simulation: *mut TriadSimulation,
    reset_params: *const TriadResetParams,
    reset_param_count: usize,
) -> bool {
    clear_last_error();
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
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_simulation_reset_all(simulation: *mut TriadSimulation) -> bool {
    clear_last_error();
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
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_simulation_request_resets(
    simulation: *mut TriadSimulation,
    env_indices: *const u32,
    env_index_count: usize,
) -> bool {
    clear_last_error();
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
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_simulation_step(simulation: *mut TriadSimulation, steps: usize) -> bool {
    clear_last_error();
    let Some(simulation) = simulation_from_ptr(simulation) else {
        set_last_error("simulation pointer was null");
        return false;
    };
    simulation
        .simulation
        .step_n(&simulation.renderer, &simulation.registry, steps.max(1));
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_simulation_readback_state(
    simulation: *mut TriadSimulation,
    out_states: *mut TriadEnvState,
    out_count: usize,
) -> bool {
    clear_last_error();
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
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_simulation_readback_observations(
    simulation: *mut TriadSimulation,
    out_observations: *mut TriadObservation,
    out_count: usize,
) -> bool {
    clear_last_error();
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
}

#[unsafe(no_mangle)]
pub extern "C" fn triad_simulation_readback_reward_done(
    simulation: *mut TriadSimulation,
    out_reward_done: *mut TriadRewardDone,
    out_count: usize,
) -> bool {
    clear_last_error();
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
