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
    position: vec4<f32>,
    velocity: vec4<f32>,
    attitude: vec4<f32>,
    angular_velocity: vec4<f32>,
    motor_thrust: vec4<f32>,
    step_count: u32,
    done: u32,
    current_gate: u32,
    current_lap: u32,
};

struct Action {
    motor_command: vec4<f32>,
};

struct Observation {
    position: vec4<f32>,
    velocity: vec4<f32>,
    attitude: vec4<f32>,
    angular_velocity: vec4<f32>,
    target_gate_position: vec4<f32>,
    target_gate_forward_progress: vec4<f32>,
    metrics: vec4<f32>,
};

struct RewardDone {
    reward: f32,
    done: u32,
    done_reason: u32,
    _pad0: u32,
    progress_reward: f32,
    distance_penalty: f32,
    alignment_reward: f32,
    tilt_penalty: f32,
    completion_bonus: f32,
    collision_penalty: f32,
    _pad1: f32,
    _pad2: f32,
};

const DONE_REASON_NONE: u32 = 0u;
const DONE_REASON_COMPLETE: u32 = 1u;
const DONE_REASON_GATE_COLLISION: u32 = 1u << 1u;
const DONE_REASON_OBSTACLE_COLLISION: u32 = 1u << 2u;
const DONE_REASON_FLOOR_COLLISION: u32 = 1u << 3u;
const DONE_REASON_OUT_OF_BOUNDS: u32 = 1u << 4u;
const DONE_REASON_STEP_LIMIT: u32 = 1u << 5u;
const DONE_REASON_EXCESSIVE_TILT: u32 = 1u << 6u;

struct SimParams {
    dt_seconds: f32,
    bounds: f32,
    min_altitude: f32,
    max_altitude: f32,
    mass: f32,
    gravity: f32,
    thrust_scale: f32,
    arm_length: f32,
    yaw_torque_scale: f32,
    linear_drag: f32,
    angular_drag: f32,
    motor_response: f32,
    drone_radius: f32,
    gate_frame_thickness: f32,
    gate_depth_half: f32,
    collision_penalty: f32,
    env_count: u32,
    max_steps: u32,
    max_gates_per_env: u32,
    max_obstacles_per_env: u32,
};

struct ResetParams {
    seed: u32,
    grammar_id: u32,
    difficulty: f32,
    curriculum_stage: u32,
};

struct EnvLayoutHeader {
    gate_offset: u32,
    gate_count: u32,
    obstacle_offset: u32,
    obstacle_count: u32,
};

struct Gate {
    center: vec4<f32>,
    half_extents: vec4<f32>,
    forward: vec4<f32>,
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

struct LayoutHeaderBuffer {
    values: array<EnvLayoutHeader>,
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
@group(0) @binding(9) var<storage, read_write> layout_headers: LayoutHeaderBuffer;

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

fn rotate_around_axis(v: vec3<f32>, axis: vec3<f32>, angle: f32) -> vec3<f32> {
    let a = normalize(axis);
    let s = sin(angle);
    let c = cos(angle);
    return v * c + cross(a, v) * s + a * dot(a, v) * (1.0 - c);
}

fn clamp_motor_command(command: vec4<f32>) -> vec4<f32> {
    return clamp(command, vec4<f32>(0.0), vec4<f32>(1.0));
}

fn body_forward(attitude: vec3<f32>) -> vec3<f32> {
    let yaw = attitude.z;
    let pitch = attitude.y;
    let heading = vec3<f32>(cos(yaw), 0.0, sin(yaw));
    let right = vec3<f32>(heading.z, 0.0, -heading.x);
    return normalize(rotate_around_axis(heading, right, pitch));
}

fn body_up(attitude: vec3<f32>) -> vec3<f32> {
    let yaw = attitude.z;
    let pitch = attitude.y;
    let roll = attitude.x;
    let heading = vec3<f32>(cos(yaw), 0.0, sin(yaw));
    let right = vec3<f32>(heading.z, 0.0, -heading.x);
    let tilted_up = rotate_around_axis(vec3<f32>(0.0, 1.0, 0.0), right, pitch);
    return normalize(rotate_around_axis(tilted_up, heading, -roll));
}

fn body_right(attitude: vec3<f32>) -> vec3<f32> {
    return normalize(cross(body_up(attitude), body_forward(attitude)));
}

fn wrap_angle(angle: f32) -> f32 {
    let two_pi = 6.283185307179586;
    return angle - floor((angle + 3.141592653589793) / two_pi) * two_pi;
}

fn gate_slot(env_index: u32, gate_index: u32) -> u32 {
    return env_index * params.max_gates_per_env + gate_index;
}

fn env_obstacle_offset(index: u32) -> u32 {
    return params.env_count * params.max_gates_per_env + index * params.max_obstacles_per_env;
}

fn obstacle_slot(env_index: u32, obstacle_index: u32) -> u32 {
    return env_obstacle_offset(env_index) + obstacle_index;
}

fn family_spacing_scale(grammar_id: u32) -> f32 {
    if (grammar_id == 1u) {
        return 1.12;
    }
    if (grammar_id == 2u) {
        return 0.94;
    }
    if (grammar_id == 3u) {
        return 1.06;
    }
    return 1.0;
}

fn family_offset_scale(grammar_id: u32) -> f32 {
    if (grammar_id == 1u) {
        return 1.15;
    }
    if (grammar_id == 2u) {
        return 0.9;
    }
    if (grammar_id == 3u) {
        return 1.25;
    }
    return 1.0;
}

fn family_radius_scale(grammar_id: u32) -> f32 {
    if (grammar_id == 1u) {
        return 1.08;
    }
    if (grammar_id == 2u) {
        return 0.92;
    }
    if (grammar_id == 3u) {
        return 1.18;
    }
    return 1.0;
}

fn resolved_turn_sign(stage: StageSpec, stage_index: u32, grammar_id: u32) -> f32 {
    let base = stage_direction(stage);
    if (grammar_id == 1u) {
        return -base;
    }
    if (grammar_id == 2u) {
        return base * select(1.0, -1.0, (stage_index & 1u) != 0u);
    }
    if (grammar_id == 3u) {
        let flip = hash_to_unit(stage_index * 977u + 0x51u) > 0.42;
        return base * select(1.0, -1.0, flip);
    }
    return base;
}

fn offset_direction(stage_index: u32, grammar_id: u32) -> f32 {
    if (grammar_id == 1u) {
        return select(-1.0, 1.0, (stage_index & 1u) == 0u);
    }
    if (grammar_id == 2u) {
        return select(1.0, -1.0, (stage_index & 1u) == 0u);
    }
    if (grammar_id == 3u) {
        return select(-1.0, 1.0, hash_to_unit(stage_index * 313u + 0x21u) > 0.5);
    }
    return 1.0;
}

fn stage_direction(stage: StageSpec) -> f32 {
    return select(-1.0, 1.0, stage.flags != 0u);
}

fn configured_gate_count(index: u32) -> u32 {
    return min(layout_headers.values[index].gate_count, params.max_gates_per_env);
}

fn generated_obstacle_count() -> u32 {
    return 0u;
}

fn env_gate_offset(index: u32) -> u32 {
    return layout_headers.values[index].gate_offset;
}

fn gate_from_stage(
    stage: StageSpec,
    cursor_position: vec2<f32>,
    cursor_forward: vec2<f32>,
    local_gate_index: u32,
    total_gate_index: u32,
    total_gate_count: u32,
    stage_index: u32,
    seed: u32,
    grammar_id: u32,
) -> Gate {
    let lateral_axis = vec2<f32>(-cursor_forward.y, cursor_forward.x);
    let progress = (f32(total_gate_index) + 0.5) / max(f32(total_gate_count), 1.0);
    let spacing = stage.spacing * family_spacing_scale(grammar_id);
    let offset_scale = family_offset_scale(grammar_id);
    let radius = max(stage.radius * family_radius_scale(grammar_id), 0.1);
    let elevation =
        sin(progress * 6.283185307179586 + hash_to_unit(seed ^ 0x44f1u) * 6.283185307179586)
            * stage.vertical_amp
        + (hash_to_unit(seed ^ (total_gate_index * 97u + 0x0f0fu)) - 0.5) * 0.12;

    var gate: Gate;
    if (stage.kind == 2u) {
        let t = (f32(local_gate_index) + 1.0) / max(f32(stage.gate_count), 1.0);
        let lateral =
            sin(t * 3.141592653589793)
            * stage.lateral_amp
            * offset_scale
            * offset_direction(stage_index, grammar_id);
        let center_2d =
            cursor_position + cursor_forward * spacing + lateral_axis * lateral;
        gate.center = vec4<f32>(center_2d.x, elevation, center_2d.y, 0.0);
        gate.forward = vec4<f32>(cursor_forward.x, 0.0, cursor_forward.y, 0.0);
    } else if (stage.kind == 3u) {
        let turn_sign = resolved_turn_sign(stage, stage_index, grammar_id);
        let exit_forward = normalize(rotate_vec2(cursor_forward, turn_sign * stage.turn_radians));
        let setup_distance = max(radius * 0.65, 2.1);
        let exit_spacing = max(radius * 0.95, 2.4);
        let corner_anchor = cursor_position + cursor_forward * setup_distance;
        if (stage.gate_count <= 1u) {
            let center_2d = corner_anchor + exit_forward * (exit_spacing * 0.35);
            let blended_forward = normalize(cursor_forward + exit_forward);
            gate.center = vec4<f32>(center_2d.x, elevation, center_2d.y, 0.0);
            gate.forward = vec4<f32>(blended_forward.x, 0.0, blended_forward.y, 0.0);
        } else {
            let normalized_index =
                f32(local_gate_index) / max(f32(stage.gate_count - 1u), 1.0);
            let center_2d = corner_anchor + exit_forward * (normalized_index * exit_spacing);
            gate.center = vec4<f32>(center_2d.x, elevation, center_2d.y, 0.0);
            gate.forward = vec4<f32>(exit_forward.x, 0.0, exit_forward.y, 0.0);
        }
    } else {
        let center_2d = cursor_position + cursor_forward * spacing;
        gate.center = vec4<f32>(center_2d.x, elevation, center_2d.y, 0.0);
        gate.forward = vec4<f32>(cursor_forward.x, 0.0, cursor_forward.y, 0.0);
    }
    gate.half_extents = vec4<f32>(
        stage.hole_half_width,
        stage.hole_half_height,
        params.gate_depth_half,
        0.0,
    );
    return gate;
}

fn next_cursor_forward(
    stage: StageSpec,
    cursor_forward: vec2<f32>,
    stage_index: u32,
    grammar_id: u32,
) -> vec2<f32> {
    if (stage.kind == 3u) {
        return normalize(rotate_vec2(
            cursor_forward,
            resolved_turn_sign(stage, stage_index, grammar_id) * stage.turn_radians,
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

fn gate_up_axis() -> vec3<f32> {
    return vec3<f32>(0.0, 1.0, 0.0);
}

fn gate_right_axis(forward: vec3<f32>) -> vec3<f32> {
    let horizontal_forward = normalize(vec3<f32>(forward.x, 0.0, forward.z));
    return normalize(cross(gate_up_axis(), horizontal_forward));
}

fn gate_hole_contains(gate: Gate, offset: vec3<f32>) -> bool {
    let gate_forward = normalize(gate.forward.xyz);
    let gate_right = gate_right_axis(gate_forward);
    let right_distance = abs(dot(offset, gate_right));
    let up_distance = abs(dot(offset, gate_up_axis()));
    let hole_half_width = max(gate.half_extents.x - params.drone_radius, 0.0);
    let hole_half_height = max(gate.half_extents.y - params.drone_radius, 0.0);
    return right_distance <= hole_half_width && up_distance <= hole_half_height;
}

fn gate_frame_collision(gate: Gate, position: vec3<f32>) -> bool {
    let gate_forward = normalize(gate.forward.xyz);
    let gate_right = gate_right_axis(gate_forward);
    let offset = position - gate.center.xyz;
    let plane_distance = abs(dot(offset, gate_forward));
    let right_distance = abs(dot(offset, gate_right));
    let up_distance = abs(dot(offset, gate_up_axis()));
    let outer_half_width = gate.half_extents.x + params.gate_frame_thickness + params.drone_radius;
    let outer_half_height = gate.half_extents.y + params.gate_frame_thickness + params.drone_radius;
    let within_outer = plane_distance <= params.gate_depth_half + params.drone_radius
        && right_distance <= outer_half_width
        && up_distance <= outer_half_height;
    return within_outer && !gate_hole_contains(gate, offset);
}

fn crossed_gate_plane(prev_position: vec3<f32>, next_position: vec3<f32>, gate: Gate) -> bool {
    let gate_forward = normalize(gate.forward.xyz);
    let prev_distance = dot(prev_position - gate.center.xyz, gate_forward);
    let next_distance = dot(next_position - gate.center.xyz, gate_forward);
    if (!(prev_distance <= 0.0 && next_distance >= 0.0)) {
        return false;
    }

    let denominator = prev_distance - next_distance;
    let interpolation = select(
        0.0,
        clamp(prev_distance / denominator, 0.0, 1.0),
        abs(denominator) > 1e-5,
    );
    let hit_position = prev_position + (next_position - prev_position) * interpolation;
    return gate_hole_contains(gate, hit_position - gate.center.xyz);
}

fn obstacle_collision(obstacle: Gate, position: vec3<f32>) -> bool {
    let obstacle_forward = normalize(obstacle.forward.xyz);
    let obstacle_right = gate_right_axis(obstacle_forward);
    let offset = position - obstacle.center.xyz;
    let local = vec3<f32>(
        dot(offset, obstacle_right),
        dot(offset, gate_up_axis()),
        dot(offset, obstacle_forward),
    );
    let clamped = clamp(local, -obstacle.half_extents.xyz, obstacle.half_extents.xyz);
    let delta = local - clamped;
    return dot(delta, delta) <= params.drone_radius * params.drone_radius;
}

fn family_segment_count(curriculum_stage: u32, grammar_id: u32) -> u32 {
    if (curriculum_stage == 0u) {
        return 3u;
    }
    if (curriculum_stage == 1u) {
        return 4u;
    }
    if (curriculum_stage == 2u) {
        if (grammar_id == 3u) {
            return 5u;
        }
        return 4u;
    }
    return 5u;
}

fn curriculum_family(curriculum_stage: u32, grammar_id: u32) -> u32 {
    let family = grammar_id % 4u;
    if (curriculum_stage == 0u) {
        return family % 2u;
    }
    if (curriculum_stage == 1u) {
        return family % 3u;
    }
    return family;
}

fn curriculum_length_scale(curriculum_stage: u32) -> f32 {
    if (curriculum_stage == 0u) {
        return 1.22;
    }
    if (curriculum_stage == 1u) {
        return 1.08;
    }
    if (curriculum_stage == 2u) {
        return 1.0;
    }
    return 1.14;
}

fn curriculum_vertical_scale(curriculum_stage: u32) -> f32 {
    if (curriculum_stage == 0u) {
        return 0.12;
    }
    if (curriculum_stage == 1u) {
        return 0.3;
    }
    if (curriculum_stage == 2u) {
        return 0.85;
    }
    return 1.35;
}

fn curriculum_hole_scale(curriculum_stage: u32) -> f32 {
    if (curriculum_stage == 0u) {
        return 1.25;
    }
    if (curriculum_stage == 1u) {
        return 1.12;
    }
    if (curriculum_stage == 3u) {
        return 0.92;
    }
    return 1.0;
}

fn family_segment_length(
    grammar_id: u32,
    segment_index: u32,
    seed: u32,
    difficulty: f32,
    curriculum_stage: u32,
    path_scale: f32,
) -> f32 {
    var base = 8.0;
    var jitter = 0.9;
    if (curriculum_stage == 0u) {
        if (segment_index == 0u) {
            base = 11.5;
        } else if (segment_index == 1u) {
            base = 10.4;
        } else {
            base = 9.1;
        }
        jitter = 0.45;
    } else if (curriculum_stage == 1u) {
        if (grammar_id == 0u) {
            if (segment_index == 0u) {
                base = 10.8;
            } else if (segment_index == 1u) {
                base = 8.6;
            } else if (segment_index == 2u) {
                base = 9.4;
            } else {
                base = 7.8;
            }
        } else if (grammar_id == 1u) {
            if (segment_index == 0u) {
                base = 9.8;
            } else if (segment_index == 1u) {
                base = 8.4;
            } else if (segment_index == 2u) {
                base = 8.8;
            } else {
                base = 8.1;
            }
        } else {
            if (segment_index == 0u) {
                base = 10.1;
            } else if (segment_index == 1u) {
                base = 7.9;
            } else if (segment_index == 2u) {
                base = 9.1;
            } else {
                base = 7.4;
            }
        }
        jitter = 0.7;
    } else if (curriculum_stage == 2u) {
        if (grammar_id == 0u) {
            if (segment_index == 0u) {
                base = 10.5;
            } else if (segment_index == 1u) {
                base = 7.8;
            } else if (segment_index == 2u) {
                base = 9.6;
            } else if (segment_index == 3u) {
                base = 7.2;
            } else {
                base = 8.9;
            }
        } else if (grammar_id == 1u) {
            if (segment_index == 0u) {
                base = 9.2;
            } else if (segment_index == 1u) {
                base = 8.6;
            } else if (segment_index == 2u) {
                base = 8.2;
            } else {
                base = 7.4;
            }
        } else if (grammar_id == 2u) {
            if (segment_index == 0u) {
                base = 12.8;
            } else if (segment_index == 1u) {
                base = 5.4;
            } else if (segment_index == 2u) {
                base = 11.6;
            } else {
                base = 6.1;
            }
            jitter = 0.7;
        } else {
            if (segment_index == 0u) {
                base = 9.6;
            } else if (segment_index == 1u) {
                base = 5.3;
            } else if (segment_index == 2u) {
                base = 5.1;
            } else if (segment_index == 3u) {
                base = 9.1;
            } else {
                base = 8.4;
            }
            jitter = 0.75;
        }
    } else {
        if (grammar_id == 0u) {
            if (segment_index == 0u) {
                base = 11.4;
            } else if (segment_index == 1u) {
                base = 7.0;
            } else if (segment_index == 2u) {
                base = 10.2;
            } else if (segment_index == 3u) {
                base = 6.8;
            } else {
                base = 9.4;
            }
        } else if (grammar_id == 1u) {
            if (segment_index == 0u) {
                base = 8.8;
            } else if (segment_index == 1u) {
                base = 8.1;
            } else if (segment_index == 2u) {
                base = 7.6;
            } else if (segment_index == 3u) {
                base = 7.4;
            } else {
                base = 8.0;
            }
        } else if (grammar_id == 2u) {
            if (segment_index == 0u) {
                base = 13.4;
            } else if (segment_index == 1u) {
                base = 4.6;
            } else if (segment_index == 2u) {
                base = 12.4;
            } else if (segment_index == 3u) {
                base = 4.8;
            } else {
                base = 10.2;
            }
        } else {
            if (segment_index == 0u) {
                base = 9.8;
            } else if (segment_index == 1u) {
                base = 5.0;
            } else if (segment_index == 2u) {
                base = 5.2;
            } else if (segment_index == 3u) {
                base = 9.3;
            } else {
                base = 7.6;
            }
        }
        jitter = 0.95;
    }
    let random = hash_to_unit(seed ^ (grammar_id * 0x9e37u + segment_index * 0x85ebu));
    return (base + (random - 0.5) * jitter + difficulty * 0.8)
        * curriculum_length_scale(curriculum_stage)
        * path_scale;
}

fn family_turn_radians(curriculum_stage: u32, grammar_id: u32, segment_index: u32) -> f32 {
    let quarter_turn = 1.5707963267948966;
    if (curriculum_stage == 0u) {
        if (grammar_id == 0u) {
            if (segment_index == 0u) {
                return 0.18;
            }
            if (segment_index == 1u) {
                return -0.14;
            }
            return 0.0;
        }
        if (segment_index == 0u) {
            return -0.16;
        }
        if (segment_index == 1u) {
            return 0.12;
        }
        return 0.0;
    }
    if (curriculum_stage == 1u) {
        if (grammar_id == 0u) {
            if (segment_index == 0u) {
                return 0.34;
            }
            if (segment_index == 1u) {
                return -0.28;
            }
            if (segment_index == 2u) {
                return 0.18;
            }
            return 0.0;
        }
        if (grammar_id == 1u) {
            if (segment_index == 0u) {
                return -0.32;
            }
            if (segment_index == 1u) {
                return 0.26;
            }
            if (segment_index == 2u) {
                return -0.18;
            }
            return 0.0;
        }
        if (segment_index == 0u) {
            return 0.22;
        }
        if (segment_index == 1u) {
            return 0.0;
        }
        if (segment_index == 2u) {
            return -0.24;
        }
        return 0.0;
    }
    if (curriculum_stage == 2u) {
        if (grammar_id == 0u) {
            if (segment_index == 0u) {
                return quarter_turn;
            }
            if (segment_index == 1u) {
                return -quarter_turn;
            }
            if (segment_index == 2u) {
                return -quarter_turn;
            }
            if (segment_index == 3u) {
                return quarter_turn;
            }
            return quarter_turn;
        }
        if (grammar_id == 1u) {
            if (segment_index < 3u) {
                return quarter_turn;
            }
            return 0.0;
        }
        if (grammar_id == 2u) {
            if (segment_index < 3u) {
                return quarter_turn;
            }
            return 0.0;
        }
        if (segment_index == 0u) {
            return -quarter_turn;
        }
        if (segment_index == 1u) {
            return quarter_turn;
        }
        if (segment_index == 2u) {
            return quarter_turn;
        }
        if (segment_index == 3u) {
            return -quarter_turn;
        }
        return 0.0;
    }
    if (grammar_id == 0u) {
        if (segment_index == 0u) {
            return quarter_turn;
        }
        if (segment_index == 1u) {
            return -quarter_turn;
        }
        if (segment_index == 2u) {
            return quarter_turn;
        }
        if (segment_index == 3u) {
            return -quarter_turn;
        }
        return quarter_turn;
    }
    if (grammar_id == 1u) {
        if (segment_index < 4u) {
            return quarter_turn;
        }
        return 0.0;
    }
    if (grammar_id == 2u) {
        if (segment_index == 0u) {
            return quarter_turn;
        }
        if (segment_index == 1u) {
            return quarter_turn;
        }
        if (segment_index == 2u) {
            return -quarter_turn;
        }
        if (segment_index == 3u) {
            return -quarter_turn;
        }
        return quarter_turn;
    }
    if (segment_index == 0u) {
        return -quarter_turn;
    }
    if (segment_index == 1u) {
        return quarter_turn;
    }
    if (segment_index == 2u) {
        return quarter_turn;
    }
    if (segment_index == 3u) {
        return -quarter_turn;
    }
    return 0.0;
}

fn path_total_length(
    grammar_id: u32,
    seed: u32,
    difficulty: f32,
    curriculum_stage: u32,
    path_scale: f32,
) -> f32 {
    let segment_count = family_segment_count(curriculum_stage, grammar_id);
    var total = 0.0;
    var segment_index = 0u;
    loop {
        if (segment_index >= segment_count) {
            break;
        }
        total = total + family_segment_length(
            grammar_id,
            segment_index,
            seed,
            difficulty,
            curriculum_stage,
            path_scale,
        );
        segment_index = segment_index + 1u;
    }
    return total;
}

fn family_path_scale(gate_count: u32, difficulty: f32, base_total_length: f32) -> f32 {
    let required_usable = max(f32(max(gate_count, 1u) - 1u), 0.0) * minimum_gate_path_spacing(difficulty);
    let required_total = required_usable + 3.6;
    return max(1.0, required_total / max(base_total_length, 1.0));
}

fn sample_family_path(
    grammar_id: u32,
    seed: u32,
    difficulty: f32,
    curriculum_stage: u32,
    course_angle: f32,
    distance_along_path: f32,
    path_scale: f32,
) -> Gate {
    let segment_count = family_segment_count(curriculum_stage, grammar_id);
    var remaining = distance_along_path;
    var cursor_position = vec2<f32>(0.0, 0.0);
    var cursor_forward = rotate_vec2(vec2<f32>(1.0, 0.0), course_angle);
    var segment_index = 0u;

    loop {
        if (segment_index >= segment_count) {
            break;
        }

        let segment_length =
            family_segment_length(
                grammar_id,
                segment_index,
                seed,
                difficulty,
                curriculum_stage,
                path_scale,
            );
        let is_last = segment_index + 1u >= segment_count;
        if (remaining <= segment_length || is_last) {
            let local_distance = min(remaining, segment_length);
            let center_2d = cursor_position + cursor_forward * local_distance;
            let path_progress =
                distance_along_path
                / max(
                    path_total_length(
                        grammar_id,
                        seed,
                        difficulty,
                        curriculum_stage,
                        path_scale,
                    ),
                    1.0,
                );
            let phase = hash_to_unit(seed ^ 0x44f1u) * 6.283185307179586;
            var elevation =
                sin(path_progress * 6.283185307179586 + phase) * (0.18 + difficulty * 0.16)
                + (hash_to_unit(seed ^ (segment_index * 97u + 0x0f0fu)) - 0.5) * 0.12;
            if (grammar_id == 1u) {
                elevation = elevation
                    + sin(path_progress * 12.566370614359172 + phase * 0.7) * 0.18;
            } else if (grammar_id == 2u) {
                elevation = elevation
                    + select(-0.22, 0.28, path_progress > 0.52)
                    + sin(path_progress * 9.42477796076938 + phase) * 0.08;
            } else if (grammar_id == 3u) {
                let hump =
                    exp(-pow((path_progress - 0.45) * 4.0, 2.0)) * (0.32 + difficulty * 0.14);
                elevation = elevation + hump;
            }
            elevation = elevation * curriculum_vertical_scale(curriculum_stage);
            let hole_scale = curriculum_hole_scale(curriculum_stage);
            let hole_half_width = (0.58 - difficulty * 0.08) * hole_scale;
            let hole_half_height = (0.58 - difficulty * 0.06) * hole_scale;
            var gate: Gate;
            gate.center = vec4<f32>(center_2d.x, elevation, center_2d.y, 0.0);
            gate.forward = vec4<f32>(cursor_forward.x, 0.0, cursor_forward.y, 0.0);
            gate.half_extents = vec4<f32>(
                hole_half_width,
                hole_half_height,
                params.gate_depth_half,
                0.0,
            );
            return gate;
        }

        remaining = remaining - segment_length;
        cursor_position = cursor_position + cursor_forward * segment_length;
        cursor_forward = normalize(rotate_vec2(
            cursor_forward,
            family_turn_radians(curriculum_stage, grammar_id, segment_index),
        ));
        segment_index = segment_index + 1u;
    }

    var fallback: Gate;
    fallback.center = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    fallback.half_extents = vec4<f32>(0.58, 0.58, params.gate_depth_half, 0.0);
    fallback.forward = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    return fallback;
}

fn minimum_gate_separation(difficulty: f32) -> f32 {
    return 2.6 + difficulty * 0.35;
}

fn minimum_gate_path_spacing(difficulty: f32) -> f32 {
    return 4.4 + difficulty * 0.6;
}

fn minimum_previous_gate_exit_distance(previous_gate: Gate) -> f32 {
    return max(previous_gate.half_extents.x, previous_gate.half_extents.y) + 1.4;
}

fn previous_target_gate(index: u32, current_gate: u32, current_lap: u32) -> Gate {
    let gate_count = max(configured_gate_count(index), 1u);
    if (current_gate > 0u) {
        return gates.values[env_gate_offset(index) + current_gate - 1u];
    }
    if (course_header.loop_enabled != 0u && current_lap > 0u) {
        return gates.values[env_gate_offset(index) + gate_count - 1u];
    }
    return gates.values[env_gate_offset(index)];
}

fn has_cleared_previous_gate_zone(index: u32, state: EnvState, position: vec3<f32>) -> bool {
    if (state.current_gate == 0u && state.current_lap == 0u) {
        return true;
    }
    let previous_gate = previous_target_gate(index, state.current_gate, state.current_lap);
    return distance(position, previous_gate.center.xyz)
        >= minimum_previous_gate_exit_distance(previous_gate);
}

fn curriculum_gate_count(curriculum_stage: u32, base_gate_count: u32) -> u32 {
    if (curriculum_stage == 0u) {
        return clamp(base_gate_count / 2u, 5u, 7u);
    }
    if (curriculum_stage == 1u) {
        return clamp((base_gate_count * 2u) / 3u, 7u, 9u);
    }
    if (curriculum_stage == 2u) {
        return max(select(0u, base_gate_count - 2u, base_gate_count >= 2u), 8u);
    }
    return base_gate_count;
}

fn reset_env(index: u32) {
    let reset = reset_params.values[index];
    let count = curriculum_gate_count(
        reset.curriculum_stage,
        min(course_header.total_gate_count, params.max_gates_per_env),
    );
    let base_slot = env_gate_offset(index);
    let family_id = curriculum_family(reset.curriculum_stage, reset.grammar_id);
    layout_headers.values[index].gate_count = count;
    layout_headers.values[index].obstacle_count = 0u;

    let course_angle =
        hash_to_unit(reset.seed ^ 0x12345u) * 6.283185307179586
        + f32(family_id) * 0.37;
    let base_total_path_length = path_total_length(
        family_id,
        reset.seed,
        reset.difficulty,
        reset.curriculum_stage,
        1.0,
    );
    let path_scale = family_path_scale(count, reset.difficulty, base_total_path_length);
    let total_path_length = path_total_length(
        family_id,
        reset.seed,
        reset.difficulty,
        reset.curriculum_stage,
        path_scale,
    );
    let entry_margin = min(1.8, total_path_length * 0.12);
    let exit_margin = min(1.2, total_path_length * 0.08);
    let usable_length = max(total_path_length - entry_margin - exit_margin, 0.5);
    var gate_index = 0u;

    loop {
        if (gate_index >= count) {
            break;
        }

        let gate_t = select(
            0.5,
            f32(gate_index) / max(f32(count - 1u), 1.0),
            count > 1u,
        );
        var distance_along_path = entry_margin + gate_t * usable_length;
        if (gate_index > 0u) {
            let previous_gate_t = f32(gate_index - 1u) / max(f32(count - 1u), 1.0);
            let previous_distance = entry_margin + previous_gate_t * usable_length;
            distance_along_path = max(
                distance_along_path,
                previous_distance + minimum_gate_path_spacing(reset.difficulty),
            );
            distance_along_path = min(entry_margin + usable_length, distance_along_path);
        }
        var gate = sample_family_path(
            family_id,
            reset.seed,
            reset.difficulty,
            reset.curriculum_stage,
            course_angle,
            distance_along_path,
            path_scale,
        );
        if (gate_index > 0u) {
            let previous_gate = gates.values[gate_slot(index, gate_index - 1u)];
            var adjustment = 0u;
            loop {
                let gate_spacing = distance(gate.center.xyz, previous_gate.center.xyz);
                if (gate_spacing >= minimum_gate_separation(reset.difficulty) || adjustment >= 6u) {
                    break;
                }
                let pull_forward = 1.2 + f32(adjustment) * 0.7;
                distance_along_path = min(entry_margin + usable_length, distance_along_path + pull_forward);
                gate = sample_family_path(
                    family_id,
                    reset.seed,
                    reset.difficulty,
                    reset.curriculum_stage,
                    course_angle,
                    distance_along_path,
                    path_scale,
                );
                adjustment = adjustment + 1u;
            }
        }
        gates.values[gate_slot(index, gate_index)] = gate;
        gate_index = gate_index + 1u;
    }

    if (count > 1u) {
        let first_gate = gates.values[base_slot];
        let last_slot = gate_slot(index, count - 1u);
        var last_gate = gates.values[last_slot];
        var adjustment = 0u;
        loop {
            let closure_distance = distance(last_gate.center.xyz, first_gate.center.xyz);
            if (closure_distance >= 4.2 || adjustment >= 5u) {
                break;
            }
            let pullback = (f32(adjustment) + 1.0) * 1.9;
            let adjusted_distance = entry_margin + max(usable_length - pullback, 0.5);
            last_gate = sample_family_path(
                family_id,
                reset.seed,
                reset.difficulty,
                reset.curriculum_stage,
                course_angle,
                adjusted_distance,
                path_scale,
            );
            gates.values[last_slot] = last_gate;
            adjustment = adjustment + 1u;
        }
    }

    loop {
        if (gate_index >= params.max_gates_per_env) {
            break;
        }
        let slot = gate_slot(index, gate_index);
        gates.values[slot].center = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        gates.values[slot].half_extents = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        gates.values[slot].forward = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        gate_index = gate_index + 1u;
    }

    let obstacle_count = generated_obstacle_count();
    var obstacle_index = 0u;
    loop {
        if (obstacle_index >= obstacle_count) {
            break;
        }

        let source_gate_index = min(obstacle_index * 2u, max(count, 1u) - 1u);
        let next_gate_index = min(source_gate_index + 1u, max(count, 1u) - 1u);
        let gate_a = gates.values[gate_slot(index, source_gate_index)];
        let gate_b = gates.values[gate_slot(index, next_gate_index)];
        let segment = gate_b.center.xyz - gate_a.center.xyz;
        let horizontal_segment = vec3<f32>(segment.x, 0.0, segment.z);
        let segment_length = length(horizontal_segment);
        let travel_forward =
            normalize(select(gate_a.forward.xyz, horizontal_segment, segment_length > 1e-5));
        let travel_right = gate_right_axis(travel_forward);
        let t = 0.35 + hash_to_unit(reset.seed ^ (obstacle_index * 193u + 0x5511u)) * 0.3;
        let centerline = gate_a.center.xyz + segment * t;
        let side_sign = select(
            -1.0,
            1.0,
            hash_to_unit(reset.seed ^ (obstacle_index * 313u + 0x33a1u)) > 0.5,
        );
        let lateral_offset =
            (gate_a.half_extents.x + 0.55 + reset.difficulty * 0.85
                + hash_to_unit(reset.seed ^ (obstacle_index * 733u + 0x91u)) * 0.35)
            * side_sign;
        let half_width =
            0.16 + hash_to_unit(reset.seed ^ (obstacle_index * 151u + 0x1717u)) * 0.1;
        let half_height =
            0.22
            + reset.difficulty * 0.18
            + hash_to_unit(reset.seed ^ (obstacle_index * 271u + 0x2021u)) * 0.16;
        let half_depth =
            0.16 + hash_to_unit(reset.seed ^ (obstacle_index * 419u + 0x0aa1u)) * 0.1;
        let center =
            centerline
            + travel_right * lateral_offset
            + vec3<f32>(0.0, half_height - 0.02, 0.0);
        let slot = obstacle_slot(index, obstacle_index);
        gates.values[slot].center = vec4<f32>(center, 0.0);
        gates.values[slot].half_extents = vec4<f32>(half_width, half_height, half_depth, 0.0);
        gates.values[slot].forward = vec4<f32>(travel_forward, 0.0);
        obstacle_index = obstacle_index + 1u;
    }

    loop {
        if (obstacle_index >= params.max_obstacles_per_env) {
            break;
        }
        let slot = obstacle_slot(index, obstacle_index);
        gates.values[slot].center = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        gates.values[slot].half_extents = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        gates.values[slot].forward = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        obstacle_index = obstacle_index + 1u;
    }

    let first_gate = gates.values[base_slot];
    let start_position = first_gate.center.xyz - first_gate.forward.xyz * 1.2 + vec3<f32>(0.0, 0.2, 0.0);
    let start_yaw = atan2(first_gate.forward.z, first_gate.forward.x);
    states.values[index].position = vec4<f32>(start_position, 0.0);
    states.values[index].velocity = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    states.values[index].attitude = vec4<f32>(0.0, 0.0, start_yaw, 0.0);
    states.values[index].angular_velocity = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    states.values[index].motor_thrust = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    states.values[index].step_count = 0u;
    states.values[index].done = 0u;
    states.values[index].current_gate = 0u;
    states.values[index].current_lap = 0u;

    observations.values[index].position = vec4<f32>(start_position, 0.0);
    observations.values[index].velocity = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    observations.values[index].attitude = vec4<f32>(0.0, 0.0, start_yaw, 0.0);
    observations.values[index].angular_velocity = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    observations.values[index].target_gate_position = first_gate.center;
    observations.values[index].target_gate_forward_progress =
        vec4<f32>(first_gate.forward.xyz, 0.0);
    observations.values[index].metrics = vec4<f32>(0.0, 0.0, 1.0, 0.0);

    reward_done.values[index].reward = 0.0;
    reward_done.values[index].done = 0u;
    reward_done.values[index].done_reason = DONE_REASON_NONE;
    reward_done.values[index]._pad0 = 0u;
    reward_done.values[index].progress_reward = 0.0;
    reward_done.values[index].distance_penalty = 0.0;
    reward_done.values[index].alignment_reward = 0.0;
    reward_done.values[index].tilt_penalty = 0.0;
    reward_done.values[index].completion_bonus = 0.0;
    reward_done.values[index].collision_penalty = 0.0;
    reward_done.values[index]._pad1 = 0.0;
    reward_done.values[index]._pad2 = 0.0;
    reset_mask.values[index] = 0u;
}

fn current_target_gate(index: u32, current_gate: u32) -> Gate {
    let safe_count = max(configured_gate_count(index), 1u);
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
        var debug = reward_done.values[index];
        let position = state.position.xyz;
        let target_position = target_gate.center.xyz;
        let to_gate = target_position - position;
        let distance_to_gate = length(to_gate);
        let target_forward = normalize(target_gate.forward.xyz);
        let gate_alignment = max(
            dot(normalize(select(target_forward, to_gate, distance_to_gate > 1e-5)), target_forward),
            -1.0,
        );
        observations.values[index].position = state.position;
        observations.values[index].velocity = state.velocity;
        observations.values[index].attitude = state.attitude;
        observations.values[index].angular_velocity = state.angular_velocity;
        observations.values[index].target_gate_position = target_gate.center;
        observations.values[index].target_gate_forward_progress = vec4<f32>(
            target_forward,
            progress_value(state.current_lap, state.current_gate, configured_gate_count(index)),
        );
        observations.values[index].metrics = vec4<f32>(
            progress_value(state.current_lap, state.current_gate, configured_gate_count(index)),
            distance_to_gate,
            gate_alignment,
            dot(state.motor_thrust, vec4<f32>(0.25)),
        );
        debug.reward = 0.0;
        debug.done = 1u;
        reward_done.values[index] = debug;
        return;
    }

    let prev_position = state.position.xyz;
    let action = clamp_motor_command(actions.values[index].motor_command);
    state.motor_thrust = state.motor_thrust
        + (action - state.motor_thrust) * params.motor_response * params.dt_seconds;

    var attitude = state.attitude.xyz;
    let forward = body_forward(attitude);
    let up = body_up(attitude);
    let right = body_right(attitude);
    let collective_thrust = dot(state.motor_thrust, vec4<f32>(1.0)) * params.thrust_scale;
    let acceleration = up * (collective_thrust / params.mass)
        - vec3<f32>(0.0, params.gravity, 0.0)
        - state.velocity.xyz * params.linear_drag;

    let roll_torque =
        ((state.motor_thrust.y + state.motor_thrust.z) - (state.motor_thrust.x + state.motor_thrust.w))
        * params.arm_length;
    let pitch_torque =
        ((state.motor_thrust.z + state.motor_thrust.w) - (state.motor_thrust.x + state.motor_thrust.y))
        * params.arm_length;
    let yaw_torque =
        ((state.motor_thrust.x + state.motor_thrust.z) - (state.motor_thrust.y + state.motor_thrust.w))
        * params.yaw_torque_scale;

    state.angular_velocity = vec4<f32>(
        state.angular_velocity.xyz
            + (vec3<f32>(roll_torque, pitch_torque, yaw_torque)
                - state.angular_velocity.xyz * params.angular_drag)
                * params.dt_seconds,
        0.0,
    );
    attitude = attitude + state.angular_velocity.xyz * params.dt_seconds;
    attitude.x = clamp(attitude.x, -0.9, 0.9);
    attitude.y = clamp(attitude.y, -0.9, 0.9);
    attitude.z = wrap_angle(attitude.z);
    state.attitude = vec4<f32>(attitude, 0.0);

    state.velocity = vec4<f32>(state.velocity.xyz + acceleration * params.dt_seconds, 0.0);
    state.position = vec4<f32>(state.position.xyz + state.velocity.xyz * params.dt_seconds, 0.0);
    state.step_count = state.step_count + 1u;

    let gate_count = configured_gate_count(index);
    let target_position = target_gate.center.xyz;
    var target_forward = normalize(target_gate.forward.xyz);
    var delta = target_position - state.position.xyz;
    var distance_to_gate = length(delta);
    let previous_gate_cleared = has_cleared_previous_gate_zone(index, state, state.position.xyz);
    let passed_gate =
        previous_gate_cleared && crossed_gate_plane(prev_position, state.position.xyz, target_gate);
    let collided_gate = !passed_gate && gate_frame_collision(target_gate, state.position.xyz);
    var collided_obstacle = false;
    var done_reason = DONE_REASON_NONE;
    var obstacle_index = 0u;
    loop {
        if (obstacle_index >= generated_obstacle_count()) {
            break;
        }
        let obstacle = gates.values[obstacle_slot(index, obstacle_index)];
        if (obstacle.half_extents.x > 0.0 && obstacle_collision(obstacle, state.position.xyz)) {
            collided_obstacle = true;
            break;
        }
        obstacle_index = obstacle_index + 1u;
    }
    if (passed_gate) {
        if (state.current_gate + 1u < gate_count) {
            state.current_gate = state.current_gate + 1u;
            target_gate = current_target_gate(index, state.current_gate);
            target_forward = normalize(target_gate.forward.xyz);
            delta = target_gate.center.xyz - state.position.xyz;
            distance_to_gate = length(delta);
        } else if (course_header.loop_enabled != 0u
            && state.current_lap + 1u < max(course_header.laps_required, 1u))
        {
            state.current_lap = state.current_lap + 1u;
            state.current_gate = 0u;
            target_gate = current_target_gate(index, 0u);
            target_forward = normalize(target_gate.forward.xyz);
            delta = target_gate.center.xyz - state.position.xyz;
            distance_to_gate = length(delta);
        } else {
            state.done = 1u;
            done_reason = done_reason | DONE_REASON_COMPLETE;
        }
    } else {
        if (collided_gate) {
            state.done = 1u;
            done_reason = done_reason | DONE_REASON_GATE_COLLISION;
        }
        if (collided_obstacle) {
            state.done = 1u;
            done_reason = done_reason | DONE_REASON_OBSTACLE_COLLISION;
        }
    }

    let collided_floor = state.position.y <= params.min_altitude + params.drone_radius;
    let out_of_bounds = abs(state.position.x) > params.bounds - params.drone_radius
        || abs(state.position.z) > params.bounds - params.drone_radius
        || state.position.y > params.max_altitude - params.drone_radius;
    let reached_step_limit = state.step_count >= params.max_steps;
    let excessive_tilt = abs(attitude.x) > 1.2 || abs(attitude.y) > 1.2;
    if (collided_floor) {
        state.done = 1u;
        done_reason = done_reason | DONE_REASON_FLOOR_COLLISION;
    }
    if (out_of_bounds) {
        state.done = 1u;
        done_reason = done_reason | DONE_REASON_OUT_OF_BOUNDS;
    }
    if (reached_step_limit) {
        state.done = 1u;
        done_reason = done_reason | DONE_REASON_STEP_LIMIT;
    }
    if (excessive_tilt) {
        state.done = 1u;
        done_reason = done_reason | DONE_REASON_EXCESSIVE_TILT;
    }
    let collided_world = collided_floor || out_of_bounds;

    states.values[index] = state;

    observations.values[index].position = state.position;
    observations.values[index].velocity = state.velocity;
    observations.values[index].attitude = state.attitude;
    observations.values[index].angular_velocity = state.angular_velocity;
    observations.values[index].target_gate_position = target_gate.center;
    let progress = progress_value(state.current_lap, state.current_gate, gate_count);
    let gate_alignment = max(
        dot(normalize(select(target_forward, delta, distance_to_gate > 1e-5)), target_forward),
        -1.0,
    );
    observations.values[index].target_gate_forward_progress = vec4<f32>(target_forward, progress);
    observations.values[index].metrics = vec4<f32>(
        progress,
        distance_to_gate,
        gate_alignment,
        dot(state.motor_thrust, vec4<f32>(0.25)),
    );

    let tilt_magnitude = abs(attitude.x) + abs(attitude.y);
    let collision_happened = collided_gate || collided_obstacle || collided_world;
    let completion_bonus = select(0.0, 4.0, passed_gate);
    let progress_reward = progress * 6.0;
    let distance_penalty = distance_to_gate * 0.35;
    let alignment_reward = gate_alignment * 0.15;
    let tilt_penalty = tilt_magnitude * 0.05;
    let collision_penalty = select(0.0, params.collision_penalty, collision_happened);
    reward_done.values[index].reward =
        progress_reward
        - distance_penalty
        + alignment_reward
        - tilt_penalty
        + completion_bonus
        - collision_penalty;
    reward_done.values[index].done = state.done;
    reward_done.values[index].done_reason = done_reason;
    reward_done.values[index]._pad0 = 0u;
    reward_done.values[index].progress_reward = progress_reward;
    reward_done.values[index].distance_penalty = distance_penalty;
    reward_done.values[index].alignment_reward = alignment_reward;
    reward_done.values[index].tilt_penalty = tilt_penalty;
    reward_done.values[index].completion_bonus = completion_bonus;
    reward_done.values[index].collision_penalty = collision_penalty;
    reward_done.values[index]._pad1 = 0.0;
    reward_done.values[index]._pad2 = 0.0;
}
"#;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EnvState {
    pub position: [f32; 3],
    pub _pad0: f32,
    pub velocity: [f32; 3],
    pub _pad1: f32,
    pub attitude: [f32; 3],
    pub _pad2: f32,
    pub angular_velocity: [f32; 3],
    pub _pad3: f32,
    pub motor_thrust: [f32; 4],
    pub step_count: u32,
    pub done: u32,
    pub current_gate: u32,
    pub current_lap: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Action {
    pub motor_command: [f32; 4],
}

impl Action {
    #[must_use]
    pub fn new(motor_command: [f32; 4]) -> Self {
        Self { motor_command }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Observation {
    pub position: [f32; 3],
    pub _pad0: f32,
    pub velocity: [f32; 3],
    pub _pad1: f32,
    pub attitude: [f32; 3],
    pub _pad2: f32,
    pub angular_velocity: [f32; 3],
    pub _pad3: f32,
    pub target_gate_position: [f32; 3],
    pub _pad4: f32,
    pub target_gate_forward: [f32; 3],
    pub progress: f32,
    pub distance_to_gate: f32,
    pub gate_alignment: f32,
    pub mean_motor_thrust: f32,
    pub _pad5: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RewardDone {
    pub reward: f32,
    pub done: u32,
    pub done_reason: u32,
    pub _pad: u32,
    pub progress_reward: f32,
    pub distance_penalty: f32,
    pub alignment_reward: f32,
    pub tilt_penalty: f32,
    pub completion_bonus: f32,
    pub collision_penalty: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ResetParams {
    pub seed: u32,
    pub grammar_id: u32,
    pub difficulty: f32,
    pub curriculum_stage: u32,
}

impl ResetParams {
    #[must_use]
    pub fn new(seed: u32, grammar_id: u32, difficulty: f32, curriculum_stage: u32) -> Self {
        Self {
            seed,
            grammar_id,
            difficulty,
            curriculum_stage,
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
    pub center: [f32; 3],
    pub _pad0: f32,
    pub half_extents: [f32; 4],
    pub forward: [f32; 3],
    pub _pad1: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SimParams {
    dt_seconds: f32,
    bounds: f32,
    min_altitude: f32,
    max_altitude: f32,
    mass: f32,
    gravity: f32,
    thrust_scale: f32,
    arm_length: f32,
    yaw_torque_scale: f32,
    linear_drag: f32,
    angular_drag: f32,
    motor_response: f32,
    drone_radius: f32,
    gate_frame_thickness: f32,
    gate_depth_half: f32,
    collision_penalty: f32,
    env_count: u32,
    max_steps: u32,
    max_gates_per_env: u32,
    max_obstacles_per_env: u32,
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
        let max_obstacles_per_env = max_obstacles_per_env(config.max_gates_per_env);
        let gate_capacity = config.env_count * primitive_capacity_per_env(config.max_gates_per_env);
        let layout_values = build_layout_headers(
            config.env_count,
            config.max_gates_per_env,
            max_obstacles_per_env,
            compiled_course.header.total_gate_count,
        );

        let reset_values = vec![1u32; config.env_count];
        let reset_params_values: Vec<ResetParams> = (0..config.env_count)
            .map(|env_index| {
                let env_index = env_index as u32;
                ResetParams::new(env_index, env_index % 4, 0.35, 1)
            })
            .collect();
        let params = SimParams {
            dt_seconds: config.dt_seconds,
            bounds: config.bounds,
            min_altitude: 0.1,
            max_altitude: config.bounds,
            mass: 1.0,
            gravity: 9.81,
            thrust_scale: 4.2,
            arm_length: 2.4,
            yaw_torque_scale: 0.45,
            linear_drag: 0.18,
            angular_drag: 0.3,
            motor_response: 8.0,
            drone_radius: 0.12,
            gate_frame_thickness: 0.08,
            gate_depth_half: 0.04,
            collision_penalty: 6.0,
            env_count: config.env_count as u32,
            max_steps: config.max_steps,
            max_gates_per_env: config.max_gates_per_env as u32,
            max_obstacles_per_env: max_obstacles_per_env as u32,
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
                layout_headers.handle(),
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
            gates.handle(),
            course_header.handle(),
            layout_headers.handle(),
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
            max_obstacles_per_env(self.config.max_gates_per_env),
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
        self.config.env_count * primitive_capacity_per_env(self.config.max_gates_per_env)
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
    layout_headers: triad_gpu::Handle<wgpu::Buffer>,
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
        .read_write(layout_headers)
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
    max_obstacles_per_env: usize,
    gate_count: u32,
) -> Vec<EnvLayoutHeader> {
    (0..env_count)
        .map(|env_index| EnvLayoutHeader {
            gate_offset: (env_index * max_gates_per_env) as u32,
            gate_count,
            obstacle_offset: (env_count * max_gates_per_env + env_index * max_obstacles_per_env)
                as u32,
            obstacle_count: 0,
        })
        .collect()
}

fn max_obstacles_per_env(max_gates_per_env: usize) -> usize {
    max_gates_per_env.saturating_div(2).max(1)
}

fn primitive_capacity_per_env(max_gates_per_env: usize) -> usize {
    max_gates_per_env + max_obstacles_per_env(max_gates_per_env)
}
