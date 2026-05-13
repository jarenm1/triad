use std::collections::VecDeque;

use glam::Vec3;
use triad_sim::{Action, CourseSpec, EnvState, Gate};

const FLOOR_ALTITUDE: f32 = 0.1;
const FLOOR_HALF_THICKNESS: f32 = 0.02;
const FLOOR_INSTANCE_COUNT: usize = 1;
const GATE_DEPTH_HALF: f32 = 0.04;
const GATE_FRAME_THICKNESS: f32 = 0.08;
const DRONE_CORE_HALF_EXTENTS: [f32; 3] = [0.06, 0.025, 0.035];
const DRONE_ARM_HALF_EXTENTS: [f32; 3] = [0.14, 0.015, 0.015];
const DRONE_MOTOR_HALF_EXTENTS: [f32; 3] = [0.028, 0.018, 0.028];
const DRONE_ARM_OFFSET: f32 = 0.11;
const DRONE_MOTOR_OFFSET: f32 = 0.22;
const DRONE_MODEL_INSTANCE_COUNT: usize = 9;
const TARGET_HALF_EXTENTS: [f32; 3] = [0.05, 0.05, 0.05];
pub(crate) const TRAIL_MAX_POINTS: usize = 96;
pub(crate) const TRAIL_INSTANCE_COUNT: usize = TRAIL_MAX_POINTS - 1;
pub(crate) const TRAIL_MIN_POINT_DISTANCE: f32 = 0.04;
pub(crate) const TRAIL_RESET_DISTANCE: f32 = 2.5;
const TRAIL_HALF_WIDTH: f32 = 0.012;
const TRAIL_HALF_HEIGHT: f32 = 0.006;
const DEBUG_VECTOR_INSTANCE_COUNT: usize = 3;
const DEBUG_VECTOR_HALF_WIDTH: f32 = 0.014;
const DEBUG_VECTOR_HALF_HEIGHT: f32 = 0.007;
const FORWARD_VECTOR_LENGTH: f32 = 0.55;
const THRUST_VECTOR_LENGTH: f32 = 0.55;
const VELOCITY_VECTOR_SCALE: f32 = 0.22;
const VELOCITY_VECTOR_MAX_LENGTH: f32 = 0.9;

pub(crate) const VISUALIZER_SHADER: &str = include_str!("../shaders/shader.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct RenderInstance {
    center: [f32; 4],
    axis_x: [f32; 4],
    axis_y: [f32; 4],
    axis_z: [f32; 4],
    half_extents: [f32; 4],
    color: [f32; 4],
}

impl RenderInstance {
    pub(crate) fn hidden() -> Self {
        Self::oriented_box(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        )
    }

    pub(crate) fn oriented_box(
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

pub(crate) fn visible_instance_capacity(max_gates_per_env: usize) -> usize {
    FLOOR_INSTANCE_COUNT
        + max_gates_per_env * 4
        + max_obstacles_per_env(max_gates_per_env)
        + TRAIL_INSTANCE_COUNT
        + DRONE_MODEL_INSTANCE_COUNT
        + DEBUG_VECTOR_INSTANCE_COUNT
        + 1
}

pub(crate) fn required_gate_capacity(course: &CourseSpec) -> usize {
    course.total_gate_count().max(1) as usize
}

pub(crate) fn gate_bar_instances(gate: Gate) -> [RenderInstance; 4] {
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

pub(crate) fn floor_instance(bounds: f32) -> RenderInstance {
    RenderInstance::oriented_box(
        [0.0, FLOOR_ALTITUDE - FLOOR_HALF_THICKNESS, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [bounds, FLOOR_HALF_THICKNESS, bounds],
        [0.18, 0.19, 0.23, 1.0],
    )
}

pub(crate) fn obstacle_instance(obstacle: Gate) -> RenderInstance {
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

pub(crate) fn drone_instances(state: EnvState) -> Vec<RenderInstance> {
    let (forward, right, up) = drone_basis(state);
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

pub(crate) fn trail_instances(points: &VecDeque<[f32; 3]>) -> Vec<RenderInstance> {
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

pub(crate) fn debug_vector_instances(state: EnvState) -> Vec<RenderInstance> {
    let (forward, _, up) = drone_basis(state);
    let center = vec3_from_array(state.position);
    let velocity = vec3_from_array(state.velocity);
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
    if velocity_length > 1e-5 {
        let scaled_velocity =
            velocity * (VELOCITY_VECTOR_SCALE.min(VELOCITY_VECTOR_MAX_LENGTH / velocity_length));
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

pub(crate) fn target_instance(gate: Gate) -> RenderInstance {
    RenderInstance::oriented_box(
        [gate.center[0], gate.center[1], gate.center[2]],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        TARGET_HALF_EXTENTS,
        [0.95, 0.2, 0.3, 1.0],
    )
}

pub(crate) fn autopilot_action(state: EnvState, target_gate: Gate) -> Action {
    let delta_x = target_gate.center[0] - state.position[0];
    let delta_y = target_gate.center[1] - state.position[1];
    let delta_z = target_gate.center[2] - state.position[2];
    let yaw = state.attitude[2];
    let heading = [yaw.cos(), yaw.sin()];
    let right = [heading[1], -heading[0]];
    let horizontal_delta = [delta_x, delta_z];
    let lateral_error = horizontal_delta[0] * right[0] + horizontal_delta[1] * right[1];
    let horizontal_distance = (horizontal_delta[0] * horizontal_delta[0]
        + horizontal_delta[1] * horizontal_delta[1])
        .sqrt();
    let safe_distance = horizontal_distance.max(1.0e-6);
    let approach_dir = [
        horizontal_delta[0] / safe_distance,
        horizontal_delta[1] / safe_distance,
    ];
    let gate_forward_length = (target_gate.forward[0] * target_gate.forward[0]
        + target_gate.forward[2] * target_gate.forward[2])
        .sqrt()
        .max(1.0e-6);
    let gate_forward = [
        target_gate.forward[0] / gate_forward_length,
        target_gate.forward[2] / gate_forward_length,
    ];
    let gate_align_weight = (1.0 - horizontal_distance / 6.0).clamp(0.0, 1.0);
    let desired_dir_raw = [
        approach_dir[0] * (1.0 - gate_align_weight) + gate_forward[0] * gate_align_weight,
        approach_dir[1] * (1.0 - gate_align_weight) + gate_forward[1] * gate_align_weight,
    ];
    let desired_dir_length = (desired_dir_raw[0] * desired_dir_raw[0]
        + desired_dir_raw[1] * desired_dir_raw[1])
        .sqrt()
        .max(1.0e-6);
    let desired_dir = [
        desired_dir_raw[0] / desired_dir_length,
        desired_dir_raw[1] / desired_dir_length,
    ];
    let desired_yaw = desired_dir[1].atan2(desired_dir[0]);
    let yaw_error = ((desired_yaw - yaw) + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
        - std::f32::consts::PI;
    let gate_right = [gate_forward[1], -gate_forward[0]];
    let cross_track_error =
        (horizontal_delta[0] * gate_right[0] + horizontal_delta[1] * gate_right[1]).abs();
    let yaw_scale = 1.0 - ((yaw_error.abs() - 0.16) / 0.32).clamp(0.0, 1.0);
    let cross_track_scale = 1.0 - ((cross_track_error - 0.08) / 0.32).clamp(0.0, 1.0);
    let approach_scale = (yaw_scale * yaw_scale * cross_track_scale)
        / (1.0 + 0.12 * lateral_error.abs() + 0.1 * delta_y.abs());

    let forward_velocity_target =
        ((horizontal_distance - 0.6) * 1.2).clamp(0.0, 2.6) * approach_scale;
    let lateral_velocity_target = (lateral_error * 1.2).clamp(-3.0, 3.0);
    let vertical_velocity_target = (delta_y * 1.15).clamp(-2.0, 2.0);
    let yaw_rate_target = (yaw_error * 2.2).clamp(-3.5, 3.5);

    Action::new([
        (0.5 + forward_velocity_target / 12.0).clamp(0.0, 1.0),
        (0.5 + lateral_velocity_target / 8.0).clamp(0.0, 1.0),
        (0.5 + vertical_velocity_target / 5.0).clamp(0.0, 1.0),
        (0.5 + yaw_rate_target / 7.0).clamp(0.0, 1.0),
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

fn drone_basis(state: EnvState) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let yaw = state.attitude[2];
    let (roll, pitch) = (state.attitude[0], state.attitude[1]);
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
    if length <= 1e-5 {
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
    if value.length_squared() <= 1e-6 {
        fallback.normalize_or_zero()
    } else {
        value.normalize()
    }
}

fn vec3_from_array(value: [f32; 3]) -> Vec3 {
    Vec3::new(value[0], value[1], value[2])
}

fn vec3_to_array(value: Vec3) -> [f32; 3] {
    [value.x, value.y, value.z]
}

fn max_obstacles_per_env(max_gates_per_env: usize) -> usize {
    max_gates_per_env.saturating_div(2).max(1)
}
