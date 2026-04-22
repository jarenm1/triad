from __future__ import annotations

import argparse
import ctypes
import json
import os
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from functools import lru_cache
from pathlib import Path
from typing import Sequence


def _library_name() -> str:
    if os.name == "nt":
        return "triad_py.dll"
    if os.uname().sysname == "Darwin":
        return "libtriad_py.dylib"
    return "libtriad_py.so"


def _default_library_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "target" / "debug" / _library_name()


def _load_library() -> ctypes.CDLL:
    explicit = os.environ.get("TRIAD_PY_LIB")
    path = Path(explicit) if explicit else _default_library_path()
    return ctypes.CDLL(str(path))


_lib = _load_library()

__all__ = [
    "ACTION_STRIDE",
    "OBSERVATION_STRIDE",
    "apply_curriculum_progress",
    "apply_curriculum_update",
    "BenchmarkResult",
    "CourseSpec",
    "CurriculumPhase",
    "CurriculumProgression",
    "CurriculumSchedule",
    "DoneReason",
    "PackedStepResult",
    "PPOConfig",
    "PPOTrainResult",
    "RolloutCollector",
    "RolloutBatch",
    "CurriculumStage",
    "SimulationConfig",
    "SimulationCore",
    "StageKind",
    "StageSpec",
    "TriadFastVecEnv",
    "TriadVecEnv",
    "TurnDirection",
    "build_basic_lap_course",
    "build_teacher_curriculum_progression",
    "build_teacher_curriculum_schedule",
    "benchmark_rollout",
    "collect_rollout_numpy",
    "main",
    "point_to_gate_policy",
    "sample_curriculum_reset_params",
    "serve_ppo_policy",
    "train_ppo",
    "load_ppo_policy",
    "zero_policy",
]


def _last_error_message() -> str:
    required = _lib.triad_last_error_message(None, 0)
    if required <= 1:
        return "triad binding call failed"
    buffer = ctypes.create_string_buffer(required)
    _lib.triad_last_error_message(buffer, len(buffer))
    return buffer.value.decode("utf-8") or "triad binding call failed"


def _require(success: bool) -> None:
    if not success:
        raise RuntimeError(_last_error_message())


def _motor_action(action: Sequence[float]) -> _TriadAction:
    if len(action) != ACTION_STRIDE:
        raise ValueError(
            f"expected {ACTION_STRIDE} motor commands per action, got {len(action)}"
        )
    return _TriadAction(
        motor_0=float(action[0]),
        motor_1=float(action[1]),
        motor_2=float(action[2]),
        motor_3=float(action[3]),
    )


@lru_cache(maxsize=1)
def _numpy_module():
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("NumPy is required for zero-copy array views") from error
    return np


class StageKind(IntEnum):
    INTRO = _lib.triad_stage_kind_intro()
    STRAIGHT = _lib.triad_stage_kind_straight()
    OFFSET = _lib.triad_stage_kind_offset()
    TURN90 = _lib.triad_stage_kind_turn90()


class TurnDirection(IntEnum):
    LEFT = _lib.triad_turn_direction_left()
    RIGHT = _lib.triad_turn_direction_right()


class CurriculumStage(IntEnum):
    INTRO = _lib.triad_curriculum_stage_intro()
    ARENA = _lib.triad_curriculum_stage_arena()
    TECHNICAL = _lib.triad_curriculum_stage_technical()
    ELEVATED = _lib.triad_curriculum_stage_elevated()


class DoneReason(IntFlag):
    NONE = 0
    COMPLETE = 1 << 0
    GATE_COLLISION = 1 << 1
    OBSTACLE_COLLISION = 1 << 2
    FLOOR_COLLISION = 1 << 3
    OUT_OF_BOUNDS = 1 << 4
    STEP_LIMIT = 1 << 5
    EXCESSIVE_TILT = 1 << 6


class _TriadStageDesc(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_uint32),
        ("gate_count", ctypes.c_uint32),
        ("spacing", ctypes.c_float),
        ("lateral_amp", ctypes.c_float),
        ("turn_degrees", ctypes.c_float),
        ("radius", ctypes.c_float),
        ("vertical_amp", ctypes.c_float),
        ("hole_half_width", ctypes.c_float),
        ("hole_half_height", ctypes.c_float),
        ("direction", ctypes.c_uint32),
    ]


class _TriadCourseStats(ctypes.Structure):
    _fields_ = [
        ("stage_count", ctypes.c_uint32),
        ("total_gate_count", ctypes.c_uint32),
        ("loop_enabled", ctypes.c_uint32),
        ("laps_required", ctypes.c_uint32),
    ]


class _TriadSimConfig(ctypes.Structure):
    _fields_ = [
        ("env_count", ctypes.c_uint32),
        ("dt_seconds", ctypes.c_float),
        ("bounds", ctypes.c_float),
        ("max_steps", ctypes.c_uint32),
        ("max_gates_per_env", ctypes.c_uint32),
        ("laps_required", ctypes.c_uint32),
    ]


class _TriadAction(ctypes.Structure):
    _fields_ = [
        ("motor_0", ctypes.c_float),
        ("motor_1", ctypes.c_float),
        ("motor_2", ctypes.c_float),
        ("motor_3", ctypes.c_float),
    ]


class _TriadResetParams(ctypes.Structure):
    _fields_ = [
        ("seed", ctypes.c_uint32),
        ("grammar_id", ctypes.c_uint32),
        ("difficulty", ctypes.c_float),
        ("curriculum_stage", ctypes.c_uint32),
    ]


class _TriadEnvState(ctypes.Structure):
    _fields_ = [
        ("position_x", ctypes.c_float),
        ("position_y", ctypes.c_float),
        ("position_z", ctypes.c_float),
        ("velocity_x", ctypes.c_float),
        ("velocity_y", ctypes.c_float),
        ("velocity_z", ctypes.c_float),
        ("roll", ctypes.c_float),
        ("pitch", ctypes.c_float),
        ("yaw", ctypes.c_float),
        ("angular_velocity_x", ctypes.c_float),
        ("angular_velocity_y", ctypes.c_float),
        ("angular_velocity_z", ctypes.c_float),
        ("motor_0", ctypes.c_float),
        ("motor_1", ctypes.c_float),
        ("motor_2", ctypes.c_float),
        ("motor_3", ctypes.c_float),
        ("step_count", ctypes.c_uint32),
        ("done", ctypes.c_uint32),
        ("current_gate", ctypes.c_uint32),
        ("current_lap", ctypes.c_uint32),
    ]


class _TriadObservation(ctypes.Structure):
    _fields_ = [
        ("position_x", ctypes.c_float),
        ("position_y", ctypes.c_float),
        ("position_z", ctypes.c_float),
        ("velocity_x", ctypes.c_float),
        ("velocity_y", ctypes.c_float),
        ("velocity_z", ctypes.c_float),
        ("roll", ctypes.c_float),
        ("pitch", ctypes.c_float),
        ("yaw", ctypes.c_float),
        ("angular_velocity_x", ctypes.c_float),
        ("angular_velocity_y", ctypes.c_float),
        ("angular_velocity_z", ctypes.c_float),
        ("target_gate_x", ctypes.c_float),
        ("target_gate_y", ctypes.c_float),
        ("target_gate_z", ctypes.c_float),
        ("target_gate_forward_x", ctypes.c_float),
        ("target_gate_forward_y", ctypes.c_float),
        ("target_gate_forward_z", ctypes.c_float),
        ("progress", ctypes.c_float),
        ("distance_to_gate", ctypes.c_float),
        ("gate_alignment", ctypes.c_float),
        ("mean_motor_thrust", ctypes.c_float),
    ]


class _TriadRewardDone(ctypes.Structure):
    _fields_ = [
        ("reward", ctypes.c_float),
        ("done", ctypes.c_uint32),
        ("done_reason", ctypes.c_uint32),
        ("_pad0", ctypes.c_uint32),
        ("progress_reward", ctypes.c_float),
        ("distance_penalty", ctypes.c_float),
        ("alignment_reward", ctypes.c_float),
        ("tilt_penalty", ctypes.c_float),
        ("completion_bonus", ctypes.c_float),
        ("collision_penalty", ctypes.c_float),
        ("_pad1", ctypes.c_float),
        ("_pad2", ctypes.c_float),
    ]


_lib.triad_last_error_message.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_lib.triad_last_error_message.restype = ctypes.c_size_t

_lib.triad_course_create.restype = ctypes.c_void_p
_lib.triad_course_create_default_drone.restype = ctypes.c_void_p
_lib.triad_course_destroy.argtypes = [ctypes.c_void_p]
_lib.triad_course_set_name.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.triad_course_set_name.restype = ctypes.c_bool
_lib.triad_course_set_loop_enabled.argtypes = [ctypes.c_void_p, ctypes.c_bool]
_lib.triad_course_set_loop_enabled.restype = ctypes.c_bool
_lib.triad_course_set_laps_required.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
_lib.triad_course_set_laps_required.restype = ctypes.c_bool
_lib.triad_course_clear_stages.argtypes = [ctypes.c_void_p]
_lib.triad_course_clear_stages.restype = ctypes.c_bool
_lib.triad_course_add_stage.argtypes = [ctypes.c_void_p, _TriadStageDesc]
_lib.triad_course_add_stage.restype = ctypes.c_bool
_lib.triad_course_get_stats.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(_TriadCourseStats),
]
_lib.triad_course_get_stats.restype = ctypes.c_bool
_lib.triad_sim_config_default.restype = _TriadSimConfig
_lib.triad_action_stride.restype = ctypes.c_size_t
_lib.triad_observation_stride.restype = ctypes.c_size_t
_lib.triad_curriculum_stage_intro.argtypes = []
_lib.triad_curriculum_stage_intro.restype = ctypes.c_uint32
_lib.triad_curriculum_stage_arena.argtypes = []
_lib.triad_curriculum_stage_arena.restype = ctypes.c_uint32
_lib.triad_curriculum_stage_technical.argtypes = []
_lib.triad_curriculum_stage_technical.restype = ctypes.c_uint32
_lib.triad_curriculum_stage_elevated.argtypes = []
_lib.triad_curriculum_stage_elevated.restype = ctypes.c_uint32

_lib.triad_simulation_create.argtypes = [_TriadSimConfig]
_lib.triad_simulation_create.restype = ctypes.c_void_p
_lib.triad_simulation_destroy.argtypes = [ctypes.c_void_p]
_lib.triad_simulation_env_count.argtypes = [ctypes.c_void_p]
_lib.triad_simulation_env_count.restype = ctypes.c_uint32
_lib.triad_simulation_set_course.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_lib.triad_simulation_set_course.restype = ctypes.c_bool
_lib.triad_simulation_set_actions.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(_TriadAction),
    ctypes.c_size_t,
]
_lib.triad_simulation_set_actions.restype = ctypes.c_bool
_lib.triad_simulation_set_reset_params.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(_TriadResetParams),
    ctypes.c_size_t,
]
_lib.triad_simulation_set_reset_params.restype = ctypes.c_bool
_lib.triad_simulation_reset_all.argtypes = [ctypes.c_void_p]
_lib.triad_simulation_reset_all.restype = ctypes.c_bool
_lib.triad_simulation_request_resets.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_size_t,
]
_lib.triad_simulation_request_resets.restype = ctypes.c_bool
_lib.triad_simulation_step.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_lib.triad_simulation_step.restype = ctypes.c_bool
_lib.triad_simulation_readback_state.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(_TriadEnvState),
    ctypes.c_size_t,
]
_lib.triad_simulation_readback_state.restype = ctypes.c_bool
_lib.triad_simulation_readback_observations.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(_TriadObservation),
    ctypes.c_size_t,
]
_lib.triad_simulation_readback_observations.restype = ctypes.c_bool
_lib.triad_simulation_readback_observations_flat.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]
_lib.triad_simulation_readback_observations_flat.restype = ctypes.c_bool
_lib.triad_simulation_readback_reward_done.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(_TriadRewardDone),
    ctypes.c_size_t,
]
_lib.triad_simulation_readback_reward_done.restype = ctypes.c_bool
_lib.triad_simulation_readback_reward_done_flat.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
]
_lib.triad_simulation_readback_reward_done_flat.restype = ctypes.c_bool
_lib.triad_simulation_step_actions_readback.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(_TriadAction),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
]
_lib.triad_simulation_step_actions_readback.restype = ctypes.c_bool
_lib.triad_simulation_step_flat_actions_readback.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
]
_lib.triad_simulation_step_flat_actions_readback.restype = ctypes.c_bool

ACTION_STRIDE = int(_lib.triad_action_stride())
OBSERVATION_STRIDE = int(_lib.triad_observation_stride())

from .curriculum import (  # noqa: E402
    CurriculumPhase,
    CurriculumProgression,
    CurriculumSchedule,
    build_teacher_curriculum_progression,
    build_teacher_curriculum_schedule,
    sample_curriculum_reset_params,
)


@dataclass
class StageSpec:
    kind: StageKind
    gate_count: int
    spacing: float = 1.5
    lateral_amp: float = 0.0
    turn_degrees: float = 0.0
    radius: float = 0.0
    vertical_amp: float = 0.0
    hole_half_width: float = 0.14
    hole_half_height: float = 0.14
    direction: TurnDirection = TurnDirection.LEFT

    @classmethod
    def intro(cls, gate_count: int, spacing: float) -> "StageSpec":
        return cls(kind=StageKind.INTRO, gate_count=gate_count, spacing=spacing)

    @classmethod
    def straight(cls, gate_count: int, spacing: float) -> "StageSpec":
        return cls(kind=StageKind.STRAIGHT, gate_count=gate_count, spacing=spacing)

    @classmethod
    def offset(cls, gate_count: int, spacing: float, lateral_amp: float) -> "StageSpec":
        return cls(
            kind=StageKind.OFFSET,
            gate_count=gate_count,
            spacing=spacing,
            lateral_amp=lateral_amp,
        )

    @classmethod
    def turn90(
        cls, gate_count: int, radius: float, direction: TurnDirection
    ) -> "StageSpec":
        return cls(
            kind=StageKind.TURN90,
            gate_count=gate_count,
            turn_degrees=90.0,
            radius=radius,
            direction=direction,
        )

    def as_ffi(self) -> _TriadStageDesc:
        return _TriadStageDesc(
            kind=int(self.kind),
            gate_count=self.gate_count,
            spacing=self.spacing,
            lateral_amp=self.lateral_amp,
            turn_degrees=self.turn_degrees,
            radius=self.radius,
            vertical_amp=self.vertical_amp,
            hole_half_width=self.hole_half_width,
            hole_half_height=self.hole_half_height,
            direction=int(self.direction),
        )


@dataclass
class SimulationConfig:
    env_count: int
    dt_seconds: float
    bounds: float
    max_steps: int
    max_gates_per_env: int
    laps_required: int

    @classmethod
    def default(cls) -> "SimulationConfig":
        config = _lib.triad_sim_config_default()
        return cls(
            env_count=config.env_count,
            dt_seconds=config.dt_seconds,
            bounds=config.bounds,
            max_steps=config.max_steps,
            max_gates_per_env=config.max_gates_per_env,
            laps_required=config.laps_required,
        )

    def as_ffi(self) -> _TriadSimConfig:
        return _TriadSimConfig(
            env_count=self.env_count,
            dt_seconds=self.dt_seconds,
            bounds=self.bounds,
            max_steps=self.max_steps,
            max_gates_per_env=self.max_gates_per_env,
            laps_required=self.laps_required,
        )


class CourseSpec:
    def __init__(self, name: str | None = "course", _handle: int | None = None) -> None:
        self._handle = _handle or _lib.triad_course_create()
        if not self._handle:
            raise RuntimeError(_last_error_message())
        if name is not None:
            self.set_name(name)

    @classmethod
    def default_drone_course(cls) -> "CourseSpec":
        handle = _lib.triad_course_create_default_drone()
        if not handle:
            raise RuntimeError(_last_error_message())
        return cls(name=None, _handle=handle)

    def close(self) -> None:
        if self._handle:
            _lib.triad_course_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def set_name(self, name: str) -> "CourseSpec":
        _require(_lib.triad_course_set_name(self._handle, name.encode("utf-8")))
        return self

    def set_loop_enabled(self, enabled: bool) -> "CourseSpec":
        _require(_lib.triad_course_set_loop_enabled(self._handle, enabled))
        return self

    def set_laps_required(self, laps_required: int) -> "CourseSpec":
        _require(_lib.triad_course_set_laps_required(self._handle, laps_required))
        return self

    def clear_stages(self) -> "CourseSpec":
        _require(_lib.triad_course_clear_stages(self._handle))
        return self

    def add_stage(self, stage: StageSpec) -> "CourseSpec":
        _require(_lib.triad_course_add_stage(self._handle, stage.as_ffi()))
        return self

    @property
    def stats(self) -> dict[str, int | bool]:
        stats = _TriadCourseStats()
        _require(_lib.triad_course_get_stats(self._handle, ctypes.byref(stats)))
        return {
            "stage_count": int(stats.stage_count),
            "total_gate_count": int(stats.total_gate_count),
            "loop_enabled": bool(stats.loop_enabled),
            "laps_required": int(stats.laps_required),
        }


@dataclass
class PackedStepResult:
    observations: ctypes.Array[ctypes.c_float]
    rewards: ctypes.Array[ctypes.c_float]
    dones: ctypes.Array[ctypes.c_uint8]
    env_count: int
    observation_stride: int = OBSERVATION_STRIDE

    def observation_rows(self) -> list[list[float]]:
        return [
            [
                float(self.observations[(env_index * self.observation_stride) + offset])
                for offset in range(self.observation_stride)
            ]
            for env_index in range(self.env_count)
        ]

    def reward_list(self) -> list[float]:
        return [float(self.rewards[index]) for index in range(self.env_count)]

    def done_list(self) -> list[bool]:
        return [bool(self.dones[index]) for index in range(self.env_count)]

    def observation_buffer(self) -> memoryview:
        return memoryview(self.observations)

    def reward_buffer(self) -> memoryview:
        return memoryview(self.rewards)

    def done_buffer(self) -> memoryview:
        return memoryview(self.dones)

    def action_buffer(self) -> memoryview:
        raise RuntimeError("PackedStepResult does not own an action buffer")

    def numpy_views(self) -> tuple[object, object, object]:
        np = _numpy_module()
        observations = np.ctypeslib.as_array(self.observations).reshape(
            self.env_count, self.observation_stride
        )
        rewards = np.ctypeslib.as_array(self.rewards)
        dones = np.ctypeslib.as_array(self.dones)
        return observations, rewards, dones

    def numpy_bool_dones(self) -> object:
        np = _numpy_module()
        return np.ctypeslib.as_array(self.dones).astype(bool)

    def to_numpy(self) -> tuple[object, object, object]:
        observations, rewards, _ = self.numpy_views()
        dones = self.numpy_bool_dones()
        return observations, rewards, dones


class SimulationCore:
    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig.default()
        self._handle = _lib.triad_simulation_create(self.config.as_ffi())
        if not self._handle:
            raise RuntimeError(_last_error_message())

    def close(self) -> None:
        if self._handle:
            _lib.triad_simulation_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()

    @property
    def env_count(self) -> int:
        return int(_lib.triad_simulation_env_count(self._handle))

    @property
    def observation_stride(self) -> int:
        return OBSERVATION_STRIDE

    @property
    def action_stride(self) -> int:
        return ACTION_STRIDE

    @property
    def flat_action_count(self) -> int:
        return self.env_count * self.action_stride

    @property
    def flat_observation_count(self) -> int:
        return self.env_count * self.observation_stride

    def create_action_buffer(self) -> ctypes.Array[ctypes.c_float]:
        return (ctypes.c_float * self.flat_action_count)()

    def _allocate_flat_observation_buffer(self) -> ctypes.Array[ctypes.c_float]:
        return (ctypes.c_float * self.flat_observation_count)()

    def _allocate_reward_buffer(self) -> ctypes.Array[ctypes.c_float]:
        return (ctypes.c_float * self.env_count)()

    def _allocate_done_buffer(self) -> ctypes.Array[ctypes.c_uint8]:
        return (ctypes.c_uint8 * self.env_count)()

    def set_course(self, course: CourseSpec) -> "SimulationCore":
        _require(_lib.triad_simulation_set_course(self._handle, course._handle))
        return self

    def set_actions(self, actions: Sequence[Sequence[float]]) -> "SimulationCore":
        ffi_actions = (_TriadAction * len(actions))(
            *(_motor_action(action) for action in actions)
        )
        _require(
            _lib.triad_simulation_set_actions(self._handle, ffi_actions, len(actions))
        )
        return self

    def set_reset_params(
        self,
        reset_params: Sequence[tuple[int, int, float] | tuple[int, int, float, int]],
    ) -> "SimulationCore":
        ffi_values = []
        for params in reset_params:
            if len(params) == 3:
                seed, grammar_id, difficulty = params
                curriculum_stage = int(CurriculumStage.ARENA)
            elif len(params) == 4:
                seed, grammar_id, difficulty, curriculum_stage = params
            else:
                raise ValueError(
                    "reset params must be (seed, grammar_id, difficulty) or "
                    "(seed, grammar_id, difficulty, curriculum_stage)"
                )
            ffi_values.append(
                _TriadResetParams(
                    seed=seed,
                    grammar_id=grammar_id,
                    difficulty=difficulty,
                    curriculum_stage=curriculum_stage,
                )
            )
        ffi_params = (_TriadResetParams * len(reset_params))(*ffi_values)
        _require(
            _lib.triad_simulation_set_reset_params(
                self._handle, ffi_params, len(reset_params)
            )
        )
        return self

    def reset_all(self) -> "SimulationCore":
        _require(_lib.triad_simulation_reset_all(self._handle))
        return self

    def request_resets(self, env_indices: Sequence[int]) -> "SimulationCore":
        ffi_indices = (ctypes.c_uint32 * len(env_indices))(*env_indices)
        _require(
            _lib.triad_simulation_request_resets(
                self._handle, ffi_indices, len(env_indices)
            )
        )
        return self

    def step(self, steps: int = 1) -> "SimulationCore":
        _require(_lib.triad_simulation_step(self._handle, steps))
        return self

    def read_observation_buffer(self) -> ctypes.Array[ctypes.c_float]:
        values = self._allocate_flat_observation_buffer()
        _require(
            _lib.triad_simulation_readback_observations_flat(
                self._handle, values, len(values)
            )
        )
        return values

    def read_reward_done_buffers(
        self,
    ) -> tuple[ctypes.Array[ctypes.c_float], ctypes.Array[ctypes.c_uint8]]:
        rewards = self._allocate_reward_buffer()
        dones = self._allocate_done_buffer()
        _require(
            _lib.triad_simulation_readback_reward_done_flat(
                self._handle,
                rewards,
                len(rewards),
                dones,
                len(dones),
            )
        )
        return rewards, dones

    def step_actions(
        self, actions: Sequence[Sequence[float]], steps: int = 1
    ) -> PackedStepResult:
        ffi_actions = (_TriadAction * len(actions))(
            *(_motor_action(action) for action in actions)
        )
        observations = self._allocate_flat_observation_buffer()
        rewards = self._allocate_reward_buffer()
        dones = self._allocate_done_buffer()
        _require(
            _lib.triad_simulation_step_actions_readback(
                self._handle,
                ffi_actions,
                len(actions),
                steps,
                observations,
                len(observations),
                rewards,
                len(rewards),
                dones,
                len(dones),
            )
        )
        return PackedStepResult(
            observations=observations,
            rewards=rewards,
            dones=dones,
            env_count=self.env_count,
        )

    def step_flat_actions_into(
        self,
        action_values: ctypes.Array[ctypes.c_float],
        result: PackedStepResult,
        steps: int = 1,
    ) -> PackedStepResult:
        _require(
            _lib.triad_simulation_step_flat_actions_readback(
                self._handle,
                action_values,
                len(action_values),
                steps,
                result.observations,
                len(result.observations),
                result.rewards,
                len(result.rewards),
                result.dones,
                len(result.dones),
            )
        )
        return result

    def get_state(self) -> list[dict[str, int | float]]:
        values = (_TriadEnvState * self.env_count)()
        _require(
            _lib.triad_simulation_readback_state(self._handle, values, len(values))
        )
        return [
            {
                "position": [
                    float(value.position_x),
                    float(value.position_y),
                    float(value.position_z),
                ],
                "velocity": [
                    float(value.velocity_x),
                    float(value.velocity_y),
                    float(value.velocity_z),
                ],
                "attitude": [float(value.roll), float(value.pitch), float(value.yaw)],
                "angular_velocity": [
                    float(value.angular_velocity_x),
                    float(value.angular_velocity_y),
                    float(value.angular_velocity_z),
                ],
                "motor_thrust": [
                    float(value.motor_0),
                    float(value.motor_1),
                    float(value.motor_2),
                    float(value.motor_3),
                ],
                "step_count": int(value.step_count),
                "done": bool(value.done),
                "current_gate": int(value.current_gate),
                "current_lap": int(value.current_lap),
            }
            for value in values
        ]

    def get_observations(self) -> list[dict[str, int | float]]:
        values = (_TriadObservation * self.env_count)()
        _require(
            _lib.triad_simulation_readback_observations(
                self._handle, values, len(values)
            )
        )
        return [
            {
                "position": [
                    float(value.position_x),
                    float(value.position_y),
                    float(value.position_z),
                ],
                "velocity": [
                    float(value.velocity_x),
                    float(value.velocity_y),
                    float(value.velocity_z),
                ],
                "attitude": [float(value.roll), float(value.pitch), float(value.yaw)],
                "angular_velocity": [
                    float(value.angular_velocity_x),
                    float(value.angular_velocity_y),
                    float(value.angular_velocity_z),
                ],
                "target_gate_position": [
                    float(value.target_gate_x),
                    float(value.target_gate_y),
                    float(value.target_gate_z),
                ],
                "target_gate_forward": [
                    float(value.target_gate_forward_x),
                    float(value.target_gate_forward_y),
                    float(value.target_gate_forward_z),
                ],
                "progress": float(value.progress),
                "distance_to_gate": float(value.distance_to_gate),
                "gate_alignment": float(value.gate_alignment),
                "mean_motor_thrust": float(value.mean_motor_thrust),
            }
            for value in values
        ]

    def get_reward_done(self) -> list[dict[str, object]]:
        values = (_TriadRewardDone * self.env_count)()
        _require(
            _lib.triad_simulation_readback_reward_done(
                self._handle, values, len(values)
            )
        )
        return [
            {
                "reward": float(value.reward),
                "done": bool(value.done),
                "done_reason": int(value.done_reason),
                "done_reason_labels": [
                    reason.name
                    for reason in DoneReason
                    if reason is not DoneReason.NONE
                    and int(value.done_reason) & int(reason)
                ],
                "progress_reward": float(value.progress_reward),
                "distance_penalty": float(value.distance_penalty),
                "alignment_reward": float(value.alignment_reward),
                "tilt_penalty": float(value.tilt_penalty),
                "completion_bonus": float(value.completion_bonus),
                "collision_penalty": float(value.collision_penalty),
            }
            for value in values
        ]


def apply_curriculum_progress(
    sim: SimulationCore,
    progress: float,
    base_seed: int = 0,
    schedule: CurriculumSchedule | None = None,
) -> list[tuple[int, int, float, int]]:
    reset_params = sample_curriculum_reset_params(
        env_count=sim.env_count,
        progress=progress,
        base_seed=base_seed,
        schedule=schedule,
    )
    sim.set_reset_params(reset_params)
    return reset_params


def apply_curriculum_update(
    sim: SimulationCore,
    update_index: int,
    progression: CurriculumProgression,
    base_seed: int = 0,
    schedule: CurriculumSchedule | None = None,
) -> list[tuple[int, int, float, int]]:
    return progression.apply(
        sim=sim,
        update_index=update_index,
        base_seed=base_seed,
        schedule=schedule,
    )


class TriadVecEnv:
    def __init__(self, sim: SimulationCore, auto_reset: bool = True) -> None:
        self.sim = sim
        self.auto_reset = auto_reset

    def reset(self) -> list[dict[str, int | float]]:
        self.sim.reset_all().step(1)
        return self.sim.get_observations()

    def step(
        self, actions: Sequence[Sequence[float]]
    ) -> tuple[
        list[dict[str, int | float]],
        list[float],
        list[bool],
        list[dict[str, int]],
    ]:
        self.sim.set_actions(actions).step(1)
        observations = self.sim.get_observations()
        reward_done = self.sim.get_reward_done()
        rewards = [float(item["reward"]) for item in reward_done]
        dones = [bool(item["done"]) for item in reward_done]
        infos = [{"env_index": env_index} for env_index in range(len(reward_done))]

        if self.auto_reset:
            done_indices = [index for index, done in enumerate(dones) if done]
            if done_indices:
                self.sim.request_resets(done_indices).step(1)
                reset_observations = self.sim.get_observations()
                for index in done_indices:
                    observations[index] = reset_observations[index]

        return observations, rewards, dones, infos


class TriadFastVecEnv:
    def __init__(self, sim: SimulationCore, auto_reset: bool = True) -> None:
        self.sim = sim
        self.auto_reset = auto_reset
        self._action_values = self.sim.create_action_buffer()
        self._result = PackedStepResult(
            observations=self.sim._allocate_flat_observation_buffer(),
            rewards=self.sim._allocate_reward_buffer(),
            dones=self.sim._allocate_done_buffer(),
            env_count=self.sim.env_count,
        )

    @property
    def action_values(self) -> ctypes.Array[ctypes.c_float]:
        return self._action_values

    @property
    def result(self) -> PackedStepResult:
        return self._result

    def action_buffer(self) -> memoryview:
        return memoryview(self._action_values)

    def numpy_action_view(self) -> object:
        np = _numpy_module()
        return np.ctypeslib.as_array(self._action_values).reshape(
            self.sim.env_count, self.sim.action_stride
        )

    def numpy_result_views(self) -> tuple[object, object, object]:
        return self._result.numpy_views()

    def set_actions_flat(self, action_values: Sequence[float]) -> "TriadFastVecEnv":
        if len(action_values) != len(self._action_values):
            raise ValueError(
                f"expected {len(self._action_values)} flat action values, got {len(action_values)}"
            )
        self._action_values[:] = action_values
        return self

    def reset(self) -> ctypes.Array[ctypes.c_float]:
        self.sim.reset_all().step(1)
        reset_obs = self.sim.read_observation_buffer()
        self._result.observations[:] = reset_obs
        return self._result.observations

    def step(
        self, actions: Sequence[Sequence[float]] | Sequence[float]
    ) -> PackedStepResult:
        if len(actions) == len(self._action_values):
            self._action_values[:] = actions
        else:
            if len(actions) != self.sim.env_count:
                raise ValueError(
                    f"expected {self.sim.env_count} motor command rows or {len(self._action_values)} flat values, got {len(actions)}"
                )
            write_index = 0
            for action in actions:
                motor_action = _motor_action(action)
                self._action_values[write_index] = motor_action.motor_0
                self._action_values[write_index + 1] = motor_action.motor_1
                self._action_values[write_index + 2] = motor_action.motor_2
                self._action_values[write_index + 3] = motor_action.motor_3
                write_index += ACTION_STRIDE
        return self.step_in_place(1)

    def step_in_place(self, steps: int = 1) -> PackedStepResult:
        result = self.sim.step_flat_actions_into(
            self._action_values, self._result, steps
        )
        if self.auto_reset:
            done_indices = [index for index, done in enumerate(result.dones) if done]
            if done_indices:
                self.sim.request_resets(done_indices).step(1)
                reset_obs = self.sim.read_observation_buffer()
                for index in done_indices:
                    start = index * self.sim.observation_stride
                    end = start + self.sim.observation_stride
                    result.observations[start:end] = reset_obs[start:end]
        return result

    def rollout(self, steps: int) -> PackedStepResult:
        return self.step_in_place(steps)

    def reset_numpy(self) -> object:
        self.reset()
        observations, _, _ = self._result.numpy_views()
        return observations

    def step_numpy(
        self, actions: Sequence[tuple[float, float]]
    ) -> tuple[object, object, object]:
        result = self.step(actions)
        return result.numpy_views()

    def rollout_numpy(self, steps: int) -> tuple[object, object, object]:
        result = self.rollout(steps)
        return result.numpy_views()


from .ppo import PPOConfig, PPOTrainResult, load_ppo_policy, serve_ppo_policy, train_ppo
from .rollout import (
    BenchmarkResult,
    RolloutBatch,
    RolloutCollector,
    benchmark_rollout,
    collect_rollout_numpy,
    point_to_gate_policy,
    zero_policy,
)


def build_basic_lap_course() -> CourseSpec:
    return (
        CourseSpec("basic-lap")
        .set_loop_enabled(True)
        .set_laps_required(1)
        .add_stage(
            StageSpec(
                kind=StageKind.INTRO,
                gate_count=1,
                spacing=3.2,
                vertical_amp=0.08,
            )
        )
        .add_stage(
            StageSpec(
                kind=StageKind.STRAIGHT,
                gate_count=2,
                spacing=4.8,
                vertical_amp=0.12,
            )
        )
        .add_stage(
            StageSpec(
                kind=StageKind.TURN90,
                gate_count=1,
                turn_degrees=90.0,
                radius=4.8,
                vertical_amp=0.22,
                direction=TurnDirection.LEFT,
            )
        )
        .add_stage(
            StageSpec(
                kind=StageKind.STRAIGHT,
                gate_count=2,
                spacing=5.1,
                vertical_amp=0.18,
            )
        )
        .add_stage(
            StageSpec(
                kind=StageKind.OFFSET,
                gate_count=2,
                spacing=4.4,
                lateral_amp=1.5,
                vertical_amp=0.3,
            )
        )
        .add_stage(
            StageSpec(
                kind=StageKind.TURN90,
                gate_count=1,
                turn_degrees=90.0,
                radius=4.5,
                vertical_amp=0.38,
                direction=TurnDirection.RIGHT,
            )
        )
        .add_stage(
            StageSpec(
                kind=StageKind.STRAIGHT,
                gate_count=2,
                spacing=5.0,
                vertical_amp=0.28,
            )
        )
        .add_stage(
            StageSpec(
                kind=StageKind.TURN90,
                gate_count=1,
                turn_degrees=90.0,
                radius=4.7,
                vertical_amp=0.2,
                direction=TurnDirection.RIGHT,
            )
        )
        .add_stage(
            StageSpec(
                kind=StageKind.STRAIGHT,
                gate_count=2,
                spacing=4.6,
                vertical_amp=0.14,
            )
        )
    )


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m triad_py")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("sim-config", help="Print the default simulation config")
    subparsers.add_parser(
        "default-course", help="Print stats for the default drone course"
    )
    subparsers.add_parser(
        "basic-lap", help="Print stats for the example basic lap course"
    )
    subparsers.add_parser(
        "core-demo", help="Create a headless core sim and print a short rollout"
    )
    subparsers.add_parser("vec-demo", help="Run the vector env wrapper for a few steps")
    subparsers.add_parser(
        "fast-demo", help="Run the packed-array fast env wrapper for a few steps"
    )
    rollout_demo = subparsers.add_parser(
        "rollout-demo", help="Collect a NumPy rollout batch with the fast env"
    )
    rollout_demo.add_argument("--horizon", type=int, default=16)
    benchmark_demo = subparsers.add_parser(
        "benchmark-demo", help="Benchmark fast rollout collection throughput"
    )
    benchmark_demo.add_argument("--horizon", type=int, default=32)
    benchmark_demo.add_argument("--iterations", type=int, default=10)
    benchmark_demo.add_argument("--warmup", type=int, default=1)
    ppo_train = subparsers.add_parser(
        "ppo-train", help="Train a PPO teacher on the Triad sim"
    )
    ppo_train.add_argument("--env-count", type=int, default=256)
    ppo_train.add_argument("--horizon", type=int, default=128)
    ppo_train.add_argument("--total-updates", type=int, default=500)
    ppo_train.add_argument("--warmup-updates", type=int, default=25)
    ppo_train.add_argument("--learning-rate", type=float, default=3.0e-4)
    ppo_train.add_argument("--gamma", type=float, default=0.99)
    ppo_train.add_argument("--gae-lambda", type=float, default=0.95)
    ppo_train.add_argument("--clip-coef", type=float, default=0.2)
    ppo_train.add_argument("--value-clip-coef", type=float, default=0.2)
    ppo_train.add_argument("--value-coef", type=float, default=0.5)
    ppo_train.add_argument("--entropy-coef", type=float, default=0.001)
    ppo_train.add_argument("--max-grad-norm", type=float, default=0.5)
    ppo_train.add_argument("--ppo-epochs", type=int, default=4)
    ppo_train.add_argument("--minibatch-size", type=int, default=4096)
    ppo_train.add_argument("--target-kl", type=float, default=0.015)
    ppo_train.add_argument("--hidden-size", type=int, default=256)
    ppo_train.add_argument("--device", default="auto")
    ppo_train.add_argument("--seed", type=int, default=0)
    ppo_train.add_argument("--log-interval", type=int, default=10)
    ppo_train.add_argument("--checkpoint", default=None)
    ppo_train.add_argument("--checkpoint-interval", type=int, default=0)
    ppo_train.add_argument(
        "--no-lr-anneal", action="store_true", help="Disable learning rate annealing"
    )
    ppo_train.add_argument(
        "--no-advantage-norm",
        action="store_true",
        help="Disable advantage normalization",
    )
    ppo_train.add_argument(
        "--no-observation-norm",
        action="store_true",
        help="Disable running observation normalization",
    )
    ppo_train.add_argument(
        "--observation-clip",
        type=float,
        default=10.0,
        help="Absolute clip applied after observation normalization",
    )
    ppo_policy_server = subparsers.add_parser(
        "ppo-policy-server", help="Serve PPO checkpoint actions over stdio"
    )
    ppo_policy_server.add_argument("--checkpoint", required=True)
    ppo_policy_server.add_argument("--device", default="auto")
    curriculum_demo = subparsers.add_parser(
        "curriculum-demo", help="Sample reset params from the default curriculum"
    )
    curriculum_demo.add_argument("--progress", type=float, default=0.0)
    curriculum_demo.add_argument("--env-count", type=int, default=8)
    curriculum_demo.add_argument("--seed", type=int, default=0)

    custom = subparsers.add_parser("custom-course", help="Build a simple custom course")
    custom.add_argument("--name", default="custom-course")
    custom.add_argument("--laps", type=int, default=1)
    custom.add_argument("--loop", action="store_true", dest="loop_enabled")
    custom.add_argument("--intro-gates", type=int, default=4)
    custom.add_argument("--straight-gates", type=int, default=4)
    custom.add_argument("--turn-gates", type=int, default=3)
    custom.add_argument("--spacing", type=float, default=1.8)
    custom.add_argument("--radius", type=float, default=2.4)
    custom.add_argument("--offset-amp", type=float, default=0.8)
    return parser


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _demo_config(course: CourseSpec) -> SimulationConfig:
    config = SimulationConfig.default()
    config.env_count = 8
    config.max_gates_per_env = max(
        config.max_gates_per_env, int(course.stats["total_gate_count"])
    )
    return config


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    if args.command == "sim-config":
        _print_json(SimulationConfig.default().__dict__)
        return 0

    if args.command == "default-course":
        course = CourseSpec.default_drone_course()
        try:
            _print_json(course.stats)
        finally:
            course.close()
        return 0

    if args.command == "basic-lap":
        course = build_basic_lap_course()
        try:
            _print_json(course.stats)
        finally:
            course.close()
        return 0

    if args.command == "core-demo":
        course = build_basic_lap_course()
        sim = SimulationCore(_demo_config(course))
        try:
            sim.set_course(course).reset_all().step(1)
            sim.set_actions([(0.0, 0.0, 0.0, 0.0)] * sim.env_count).step(4)
            _print_json(
                {
                    "env_count": sim.env_count,
                    "state_head": sim.get_state()[:2],
                    "reward_done_head": sim.get_reward_done()[:2],
                }
            )
        finally:
            sim.close()
            course.close()
        return 0

    if args.command == "vec-demo":
        course = build_basic_lap_course()
        sim = SimulationCore(_demo_config(course))
        env = TriadVecEnv(sim)
        try:
            observations = env.reset()
            step_result = env.step([(0.0, 0.0, 0.0, 0.0)] * sim.env_count)
            _print_json(
                {
                    "reset_head": observations[:2],
                    "step_head": step_result[0][:2],
                    "reward_head": step_result[1][:4],
                    "done_head": step_result[2][:4],
                }
            )
        finally:
            sim.close()
            course.close()
        return 0

    if args.command == "fast-demo":
        course = build_basic_lap_course()
        sim = SimulationCore(_demo_config(course))
        env = TriadFastVecEnv(sim)
        try:
            observations = env.reset()
            result = env.step([(0.0, 0.0, 0.0, 0.0)] * sim.env_count)
            _print_json(
                {
                    "reset_obs_len": len(observations),
                    "step_obs_len": len(result.observations),
                    "reward_head": result.reward_list()[:4],
                    "done_head": result.done_list()[:4],
                    "obs_head": result.observation_rows()[:2],
                }
            )
        finally:
            sim.close()
            course.close()
        return 0

    if args.command == "rollout-demo":
        course = build_basic_lap_course()
        sim = SimulationCore(_demo_config(course))
        env = TriadFastVecEnv(sim)
        try:
            env.reset_numpy()
            batch = collect_rollout_numpy(env, args.horizon, point_to_gate_policy)
            _print_json(
                {
                    "horizon": args.horizon,
                    "obs_shape": list(batch.observations.shape),
                    "actions_shape": list(batch.actions.shape),
                    "rewards_shape": list(batch.rewards.shape),
                    "dones_shape": list(batch.dones.shape),
                    "reward_head": batch.rewards[0, :4].tolist(),
                    "done_head": batch.dones[0, :4].tolist(),
                }
            )
        finally:
            sim.close()
            course.close()
        return 0

    if args.command == "benchmark-demo":
        course = build_basic_lap_course()
        sim = SimulationCore(_demo_config(course))
        env = TriadFastVecEnv(sim)
        try:
            env.reset_numpy()
            result = benchmark_rollout(
                env,
                args.horizon,
                args.iterations,
                point_to_gate_policy,
                args.warmup,
            )
            _print_json(result.__dict__)
        finally:
            sim.close()
            course.close()
        return 0

    if args.command == "curriculum-demo":
        schedule = build_teacher_curriculum_schedule()
        phase = schedule.phase_for_progress(args.progress)
        _print_json(
            {
                "progress": max(0.0, min(1.0, float(args.progress))),
                "phase": phase.name,
                "curriculum_stage": int(phase.curriculum_stage),
                "difficulty_range": [
                    phase.difficulty_min,
                    phase.difficulty_max,
                ],
                "grammar_ids": list(phase.grammar_ids),
                "reset_params_head": sample_curriculum_reset_params(
                    args.env_count,
                    args.progress,
                    args.seed,
                    schedule,
                )[: min(args.env_count, 4)],
            }
        )
        return 0

    if args.command == "ppo-train":
        config = PPOConfig(
            env_count=args.env_count,
            horizon=args.horizon,
            total_updates=args.total_updates,
            warmup_updates=args.warmup_updates,
            learning_rate=args.learning_rate,
            anneal_learning_rate=not args.no_lr_anneal,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_coef=args.clip_coef,
            value_clip_coef=None if args.value_clip_coef <= 0.0 else args.value_clip_coef,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            normalize_advantages=not args.no_advantage_norm,
            target_kl=None if args.target_kl <= 0.0 else args.target_kl,
            normalize_observations=not args.no_observation_norm,
            observation_clip=args.observation_clip,
            hidden_size=args.hidden_size,
            device=args.device,
            seed=args.seed,
            log_interval=args.log_interval,
            checkpoint_path=args.checkpoint,
            checkpoint_interval=args.checkpoint_interval,
        )
        result = train_ppo(config)
        _print_json(result.__dict__)
        return 0

    if args.command == "ppo-policy-server":
        return serve_ppo_policy(args.checkpoint, device=args.device)

    course = (
        CourseSpec(args.name)
        .set_loop_enabled(args.loop_enabled)
        .set_laps_required(args.laps)
        .add_stage(StageSpec.intro(args.intro_gates, args.spacing))
        .add_stage(
            StageSpec.offset(args.straight_gates, args.spacing + 0.2, args.offset_amp)
        )
        .add_stage(StageSpec.turn90(args.turn_gates, args.radius, TurnDirection.LEFT))
    )
    try:
        _print_json(course.stats)
    finally:
        course.close()
    return 0
