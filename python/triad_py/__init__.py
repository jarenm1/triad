from __future__ import annotations

import argparse
import ctypes
import json
import os
from dataclasses import dataclass
from enum import IntEnum
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
    "CourseSpec",
    "SimulationConfig",
    "SimulationCore",
    "StageKind",
    "StageSpec",
    "TriadVecEnv",
    "TurnDirection",
    "build_basic_lap_course",
    "main",
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


class StageKind(IntEnum):
    INTRO = _lib.triad_stage_kind_intro()
    STRAIGHT = _lib.triad_stage_kind_straight()
    OFFSET = _lib.triad_stage_kind_offset()
    TURN90 = _lib.triad_stage_kind_turn90()


class TurnDirection(IntEnum):
    LEFT = _lib.triad_turn_direction_left()
    RIGHT = _lib.triad_turn_direction_right()


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
        ("accel_x", ctypes.c_float),
        ("accel_y", ctypes.c_float),
    ]


class _TriadResetParams(ctypes.Structure):
    _fields_ = [
        ("seed", ctypes.c_uint32),
        ("grammar_id", ctypes.c_uint32),
        ("difficulty", ctypes.c_float),
    ]


class _TriadEnvState(ctypes.Structure):
    _fields_ = [
        ("position_x", ctypes.c_float),
        ("position_y", ctypes.c_float),
        ("velocity_x", ctypes.c_float),
        ("velocity_y", ctypes.c_float),
        ("step_count", ctypes.c_uint32),
        ("done", ctypes.c_uint32),
        ("current_gate", ctypes.c_uint32),
        ("current_lap", ctypes.c_uint32),
    ]


class _TriadObservation(ctypes.Structure):
    _fields_ = [
        ("position_x", ctypes.c_float),
        ("position_y", ctypes.c_float),
        ("velocity_x", ctypes.c_float),
        ("velocity_y", ctypes.c_float),
        ("target_gate_x", ctypes.c_float),
        ("target_gate_y", ctypes.c_float),
        ("progress", ctypes.c_float),
    ]


class _TriadRewardDone(ctypes.Structure):
    _fields_ = [
        ("reward", ctypes.c_float),
        ("done", ctypes.c_uint32),
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
_lib.triad_simulation_readback_reward_done.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(_TriadRewardDone),
    ctypes.c_size_t,
]
_lib.triad_simulation_readback_reward_done.restype = ctypes.c_bool


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

    def set_course(self, course: CourseSpec) -> "SimulationCore":
        _require(_lib.triad_simulation_set_course(self._handle, course._handle))
        return self

    def set_actions(self, actions: Sequence[tuple[float, float]]) -> "SimulationCore":
        ffi_actions = (_TriadAction * len(actions))(
            *(_TriadAction(accel_x=x, accel_y=y) for x, y in actions)
        )
        _require(
            _lib.triad_simulation_set_actions(self._handle, ffi_actions, len(actions))
        )
        return self

    def set_reset_params(
        self, reset_params: Sequence[tuple[int, int, float]]
    ) -> "SimulationCore":
        ffi_params = (_TriadResetParams * len(reset_params))(
            *(
                _TriadResetParams(
                    seed=seed, grammar_id=grammar_id, difficulty=difficulty
                )
                for seed, grammar_id, difficulty in reset_params
            )
        )
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

    def get_state(self) -> list[dict[str, int | float]]:
        values = (_TriadEnvState * self.env_count)()
        _require(
            _lib.triad_simulation_readback_state(self._handle, values, len(values))
        )
        return [
            {
                "position": [float(value.position_x), float(value.position_y)],
                "velocity": [float(value.velocity_x), float(value.velocity_y)],
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
                "position": [float(value.position_x), float(value.position_y)],
                "velocity": [float(value.velocity_x), float(value.velocity_y)],
                "target_gate_position": [
                    float(value.target_gate_x),
                    float(value.target_gate_y),
                ],
                "progress": float(value.progress),
            }
            for value in values
        ]

    def get_reward_done(self) -> list[dict[str, int | float]]:
        values = (_TriadRewardDone * self.env_count)()
        _require(
            _lib.triad_simulation_readback_reward_done(
                self._handle, values, len(values)
            )
        )
        return [
            {"reward": float(value.reward), "done": bool(value.done)}
            for value in values
        ]


class TriadVecEnv:
    def __init__(self, sim: SimulationCore, auto_reset: bool = True) -> None:
        self.sim = sim
        self.auto_reset = auto_reset

    def reset(self) -> list[dict[str, int | float]]:
        self.sim.reset_all().step(1)
        return self.sim.get_observations()

    def step(
        self, actions: Sequence[tuple[float, float]]
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


def build_basic_lap_course() -> CourseSpec:
    return (
        CourseSpec("basic-lap")
        .set_loop_enabled(True)
        .set_laps_required(1)
        .add_stage(StageSpec.intro(4, 1.6))
        .add_stage(
            StageSpec(
                kind=StageKind.OFFSET,
                gate_count=5,
                spacing=1.9,
                lateral_amp=0.8,
                vertical_amp=0.35,
            )
        )
        .add_stage(
            StageSpec(
                kind=StageKind.TURN90,
                gate_count=3,
                turn_degrees=90.0,
                radius=2.6,
                vertical_amp=0.5,
                direction=TurnDirection.LEFT,
            )
        )
        .add_stage(
            StageSpec(
                kind=StageKind.STRAIGHT,
                gate_count=4,
                spacing=2.2,
                vertical_amp=0.25,
            )
        )
        .add_stage(
            StageSpec(
                kind=StageKind.TURN90,
                gate_count=3,
                turn_degrees=90.0,
                radius=2.6,
                vertical_amp=0.6,
                direction=TurnDirection.LEFT,
            )
        )
        .add_stage(
            StageSpec(
                kind=StageKind.OFFSET,
                gate_count=5,
                spacing=1.9,
                lateral_amp=1.0,
                vertical_amp=0.4,
            )
        )
        .add_stage(
            StageSpec(
                kind=StageKind.TURN90,
                gate_count=3,
                turn_degrees=90.0,
                radius=2.6,
                vertical_amp=0.55,
                direction=TurnDirection.LEFT,
            )
        )
        .add_stage(
            StageSpec(
                kind=StageKind.STRAIGHT,
                gate_count=4,
                spacing=2.2,
                vertical_amp=0.2,
            )
        )
        .add_stage(
            StageSpec(
                kind=StageKind.TURN90,
                gate_count=3,
                turn_degrees=90.0,
                radius=2.6,
                vertical_amp=0.3,
                direction=TurnDirection.LEFT,
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
            sim.set_actions([(0.0, 0.0)] * sim.env_count).step(4)
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
            step_result = env.step([(0.0, 0.0)] * sim.env_count)
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
