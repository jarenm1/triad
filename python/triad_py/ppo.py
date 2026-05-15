from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np

from . import (
    CourseSpec,
    CurriculumStage,
    SimulationConfig,
    SimulationCore,
    StageKind,
    StageSpec,
    TriadFastVecEnv,
    TurnDirection,
    build_teacher_curriculum_schedule,
)
from .training_log import DONE_REASON_BITS, DoneReasonTracker, PPOTrainingLogger

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - runtime dependency
    torch = None
    nn = None


ACTION_SPACE_VERSION = "velocity_yaw_setpoint_v3_smooth_asym"
OBSERVATION_SPACE_VERSION = "teacher_privileged_v2_fixed_metrics"
TASK_SPEC_SCHEMA_VERSION = 1
CRITIC_CONTEXT_SCALAR_DIM = 5

DEFAULT_TRAINING_COURSE_SPEC: dict[str, object] = {
    "name": "basic-lap",
    "loop_enabled": True,
    "laps_required": 1,
    "stages": [
        {
            "kind": "INTRO",
            "gate_count": 1,
            "spacing": 3.2,
            "lateral_amp": 0.0,
            "turn_degrees": 0.0,
            "radius": 0.0,
            "vertical_amp": 0.08,
            "hole_half_width": 0.6,
            "hole_half_height": 0.6,
            "direction": "LEFT",
        },
        {
            "kind": "STRAIGHT",
            "gate_count": 2,
            "spacing": 4.8,
            "lateral_amp": 0.0,
            "turn_degrees": 0.0,
            "radius": 0.0,
            "vertical_amp": 0.12,
            "hole_half_width": 0.6,
            "hole_half_height": 0.6,
            "direction": "LEFT",
        },
        {
            "kind": "TURN90",
            "gate_count": 1,
            "spacing": 1.5,
            "lateral_amp": 0.0,
            "turn_degrees": 90.0,
            "radius": 4.8,
            "vertical_amp": 0.22,
            "hole_half_width": 0.6,
            "hole_half_height": 0.6,
            "direction": "LEFT",
        },
        {
            "kind": "STRAIGHT",
            "gate_count": 2,
            "spacing": 5.1,
            "lateral_amp": 0.0,
            "turn_degrees": 0.0,
            "radius": 0.0,
            "vertical_amp": 0.18,
            "hole_half_width": 0.6,
            "hole_half_height": 0.6,
            "direction": "LEFT",
        },
        {
            "kind": "OFFSET",
            "gate_count": 2,
            "spacing": 4.4,
            "lateral_amp": 1.5,
            "turn_degrees": 0.0,
            "radius": 0.0,
            "vertical_amp": 0.3,
            "hole_half_width": 0.6,
            "hole_half_height": 0.6,
            "direction": "LEFT",
        },
        {
            "kind": "TURN90",
            "gate_count": 1,
            "spacing": 1.5,
            "lateral_amp": 0.0,
            "turn_degrees": 90.0,
            "radius": 4.5,
            "vertical_amp": 0.38,
            "hole_half_width": 0.6,
            "hole_half_height": 0.6,
            "direction": "RIGHT",
        },
        {
            "kind": "STRAIGHT",
            "gate_count": 2,
            "spacing": 5.0,
            "lateral_amp": 0.0,
            "turn_degrees": 0.0,
            "radius": 0.0,
            "vertical_amp": 0.28,
            "hole_half_width": 0.6,
            "hole_half_height": 0.6,
            "direction": "LEFT",
        },
        {
            "kind": "TURN90",
            "gate_count": 1,
            "spacing": 1.5,
            "lateral_amp": 0.0,
            "turn_degrees": 90.0,
            "radius": 4.7,
            "vertical_amp": 0.2,
            "hole_half_width": 0.6,
            "hole_half_height": 0.6,
            "direction": "RIGHT",
        },
        {
            "kind": "STRAIGHT",
            "gate_count": 2,
            "spacing": 4.6,
            "lateral_amp": 0.0,
            "turn_degrees": 0.0,
            "radius": 0.0,
            "vertical_amp": 0.14,
            "hole_half_width": 0.6,
            "hole_half_height": 0.6,
            "direction": "LEFT",
        },
    ],
}


def _require_torch():
    if torch is None or nn is None:
        raise RuntimeError(
            "PyTorch is required for PPO training. Install with `uv sync --extra training`."
        )
    return torch, nn


def _create_tensorboard_writer(config: "PPOConfig"):
    if config.tensorboard_dir is None:
        return None, None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "TensorBoard logging requires `tensorboard`. Install with `uv sync --extra training`."
        ) from error

    checkpoint_stem = (
        Path(config.checkpoint_path).stem if config.checkpoint_path else "ppo"
    )
    run_name = (
        config.run_name
        or f"{checkpoint_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    log_dir = Path(config.tensorboard_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    return writer, log_dir


def _resolve_device(requested: str) -> str:
    torch_module, _ = _require_torch()
    if requested != "auto":
        return requested
    if torch_module.backends.mps.is_available():
        return "mps"
    if torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class PPOConfig:
    env_count: int = 256
    task_name: str = "drone-racing-basic-lap"
    task_version: str = "v1"
    horizon: int = 128
    total_updates: int = 500
    warmup_updates: int = 25
    max_episode_steps: int = 2048
    dynamics_randomization_scale: float = 1.0
    actuator_randomization_scale: float = 1.0
    spawn_randomization_scale: float = 1.0
    forward_progress_reward_scale: float = 0.45
    backward_progress_reward_scale: float = 0.8
    gate_pass_reward: float = 10.0
    gate_pass_speed_reward_scale: float = 0.35
    gate_pass_speed_reward_cap: float = 12.0
    course_completion_reward: float = 48.0
    time_penalty_base: float = 0.0005
    time_penalty_delay: float = 3.0
    time_penalty_scale: float = 0.004
    time_penalty_cap: float = 0.001
    collision_penalty: float = 14.0
    out_of_bounds_penalty: float = 24.0
    bootstrap_distance_reward_scale: float = 1.6
    bootstrap_alignment_reward_scale: float = 0.3
    bootstrap_centering_reward_scale: float = 0.35
    bootstrap_velocity_alignment_reward_scale: float = 0.2
    bootstrap_gate_pass_speed_reward_scale: float = 0.0
    bootstrap_time_penalty_scale: float = 0.2
    bootstrap_collision_penalty_scale: float = 0.35
    bootstrap_out_of_bounds_penalty_scale: float = 0.4
    learning_rate: float = 3.0e-4
    anneal_learning_rate: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_clip_coef: float | None = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    action_smoothness_reward_scale: float = 0.035
    action_smoothness_loss_coef: float = 0.015
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 4096
    normalize_advantages: bool = True
    target_kl: float | None = 0.015
    normalize_observations: bool = True
    observation_clip: float = 10.0
    hidden_size: int = 256
    seed: int = 0
    device: str = "auto"
    log_interval: int = 10
    pretty_log: bool = True
    json_log: bool = True
    tensorboard_dir: str | None = "runs"
    run_name: str | None = None
    checkpoint_path: str | None = None
    resume_from: str | None = None
    resume_model_only: bool = False
    checkpoint_interval: int = 0
    curriculum_eval_interval: int = 10
    curriculum_eval_env_count: int = 64
    curriculum_mastery_window: int = 3
    curriculum_min_stage_updates: int = 20
    curriculum_completion_threshold: float = 0.6
    curriculum_progress_threshold: float = 0.9
    curriculum_gate_pass_threshold: float = 0.85
    curriculum_current_weight: float = 0.7
    curriculum_previous_weight: float = 0.2
    curriculum_easy_weight: float = 0.1
    curriculum_holdout_seed: int = 131071
    adaptive_noise_enabled: bool = True
    adaptive_noise_initial: float = 0.05
    adaptive_noise_min: float = 0.0
    adaptive_noise_max: float = 0.55
    adaptive_noise_step_up: float = 0.025
    adaptive_noise_step_down: float = 0.05
    adaptive_noise_target_failure_rate: float = 0.25
    soft_failure_curriculum_enabled: bool = True
    soft_failure_initial: float = 1.0
    soft_failure_min: float = 0.0
    soft_failure_step_up: float = 0.025
    soft_failure_step_down: float = 0.04
    soft_failure_target_failure_rate: float = 0.18
    pretrain_updates: int = 0
    pretrain_epochs: int = 4
    pretrain_minibatch_size: int = 4096
    pretrain_bootstrap_weight: float = 1.0
    pretrain_intro_weight: float = 0.0
    pretrain_max_gate_distance: float = 1.25


@dataclass
class PPOUpdateStats:
    update_index: int
    progress: float
    phase: str
    phase_index: int
    phase_updates: int
    mean_reward: float
    mean_done_rate: float
    completed_episodes: int
    mean_episode_return: float
    mean_episode_length: float
    eval_completion_rate: float
    eval_gate_pass_rate: float
    eval_mean_progress: float
    eval_mean_episode_return: float
    eval_mean_episode_length: float
    phase_advanced: bool
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    action_smoothness_loss: float
    mean_action_delta: float
    mean_action_smoothness_penalty: float
    explained_variance: float
    learning_rate: float
    elapsed_seconds: float


@dataclass
class PPOTrainResult:
    final_update: int
    final_progress: float
    final_phase: str
    final_phase_index: int
    device: str
    env_count: int
    horizon: int
    total_env_steps: int
    checkpoint_path: str | None


@dataclass
class CurriculumEvalStats:
    phase_index: int
    phase: str
    completion_rate: float
    gate_pass_rate: float
    mean_progress: float
    mean_episode_return: float
    mean_episode_length: float
    per_env: list["CurriculumEvalEnvResult"] | None = None
    gate_metrics: dict[str, object] | None = None


@dataclass
class CurriculumEvalEnvResult:
    env_index: int
    seed: int
    grammar_id: int
    difficulty: float
    curriculum_stage: int
    completed: bool
    done: bool
    done_reason: int
    done_reason_labels: list[str]
    episode_return: float
    episode_length: int
    best_progress: float
    final_progress: float
    current_gate: int
    current_lap: int
    step_count: int
    position: list[float]
    velocity: list[float]
    attitude: list[float]
    target_gate_position: list[float]
    distance_to_gate: float
    gate_alignment: float
    reward: float
    shaping_reward: float
    out_of_bounds_penalty: float
    time_penalty: float
    sparse_objective_reward: float
    collision_penalty: float


@dataclass
class PhaseBestEvalSnapshot:
    phase_index: int
    eval_stats: CurriculumEvalStats
    model_state: dict[str, object]
    optimizer_state: dict[str, object]
    observation_normalizer_state: dict[str, object] | None
    update_index: int
    total_env_steps: int


@dataclass
class PPOEvalResult:
    checkpoint_path: str
    checkpoint_update_index: int | None
    checkpoint_total_env_steps: int | None
    device: str
    eval_env_count: int
    max_episode_steps: int
    per_env: bool
    task_name: str | None
    task_version: str | None
    task_hash: str | None
    results: list[CurriculumEvalStats]


def _curriculum_eval_env_result_from_dict(
    payload: dict[str, object],
) -> CurriculumEvalEnvResult:
    return CurriculumEvalEnvResult(
        env_index=int(payload["env_index"]),
        seed=int(payload["seed"]),
        grammar_id=int(payload["grammar_id"]),
        difficulty=float(payload["difficulty"]),
        curriculum_stage=int(payload["curriculum_stage"]),
        completed=bool(payload["completed"]),
        done=bool(payload["done"]),
        done_reason=int(payload["done_reason"]),
        done_reason_labels=list(payload["done_reason_labels"]),
        episode_return=float(payload["episode_return"]),
        episode_length=int(payload["episode_length"]),
        best_progress=float(payload["best_progress"]),
        final_progress=float(payload["final_progress"]),
        current_gate=int(payload["current_gate"]),
        current_lap=int(payload["current_lap"]),
        step_count=int(payload["step_count"]),
        position=list(payload["position"]),
        velocity=list(payload["velocity"]),
        attitude=list(payload["attitude"]),
        target_gate_position=list(payload["target_gate_position"]),
        distance_to_gate=float(payload["distance_to_gate"]),
        gate_alignment=float(payload["gate_alignment"]),
        reward=float(payload["reward"]),
        shaping_reward=float(payload["shaping_reward"]),
        out_of_bounds_penalty=float(payload["out_of_bounds_penalty"]),
        time_penalty=float(payload["time_penalty"]),
        sparse_objective_reward=float(payload["sparse_objective_reward"]),
        collision_penalty=float(payload["collision_penalty"]),
    )


def _curriculum_eval_stats_from_dict(
    payload: dict[str, object],
) -> CurriculumEvalStats:
    per_env_payload = payload.get("per_env")
    per_env = None
    if per_env_payload is not None:
        per_env = [
            _curriculum_eval_env_result_from_dict(item) for item in per_env_payload
        ]
    return CurriculumEvalStats(
        phase_index=int(payload["phase_index"]),
        phase=str(payload["phase"]),
        completion_rate=float(payload["completion_rate"]),
        gate_pass_rate=float(
            payload.get("gate_pass_rate", payload["completion_rate"])
        ),
        mean_progress=float(payload["mean_progress"]),
        mean_episode_return=float(payload["mean_episode_return"]),
        mean_episode_length=float(payload["mean_episode_length"]),
        per_env=per_env,
        gate_metrics=payload.get("gate_metrics"),
    )


def _safe_checkpoint_stem(name: str) -> str:
    stem = "".join(
        character if character.isalnum() or character in ("-", "_", ".") else "_"
        for character in name.strip()
    )
    return stem or "ppo"


def _resolve_checkpoint_path(config: PPOConfig) -> Path | None:
    if config.checkpoint_path:
        return Path(config.checkpoint_path)
    if config.resume_from:
        return Path(config.resume_from)
    if config.run_name:
        return Path("checkpoints") / f"{_safe_checkpoint_stem(config.run_name)}.pt"
    return None


def _best_checkpoint_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(
        f"{checkpoint_path.stem}.best{checkpoint_path.suffix}"
    )


def _eval_score(
    eval_stats: CurriculumEvalStats,
) -> tuple[float, float, float, float, float]:
    return (
        float(eval_stats.completion_rate),
        float(eval_stats.gate_pass_rate),
        float(eval_stats.mean_progress),
        float(eval_stats.mean_episode_return),
        -float(eval_stats.mean_episode_length),
    )


def _restore_best_phase_snapshot(
    snapshot: PhaseBestEvalSnapshot,
    *,
    model: "ActorCritic",
    optimizer,
    observation_normalizer: "RunningMeanStd | None",
) -> "RunningMeanStd | None":
    model.load_state_dict(snapshot.model_state)
    optimizer.load_state_dict(snapshot.optimizer_state)
    if snapshot.observation_normalizer_state is None:
        return None
    restored = RunningMeanStd.from_state_dict(snapshot.observation_normalizer_state)
    if observation_normalizer is None:
        return restored
    observation_normalizer.mean = restored.mean
    observation_normalizer.var = restored.var
    observation_normalizer.count = restored.count
    observation_normalizer.clip = restored.clip
    return observation_normalizer


@dataclass
class LoadedPPOPolicy:
    model: "ActorCritic"
    device: str
    config: dict[str, object]
    observation_normalizer: "RunningMeanStd | None"
    task_spec: dict[str, object] | None

    def predict(
        self, observations: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        torch_module, _ = _require_torch()
        observation_array = np.asarray(observations, dtype=np.float32)
        if observation_array.ndim == 1:
            observation_array = observation_array[None, :]
        if self.observation_normalizer is not None:
            observation_array = self.observation_normalizer.normalize(observation_array)

        obs_tensor = torch_module.as_tensor(
            observation_array, dtype=torch_module.float32, device=self.device
        )
        with torch_module.no_grad():
            if deterministic:
                actions, values = self.model.act_deterministic(obs_tensor)
            else:
                actions, _, _, values = self.model.act(obs_tensor)

        return (
            actions.detach().cpu().numpy(),
            values.detach().cpu().numpy(),
        )


if nn is not None:

    def _init_linear(layer, std: float = math.sqrt(2.0), bias: float = 0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias)
        return layer

    class ActorCritic(nn.Module):
        def __init__(
            self,
            observation_dim: int,
            action_dim: int,
            hidden_size: int,
            critic_observation_dim: int | None = None,
        ):
            super().__init__()
            self.observation_dim = int(observation_dim)
            self.critic_observation_dim = int(
                observation_dim if critic_observation_dim is None else critic_observation_dim
            )
            self.actor_body = nn.Sequential(
                _init_linear(nn.Linear(observation_dim, hidden_size)),
                nn.Tanh(),
                _init_linear(nn.Linear(hidden_size, hidden_size)),
                nn.Tanh(),
            )
            self.critic_body = nn.Sequential(
                _init_linear(nn.Linear(self.critic_observation_dim, hidden_size)),
                nn.Tanh(),
                _init_linear(nn.Linear(hidden_size, hidden_size)),
                nn.Tanh(),
            )
            self.policy_mean = _init_linear(
                nn.Linear(hidden_size, action_dim), std=0.01
            )
            self.value_head = _init_linear(nn.Linear(hidden_size, 1), std=1.0)
            self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

        def _default_critic_observations(self, observations):
            if self.critic_observation_dim == observations.shape[-1]:
                return observations
            extra_dim = self.critic_observation_dim - observations.shape[-1]
            if extra_dim < 0:
                return observations[..., : self.critic_observation_dim]
            zeros = observations.new_zeros((*observations.shape[:-1], extra_dim))
            return torch.cat((observations, zeros), dim=-1)

        def forward(self, observations, critic_observations=None):
            if critic_observations is None:
                critic_observations = self._default_critic_observations(observations)
            actor_features = self.actor_body(observations)
            critic_features = self.critic_body(critic_observations)
            return self.policy_mean(actor_features), self.value_head(
                critic_features
            ).squeeze(-1)

        def act(self, observations, critic_observations=None):
            mean, value = self.forward(observations, critic_observations)
            std = torch.exp(self.log_std).expand_as(mean)
            dist = torch.distributions.Normal(mean, std)
            raw_action = dist.rsample()
            actions = torch.sigmoid(raw_action)
            log_prob = _squashed_log_prob(dist, raw_action, actions)
            entropy = dist.entropy().sum(dim=-1)
            return actions, log_prob, entropy, value

        def act_deterministic(self, observations, critic_observations=None):
            mean, value = self.forward(observations, critic_observations)
            actions = torch.sigmoid(mean)
            return actions, value

        def evaluate_actions(self, observations, actions, critic_observations=None):
            mean, value = self.forward(observations, critic_observations)
            std = torch.exp(self.log_std).expand_as(mean)
            dist = torch.distributions.Normal(mean, std)
            safe_actions = actions.clamp(1.0e-5, 1.0 - 1.0e-5)
            raw_action = torch.logit(safe_actions)
            log_prob = _squashed_log_prob(dist, raw_action, safe_actions)
            entropy = dist.entropy().sum(dim=-1)
            mean_action = torch.sigmoid(mean)
            return log_prob, entropy, value, mean_action

else:

    class ActorCritic:  # pragma: no cover - runtime dependency guard
        def __init__(self, *args, **kwargs):
            del args, kwargs
            _require_torch()


def _squashed_log_prob(dist, raw_action, squashed_action):
    log_det = torch.log(squashed_action) + torch.log1p(-squashed_action)
    return (dist.log_prob(raw_action) - log_det).sum(dim=-1)


def _set_global_seeds(seed: int) -> None:
    torch_module, _ = _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def _copy_json_value(value: object) -> object:
    return json.loads(json.dumps(value))


def _build_default_training_course() -> CourseSpec:
    course = (
        CourseSpec(str(DEFAULT_TRAINING_COURSE_SPEC["name"]))
        .set_loop_enabled(bool(DEFAULT_TRAINING_COURSE_SPEC["loop_enabled"]))
        .set_laps_required(int(DEFAULT_TRAINING_COURSE_SPEC["laps_required"]))
    )
    for stage in DEFAULT_TRAINING_COURSE_SPEC["stages"]:
        assert isinstance(stage, dict)
        course.add_stage(
            StageSpec(
                kind=StageKind[str(stage["kind"])],
                gate_count=int(stage["gate_count"]),
                spacing=float(stage["spacing"]),
                lateral_amp=float(stage["lateral_amp"]),
                turn_degrees=float(stage["turn_degrees"]),
                radius=float(stage["radius"]),
                vertical_amp=float(stage["vertical_amp"]),
                hole_half_width=float(stage["hole_half_width"]),
                hole_half_height=float(stage["hole_half_height"]),
                direction=TurnDirection[str(stage["direction"])],
            )
        )
    return course


def _default_training_course() -> CourseSpec:
    return _build_default_training_course()


def _training_sim_config(course: CourseSpec, config: PPOConfig) -> SimulationConfig:
    sim_config = SimulationConfig.default()
    sim_config.env_count = config.env_count
    sim_config.max_steps = max(sim_config.max_steps, config.max_episode_steps)
    sim_config.max_gates_per_env = max(
        sim_config.max_gates_per_env, int(course.stats["total_gate_count"])
    )
    sim_config.dynamics_randomization_scale = config.dynamics_randomization_scale
    sim_config.actuator_randomization_scale = config.actuator_randomization_scale
    sim_config.spawn_randomization_scale = config.spawn_randomization_scale
    sim_config.forward_progress_reward_scale = config.forward_progress_reward_scale
    sim_config.backward_progress_reward_scale = config.backward_progress_reward_scale
    sim_config.gate_pass_reward = config.gate_pass_reward
    sim_config.gate_pass_speed_reward_scale = config.gate_pass_speed_reward_scale
    sim_config.gate_pass_speed_reward_cap = config.gate_pass_speed_reward_cap
    sim_config.course_completion_reward = config.course_completion_reward
    sim_config.time_penalty_base = config.time_penalty_base
    sim_config.time_penalty_delay = config.time_penalty_delay
    sim_config.time_penalty_scale = config.time_penalty_scale
    sim_config.time_penalty_cap = config.time_penalty_cap
    sim_config.collision_penalty = config.collision_penalty
    sim_config.out_of_bounds_penalty = config.out_of_bounds_penalty
    sim_config.bootstrap_distance_reward_scale = config.bootstrap_distance_reward_scale
    sim_config.bootstrap_alignment_reward_scale = (
        config.bootstrap_alignment_reward_scale
    )
    sim_config.bootstrap_centering_reward_scale = (
        config.bootstrap_centering_reward_scale
    )
    sim_config.bootstrap_velocity_alignment_reward_scale = (
        config.bootstrap_velocity_alignment_reward_scale
    )
    sim_config.bootstrap_gate_pass_speed_reward_scale = (
        config.bootstrap_gate_pass_speed_reward_scale
    )
    sim_config.bootstrap_time_penalty_scale = config.bootstrap_time_penalty_scale
    sim_config.bootstrap_collision_penalty_scale = (
        config.bootstrap_collision_penalty_scale
    )
    sim_config.bootstrap_out_of_bounds_penalty_scale = (
        config.bootstrap_out_of_bounds_penalty_scale
    )
    return sim_config


def _evaluation_sim_config(course: CourseSpec, config: PPOConfig) -> SimulationConfig:
    sim_config = _training_sim_config(course, config)
    sim_config.env_count = max(
        1, min(config.curriculum_eval_env_count, config.env_count)
    )
    return sim_config


def _task_reward_config(config: PPOConfig) -> dict[str, object]:
    return {
        "forward_progress_reward_scale": config.forward_progress_reward_scale,
        "backward_progress_reward_scale": config.backward_progress_reward_scale,
        "gate_pass_reward": config.gate_pass_reward,
        "gate_pass_speed_reward_scale": config.gate_pass_speed_reward_scale,
        "gate_pass_speed_reward_cap": config.gate_pass_speed_reward_cap,
        "course_completion_reward": config.course_completion_reward,
        "time_penalty_base": config.time_penalty_base,
        "time_penalty_delay": config.time_penalty_delay,
        "time_penalty_scale": config.time_penalty_scale,
        "time_penalty_cap": config.time_penalty_cap,
        "collision_penalty": config.collision_penalty,
        "out_of_bounds_penalty": config.out_of_bounds_penalty,
        "bootstrap_distance_reward_scale": config.bootstrap_distance_reward_scale,
        "bootstrap_alignment_reward_scale": config.bootstrap_alignment_reward_scale,
        "bootstrap_centering_reward_scale": config.bootstrap_centering_reward_scale,
        "bootstrap_velocity_alignment_reward_scale": config.bootstrap_velocity_alignment_reward_scale,
        "bootstrap_gate_pass_speed_reward_scale": config.bootstrap_gate_pass_speed_reward_scale,
        "bootstrap_time_penalty_scale": config.bootstrap_time_penalty_scale,
        "bootstrap_collision_penalty_scale": config.bootstrap_collision_penalty_scale,
        "bootstrap_out_of_bounds_penalty_scale": config.bootstrap_out_of_bounds_penalty_scale,
        "camera_alignment_reward": "stage_gated_lookahead_v1",
        "action_smoothness_reward_scale": config.action_smoothness_reward_scale,
        "action_smoothness_loss_coef": config.action_smoothness_loss_coef,
    }


def _task_randomization_config(config: PPOConfig) -> dict[str, object]:
    return {
        "dynamics_randomization_scale": config.dynamics_randomization_scale,
        "actuator_randomization_scale": config.actuator_randomization_scale,
        "spawn_randomization_scale": config.spawn_randomization_scale,
    }


def _task_curriculum_config(config: PPOConfig, schedule) -> dict[str, object]:
    return {
        "schedule_name": "axis_primitive_curriculum_v7",
        "start_gate_replay": FAILURE_REPLAY_STRATEGY,
        "direction_replay": "eval_weak_direction_bias_v1",
        "phases": [
            {
                "name": phase.name,
                "progress_end": phase.progress_end,
                "curriculum_stage": int(phase.curriculum_stage),
                "difficulty_min": phase.difficulty_min,
                "difficulty_max": phase.difficulty_max,
                "grammar_ids": list(phase.grammar_ids),
                "axis_ranges": asdict(phase.axis_ranges),
            }
            for phase in schedule.phases
        ],
        "mastery_window": config.curriculum_mastery_window,
        "min_stage_updates": config.curriculum_min_stage_updates,
        "completion_threshold": config.curriculum_completion_threshold,
        "progress_threshold": config.curriculum_progress_threshold,
        "gate_pass_threshold": config.curriculum_gate_pass_threshold,
        "current_weight": config.curriculum_current_weight,
        "previous_weight": config.curriculum_previous_weight,
        "easy_weight": config.curriculum_easy_weight,
        "holdout_seed": config.curriculum_holdout_seed,
        "adaptive_noise_enabled": config.adaptive_noise_enabled,
        "adaptive_noise_initial": config.adaptive_noise_initial,
        "adaptive_noise_min": config.adaptive_noise_min,
        "adaptive_noise_max": config.adaptive_noise_max,
        "adaptive_noise_step_up": config.adaptive_noise_step_up,
        "adaptive_noise_step_down": config.adaptive_noise_step_down,
        "adaptive_noise_target_failure_rate": config.adaptive_noise_target_failure_rate,
        "soft_failure_curriculum_enabled": config.soft_failure_curriculum_enabled,
        "soft_failure_initial": config.soft_failure_initial,
        "soft_failure_min": config.soft_failure_min,
        "soft_failure_step_up": config.soft_failure_step_up,
        "soft_failure_step_down": config.soft_failure_step_down,
        "soft_failure_target_failure_rate": config.soft_failure_target_failure_rate,
    }


def _build_training_task_spec(config: PPOConfig, schedule) -> dict[str, object]:
    course = _default_training_course()
    try:
        sim_config = _training_sim_config(course, config)
    finally:
        course.close()
    payload = {
        "schema_version": TASK_SPEC_SCHEMA_VERSION,
        "task_name": config.task_name,
        "task_version": config.task_version,
        "action_space_version": ACTION_SPACE_VERSION,
        "observation_space_version": OBSERVATION_SPACE_VERSION,
        "environment": {
            "dt_seconds": sim_config.dt_seconds,
            "bounds": sim_config.bounds,
            "max_episode_steps": config.max_episode_steps,
            "max_gates_per_env": sim_config.max_gates_per_env,
            "laps_required": sim_config.laps_required,
        },
        "course": {
            "template": "basic-lap",
            **_copy_json_value(DEFAULT_TRAINING_COURSE_SPEC),
        },
        "reward": _task_reward_config(config),
        "randomization": _task_randomization_config(config),
        "curriculum": _task_curriculum_config(config, schedule),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["task_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return payload


def _resolve_artifact_dir(
    config: PPOConfig,
    *,
    tensorboard_log_dir: str | None,
) -> Path | None:
    if tensorboard_log_dir is not None:
        return Path(tensorboard_log_dir)
    if config.checkpoint_path:
        checkpoint_path = Path(config.checkpoint_path)
        return checkpoint_path.parent / f"{checkpoint_path.stem}_artifacts"
    return None


def _save_run_artifacts(
    artifact_dir: Path | None,
    *,
    config: PPOConfig,
    task_spec: dict[str, object],
) -> None:
    if artifact_dir is None:
        return
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "task_spec.json").write_text(
        json.dumps(task_spec, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "training_config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _hash_to_unit(seed: int) -> float:
    mixed = (seed * 747796405 + 2891336453) & 0xFFFFFFFF
    masked = mixed & 0x00FFFFFF
    return masked / 16777215.0


def _hash_to_signed(seed: int) -> float:
    return _hash_to_unit(seed) * 2.0 - 1.0


def _curriculum_dynamics_scale(curriculum_stage: int) -> float:
    if curriculum_stage == int(CurriculumStage.BOOTSTRAP):
        return 0.0
    if curriculum_stage == int(CurriculumStage.INTRO):
        return 0.35
    if curriculum_stage == int(CurriculumStage.ARENA):
        return 0.55
    if curriculum_stage == int(CurriculumStage.TECHNICAL):
        return 0.8
    return 1.0


def _randomized_positive_scale(seed: int, salt: int, magnitude: float) -> float:
    return max(0.25, 1.0 + _hash_to_signed((seed ^ salt) & 0xFFFFFFFF) * magnitude)


def _teacher_point_to_gate_actions(observations: np.ndarray) -> np.ndarray:
    position = observations[:, 0:3]
    attitude = observations[:, 6:9]
    target_gate_position = observations[:, 12:15]
    target_gate_forward = observations[:, 15:18]
    body_velocity = observations[:, 22:25]
    target_gate_body = observations[:, 25:28]

    target_delta = target_gate_position - position
    yaw = attitude[:, 2]

    horizontal_delta = target_delta[:, [0, 2]]
    lateral_error = target_gate_body[:, 0]
    altitude_error = target_gate_body[:, 1]
    horizontal_distance = np.linalg.norm(horizontal_delta, axis=1)
    safe_distance = np.maximum(horizontal_distance, 1.0e-6)
    approach_dir = horizontal_delta / safe_distance[:, None]
    gate_forward_xz = target_gate_forward[:, [0, 2]]
    gate_forward_norm = np.maximum(
        np.linalg.norm(gate_forward_xz, axis=1, keepdims=True),
        1.0e-6,
    )
    gate_forward_unit = gate_forward_xz / gate_forward_norm
    gate_align_weight = np.clip(1.0 - horizontal_distance / 3.5, 0.0, 1.0)
    desired_dir = (
        approach_dir * (1.0 - gate_align_weight)[:, None]
        + gate_forward_unit * gate_align_weight[:, None]
    )
    desired_dir_norm = np.maximum(
        np.linalg.norm(desired_dir, axis=1, keepdims=True),
        1.0e-6,
    )
    desired_dir = desired_dir / desired_dir_norm
    desired_yaw = np.arctan2(desired_dir[:, 1], desired_dir[:, 0])
    yaw_error = (desired_yaw - yaw + np.pi) % (2.0 * np.pi) - np.pi
    yaw_scale = 1.0 - np.clip((np.abs(yaw_error) - 0.25) / 0.65, 0.0, 1.0)
    lateral_scale = 1.0 - np.clip((np.abs(lateral_error) - 0.25) / 1.25, 0.0, 1.0)
    approach_scale = np.maximum(0.35, yaw_scale * lateral_scale)
    lateral_control = np.sign(lateral_error) * np.maximum(np.abs(lateral_error) - 0.12, 0.0)
    altitude_control = np.sign(altitude_error) * np.maximum(np.abs(altitude_error) - 0.35, 0.0)
    forward_velocity_target = (
        np.clip(
            (horizontal_distance - 0.3) * 1.6,
            0.35,
            2.8,
        )
        * approach_scale
    )
    lateral_velocity_target = np.clip(
        lateral_control * 1.0 - body_velocity[:, 0] * 0.15,
        -2.2,
        2.2,
    )
    vertical_velocity_target = np.clip(
        np.maximum(altitude_control * 0.7, -0.2) - body_velocity[:, 1] * 0.05,
        -0.4,
        1.1,
    )
    yaw_rate_target = np.clip(yaw_error * 2.0, -3.2, 3.2)

    actions = np.empty((observations.shape[0], 4), dtype=np.float32)
    actions[:, 0] = np.clip(0.5 + forward_velocity_target / 12.0, 0.0, 1.0)
    actions[:, 1] = np.clip(0.5 + lateral_velocity_target / 8.0, 0.0, 1.0)
    actions[:, 2] = np.clip(0.5 + vertical_velocity_target / 5.0, 0.0, 1.0)
    actions[:, 3] = np.clip(0.5 + yaw_rate_target / 7.0, 0.0, 1.0)
    return actions


def _summarize_array(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def _randomization_preview(
    reset_params: list[tuple],
    config: PPOConfig,
) -> dict[str, object]:
    if not reset_params:
        return {}

    expanded = [_expand_reset_param(item) for item in reset_params]
    seeds = np.asarray([item[0] for item in expanded], dtype=np.uint32)
    difficulties = np.asarray([item[2] for item in expanded], dtype=np.float32)
    curriculum_stages = np.asarray([item[3] for item in expanded], dtype=np.uint32)
    gate_count_levels = np.asarray([item[4] for item in expanded], dtype=np.float32)
    gate_size_levels = np.asarray([item[5] for item in expanded], dtype=np.float32)
    spacing_levels = np.asarray([item[6] for item in expanded], dtype=np.float32)
    verticality_levels = np.asarray([item[7] for item in expanded], dtype=np.float32)
    adaptive_noise_levels = np.asarray([item[8] for item in expanded], dtype=np.float32)
    spawn_noise_levels = np.asarray([item[9] for item in expanded], dtype=np.float32)
    dynamics_noise_levels = np.asarray([item[10] for item in expanded], dtype=np.float32)
    obstacle_density_levels = np.asarray([item[11] for item in expanded], dtype=np.float32)
    path_curvature_levels = np.asarray([item[12] for item in expanded], dtype=np.float32)
    soft_failure_levels = np.asarray([item[13] for item in expanded], dtype=np.float32)

    dynamics_strength = (
        (0.04 + np.clip(dynamics_noise_levels, 0.0, 1.0) * 0.18)
        * np.asarray(
            [_curriculum_dynamics_scale(int(stage)) for stage in curriculum_stages],
            dtype=np.float32,
        )
        * float(config.dynamics_randomization_scale)
    )
    actuator_strength = (
        dynamics_strength * 0.25 * float(config.actuator_randomization_scale)
    )

    def scales(salt: int, multiplier: float = 1.0) -> np.ndarray:
        return np.asarray(
            [
                _randomized_positive_scale(
                    int(seed), salt, float(strength) * multiplier
                )
                for seed, strength in zip(seeds, dynamics_strength, strict=True)
            ],
            dtype=np.float32,
        )

    def actuator_scales(salt: int) -> np.ndarray:
        return np.asarray(
            [
                _randomized_positive_scale(int(seed), salt, float(strength))
                for seed, strength in zip(seeds, actuator_strength, strict=True)
            ],
            dtype=np.float32,
        )

    motor_scales = np.stack(
        (
            actuator_scales(0x6001),
            actuator_scales(0x6003),
            actuator_scales(0x6005),
            actuator_scales(0x6007),
        ),
        axis=1,
    )

    spawn_scale = float(config.spawn_randomization_scale)
    lateral_spawn_offset = np.asarray(
        [
            _hash_to_signed(int(seed) ^ 0x611)
            * (0.08 + float(spawn_level) * 0.18)
            * spawn_scale
            for seed, spawn_level in zip(seeds, spawn_noise_levels, strict=True)
        ],
        dtype=np.float32,
    )
    yaw_offset = np.asarray(
        [_hash_to_signed(int(seed) ^ 0x1551) * 0.25 * spawn_scale for seed in seeds],
        dtype=np.float32,
    )

    return {
        "difficulty": _summarize_array(difficulties),
        "gate_count_level": _summarize_array(gate_count_levels),
        "gate_size_level": _summarize_array(gate_size_levels),
        "spacing_level": _summarize_array(spacing_levels),
        "verticality_level": _summarize_array(verticality_levels),
        "adaptive_gate_noise": _summarize_array(adaptive_noise_levels),
        "spawn_noise_level": _summarize_array(spawn_noise_levels),
        "dynamics_noise_level": _summarize_array(dynamics_noise_levels),
        "obstacle_density_level": _summarize_array(obstacle_density_levels),
        "path_curvature_level": _summarize_array(path_curvature_levels),
        "soft_failure": _summarize_array(soft_failure_levels),
        "dynamics_strength": _summarize_array(dynamics_strength),
        "mass_scale": _summarize_array(scales(0x4001, 0.35)),
        "gravity_scale": _summarize_array(scales(0x4003, 0.05)),
        "thrust_scale": _summarize_array(scales(0x4005, 0.35)),
        "arm_length_scale": _summarize_array(scales(0x4007, 0.12)),
        "yaw_torque_scale": _summarize_array(scales(0x4009, 0.18)),
        "linear_drag_scale": _summarize_array(scales(0x4011, 0.45)),
        "angular_drag_scale": _summarize_array(scales(0x4013, 0.45)),
        "motor_response_scale": _summarize_array(scales(0x4015, 0.35)),
        "motor_gain_scale": _summarize_array(motor_scales.reshape(-1)),
        "spawn_lateral_offset_m": _summarize_array(lateral_spawn_offset),
        "spawn_yaw_offset_rad": _summarize_array(yaw_offset),
    }


def _neutral_actions(env_count: int, action_dim: int) -> np.ndarray:
    return np.full((env_count, action_dim), 0.5, dtype=np.float32)


def _critic_context_features(
    previous_actions: np.ndarray,
    *,
    step_fraction: float,
    phase_progress: float,
) -> np.ndarray:
    previous = np.asarray(previous_actions, dtype=np.float32)
    env_count = previous.shape[0]
    step = np.full((env_count, 1), float(step_fraction), dtype=np.float32)
    remaining = np.full((env_count, 1), 1.0 - float(step_fraction), dtype=np.float32)
    phase = np.full((env_count, 1), float(phase_progress), dtype=np.float32)
    mean_action = previous.mean(axis=1, keepdims=True)
    action_energy = np.mean(np.square(previous - 0.5), axis=1, keepdims=True)
    return np.concatenate(
        (previous, step, remaining, phase, mean_action, action_energy),
        axis=1,
    )


def _build_critic_observations(
    normalized_observations: np.ndarray,
    previous_actions: np.ndarray,
    *,
    step_fraction: float,
    phase_progress: float,
) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(normalized_observations, dtype=np.float32),
            _critic_context_features(
                previous_actions,
                step_fraction=step_fraction,
                phase_progress=phase_progress,
            ),
        ),
        axis=1,
    )


def _allocate_rollout_arrays(
    config: PPOConfig,
    env_count: int,
    obs_dim: int,
    action_dim: int,
    critic_obs_dim: int,
):
    return {
        "observations": np.empty(
            (config.horizon, env_count, obs_dim), dtype=np.float32
        ),
        "critic_observations": np.empty(
            (config.horizon, env_count, critic_obs_dim), dtype=np.float32
        ),
        "previous_actions": np.empty(
            (config.horizon, env_count, action_dim), dtype=np.float32
        ),
        "action_deltas": np.empty(
            (config.horizon, env_count, action_dim), dtype=np.float32
        ),
        "action_smoothness_penalties": np.empty(
            (config.horizon, env_count), dtype=np.float32
        ),
        "actions": np.empty((config.horizon, env_count, action_dim), dtype=np.float32),
        "log_probs": np.empty((config.horizon, env_count), dtype=np.float32),
        "rewards": np.empty((config.horizon, env_count), dtype=np.float32),
        "dones": np.empty((config.horizon, env_count), dtype=np.float32),
        "values": np.empty((config.horizon, env_count), dtype=np.float32),
        "advantages": np.empty((config.horizon, env_count), dtype=np.float32),
        "returns": np.empty((config.horizon, env_count), dtype=np.float32),
    }


class RunningMeanStd:
    def __init__(
        self,
        shape: int | tuple[int, ...],
        *,
        mean: np.ndarray | None = None,
        var: np.ndarray | None = None,
        count: float = 1.0e-4,
        clip: float = 10.0,
    ):
        resolved_shape = (shape,) if isinstance(shape, int) else shape
        self.mean = (
            np.zeros(resolved_shape, dtype=np.float64)
            if mean is None
            else np.asarray(mean, dtype=np.float64)
        )
        self.var = (
            np.ones(resolved_shape, dtype=np.float64)
            if var is None
            else np.asarray(var, dtype=np.float64)
        )
        self.count = float(count)
        self.clip = float(clip)

    def update(self, values: np.ndarray) -> None:
        batch = np.asarray(values, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch[None, :]
        if batch.shape[0] == 0:
            return
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = float(batch.shape[0])
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * (batch_count / total_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        correction = np.square(delta) * (self.count * batch_count / total_count)
        new_var = (m_a + m_b + correction) / total_count

        self.mean = new_mean
        self.var = np.maximum(new_var, 1.0e-6)
        self.count = total_count

    def normalize(self, values: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(values, dtype=np.float32) - self.mean) / np.sqrt(
            self.var + 1.0e-8
        )
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean.astype(np.float32),
            "var": self.var.astype(np.float32),
            "count": float(self.count),
            "clip": float(self.clip),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "RunningMeanStd":
        mean = np.asarray(state["mean"], dtype=np.float64)
        return cls(
            shape=tuple(mean.shape),
            mean=mean,
            var=np.asarray(state["var"], dtype=np.float64),
            count=float(state.get("count", 1.0e-4)),
            clip=float(state.get("clip", 10.0)),
        )


class EpisodeTracker:
    def __init__(self, env_count: int):
        self.current_returns = np.zeros(env_count, dtype=np.float32)
        self.current_lengths = np.zeros(env_count, dtype=np.int32)
        self.completed_returns: list[float] = []
        self.completed_lengths: list[int] = []

    def begin_rollout(self) -> None:
        self.completed_returns.clear()
        self.completed_lengths.clear()

    def record_step(self, rewards: np.ndarray, dones: np.ndarray) -> None:
        self.current_returns += rewards
        self.current_lengths += 1

        done_indices = np.flatnonzero(dones)
        if done_indices.size == 0:
            return

        self.completed_returns.extend(self.current_returns[done_indices].tolist())
        self.completed_lengths.extend(self.current_lengths[done_indices].tolist())
        self.current_returns[done_indices] = 0.0
        self.current_lengths[done_indices] = 0

    def rollout_stats(self) -> tuple[int, float, float]:
        if not self.completed_returns:
            return 0, 0.0, 0.0
        return (
            len(self.completed_returns),
            float(np.mean(self.completed_returns)),
            float(np.mean(self.completed_lengths)),
        )


RESET_START_GATE_SHIFT = 8
RESET_START_GATE_MASK = 0xFF
RESET_ADAPTIVE_NOISE_SHIFT = 16
RESET_SOFT_FAILURE_SHIFT = 20
RESET_CURRICULUM_BUCKETS = 15
RESET_PARAM_AXIS_FIELD_COUNT = 15
FAILURE_REPLAY_STRATEGY = "transition_into_failure_v1"
DEFAULT_TRAINING_GATE_COUNT = int(
    sum(int(stage["gate_count"]) for stage in DEFAULT_TRAINING_COURSE_SPEC["stages"])
)


def _legacy_axis_values(
    difficulty: float,
) -> tuple[float, float, float, float, float, float, float, float, float, float, int]:
    return (
        float(difficulty),
        float(difficulty),
        float(difficulty),
        float(difficulty),
        0.0,
        float(difficulty),
        float(difficulty),
        0.0,
        float(difficulty),
        0.0,
        0,
    )


def _expand_reset_param(
    params: Sequence[float],
) -> tuple[
    int,
    int,
    float,
    int,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    int,
]:
    if len(params) == 4:
        seed, grammar_id, difficulty, curriculum_stage = params
        return (
            int(seed),
            int(grammar_id),
            float(difficulty),
            int(curriculum_stage),
            *_legacy_axis_values(float(difficulty)),
        )
    if len(params) != RESET_PARAM_AXIS_FIELD_COUNT:
        raise ValueError(f"expected reset param length 4 or 15, got {len(params)}")
    (
        seed,
        grammar_id,
        difficulty,
        curriculum_stage,
        gate_count_level,
        gate_size_level,
        spacing_level,
        verticality_level,
        gate_pose_noise_level,
        spawn_noise_level,
        dynamics_noise_level,
        obstacle_density_level,
        path_curvature_level,
        soft_failure_level,
        start_gate,
    ) = params
    return (
        int(seed),
        int(grammar_id),
        float(difficulty),
        int(curriculum_stage),
        float(gate_count_level),
        float(gate_size_level),
        float(spacing_level),
        float(verticality_level),
        float(gate_pose_noise_level),
        float(spawn_noise_level),
        float(dynamics_noise_level),
        float(obstacle_density_level),
        float(path_curvature_level),
        float(soft_failure_level),
        int(start_gate),
    )


def _estimated_curriculum_gate_count(
    curriculum_stage: int,
    gate_count_level: float,
    *,
    base_gate_count: int = DEFAULT_TRAINING_GATE_COUNT,
) -> int:
    if curriculum_stage == int(CurriculumStage.BOOTSTRAP):
        return 2 if gate_count_level >= 0.015 and base_gate_count >= 2 else 1
    if curriculum_stage == int(CurriculumStage.INTRO):
        if gate_count_level < 0.02:
            return 2
        if gate_count_level < 0.04:
            return 3
        return 4
    if curriculum_stage == int(CurriculumStage.ARENA):
        if gate_count_level < 0.05:
            return 3
        if gate_count_level < 0.1:
            return 4
        if gate_count_level < 0.18:
            return 5
        return max(5, min(6, (base_gate_count + 1) // 2))
    if curriculum_stage == int(CurriculumStage.TECHNICAL):
        max_stage_count = min(max(base_gate_count, 1), 8)
        min_stage_count = min(6, max_stage_count)
        if gate_count_level < 0.45:
            return min_stage_count
        if gate_count_level < 0.65:
            return min(min_stage_count + 1, max_stage_count)
        return max_stage_count
    max_stage_count = max(base_gate_count, 1)
    min_stage_count = min(8, max_stage_count)
    scaled_count = int(
        (min_stage_count + (max_stage_count - min_stage_count) * gate_count_level)
        + 0.5
    )
    return max(min_stage_count, min(scaled_count, max_stage_count))


def _stage_adaptive_noise_cap(curriculum_stage: int) -> float:
    if curriculum_stage == int(CurriculumStage.BOOTSTRAP):
        return 0.03
    if curriculum_stage == int(CurriculumStage.INTRO):
        return 0.12
    if curriculum_stage == int(CurriculumStage.ARENA):
        return 0.28
    if curriculum_stage == int(CurriculumStage.TECHNICAL):
        return 0.42
    return 0.55


def _stage_soft_failure_cap(curriculum_stage: int) -> float:
    if curriculum_stage == int(CurriculumStage.BOOTSTRAP):
        return 1.0
    if curriculum_stage == int(CurriculumStage.INTRO):
        return 0.75
    if curriculum_stage == int(CurriculumStage.ARENA):
        return 0.35
    return 0.0


def _phase_soft_failure_cap(phase) -> float:
    stage_cap = _stage_soft_failure_cap(int(phase.curriculum_stage))
    if str(phase.name).startswith("circle_"):
        return min(stage_cap, 0.25)
    return stage_cap


def _quantize_curriculum_level(value: float) -> int:
    return int(
        max(
            0,
            min(
                RESET_CURRICULUM_BUCKETS,
                round(float(value) * RESET_CURRICULUM_BUCKETS),
            ),
        )
    )


def _encode_training_controls_in_grammar_id(
    grammar_id: int,
    *,
    start_gate: int,
    adaptive_noise_level: float,
    soft_failure_level: float,
) -> int:
    base_grammar_id = int(grammar_id) & 0xFF
    start_bucket = max(0, min(RESET_START_GATE_MASK, int(start_gate)))
    noise_bucket = _quantize_curriculum_level(adaptive_noise_level)
    soft_failure_bucket = _quantize_curriculum_level(soft_failure_level)
    return (
        base_grammar_id
        | (start_bucket << RESET_START_GATE_SHIFT)
        | (noise_bucket << RESET_ADAPTIVE_NOISE_SHIFT)
        | (soft_failure_bucket << RESET_SOFT_FAILURE_SHIFT)
    )


def _failure_replay_key(
    curriculum_stage: int, gate_count: int, grammar_id: int | str
) -> str:
    return f"{int(curriculum_stage)}:{int(gate_count)}:{grammar_id}"


def _failure_replay_direction_key(curriculum_stage: int, gate_count: int) -> str:
    return f"{int(curriculum_stage)}:{int(gate_count)}"


def _with_training_start_gate_resets(
    reset_params: list[tuple],
    *,
    base_seed: int,
    adaptive_noise_level: float,
    soft_failure_level: float,
    failure_replay_start_gates: dict[str, int] | None = None,
    failure_replay_grammar_ids: dict[str, int] | None = None,
) -> list[tuple]:
    encoded: list[tuple] = []
    for env_index, params in enumerate(reset_params):
        (
            seed,
            grammar_id,
            difficulty,
            curriculum_stage,
            gate_count_level,
            gate_size_level,
            spacing_level,
            verticality_level,
            gate_pose_noise_level,
            spawn_noise_level,
            dynamics_noise_level,
            obstacle_density_level,
            path_curvature_level,
            phase_soft_failure_level,
            _start_gate,
        ) = _expand_reset_param(params)
        curriculum_stage = int(curriculum_stage)
        gate_count = _estimated_curriculum_gate_count(
            curriculum_stage,
            float(gate_count_level),
        )
        env_rng = random.Random(
            int(seed) ^ int(base_seed) ^ (env_index * 0x9E37_79B9) ^ 0xA511_E9B3
        )
        if (
            failure_replay_grammar_ids
            and gate_count > 2
            and curriculum_stage == int(CurriculumStage.INTRO)
        ):
            weak_grammar_id = failure_replay_grammar_ids.get(
                _failure_replay_direction_key(curriculum_stage, gate_count)
            )
            if weak_grammar_id is not None and env_rng.random() < 0.35:
                grammar_id = int(weak_grammar_id)
        start_gate = 0
        if gate_count > 2 and curriculum_stage != int(CurriculumStage.BOOTSTRAP):
            start_gate_probability = 0.50
            if curriculum_stage == int(CurriculumStage.INTRO):
                start_gate_probability = 0.60
                if gate_count >= 4:
                    start_gate_probability = 0.70
            if env_rng.random() < start_gate_probability:
                if curriculum_stage == int(CurriculumStage.INTRO):
                    replay_roll = env_rng.random()
                    preferred_start_gate = None
                    if failure_replay_start_gates:
                        preferred_start_gate = failure_replay_start_gates.get(
                            _failure_replay_key(
                                curriculum_stage, gate_count, int(grammar_id) & 0xFF
                            )
                        )
                        if preferred_start_gate is None:
                            preferred_start_gate = failure_replay_start_gates.get(
                                _failure_replay_key(curriculum_stage, gate_count, "*")
                            )
                    if preferred_start_gate is not None and replay_roll < 0.75:
                        start_gate = max(
                            0, min(gate_count - 1, int(preferred_start_gate))
                        )
                    elif gate_count == 3:
                        start_gate = 1 if replay_roll < 0.75 else 0
                    elif gate_count >= 4:
                        if replay_roll < 0.60:
                            start_gate = max(0, gate_count - 2)
                        elif replay_roll < 0.85:
                            start_gate = max(0, gate_count - 3)
                        else:
                            start_gate = env_rng.randrange(0, gate_count)
                    else:
                        start_gate = env_rng.randrange(0, gate_count)
                else:
                    start_gate = env_rng.randrange(1, gate_count)
        encoded.append(
            (
                int(seed),
                int(grammar_id) & 0xFF,
                float(difficulty),
                curriculum_stage,
                float(gate_count_level),
                float(gate_size_level),
                float(spacing_level),
                float(verticality_level),
                min(
                    max(float(gate_pose_noise_level), float(adaptive_noise_level)),
                    _stage_adaptive_noise_cap(curriculum_stage),
                ),
                float(spawn_noise_level),
                float(dynamics_noise_level),
                float(obstacle_density_level),
                float(path_curvature_level),
                min(
                    max(float(phase_soft_failure_level), float(soft_failure_level)),
                    _stage_soft_failure_cap(curriculum_stage),
                ),
                int(start_gate),
            )
        )
    return encoded


class MasteryCurriculumController:
    def __init__(self, schedule, config: PPOConfig):
        self.schedule = schedule
        self.config = config
        self.current_phase_index = 0
        self.phase_update_count = 0
        self.adaptive_noise_level = float(
            max(
                config.adaptive_noise_min,
                min(config.adaptive_noise_max, config.adaptive_noise_initial),
            )
        )
        self.soft_failure_level = float(
            max(config.soft_failure_min, min(1.0, config.soft_failure_initial))
        )
        self.latest_eval_stats: CurriculumEvalStats | None = None
        self.failure_replay_start_gates: dict[str, int] = {}
        self.failure_replay_grammar_ids: dict[str, int] = {}
        self._history = {
            phase_index: deque(maxlen=config.curriculum_mastery_window)
            for phase_index in range(len(schedule.phases))
        }

    def current_phase(self):
        return self.schedule.phase_at_index(self.current_phase_index)

    def progress(self) -> float:
        if len(self.schedule.phases) <= 1:
            return 1.0
        return self.current_phase_index / float(len(self.schedule.phases) - 1)

    def note_update(self) -> int:
        self.phase_update_count += 1
        return self.phase_update_count

    def training_adaptive_noise_level(self) -> float:
        if not self.config.adaptive_noise_enabled:
            return 0.0
        phase_cap = _stage_adaptive_noise_cap(int(self.current_phase().curriculum_stage))
        return float(
            max(
                self.config.adaptive_noise_min,
                min(self.config.adaptive_noise_max, self.adaptive_noise_level, phase_cap),
            )
        )

    def training_soft_failure_level(self) -> float:
        if not self.config.soft_failure_curriculum_enabled:
            return 0.0
        phase_cap = _phase_soft_failure_cap(self.current_phase())
        return float(max(self.config.soft_failure_min, min(self.soft_failure_level, phase_cap)))

    def adaptive_curriculum_state(self) -> dict[str, float | bool]:
        return {
            "adaptive_noise_enabled": bool(self.config.adaptive_noise_enabled),
            "adaptive_noise_level": float(self.adaptive_noise_level),
            "training_noise_level": self.training_adaptive_noise_level(),
            "soft_failure_enabled": bool(self.config.soft_failure_curriculum_enabled),
            "soft_failure_level": float(self.soft_failure_level),
            "training_soft_failure_level": self.training_soft_failure_level(),
        }

    def record_rollout_feedback(self, done_reasons: DoneReasonSummary) -> None:
        rates = done_reasons.rates
        failure_rate = float(
            rates.get("missed_gate", 0.0)
            + rates.get("gate_collision", 0.0)
            + rates.get("obstacle_collision", 0.0)
            + rates.get("floor_collision", 0.0)
            + rates.get("out_of_bounds", 0.0)
        )
        complete_rate = float(rates.get("complete", 0.0))

        if self.config.adaptive_noise_enabled and done_reasons.total_terminations > 0:
            if failure_rate > self.config.adaptive_noise_target_failure_rate:
                self.adaptive_noise_level -= self.config.adaptive_noise_step_down
            elif (
                failure_rate < self.config.adaptive_noise_target_failure_rate * 0.5
                and complete_rate > 0.05
            ):
                self.adaptive_noise_level += self.config.adaptive_noise_step_up
            self.adaptive_noise_level = float(
                max(
                    self.config.adaptive_noise_min,
                    min(self.config.adaptive_noise_max, self.adaptive_noise_level),
                )
            )

        if (
            self.config.soft_failure_curriculum_enabled
            and done_reasons.total_terminations > 0
        ):
            if failure_rate > self.config.soft_failure_target_failure_rate:
                self.soft_failure_level += self.config.soft_failure_step_up
            elif complete_rate > 0.05 or failure_rate < self.config.soft_failure_target_failure_rate:
                self.soft_failure_level -= self.config.soft_failure_step_down
            self.soft_failure_level = float(
                max(self.config.soft_failure_min, min(1.0, self.soft_failure_level))
            )

    def training_phase_mix(self) -> dict[str, float]:
        if self.current_phase_index <= 0:
            return {self.current_phase().name: 1.0}

        if self.current_phase_index == 1:
            return {
                self.schedule.phase_at_index(0).name: 0.45,
                self.schedule.phase_at_index(1).name: 0.55,
            }

        raw_mix: dict[int, float] = {}
        current_weight = self.config.curriculum_current_weight
        previous_weight = self.config.curriculum_previous_weight
        easy_weight = self.config.curriculum_easy_weight
        named_rehearsal: dict[str, float] = {}
        if self.current_phase().name == "straight_chain":
            current_weight = 0.80
            previous_weight = 0.15
            easy_weight = 0.05
        if self.current_phase().name == "circle_intro":
            current_weight = 0.85
            previous_weight = 0.10
            easy_weight = 0.05
        if self.current_phase().name in {"circle_mastery", "circle_chain"}:
            current_weight = 0.75
            previous_weight = 0.10
            easy_weight = 0.05
            named_rehearsal["straight_chain"] = 0.10

        def add_weight(phase_index: int, weight: float) -> None:
            if phase_index < 0 or weight <= 0.0:
                return
            raw_mix[phase_index] = raw_mix.get(phase_index, 0.0) + weight

        def add_phase_weight(phase_name: str, weight: float) -> None:
            for phase_index, phase in enumerate(self.schedule.phases):
                if phase.name == phase_name:
                    add_weight(phase_index, weight)
                    return

        add_weight(self.current_phase_index, current_weight)
        add_weight(self.current_phase_index - 1, previous_weight)
        add_weight(0, easy_weight)
        for phase_name, weight in named_rehearsal.items():
            add_phase_weight(phase_name, weight)

        total = sum(raw_mix.values())
        if total <= 0.0:
            return {self.current_phase().name: 1.0}

        return {
            self.schedule.phase_at_index(phase_index).name: weight / total
            for phase_index, weight in sorted(raw_mix.items())
        }

    def sample_training_reset_params(
        self, env_count: int, base_seed: int
    ) -> list[tuple]:
        if self.current_phase_index <= 0:
            reset_params = self.schedule.sample_reset_params_for_phase(
                env_count=env_count,
                phase_index=0,
                base_seed=base_seed,
            )
            return _with_training_start_gate_resets(
                reset_params,
                base_seed=base_seed,
                adaptive_noise_level=self.training_adaptive_noise_level(),
                soft_failure_level=self.training_soft_failure_level(),
                failure_replay_start_gates=self.failure_replay_start_gates,
                failure_replay_grammar_ids=self.failure_replay_grammar_ids,
            )

        phase_mix = self.training_phase_mix()
        phase_indices = [
            phase_index
            for phase_index, phase in enumerate(self.schedule.phases)
            if phase.name in phase_mix
        ]
        weights = [
            phase_mix[self.schedule.phase_at_index(i).name] for i in phase_indices
        ]
        reset_params = self.schedule.sample_reset_params_mixture(
            env_count=env_count,
            phase_indices=phase_indices,
            weights=weights,
            base_seed=base_seed,
        )
        return _with_training_start_gate_resets(
            reset_params,
            base_seed=base_seed,
            adaptive_noise_level=self.training_adaptive_noise_level(),
            soft_failure_level=self.training_soft_failure_level(),
            failure_replay_start_gates=self.failure_replay_start_gates,
            failure_replay_grammar_ids=self.failure_replay_grammar_ids,
        )

    def should_evaluate(self, update_index: int, total_updates: int) -> bool:
        if self.config.curriculum_eval_interval <= 0:
            return update_index == total_updates - 1
        return (
            (update_index + 1) % self.config.curriculum_eval_interval == 0
            or update_index == total_updates - 1
        )

    def _mastery_thresholds(self, phase_name: str) -> tuple[float, float, float]:
        early_thresholds = {
            "gate_approach": (0.70, 0.70, 0.70),
            "gate_pass": (0.75, 0.85, 0.80),
            "straight_intro": (0.70, 0.85, 0.75),
            "straight_chain": (0.70, 0.88, 0.82),
            "circle_intro": (0.50, 0.72, 0.65),
            "circle_mastery": (0.60, 0.82, 0.75),
            "circle_chain": (0.62, 0.84, 0.78),
        }
        return early_thresholds.get(
            phase_name,
            (
                self.config.curriculum_completion_threshold,
                self.config.curriculum_gate_pass_threshold,
                self.config.curriculum_progress_threshold,
            ),
        )

    def _update_failure_replay(self, eval_stats: CurriculumEvalStats) -> None:
        gate_metrics = eval_stats.gate_metrics
        if not isinstance(gate_metrics, dict):
            return
        gate_count_distribution = gate_metrics.get("gate_count_distribution")
        if not isinstance(gate_count_distribution, dict) or not gate_count_distribution:
            return
        gate_count_text, _ = max(
            gate_count_distribution.items(), key=lambda item: int(item[1])
        )
        try:
            gate_count = int(gate_count_text)
        except ValueError:
            return
        if gate_count <= 2:
            return

        phase = self.schedule.phase_at_index(eval_stats.phase_index)
        curriculum_stage = int(phase.curriculum_stage)

        failure_counts = gate_metrics.get("failure_counts")
        if isinstance(failure_counts, dict) and failure_counts:
            gate_key, _ = max(failure_counts.items(), key=lambda item: int(item[1]))
            failed_gate = _gate_number_from_metric_key(gate_key)
            if failed_gate is not None and failed_gate > 1:
                self.failure_replay_start_gates[
                    _failure_replay_key(curriculum_stage, gate_count, "*")
                ] = max(0, min(gate_count - 1, failed_gate - 2))

        direction_metrics = gate_metrics.get("direction_metrics")
        if isinstance(direction_metrics, dict) and len(direction_metrics) >= 2:
            candidates: list[tuple[int, float, int]] = []
            for grammar_id_text, metrics in direction_metrics.items():
                if not isinstance(metrics, dict):
                    continue
                try:
                    grammar_id = int(grammar_id_text)
                    completion_rate = float(metrics.get("completion_rate", 0.0))
                    episodes = int(metrics.get("episodes", 0))
                except (TypeError, ValueError):
                    continue
                if episodes > 0:
                    candidates.append((grammar_id, completion_rate, episodes))
            if len(candidates) >= 2:
                weakest = min(candidates, key=lambda item: item[1])
                strongest = max(candidates, key=lambda item: item[1])
                if strongest[1] - weakest[1] >= 0.20:
                    self.failure_replay_grammar_ids[
                        _failure_replay_direction_key(curriculum_stage, gate_count)
                    ] = weakest[0]

        failures_by_direction = gate_metrics.get("failure_counts_by_direction")
        if not isinstance(failures_by_direction, dict):
            return
        for grammar_id_text, direction_failures in failures_by_direction.items():
            if not isinstance(direction_failures, dict) or not direction_failures:
                continue
            gate_key, _ = max(
                direction_failures.items(), key=lambda item: int(item[1])
            )
            failed_gate = _gate_number_from_metric_key(gate_key)
            if failed_gate is None or failed_gate <= 1:
                continue
            self.failure_replay_start_gates[
                _failure_replay_key(curriculum_stage, gate_count, grammar_id_text)
            ] = max(0, min(gate_count - 1, failed_gate - 2))

    def record_eval(self, eval_stats: CurriculumEvalStats, update_index: int) -> bool:
        self.latest_eval_stats = eval_stats
        self._update_failure_replay(eval_stats)
        history = self._history[eval_stats.phase_index]
        history.append(eval_stats)

        if self.current_phase_index >= len(self.schedule.phases) - 1:
            return False
        if update_index < self.config.warmup_updates:
            return False
        if self.phase_update_count < self.config.curriculum_min_stage_updates:
            return False
        if len(history) < self.config.curriculum_mastery_window:
            return False

        mean_completion = float(np.mean([item.completion_rate for item in history]))
        mean_gate_pass = float(np.mean([item.gate_pass_rate for item in history]))
        mean_progress = float(np.mean([item.mean_progress for item in history]))
        completion_threshold, gate_pass_threshold, progress_threshold = (
            self._mastery_thresholds(eval_stats.phase)
        )
        if (
            mean_completion < completion_threshold
            or mean_gate_pass < gate_pass_threshold
            or mean_progress < progress_threshold
        ):
            return False

        self.current_phase_index += 1
        self.phase_update_count = 0
        self.soft_failure_level = min(
            self.soft_failure_level,
            _stage_soft_failure_cap(int(self.current_phase().curriculum_stage)),
        )
        self.adaptive_noise_level = min(
            self.adaptive_noise_level,
            _stage_adaptive_noise_cap(int(self.current_phase().curriculum_stage)),
        )
        return True

    def state_dict(self) -> dict[str, object]:
        return {
            "current_phase_index": int(self.current_phase_index),
            "phase_update_count": int(self.phase_update_count),
            "adaptive_noise_level": float(self.adaptive_noise_level),
            "soft_failure_level": float(self.soft_failure_level),
            "latest_eval_stats": None
            if self.latest_eval_stats is None
            else asdict(self.latest_eval_stats),
            "failure_replay_strategy": FAILURE_REPLAY_STRATEGY,
            "failure_replay_start_gates": dict(self.failure_replay_start_gates),
            "failure_replay_grammar_ids": dict(self.failure_replay_grammar_ids),
            "history": {
                str(phase_index): [asdict(item) for item in history]
                for phase_index, history in self._history.items()
            },
        }

    def load_state_dict(self, state: dict[str, object] | None) -> None:
        if not state:
            return

        self.current_phase_index = int(
            max(
                0,
                min(
                    int(state.get("current_phase_index", 0)),
                    len(self.schedule.phases) - 1,
                ),
            )
        )
        self.phase_update_count = int(max(0, int(state.get("phase_update_count", 0))))
        self.adaptive_noise_level = float(
            max(
                self.config.adaptive_noise_min,
                min(
                    self.config.adaptive_noise_max,
                    float(state.get("adaptive_noise_level", self.adaptive_noise_level)),
                ),
            )
        )
        self.soft_failure_level = float(
            max(
                self.config.soft_failure_min,
                min(1.0, float(state.get("soft_failure_level", self.soft_failure_level))),
            )
        )

        self.latest_eval_stats = None
        latest_eval_stats = state.get("latest_eval_stats")
        if isinstance(latest_eval_stats, dict):
            self.latest_eval_stats = _curriculum_eval_stats_from_dict(latest_eval_stats)

        if state.get("failure_replay_strategy") == FAILURE_REPLAY_STRATEGY:
            replay_state = state.get("failure_replay_start_gates")
            if isinstance(replay_state, dict):
                self.failure_replay_start_gates = {
                    str(key): int(value) for key, value in replay_state.items()
                }
            replay_grammar_state = state.get("failure_replay_grammar_ids")
            if isinstance(replay_grammar_state, dict):
                self.failure_replay_grammar_ids = {
                    str(key): int(value) for key, value in replay_grammar_state.items()
                }
        elif self.latest_eval_stats is not None:
            self._update_failure_replay(self.latest_eval_stats)

        self._history = {
            phase_index: deque(maxlen=self.config.curriculum_mastery_window)
            for phase_index in range(len(self.schedule.phases))
        }
        history_payload = state.get("history")
        if not isinstance(history_payload, dict):
            return

        for phase_index_text, values in history_payload.items():
            try:
                phase_index = int(phase_index_text)
            except (TypeError, ValueError):
                continue
            history = self._history.get(phase_index)
            if history is None or not isinstance(values, list):
                continue
            for item in values:
                if isinstance(item, dict):
                    history.append(_curriculum_eval_stats_from_dict(item))


COMPLETE_DONE_REASON = 1 << 0
PROGRESS_OBSERVATION_INDEX = 18
POSITION_OBSERVATION_SLICE = slice(0, 3)
TARGET_GATE_POSITION_OBSERVATION_SLICE = slice(12, 15)


def _select_pretrain_rollouts(
    obs_batch: np.ndarray,
    completed: np.ndarray,
    config: PPOConfig,
) -> np.ndarray:
    position = obs_batch[:, :, POSITION_OBSERVATION_SLICE]
    target_gate_position = obs_batch[:, :, TARGET_GATE_POSITION_OBSERVATION_SLICE]
    min_gate_distance = np.linalg.norm(target_gate_position - position, axis=2).min(
        axis=0
    )
    selected = completed | (min_gate_distance <= config.pretrain_max_gate_distance)
    if np.any(selected):
        return selected

    keep_count = max(1, obs_batch.shape[1] // 4)
    best_indices = np.argsort(min_gate_distance)[:keep_count]
    selected = np.zeros(obs_batch.shape[1], dtype=bool)
    selected[best_indices] = True
    return selected


def _done_reason_labels(done_reason: int) -> list[str]:
    labels = [label for bit, label in DONE_REASON_BITS if done_reason & int(bit) != 0]
    return labels or ["none"]


def _gate_metric_key(gate_number: int) -> str:
    return f"gate_{max(1, int(gate_number))}"


def _gate_number_from_metric_key(gate_key: object) -> int | None:
    text = str(gate_key)
    if not text.startswith("gate_"):
        return None
    try:
        return int(text.removeprefix("gate_"))
    except ValueError:
        return None


def _counter_to_gate_dict(counter: Counter[int]) -> dict[str, int]:
    return {
        _gate_metric_key(gate_number): int(count)
        for gate_number, count in sorted(counter.items())
        if int(count) > 0
    }


def _summarize_gate_steps(
    samples_by_gate: dict[int, list[int]],
) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for gate_number, samples in sorted(samples_by_gate.items()):
        if not samples:
            continue
        values = np.asarray(samples, dtype=np.float32)
        summary[_gate_metric_key(gate_number)] = {
            "count": int(values.size),
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50.0)),
            "p90": float(np.percentile(values, 90.0)),
        }
    return summary


def _target_gate_number(current_gate: int, gate_count: int) -> int:
    if gate_count <= 0:
        return 1
    return max(1, min(int(current_gate) + 1, int(gate_count)))


def _current_gates_from_progress(
    progress_values: np.ndarray, gate_counts: np.ndarray
) -> np.ndarray:
    scaled = np.asarray(progress_values, dtype=np.float32) * np.asarray(
        gate_counts, dtype=np.float32
    )
    return np.clip(
        np.floor(scaled + 1.0e-4).astype(np.int32),
        0,
        np.asarray(gate_counts, dtype=np.int32),
    )


def _evaluate_curriculum_phase(
    env: TriadFastVecEnv,
    model: ActorCritic,
    *,
    device: str,
    observation_normalizer: RunningMeanStd | None,
    schedule,
    phase_index: int,
    base_seed: int,
    max_episode_steps: int,
    include_per_env: bool = False,
) -> CurriculumEvalStats:
    torch_module, _ = _require_torch()
    reset_params = schedule.sample_reset_params_for_phase(
        env_count=env.sim.env_count,
        phase_index=phase_index,
        base_seed=base_seed,
    )
    expanded_reset_params = [_expand_reset_param(item) for item in reset_params]
    grammar_ids = np.asarray(
        [int(item[1]) & 0xFF for item in expanded_reset_params], dtype=np.int32
    )
    gate_counts = np.asarray(
        [
            _estimated_curriculum_gate_count(int(item[3]), float(item[4]))
            for item in expanded_reset_params
        ],
        dtype=np.int32,
    )
    env.sim.set_reset_params(reset_params)
    observations = env.reset_numpy().copy()

    active = np.ones(env.sim.env_count, dtype=bool)
    completed = np.zeros(env.sim.env_count, dtype=bool)
    returns = np.zeros(env.sim.env_count, dtype=np.float32)
    lengths = np.zeros(env.sim.env_count, dtype=np.int32)
    best_progress = observations[:, PROGRESS_OBSERVATION_INDEX].copy()
    last_gate = _current_gates_from_progress(
        observations[:, PROGRESS_OBSERVATION_INDEX], gate_counts
    )
    gate_entry_steps = np.zeros(env.sim.env_count, dtype=np.int32)
    terminal_recorded = np.zeros(env.sim.env_count, dtype=bool)
    gate_attempt_counts: Counter[int] = Counter()
    gate_success_counts: Counter[int] = Counter()
    gate_failure_counts: Counter[int] = Counter()
    gate_completion_counts: Counter[int] = Counter()
    gate_success_steps: dict[int, list[int]] = defaultdict(list)
    gate_failure_steps: dict[int, list[int]] = defaultdict(list)
    failure_reason_by_gate: dict[int, Counter[str]] = defaultdict(Counter)
    failure_gate_by_direction: dict[int, Counter[int]] = defaultdict(Counter)
    for env_index in range(env.sim.env_count):
        if int(last_gate[env_index]) < int(gate_counts[env_index]):
            gate_attempt_counts[
                _target_gate_number(last_gate[env_index], gate_counts[env_index])
            ] += 1

    for _ in range(max_episode_steps):
        obs_tensor = _observations_to_tensor(
            observations,
            device=device,
            observation_normalizer=observation_normalizer,
        )
        with torch_module.no_grad():
            actions, _ = model.act_deterministic(obs_tensor)

        action_values = actions.detach().cpu().numpy()
        action_values[~active, 0] = 0.0
        action_values[~active, 1:] = 0.5
        env.numpy_action_view()[:, :] = action_values

        step_result = env.step_in_place()
        next_observations, rewards, dones = step_result.numpy_views()
        done_flags = dones.astype(bool, copy=False)
        done_reasons = step_result.numpy_done_reasons()

        returns[active] += rewards[active]
        lengths[active] += 1
        best_progress = np.maximum(
            best_progress,
            next_observations[:, PROGRESS_OBSERVATION_INDEX],
        )
        current_gates = _current_gates_from_progress(
            next_observations[:, PROGRESS_OBSERVATION_INDEX], gate_counts
        )
        for env_index in np.flatnonzero(active):
            previous_gate = int(last_gate[env_index])
            current_gate = int(current_gates[env_index])
            gate_count = int(gate_counts[env_index])
            if current_gate <= previous_gate:
                continue
            passed_gate_number = max(1, min(current_gate, gate_count))
            gate_success_counts[passed_gate_number] += 1
            gate_success_steps[passed_gate_number].append(
                max(1, int(lengths[env_index]) - int(gate_entry_steps[env_index]))
            )
            gate_entry_steps[env_index] = int(lengths[env_index])
            last_gate[env_index] = min(current_gate, gate_count)
            if current_gate < gate_count:
                gate_attempt_counts[current_gate + 1] += 1

        newly_done = active & done_flags
        if np.any(newly_done):
            completed[newly_done] = (
                done_reasons[newly_done] & np.uint32(COMPLETE_DONE_REASON)
            ) != 0
            for env_index in np.flatnonzero(newly_done):
                terminal_recorded[env_index] = True
                gate_count = int(gate_counts[env_index])
                if bool(completed[env_index]):
                    gate_completion_counts[gate_count] += 1
                    continue
                target_gate = _target_gate_number(
                    current_gates[env_index], gate_count
                )
                gate_failure_counts[target_gate] += 1
                gate_failure_steps[target_gate].append(
                    max(1, int(lengths[env_index]) - int(gate_entry_steps[env_index]))
                )
                labels = _done_reason_labels(int(done_reasons[env_index]))
                for label in labels:
                    failure_reason_by_gate[target_gate][label] += 1
                failure_gate_by_direction[int(grammar_ids[env_index])][
                    target_gate
                ] += 1
            active[newly_done] = False

        observations = next_observations.copy()
        if not np.any(active):
            break

    phase = schedule.phase_at_index(phase_index)
    final_states = env.sim.get_state()
    final_reward_done = env.sim.get_reward_done()
    for env_index in np.flatnonzero(~terminal_recorded):
        gate_count = int(gate_counts[env_index])
        current_gate = int(final_states[env_index]["current_gate"])
        if bool(completed[env_index]):
            gate_completion_counts[gate_count] += 1
            continue
        target_gate = _target_gate_number(current_gate, gate_count)
        gate_failure_counts[target_gate] += 1
        gate_failure_steps[target_gate].append(
            max(1, int(lengths[env_index]) - int(gate_entry_steps[env_index]))
        )
        reason = int(final_reward_done[env_index]["done_reason"])
        labels = _done_reason_labels(reason)
        if labels == ["none"]:
            labels = ["eval_timeout"]
        for label in labels:
            failure_reason_by_gate[target_gate][label] += 1
        failure_gate_by_direction[int(grammar_ids[env_index])][target_gate] += 1
    gate_passed = np.asarray(
        [
            int(final_states[env_index]["current_gate"]) > 0
            or (
                int(final_reward_done[env_index]["done_reason"])
                & int(COMPLETE_DONE_REASON)
            )
            != 0
            for env_index in range(env.sim.env_count)
        ],
        dtype=bool,
    )
    per_env = None
    if include_per_env:
        final_observations = env.sim.get_observations()
        per_env = [
            CurriculumEvalEnvResult(
                env_index=env_index,
                seed=int(seed),
                grammar_id=int(grammar_id),
                difficulty=float(difficulty),
                curriculum_stage=int(curriculum_stage),
                completed=bool(completed[env_index]),
                done=bool(final_states[env_index]["done"]),
                done_reason=int(final_reward_done[env_index]["done_reason"]),
                done_reason_labels=_done_reason_labels(
                    int(final_reward_done[env_index]["done_reason"])
                ),
                episode_return=float(returns[env_index]),
                episode_length=int(lengths[env_index]),
                best_progress=float(best_progress[env_index]),
                final_progress=float(final_observations[env_index]["progress"]),
                current_gate=int(final_states[env_index]["current_gate"]),
                current_lap=int(final_states[env_index]["current_lap"]),
                step_count=int(final_states[env_index]["step_count"]),
                position=list(final_states[env_index]["position"]),
                velocity=list(final_states[env_index]["velocity"]),
                attitude=list(final_states[env_index]["attitude"]),
                target_gate_position=list(
                    final_observations[env_index]["target_gate_position"]
                ),
                distance_to_gate=float(
                    final_observations[env_index]["distance_to_gate"]
                ),
                gate_alignment=float(final_observations[env_index]["gate_alignment"]),
                reward=float(final_reward_done[env_index]["reward"]),
                shaping_reward=float(final_reward_done[env_index]["shaping_reward"]),
                out_of_bounds_penalty=float(
                    final_reward_done[env_index]["out_of_bounds_penalty"]
                ),
                time_penalty=float(final_reward_done[env_index]["time_penalty"]),
                sparse_objective_reward=float(
                    final_reward_done[env_index]["sparse_objective_reward"]
                ),
                collision_penalty=float(
                    final_reward_done[env_index]["collision_penalty"]
                ),
            )
            for env_index, (
                seed,
                grammar_id,
                difficulty,
                curriculum_stage,
                *_axis_values,
            ) in enumerate(expanded_reset_params)
        ]
    direction_metrics = {}
    for grammar_id in sorted(set(int(item) for item in grammar_ids.tolist())):
        direction_mask = grammar_ids == grammar_id
        direction_metrics[str(grammar_id)] = {
            "episodes": int(np.count_nonzero(direction_mask)),
            "completion_rate": float(np.mean(completed[direction_mask]))
            if np.any(direction_mask)
            else 0.0,
            "mean_progress": float(np.mean(best_progress[direction_mask]))
            if np.any(direction_mask)
            else 0.0,
        }
    gate_metrics: dict[str, object] = {
        "gate_count_distribution": {
            str(gate_count): int(count)
            for gate_count, count in sorted(Counter(gate_counts.tolist()).items())
        },
        "attempt_counts": _counter_to_gate_dict(gate_attempt_counts),
        "success_counts": _counter_to_gate_dict(gate_success_counts),
        "failure_counts": _counter_to_gate_dict(gate_failure_counts),
        "completion_counts": _counter_to_gate_dict(gate_completion_counts),
        "success_steps": _summarize_gate_steps(gate_success_steps),
        "failure_steps": _summarize_gate_steps(gate_failure_steps),
        "failure_reasons_by_gate": {
            _gate_metric_key(gate_number): {
                reason: int(count) for reason, count in sorted(reason_counts.items())
            }
            for gate_number, reason_counts in sorted(failure_reason_by_gate.items())
        },
        "failure_counts_by_direction": {
            str(grammar_id): _counter_to_gate_dict(counter)
            for grammar_id, counter in sorted(failure_gate_by_direction.items())
        },
        "direction_metrics": direction_metrics,
    }
    return CurriculumEvalStats(
        phase_index=phase_index,
        phase=phase.name,
        completion_rate=float(np.mean(completed)),
        gate_pass_rate=float(np.mean(gate_passed)),
        mean_progress=float(np.mean(best_progress)),
        mean_episode_return=float(np.mean(returns)),
        mean_episode_length=float(np.mean(lengths)),
        per_env=per_env,
        gate_metrics=gate_metrics,
    )


def _observations_to_tensor(
    observations: np.ndarray,
    *,
    device: str,
    observation_normalizer: RunningMeanStd | None,
):
    torch_module, _ = _require_torch()
    source = observations
    if observation_normalizer is not None:
        source = observation_normalizer.normalize(source)
    return torch_module.as_tensor(source, dtype=torch_module.float32, device=device)


def _collect_rollout(
    env: TriadFastVecEnv,
    model: ActorCritic,
    config: PPOConfig,
    device: str,
    observations: np.ndarray,
    previous_actions: np.ndarray,
    phase_progress: float,
    storage: dict[str, np.ndarray],
    observation_normalizer: RunningMeanStd | None,
    episode_tracker: EpisodeTracker,
    done_reason_tracker: DoneReasonTracker,
):
    torch_module, _ = _require_torch()

    for step_index in range(config.horizon):
        storage["observations"][step_index] = observations
        storage["previous_actions"][step_index] = previous_actions
        if observation_normalizer is not None:
            observation_normalizer.update(observations)
        normalized_observations = (
            observation_normalizer.normalize(observations)
            if observation_normalizer is not None
            else observations
        )
        critic_observations = _build_critic_observations(
            normalized_observations,
            previous_actions,
            step_fraction=step_index / max(config.horizon, 1),
            phase_progress=phase_progress,
        )
        storage["critic_observations"][step_index] = critic_observations
        obs_tensor = torch_module.as_tensor(
            normalized_observations, dtype=torch_module.float32, device=device
        )
        critic_obs_tensor = torch_module.as_tensor(
            critic_observations, dtype=torch_module.float32, device=device
        )
        with torch_module.no_grad():
            actions, log_probs, _, values = model.act(obs_tensor, critic_obs_tensor)

        action_values = actions.detach().cpu().numpy()
        action_deltas = action_values - previous_actions
        action_smoothness_penalties = (
            np.mean(np.square(action_deltas), axis=1).astype(np.float32)
            * float(config.action_smoothness_reward_scale)
        )
        env.numpy_action_view()[:, :] = action_values
        step_result = env.step_in_place()
        next_observations, rewards, dones = step_result.numpy_views()
        adjusted_rewards = rewards - action_smoothness_penalties

        storage["actions"][step_index] = action_values
        storage["action_deltas"][step_index] = action_deltas
        storage["action_smoothness_penalties"][step_index] = action_smoothness_penalties
        storage["log_probs"][step_index] = log_probs.detach().cpu().numpy()
        storage["values"][step_index] = values.detach().cpu().numpy()
        storage["rewards"][step_index] = adjusted_rewards
        done_flags = dones.astype(bool, copy=False)
        storage["dones"][step_index] = done_flags.astype(np.float32)
        episode_tracker.record_step(adjusted_rewards, done_flags)
        done_reason_tracker.record_step(done_flags, step_result.numpy_done_reasons())

        observations = next_observations.copy()
        previous_actions = action_values.copy()
        if np.any(done_flags):
            previous_actions[done_flags] = 0.5

    normalized_observations = (
        observation_normalizer.normalize(observations)
        if observation_normalizer is not None
        else observations
    )
    critic_observations = _build_critic_observations(
        normalized_observations,
        previous_actions,
        step_fraction=1.0,
        phase_progress=phase_progress,
    )
    obs_tensor = torch_module.as_tensor(
        normalized_observations, dtype=torch_module.float32, device=device
    )
    critic_obs_tensor = torch_module.as_tensor(
        critic_observations, dtype=torch_module.float32, device=device
    )
    with torch_module.no_grad():
        _, next_value = model.forward(obs_tensor, critic_obs_tensor)

    return observations, previous_actions, next_value.detach().cpu().numpy()


def _compute_gae(
    storage: dict[str, np.ndarray],
    next_value: np.ndarray,
    gamma: float,
    gae_lambda: float,
):
    last_advantage = np.zeros_like(next_value, dtype=np.float32)
    for step_index in range(storage["rewards"].shape[0] - 1, -1, -1):
        if step_index == storage["rewards"].shape[0] - 1:
            next_non_terminal = 1.0 - storage["dones"][step_index]
            next_values = next_value
        else:
            next_non_terminal = 1.0 - storage["dones"][step_index]
            next_values = storage["values"][step_index + 1]

        delta = (
            storage["rewards"][step_index]
            + gamma * next_values * next_non_terminal
            - storage["values"][step_index]
        )
        last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        storage["advantages"][step_index] = last_advantage

    storage["returns"][:, :] = storage["advantages"] + storage["values"]


def _flatten_rollout(storage: dict[str, np.ndarray]):
    return {
        key: value.reshape((-1,) + value.shape[2:])
        if value.ndim > 2
        else value.reshape(-1)
        for key, value in storage.items()
    }


def _explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = np.var(y_true)
    if var_y <= 1.0e-8:
        return 0.0
    return float(1.0 - np.var(y_true - y_pred) / var_y)


def _collect_pretrain_batch(
    env: TriadFastVecEnv,
    schedule,
    config: PPOConfig,
    *,
    base_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(schedule.phases) > 1 and config.pretrain_intro_weight > 0.0:
        reset_params = schedule.sample_reset_params_mixture(
            env_count=env.sim.env_count,
            phase_indices=[0, 1],
            weights=[
                max(config.pretrain_bootstrap_weight, 0.0),
                max(config.pretrain_intro_weight, 0.0),
            ],
            base_seed=base_seed,
        )
    else:
        reset_params = schedule.sample_reset_params_for_phase(
            env_count=env.sim.env_count,
            phase_index=0,
            base_seed=base_seed,
        )

    env.sim.set_reset_params(reset_params)
    observations = env.reset_numpy().copy()
    obs_batch = np.empty(
        (config.horizon, env.sim.env_count, env.sim.observation_stride),
        dtype=np.float32,
    )
    action_batch = np.empty(
        (config.horizon, env.sim.env_count, env.sim.action_stride),
        dtype=np.float32,
    )
    completed = np.zeros(env.sim.env_count, dtype=bool)

    for step_index in range(config.horizon):
        obs_batch[step_index] = observations
        teacher_actions = _teacher_point_to_gate_actions(observations)
        action_batch[step_index] = teacher_actions
        env.numpy_action_view()[:, :] = teacher_actions
        step_result = env.step_in_place()
        next_observations, _, dones = step_result.numpy_views()
        done_reasons = step_result.numpy_done_reasons()
        done_flags = dones.astype(bool, copy=False)
        completed |= done_flags & (
            (done_reasons & np.uint32(COMPLETE_DONE_REASON)) != 0
        )
        observations = next_observations.copy()

    selected = _select_pretrain_rollouts(obs_batch, completed, config)
    return (
        obs_batch[:, selected, :].reshape((-1, env.sim.observation_stride)),
        action_batch[:, selected, :].reshape((-1, env.sim.action_stride)),
    )


def _behavior_clone_pretrain(
    env: TriadFastVecEnv,
    model: "ActorCritic",
    optimizer,
    config: PPOConfig,
    *,
    device: str,
    observation_normalizer: RunningMeanStd | None,
    schedule,
) -> RunningMeanStd | None:
    torch_module, _ = _require_torch()
    if config.pretrain_updates <= 0:
        return observation_normalizer

    minibatch_size = max(1, int(config.pretrain_minibatch_size))
    for pretrain_update in range(config.pretrain_updates):
        flat_observations, flat_actions = _collect_pretrain_batch(
            env,
            schedule,
            config,
            base_seed=config.seed ^ (pretrain_update * 0x9E37_79B9),
        )
        if observation_normalizer is not None:
            observation_normalizer.update(flat_observations)
            flat_observations = observation_normalizer.normalize(flat_observations)

        obs_batch = torch_module.as_tensor(
            flat_observations, dtype=torch_module.float32, device=device
        )
        actions_batch = torch_module.as_tensor(
            flat_actions, dtype=torch_module.float32, device=device
        )
        batch_size = obs_batch.shape[0]

        for _ in range(config.pretrain_epochs):
            permutation = torch_module.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                indices = permutation[start : start + minibatch_size]
                predicted_mean, _ = model.forward(obs_batch[indices])
                predicted_actions = torch.sigmoid(predicted_mean)
                loss = torch_module.nn.functional.mse_loss(
                    predicted_actions,
                    actions_batch[indices],
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch_module.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm
                )
                optimizer.step()

    return observation_normalizer


def _save_checkpoint(
    checkpoint_path: Path,
    *,
    model: ActorCritic,
    optimizer,
    config: PPOConfig,
    task_spec: dict[str, object],
    observation_normalizer: RunningMeanStd | None,
    curriculum: MasteryCurriculumController,
    update_index: int,
    total_env_steps: int,
) -> None:
    torch_module, _ = _require_torch()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch_module.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "task_spec": _copy_json_value(task_spec),
            "action_space_version": ACTION_SPACE_VERSION,
            "observation_space_version": OBSERVATION_SPACE_VERSION,
            "observation_normalizer": None
            if observation_normalizer is None
            else observation_normalizer.state_dict(),
            "curriculum": curriculum.state_dict(),
            "update_index": int(update_index),
            "total_env_steps": int(total_env_steps),
        },
        checkpoint_path,
    )


def _upgrade_legacy_model_state(
    model_state: dict[str, object],
) -> dict[str, object]:
    if "actor_body.0.weight" in model_state:
        return model_state

    if "body.0.weight" not in model_state:
        return model_state

    upgraded = {
        "actor_body.0.weight": model_state["body.0.weight"],
        "actor_body.0.bias": model_state["body.0.bias"],
        "actor_body.2.weight": model_state["body.2.weight"],
        "actor_body.2.bias": model_state["body.2.bias"],
        "critic_body.0.weight": model_state["body.0.weight"],
        "critic_body.0.bias": model_state["body.0.bias"],
        "critic_body.2.weight": model_state["body.2.weight"],
        "critic_body.2.bias": model_state["body.2.bias"],
        "policy_mean.weight": model_state["policy_mean.weight"],
        "policy_mean.bias": model_state["policy_mean.bias"],
        "value_head.weight": model_state["value_head.weight"],
        "value_head.bias": model_state["value_head.bias"],
        "log_std": model_state["log_std"],
    }
    return upgraded


def _require_supported_action_space(
    checkpoint: dict[str, object], *, context: str
) -> None:
    action_space_version = checkpoint.get("action_space_version")
    if action_space_version == ACTION_SPACE_VERSION:
        return
    if action_space_version is None:
        raise ValueError(
            f"{context} checkpoint uses the legacy motor-action policy interface and "
            "cannot be loaded after the pilot-controls action-space change"
        )
    raise ValueError(
        f"{context} checkpoint action space {action_space_version!r} is not supported; "
        f"expected {ACTION_SPACE_VERSION!r}"
    )


def _require_supported_observation_space(
    checkpoint: dict[str, object], *, context: str
) -> None:
    observation_space_version = checkpoint.get("observation_space_version")
    if observation_space_version == OBSERVATION_SPACE_VERSION:
        return
    if observation_space_version is None:
        raise ValueError(
            f"{context} checkpoint uses the legacy teacher observation interface and "
            "cannot be loaded after the privileged-teacher observation change"
        )
    raise ValueError(
        f"{context} checkpoint observation space {observation_space_version!r} is not "
        f"supported; expected {OBSERVATION_SPACE_VERSION!r}"
    )


def load_ppo_policy(
    checkpoint_path: str | Path, device: str = "auto"
) -> LoadedPPOPolicy:
    torch_module, _ = _require_torch()
    resolved_device = _resolve_device(device)
    checkpoint = torch_module.load(
        Path(checkpoint_path), map_location=resolved_device, weights_only=False
    )
    _require_supported_action_space(checkpoint, context="PPO")
    _require_supported_observation_space(checkpoint, context="PPO")
    model_state = _upgrade_legacy_model_state(checkpoint["model"])
    observation_key = (
        "actor_body.0.weight"
        if "actor_body.0.weight" in model_state
        else "body.0.weight"
    )
    observation_dim = int(model_state[observation_key].shape[1])
    hidden_size = int(model_state[observation_key].shape[0])
    action_dim = int(model_state["policy_mean.weight"].shape[0])
    critic_observation_dim = int(
        model_state.get("critic_body.0.weight", model_state[observation_key]).shape[1]
    )
    model = ActorCritic(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        critic_observation_dim=critic_observation_dim,
    ).to(resolved_device)
    model.load_state_dict(model_state)
    model.eval()
    observation_normalizer_state = checkpoint.get("observation_normalizer")
    return LoadedPPOPolicy(
        model=model,
        device=resolved_device,
        config=dict(checkpoint.get("config", {})),
        observation_normalizer=None
        if observation_normalizer_state is None
        else RunningMeanStd.from_state_dict(observation_normalizer_state),
        task_spec=checkpoint.get("task_spec"),
    )


def _config_from_mapping(config_data: dict[str, object]) -> PPOConfig:
    valid_fields = {field.name for field in fields(PPOConfig)}
    overrides = {
        key: value for key, value in config_data.items() if key in valid_fields
    }
    return PPOConfig(**overrides)


def evaluate_ppo_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str = "auto",
    phase_index: int | None = None,
    eval_env_count: int | None = None,
    max_episode_steps: int | None = None,
    base_seed: int | None = None,
    per_env: bool = False,
) -> PPOEvalResult:
    policy = load_ppo_policy(checkpoint_path, device=device)
    torch_module, _ = _require_torch()
    checkpoint = torch_module.load(
        Path(checkpoint_path), map_location=policy.device, weights_only=False
    )
    config = _config_from_mapping(policy.config)
    if eval_env_count is not None:
        config.curriculum_eval_env_count = eval_env_count
    if max_episode_steps is not None:
        config.max_episode_steps = max_episode_steps

    schedule = build_teacher_curriculum_schedule()
    phase_indices: list[int]
    if phase_index is None:
        phase_indices = list(range(len(schedule.phases)))
    else:
        if phase_index < 0 or phase_index >= len(schedule.phases):
            raise ValueError(
                f"phase_index must be between 0 and {len(schedule.phases) - 1}, got {phase_index}"
            )
        phase_indices = [phase_index]

    eval_course = _default_training_course()
    eval_sim = SimulationCore(_evaluation_sim_config(eval_course, config))
    eval_sim.set_course(eval_course)
    eval_env = TriadFastVecEnv(eval_sim, auto_reset=False)
    resolved_eval_env_count = eval_sim.env_count

    resolved_base_seed = (
        config.curriculum_holdout_seed if base_seed is None else int(base_seed)
    )

    try:
        results = [
            _evaluate_curriculum_phase(
                eval_env,
                policy.model,
                device=policy.device,
                observation_normalizer=policy.observation_normalizer,
                schedule=schedule,
                phase_index=current_phase_index,
                base_seed=resolved_base_seed + current_phase_index * 104729,
                max_episode_steps=config.max_episode_steps,
                include_per_env=per_env,
            )
            for current_phase_index in phase_indices
        ]
    finally:
        eval_sim.close()
        eval_course.close()

    return PPOEvalResult(
        checkpoint_path=str(checkpoint_path),
        checkpoint_update_index=checkpoint.get("update_index"),
        checkpoint_total_env_steps=checkpoint.get("total_env_steps"),
        device=policy.device,
        eval_env_count=resolved_eval_env_count,
        max_episode_steps=config.max_episode_steps,
        per_env=per_env,
        task_name=None
        if policy.task_spec is None
        else str(policy.task_spec.get("task_name")),
        task_version=None
        if policy.task_spec is None
        else str(policy.task_spec.get("task_version")),
        task_hash=None
        if policy.task_spec is None
        else str(policy.task_spec.get("task_hash")),
        results=results,
    )


def _parse_text_policy_request(
    stripped: str,
) -> tuple[str, np.ndarray | None, bool, bool]:
    if stripped == "shutdown":
        return "shutdown", None, True, True

    tokens = stripped.split()
    if not tokens:
        raise ValueError("empty policy request")
    if tokens[0] != "predict":
        raise ValueError(f"unknown policy request kind: {tokens[0]}")
    if len(tokens) < 4:
        raise ValueError("predict request missing metadata")

    env_count = int(tokens[1])
    observation_dim = int(tokens[2])
    deterministic = bool(int(tokens[3]))
    expected_values = env_count * observation_dim
    raw_values = tokens[4:]
    if len(raw_values) != expected_values:
        raise ValueError(
            "predict request size mismatch: "
            f"expected {expected_values} values, got {len(raw_values)}"
        )

    observations = np.asarray(raw_values, dtype=np.float32).reshape(
        env_count, observation_dim
    )
    return "predict", observations, deterministic, True


def serve_ppo_policy(checkpoint_path: str | Path, device: str = "auto") -> int:
    policy = load_ppo_policy(checkpoint_path, device=device)
    sys.stdout.write("ready\n")
    sys.stdout.flush()
    for line in sys.stdin:
        stripped = line.strip()
        if not stripped:
            continue
        text_mode = not stripped.startswith("{")
        try:
            if text_mode:
                kind, observations, deterministic, _ = _parse_text_policy_request(stripped)
                if kind == "shutdown":
                    return 0
            else:
                request = json.loads(stripped)
                if request.get("kind") == "shutdown":
                    return 0
                observations = np.asarray(request["observations"], dtype=np.float32)
                deterministic = bool(request.get("deterministic", True))

            actions, values = policy.predict(observations, deterministic=deterministic)
            if text_mode:
                flat_actions = " ".join(
                    format(float(value), ".9g") for value in actions.reshape(-1)
                )
                response = (
                    f"ok {actions.shape[0]} {actions.shape[1]} {flat_actions}\n"
                    if flat_actions
                    else f"ok {actions.shape[0]} {actions.shape[1]}\n"
                )
                sys.stdout.write(response)
            else:
                response = {
                    "actions": actions.tolist(),
                    "values": values.tolist(),
                }
                sys.stdout.write(json.dumps(response) + "\n")
        except Exception as error:  # pragma: no cover - IPC guard
            if text_mode:
                message = " ".join(str(error).split())
                sys.stdout.write(f"error {message}\n")
            else:
                sys.stdout.write(json.dumps({"error": str(error)}) + "\n")

        sys.stdout.flush()
    return 0


def train_ppo(config: PPOConfig) -> PPOTrainResult:
    torch_module, _ = _require_torch()
    _set_global_seeds(config.seed)
    device = _resolve_device(config.device)

    schedule = build_teacher_curriculum_schedule()
    curriculum = MasteryCurriculumController(schedule, config)
    task_spec = _build_training_task_spec(config, schedule)

    course = _default_training_course()
    sim = SimulationCore(_training_sim_config(course, config))
    sim.set_course(course)
    env = TriadFastVecEnv(sim)
    eval_course = _default_training_course()
    eval_sim = SimulationCore(_evaluation_sim_config(eval_course, config))
    eval_sim.set_course(eval_course)
    eval_env = TriadFastVecEnv(eval_sim, auto_reset=False)

    checkpoint_path = _resolve_checkpoint_path(config)
    if checkpoint_path is not None:
        config.checkpoint_path = str(checkpoint_path)
    best_checkpoint_path = (
        None if checkpoint_path is None else _best_checkpoint_path(checkpoint_path)
    )
    resume_path = Path(config.resume_from) if config.resume_from else None
    if resume_path is not None:
        config.resume_from = str(resume_path)

    try:
        critic_observation_dim = (
            sim.observation_stride + sim.action_stride + CRITIC_CONTEXT_SCALAR_DIM
        )
        model = ActorCritic(
            observation_dim=sim.observation_stride,
            action_dim=sim.action_stride,
            hidden_size=config.hidden_size,
            critic_observation_dim=critic_observation_dim,
        ).to(device)
        optimizer = torch_module.optim.Adam(model.parameters(), lr=config.learning_rate)
        observation_normalizer = (
            RunningMeanStd(
                sim.observation_stride,
                clip=config.observation_clip,
            )
            if config.normalize_observations
            else None
        )
        episode_tracker = EpisodeTracker(sim.env_count)
        done_reason_tracker = DoneReasonTracker()
        tensorboard_writer, tensorboard_log_dir = _create_tensorboard_writer(config)
        training_logger = PPOTrainingLogger(
            asdict(config),
            json_output=config.json_log,
            pretty_output=config.pretty_log,
            tensorboard_writer=tensorboard_writer,
            tensorboard_log_dir=(
                None if tensorboard_log_dir is None else str(tensorboard_log_dir)
            ),
            task_metadata={
                "task_name": task_spec["task_name"],
                "task_version": task_spec["task_version"],
                "task_hash": task_spec["task_hash"],
            },
        )
        artifact_dir = _resolve_artifact_dir(
            config,
            tensorboard_log_dir=(
                None if tensorboard_log_dir is None else str(tensorboard_log_dir)
            ),
        )
        _save_run_artifacts(artifact_dir, config=config, task_spec=task_spec)
        start_update = 0
        total_env_steps = 0
        training_logger.emit_started(initial_phase=curriculum.current_phase().name)
        if resume_path is not None:
            training_logger.emit_status(
                f"loading checkpoint {resume_path}",
                resume_from=str(resume_path),
            )
            checkpoint = torch_module.load(
                resume_path, map_location=device, weights_only=False
            )
            _require_supported_action_space(checkpoint, context="resume")
            _require_supported_observation_space(checkpoint, context="resume")
            model.load_state_dict(_upgrade_legacy_model_state(checkpoint["model"]))
            optimizer_state = checkpoint.get("optimizer")
            if optimizer_state is not None and not config.resume_model_only:
                optimizer.load_state_dict(optimizer_state)
            observation_normalizer_state = checkpoint.get("observation_normalizer")
            if (
                observation_normalizer is not None
                and observation_normalizer_state is not None
            ):
                observation_normalizer = RunningMeanStd.from_state_dict(
                    observation_normalizer_state
                )
            if not config.resume_model_only:
                curriculum.load_state_dict(checkpoint.get("curriculum"))
                start_update = max(0, int(checkpoint.get("update_index", -1)) + 1)
                total_env_steps = max(0, int(checkpoint.get("total_env_steps", 0)))
        elif config.pretrain_updates > 0:
            training_logger.emit_status(
                "running behavior-clone pretrain",
                pretrain_updates=config.pretrain_updates,
                pretrain_epochs=config.pretrain_epochs,
                pretrain_minibatch_size=config.pretrain_minibatch_size,
            )
            observation_normalizer = _behavior_clone_pretrain(
                env,
                model,
                optimizer,
                config,
                device=device,
                observation_normalizer=observation_normalizer,
                schedule=schedule,
            )
        reset_params = curriculum.sample_training_reset_params(
            env_count=sim.env_count,
            base_seed=config.seed + start_update,
        )
        sim.set_reset_params(reset_params)
        del reset_params
        observations = env.reset_numpy().copy()
        previous_actions = _neutral_actions(sim.env_count, sim.action_stride)

        storage = _allocate_rollout_arrays(
            config,
            sim.env_count,
            sim.observation_stride,
            sim.action_stride,
            critic_observation_dim,
        )

        final_phase = curriculum.current_phase().name
        final_progress = curriculum.progress()
        best_phase_snapshot: PhaseBestEvalSnapshot | None = None
        end_update = start_update + config.total_updates

        for update_index in range(start_update, end_update):
            started_at = perf_counter()
            phase_index = curriculum.current_phase_index
            phase = curriculum.current_phase()
            if (
                best_phase_snapshot is not None
                and best_phase_snapshot.phase_index != phase_index
            ):
                best_phase_snapshot = None
            progress = curriculum.progress()
            final_phase = phase.name
            final_progress = progress
            session_update_index = update_index - start_update

            if config.anneal_learning_rate:
                if config.total_updates <= 1:
                    current_learning_rate = config.learning_rate
                else:
                    frac = 1.0 - (session_update_index / (config.total_updates - 1))
                    current_learning_rate = config.learning_rate * frac
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_learning_rate
            else:
                current_learning_rate = config.learning_rate

            reset_params = curriculum.sample_training_reset_params(
                env_count=sim.env_count,
                base_seed=config.seed + update_index,
            )
            sim.set_reset_params(reset_params)
            phase_updates = curriculum.note_update()
            episode_tracker.begin_rollout()
            done_reason_tracker.reset_rollout()

            observations, previous_actions, next_value = _collect_rollout(
                env,
                model,
                config,
                device,
                observations,
                previous_actions,
                progress,
                storage,
                observation_normalizer,
                episode_tracker,
                done_reason_tracker,
            )
            done_reason_summary = done_reason_tracker.summary()
            curriculum.record_rollout_feedback(done_reason_summary)
            _compute_gae(storage, next_value, config.gamma, config.gae_lambda)

            batch = _flatten_rollout(storage)
            advantages = batch["advantages"].copy()
            if config.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1.0e-8
                )
            batch_observations = batch["observations"]
            if observation_normalizer is not None:
                batch_observations = observation_normalizer.normalize(
                    batch_observations
                )

            obs_batch = torch_module.as_tensor(
                batch_observations, dtype=torch_module.float32, device=device
            )
            actions_batch = torch_module.as_tensor(
                batch["actions"], dtype=torch_module.float32, device=device
            )
            critic_obs_batch = torch_module.as_tensor(
                batch["critic_observations"], dtype=torch_module.float32, device=device
            )
            previous_actions_batch = torch_module.as_tensor(
                batch["previous_actions"], dtype=torch_module.float32, device=device
            )
            old_log_probs_batch = torch_module.as_tensor(
                batch["log_probs"], dtype=torch_module.float32, device=device
            )
            returns_batch = torch_module.as_tensor(
                batch["returns"], dtype=torch_module.float32, device=device
            )
            old_values_batch = torch_module.as_tensor(
                batch["values"], dtype=torch_module.float32, device=device
            )
            advantages_batch = torch_module.as_tensor(
                advantages, dtype=torch_module.float32, device=device
            )

            batch_size = obs_batch.shape[0]
            minibatch_size = min(config.minibatch_size, batch_size)

            policy_losses: list[float] = []
            value_losses: list[float] = []
            entropy_values: list[float] = []
            approx_kl_values: list[float] = []
            clip_fractions: list[float] = []
            action_smoothness_losses: list[float] = []

            for _ in range(config.ppo_epochs):
                permutation = torch_module.randperm(batch_size, device=device)
                epoch_approx_kl_values: list[float] = []
                for start in range(0, batch_size, minibatch_size):
                    indices = permutation[start : start + minibatch_size]

                    new_log_probs, entropy, new_values, mean_actions = model.evaluate_actions(
                        obs_batch[indices],
                        actions_batch[indices],
                        critic_obs_batch[indices],
                    )
                    log_ratio = new_log_probs - old_log_probs_batch[indices]
                    ratio = log_ratio.exp()

                    unclipped = -advantages_batch[indices] * ratio
                    clipped = -advantages_batch[indices] * torch_module.clamp(
                        ratio,
                        1.0 - config.clip_coef,
                        1.0 + config.clip_coef,
                    )
                    policy_loss = torch_module.max(unclipped, clipped).mean()

                    if config.value_clip_coef is None:
                        value_loss = 0.5 * (
                            (returns_batch[indices] - new_values).pow(2).mean()
                        )
                    else:
                        value_delta = new_values - old_values_batch[indices]
                        clipped_values = old_values_batch[indices] + torch_module.clamp(
                            value_delta,
                            -config.value_clip_coef,
                            config.value_clip_coef,
                        )
                        unclipped_value_loss = (
                            returns_batch[indices] - new_values
                        ).pow(2)
                        clipped_value_loss = (
                            returns_batch[indices] - clipped_values
                        ).pow(2)
                        value_loss = (
                            0.5
                            * torch_module.max(
                                unclipped_value_loss,
                                clipped_value_loss,
                            ).mean()
                        )
                    entropy_loss = entropy.mean()
                    action_smoothness_loss = (
                        (mean_actions - previous_actions_batch[indices])
                        .pow(2)
                        .mean()
                    )

                    loss = (
                        policy_loss
                        + config.value_coef * value_loss
                        - config.entropy_coef * entropy_loss
                        + config.action_smoothness_loss_coef * action_smoothness_loss
                    )

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch_module.nn.utils.clip_grad_norm_(
                        model.parameters(), config.max_grad_norm
                    )
                    optimizer.step()

                    with torch_module.no_grad():
                        approx_kl = float(((ratio - 1.0) - log_ratio).mean().item())
                        clip_fraction = float(
                            ((ratio - 1.0).abs() > config.clip_coef)
                            .float()
                            .mean()
                            .item()
                        )
                        policy_losses.append(float(policy_loss.item()))
                        value_losses.append(float(value_loss.item()))
                        entropy_values.append(float(entropy_loss.item()))
                        action_smoothness_losses.append(
                            float(action_smoothness_loss.item())
                        )
                        approx_kl_values.append(approx_kl)
                        clip_fractions.append(clip_fraction)
                        epoch_approx_kl_values.append(approx_kl)

                if (
                    config.target_kl is not None
                    and epoch_approx_kl_values
                    and float(np.mean(epoch_approx_kl_values)) > config.target_kl
                ):
                    break

            total_env_steps += config.horizon * sim.env_count
            elapsed = perf_counter() - started_at
            completed_episodes, mean_episode_return, mean_episode_length = (
                episode_tracker.rollout_stats()
            )
            eval_stats = curriculum.latest_eval_stats
            phase_advanced = False
            if curriculum.should_evaluate(update_index, end_update):
                eval_stats = _evaluate_curriculum_phase(
                    eval_env,
                    model,
                    device=device,
                    observation_normalizer=observation_normalizer,
                    schedule=schedule,
                    phase_index=phase_index,
                    base_seed=config.curriculum_holdout_seed + phase_index * 104729,
                    max_episode_steps=config.max_episode_steps,
                )
                if (
                    best_phase_snapshot is None
                    or best_phase_snapshot.phase_index != phase_index
                    or _eval_score(eval_stats)
                    > _eval_score(best_phase_snapshot.eval_stats)
                ):
                    best_phase_snapshot = PhaseBestEvalSnapshot(
                        phase_index=phase_index,
                        eval_stats=copy.deepcopy(eval_stats),
                        model_state=copy.deepcopy(model.state_dict()),
                        optimizer_state=copy.deepcopy(optimizer.state_dict()),
                        observation_normalizer_state=None
                        if observation_normalizer is None
                        else copy.deepcopy(observation_normalizer.state_dict()),
                        update_index=update_index,
                        total_env_steps=total_env_steps,
                    )
                    if best_checkpoint_path is not None:
                        _save_checkpoint(
                            best_checkpoint_path,
                            model=model,
                            optimizer=optimizer,
                            config=config,
                            task_spec=task_spec,
                            observation_normalizer=observation_normalizer,
                            curriculum=curriculum,
                            update_index=update_index,
                            total_env_steps=total_env_steps,
                        )
                        training_logger.emit_checkpoint_saved(
                            checkpoint_path=str(best_checkpoint_path),
                            update_index=update_index,
                            total_env_steps=total_env_steps,
                        )
                phase_advanced = curriculum.record_eval(eval_stats, update_index)
                if (
                    phase_advanced
                    and best_phase_snapshot is not None
                    and best_phase_snapshot.phase_index == phase_index
                ):
                    observation_normalizer = _restore_best_phase_snapshot(
                        best_phase_snapshot,
                        model=model,
                        optimizer=optimizer,
                        observation_normalizer=observation_normalizer,
                    )
                    if best_checkpoint_path is not None:
                        _save_checkpoint(
                            best_checkpoint_path,
                            model=model,
                            optimizer=optimizer,
                            config=config,
                            task_spec=task_spec,
                            observation_normalizer=observation_normalizer,
                            curriculum=curriculum,
                            update_index=best_phase_snapshot.update_index,
                            total_env_steps=best_phase_snapshot.total_env_steps,
                        )
                        training_logger.emit_checkpoint_saved(
                            checkpoint_path=str(best_checkpoint_path),
                            update_index=best_phase_snapshot.update_index,
                            total_env_steps=best_phase_snapshot.total_env_steps,
                        )
                    best_phase_snapshot = None

            stats = PPOUpdateStats(
                update_index=update_index,
                progress=progress,
                phase=phase.name,
                phase_index=phase_index,
                phase_updates=phase_updates,
                mean_reward=float(storage["rewards"].mean()),
                mean_done_rate=float(storage["dones"].mean()),
                completed_episodes=completed_episodes,
                mean_episode_return=mean_episode_return,
                mean_episode_length=mean_episode_length,
                eval_completion_rate=0.0
                if eval_stats is None
                else eval_stats.completion_rate,
                eval_gate_pass_rate=0.0
                if eval_stats is None
                else eval_stats.gate_pass_rate,
                eval_mean_progress=0.0
                if eval_stats is None
                else eval_stats.mean_progress,
                eval_mean_episode_return=0.0
                if eval_stats is None
                else eval_stats.mean_episode_return,
                eval_mean_episode_length=0.0
                if eval_stats is None
                else eval_stats.mean_episode_length,
                phase_advanced=phase_advanced,
                policy_loss=float(np.mean(policy_losses)) if policy_losses else 0.0,
                value_loss=float(np.mean(value_losses)) if value_losses else 0.0,
                entropy=float(np.mean(entropy_values)) if entropy_values else 0.0,
                approx_kl=float(np.mean(approx_kl_values)) if approx_kl_values else 0.0,
                clip_fraction=float(np.mean(clip_fractions)) if clip_fractions else 0.0,
                action_smoothness_loss=float(np.mean(action_smoothness_losses))
                if action_smoothness_losses
                else 0.0,
                mean_action_delta=float(np.mean(np.abs(batch["action_deltas"]))),
                mean_action_smoothness_penalty=float(
                    np.mean(batch["action_smoothness_penalties"])
                ),
                explained_variance=_explained_variance(
                    batch["values"], batch["returns"]
                ),
                learning_rate=float(current_learning_rate),
                elapsed_seconds=elapsed,
            )

            if (
                update_index % config.log_interval == 0
                or update_index == end_update - 1
            ):
                training_logger.emit_update(
                    asdict(stats),
                    done_reasons=done_reason_summary,
                    curriculum={
                        "phase_index": phase_index,
                        "phase_updates": phase_updates,
                        "phase_mix": curriculum.training_phase_mix(),
                        "adaptive": curriculum.adaptive_curriculum_state(),
                        "failure_replay_start_gates": dict(
                            curriculum.failure_replay_start_gates
                        ),
                        "failure_replay_grammar_ids": dict(
                            curriculum.failure_replay_grammar_ids
                        ),
                        "randomization_preview": _randomization_preview(
                            reset_params, config
                        ),
                        "eval": None
                        if eval_stats is None
                        else {
                            "completion_rate": eval_stats.completion_rate,
                            "gate_pass_rate": eval_stats.gate_pass_rate,
                            "mean_progress": eval_stats.mean_progress,
                            "mean_episode_return": eval_stats.mean_episode_return,
                            "mean_episode_length": eval_stats.mean_episode_length,
                            "gate_metrics": eval_stats.gate_metrics,
                        },
                        "phase_advanced": phase_advanced,
                        "next_phase": curriculum.current_phase().name,
                    },
                )

            if (
                checkpoint_path is not None
                and config.checkpoint_interval > 0
                and (update_index + 1) % config.checkpoint_interval == 0
            ):
                _save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    task_spec=task_spec,
                    observation_normalizer=observation_normalizer,
                    curriculum=curriculum,
                    update_index=update_index,
                    total_env_steps=total_env_steps,
                )
                training_logger.emit_checkpoint_saved(
                    checkpoint_path=str(checkpoint_path),
                    update_index=update_index,
                    total_env_steps=total_env_steps,
                )

        if checkpoint_path is not None:
            _save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                config=config,
                task_spec=task_spec,
                observation_normalizer=observation_normalizer,
                curriculum=curriculum,
                update_index=end_update - 1,
                total_env_steps=total_env_steps,
            )
            training_logger.emit_checkpoint_saved(
                checkpoint_path=str(checkpoint_path),
                update_index=end_update - 1,
                total_env_steps=total_env_steps,
            )

        result = PPOTrainResult(
            final_update=end_update - 1,
            final_progress=curriculum.progress(),
            final_phase=curriculum.current_phase().name,
            final_phase_index=curriculum.current_phase_index,
            device=device,
            env_count=sim.env_count,
            horizon=config.horizon,
            total_env_steps=total_env_steps,
            checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        )
        training_logger.emit_completed(asdict(result))
        return result
    finally:
        if "tensorboard_writer" in locals() and tensorboard_writer is not None:
            tensorboard_writer.close()
        eval_sim.close()
        eval_course.close()
        sim.close()
        course.close()
