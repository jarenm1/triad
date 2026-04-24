from __future__ import annotations

from collections import deque
from datetime import datetime
import json
import math
import random
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from time import perf_counter

import numpy as np

from . import (
    CourseSpec,
    SimulationConfig,
    SimulationCore,
    TriadFastVecEnv,
    build_teacher_curriculum_schedule,
)
from .training_log import DONE_REASON_BITS, DoneReasonTracker, PPOTrainingLogger

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - runtime dependency
    torch = None
    nn = None


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

    checkpoint_stem = Path(config.checkpoint_path).stem if config.checkpoint_path else "ppo"
    run_name = config.run_name or f"{checkpoint_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    horizon: int = 128
    total_updates: int = 500
    warmup_updates: int = 25
    max_episode_steps: int = 2048
    dynamics_randomization_scale: float = 1.0
    actuator_randomization_scale: float = 1.0
    spawn_randomization_scale: float = 1.0
    learning_rate: float = 3.0e-4
    anneal_learning_rate: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_clip_coef: float | None = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.001
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
    checkpoint_interval: int = 0
    curriculum_eval_interval: int = 10
    curriculum_eval_env_count: int = 64
    curriculum_mastery_window: int = 3
    curriculum_min_stage_updates: int = 20
    curriculum_completion_threshold: float = 0.6
    curriculum_progress_threshold: float = 0.9
    curriculum_current_weight: float = 0.7
    curriculum_previous_weight: float = 0.2
    curriculum_easy_weight: float = 0.1
    curriculum_holdout_seed: int = 131071


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
    eval_mean_progress: float
    eval_mean_episode_return: float
    eval_mean_episode_length: float
    phase_advanced: bool
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
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
    mean_progress: float
    mean_episode_return: float
    mean_episode_length: float
    per_env: list["CurriculumEvalEnvResult"] | None = None


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
    time_penalty: float
    sparse_objective_reward: float
    collision_penalty: float


@dataclass
class PPOEvalResult:
    checkpoint_path: str
    checkpoint_update_index: int | None
    checkpoint_total_env_steps: int | None
    device: str
    eval_env_count: int
    max_episode_steps: int
    per_env: bool
    results: list[CurriculumEvalStats]


@dataclass
class LoadedPPOPolicy:
    model: "ActorCritic"
    device: str
    config: dict[str, object]
    observation_normalizer: "RunningMeanStd | None"

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
        def __init__(self, observation_dim: int, action_dim: int, hidden_size: int):
            super().__init__()
            self.actor_body = nn.Sequential(
                _init_linear(nn.Linear(observation_dim, hidden_size)),
                nn.Tanh(),
                _init_linear(nn.Linear(hidden_size, hidden_size)),
                nn.Tanh(),
            )
            self.critic_body = nn.Sequential(
                _init_linear(nn.Linear(observation_dim, hidden_size)),
                nn.Tanh(),
                _init_linear(nn.Linear(hidden_size, hidden_size)),
                nn.Tanh(),
            )
            self.policy_mean = _init_linear(
                nn.Linear(hidden_size, action_dim), std=0.01
            )
            self.value_head = _init_linear(nn.Linear(hidden_size, 1), std=1.0)
            self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

        def forward(self, observations):
            actor_features = self.actor_body(observations)
            critic_features = self.critic_body(observations)
            return self.policy_mean(actor_features), self.value_head(
                critic_features
            ).squeeze(-1)

        def act(self, observations):
            mean, value = self.forward(observations)
            std = torch.exp(self.log_std).expand_as(mean)
            dist = torch.distributions.Normal(mean, std)
            raw_action = dist.rsample()
            actions = torch.sigmoid(raw_action)
            log_prob = _squashed_log_prob(dist, raw_action, actions)
            entropy = dist.entropy().sum(dim=-1)
            return actions, log_prob, entropy, value

        def act_deterministic(self, observations):
            mean, value = self.forward(observations)
            actions = torch.sigmoid(mean)
            return actions, value

        def evaluate_actions(self, observations, actions):
            mean, value = self.forward(observations)
            std = torch.exp(self.log_std).expand_as(mean)
            dist = torch.distributions.Normal(mean, std)
            safe_actions = actions.clamp(1.0e-5, 1.0 - 1.0e-5)
            raw_action = torch.logit(safe_actions)
            log_prob = _squashed_log_prob(dist, raw_action, safe_actions)
            entropy = dist.entropy().sum(dim=-1)
            return log_prob, entropy, value

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


def _default_training_course() -> CourseSpec:
    return CourseSpec.default_drone_course()


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
    return sim_config


def _evaluation_sim_config(course: CourseSpec, config: PPOConfig) -> SimulationConfig:
    sim_config = _training_sim_config(course, config)
    sim_config.env_count = max(1, min(config.curriculum_eval_env_count, config.env_count))
    return sim_config


def _hash_to_unit(seed: int) -> float:
    mixed = (seed * 747796405 + 2891336453) & 0xFFFFFFFF
    masked = mixed & 0x00FFFFFF
    return masked / 16777215.0


def _hash_to_signed(seed: int) -> float:
    return _hash_to_unit(seed) * 2.0 - 1.0


def _curriculum_dynamics_scale(curriculum_stage: int) -> float:
    if curriculum_stage == 0:
        return 0.35
    if curriculum_stage == 1:
        return 0.55
    if curriculum_stage == 2:
        return 0.8
    return 1.0


def _randomized_positive_scale(seed: int, salt: int, magnitude: float) -> float:
    return max(0.25, 1.0 + _hash_to_signed((seed ^ salt) & 0xFFFFFFFF) * magnitude)


def _summarize_array(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def _randomization_preview(
    reset_params: list[tuple[int, int, float, int]],
    config: PPOConfig,
) -> dict[str, object]:
    if not reset_params:
        return {}

    seeds = np.asarray([item[0] for item in reset_params], dtype=np.uint32)
    difficulties = np.asarray([item[2] for item in reset_params], dtype=np.float32)
    curriculum_stages = np.asarray([item[3] for item in reset_params], dtype=np.uint32)

    dynamics_strength = (
        0.04 + np.clip(difficulties, 0.0, 1.0) * 0.18
    ) * np.asarray(
        [_curriculum_dynamics_scale(int(stage)) for stage in curriculum_stages],
        dtype=np.float32,
    ) * float(config.dynamics_randomization_scale)
    actuator_strength = dynamics_strength * 0.25 * float(config.actuator_randomization_scale)

    def scales(salt: int, multiplier: float = 1.0) -> np.ndarray:
        return np.asarray(
            [
                _randomized_positive_scale(int(seed), salt, float(strength) * multiplier)
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
            _hash_to_signed(int(seed) ^ 0x611) * (0.08 + float(diff) * 0.18) * spawn_scale
            for seed, diff in zip(seeds, difficulties, strict=True)
        ],
        dtype=np.float32,
    )
    yaw_offset = np.asarray(
        [
            _hash_to_signed(int(seed) ^ 0x1551) * 0.25 * spawn_scale
            for seed in seeds
        ],
        dtype=np.float32,
    )

    return {
        "difficulty": _summarize_array(difficulties),
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


def _allocate_rollout_arrays(
    config: PPOConfig, env_count: int, obs_dim: int, action_dim: int
):
    return {
        "observations": np.empty(
            (config.horizon, env_count, obs_dim), dtype=np.float32
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


class MasteryCurriculumController:
    def __init__(self, schedule, config: PPOConfig):
        self.schedule = schedule
        self.config = config
        self.current_phase_index = 0
        self.phase_update_count = 0
        self.latest_eval_stats: CurriculumEvalStats | None = None
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

    def training_phase_mix(self) -> dict[str, float]:
        if self.current_phase_index <= 0:
            return {self.current_phase().name: 1.0}

        raw_mix: dict[int, float] = {}

        def add_weight(phase_index: int, weight: float) -> None:
            if phase_index < 0 or weight <= 0.0:
                return
            raw_mix[phase_index] = raw_mix.get(phase_index, 0.0) + weight

        add_weight(self.current_phase_index, self.config.curriculum_current_weight)
        add_weight(self.current_phase_index - 1, self.config.curriculum_previous_weight)
        add_weight(0, self.config.curriculum_easy_weight)

        total = sum(raw_mix.values())
        if total <= 0.0:
            return {self.current_phase().name: 1.0}

        return {
            self.schedule.phase_at_index(phase_index).name: weight / total
            for phase_index, weight in sorted(raw_mix.items())
        }

    def sample_training_reset_params(
        self, env_count: int, base_seed: int
    ) -> list[tuple[int, int, float, int]]:
        if self.current_phase_index <= 0:
            return self.schedule.sample_reset_params_for_phase(
                env_count=env_count,
                phase_index=0,
                base_seed=base_seed,
            )

        phase_mix = self.training_phase_mix()
        phase_indices = [
            phase_index
            for phase_index, phase in enumerate(self.schedule.phases)
            if phase.name in phase_mix
        ]
        weights = [phase_mix[self.schedule.phase_at_index(i).name] for i in phase_indices]
        return self.schedule.sample_reset_params_mixture(
            env_count=env_count,
            phase_indices=phase_indices,
            weights=weights,
            base_seed=base_seed,
        )

    def should_evaluate(self, update_index: int, total_updates: int) -> bool:
        if self.config.curriculum_eval_interval <= 0:
            return update_index == total_updates - 1
        return (
            (update_index + 1) % self.config.curriculum_eval_interval == 0
            or update_index == total_updates - 1
        )

    def record_eval(self, eval_stats: CurriculumEvalStats, update_index: int) -> bool:
        self.latest_eval_stats = eval_stats
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
        mean_progress = float(np.mean([item.mean_progress for item in history]))
        if (
            mean_completion < self.config.curriculum_completion_threshold
            or mean_progress < self.config.curriculum_progress_threshold
        ):
            return False

        self.current_phase_index += 1
        self.phase_update_count = 0
        return True


COMPLETE_DONE_REASON = 1 << 0
PROGRESS_OBSERVATION_INDEX = 18


def _done_reason_labels(done_reason: int) -> list[str]:
    labels = [label for bit, label in DONE_REASON_BITS if done_reason & int(bit) != 0]
    return labels or ["none"]


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
    env.sim.set_reset_params(reset_params)
    observations = env.reset_numpy().copy()

    active = np.ones(env.sim.env_count, dtype=bool)
    completed = np.zeros(env.sim.env_count, dtype=bool)
    returns = np.zeros(env.sim.env_count, dtype=np.float32)
    lengths = np.zeros(env.sim.env_count, dtype=np.int32)
    best_progress = observations[:, PROGRESS_OBSERVATION_INDEX].copy()

    for _ in range(max_episode_steps):
        obs_tensor = _observations_to_tensor(
            observations,
            device=device,
            observation_normalizer=observation_normalizer,
        )
        with torch_module.no_grad():
            actions, _ = model.act_deterministic(obs_tensor)

        action_values = actions.detach().cpu().numpy()
        action_values[~active] = 0.0
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

        newly_done = active & done_flags
        if np.any(newly_done):
            completed[newly_done] = (
                done_reasons[newly_done] & np.uint32(COMPLETE_DONE_REASON)
            ) != 0
            active[newly_done] = False

        observations = next_observations.copy()
        if not np.any(active):
            break

    phase = schedule.phase_at_index(phase_index)
    per_env = None
    if include_per_env:
        final_states = env.sim.get_state()
        final_observations = env.sim.get_observations()
        final_reward_done = env.sim.get_reward_done()
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
                distance_to_gate=float(final_observations[env_index]["distance_to_gate"]),
                gate_alignment=float(final_observations[env_index]["gate_alignment"]),
                reward=float(final_reward_done[env_index]["reward"]),
                shaping_reward=float(final_reward_done[env_index]["shaping_reward"]),
                time_penalty=float(final_reward_done[env_index]["time_penalty"]),
                sparse_objective_reward=float(
                    final_reward_done[env_index]["sparse_objective_reward"]
                ),
                collision_penalty=float(final_reward_done[env_index]["collision_penalty"]),
            )
            for env_index, (
                grammar_id,
                curriculum_stage,
                difficulty,
                seed,
            ) in enumerate(reset_params)
        ]
    return CurriculumEvalStats(
        phase_index=phase_index,
        phase=phase.name,
        completion_rate=float(np.mean(completed)),
        mean_progress=float(np.mean(best_progress)),
        mean_episode_return=float(np.mean(returns)),
        mean_episode_length=float(np.mean(lengths)),
        per_env=per_env,
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
    storage: dict[str, np.ndarray],
    observation_normalizer: RunningMeanStd | None,
    episode_tracker: EpisodeTracker,
    done_reason_tracker: DoneReasonTracker,
):
    torch_module, _ = _require_torch()

    for step_index in range(config.horizon):
        storage["observations"][step_index] = observations
        if observation_normalizer is not None:
            observation_normalizer.update(observations)
        obs_tensor = _observations_to_tensor(
            observations,
            device=device,
            observation_normalizer=observation_normalizer,
        )
        with torch_module.no_grad():
            actions, log_probs, _, values = model.act(obs_tensor)

        action_values = actions.detach().cpu().numpy()
        env.numpy_action_view()[:, :] = action_values
        step_result = env.step_in_place()
        next_observations, rewards, dones = step_result.numpy_views()

        storage["actions"][step_index] = action_values
        storage["log_probs"][step_index] = log_probs.detach().cpu().numpy()
        storage["values"][step_index] = values.detach().cpu().numpy()
        storage["rewards"][step_index] = rewards
        done_flags = dones.astype(bool, copy=False)
        storage["dones"][step_index] = done_flags.astype(np.float32)
        episode_tracker.record_step(rewards, done_flags)
        done_reason_tracker.record_step(done_flags, step_result.numpy_done_reasons())

        observations = next_observations.copy()

    obs_tensor = _observations_to_tensor(
        observations,
        device=device,
        observation_normalizer=observation_normalizer,
    )
    with torch_module.no_grad():
        _, next_value = model.forward(obs_tensor)

    return observations, next_value.detach().cpu().numpy()


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


def _save_checkpoint(
    checkpoint_path: Path,
    *,
    model: ActorCritic,
    optimizer,
    config: PPOConfig,
    observation_normalizer: RunningMeanStd | None,
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
            "observation_normalizer": None
            if observation_normalizer is None
            else observation_normalizer.state_dict(),
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


def load_ppo_policy(
    checkpoint_path: str | Path, device: str = "auto"
) -> LoadedPPOPolicy:
    torch_module, _ = _require_torch()
    resolved_device = _resolve_device(device)
    checkpoint = torch_module.load(
        Path(checkpoint_path), map_location=resolved_device, weights_only=False
    )
    model_state = _upgrade_legacy_model_state(checkpoint["model"])
    observation_key = (
        "actor_body.0.weight"
        if "actor_body.0.weight" in model_state
        else "body.0.weight"
    )
    observation_dim = int(model_state[observation_key].shape[1])
    hidden_size = int(model_state[observation_key].shape[0])
    action_dim = int(model_state["policy_mean.weight"].shape[0])
    model = ActorCritic(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
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
        results=results,
    )


def serve_ppo_policy(checkpoint_path: str | Path, device: str = "auto") -> int:
    policy = load_ppo_policy(checkpoint_path, device=device)
    for line in sys.stdin:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            request = json.loads(stripped)
            if request.get("kind") == "shutdown":
                return 0

            observations = np.asarray(request["observations"], dtype=np.float32)
            deterministic = bool(request.get("deterministic", True))
            actions, values = policy.predict(
                observations, deterministic=deterministic
            )
            response = {
                "actions": actions.tolist(),
                "values": values.tolist(),
            }
        except Exception as error:  # pragma: no cover - IPC guard
            response = {"error": str(error)}

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
    return 0


def train_ppo(config: PPOConfig) -> PPOTrainResult:
    torch_module, _ = _require_torch()
    _set_global_seeds(config.seed)
    device = _resolve_device(config.device)

    schedule = build_teacher_curriculum_schedule()
    curriculum = MasteryCurriculumController(schedule, config)

    course = _default_training_course()
    sim = SimulationCore(_training_sim_config(course, config))
    sim.set_course(course)
    env = TriadFastVecEnv(sim)
    eval_course = _default_training_course()
    eval_sim = SimulationCore(_evaluation_sim_config(eval_course, config))
    eval_sim.set_course(eval_course)
    eval_env = TriadFastVecEnv(eval_sim, auto_reset=False)

    checkpoint_path = Path(config.checkpoint_path) if config.checkpoint_path else None

    try:
        model = ActorCritic(
            observation_dim=sim.observation_stride,
            action_dim=sim.action_stride,
            hidden_size=config.hidden_size,
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
        )
        reset_params = curriculum.sample_training_reset_params(
            env_count=sim.env_count,
            base_seed=config.seed,
        )
        sim.set_reset_params(reset_params)
        del reset_params
        observations = env.reset_numpy().copy()

        storage = _allocate_rollout_arrays(
            config,
            sim.env_count,
            sim.observation_stride,
            sim.action_stride,
        )

        total_env_steps = 0
        final_phase = curriculum.current_phase().name
        final_progress = curriculum.progress()
        training_logger.emit_started(initial_phase=final_phase)

        for update_index in range(config.total_updates):
            started_at = perf_counter()
            phase_index = curriculum.current_phase_index
            phase = curriculum.current_phase()
            progress = curriculum.progress()
            final_phase = phase.name
            final_progress = progress

            if config.anneal_learning_rate:
                if config.total_updates <= 1:
                    current_learning_rate = config.learning_rate
                else:
                    frac = 1.0 - (update_index / (config.total_updates - 1))
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

            observations, next_value = _collect_rollout(
                env,
                model,
                config,
                device,
                observations,
                storage,
                observation_normalizer,
                episode_tracker,
                done_reason_tracker,
            )
            _compute_gae(storage, next_value, config.gamma, config.gae_lambda)

            batch = _flatten_rollout(storage)
            advantages = batch["advantages"].copy()
            if config.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1.0e-8
                )
            batch_observations = batch["observations"]
            if observation_normalizer is not None:
                batch_observations = observation_normalizer.normalize(batch_observations)

            obs_batch = torch_module.as_tensor(
                batch_observations, dtype=torch_module.float32, device=device
            )
            actions_batch = torch_module.as_tensor(
                batch["actions"], dtype=torch_module.float32, device=device
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

            for _ in range(config.ppo_epochs):
                permutation = torch_module.randperm(batch_size, device=device)
                epoch_approx_kl_values: list[float] = []
                for start in range(0, batch_size, minibatch_size):
                    indices = permutation[start : start + minibatch_size]

                    new_log_probs, entropy, new_values = model.evaluate_actions(
                        obs_batch[indices],
                        actions_batch[indices],
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
                        value_loss = 0.5 * torch_module.max(
                            unclipped_value_loss,
                            clipped_value_loss,
                        ).mean()
                    entropy_loss = entropy.mean()

                    loss = (
                        policy_loss
                        + config.value_coef * value_loss
                        - config.entropy_coef * entropy_loss
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
            if curriculum.should_evaluate(update_index, config.total_updates):
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
                phase_advanced = curriculum.record_eval(eval_stats, update_index)

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
                eval_completion_rate=0.0 if eval_stats is None else eval_stats.completion_rate,
                eval_mean_progress=0.0 if eval_stats is None else eval_stats.mean_progress,
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
                explained_variance=_explained_variance(
                    batch["values"], batch["returns"]
                ),
                learning_rate=float(current_learning_rate),
                elapsed_seconds=elapsed,
            )

            if (
                update_index % config.log_interval == 0
                or update_index == config.total_updates - 1
            ):
                training_logger.emit_update(
                    asdict(stats),
                    done_reasons=done_reason_tracker.summary(),
                    curriculum={
                        "phase_index": phase_index,
                        "phase_updates": phase_updates,
                        "phase_mix": curriculum.training_phase_mix(),
                        "randomization_preview": _randomization_preview(
                            reset_params, config
                        ),
                        "eval": None
                        if eval_stats is None
                        else {
                            "completion_rate": eval_stats.completion_rate,
                            "mean_progress": eval_stats.mean_progress,
                            "mean_episode_return": eval_stats.mean_episode_return,
                            "mean_episode_length": eval_stats.mean_episode_length,
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
                    observation_normalizer=observation_normalizer,
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
                observation_normalizer=observation_normalizer,
                update_index=config.total_updates - 1,
                total_env_steps=total_env_steps,
            )
            training_logger.emit_checkpoint_saved(
                checkpoint_path=str(checkpoint_path),
                update_index=config.total_updates - 1,
                total_env_steps=total_env_steps,
            )

        result = PPOTrainResult(
            final_update=config.total_updates - 1,
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
