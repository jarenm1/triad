from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from . import (
    CourseSpec,
    SimulationConfig,
    SimulationCore,
    TriadFastVecEnv,
    apply_curriculum_update,
    build_teacher_curriculum_progression,
    build_teacher_curriculum_schedule,
)

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
    checkpoint_path: str | None = None
    checkpoint_interval: int = 0


@dataclass
class PPOUpdateStats:
    update_index: int
    progress: float
    phase: str
    mean_reward: float
    mean_done_rate: float
    completed_episodes: int
    mean_episode_return: float
    mean_episode_length: float
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
    device: str
    env_count: int
    horizon: int
    total_env_steps: int
    checkpoint_path: str | None


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
    sim_config.max_gates_per_env = max(
        sim_config.max_gates_per_env, int(course.stats["total_gate_count"])
    )
    return sim_config


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
        next_observations, rewards, dones = env.step_in_place().numpy_views()

        storage["actions"][step_index] = action_values
        storage["log_probs"][step_index] = log_probs.detach().cpu().numpy()
        storage["values"][step_index] = values.detach().cpu().numpy()
        storage["rewards"][step_index] = rewards
        done_flags = dones.astype(bool, copy=False)
        storage["dones"][step_index] = done_flags.astype(np.float32)
        episode_tracker.record_step(rewards, done_flags)

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
    progression = build_teacher_curriculum_progression(
        total_updates=config.total_updates,
        warmup_updates=config.warmup_updates,
    )

    course = _default_training_course()
    sim = SimulationCore(_training_sim_config(course, config))
    sim.set_course(course)
    env = TriadFastVecEnv(sim)

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

        reset_params = apply_curriculum_update(
            sim,
            update_index=0,
            progression=progression,
            base_seed=config.seed,
            schedule=schedule,
        )
        del reset_params
        observations = env.reset_numpy().copy()

        storage = _allocate_rollout_arrays(
            config,
            sim.env_count,
            sim.observation_stride,
            sim.action_stride,
        )

        total_env_steps = 0
        final_phase = schedule.phase_for_progress(0.0).name
        final_progress = 0.0

        for update_index in range(config.total_updates):
            started_at = perf_counter()
            progress = progression.progress_for_update(update_index)
            phase = schedule.phase_for_progress(progress)
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

            apply_curriculum_update(
                sim,
                update_index=update_index,
                progression=progression,
                base_seed=config.seed + update_index,
                schedule=schedule,
            )
            episode_tracker.begin_rollout()

            observations, next_value = _collect_rollout(
                env,
                model,
                config,
                device,
                observations,
                storage,
                observation_normalizer,
                episode_tracker,
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

            stats = PPOUpdateStats(
                update_index=update_index,
                progress=progress,
                phase=phase.name,
                mean_reward=float(storage["rewards"].mean()),
                mean_done_rate=float(storage["dones"].mean()),
                completed_episodes=completed_episodes,
                mean_episode_return=mean_episode_return,
                mean_episode_length=mean_episode_length,
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
                print(json.dumps(asdict(stats), sort_keys=True))

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

        return PPOTrainResult(
            final_update=config.total_updates - 1,
            final_progress=final_progress,
            final_phase=final_phase,
            device=device,
            env_count=sim.env_count,
            horizon=config.horizon,
            total_env_steps=total_env_steps,
            checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        )
    finally:
        sim.close()
        course.close()
