from __future__ import annotations

import json
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
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 4096
    hidden_size: int = 256
    seed: int = 0
    device: str = "auto"
    log_interval: int = 10
    checkpoint_path: str | None = None


@dataclass
class PPOUpdateStats:
    update_index: int
    progress: float
    phase: str
    mean_reward: float
    mean_done_rate: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    explained_variance: float
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

    def predict(
        self, observations: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        torch_module, _ = _require_torch()
        observation_array = np.asarray(observations, dtype=np.float32)
        if observation_array.ndim == 1:
            observation_array = observation_array[None, :]

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

    class ActorCritic(nn.Module):
        def __init__(self, observation_dim: int, action_dim: int, hidden_size: int):
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(observation_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
            )
            self.policy_mean = nn.Linear(hidden_size, action_dim)
            self.value_head = nn.Linear(hidden_size, 1)
            self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

        def forward(self, observations):
            features = self.body(observations)
            return self.policy_mean(features), self.value_head(features).squeeze(-1)

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


def _collect_rollout(
    env: TriadFastVecEnv,
    model: ActorCritic,
    config: PPOConfig,
    device: str,
    observations: np.ndarray,
    storage: dict[str, np.ndarray],
):
    torch_module, _ = _require_torch()

    for step_index in range(config.horizon):
        storage["observations"][step_index] = observations
        obs_tensor = torch_module.as_tensor(
            observations, dtype=torch_module.float32, device=device
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
        storage["dones"][step_index] = dones.astype(np.float32)

        observations = next_observations.copy()

    obs_tensor = torch_module.as_tensor(
        observations, dtype=torch_module.float32, device=device
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


def load_ppo_policy(
    checkpoint_path: str | Path, device: str = "auto"
) -> LoadedPPOPolicy:
    torch_module, _ = _require_torch()
    resolved_device = _resolve_device(device)
    checkpoint = torch_module.load(
        Path(checkpoint_path), map_location=resolved_device, weights_only=False
    )
    model_state = checkpoint["model"]
    observation_dim = int(model_state["body.0.weight"].shape[1])
    hidden_size = int(model_state["body.0.weight"].shape[0])
    action_dim = int(model_state["policy_mean.weight"].shape[0])
    model = ActorCritic(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
    ).to(resolved_device)
    model.load_state_dict(model_state)
    model.eval()
    return LoadedPPOPolicy(
        model=model,
        device=resolved_device,
        config=dict(checkpoint.get("config", {})),
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

        for update_index in range(config.total_updates):
            started_at = perf_counter()
            progress = progression.progress_for_update(update_index)
            phase = schedule.phase_for_progress(progress)
            final_phase = phase.name

            apply_curriculum_update(
                sim,
                update_index=update_index,
                progression=progression,
                base_seed=config.seed + update_index,
                schedule=schedule,
            )
            observations = env.reset_numpy().copy()

            observations, next_value = _collect_rollout(
                env,
                model,
                config,
                device,
                observations,
                storage,
            )
            _compute_gae(storage, next_value, config.gamma, config.gae_lambda)

            batch = _flatten_rollout(storage)
            advantages = batch["advantages"]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)

            obs_batch = torch_module.as_tensor(
                batch["observations"], dtype=torch_module.float32, device=device
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
            advantages_batch = torch_module.as_tensor(
                advantages, dtype=torch_module.float32, device=device
            )

            batch_size = obs_batch.shape[0]
            minibatch_size = min(config.minibatch_size, batch_size)

            policy_loss_value = 0.0
            value_loss_value = 0.0
            entropy_value = 0.0
            approx_kl_value = 0.0

            for _ in range(config.ppo_epochs):
                permutation = torch_module.randperm(batch_size, device=device)
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

                    value_loss = 0.5 * (
                        (returns_batch[indices] - new_values).pow(2).mean()
                    )
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
                        policy_loss_value = float(policy_loss.item())
                        value_loss_value = float(value_loss.item())
                        entropy_value = float(entropy_loss.item())
                        approx_kl_value = float(
                            ((ratio - 1.0) - log_ratio).mean().item()
                        )

            total_env_steps += config.horizon * sim.env_count
            elapsed = perf_counter() - started_at

            stats = PPOUpdateStats(
                update_index=update_index,
                progress=progress,
                phase=phase.name,
                mean_reward=float(storage["rewards"].mean()),
                mean_done_rate=float(storage["dones"].mean()),
                policy_loss=policy_loss_value,
                value_loss=value_loss_value,
                entropy=entropy_value,
                approx_kl=approx_kl_value,
                explained_variance=_explained_variance(
                    batch["values"], batch["returns"]
                ),
                elapsed_seconds=elapsed,
            )

            if (
                update_index % config.log_interval == 0
                or update_index == config.total_updates - 1
            ):
                print(json.dumps(asdict(stats), sort_keys=True))

        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch_module.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(config),
                },
                checkpoint_path,
            )

        return PPOTrainResult(
            final_update=config.total_updates - 1,
            final_progress=progression.progress_for_update(config.total_updates),
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
