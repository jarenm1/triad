from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

from . import TriadFastVecEnv, _numpy_module


@dataclass
class RolloutBatch:
    observations: object
    actions: object
    rewards: object
    dones: object
    final_observations: object

    @property
    def horizon(self) -> int:
        return int(self.observations.shape[0])

    @property
    def env_count(self) -> int:
        return int(self.observations.shape[1])

    @property
    def observation_dim(self) -> int:
        return int(self.observations.shape[2])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[2])

    @property
    def total_steps(self) -> int:
        return self.horizon * self.env_count


@dataclass
class BenchmarkResult:
    iterations: int
    horizon: int
    env_count: int
    elapsed_seconds: float
    total_env_steps: int
    env_steps_per_second: float
    rollouts_per_second: float


PolicyFn = Callable[[object, object, int], None]


def zero_policy(observations: object, actions: object, step_index: int) -> None:
    del observations, step_index
    actions[:, 0] = 0.0
    actions[:, 1:] = 0.5


def point_to_gate_policy(
    observations: object, actions: object, step_index: int
) -> None:
    del step_index
    np = _numpy_module()

    position = observations[:, 0:3]
    velocity = observations[:, 3:6]
    attitude = observations[:, 6:9]
    angular_velocity = observations[:, 9:12]
    target_gate_position = observations[:, 12:15]
    target_gate_forward = observations[:, 15:18]

    target_delta = target_gate_position - position
    yaw = attitude[:, 2]
    heading = np.stack((np.cos(yaw), np.sin(yaw)), axis=1)
    right = np.stack((heading[:, 1], -heading[:, 0]), axis=1)

    horizontal_delta = target_delta[:, [0, 2]]
    horizontal_velocity = velocity[:, [0, 2]]
    forward_error = np.sum(horizontal_delta * heading, axis=1)
    lateral_error = np.sum(horizontal_delta * right, axis=1)
    forward_velocity = np.sum(horizontal_velocity * heading, axis=1)
    lateral_velocity = np.sum(horizontal_velocity * right, axis=1)

    desired_yaw = np.arctan2(target_gate_forward[:, 2], target_gate_forward[:, 0])
    yaw_error = desired_yaw - yaw
    yaw_error = (yaw_error + np.pi) % (2.0 * np.pi) - np.pi

    altitude_error = target_delta[:, 1]
    vertical_velocity = velocity[:, 1]

    collective = 0.48 + altitude_error * 0.22 - vertical_velocity * 0.08
    roll_rate_cmd = np.clip(-lateral_error * 0.9 + lateral_velocity * 0.45, -6.0, 6.0)
    pitch_rate_cmd = np.clip(forward_error * 0.9 - forward_velocity * 0.45, -6.0, 6.0)
    yaw_rate_cmd = np.clip(yaw_error * 2.4 - angular_velocity[:, 2] * 0.35, -4.0, 4.0)

    actions[:, 0] = np.clip(collective, 0.0, 1.0)
    actions[:, 1] = np.clip(0.5 + roll_rate_cmd / 15.0, 0.0, 1.0)
    actions[:, 2] = np.clip(0.5 + pitch_rate_cmd / 15.0, 0.0, 1.0)
    actions[:, 3] = np.clip(0.5 + yaw_rate_cmd / 9.0, 0.0, 1.0)


class RolloutCollector:
    def __init__(self, env: TriadFastVecEnv, horizon: int) -> None:
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")

        np = _numpy_module()
        self.env = env
        self.horizon = horizon
        self._obs_view, _, _ = env.numpy_result_views()
        self._action_view = env.numpy_action_view()
        self.batch = RolloutBatch(
            observations=np.empty(
                (horizon, env.sim.env_count, env.sim.observation_stride),
                dtype=self._obs_view.dtype,
            ),
            actions=np.empty(
                (horizon, env.sim.env_count, env.sim.action_stride),
                dtype=self._action_view.dtype,
            ),
            rewards=np.empty((horizon, env.sim.env_count), dtype=np.float32),
            dones=np.empty((horizon, env.sim.env_count), dtype=np.uint8),
            final_observations=np.empty_like(self._obs_view),
        )

    @property
    def action_view(self) -> object:
        return self._action_view

    @property
    def observation_view(self) -> object:
        return self._obs_view

    def collect(self, policy: PolicyFn | None = None) -> RolloutBatch:
        policy = policy or zero_policy

        for step_index in range(self.horizon):
            policy(self._obs_view, self._action_view, step_index)
            self.batch.actions[step_index, :, :] = self._action_view
            step_obs, step_rewards, step_dones = self.env.step_in_place().numpy_views()
            self.batch.observations[step_index, :, :] = step_obs
            self.batch.rewards[step_index, :] = step_rewards
            self.batch.dones[step_index, :] = step_dones

        self.batch.final_observations[:, :] = self._obs_view
        return self.batch

    def benchmark(
        self,
        iterations: int,
        policy: PolicyFn | None = None,
        warmup_iterations: int = 1,
    ) -> BenchmarkResult:
        if iterations <= 0:
            raise ValueError(f"iterations must be positive, got {iterations}")
        if warmup_iterations < 0:
            raise ValueError(
                f"warmup_iterations must be non-negative, got {warmup_iterations}"
            )

        policy = policy or zero_policy
        for _ in range(warmup_iterations):
            self.collect(policy)

        start = perf_counter()
        for _ in range(iterations):
            self.collect(policy)
        elapsed_seconds = perf_counter() - start

        total_env_steps = iterations * self.horizon * self.env.sim.env_count
        return BenchmarkResult(
            iterations=iterations,
            horizon=self.horizon,
            env_count=self.env.sim.env_count,
            elapsed_seconds=elapsed_seconds,
            total_env_steps=total_env_steps,
            env_steps_per_second=total_env_steps / elapsed_seconds,
            rollouts_per_second=iterations / elapsed_seconds,
        )


def collect_rollout_numpy(
    env: TriadFastVecEnv,
    horizon: int,
    policy: PolicyFn | None = None,
) -> RolloutBatch:
    collector = RolloutCollector(env, horizon)
    return collector.collect(policy)


def benchmark_rollout(
    env: TriadFastVecEnv,
    horizon: int,
    iterations: int,
    policy: PolicyFn | None = None,
    warmup_iterations: int = 1,
) -> BenchmarkResult:
    collector = RolloutCollector(env, horizon)
    return collector.benchmark(iterations, policy, warmup_iterations)
