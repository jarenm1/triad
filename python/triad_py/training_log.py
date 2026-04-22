from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Mapping

import numpy as np

DONE_REASON_BITS: tuple[tuple[int, str], ...] = (
    (1 << 0, "complete"),
    (1 << 1, "gate_collision"),
    (1 << 2, "obstacle_collision"),
    (1 << 3, "floor_collision"),
    (1 << 4, "out_of_bounds"),
    (1 << 5, "step_limit"),
    (1 << 6, "excessive_tilt"),
)
_KNOWN_DONE_REASON_MASK = sum(bit for bit, _ in DONE_REASON_BITS)


@dataclass(frozen=True)
class DoneReasonSummary:
    total_terminations: int
    multi_reason_terminations: int
    counts: dict[str, int]
    rates: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "total_terminations": self.total_terminations,
            "multi_reason_terminations": self.multi_reason_terminations,
            "counts": self.counts,
            "rates": self.rates,
        }


class DoneReasonTracker:
    def __init__(self) -> None:
        self._labels = [label for _, label in DONE_REASON_BITS]
        self.reset_rollout()

    def reset_rollout(self) -> None:
        self._counts = {label: 0 for label in self._labels}
        self._counts["unknown"] = 0
        self._total_terminations = 0
        self._multi_reason_terminations = 0

    def record_step(self, done_flags: np.ndarray, done_reasons: np.ndarray) -> None:
        terminated = np.asarray(done_reasons, dtype=np.uint32)[
            np.asarray(done_flags, dtype=bool)
        ]
        if terminated.size == 0:
            return

        self._total_terminations += int(terminated.size)
        self._multi_reason_terminations += int(
            np.count_nonzero((terminated != 0) & ((terminated & (terminated - 1)) != 0))
        )

        for bit, label in DONE_REASON_BITS:
            self._counts[label] += int(np.count_nonzero((terminated & bit) != 0))

        self._counts["unknown"] += int(
            np.count_nonzero((terminated & ~np.uint32(_KNOWN_DONE_REASON_MASK)) != 0)
        )

    def summary(self) -> DoneReasonSummary:
        if self._total_terminations <= 0:
            rates = {label: 0.0 for label in self._counts}
        else:
            rates = {
                label: count / float(self._total_terminations)
                for label, count in self._counts.items()
            }

        return DoneReasonSummary(
            total_terminations=self._total_terminations,
            multi_reason_terminations=self._multi_reason_terminations,
            counts=dict(self._counts),
            rates=rates,
        )


class PPOTrainingLogger:
    def __init__(self, config: Mapping[str, object]) -> None:
        self._config = dict(config)
        self._env_steps_per_update = int(config["env_count"]) * int(config["horizon"])

    def emit_started(self, *, initial_phase: str) -> None:
        self._emit(
            {
                "event": "ppo.start",
                "config": self._config,
                "rollout": {"env_steps_per_update": self._env_steps_per_update},
                "curriculum": {"initial_phase": initial_phase},
            }
        )

    def emit_update(
        self,
        stats: Mapping[str, object],
        *,
        done_reasons: DoneReasonSummary,
        curriculum: Mapping[str, object] | None = None,
    ) -> None:
        elapsed_seconds = float(stats["elapsed_seconds"])
        throughput = (
            self._env_steps_per_update / elapsed_seconds
            if elapsed_seconds > 1.0e-9
            else 0.0
        )
        payload = {
            "event": "ppo.update",
            "update": {
                "index": int(stats["update_index"]),
                "progress": float(stats["progress"]),
                "phase": str(stats["phase"]),
            },
            "rollout": {
                "env_steps": self._env_steps_per_update,
                "mean_reward": float(stats["mean_reward"]),
                "mean_done_rate": float(stats["mean_done_rate"]),
                "completed_episodes": int(stats["completed_episodes"]),
                "mean_episode_return": float(stats["mean_episode_return"]),
                "mean_episode_length": float(stats["mean_episode_length"]),
            },
            "optimization": {
                "policy_loss": float(stats["policy_loss"]),
                "value_loss": float(stats["value_loss"]),
                "entropy": float(stats["entropy"]),
                "approx_kl": float(stats["approx_kl"]),
                "clip_fraction": float(stats["clip_fraction"]),
                "explained_variance": float(stats["explained_variance"]),
                "learning_rate": float(stats["learning_rate"]),
            },
            "done_reasons": done_reasons.as_dict(),
            "timing": {
                "elapsed_seconds": elapsed_seconds,
                "env_steps_per_second": throughput,
            },
        }
        if curriculum is not None:
            payload["curriculum"] = dict(curriculum)
        self._emit(payload)

    def emit_checkpoint_saved(
        self, *, checkpoint_path: str, update_index: int, total_env_steps: int
    ) -> None:
        self._emit(
            {
                "event": "ppo.checkpoint_saved",
                "checkpoint": {
                    "path": checkpoint_path,
                    "update_index": int(update_index),
                    "total_env_steps": int(total_env_steps),
                },
            }
        )

    def emit_completed(self, result: Mapping[str, object]) -> None:
        self._emit({"event": "ppo.complete", "result": dict(result)})

    def _emit(self, payload: Mapping[str, object]) -> None:
        sys.stdout.write(json.dumps(dict(payload), sort_keys=True) + "\n")
        sys.stdout.flush()
