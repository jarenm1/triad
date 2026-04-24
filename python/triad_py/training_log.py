from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

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
_DONE_REASON_SHORT_LABELS = {
    "complete": "comp",
    "gate_collision": "gate",
    "obstacle_collision": "obs",
    "floor_collision": "floor",
    "out_of_bounds": "oob",
    "step_limit": "step",
    "excessive_tilt": "tilt",
    "unknown": "unk",
}


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
    def __init__(
        self,
        config: Mapping[str, object],
        *,
        json_output: bool = True,
        pretty_output: bool = True,
        tensorboard_writer: Any | None = None,
        tensorboard_log_dir: str | None = None,
    ) -> None:
        self._config = dict(config)
        self._env_steps_per_update = int(config["env_count"]) * int(config["horizon"])
        self._json_output = json_output
        self._pretty_output = pretty_output
        self._tensorboard_writer = tensorboard_writer
        self._tensorboard_log_dir = tensorboard_log_dir

    def emit_started(self, *, initial_phase: str) -> None:
        self._emit(
            {
                "event": "ppo.start",
                "config": self._config,
                "rollout": {"env_steps_per_update": self._env_steps_per_update},
                "curriculum": {"initial_phase": initial_phase},
            }
        )
        if self._pretty_output:
            message = (
                f"[ppo] start phase={initial_phase} "
                f"envs={self._config['env_count']} horizon={self._config['horizon']} "
                f"device={self._config['device']}"
            )
            if self._tensorboard_log_dir is not None:
                message += f" tb={self._tensorboard_log_dir}"
            self._emit_pretty(message)
        if self._tensorboard_writer is not None:
            self._tensorboard_writer.add_text(
                "run/config_json",
                json.dumps(self._config, indent=2, sort_keys=True),
                0,
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
        self._emit_tensorboard(stats, done_reasons=done_reasons, curriculum=curriculum)
        if self._pretty_output:
            self._emit_pretty(self._format_update_line(payload))

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
        if self._pretty_output:
            self._emit_pretty(
                f"[ppo] checkpoint update={update_index} steps={total_env_steps} path={checkpoint_path}"
            )

    def emit_completed(self, result: Mapping[str, object]) -> None:
        self._emit({"event": "ppo.complete", "result": dict(result)})
        if self._pretty_output:
            self._emit_pretty(
                f"[ppo] complete update={int(result['final_update'])} "
                f"phase={result['final_phase']} progress={float(result['final_progress']):.3f} "
                f"steps={int(result['total_env_steps'])}"
            )
        if self._tensorboard_writer is not None:
            self._tensorboard_writer.add_text(
                "run/result_json",
                json.dumps(dict(result), indent=2, sort_keys=True),
                int(result["total_env_steps"]),
            )
            self._tensorboard_writer.flush()

    def _emit(self, payload: Mapping[str, object]) -> None:
        if self._json_output:
            sys.stdout.write(json.dumps(dict(payload), sort_keys=True) + "\n")
            sys.stdout.flush()

    def _emit_pretty(self, message: str) -> None:
        sys.stderr.write(message + "\n")
        sys.stderr.flush()

    def _emit_tensorboard(
        self,
        stats: Mapping[str, object],
        *,
        done_reasons: DoneReasonSummary,
        curriculum: Mapping[str, object] | None,
    ) -> None:
        writer = self._tensorboard_writer
        if writer is None:
            return
        update_index = int(stats["update_index"])
        env_step = (update_index + 1) * self._env_steps_per_update
        writer.add_scalar("rollout/mean_reward", float(stats["mean_reward"]), env_step)
        writer.add_scalar(
            "rollout/mean_done_rate", float(stats["mean_done_rate"]), env_step
        )
        writer.add_scalar(
            "rollout/completed_episodes", int(stats["completed_episodes"]), env_step
        )
        writer.add_scalar(
            "rollout/mean_episode_return", float(stats["mean_episode_return"]), env_step
        )
        writer.add_scalar(
            "rollout/mean_episode_length", float(stats["mean_episode_length"]), env_step
        )
        writer.add_scalar("optimization/policy_loss", float(stats["policy_loss"]), env_step)
        writer.add_scalar("optimization/value_loss", float(stats["value_loss"]), env_step)
        writer.add_scalar("optimization/entropy", float(stats["entropy"]), env_step)
        writer.add_scalar("optimization/approx_kl", float(stats["approx_kl"]), env_step)
        writer.add_scalar(
            "optimization/clip_fraction", float(stats["clip_fraction"]), env_step
        )
        writer.add_scalar(
            "optimization/explained_variance",
            float(stats["explained_variance"]),
            env_step,
        )
        writer.add_scalar(
            "optimization/learning_rate", float(stats["learning_rate"]), env_step
        )
        elapsed_seconds = float(stats["elapsed_seconds"])
        throughput = (
            self._env_steps_per_update / elapsed_seconds
            if elapsed_seconds > 1.0e-9
            else 0.0
        )
        writer.add_scalar("timing/env_steps_per_second", throughput, env_step)
        writer.add_scalar("curriculum/progress", float(stats["progress"]), env_step)
        writer.add_scalar("curriculum/phase_index", int(stats["phase_index"]), env_step)
        writer.add_scalar("curriculum/phase_updates", int(stats["phase_updates"]), env_step)
        writer.add_scalar(
            "curriculum/phase_advanced",
            1.0 if bool(stats["phase_advanced"]) else 0.0,
            env_step,
        )
        writer.add_scalar(
            "eval/completion_rate", float(stats["eval_completion_rate"]), env_step
        )
        writer.add_scalar("eval/mean_progress", float(stats["eval_mean_progress"]), env_step)
        writer.add_scalar(
            "eval/mean_episode_return",
            float(stats["eval_mean_episode_return"]),
            env_step,
        )
        writer.add_scalar(
            "eval/mean_episode_length",
            float(stats["eval_mean_episode_length"]),
            env_step,
        )
        for label, rate in done_reasons.rates.items():
            writer.add_scalar(f"done_reasons/{label}_rate", float(rate), env_step)
        if curriculum is not None:
            phase_mix = curriculum.get("phase_mix")
            if isinstance(phase_mix, Mapping):
                for phase_name, weight in phase_mix.items():
                    writer.add_scalar(
                        f"curriculum/phase_mix/{phase_name}",
                        float(weight),
                        env_step,
                    )
        writer.flush()

    def _format_update_line(self, payload: Mapping[str, object]) -> str:
        update = payload["update"]
        rollout = payload["rollout"]
        optimization = payload["optimization"]
        done_reasons = payload["done_reasons"]
        timing = payload["timing"]
        curriculum = payload.get("curriculum")

        done_rates = done_reasons["rates"]
        dominant_done = sorted(
            (
                (label, rate)
                for label, rate in done_rates.items()
                if label != "unknown" and float(rate) > 0.0
            ),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:2]
        done_text = (
            " ".join(
                f"{_DONE_REASON_SHORT_LABELS.get(label, label)}={float(rate):.0%}"
                for label, rate in dominant_done
            )
            if dominant_done
            else "none"
        )

        eval_text = ""
        if isinstance(curriculum, Mapping) and isinstance(curriculum.get("eval"), Mapping):
            eval_payload = curriculum["eval"]
            eval_text = (
                f" | eval prog={float(eval_payload['mean_progress']):.3f}"
                f" comp={float(eval_payload['completion_rate']):.1%}"
                f" ret={float(eval_payload['mean_episode_return']):.1f}"
            )

        return (
            f"[u{int(update['index']):04d} {str(update['phase'])} p={float(update['progress']):.3f}] "
            f"train ret={float(rollout['mean_episode_return']):.1f} "
            f"len={float(rollout['mean_episode_length']):.0f} "
            f"r={float(rollout['mean_reward']):+.3f}"
            f"{eval_text}"
            f" | done {done_text}"
            f" | kl={float(optimization['approx_kl']):.4f}"
            f" clip={float(optimization['clip_fraction']):.3f}"
            f" lr={float(optimization['learning_rate']):.2e}"
            f" | {float(timing['env_steps_per_second'])/1000.0:.1f}k/s"
        )
