from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Sequence

from . import CurriculumStage


@dataclass(frozen=True)
class CurriculumPhase:
    name: str
    progress_end: float
    curriculum_stage: CurriculumStage
    difficulty_min: float
    difficulty_max: float
    grammar_ids: tuple[int, ...] = (0, 1, 2, 3)

    def sample_difficulty(self, rng: Random) -> float:
        if self.difficulty_max <= self.difficulty_min:
            return float(self.difficulty_min)
        return rng.uniform(self.difficulty_min, self.difficulty_max)

    def sample_grammar_id(self, rng: Random) -> int:
        return int(self.grammar_ids[rng.randrange(len(self.grammar_ids))])


@dataclass(frozen=True)
class CurriculumSchedule:
    phases: tuple[CurriculumPhase, ...]

    def __post_init__(self) -> None:
        if not self.phases:
            raise ValueError("curriculum schedule requires at least one phase")

        previous_end = 0.0
        for phase in self.phases:
            if phase.progress_end <= previous_end:
                raise ValueError("curriculum phases must have increasing progress_end")
            if phase.progress_end > 1.0:
                raise ValueError("curriculum phase progress_end must be <= 1.0")
            if not phase.grammar_ids:
                raise ValueError("curriculum phase must allow at least one grammar_id")
            previous_end = phase.progress_end

        if self.phases[-1].progress_end != 1.0:
            raise ValueError("last curriculum phase must end at progress 1.0")

    def phase_for_progress(self, progress: float) -> CurriculumPhase:
        clamped = max(0.0, min(1.0, float(progress)))
        for phase in self.phases:
            if clamped <= phase.progress_end:
                return phase
        return self.phases[-1]

    def sample_reset_params(
        self,
        env_count: int,
        progress: float,
        base_seed: int = 0,
    ) -> list[tuple[int, int, float, int]]:
        if env_count <= 0:
            raise ValueError(f"env_count must be positive, got {env_count}")

        phase = self.phase_for_progress(progress)
        rng = Random(base_seed)
        reset_params: list[tuple[int, int, float, int]] = []
        for env_index in range(env_count):
            env_seed = rng.getrandbits(32)
            env_rng = Random(env_seed ^ env_index)
            reset_params.append(
                (
                    env_seed,
                    phase.sample_grammar_id(env_rng),
                    phase.sample_difficulty(env_rng),
                    int(phase.curriculum_stage),
                )
            )
        return reset_params


@dataclass(frozen=True)
class CurriculumProgression:
    total_updates: int
    warmup_updates: int = 0
    start_progress: float = 0.0
    end_progress: float = 1.0

    def __post_init__(self) -> None:
        if self.total_updates <= 0:
            raise ValueError("total_updates must be positive")
        if self.warmup_updates < 0:
            raise ValueError("warmup_updates must be non-negative")

    def progress_for_update(self, update_index: int) -> float:
        if update_index < 0:
            raise ValueError("update_index must be non-negative")
        if update_index <= self.warmup_updates:
            return float(self.start_progress)

        ramp_updates = max(1, self.total_updates - self.warmup_updates)
        ramp_index = min(update_index - self.warmup_updates, ramp_updates)
        alpha = ramp_index / ramp_updates
        progress = self.start_progress + (
            (self.end_progress - self.start_progress) * alpha
        )
        return max(0.0, min(1.0, float(progress)))

    def apply(
        self,
        sim: object,
        update_index: int,
        base_seed: int = 0,
        schedule: CurriculumSchedule | None = None,
    ) -> list[tuple[int, int, float, int]]:
        progress = self.progress_for_update(update_index)
        active_schedule = schedule or build_teacher_curriculum_schedule()
        reset_params = active_schedule.sample_reset_params(
            env_count=int(sim.env_count),
            progress=progress,
            base_seed=base_seed,
        )
        sim.set_reset_params(reset_params)
        return reset_params


def build_teacher_curriculum_schedule() -> CurriculumSchedule:
    return CurriculumSchedule(
        phases=(
            CurriculumPhase(
                name="intro",
                progress_end=0.2,
                curriculum_stage=CurriculumStage.INTRO,
                difficulty_min=0.0,
                difficulty_max=0.15,
                grammar_ids=(0, 1),
            ),
            CurriculumPhase(
                name="offset",
                progress_end=0.45,
                curriculum_stage=CurriculumStage.ARENA,
                difficulty_min=0.1,
                difficulty_max=0.35,
                grammar_ids=(0, 1, 2),
            ),
            CurriculumPhase(
                name="arena",
                progress_end=0.75,
                curriculum_stage=CurriculumStage.TECHNICAL,
                difficulty_min=0.3,
                difficulty_max=0.65,
                grammar_ids=(0, 1, 2, 3),
            ),
            CurriculumPhase(
                name="hard",
                progress_end=1.0,
                curriculum_stage=CurriculumStage.ELEVATED,
                difficulty_min=0.6,
                difficulty_max=1.0,
                grammar_ids=(0, 1, 2, 3),
            ),
        )
    )


def sample_curriculum_reset_params(
    env_count: int,
    progress: float,
    base_seed: int = 0,
    schedule: CurriculumSchedule | None = None,
) -> list[tuple[int, int, float, int]]:
    active_schedule = schedule or build_teacher_curriculum_schedule()
    return active_schedule.sample_reset_params(env_count, progress, base_seed)


def build_teacher_curriculum_progression(
    total_updates: int,
    warmup_updates: int = 0,
) -> CurriculumProgression:
    return CurriculumProgression(
        total_updates=total_updates,
        warmup_updates=warmup_updates,
    )
