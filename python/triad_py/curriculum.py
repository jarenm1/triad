from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Sequence

from . import CurriculumStage


PRIMITIVE_STRAIGHT = 0
PRIMITIVE_CIRCLE_CW = 1
PRIMITIVE_CIRCLE_CCW = 2
PRIMITIVE_ZIGZAG = 3
PRIMITIVE_ELLIPSE = 4
PRIMITIVE_MIXED = 5


@dataclass(frozen=True)
class CurriculumAxes:
    gate_count_level: float = 0.0
    gate_size_level: float = 0.0
    spacing_level: float = 0.0
    verticality_level: float = 0.0
    gate_pose_noise_level: float = 0.0
    spawn_noise_level: float = 0.0
    dynamics_noise_level: float = 0.0
    obstacle_density_level: float = 0.0
    path_curvature_level: float = 0.0
    soft_failure_level: float = 0.0
    start_gate: int = 0

    def summary_difficulty(self) -> float:
        geometry = (
            self.gate_count_level,
            self.gate_size_level,
            self.spacing_level,
            self.verticality_level,
            self.path_curvature_level,
        )
        randomization = (
            self.gate_pose_noise_level,
            self.spawn_noise_level,
            self.dynamics_noise_level,
            self.obstacle_density_level,
        )
        return float((sum(geometry) + 0.5 * sum(randomization)) / 7.0)

    def as_reset_tuple_tail(
        self,
    ) -> tuple[float, float, float, float, float, float, float, float, float, float, int]:
        return (
            float(self.gate_count_level),
            float(self.gate_size_level),
            float(self.spacing_level),
            float(self.verticality_level),
            float(self.gate_pose_noise_level),
            float(self.spawn_noise_level),
            float(self.dynamics_noise_level),
            float(self.obstacle_density_level),
            float(self.path_curvature_level),
            float(self.soft_failure_level),
            int(self.start_gate),
        )


AxisRange = tuple[float, float]


@dataclass(frozen=True)
class CurriculumAxisRanges:
    gate_count_level: AxisRange = (0.0, 0.0)
    gate_size_level: AxisRange = (0.0, 0.0)
    spacing_level: AxisRange = (0.0, 0.0)
    verticality_level: AxisRange = (0.0, 0.0)
    gate_pose_noise_level: AxisRange = (0.0, 0.0)
    spawn_noise_level: AxisRange = (0.0, 0.0)
    dynamics_noise_level: AxisRange = (0.0, 0.0)
    obstacle_density_level: AxisRange = (0.0, 0.0)
    path_curvature_level: AxisRange = (0.0, 0.0)
    soft_failure_level: AxisRange = (0.0, 0.0)

    @staticmethod
    def _sample_range(rng: Random, value_range: AxisRange) -> float:
        low, high = value_range
        if high <= low:
            return float(low)
        return rng.uniform(float(low), float(high))

    def sample(self, rng: Random) -> CurriculumAxes:
        return CurriculumAxes(
            gate_count_level=self._sample_range(rng, self.gate_count_level),
            gate_size_level=self._sample_range(rng, self.gate_size_level),
            spacing_level=self._sample_range(rng, self.spacing_level),
            verticality_level=self._sample_range(rng, self.verticality_level),
            gate_pose_noise_level=self._sample_range(rng, self.gate_pose_noise_level),
            spawn_noise_level=self._sample_range(rng, self.spawn_noise_level),
            dynamics_noise_level=self._sample_range(rng, self.dynamics_noise_level),
            obstacle_density_level=self._sample_range(rng, self.obstacle_density_level),
            path_curvature_level=self._sample_range(rng, self.path_curvature_level),
            soft_failure_level=self._sample_range(rng, self.soft_failure_level),
        )


@dataclass(frozen=True)
class CurriculumPhase:
    name: str
    progress_end: float
    curriculum_stage: CurriculumStage
    difficulty_min: float
    difficulty_max: float
    grammar_ids: tuple[int, ...] = (
        PRIMITIVE_STRAIGHT,
        PRIMITIVE_CIRCLE_CW,
        PRIMITIVE_CIRCLE_CCW,
        PRIMITIVE_ZIGZAG,
        PRIMITIVE_ELLIPSE,
    )
    axis_ranges: CurriculumAxisRanges = field(default_factory=CurriculumAxisRanges)

    def sample_difficulty(self, rng: Random) -> float:
        if self.difficulty_max <= self.difficulty_min:
            return float(self.difficulty_min)
        return rng.uniform(self.difficulty_min, self.difficulty_max)

    def sample_grammar_id(self, rng: Random) -> int:
        return int(self.grammar_ids[rng.randrange(len(self.grammar_ids))])

    def sample_axes(self, rng: Random) -> CurriculumAxes:
        return self.axis_ranges.sample(rng)


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

    def phase_index_for_progress(self, progress: float) -> int:
        clamped = max(0.0, min(1.0, float(progress)))
        for phase_index, phase in enumerate(self.phases):
            if clamped <= phase.progress_end:
                return phase_index
        return len(self.phases) - 1

    def phase_at_index(self, phase_index: int) -> CurriculumPhase:
        if phase_index < 0 or phase_index >= len(self.phases):
            raise IndexError(
                f"phase_index out of range: {phase_index} not in [0, {len(self.phases)})"
            )
        return self.phases[phase_index]

    def sample_reset_params(
        self,
        env_count: int,
        progress: float,
        base_seed: int = 0,
    ) -> list[tuple]:
        if env_count <= 0:
            raise ValueError(f"env_count must be positive, got {env_count}")

        phase = self.phase_for_progress(progress)
        rng = Random(base_seed)
        reset_params: list[tuple] = []
        for env_index in range(env_count):
            env_seed = rng.getrandbits(32)
            env_rng = Random(env_seed ^ env_index)
            axes = phase.sample_axes(env_rng)
            reset_params.append(
                (
                    env_seed,
                    phase.sample_grammar_id(env_rng),
                    axes.summary_difficulty(),
                    int(phase.curriculum_stage),
                    *axes.as_reset_tuple_tail(),
                )
            )
        return reset_params

    def sample_reset_params_for_phase(
        self,
        env_count: int,
        phase_index: int,
        base_seed: int = 0,
    ) -> list[tuple]:
        if env_count <= 0:
            raise ValueError(f"env_count must be positive, got {env_count}")

        phase = self.phase_at_index(phase_index)
        rng = Random(base_seed)
        reset_params: list[tuple] = []
        for env_index in range(env_count):
            env_seed = rng.getrandbits(32)
            env_rng = Random(env_seed ^ env_index ^ (phase_index * 0x9E3779B9))
            axes = phase.sample_axes(env_rng)
            reset_params.append(
                (
                    env_seed,
                    phase.sample_grammar_id(env_rng),
                    axes.summary_difficulty(),
                    int(phase.curriculum_stage),
                    *axes.as_reset_tuple_tail(),
                )
            )
        return reset_params

    def sample_reset_params_mixture(
        self,
        env_count: int,
        phase_indices: Sequence[int],
        weights: Sequence[float],
        base_seed: int = 0,
    ) -> list[tuple]:
        if env_count <= 0:
            raise ValueError(f"env_count must be positive, got {env_count}")
        if len(phase_indices) != len(weights):
            raise ValueError("phase_indices and weights must have the same length")
        if not phase_indices:
            raise ValueError("phase_indices must not be empty")

        filtered: list[tuple[int, float]] = []
        for phase_index, weight in zip(phase_indices, weights, strict=True):
            if weight <= 0.0:
                continue
            self.phase_at_index(int(phase_index))
            filtered.append((int(phase_index), float(weight)))
        if not filtered:
            raise ValueError("at least one curriculum mixture weight must be positive")

        total_weight = sum(weight for _, weight in filtered)
        cumulative_weights: list[tuple[int, float]] = []
        running = 0.0
        for phase_index, weight in filtered:
            running += weight / total_weight
            cumulative_weights.append((phase_index, running))

        rng = Random(base_seed)
        reset_params: list[tuple] = []
        for env_index in range(env_count):
            draw = rng.random()
            selected_phase_index = cumulative_weights[-1][0]
            for phase_index, threshold in cumulative_weights:
                if draw <= threshold:
                    selected_phase_index = phase_index
                    break
            phase = self.phase_at_index(selected_phase_index)
            env_seed = rng.getrandbits(32)
            env_rng = Random(env_seed ^ env_index ^ (selected_phase_index * 0x9E3779B9))
            axes = phase.sample_axes(env_rng)
            reset_params.append(
                (
                    env_seed,
                    phase.sample_grammar_id(env_rng),
                    axes.summary_difficulty(),
                    int(phase.curriculum_stage),
                    *axes.as_reset_tuple_tail(),
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
                name="gate_approach",
                progress_end=0.12,
                curriculum_stage=CurriculumStage.BOOTSTRAP,
                difficulty_min=0.0,
                difficulty_max=0.012,
                grammar_ids=(PRIMITIVE_STRAIGHT,),
                axis_ranges=CurriculumAxisRanges(
                    gate_count_level=(0.0, 0.012),
                    gate_size_level=(0.0, 0.02),
                    spacing_level=(0.0, 0.03),
                    spawn_noise_level=(0.0, 0.04),
                ),
            ),
            CurriculumPhase(
                name="gate_pass",
                progress_end=0.26,
                curriculum_stage=CurriculumStage.BOOTSTRAP,
                difficulty_min=0.011,
                difficulty_max=0.039,
                grammar_ids=(PRIMITIVE_STRAIGHT,),
                axis_ranges=CurriculumAxisRanges(
                    gate_count_level=(0.016, 0.03),
                    gate_size_level=(0.02, 0.05),
                    spacing_level=(0.03, 0.08),
                    verticality_level=(0.0, 0.03),
                    spawn_noise_level=(0.02, 0.06),
                    path_curvature_level=(0.0, 0.05),
                ),
            ),
            CurriculumPhase(
                name="straight_intro",
                progress_end=0.36,
                curriculum_stage=CurriculumStage.INTRO,
                difficulty_min=0.020,
                difficulty_max=0.082,
                grammar_ids=(PRIMITIVE_STRAIGHT,),
                axis_ranges=CurriculumAxisRanges(
                    gate_count_level=(0.0, 0.025),
                    gate_size_level=(0.05, 0.12),
                    spacing_level=(0.06, 0.16),
                    verticality_level=(0.0, 0.06),
                    spawn_noise_level=(0.04, 0.10),
                    dynamics_noise_level=(0.02, 0.08),
                    path_curvature_level=(0.0, 0.12),
                ),
            ),
            CurriculumPhase(
                name="straight_chain",
                progress_end=0.46,
                curriculum_stage=CurriculumStage.INTRO,
                difficulty_min=0.038,
                difficulty_max=0.095,
                grammar_ids=(PRIMITIVE_STRAIGHT,),
                axis_ranges=CurriculumAxisRanges(
                    gate_count_level=(0.02, 0.038),
                    gate_size_level=(0.0, 0.08),
                    spacing_level=(0.08, 0.18),
                    verticality_level=(0.0, 0.04),
                    spawn_noise_level=(0.03, 0.09),
                    dynamics_noise_level=(0.0, 0.04),
                    path_curvature_level=(0.0, 0.02),
                ),
            ),
            CurriculumPhase(
                name="circle_intro",
                progress_end=0.56,
                curriculum_stage=CurriculumStage.INTRO,
                difficulty_min=0.032,
                difficulty_max=0.074,
                grammar_ids=(
                    PRIMITIVE_CIRCLE_CW,
                    PRIMITIVE_CIRCLE_CCW,
                ),
                axis_ranges=CurriculumAxisRanges(
                    gate_count_level=(0.0, 0.018),
                    gate_size_level=(0.0, 0.08),
                    spacing_level=(0.08, 0.18),
                    verticality_level=(0.0, 0.03),
                    spawn_noise_level=(0.02, 0.08),
                    dynamics_noise_level=(0.0, 0.04),
                    path_curvature_level=(0.0, 0.12),
                ),
            ),
            CurriculumPhase(
                name="circle_mastery",
                progress_end=0.66,
                curriculum_stage=CurriculumStage.INTRO,
                difficulty_min=0.063,
                difficulty_max=0.135,
                grammar_ids=(
                    PRIMITIVE_CIRCLE_CW,
                    PRIMITIVE_CIRCLE_CCW,
                ),
                axis_ranges=CurriculumAxisRanges(
                    gate_count_level=(0.02, 0.038),
                    gate_size_level=(0.04, 0.14),
                    spacing_level=(0.10, 0.24),
                    verticality_level=(0.0, 0.08),
                    spawn_noise_level=(0.04, 0.12),
                    dynamics_noise_level=(0.03, 0.10),
                    path_curvature_level=(0.10, 0.28),
                ),
            ),
            CurriculumPhase(
                name="circle_chain",
                progress_end=0.75,
                curriculum_stage=CurriculumStage.INTRO,
                difficulty_min=0.112,
                difficulty_max=0.211,
                grammar_ids=(
                    PRIMITIVE_CIRCLE_CW,
                    PRIMITIVE_CIRCLE_CCW,
                ),
                axis_ranges=CurriculumAxisRanges(
                    gate_count_level=(0.04, 0.075),
                    gate_size_level=(0.10, 0.22),
                    spacing_level=(0.14, 0.30),
                    verticality_level=(0.03, 0.12),
                    spawn_noise_level=(0.06, 0.15),
                    dynamics_noise_level=(0.06, 0.14),
                    path_curvature_level=(0.22, 0.42),
                ),
            ),
            CurriculumPhase(
                name="arena_straights",
                progress_end=0.82,
                curriculum_stage=CurriculumStage.ARENA,
                difficulty_min=0.077,
                difficulty_max=0.181,
                grammar_ids=(PRIMITIVE_STRAIGHT,),
                axis_ranges=CurriculumAxisRanges(
                    gate_count_level=(0.0, 0.08),
                    gate_size_level=(0.12, 0.25),
                    spacing_level=(0.14, 0.30),
                    verticality_level=(0.05, 0.18),
                    spawn_noise_level=(0.08, 0.18),
                    dynamics_noise_level=(0.08, 0.18),
                    path_curvature_level=(0.15, 0.28),
                ),
            ),
            CurriculumPhase(
                name="arena_zigzag",
                progress_end=0.89,
                curriculum_stage=CurriculumStage.ARENA,
                difficulty_min=0.140,
                difficulty_max=0.286,
                grammar_ids=(PRIMITIVE_ZIGZAG,),
                axis_ranges=CurriculumAxisRanges(
                    gate_count_level=(0.08, 0.18),
                    gate_size_level=(0.18, 0.34),
                    spacing_level=(0.22, 0.42),
                    verticality_level=(0.10, 0.26),
                    spawn_noise_level=(0.12, 0.24),
                    dynamics_noise_level=(0.12, 0.26),
                    path_curvature_level=(0.28, 0.55),
                ),
            ),
            CurriculumPhase(
                name="technical_turns",
                progress_end=0.945,
                curriculum_stage=CurriculumStage.TECHNICAL,
                difficulty_min=0.257,
                difficulty_max=0.451,
                grammar_ids=(
                    PRIMITIVE_CIRCLE_CW,
                    PRIMITIVE_CIRCLE_CCW,
                    PRIMITIVE_ZIGZAG,
                    PRIMITIVE_ELLIPSE,
                ),
                axis_ranges=CurriculumAxisRanges(
                    gate_count_level=(0.30, 0.55),
                    gate_size_level=(0.28, 0.48),
                    spacing_level=(0.30, 0.55),
                    verticality_level=(0.18, 0.42),
                    spawn_noise_level=(0.18, 0.34),
                    dynamics_noise_level=(0.20, 0.42),
                    path_curvature_level=(0.55, 0.78),
                ),
            ),
            CurriculumPhase(
                name="primitive_mix",
                progress_end=0.985,
                curriculum_stage=CurriculumStage.TECHNICAL,
                difficulty_min=0.356,
                difficulty_max=0.595,
                grammar_ids=(
                    PRIMITIVE_STRAIGHT,
                    PRIMITIVE_CIRCLE_CW,
                    PRIMITIVE_CIRCLE_CCW,
                    PRIMITIVE_ZIGZAG,
                    PRIMITIVE_ELLIPSE,
                    PRIMITIVE_MIXED,
                ),
                axis_ranges=CurriculumAxisRanges(
                    gate_count_level=(0.45, 0.75),
                    gate_size_level=(0.38, 0.62),
                    spacing_level=(0.42, 0.70),
                    verticality_level=(0.30, 0.62),
                    gate_pose_noise_level=(0.00, 0.06),
                    spawn_noise_level=(0.25, 0.45),
                    dynamics_noise_level=(0.34, 0.60),
                    path_curvature_level=(0.65, 0.92),
                ),
            ),
            CurriculumPhase(
                name="hard_lap_mix",
                progress_end=1.0,
                curriculum_stage=CurriculumStage.ELEVATED,
                difficulty_min=0.502,
                difficulty_max=0.771,
                grammar_ids=(
                    PRIMITIVE_STRAIGHT,
                    PRIMITIVE_CIRCLE_CW,
                    PRIMITIVE_CIRCLE_CCW,
                    PRIMITIVE_ZIGZAG,
                    PRIMITIVE_ELLIPSE,
                    PRIMITIVE_MIXED,
                ),
                axis_ranges=CurriculumAxisRanges(
                    gate_count_level=(0.70, 1.0),
                    gate_size_level=(0.55, 0.82),
                    spacing_level=(0.58, 0.90),
                    verticality_level=(0.45, 0.90),
                    gate_pose_noise_level=(0.02, 0.10),
                    spawn_noise_level=(0.35, 0.60),
                    dynamics_noise_level=(0.50, 0.85),
                    path_curvature_level=(0.80, 1.0),
                ),
            ),
        )
    )


def sample_curriculum_reset_params(
    env_count: int,
    progress: float,
    base_seed: int = 0,
    schedule: CurriculumSchedule | None = None,
) -> list[tuple]:
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
