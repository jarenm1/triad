use crate::constants::{
    DONE_REASON_COMPLETE, DONE_REASON_EXCESSIVE_TILT, DONE_REASON_FLOOR_COLLISION,
    DONE_REASON_GATE_COLLISION, DONE_REASON_OBSTACLE_COLLISION, DONE_REASON_OUT_OF_BOUNDS,
    DONE_REASON_STEP_LIMIT, VISUALIZER_ENV_COUNT,
};
use crate::replay::{ReplayPlayback, ReplaySnapshot, ReplayState};
use triad_window::egui;

#[derive(Debug)]
pub(crate) struct UiState {
    pub(crate) replay: ReplayState,
    pub(crate) selected_env: usize,
    pub(crate) difficulty: f32,
    pub(crate) curriculum_phase: usize,
    pub(crate) curriculum_stage: u32,
    pub(crate) seed_base: u32,
    pub(crate) request_reset_selected: bool,
    pub(crate) request_reset_all: bool,
    pub(crate) request_randomize: bool,
    pub(crate) gate_count: u32,
    pub(crate) current_gate: u32,
    pub(crate) done: bool,
    pub(crate) position: [f32; 3],
    pub(crate) reward: f32,
    pub(crate) done_reason_bits: u32,
    pub(crate) progress: f32,
    pub(crate) distance_to_gate: f32,
    pub(crate) gate_alignment: f32,
    pub(crate) mean_motor_thrust: f32,
    pub(crate) shaping_reward: f32,
    pub(crate) time_penalty: f32,
    pub(crate) sparse_objective_reward: f32,
    pub(crate) collision_penalty: f32,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            replay: ReplayState::default(),
            selected_env: 0,
            difficulty: 0.35,
            curriculum_phase: 0,
            curriculum_stage: 0,
            seed_base: 1,
            request_reset_selected: false,
            request_reset_all: false,
            request_randomize: false,
            gate_count: 0,
            current_gate: 0,
            done: false,
            position: [0.0, 0.0, 0.0],
            reward: 0.0,
            done_reason_bits: 0,
            progress: 0.0,
            distance_to_gate: 0.0,
            gate_alignment: 0.0,
            mean_motor_thrust: 0.0,
            shaping_reward: 0.0,
            time_penalty: 0.0,
            sparse_objective_reward: 0.0,
            collision_penalty: 0.0,
        }
    }
}

impl UiState {
    pub(crate) fn snapshot(&mut self, env_count: usize) -> UiSnapshot {
        self.curriculum_phase = self
            .curriculum_phase
            .min(CURRICULUM_PHASES.len().saturating_sub(1));
        self.curriculum_stage = curriculum_phase_profile(self.curriculum_phase).curriculum_stage;

        let snapshot = UiSnapshot {
            replay: self.replay.snapshot(),
            selected_env: self.selected_env.min(env_count.saturating_sub(1)),
            difficulty: self.difficulty,
            curriculum_phase: self.curriculum_phase,
            curriculum_stage: self.curriculum_stage,
            seed_base: self.seed_base,
            reset_selected: self.request_reset_selected,
            reset_all: self.request_reset_all,
            randomize: self.request_randomize,
        };

        self.request_reset_selected = false;
        self.request_reset_all = false;
        self.request_randomize = false;
        snapshot
    }
}

#[derive(Debug, Clone)]
pub(crate) struct UiSnapshot {
    pub(crate) replay: ReplaySnapshot,
    pub(crate) selected_env: usize,
    pub(crate) difficulty: f32,
    pub(crate) curriculum_phase: usize,
    pub(crate) curriculum_stage: u32,
    pub(crate) seed_base: u32,
    pub(crate) reset_selected: bool,
    pub(crate) reset_all: bool,
    pub(crate) randomize: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct CurriculumPhaseProfile {
    pub(crate) name: &'static str,
    pub(crate) curriculum_stage: u32,
    pub(crate) grammar_ids: &'static [u32],
    pub(crate) difficulty_min: f32,
    pub(crate) difficulty_max: f32,
}

pub(crate) const CURRICULUM_PHASES: &[CurriculumPhaseProfile] = &[
    CurriculumPhaseProfile {
        name: "discover_gate",
        curriculum_stage: 0,
        grammar_ids: &[0],
        difficulty_min: 0.0,
        difficulty_max: 0.005,
    },
    CurriculumPhaseProfile {
        name: "align_gate",
        curriculum_stage: 0,
        grammar_ids: &[0],
        difficulty_min: 0.005,
        difficulty_max: 0.015,
    },
    CurriculumPhaseProfile {
        name: "pass_gate",
        curriculum_stage: 0,
        grammar_ids: &[0],
        difficulty_min: 0.015,
        difficulty_max: 0.03,
    },
    CurriculumPhaseProfile {
        name: "exit_gate",
        curriculum_stage: 1,
        grammar_ids: &[0],
        difficulty_min: 0.0,
        difficulty_max: 0.02,
    },
    CurriculumPhaseProfile {
        name: "chain_two",
        curriculum_stage: 1,
        grammar_ids: &[0],
        difficulty_min: 0.02,
        difficulty_max: 0.04,
    },
    CurriculumPhaseProfile {
        name: "offset",
        curriculum_stage: 2,
        grammar_ids: &[0, 1],
        difficulty_min: 0.05,
        difficulty_max: 0.22,
    },
    CurriculumPhaseProfile {
        name: "arena",
        curriculum_stage: 3,
        grammar_ids: &[0, 1, 2, 3],
        difficulty_min: 0.3,
        difficulty_max: 0.65,
    },
    CurriculumPhaseProfile {
        name: "hard",
        curriculum_stage: 4,
        grammar_ids: &[0, 1, 2, 3],
        difficulty_min: 0.6,
        difficulty_max: 1.0,
    },
];

pub(crate) fn curriculum_phase_profile(curriculum_phase: usize) -> CurriculumPhaseProfile {
    CURRICULUM_PHASES[curriculum_phase.min(CURRICULUM_PHASES.len().saturating_sub(1))]
}

pub(crate) fn draw_visualizer_ui(ctx: &egui::Context, ui: &mut UiState) {
    egui::Window::new("Visualizer")
        .default_pos(egui::pos2(12.0, 84.0))
        .show(ctx, |panel| {
            panel.horizontal(|row| {
                if row.button("Play").clicked() {
                    ui.replay.playback = ReplayPlayback::Playing;
                }
                if row.button("Pause").clicked() {
                    ui.replay.playback = ReplayPlayback::Paused;
                }
                if row.button("Step").clicked() {
                    ui.replay.request_step = true;
                    ui.replay.playback = ReplayPlayback::Paused;
                }
            });
            panel.checkbox(&mut ui.replay.auto_pause_on_terminal, "Auto Pause Terminal");
            panel.label(format!("Replay Status: {}", ui.replay.status));
            panel.horizontal(|row| {
                row.label("Model");
                row.text_edit_singleline(&mut ui.replay.model_path);
                if row.button("Browse").clicked() {
                    if let Some(path) = pick_model_file() {
                        ui.replay.model_path = path;
                        ui.replay.request_load_model = true;
                    }
                }
                if row.button("Load").clicked() {
                    ui.replay.request_load_model = true;
                }
            });
            if panel.button("Reset Selected").clicked() {
                ui.request_reset_selected = true;
            }
            if panel.button("Reset All").clicked() {
                ui.request_reset_all = true;
            }
            if panel.button("Randomize").clicked() {
                ui.request_randomize = true;
                ui.seed_base = ui.seed_base.wrapping_add(1);
            }

            panel.separator();
            panel.add(
                egui::Slider::new(
                    &mut ui.selected_env,
                    0..=(VISUALIZER_ENV_COUNT.saturating_sub(1)),
                )
                .text("Selected Env"),
            );
            panel.add(egui::Slider::new(&mut ui.difficulty, 0.0..=1.0).text("Phase Difficulty"));

            let selected_phase = ui
                .curriculum_phase
                .min(CURRICULUM_PHASES.len().saturating_sub(1));
            egui::ComboBox::from_label("Curriculum Phase")
                .selected_text(CURRICULUM_PHASES[selected_phase].name)
                .show_ui(panel, |combo| {
                    for (phase_index, phase) in CURRICULUM_PHASES.iter().enumerate() {
                        combo.selectable_value(&mut ui.curriculum_phase, phase_index, phase.name);
                    }
                });

            ui.curriculum_phase = ui
                .curriculum_phase
                .min(CURRICULUM_PHASES.len().saturating_sub(1));
            let phase = CURRICULUM_PHASES[ui.curriculum_phase];
            ui.curriculum_stage = phase.curriculum_stage;
            let actual_difficulty = phase.difficulty_min
                + ui.difficulty.clamp(0.0, 1.0)
                    * (phase.difficulty_max - phase.difficulty_min).max(0.0);
            panel.label(format!(
                "Sim Stage: {} | Difficulty: {:.4}..{:.4} -> {:.4}",
                phase.curriculum_stage,
                phase.difficulty_min,
                phase.difficulty_max,
                actual_difficulty
            ));

            panel.separator();
            panel.label(format!("Gate Count: {}", ui.gate_count));
            panel.label(format!("Current Gate: {}", ui.current_gate));
            panel.label(format!("Done: {}", ui.done));
            panel.label(format!(
                "Done Reason: {}",
                format_done_reasons(ui.done_reason_bits)
            ));
            panel.label(format!("Reward: {:.3}", ui.reward));
            panel.label(format!("Progress: {:.3}", ui.progress));
            panel.label(format!("Distance To Gate: {:.3}", ui.distance_to_gate));
            panel.label(format!("Gate Alignment: {:.3}", ui.gate_alignment));
            panel.label(format!("Mean Motor Thrust: {:.3}", ui.mean_motor_thrust));
            panel.label(format!(
                "Position: {:.2}, {:.2}, {:.2}",
                ui.position[0], ui.position[1], ui.position[2]
            ));

            panel.separator();
            panel.label("Reward Breakdown");
            panel.label(format!("  shaping: {:+.3}", ui.shaping_reward));
            panel.label(format!(
                "  sparse objective: +{:.3}",
                ui.sparse_objective_reward
            ));
            panel.label(format!("  time: -{:.3}", ui.time_penalty));
            panel.label(format!("  collision: -{:.3}", ui.collision_penalty));
        });
}

pub(crate) fn format_done_reasons(done_reason_bits: u32) -> String {
    if done_reason_bits == 0 {
        return "none".to_string();
    }

    let mut labels = Vec::new();
    if done_reason_bits & DONE_REASON_COMPLETE != 0 {
        labels.push("complete");
    }
    if done_reason_bits & DONE_REASON_GATE_COLLISION != 0 {
        labels.push("gate_collision");
    }
    if done_reason_bits & DONE_REASON_OBSTACLE_COLLISION != 0 {
        labels.push("obstacle_collision");
    }
    if done_reason_bits & DONE_REASON_FLOOR_COLLISION != 0 {
        labels.push("floor_collision");
    }
    if done_reason_bits & DONE_REASON_OUT_OF_BOUNDS != 0 {
        labels.push("out_of_bounds");
    }
    if done_reason_bits & DONE_REASON_STEP_LIMIT != 0 {
        labels.push("step_limit");
    }
    if done_reason_bits & DONE_REASON_EXCESSIVE_TILT != 0 {
        labels.push("excessive_tilt");
    }
    labels.join(", ")
}

#[cfg(not(target_arch = "wasm32"))]
fn pick_model_file() -> Option<String> {
    rfd::FileDialog::new()
        .set_title("Load Replay Model")
        .add_filter("Model checkpoint", &["pt", "pth", "onnx"])
        .add_filter("All files", &["*"])
        .pick_file()
        .map(|path| path.display().to_string())
}

#[cfg(target_arch = "wasm32")]
fn pick_model_file() -> Option<String> {
    None
}
