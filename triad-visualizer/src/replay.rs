use triad_sim::{Action, EnvState, Gate, RewardDone};

use crate::render::autopilot_action;
use crate::ui::format_done_reasons;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReplayPlayback {
    Paused,
    Playing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReplayController {
    Heuristic,
}

#[derive(Debug)]
pub(crate) struct ReplayState {
    pub(crate) playback: ReplayPlayback,
    pub(crate) controller: ReplayController,
    pub(crate) model_path: String,
    pub(crate) auto_pause_on_terminal: bool,
    pub(crate) request_step: bool,
    pub(crate) request_load_model: bool,
    pub(crate) status: String,
}

impl Default for ReplayState {
    fn default() -> Self {
        Self {
            playback: ReplayPlayback::Paused,
            controller: ReplayController::Heuristic,
            model_path: String::new(),
            auto_pause_on_terminal: true,
            request_step: false,
            request_load_model: false,
            status: "Heuristic replay ready".to_string(),
        }
    }
}

impl ReplayState {
    pub(crate) fn snapshot(&mut self) -> ReplaySnapshot {
        let snapshot = ReplaySnapshot {
            playback: self.playback,
            controller: self.controller,
            model_path: self.model_path.clone(),
            auto_pause_on_terminal: self.auto_pause_on_terminal,
            step_requested: self.request_step,
            load_model_requested: self.request_load_model,
        };
        self.request_step = false;
        self.request_load_model = false;
        snapshot
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ReplaySnapshot {
    pub(crate) playback: ReplayPlayback,
    pub(crate) controller: ReplayController,
    pub(crate) model_path: String,
    pub(crate) auto_pause_on_terminal: bool,
    pub(crate) step_requested: bool,
    pub(crate) load_model_requested: bool,
}

impl ReplaySnapshot {
    pub(crate) fn should_advance(&self) -> bool {
        self.playback == ReplayPlayback::Playing || self.step_requested
    }
}

pub(crate) fn apply_replay_controller(
    controller: ReplayController,
    actions: &mut [Action],
    states: &[EnvState],
    target_gate_for_env: impl Fn(usize, &EnvState) -> Option<Gate>,
) -> &'static str {
    actions.fill(Action::idle());
    match controller {
        ReplayController::Heuristic => {
            for (env_index, state) in states.iter().copied().enumerate() {
                if let Some(target_gate) = target_gate_for_env(env_index, &state) {
                    actions[env_index] = autopilot_action(state, target_gate);
                }
            }
            "Replay active with heuristic controller"
        }
    }
}

pub(crate) fn terminal_pause_status(reward_done: RewardDone) -> String {
    format!(
        "Replay paused at terminal state: {}",
        format_done_reasons(reward_done.done_reason)
    )
}
