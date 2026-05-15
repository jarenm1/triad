use std::io::{BufRead, BufReader, Write};
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};
#[cfg(not(target_arch = "wasm32"))]
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use triad_sim::{Action, EnvState, Gate, Observation, RewardDone};

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
    Policy,
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
        ReplayController::Policy => "Replay active with PPO policy",
    }
}

pub(crate) fn terminal_pause_status(reward_done: RewardDone) -> String {
    format!(
        "Replay paused at terminal state: {}",
        format_done_reasons(reward_done.done_reason)
    )
}

const POLICY_OBSERVATION_DIM: usize = 37;
const POLICY_ACTION_DIM: usize = 4;

pub(crate) fn observation_to_vec(observation: Observation) -> Vec<f32> {
    vec![
        observation.position[0],
        observation.position[1],
        observation.position[2],
        observation.velocity[0],
        observation.velocity[1],
        observation.velocity[2],
        observation.attitude[0],
        observation.attitude[1],
        observation.attitude[2],
        observation.angular_velocity[0],
        observation.angular_velocity[1],
        observation.angular_velocity[2],
        observation.target_gate_position[0],
        observation.target_gate_position[1],
        observation.target_gate_position[2],
        observation.target_gate_forward[0],
        observation.target_gate_forward[1],
        observation.target_gate_forward[2],
        observation.progress,
        observation.distance_to_gate,
        observation.gate_alignment,
        observation.mean_motor_thrust,
        observation.privileged_velocity_body[0],
        observation.privileged_velocity_body[1],
        observation.privileged_velocity_body[2],
        observation.privileged_target_gate_body[0],
        observation.privileged_target_gate_body[1],
        observation.privileged_target_gate_body[2],
        observation.privileged_target_gate_forward_body[0],
        observation.privileged_target_gate_forward_body[1],
        observation.privileged_target_gate_forward_body[2],
        observation.privileged_next_gate_body[0],
        observation.privileged_next_gate_body[1],
        observation.privileged_next_gate_body[2],
        observation.privileged_next_gate_forward_body[0],
        observation.privileged_next_gate_forward_body[1],
        observation.privileged_next_gate_forward_body[2],
    ]
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct PolicyClient {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

#[cfg(not(target_arch = "wasm32"))]
impl PolicyClient {
    pub(crate) fn spawn(checkpoint_path: &Path) -> Result<Self, String> {
        let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .ok_or_else(|| "workspace root not found".to_string())?
            .to_path_buf();
        let mut candidates = Vec::new();
        if let Some(requested) = std::env::var("TRIAD_VISUALIZER_PYTHON")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
        {
            candidates.push(requested);
        }
        if let Some(active_venv) = std::env::var("VIRTUAL_ENV")
            .ok()
            .map(PathBuf::from)
            .filter(|path| path.join("bin/python").is_file())
        {
            candidates.push(active_venv.join("bin/python").display().to_string());
        }
        let workspace_venv = workspace_root.join(".venv").join("bin/python");
        if workspace_venv.is_file() {
            candidates.push(workspace_venv.display().to_string());
        }
        candidates.push("python3".to_string());
        candidates.push("python".to_string());
        candidates.dedup();

        let checkpoint = checkpoint_path
            .canonicalize()
            .unwrap_or_else(|_| checkpoint_path.to_path_buf());

        let mut last_error = None;
        for program in candidates {
            match Self::spawn_with_program(&program, &workspace_root, &checkpoint) {
                Ok(client) => return Ok(client),
                Err(error) => last_error = Some(format!("{program}: {error}")),
            }
        }

        Err(last_error.unwrap_or_else(|| "failed to start policy server".to_string()))
    }

    fn spawn_with_program(
        program: &str,
        workspace_root: &Path,
        checkpoint_path: &Path,
    ) -> Result<Self, String> {
        let mut child = Command::new(program)
            .arg("-u")
            .arg("-m")
            .arg("triad_py")
            .arg("ppo-policy-server")
            .arg("--checkpoint")
            .arg(checkpoint_path)
            .current_dir(workspace_root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| format!("spawn failed: {error}"))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "policy server stdin unavailable".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "policy server stdout unavailable".to_string())?;

        let mut client = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        };

        let mut ready = String::new();
        client
            .stdout
            .read_line(&mut ready)
            .map_err(|error| format!("failed to read policy server readiness: {error}"))?;
        if ready.trim() != "ready" {
            return Err(if ready.trim().is_empty() {
                "policy server exited before becoming ready".to_string()
            } else {
                format!("unexpected policy server handshake: {}", ready.trim())
            });
        }

        Ok(client)
    }

    pub(crate) fn infer(
        &mut self,
        observations: &[Observation],
        actions: &mut [Action],
    ) -> Result<(), String> {
        let observation_count = observations.len();
        if actions.len() != observation_count {
            return Err(format!(
                "policy input count mismatch: {} observations for {} action slots",
                observation_count,
                actions.len()
            ));
        }

        let mut request =
            String::with_capacity(32 + observation_count * POLICY_OBSERVATION_DIM * 12);
        request.push_str("predict ");
        request.push_str(&observation_count.to_string());
        request.push(' ');
        request.push_str(&POLICY_OBSERVATION_DIM.to_string());
        request.push_str(" 1");
        for observation in observations.iter().copied() {
            for value in observation_to_vec(observation) {
                request.push(' ');
                request.push_str(&value.to_string());
            }
        }
        request.push('\n');

        self.stdin
            .write_all(request.as_bytes())
            .and_then(|_| self.stdin.flush())
            .map_err(|error| format!("failed to write policy request: {error}"))?;

        let mut response = String::new();
        self.stdout
            .read_line(&mut response)
            .map_err(|error| format!("failed to read policy response: {error}"))?;
        if response.trim().is_empty() {
            return Err("policy server returned empty response".to_string());
        }

        let trimmed = response.trim();
        if let Some(error) = trimmed.strip_prefix("error ") {
            return Err(error.to_string());
        }
        let mut tokens = trimmed.split_whitespace();
        let Some(kind) = tokens.next() else {
            return Err("policy response was empty".to_string());
        };
        if kind != "ok" {
            return Err(format!("unknown policy response kind: {kind}"));
        }

        let response_env_count = tokens
            .next()
            .ok_or_else(|| "policy response missing env count".to_string())?
            .parse::<usize>()
            .map_err(|error| format!("invalid policy env count: {error}"))?;
        if response_env_count != actions.len() {
            return Err(format!(
                "policy action count mismatch: expected {}, got {}",
                actions.len(),
                response_env_count
            ));
        }

        let response_action_dim = tokens
            .next()
            .ok_or_else(|| "policy response missing action width".to_string())?
            .parse::<usize>()
            .map_err(|error| format!("invalid policy action width: {error}"))?;
        if response_action_dim != POLICY_ACTION_DIM {
            return Err(format!(
                "policy action width mismatch: expected {}, got {}",
                POLICY_ACTION_DIM, response_action_dim
            ));
        }

        for slot in actions.iter_mut() {
            let mut command = [0.5; POLICY_ACTION_DIM];
            for value in &mut command {
                *value = tokens
                    .next()
                    .ok_or_else(|| "policy response had too few action values".to_string())?
                    .parse::<f32>()
                    .map_err(|error| format!("invalid policy action value: {error}"))?;
            }
            *slot = Action::new(command);
        }
        if tokens.next().is_some() {
            return Err("policy response had extra trailing action values".to_string());
        }

        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for PolicyClient {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "shutdown");
        let _ = self.stdin.flush();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
