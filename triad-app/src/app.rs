//! Application state and main run loop.

use crate::layers::LayerMode;
use crate::multi_delegate::{MultiDelegate, MultiInitData};
use glam::Vec3;
use std::error::Error;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use triad_gpu::{SceneBounds, ply_loader};
use triad_window::{
    CameraControl, CameraIntent, CameraPose, FrameUpdate, InputState,
    IntentMode, KeyCode, MouseButton, PhysicalKey,
};

/// Shared mode state that can be read by the delegate and written by the UI/input handler.
/// Uses atomic operations for thread-safe access.
pub type ModeSignal = Arc<AtomicU8>;

/// Create a new mode signal with the given initial mode.
pub fn create_mode_signal(mode: LayerMode) -> ModeSignal {
    Arc::new(AtomicU8::new(mode as u8))
}

/// Read the current mode from a signal.
pub fn read_mode(signal: &ModeSignal) -> LayerMode {
    match signal.load(Ordering::Relaxed) {
        0 => LayerMode::Points,
        1 => LayerMode::Gaussians,
        2 => LayerMode::Triangles,
        _ => LayerMode::Points,
    }
}

/// Write a new mode to a signal.
pub fn write_mode(signal: &ModeSignal, mode: LayerMode) {
    signal.store(mode as u8, Ordering::Relaxed);
}

/// Run the application with unified multi-mode rendering.
pub fn run(file: Option<PathBuf>, mode: &str) -> Result<(), Box<dyn Error>> {
    let initial_mode = match mode.to_lowercase().as_str() {
        "points" | "point" => LayerMode::Points,
        "gaussians" | "gaussian" => LayerMode::Gaussians,
        "triangles" | "triangle" => LayerMode::Triangles,
        _ => {
            eprintln!("Unknown mode '{}', defaulting to points", mode);
            LayerMode::Points
        }
    };

    let file = file.unwrap_or_else(|| {
        let default = PathBuf::from("goat.ply");
        if default.exists() {
            default
        } else {
            eprintln!("No PLY file specified and no default file found");
            std::process::exit(1);
        }
    });

    // Compute scene center for orbit camera
    let scene_center = compute_scene_center(&file)?;

    // Create shared mode signal
    let mode_signal = create_mode_signal(initial_mode);
    
    // Create init data for multi-delegate
    let init_data = MultiInitData::new(file, initial_mode, mode_signal.clone());

    // Clone for the callback
    let mode_signal_cb = mode_signal.clone();

    triad_window::run_with_delegate_config::<MultiDelegate, _>(
        "Triad Viewer - Press 1/2/3 to switch modes",
        init_data,
        move |controls| {
            let target = scene_center;
            let mode_signal = mode_signal_cb;

            controls
                .clear_controllers()
                .single_active(true)
                .add_controller_with_priority(Box::new(OrbitCam::new(target, 10.0)), 0)
                .on_frame(move |ctx: FrameUpdate| {
                    // Mode switching with number keys
                    if ctx.input.just_pressed(PhysicalKey::Code(KeyCode::Digit1)) {
                        write_mode(&mode_signal, LayerMode::Points);
                        tracing::info!("Switched to Points mode");
                    }
                    if ctx.input.just_pressed(PhysicalKey::Code(KeyCode::Digit2)) {
                        write_mode(&mode_signal, LayerMode::Gaussians);
                        tracing::info!("Switched to Gaussians mode");
                    }
                    if ctx.input.just_pressed(PhysicalKey::Code(KeyCode::Digit3)) {
                        write_mode(&mode_signal, LayerMode::Triangles);
                        tracing::info!("Switched to Triangles mode");
                    }

                    // Tab to cycle modes
                    if ctx.input.just_pressed(PhysicalKey::Code(KeyCode::Tab)) {
                        let current = read_mode(&mode_signal);
                        write_mode(&mode_signal, current.next());
                        tracing::info!("Cycled to {} mode", read_mode(&mode_signal));
                    }

                    // Reset camera with R
                    if ctx.input.just_pressed(PhysicalKey::Code(KeyCode::KeyR)) {
                        let mut pose = *ctx.pose;
                        pose.position = target + Vec3::new(0.0, 0.0, 10.0);
                        let forward = (target - pose.position).normalize_or_zero();
                        pose.yaw = forward.x.atan2(-forward.z);
                        pose.pitch = forward.y.asin().clamp(
                            -std::f32::consts::FRAC_PI_2,
                            std::f32::consts::FRAC_PI_2,
                        );
                        pose.roll = 0.0;
                        *ctx.reset_request = Some(pose);
                    }

                    // Help
                    if ctx.input.just_pressed(PhysicalKey::Code(KeyCode::KeyH)) {
                        tracing::info!("=== Controls ===");
                        tracing::info!("1 - Points mode");
                        tracing::info!("2 - Gaussians mode");
                        tracing::info!("3 - Triangles mode");
                        tracing::info!("Tab - Cycle modes");
                        tracing::info!("R - Reset camera");
                        tracing::info!("H - Show this help");
                        tracing::info!("ESC - Quit");
                        tracing::info!("Left mouse - Orbit");
                        tracing::info!("Shift+Left mouse - Pan");
                        tracing::info!("Scroll - Zoom");
                    }
                });
        },
    )
}

fn compute_scene_center(ply_path: &PathBuf) -> Result<Vec3, Box<dyn Error>> {
    let ply_path_str = ply_path
        .to_str()
        .ok_or_else(|| format!("PLY path {:?} is not valid UTF-8", ply_path))?;

    let vertices = ply_loader::load_vertices_from_ply(ply_path_str)?;
    
    if vertices.is_empty() {
        return Ok(Vec3::ZERO);
    }

    let bounds = SceneBounds::from_positions(vertices.iter().map(|v| v.position));
    Ok(bounds.center)
}

/// Blender-like orbit camera controller.
struct OrbitCam {
    focus: Vec3,
    yaw: f32,
    pitch: f32,
    radius: f32,
    rotation_sensitivity: f32,
    pan_sensitivity: f32,
    zoom_sensitivity: f32,
}

impl OrbitCam {
    fn new(focus: Vec3, radius: f32) -> Self {
        Self {
            focus,
            yaw: 0.0,
            pitch: 0.0,
            radius,
            rotation_sensitivity: 0.005,
            pan_sensitivity: 0.0025,
            zoom_sensitivity: 0.5,
        }
    }

    fn pose_from_state(&self) -> CameraPose {
        let dir = glam::Quat::from_euler(glam::EulerRot::YXZ, self.yaw, self.pitch, 0.0) * -Vec3::Z;
        let position = self.focus - dir.normalize() * self.radius;
        let forward = (self.focus - position).normalize_or_zero();
        let yaw = forward.x.atan2(-forward.z);
        let pitch = forward.y.asin().clamp(
            -std::f32::consts::FRAC_PI_2,
            std::f32::consts::FRAC_PI_2,
        );
        CameraPose {
            position,
            yaw,
            pitch,
            roll: 0.0,
        }
    }
}

impl CameraControl for OrbitCam {
    fn update(&mut self, _dt: f32, input: &InputState, _current: &CameraPose) -> Option<CameraIntent> {
        let delta = input.mouse_delta();
        let shift = input.key_down(PhysicalKey::Code(KeyCode::ShiftLeft))
            || input.key_down(PhysicalKey::Code(KeyCode::ShiftRight));

        if input.mouse_button_down(MouseButton::Left) && !shift {
            self.yaw -= delta.x * self.rotation_sensitivity;
            self.pitch = (self.pitch - delta.y * self.rotation_sensitivity).clamp(
                -std::f32::consts::FRAC_PI_2 + 0.01,
                std::f32::consts::FRAC_PI_2 - 0.01,
            );
        } else if input.mouse_button_down(MouseButton::Left) && shift {
            let right = glam::Quat::from_euler(glam::EulerRot::YXZ, self.yaw, 0.0, 0.0) * Vec3::X;
            let up = Vec3::Y;
            let pan = (-delta.x * self.pan_sensitivity * self.radius) * right
                + (delta.y * self.pan_sensitivity * self.radius) * up;
            self.focus += pan;
        }

        if input.scroll_delta() != 0.0 {
            let factor = 1.0 - input.scroll_delta() * 0.1 * self.zoom_sensitivity;
            self.radius = (self.radius * factor).clamp(0.2, 500.0);
        }

        Some(CameraIntent {
            pose: self.pose_from_state(),
            mode: IntentMode::Override,
        })
    }

    fn on_reset(&mut self, _pose: &CameraPose) {
        self.focus = Vec3::ZERO;
        self.yaw = 0.0;
        self.pitch = 0.0;
        self.radius = 10.0;
    }
}
