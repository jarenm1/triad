//! Scriptable camera system - decoupled from GPU and app logic.

use glam::Vec3;
use std::error::Error;

/// Camera update context provided to scripts each frame.
pub struct CameraUpdateContext {
    /// Time since last frame in seconds.
    pub dt: f32,
    /// Current camera position.
    pub position: Vec3,
    /// Current camera center/focus point.
    pub center: Vec3,
    /// Current yaw angle in radians.
    pub yaw: f32,
    /// Current pitch angle in radians.
    pub pitch: f32,
    /// Current roll angle in radians.
    pub roll: f32,
}

/// Camera intent produced by a script.
pub struct CameraIntent {
    /// Target position.
    pub position: Option<Vec3>,
    /// Target center/focus point.
    pub center: Option<Vec3>,
    /// Target yaw angle in radians.
    pub yaw: Option<f32>,
    /// Target pitch angle in radians.
    pub pitch: Option<f32>,
    /// Target roll angle in radians.
    pub roll: Option<f32>,
}

impl CameraIntent {
    /// Create an empty intent (no changes).
    pub fn new() -> Self {
        Self {
            position: None,
            center: None,
            yaw: None,
            pitch: None,
            roll: None,
        }
    }

    /// Set position.
    pub fn with_position(mut self, position: Vec3) -> Self {
        self.position = Some(position);
        self
    }

    /// Set center.
    pub fn with_center(mut self, center: Vec3) -> Self {
        self.center = Some(center);
        self
    }

    /// Set yaw.
    pub fn with_yaw(mut self, yaw: f32) -> Self {
        self.yaw = Some(yaw);
        self
    }

    /// Set pitch.
    pub fn with_pitch(mut self, pitch: f32) -> Self {
        self.pitch = Some(pitch);
        self
    }

    /// Set roll.
    pub fn with_roll(mut self, roll: f32) -> Self {
        self.roll = Some(roll);
        self
    }
}

impl Default for CameraIntent {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for camera script setup.
pub struct CameraScriptConfig {
    /// Initial camera position.
    pub initial_position: Option<Vec3>,
    /// Initial camera center.
    pub initial_center: Option<Vec3>,
    /// Initial yaw.
    pub initial_yaw: Option<f32>,
    /// Initial pitch.
    pub initial_pitch: Option<f32>,
    /// Initial roll.
    pub initial_roll: Option<f32>,
}

impl CameraScriptConfig {
    /// Create a new config.
    pub fn new() -> Self {
        Self {
            initial_position: None,
            initial_center: None,
            initial_yaw: None,
            initial_pitch: None,
            initial_roll: None,
        }
    }

    /// Set initial position.
    pub fn with_position(mut self, position: Vec3) -> Self {
        self.initial_position = Some(position);
        self
    }

    /// Set initial center.
    pub fn with_center(mut self, center: Vec3) -> Self {
        self.initial_center = Some(center);
        self
    }

    /// Set initial yaw.
    pub fn with_yaw(mut self, yaw: f32) -> Self {
        self.initial_yaw = Some(yaw);
        self
    }

    /// Set initial pitch.
    pub fn with_pitch(mut self, pitch: f32) -> Self {
        self.initial_pitch = Some(pitch);
        self
    }

    /// Set initial roll.
    pub fn with_roll(mut self, roll: f32) -> Self {
        self.initial_roll = Some(roll);
        self
    }
}

impl Default for CameraScriptConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for scriptable camera controllers.
/// 
/// This trait allows camera control logic to be decoupled from GPU and app logic.
/// Implementations define how the camera should behave through `setup()` and `update()`.
pub trait CameraScript: Send + Sync {
    /// Setup the camera script - configure initial state, bindings, etc.
    /// This is called once when the camera script is initialized.
    fn setup(&mut self, config: &mut CameraScriptConfig) -> Result<(), Box<dyn Error>> {
        let _ = config;
        Ok(())
    }

    /// Update the camera each frame.
    /// Returns a CameraIntent describing desired camera changes.
    fn update(&mut self, ctx: &CameraUpdateContext) -> CameraIntent {
        let _ = ctx;
        CameraIntent::new()
    }

    /// Called when the camera is reset.
    fn on_reset(&mut self, _ctx: &CameraUpdateContext) {
        // Default: no-op
    }
}

/// Adapter to convert a CameraScript into a triad-window CameraControl.
pub struct CameraScriptAdapter {
    script: Box<dyn CameraScript>,
    current_pose: Option<triad_window::CameraPose>,
}

impl CameraScriptAdapter {
    /// Create a new adapter from a CameraScript.
    pub fn new(script: Box<dyn CameraScript>) -> Self {
        Self {
            script,
            current_pose: None,
        }
    }
}

impl triad_window::CameraControl for CameraScriptAdapter {
    fn update(
        &mut self,
        dt: f32,
        _input: &triad_window::InputState,
        current: &triad_window::CameraPose,
    ) -> Option<triad_window::CameraIntent> {
        // Convert current pose to update context
        let ctx = CameraUpdateContext {
            dt,
            position: current.position,
            center: current.center,
            yaw: current.yaw,
            pitch: current.pitch,
            roll: current.roll,
        };

        // Get intent from script
        let intent = self.script.update(&ctx);

        // Convert intent to window CameraIntent
        let mut new_pose = *current;
        let mut changed = false;

        if let Some(pos) = intent.position {
            new_pose.position = pos;
            changed = true;
        }
        if let Some(center) = intent.center {
            new_pose.center = center;
            changed = true;
        }
        if let Some(yaw) = intent.yaw {
            new_pose.yaw = yaw;
            changed = true;
        }
        if let Some(pitch) = intent.pitch {
            new_pose.pitch = pitch;
            changed = true;
        }
        if let Some(roll) = intent.roll {
            new_pose.roll = roll;
            changed = true;
        }

        if changed {
            Some(triad_window::CameraIntent {
                pose: new_pose,
                mode: triad_window::IntentMode::Override,
            })
        } else {
            None
        }
    }

    fn on_reset(&mut self, pose: &triad_window::CameraPose) {
        let ctx = CameraUpdateContext {
            dt: 0.0,
            position: pose.position,
            center: pose.center,
            yaw: pose.yaw,
            pitch: pose.pitch,
            roll: pose.roll,
        };
        self.script.on_reset(&ctx);
        self.current_pose = Some(*pose);
    }
}
