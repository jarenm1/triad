//! 4D Gaussian representation for time-varying scenes

use glam::{Quat, Vec3};
use triad_gpu::GaussianPoint;

/// A 4D Gaussian that can vary over time
///
/// This extends the 3D GaussianPoint with temporal information.
/// For real-time rendering, we support both keyframe-based and
/// continuous temporal interpolation.
#[derive(Debug, Clone)]
pub struct Gaussian4D {
    /// Base Gaussian properties (position, color, opacity, rotation, scale)
    pub base: GaussianPoint,
    /// Time at which this Gaussian is defined (in seconds)
    pub time: f64,
    /// Optional velocity for linear temporal interpolation
    pub velocity: Option<Vec3>,
    /// Optional time-varying rotation delta
    pub rotation_delta: Option<Quat>,
    /// Optional time-varying scale delta
    pub scale_delta: Option<Vec3>,
}

impl Gaussian4D {
    /// Create a new 4D Gaussian at a specific time
    pub fn new(base: GaussianPoint, time: f64) -> Self {
        Self {
            base,
            time,
            velocity: None,
            rotation_delta: None,
            scale_delta: None,
        }
    }

    /// Create a 4D Gaussian with velocity for linear interpolation
    pub fn with_velocity(base: GaussianPoint, time: f64, velocity: Vec3) -> Self {
        Self {
            base,
            time,
            velocity: Some(velocity),
            rotation_delta: None,
            scale_delta: None,
        }
    }

    /// Evaluate the Gaussian at a specific time
    /// Returns a 3D GaussianPoint suitable for rendering
    pub fn evaluate_at(&self, target_time: f64) -> GaussianPoint {
        let dt = (target_time - self.time) as f32;

        // Interpolate position if velocity is available
        let position = if let Some(vel) = self.velocity {
            let base_pos = Vec3::from_slice(&self.base.position);
            let new_pos = base_pos + vel * dt;
            [new_pos.x, new_pos.y, new_pos.z]
        } else {
            self.base.position
        };

        // Interpolate rotation if delta is available
        let rotation = if let Some(rot_delta) = self.rotation_delta {
            // Convert [x, y, z, w] array to Quat
            let base_rot = Quat::from_xyzw(
                self.base.rotation[0],
                self.base.rotation[1],
                self.base.rotation[2],
                self.base.rotation[3],
            );
            let new_rot = base_rot.slerp(base_rot * rot_delta, dt);
            [new_rot.x, new_rot.y, new_rot.z, new_rot.w]
        } else {
            self.base.rotation
        };

        // Interpolate scale if delta is available
        let scale = if let Some(scale_delta) = self.scale_delta {
            let base_scale = Vec3::from_slice(&self.base.scale);
            let new_scale = base_scale + scale_delta * dt;
            [new_scale.x, new_scale.y, new_scale.z, 0.0]
        } else {
            self.base.scale
        };

        GaussianPoint {
            position,
            _pad0: self.base._pad0,
            color_opacity: self.base.color_opacity,
            rotation,
            scale,
        }
    }

    /// Convert to a standard 3D GaussianPoint (at base time)
    pub fn to_3d(&self) -> GaussianPoint {
        self.base
    }
}

impl From<GaussianPoint> for Gaussian4D {
    fn from(base: GaussianPoint) -> Self {
        Self::new(base, 0.0)
    }
}
