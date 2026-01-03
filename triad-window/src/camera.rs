use glam::{Mat4, Vec2, Vec3};

/// Camera pose representing position and orientation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraPose {
    /// Center/focus point that the camera orbits around.
    pub center: Vec3,
    /// Camera position in world space.
    pub position: Vec3,
    /// Yaw angle in radians (rotation around Y axis).
    pub yaw: f32,
    /// Pitch angle in radians (rotation around X axis).
    pub pitch: f32,
    /// Roll angle in radians (rotation around Z axis).
    pub roll: f32,
}

impl CameraPose {
    /// Create a new camera pose.
    pub fn new(position: Vec3, center: Vec3) -> Self {
        let forward = (center - position).normalize_or_zero();
        let yaw = forward.x.atan2(-forward.z);
        let pitch = forward
            .y
            .asin()
            .clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
        Self {
            center,
            position,
            yaw,
            pitch,
            roll: 0.0,
        }
    }

    /// Orbit around the center point.
    pub fn orbit_around_center(&mut self, delta: Vec2, sensitivity: f32) {
        self.yaw -= delta.x * sensitivity;
        self.pitch = (self.pitch - delta.y * sensitivity).clamp(
            -std::f32::consts::FRAC_PI_2 + 0.01,
            std::f32::consts::FRAC_PI_2 - 0.01,
        );

        // Update position based on new angles
        let dir = glam::Quat::from_euler(glam::EulerRot::YXZ, self.yaw, self.pitch, 0.0) * -Vec3::Z;
        let distance = (self.position - self.center).length();
        self.position = self.center - dir.normalize() * distance;
    }

    /// Pan the camera and center together.
    pub fn pan(&mut self, delta: Vec2, sensitivity: f32) {
        let right = glam::Quat::from_euler(glam::EulerRot::YXZ, self.yaw, 0.0, 0.0) * Vec3::X;
        let up = Vec3::Y;
        let distance = (self.position - self.center).length();
        let pan = (-delta.x * sensitivity * distance) * right
            + (delta.y * sensitivity * distance) * up;
        self.center += pan;
        self.position += pan;
    }

    /// Zoom in/out by changing distance to center.
    pub fn zoom(&mut self, amount: f32) {
        let direction = (self.position - self.center).normalize_or_zero();
        let distance = (self.position - self.center).length();
        let new_distance = (distance + amount).max(0.1).min(1000.0);
        self.position = self.center + direction * new_distance;
    }
}

/// Camera that manages position and view matrix.
pub struct Camera {
    pose: CameraPose,
}

impl Camera {
    /// Creates a camera at the given position looking at the center.
    pub fn new(position: Vec3, center: Vec3) -> Self {
        Self {
            pose: CameraPose::new(position, center),
        }
    }

    /// Get the current pose.
    pub fn pose(&self) -> CameraPose {
        self.pose
    }

    /// Apply a pose to the camera.
    pub fn apply_pose(&mut self, pose: &CameraPose) {
        self.pose = *pose;
    }

    /// Get the camera position.
    pub fn position(&self) -> Vec3 {
        self.pose.position
    }

    /// Get the view matrix.
    pub fn view_matrix(&self) -> Mat4 {
        let forward = (self.pose.center - self.pose.position).normalize_or_zero();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        Mat4::look_to_rh(self.pose.position, forward, up)
    }
}

/// Projection matrix configuration.
pub struct Projection {
    width: u32,
    height: u32,
    fov: f32,
    near: f32,
    far: f32,
}

impl Projection {
    /// Create a new projection.
    pub fn new(width: u32, height: u32, fov: f32, near: f32, far: f32) -> Self {
        Self {
            width,
            height,
            fov,
            near,
            far,
        }
    }

    /// Get the projection matrix.
    pub fn matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov, self.width as f32 / self.height as f32, self.near, self.far)
    }

    /// Update the projection size.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    /// Update the projection size (alias for resize).
    pub fn update_size(&mut self, width: u32, height: u32) {
        self.resize(width, height);
    }

    /// Set the far plane distance.
    pub fn set_far(&mut self, far: f32) {
        self.far = far;
    }

    /// Get the far plane distance.
    pub fn far(&self) -> f32 {
        self.far
    }
}

/// Trait for camera controllers (legacy - use CameraControl in controls.rs instead).
pub trait CameraController {
    fn update() {}
    fn setup() {}
}
