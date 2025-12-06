use glam::{EulerRot, Mat4, Quat, Vec2, Vec3};

#[derive(Debug)]
pub struct Camera {
    position: Vec3,
    yaw: f32,
    pitch: f32,
    roll: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraPose {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
}

impl Camera {
    pub fn new(position: Vec3, target: Vec3) -> Self {
        let forward = (target - position).normalize_or_zero();
        let yaw = forward.x.atan2(-forward.z);
        let pitch = forward
            .y
            .asin()
            .clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
        Self {
            position,
            yaw,
            pitch,
            roll: 0.0,
        }
    }

    pub fn position(&self) -> Vec3 {
        self.position
    }

    pub fn pose(&self) -> CameraPose {
        CameraPose {
            position: self.position,
            yaw: self.yaw,
            pitch: self.pitch,
            roll: self.roll,
        }
    }

    pub fn apply_pose(&mut self, pose: &CameraPose) {
        self.position = pose.position;
        self.yaw = pose.yaw;
        self.pitch = pose.pitch;
        self.roll = pose.roll;
    }

    pub fn set_position(&mut self, position: Vec3) {
        self.position = position;
    }

    pub fn look_at(&mut self, target: Vec3) {
        let forward = (target - self.position).normalize_or_zero();
        self.yaw = forward.x.atan2(-forward.z);
        self.pitch = forward
            .y
            .asin()
            .clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
        // Reset roll to keep horizon level when tracking a target.
        self.roll = 0.0;
    }

    pub fn set_orientation(&mut self, yaw: f32, pitch: f32, roll: f32) {
        self.yaw = yaw;
        self.pitch = pitch.clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
        self.roll = roll;
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_to_rh(self.position, self.forward(), self.up())
    }

    pub fn forward(&self) -> Vec3 {
        (self.orientation() * -Vec3::Z).normalize()
    }

    pub fn right(&self) -> Vec3 {
        (self.orientation() * Vec3::X).normalize()
    }

    pub fn up(&self) -> Vec3 {
        (self.orientation() * Vec3::Y).normalize()
    }

    fn orientation(&self) -> Quat {
        Quat::from_euler(EulerRot::YXZ, self.yaw, self.pitch, self.roll)
    }

    pub fn orbit(&mut self, delta: Vec2, sensitivity: f32) {
        self.yaw -= delta.x * sensitivity;
        self.pitch = (self.pitch - delta.y * sensitivity).clamp(
            -std::f32::consts::FRAC_PI_2 + 0.01,
            std::f32::consts::FRAC_PI_2 - 0.01,
        );
    }

    pub fn roll(&mut self, delta: f32, sensitivity: f32) {
        self.roll = (self.roll - delta * sensitivity) % (std::f32::consts::TAU);
    }

    pub fn pan(&mut self, delta: Vec2, sensitivity: f32) {
        let translation =
            (-delta.x * sensitivity) * self.right() + (delta.y * sensitivity) * self.up();
        self.position += translation;
    }

    pub fn dolly(&mut self, amount: f32) {
        self.position += self.forward() * amount;
    }
}

impl CameraPose {
    pub fn forward(&self) -> Vec3 {
        (self.orientation() * -Vec3::Z).normalize()
    }

    pub fn right(&self) -> Vec3 {
        (self.orientation() * Vec3::X).normalize()
    }

    pub fn up(&self) -> Vec3 {
        (self.orientation() * Vec3::Y).normalize()
    }

    fn orientation(&self) -> Quat {
        Quat::from_euler(EulerRot::YXZ, self.yaw, self.pitch, self.roll)
    }

    pub fn orbit(&mut self, delta: Vec2, sensitivity: f32) {
        self.yaw -= delta.x * sensitivity;
        self.pitch = (self.pitch - delta.y * sensitivity).clamp(
            -std::f32::consts::FRAC_PI_2 + 0.01,
            std::f32::consts::FRAC_PI_2 - 0.01,
        );
    }

    pub fn roll(&mut self, delta: f32, sensitivity: f32) {
        self.roll = (self.roll - delta * sensitivity) % (std::f32::consts::TAU);
    }

    pub fn pan(&mut self, delta: Vec2, sensitivity: f32) {
        let translation =
            (-delta.x * sensitivity) * self.right() + (delta.y * sensitivity) * self.up();
        self.position += translation;
    }

    pub fn dolly(&mut self, amount: f32) {
        self.position += self.forward() * amount;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Projection {
    pub fovy: f32,
    pub aspect: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Projection {
    pub fn new(width: u32, height: u32, fovy: f32, znear: f32, zfar: f32) -> Self {
        Self {
            fovy,
            aspect: width as f32 / height.max(1) as f32,
            znear,
            zfar,
        }
    }

    pub fn update_size(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height.max(1) as f32;
    }

    pub fn matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fovy, self.aspect, self.znear, self.zfar)
    }
}
