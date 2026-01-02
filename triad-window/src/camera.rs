use glam::Vec3;

pub struct Camera {
    position: Vec3,
    rotation: Vec3,
}

impl Camera {
    /// Creates a camera at (0, 0, 0)
    ///
    pub fn new() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 0.0),
            rotation: Vec3::new(0.0, 0.0, 0.0),
        }
    }
}

pub trait CameraController {
    fn update() {}
    fn setup() {}
}
