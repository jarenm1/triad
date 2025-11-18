use glam::{EulerRot, Mat4, Quat, Vec2, Vec3};
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

#[derive(Debug)]
pub struct Camera {
    position: Vec3,
    yaw: f32,
    pitch: f32,
    roll: f32,
}

impl Camera {
    pub fn new(position: Vec3, target: Vec3) -> Self {
        let forward = (target - position).normalize_or_zero();
        let yaw = forward.x.atan2(-forward.z);
        let pitch = forward.y.asin().clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
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
        let translation = (-delta.x * sensitivity) * self.right() + (delta.y * sensitivity) * self.up();
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

#[derive(Debug)]
pub struct CameraController {
    rotate_button: MouseButton,
    pan_button: MouseButton,
    roll_button: MouseButton,
    drag_state: Option<DragState>,
    rotation_sensitivity: f32,
    pan_sensitivity: f32,
    roll_sensitivity: f32,
    scroll_sensitivity: f32,
    translate_sensitivity: f32,
    ctrl_pressed: bool,
}

#[derive(Debug, Clone, Copy)]
enum DragMode {
    Rotate,
    Pan,
    Roll,
    Translate,
}

#[derive(Debug, Clone, Copy)]
struct DragState {
    mode: DragMode,
    last: Vec2,
}

impl DragState {
    fn new(mode: DragMode, pos: Vec2) -> Self {
        Self { mode, last: pos }
    }

    fn delta(&mut self, pos: Vec2) -> Vec2 {
        let delta = pos - self.last;
        self.last = pos;
        delta
    }
}

impl CameraController {
    pub fn new() -> Self {
        Self {
            rotate_button: MouseButton::Left,
            pan_button: MouseButton::Middle,
            roll_button: MouseButton::Right,
            drag_state: None,
            rotation_sensitivity: 0.005,
            pan_sensitivity: 0.0025,
            roll_sensitivity: 0.004,
            scroll_sensitivity: 0.2,
            translate_sensitivity: 0.2,
            ctrl_pressed: false,
        }
    }

    pub fn process_event(&mut self, event: &WindowEvent, camera: &mut Camera) -> bool {
        match event {
            WindowEvent::MouseInput { state, button, .. } => {
                if *state == ElementState::Pressed {
                    let mode = if *button == self.rotate_button {
                        Some(DragMode::Rotate)
                    } else if *button == self.pan_button {
                        Some(DragMode::Pan)
                    } else if *button == self.roll_button && !self.ctrl_pressed {
                        Some(DragMode::Roll)
                    } else if *button == self.roll_button && self.ctrl_pressed {
                        Some(DragMode::Translate)
                    } else {
                        None
                    };

                    if let Some(mode) = mode {
                        self.drag_state = Some(DragState::new(mode, Vec2::ZERO));
                        return true;
                    }
                } else if self
                    .drag_state
                    .as_ref()
                    .map(|state| match (state.mode, button) {
                        (DragMode::Rotate, btn) if *btn == self.rotate_button => true,
                        (DragMode::Pan, btn) if *btn == self.pan_button => true,
                        (DragMode::Roll, btn) if *btn == self.roll_button => true,
                        (DragMode::Translate, btn) if *btn == self.roll_button => true,
                        _ => false,
                    })
                    .unwrap_or(false)
                {
                    self.drag_state = None;
                    return true;
                }
                false
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(state) = self.drag_state.as_mut() {
                    let current = Vec2::new(position.x as f32, position.y as f32);
                    if state.last == Vec2::ZERO {
                        state.last = current;
                        return true;
                    }
                    let delta = state.delta(current);
                    match state.mode {
                        DragMode::Rotate => camera.orbit(delta, self.rotation_sensitivity),
                        DragMode::Pan => camera.pan(delta, self.pan_sensitivity),
                        DragMode::Roll => {
                            camera.roll(delta.x, self.roll_sensitivity);
                            camera.dolly(-delta.y * self.scroll_sensitivity * 0.1);
                        }
                        DragMode::Translate => {
                            let forward = camera.forward();
                            let right = camera.right();
                            let translation = (-delta.y * self.translate_sensitivity) * forward
                                + (-delta.x * self.translate_sensitivity) * right;
                            camera.position += translation;
                        }
                    }
                    return true;
                }
                false
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                };
                camera.dolly(amount * self.scroll_sensitivity);
                true
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state,
                        ..
                    },
                ..
            } => {
                let pressed = *state == ElementState::Pressed;
                match key {
                    KeyCode::ControlLeft | KeyCode::ControlRight => {
                        self.ctrl_pressed = pressed;
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }
}

