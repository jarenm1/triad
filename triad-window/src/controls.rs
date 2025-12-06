use std::collections::HashSet;

use glam::Vec2;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::PhysicalKey;

use crate::camera::{Camera, CameraPose};

/// How an intent should be applied relative to the current pose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntentMode {
    Override,
    Additive,
}

/// A camera intent produced by a controller.
#[derive(Debug, Clone, Copy)]
pub struct CameraIntent {
    pub pose: CameraPose,
    pub mode: IntentMode,
}

/// Per-frame snapshot of input state that controllers and hooks can inspect.
#[derive(Debug, Default)]
pub struct InputState {
    mouse_position: Option<Vec2>,
    mouse_delta: Vec2,
    scroll_delta: f32,
    keys_down: HashSet<PhysicalKey>,
    keys_pressed: HashSet<PhysicalKey>,
    keys_released: HashSet<PhysicalKey>,
    mouse_down: HashSet<MouseButton>,
}

impl InputState {
    pub fn mouse_position(&self) -> Option<Vec2> {
        self.mouse_position
    }

    pub fn mouse_delta(&self) -> Vec2 {
        self.mouse_delta
    }

    pub fn scroll_delta(&self) -> f32 {
        self.scroll_delta
    }

    pub fn key_down(&self, key: PhysicalKey) -> bool {
        self.keys_down.contains(&key)
    }

    pub fn just_pressed(&self, key: PhysicalKey) -> bool {
        self.keys_pressed.contains(&key)
    }

    pub fn just_released(&self, key: PhysicalKey) -> bool {
        self.keys_released.contains(&key)
    }

    pub fn mouse_button_down(&self, button: MouseButton) -> bool {
        self.mouse_down.contains(&button)
    }

    pub fn is_ctrl_pressed(&self) -> bool {
        self.key_down(PhysicalKey::Code(winit::keyboard::KeyCode::ControlLeft))
            || self.key_down(PhysicalKey::Code(winit::keyboard::KeyCode::ControlRight))
    }

    fn end_frame(&mut self) {
        self.mouse_delta = Vec2::ZERO;
        self.scroll_delta = 0.0;
        self.keys_pressed.clear();
        self.keys_released.clear();
    }

    fn record_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                let current = Vec2::new(position.x as f32, position.y as f32);
                if let Some(prev) = self.mouse_position {
                    self.mouse_delta += current - prev;
                }
                self.mouse_position = Some(current);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                };
                self.scroll_delta += amount;
            }
            WindowEvent::MouseInput { state, button, .. } => {
                match state {
                    ElementState::Pressed => {
                        self.mouse_down.insert(*button);
                    }
                    ElementState::Released => {
                        self.mouse_down.remove(button);
                    }
                };
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let key = event.physical_key;
                match event.state {
                    ElementState::Pressed => {
                        if self.keys_down.insert(key.clone()) {
                            self.keys_pressed.insert(key.clone());
                        }
                        self.keys_released.remove(&key);
                    }
                    ElementState::Released => {
                        self.keys_down.remove(&key);
                        self.keys_released.insert(key.clone());
                        self.keys_pressed.remove(&key);
                    }
                }
            }
            _ => {}
        }
    }
}

/// Trait for camera controllers. Implementations can react to events and emit intents.
pub trait CameraControl: Send {
    /// Handle an individual window event. Return true if the event was consumed.
    fn handle_event(&mut self, event: &WindowEvent, input: &mut InputState) -> bool {
        let _ = (event, input);
        false
    }

    /// Per-frame update hook. Receives the current pose and should return an intent to apply.
    fn update(
        &mut self,
        _dt: f32,
        _input: &InputState,
        _current: &CameraPose,
    ) -> Option<CameraIntent> {
        None
    }

    /// Called when the camera is reset so controllers can re-seed their state.
    fn on_reset(&mut self, _pose: &CameraPose) {}
}

struct ControllerEntry {
    priority: i32,
    controller: Box<dyn CameraControl>,
}

pub struct FrameUpdate<'a> {
    pub dt: f32,
    pub input: &'a InputState,
    pub pose: &'a mut CameraPose,
    pub reset_request: &'a mut Option<CameraPose>,
}

/// Collection of controls plus frame hooks. Acts as the user-facing builder.
pub struct Controls {
    input: InputState,
    controllers: Vec<ControllerEntry>,
    frame_hooks: Vec<Box<dyn FnMut(FrameUpdate<'_>) + Send>>,
    reset: Option<CameraPose>,
    single_active: bool,
}

impl Controls {
    pub fn new() -> Self {
        let mut controls = Self::empty();
        controls.add_mouse_controller(MouseController::default(), 0);
        controls
    }

    pub fn empty() -> Self {
        Self {
            input: InputState::default(),
            controllers: Vec::new(),
            frame_hooks: Vec::new(),
            reset: None,
            single_active: false,
        }
    }

    /// Remove all controllers (useful when opting out of defaults).
    pub fn clear_controllers(&mut self) -> &mut Self {
        self.controllers.clear();
        self
    }

    /// Whether only the highest-priority controller should run.
    pub fn single_active(&mut self, value: bool) -> &mut Self {
        self.single_active = value;
        self
    }

    pub fn add_mouse_controller(
        &mut self,
        controller: MouseController,
        priority: i32,
    ) -> &mut Self {
        self.add_controller_with_priority(Box::new(controller), priority)
    }

    pub fn add_controller_with_priority(
        &mut self,
        controller: Box<dyn CameraControl>,
        priority: i32,
    ) -> &mut Self {
        self.controllers.push(ControllerEntry {
            priority,
            controller,
        });
        self.controllers.sort_by(|a, b| a.priority.cmp(&b.priority));
        self
    }

    pub fn add_controller(&mut self, controller: Box<dyn CameraControl>) -> &mut Self {
        self.add_controller_with_priority(controller, 0)
    }

    pub fn on_frame<F>(&mut self, hook: F) -> &mut Self
    where
        F: FnMut(FrameUpdate<'_>) + Send + 'static,
    {
        self.frame_hooks.push(Box::new(hook));
        self
    }

    /// Request a camera reset that will be applied before the next update.
    pub fn request_reset(&mut self, pose: CameraPose) {
        self.reset = Some(pose);
    }

    pub fn handle_event(&mut self, event: &WindowEvent) -> bool {
        self.input.record_event(event);
        // Higher priority processed last? For events, process all.
        self.controllers
            .iter_mut()
            .any(|entry| entry.controller.handle_event(event, &mut self.input))
    }

    pub fn update(&mut self, dt: f32, camera: &mut Camera) {
        // Apply pending reset and re-seed controllers.
        if let Some(reset_pose) = self.reset.take() {
            camera.apply_pose(&reset_pose);
            for entry in self.controllers.iter_mut() {
                entry.controller.on_reset(&reset_pose);
            }
        }

        let mut working_pose = camera.pose();

        let controller_iter: Box<dyn Iterator<Item = &mut ControllerEntry>> = if self.single_active
        {
            if let Some(last) = self.controllers.iter_mut().max_by_key(|e| e.priority) {
                Box::new(std::iter::once(last))
            } else {
                Box::new([].into_iter())
            }
        } else {
            Box::new(self.controllers.iter_mut())
        };

        for entry in controller_iter {
            if let Some(intent) = entry.controller.update(dt, &self.input, &working_pose) {
                apply_intent(&mut working_pose, intent);
            }
        }

        let mut requested_reset = None;
        for hook in self.frame_hooks.iter_mut() {
            hook(FrameUpdate {
                dt,
                input: &self.input,
                pose: &mut working_pose,
                reset_request: &mut requested_reset,
            });
        }

        if let Some(reset_pose) = requested_reset {
            // Apply immediately and notify controllers so state is latched.
            camera.apply_pose(&reset_pose);
            for entry in self.controllers.iter_mut() {
                entry.controller.on_reset(&reset_pose);
            }
            working_pose = reset_pose;
        }

        camera.apply_pose(&working_pose);
        self.input.end_frame();
    }

    pub fn input(&self) -> &InputState {
        &self.input
    }
}

impl Default for Controls {
    fn default() -> Self {
        Self::new()
    }
}

fn apply_intent(pose: &mut CameraPose, intent: CameraIntent) {
    match intent.mode {
        IntentMode::Override => {
            *pose = intent.pose;
        }
        IntentMode::Additive => {
            pose.position += intent.pose.position;
            pose.yaw += intent.pose.yaw;
            pose.pitch = (pose.pitch + intent.pose.pitch)
                .clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
            pose.roll += intent.pose.roll;
        }
    }
}

#[derive(Debug)]
pub struct MouseController {
    rotate_button: MouseButton,
    pan_button: MouseButton,
    roll_button: MouseButton,
    drag_state: Option<DragState>,
    rotation_sensitivity: f32,
    pan_sensitivity: f32,
    roll_sensitivity: f32,
    scroll_sensitivity: f32,
    translate_sensitivity: f32,
}

impl MouseController {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn rotation_sensitivity(&mut self, value: f32) -> &mut Self {
        self.rotation_sensitivity = value;
        self
    }

    pub fn pan_sensitivity(&mut self, value: f32) -> &mut Self {
        self.pan_sensitivity = value;
        self
    }

    pub fn roll_sensitivity(&mut self, value: f32) -> &mut Self {
        self.roll_sensitivity = value;
        self
    }

    pub fn scroll_sensitivity(&mut self, value: f32) -> &mut Self {
        self.scroll_sensitivity = value;
        self
    }

    pub fn translate_sensitivity(&mut self, value: f32) -> &mut Self {
        self.translate_sensitivity = value;
        self
    }
}

impl Default for MouseController {
    fn default() -> Self {
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
        }
    }
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
    last: Option<Vec2>,
}

impl DragState {
    fn new(mode: DragMode, pos: Option<Vec2>) -> Self {
        Self { mode, last: pos }
    }
}

impl CameraControl for MouseController {
    fn handle_event(&mut self, event: &WindowEvent, input: &mut InputState) -> bool {
        match event {
            WindowEvent::MouseInput { state, button, .. } => {
                if *state == ElementState::Pressed {
                    let mode = if *button == self.rotate_button {
                        Some(DragMode::Rotate)
                    } else if *button == self.pan_button {
                        Some(DragMode::Pan)
                    } else if *button == self.roll_button && !input.is_ctrl_pressed() {
                        Some(DragMode::Roll)
                    } else if *button == self.roll_button && input.is_ctrl_pressed() {
                        Some(DragMode::Translate)
                    } else {
                        None
                    };

                    if let Some(mode) = mode {
                        let start = input.mouse_position();
                        self.drag_state = Some(DragState::new(mode, start));
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
            _ => false,
        }
    }

    fn update(
        &mut self,
        _dt: f32,
        input: &InputState,
        current: &CameraPose,
    ) -> Option<CameraIntent> {
        let mut pose = *current;

        if let Some(state) = self.drag_state.as_mut() {
            if let Some(current_pos) = input.mouse_position() {
                if let Some(last) = state.last {
                    let delta = current_pos - last;
                    state.last = Some(current_pos);
                    match state.mode {
                        DragMode::Rotate => pose.orbit(delta, self.rotation_sensitivity),
                        DragMode::Pan => pose.pan(delta, self.pan_sensitivity),
                        DragMode::Roll => {
                            pose.roll(delta.x, self.roll_sensitivity);
                            pose.dolly(-delta.y * self.scroll_sensitivity * 0.1);
                        }
                        DragMode::Translate => {
                            let forward = pose.forward();
                            let right = pose.right();
                            let translation = (-delta.y * self.translate_sensitivity) * forward
                                + (-delta.x * self.translate_sensitivity) * right;
                            pose.position += translation;
                        }
                    }
                } else {
                    state.last = Some(current_pos);
                }
            }
        }

        let scroll = input.scroll_delta();
        if scroll != 0.0 {
            pose.dolly(scroll * self.scroll_sensitivity);
        }

        if pose == *current {
            None
        } else {
            Some(CameraIntent {
                pose,
                mode: IntentMode::Override,
            })
        }
    }

    fn on_reset(&mut self, _pose: &CameraPose) {
        self.drag_state = None;
    }
}
