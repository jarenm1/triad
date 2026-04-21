mod app;
mod camera;
mod camera_uniforms;
pub mod controls;

// Re-export types from triad-gpu
// Note: RenderDelegate and SceneBounds have been removed

pub use app::{RendererManager, WindowConfig, egui, run_with_renderer_config};
pub use camera::{Camera, CameraController, CameraPose, Projection};
pub use camera_uniforms::CameraUniforms;
pub use controls::{
    CameraControl, CameraIntent, Controls, FrameUpdate, InputState, IntentMode, MouseController,
};
pub use winit::event::MouseButton;
pub use winit::keyboard::{KeyCode, PhysicalKey};
