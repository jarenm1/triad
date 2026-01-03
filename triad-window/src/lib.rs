mod app;
mod camera;
pub mod controls;

// Re-export render delegate types from triad-gpu
pub use triad_gpu::{
    GaussianDelegate, GaussianInitData, PointDelegate, PointInitData, RenderContext,
    RenderDelegate, SceneBounds, TriangleDelegate, TriangleInitData,
};

pub use app::{egui, run_with_delegate, run_with_delegate_config};
pub use camera::{Camera, CameraController, CameraPose, Projection};
pub use controls::{
    CameraControl, CameraIntent, Controls, FrameUpdate, InputState, IntentMode, MouseController,
};
pub use winit::event::MouseButton;
pub use winit::keyboard::{KeyCode, PhysicalKey};
