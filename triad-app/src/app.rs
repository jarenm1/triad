//! Application state and main run loop.

use crate::layers::LayerMode;
use crate::multi_delegate::{MultiDelegate, MultiInitData};
use glam::Vec3;
use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use triad_gpu::{SceneBounds, ply_loader};
use triad_window::{
    CameraControl, CameraIntent, CameraPose, FrameUpdate, InputState, IntentMode, KeyCode,
    MouseButton, PhysicalKey, egui,
};
