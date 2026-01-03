//! Triad Sim Application

mod app;
mod camera;
mod layers;
mod menu;
mod multi_delegate;

use app::{AppBuilder, ModeSignal, MultiDelegateConfig};
use clap::Parser;
use menu::{MenuAction, create_full_menu};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU8;
use std::sync::mpsc;

/// Command-line arguments for the Triad application.
#[derive(Parser, Debug)]
#[command(name = "triad")]
#[command(about = "Triad - A visualization and reconstruction tool")]
pub struct Args {
    /// Path to PLY file to load (optional - can be loaded at runtime)
    #[arg(short, long)]
    pub ply: Option<PathBuf>,

    /// Initial visualization mode
    #[arg(short, long, default_value = "points")]
    pub mode: String,

    /// Window width
    #[arg(long, default_value_t = 1280)]
    pub width: u32,

    /// Window height
    #[arg(long, default_value_t = 720)]
    pub height: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Parse mode
    let initial_mode = match args.mode.to_lowercase().as_str() {
        "points" => layers::LayerMode::Points,
        "gaussians" => layers::LayerMode::Gaussians,
        "triangles" => layers::LayerMode::Triangles,
        _ => {
            eprintln!("Unknown mode: {}. Using 'points'", args.mode);
            layers::LayerMode::Points
        }
    };

    // Create mode signal for menu (shared with delegate)
    let mode_signal: ModeSignal = Arc::new(AtomicU8::new(match initial_mode {
        layers::LayerMode::Points => 0,
        layers::LayerMode::Gaussians => 1,
        layers::LayerMode::Triangles => 2,
    }));

    // Create channel for menu actions
    let (action_sender, action_receiver) = mpsc::channel();

    // Create channel for PLY loading (separate from menu actions for direct delegate communication)
    let (ply_sender, ply_receiver) = mpsc::channel();

    // Spawn a thread to handle menu actions
    let ply_sender_clone = ply_sender.clone();
    std::thread::spawn(move || {
        while let Ok(action) = action_receiver.recv() {
            match action {
                MenuAction::ImportPly(path) => {
                    tracing::info!("Menu action: Import PLY from {:?}", path);
                    // Send PLY path to delegate for runtime loading
                    if let Err(e) = ply_sender_clone.send(path) {
                        tracing::warn!("Failed to send PLY path to delegate: {}", e);
                    }
                }
                MenuAction::ConnectCamera(index) => {
                    tracing::info!("Menu action: Connect to camera device {}", index);
                    // TODO: Implement camera connection
                    // This would start a camera stream and potentially feed it to the reconstruction system
                }
                MenuAction::DisconnectCamera => {
                    tracing::info!("Menu action: Disconnect camera");
                    // TODO: Implement camera disconnection
                }
            }
        }
    });

    // Create app builder
    let mut builder = AppBuilder::new()
        .with_title("Triad Viewer")
        .with_window_size(args.width, args.height)
        .with_multi_delegate(MultiDelegateConfig {
            initial_mode,
            point_size: 0.01,
        })
        .with_mode_signal(mode_signal.clone());

    // Set PLY path if provided
    if let Some(ply_path) = args.ply {
        builder = builder.with_ply_path(ply_path);
    }

    // Configure menu with mode signal and action sender
    let menu_fn = create_full_menu(mode_signal, action_sender);
    builder = builder.with_menu(menu_fn).with_ply_receiver(ply_receiver);

    // Configure controls - enable mouse camera controls
    builder = builder.with_controls(|controls| {
        // Add mouse controller for camera interaction
        // Controls:
        // - Left mouse drag: Orbit camera around the center point
        // - Shift + Left mouse drag: Pan (move center point and camera together)
        // - Mouse wheel: Zoom in/out (change distance to center)
        use triad_window::MouseController;
        controls.add_mouse_controller(MouseController::default(), 0);
    });

    // Run the app
    builder.run()
}
