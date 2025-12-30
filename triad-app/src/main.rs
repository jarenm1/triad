//! Triad Application
//!
//! Main application for point cloud visualization and 3D reconstruction.
//!
//! Features:
//! - Multiple visualization modes (points, gaussians, triangles)
//! - Layer system for toggling between modes
//! - Video capture integration
//! - Debug UI overlays

mod app;
mod layers;
mod multi_delegate;

use clap::Parser;
use std::path::PathBuf;

/// Triad - Point Cloud Visualization and Reconstruction
#[derive(Parser, Debug)]
#[command(name = "triad")]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to PLY file to load
    #[arg(short, long)]
    file: Option<PathBuf>,

    /// Initial visualization mode (points, gaussians, triangles)
    #[arg(short, long, default_value = "points")]
    mode: String,
}

fn main() {
    // Note: tracing is initialized by triad-window's run_with_delegate_config
    // Do NOT initialize it here to avoid double-init panic

    let args = Args::parse();

    // Print startup info after tracing is initialized (inside run)
    if let Err(e) = app::run(args.file, &args.mode) {
        eprintln!("Application error: {}", e);
        std::process::exit(1);
    }
}

