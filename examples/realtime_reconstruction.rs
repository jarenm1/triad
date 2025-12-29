//! Real-time scene reconstruction example
//!
//! This example demonstrates how to use the triad-train crate for
//! real-time scene reconstruction from point cloud data.
//!
//! Usage:
//!   cargo run --example realtime_reconstruction -- <path_to_ply>

use std::error::Error;
use std::path::PathBuf;
use tracing::info;
use triad_data::load_vertices_from_ply;
use triad_train::{
    ingest::point_cloud::{Point, PointCloud},
    reconstruction::{GaussianInitializer, InitializationStrategy, SceneUpdater, UpdateStrategy},
    scene::SceneGraph,
};

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let ply_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .expect("Please provide a PLY file path");

    info!("Loading point cloud from: {:?}", ply_path);

    // Load vertices from PLY file
    let ply_path_str = ply_path
        .to_str()
        .ok_or_else(|| "PLY path is not valid UTF-8")?;
    let vertices = load_vertices_from_ply(ply_path_str)?;

    info!("Loaded {} vertices", vertices.len());

    // Convert to PointCloud format
    let points: Vec<Point> = vertices
        .iter()
        .map(|v| Point::new(v.position, v.color))
        .collect();

    let point_cloud = PointCloud::with_timestamp(points, 0.0);

    // Create scene graph and updater
    let mut scene = SceneGraph::new();
    let initializer = GaussianInitializer::new(InitializationStrategy::OnePerPoint);
    let updater = SceneUpdater::new(initializer, UpdateStrategy::Append);

    // Update scene with point cloud
    info!("Initializing scene with {} points", point_cloud.len());
    updater.update_from_point_cloud(&mut scene, &point_cloud, 0.0);

    // Query Gaussians at time 0.0
    let gaussians = scene.gaussians_at(0.0);
    info!(
        "Scene contains {} Gaussians at time 0.0",
        gaussians.len()
    );

    // Get time range
    if let Some(time_range) = scene.time_range() {
        info!(
            "Scene time range: {:.2}s to {:.2}s (duration: {:.2}s)",
            time_range.start,
            time_range.end,
            time_range.duration()
        );
    }

    info!("Real-time reconstruction example completed successfully!");
    Ok(())
}
