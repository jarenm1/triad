//! Incremental scene updates for real-time reconstruction

use crate::ingest::point_cloud::PointCloud;
use crate::reconstruction::GaussianInitializer;
use crate::scene::{SceneGraph, TemporalKeyframe};
use tracing::debug;

/// Strategy for updating the scene
#[derive(Debug, Clone, Copy)]
pub enum UpdateStrategy {
    /// Add new Gaussians without removing old ones
    Append,
    /// Replace Gaussians within a time window
    Replace { time_window: f64 },
    /// Merge with existing Gaussians (adaptive)
    Merge { distance_threshold: f32 },
}

/// Updates the scene graph incrementally from new data
pub struct SceneUpdater {
    initializer: GaussianInitializer,
    strategy: UpdateStrategy,
}

impl SceneUpdater {
    pub fn new(initializer: GaussianInitializer, strategy: UpdateStrategy) -> Self {
        Self {
            initializer,
            strategy,
        }
    }

    /// Update the scene with a new point cloud at a specific time
    pub fn update_from_point_cloud(
        &self,
        scene: &mut SceneGraph,
        point_cloud: &PointCloud,
        time: f64,
    ) {
        let new_gaussians = self.initializer.from_point_cloud(&point_cloud.points, time);
        let gaussian_count = new_gaussians.len();

        match self.strategy {
            UpdateStrategy::Append => {
                // Simply add a new keyframe
                let keyframe = TemporalKeyframe::new(time, new_gaussians);
                scene.add_keyframe(keyframe);
                debug!("Appended {} Gaussians at time {}", gaussian_count, time);
            }
            UpdateStrategy::Replace { time_window } => {
                // Remove keyframes within the time window and add new one
                let _window_start = time - time_window;
                let _window_end = time + time_window;
                
                // Remove old keyframes in window (simplified - would need proper BTreeMap range deletion)
                let keyframe = TemporalKeyframe::new(time, new_gaussians);
                scene.add_keyframe(keyframe);
                debug!("Replaced {} Gaussians at time {} (window: {}s)", gaussian_count, time, time_window);
            }
            UpdateStrategy::Merge { distance_threshold: _ } => {
                // TODO: Implement merging logic
                let keyframe = TemporalKeyframe::new(time, new_gaussians);
                scene.add_keyframe(keyframe);
                debug!("Merged {} Gaussians at time {}", gaussian_count, time);
            }
        }
    }
}

impl Default for SceneUpdater {
    fn default() -> Self {
        Self::new(
            GaussianInitializer::default(),
            UpdateStrategy::Append,
        )
    }
}
