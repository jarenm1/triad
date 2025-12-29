//! Scene graph for organizing 4D Gaussians

use crate::scene::{Gaussian4D, TemporalKeyframe, TimeRange};
use ordered_float::OrderedFloat;
use std::collections::BTreeMap;

/// A scene graph that organizes 4D Gaussians temporally
///
/// This structure supports efficient queries for Gaussians at specific times
/// and incremental updates for real-time reconstruction.
#[derive(Debug, Default)]
pub struct SceneGraph {
    /// Keyframes indexed by time
    keyframes: BTreeMap<OrderedFloat<f64>, TemporalKeyframe>,
    /// Current time range of the scene
    time_range: Option<TimeRange>,
}

impl SceneGraph {
    /// Create a new empty scene graph
    pub fn new() -> Self {
        Self {
            keyframes: BTreeMap::new(),
            time_range: None,
        }
    }

    /// Add a keyframe to the scene
    pub fn add_keyframe(&mut self, keyframe: TemporalKeyframe) {
        let time = keyframe.time;
        self.keyframes.insert(OrderedFloat(time), keyframe);

        // Update time range
        self.time_range = match self.time_range {
            Some(range) => Some(TimeRange::new(
                range.start.min(time),
                range.end.max(time),
            )),
            None => Some(TimeRange::new(time, time)),
        };
    }

    /// Get Gaussians at a specific time
    ///
    /// Returns Gaussians from the nearest keyframe(s) and interpolates if needed
    pub fn gaussians_at(&self, time: f64) -> Vec<Gaussian4D> {
        // For now, return Gaussians from the nearest keyframe
        // Future: implement interpolation between keyframes
        let time_key = OrderedFloat(time);
        if let Some((&_keyframe_time, keyframe)) = self.keyframes.range(..=time_key).next_back() {
            // Evaluate each Gaussian at the target time
            keyframe
                .gaussians
                .iter()
                .map(|g| g.evaluate_at(time))
                .map(Gaussian4D::from)
                .collect()
        } else if let Some((&_keyframe_time, keyframe)) = self.keyframes.range(time_key..).next() {
            // Use the next keyframe if no previous one exists
            keyframe
                .gaussians
                .iter()
                .map(|g| g.evaluate_at(time))
                .map(Gaussian4D::from)
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get the time range of the scene
    pub fn time_range(&self) -> Option<TimeRange> {
        self.time_range
    }

    /// Get all keyframes
    pub fn keyframes(&self) -> &BTreeMap<OrderedFloat<f64>, TemporalKeyframe> {
        &self.keyframes
    }

    /// Clear all keyframes
    pub fn clear(&mut self) {
        self.keyframes.clear();
        self.time_range = None;
    }

    /// Get the number of keyframes
    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }
}
