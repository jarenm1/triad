//! Temporal management for 4D Gaussians

use crate::scene::Gaussian4D;

/// A time range for temporal queries
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeRange {
    pub start: f64,
    pub end: f64,
}

impl TimeRange {
    pub fn new(start: f64, end: f64) -> Self {
        Self { start, end }
    }

    pub fn contains(&self, time: f64) -> bool {
        time >= self.start && time <= self.end
    }

    pub fn duration(&self) -> f64 {
        self.end - self.start
    }
}

/// A temporal keyframe for efficient time-based queries
#[derive(Debug, Clone)]
pub struct TemporalKeyframe {
    pub time: f64,
    pub gaussians: Vec<Gaussian4D>,
}

impl TemporalKeyframe {
    pub fn new(time: f64, gaussians: Vec<Gaussian4D>) -> Self {
        Self { time, gaussians }
    }

    pub fn len(&self) -> usize {
        self.gaussians.len()
    }

    pub fn is_empty(&self) -> bool {
        self.gaussians.is_empty()
    }
}
