//! Camera stream interfaces for real-time data ingestion

use image::RgbImage;
use glam::Vec3;

/// A single frame from a camera stream
#[derive(Debug, Clone)]
pub struct CameraFrame {
    /// RGB image data
    pub image: RgbImage,
    /// Camera pose (position)
    pub position: Vec3,
    /// Camera orientation (quaternion: x, y, z, w)
    pub rotation: [f32; 4],
    /// Timestamp in seconds (relative to stream start)
    pub timestamp: f64,
    /// Optional depth map (same resolution as image)
    pub depth: Option<Vec<f32>>,
}

impl CameraFrame {
    /// Create a new camera frame
    pub fn new(
        image: RgbImage,
        position: Vec3,
        rotation: [f32; 4],
        timestamp: f64,
    ) -> Self {
        Self {
            image,
            position,
            rotation,
            timestamp,
            depth: None,
        }
    }

    /// Create a new RGBD camera frame with depth
    pub fn with_depth(
        image: RgbImage,
        position: Vec3,
        rotation: [f32; 4],
        timestamp: f64,
        depth: Vec<f32>,
    ) -> Self {
        Self {
            image,
            position,
            rotation,
            timestamp,
            depth: Some(depth),
        }
    }

    /// Get image dimensions (width, height)
    pub fn dimensions(&self) -> (u32, u32) {
        self.image.dimensions()
    }
}

/// Trait for camera stream sources
pub trait CameraStream {
    /// Get the next frame from the stream
    /// Returns None when the stream ends
    fn next_frame(&mut self) -> Result<Option<CameraFrame>, StreamError>;

    /// Get the frame rate (frames per second), if known
    fn frame_rate(&self) -> Option<f32>;

    /// Check if the stream is still active
    fn is_active(&self) -> bool;
}

/// Errors that can occur during stream processing
#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Image decoding error: {0}")]
    ImageDecode(#[from] image::ImageError),
    #[error("Stream ended unexpectedly")]
    StreamEnded,
    #[error("Invalid frame data: {0}")]
    InvalidData(String),
}
