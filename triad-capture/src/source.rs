//! Common capture source types and traits.

use glam::Vec3;
use image::RgbImage;
use thiserror::Error;

/// Errors that can occur during capture.
#[derive(Debug, Error)]
pub enum CaptureError {
    #[error("Device not found: {0}")]
    DeviceNotFound(String),

    #[error("Failed to open device: {0}")]
    OpenFailed(String),

    #[error("Failed to capture frame: {0}")]
    CaptureFailed(String),

    #[error("Stream ended")]
    StreamEnded,

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Raw frame data from a capture source.
#[derive(Debug, Clone)]
pub struct FrameData {
    /// RGB image data.
    pub image: RgbImage,
    /// Frame timestamp in seconds (relative to stream start).
    pub timestamp: f64,
    /// Frame number.
    pub frame_number: u64,
}

impl FrameData {
    /// Create a new frame.
    pub fn new(image: RgbImage, timestamp: f64, frame_number: u64) -> Self {
        Self {
            image,
            timestamp,
            frame_number,
        }
    }

    /// Get image dimensions (width, height).
    pub fn dimensions(&self) -> (u32, u32) {
        self.image.dimensions()
    }
}

/// Trait for capture sources that provide video frames.
///
/// This is a lower-level trait than `CameraStream` - it doesn't include
/// camera pose information. Use this for raw video capture.
pub trait CaptureSource {
    /// Get the next frame from the source.
    fn next_frame(&mut self) -> Result<Option<FrameData>, CaptureError>;

    /// Get the frame rate, if known.
    fn frame_rate(&self) -> Option<f32>;

    /// Get the resolution (width, height).
    fn resolution(&self) -> (u32, u32);

    /// Check if the source is still active.
    fn is_active(&self) -> bool;

    /// Stop capturing.
    fn stop(&mut self);
}

/// A capture source with known camera pose (implements CameraStream).
pub struct PosedCaptureSource<S: CaptureSource> {
    source: S,
    position: Vec3,
    rotation: [f32; 4],
}

impl<S: CaptureSource> PosedCaptureSource<S> {
    /// Create a new posed capture source.
    pub fn new(source: S) -> Self {
        Self {
            source,
            position: Vec3::ZERO,
            rotation: [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Set the camera position.
    pub fn with_position(mut self, position: Vec3) -> Self {
        self.position = position;
        self
    }

    /// Set the camera rotation (quaternion).
    pub fn with_rotation(mut self, rotation: [f32; 4]) -> Self {
        self.rotation = rotation;
        self
    }

    /// Update the camera pose.
    pub fn set_pose(&mut self, position: Vec3, rotation: [f32; 4]) {
        self.position = position;
        self.rotation = rotation;
    }

    /// Get the underlying source.
    pub fn inner(&self) -> &S {
        &self.source
    }

    /// Get the underlying source mutably.
    pub fn inner_mut(&mut self) -> &mut S {
        &mut self.source
    }
}

impl<S: CaptureSource> triad_train::ingest::CameraStream for PosedCaptureSource<S> {
    fn next_frame(&mut self) -> Result<Option<triad_train::ingest::CameraFrame>, triad_train::ingest::StreamError> {
        match self.source.next_frame() {
            Ok(Some(frame)) => Ok(Some(triad_train::ingest::CameraFrame::new(
                frame.image,
                self.position,
                self.rotation,
                frame.timestamp,
            ))),
            Ok(None) => Ok(None),
            Err(e) => Err(triad_train::ingest::StreamError::InvalidData(e.to_string())),
        }
    }

    fn frame_rate(&self) -> Option<f32> {
        self.source.frame_rate()
    }

    fn is_active(&self) -> bool {
        self.source.is_active()
    }
}

