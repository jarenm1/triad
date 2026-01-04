//! Webcam capture using nokhwa.

use crate::source::{CaptureError, CaptureSource, FrameData};
use image::RgbImage;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::Camera;
use std::time::Instant;
use tracing::{debug, info};

/// Webcam capture source.
pub struct WebcamCapture {
    camera: Camera,
    start_time: Instant,
    frame_count: u64,
    active: bool,
    resolution: (u32, u32),
}

impl WebcamCapture {
    /// Create a new webcam capture from device index.
    pub fn new(index: u32) -> Result<Self, CaptureError> {
        Self::with_resolution(index, 1280, 720)
    }

    /// Create a new webcam capture with specific resolution.
    pub fn with_resolution(index: u32, width: u32, height: u32) -> Result<Self, CaptureError> {
        info!("Opening webcam {} at {}x{}", index, width, height);

        let camera_index = CameraIndex::Index(index);
        let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestResolution);

        let camera = Camera::new(camera_index, requested)
            .map_err(|e| CaptureError::OpenFailed(e.to_string()))?;

        let resolution = camera.resolution();
        info!(
            "Webcam opened: {}x{} @ {:?} fps",
            resolution.width(),
            resolution.height(),
            camera.frame_rate()
        );

        Ok(Self {
            camera,
            start_time: Instant::now(),
            frame_count: 0,
            active: true,
            resolution: (resolution.width(), resolution.height()),
        })
    }

    /// List available webcam devices.
    pub fn list_devices() -> Result<Vec<String>, CaptureError> {
        let devices = nokhwa::query(nokhwa::utils::ApiBackend::Auto)
            .map_err(|e| CaptureError::DeviceNotFound(e.to_string()))?;

        Ok(devices
            .into_iter()
            .map(|info| format!("{}: {}", info.index(), info.human_name()))
            .collect())
    }
}

impl CaptureSource for WebcamCapture {
    fn next_frame(&mut self) -> Result<Option<FrameData>, CaptureError> {
        if !self.active {
            return Ok(None);
        }

        let frame = self
            .camera
            .frame()
            .map_err(|e| CaptureError::CaptureFailed(e.to_string()))?;

        let decoded = frame
            .decode_image::<RgbFormat>()
            .map_err(|e| CaptureError::CaptureFailed(e.to_string()))?;

        let timestamp = self.start_time.elapsed().as_secs_f64();
        self.frame_count += 1;

        debug!(
            "Captured frame {} at {:.3}s",
            self.frame_count, timestamp
        );

        // Convert to RgbImage
        let (width, height) = (decoded.width(), decoded.height());
        let rgb_image = RgbImage::from_raw(width, height, decoded.into_raw())
            .ok_or_else(|| CaptureError::CaptureFailed("Failed to create RGB image".to_string()))?;

        Ok(Some(FrameData::new(rgb_image, timestamp, self.frame_count)))
    }

    fn frame_rate(&self) -> Option<f32> {
        Some(self.camera.frame_rate() as f32)
    }

    fn resolution(&self) -> (u32, u32) {
        self.resolution
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn stop(&mut self) {
        self.active = false;
        info!("Webcam capture stopped after {} frames", self.frame_count);
    }
}

impl Drop for WebcamCapture {
    fn drop(&mut self) {
        self.stop();
    }
}

