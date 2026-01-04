//! Triad Capture - Video capture from multiple sources
//!
//! This crate provides implementations of the `CameraStream` trait from triad-train
//! for capturing video from various sources:
//!
//! - Webcams (via nokhwa, requires `webcam` feature)
//! - Video files (planned, via ffmpeg)
//! - Network streams (planned, RTSP)
//!
//! ## Example
//!
//! ```ignore
//! use triad_capture::WebcamCapture;
//! use triad_train::ingest::CameraStream;
//!
//! let mut camera = WebcamCapture::new(0)?;
//! while let Some(frame) = camera.next_frame()? {
//!     // Process frame...
//! }
//! ```

mod source;

#[cfg(feature = "webcam")]
mod webcam;

pub use source::{CaptureError, CaptureSource, FrameData};

#[cfg(feature = "webcam")]
pub use webcam::WebcamCapture;

// Re-export CameraStream trait for convenience
pub use triad_train::ingest::{CameraFrame, CameraStream, StreamError};

