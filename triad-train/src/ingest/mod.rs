//! Data ingestion module
//!
//! Provides interfaces and implementations for ingesting data from various sources:
//! - Camera streams (RGB, RGBD)
//! - Point clouds
//! - PLY files (streaming support)
//! - Frame synchronization

pub mod camera;
pub mod point_cloud;
pub mod ply_stream;

pub use camera::{CameraFrame, CameraStream, StreamError};
pub use point_cloud::{PointCloud, PointCloudFrame};
pub use ply_stream::PlyStream;
