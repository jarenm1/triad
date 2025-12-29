//! Triad Training Crate
//!
//! This crate provides infrastructure for training 4D Gaussian Splatting models
//! and real-time scene reconstruction. It focuses on data ingest pipelines and
//! scene representation for dynamic, time-varying scenes.
//!
//! ## Modules
//!
//! - [`ingest`]: Data ingestion from various sources (cameras, point clouds, files)
//! - [`scene`]: 4D Gaussian scene representation and temporal management
//! - [`reconstruction`]: Real-time scene reconstruction algorithms
//! - [`train`]: Training infrastructure (loss functions, optimizers)

pub mod ingest;
pub mod reconstruction;
pub mod scene;

// Training module will be added later when ML framework integration is needed
// pub mod train;

pub use scene::Gaussian4D;
