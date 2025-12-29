//! Real-time scene reconstruction algorithms
//!
//! This module provides algorithms for incrementally building and updating
//! Gaussian scenes from streaming data sources.

pub mod initializer;
pub mod updater;

pub use initializer::{GaussianInitializer, InitializationStrategy};
pub use updater::{SceneUpdater, UpdateStrategy};
