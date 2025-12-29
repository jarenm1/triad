//! Scene representation for 4D Gaussian Splatting
//!
//! This module provides data structures and algorithms for representing
//! time-varying Gaussian scenes.

pub mod gaussian_4d;
pub mod scene_graph;
pub mod temporal;

pub use gaussian_4d::Gaussian4D;
pub use scene_graph::SceneGraph;
pub use temporal::{TemporalKeyframe, TimeRange};
