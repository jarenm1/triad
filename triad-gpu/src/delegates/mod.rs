//! Render delegate implementations for different visualization modes.
//!
//! - [`PointDelegate`] - Point cloud rendering
//! - [`GaussianDelegate`] - Gaussian splatting rendering
//! - [`TriangleDelegate`] - Triangle splatting rendering

mod point;
mod gaussian;
mod triangle;

pub use point::{PointDelegate, PointInitData};
pub use gaussian::{GaussianDelegate, GaussianInitData};
pub use triangle::{TriangleDelegate, TriangleInitData};

