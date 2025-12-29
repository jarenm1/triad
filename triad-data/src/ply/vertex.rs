//! PLY vertex data structures

use glam::Vec3;

/// Intermediate vertex data extracted from PLY for triangle building.
#[derive(Debug, Clone)]
pub struct PlyVertex {
    pub position: Vec3,
    pub color: Vec3,
    pub opacity: f32,
}
