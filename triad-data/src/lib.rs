//! Triad Data Crate
//!
//! Data loading and processing utilities for point clouds, PLY files, and geometry.
//! This crate is GPU-agnostic and focuses on data parsing and geometric operations.

pub mod ply;
pub mod triangulation;

pub use ply::{PlyVertex, load_vertices_from_ply, ply_has_faces};
pub use triangulation::{triangulate_points, ProjectionPlane};
