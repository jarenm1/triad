//! Triad Data Crate
//!
//! Data loading and processing utilities for point clouds, PLY files, and geometry.
//! This crate is GPU-agnostic and focuses on data parsing and geometric operations.
//!
//! ## Core Types
//!
//! - [`Point`] - Simple colored point in 3D space
//! - [`Gaussian`] - 3D Gaussian splat with position, rotation, scale, color
//! - [`Triangle`] - Triangle primitive with vertices and color
//!
//! ## Data Loading
//!
//! - [`load_vertices_from_ply`] - Load PLY files as vertex data
//! - [`triangulate_points`] - Generate triangles from point clouds

pub mod ply;
pub mod triangulation;
pub mod types;

pub use ply::{PlyVertex, load_vertices_from_ply, ply_has_faces};
pub use triangulation::{ProjectionPlane, triangulate_points};
pub use types::{Gaussian, Point, Triangle};
