//! Delaunay triangulation for point clouds.
//!
//! This module provides functions to generate triangles from unstructured
//! point clouds using Delaunay triangulation.

use crate::TrianglePrimitive;
use crate::ply_loader::PlyVertex;
use delaunator::{Point, triangulate};
use glam::Vec3;
use tracing::{debug, info, warn};

/// Perform 2D Delaunay triangulation on a set of 3D points.
///
/// Projects points onto the XY plane for triangulation, then uses the
/// original 3D positions for the resulting triangles.
///
/// Returns a list of triangle index triplets `[i0, i1, i2]`.
pub fn triangulate_points_xy(positions: &[Vec3]) -> Vec<[usize; 3]> {
    if positions.len() < 3 {
        warn!("Not enough points for triangulation (need at least 3)");
        return Vec::new();
    }

    // Project to XY plane for 2D Delaunay
    let points: Vec<Point> = positions
        .iter()
        .map(|p| Point {
            x: p.x as f64,
            y: p.y as f64,
        })
        .collect();

    let result = triangulate(&points);

    // Convert triangle indices to triplets
    let mut triangles = Vec::with_capacity(result.triangles.len() / 3);
    for chunk in result.triangles.chunks_exact(3) {
        triangles.push([chunk[0], chunk[1], chunk[2]]);
    }

    debug!(
        "Triangulated {} points into {} triangles (XY projection)",
        positions.len(),
        triangles.len()
    );

    triangles
}

/// Perform 2D Delaunay triangulation on a set of 3D points.
///
/// Projects points onto the XZ plane for triangulation, then uses the
/// original 3D positions for the resulting triangles.
///
/// Returns a list of triangle index triplets `[i0, i1, i2]`.
pub fn triangulate_points_xz(positions: &[Vec3]) -> Vec<[usize; 3]> {
    if positions.len() < 3 {
        warn!("Not enough points for triangulation (need at least 3)");
        return Vec::new();
    }

    // Project to XZ plane for 2D Delaunay
    let points: Vec<Point> = positions
        .iter()
        .map(|p| Point {
            x: p.x as f64,
            y: p.z as f64,
        })
        .collect();

    let result = triangulate(&points);

    // Convert triangle indices to triplets
    let mut triangles = Vec::with_capacity(result.triangles.len() / 3);
    for chunk in result.triangles.chunks_exact(3) {
        triangles.push([chunk[0], chunk[1], chunk[2]]);
    }

    debug!(
        "Triangulated {} points into {} triangles (XZ projection)",
        positions.len(),
        triangles.len()
    );

    triangles
}

/// Determine the best projection plane for triangulation based on point distribution.
///
/// Returns the plane with the largest spread in the point cloud.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionPlane {
    XY,
    XZ,
    YZ,
}

/// Analyze point cloud and determine the best projection plane for triangulation.
pub fn best_projection_plane(positions: &[Vec3]) -> ProjectionPlane {
    if positions.is_empty() {
        return ProjectionPlane::XY;
    }

    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);

    for pos in positions {
        min = min.min(*pos);
        max = max.max(*pos);
    }

    let extent = max - min;

    // Choose the plane that maximizes spread (avoid the smallest dimension)
    if extent.z <= extent.x && extent.z <= extent.y {
        ProjectionPlane::XY // Z is smallest, project onto XY
    } else if extent.y <= extent.x && extent.y <= extent.z {
        ProjectionPlane::XZ // Y is smallest, project onto XZ
    } else {
        ProjectionPlane::YZ // X is smallest, project onto YZ
    }
}

/// Perform 2D Delaunay triangulation on a set of 3D points.
///
/// Automatically selects the best projection plane based on point distribution.
///
/// Returns a list of triangle index triplets `[i0, i1, i2]`.
pub fn triangulate_points(positions: &[Vec3]) -> Vec<[usize; 3]> {
    if positions.len() < 3 {
        warn!("Not enough points for triangulation (need at least 3)");
        return Vec::new();
    }

    let plane = best_projection_plane(positions);
    debug!("Using projection plane: {:?}", plane);

    let points: Vec<Point> = match plane {
        ProjectionPlane::XY => positions
            .iter()
            .map(|p| Point {
                x: p.x as f64,
                y: p.y as f64,
            })
            .collect(),
        ProjectionPlane::XZ => positions
            .iter()
            .map(|p| Point {
                x: p.x as f64,
                y: p.z as f64,
            })
            .collect(),
        ProjectionPlane::YZ => positions
            .iter()
            .map(|p| Point {
                x: p.y as f64,
                y: p.z as f64,
            })
            .collect(),
    };

    let result = triangulate(&points);

    // Convert triangle indices to triplets
    let mut triangles = Vec::with_capacity(result.triangles.len() / 3);
    for chunk in result.triangles.chunks_exact(3) {
        triangles.push([chunk[0], chunk[1], chunk[2]]);
    }

    info!(
        "Triangulated {} points into {} triangles (plane: {:?})",
        positions.len(),
        triangles.len(),
        plane
    );

    triangles
}

/// Build triangle primitives from PLY vertex data using Delaunay triangulation.
///
/// This is the main entry point for converting point clouds to triangles.
pub fn build_triangles_from_vertices(vertices: &[PlyVertex]) -> Vec<TrianglePrimitive> {
    if vertices.len() < 3 {
        warn!("Not enough vertices to build triangles");
        return Vec::new();
    }

    // Extract positions for triangulation
    let positions: Vec<Vec3> = vertices.iter().map(|v| v.position).collect();

    // Triangulate
    let triangle_indices = triangulate_points(&positions);

    // Build primitives
    let mut triangles = Vec::with_capacity(triangle_indices.len());

    for [i0, i1, i2] in triangle_indices {
        let v0 = &vertices[i0];
        let v1 = &vertices[i1];
        let v2 = &vertices[i2];

        // Average vertex colors and opacities
        let avg_color = (v0.color + v1.color + v2.color) / 3.0;
        let avg_opacity = (v0.opacity + v1.opacity + v2.opacity) / 3.0;

        triangles.push(TrianglePrimitive::new(
            v0.position,
            v1.position,
            v2.position,
            avg_color,
            avg_opacity,
        ));
    }

    info!(
        "Built {} triangle primitives from vertices",
        triangles.len()
    );
    triangles
}

/// Filter triangles by maximum edge length.
///
/// Removes triangles where any edge exceeds the given threshold.
/// Useful for removing "long" triangles that span gaps in the point cloud.
pub fn filter_by_edge_length(
    triangles: &[TrianglePrimitive],
    max_edge_length: f32,
) -> Vec<TrianglePrimitive> {
    let max_sq = max_edge_length * max_edge_length;

    triangles
        .iter()
        .filter(|t| {
            let v0 = t.vertex0();
            let v1 = t.vertex1();
            let v2 = t.vertex2();

            let e0_sq = (v1 - v0).length_squared();
            let e1_sq = (v2 - v1).length_squared();
            let e2_sq = (v0 - v2).length_squared();

            e0_sq <= max_sq && e1_sq <= max_sq && e2_sq <= max_sq
        })
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangulate_simple() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 1.0, 0.0),
            Vec3::new(0.5, 0.5, 0.0),
        ];

        let triangles = triangulate_points(&positions);
        assert!(!triangles.is_empty());
    }

    #[test]
    fn test_triangulate_insufficient_points() {
        let positions = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];

        let triangles = triangulate_points(&positions);
        assert!(triangles.is_empty());
    }
}
