//! Delaunay triangulation for point clouds.
//!
//! This module provides functions to generate triangles from unstructured
//! point clouds using Delaunay triangulation.

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

    #[test]
    fn test_triangulate_empty_input() {
        let positions = vec![];
        let triangles = triangulate_points(&positions);
        assert!(triangles.is_empty());
    }

    #[test]
    fn test_triangulate_single_point() {
        let positions = vec![Vec3::new(0.0, 0.0, 0.0)];
        let triangles = triangulate_points(&positions);
        assert!(triangles.is_empty());
    }

    #[test]
    fn test_triangulate_points_xy() {
        // Test XY plane projection explicitly
        let positions = vec![
            Vec3::new(0.0, 0.0, 5.0), // Z should be ignored
            Vec3::new(1.0, 0.0, 10.0),
            Vec3::new(0.5, 1.0, 15.0),
            Vec3::new(0.5, 0.5, 20.0),
        ];

        let triangles = triangulate_points_xy(&positions);
        assert!(!triangles.is_empty());

        // Verify all indices are valid
        for triangle in &triangles {
            for &idx in triangle {
                assert!(idx < positions.len() as usize);
            }
        }
    }

    #[test]
    fn test_triangulate_points_xz() {
        // Test XZ plane projection explicitly
        let positions = vec![
            Vec3::new(0.0, 5.0, 0.0), // Y should be ignored
            Vec3::new(1.0, 10.0, 0.0),
            Vec3::new(0.5, 15.0, 1.0),
            Vec3::new(0.5, 20.0, 0.5),
        ];

        let triangles = triangulate_points_xz(&positions);
        assert!(!triangles.is_empty());

        // Verify all indices are valid
        for triangle in &triangles {
            for &idx in triangle {
                assert!(idx < positions.len() as usize);
            }
        }
    }

    #[test]
    fn test_triangulate_grid_pattern() {
        // Create a grid of points to test delaunator with regular patterns
        let mut positions = Vec::new();
        for y in 0..5 {
            for x in 0..5 {
                positions.push(Vec3::new(x as f32, y as f32, 0.0));
            }
        }

        let triangles = triangulate_points(&positions);
        assert!(!triangles.is_empty());

        // Grid of 5x5 = 25 points should produce many triangles
        // Delaunay triangulation of a grid typically produces ~2*(n-2) triangles
        // where n is the number of points
        assert!(triangles.len() >= 20);
    }

    #[test]
    fn test_triangulate_collinear_points() {
        // Test with collinear points (edge case for delaunator)
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        ];

        // Collinear points might produce degenerate triangles or fail
        // The delaunator library should handle this gracefully
        let triangles = triangulate_points(&positions);
        // Result may be empty or have degenerate triangles, both are acceptable
    }

    #[test]
    fn test_triangulate_duplicate_points() {
        // Test with duplicate points (edge case)
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0), // Duplicate
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 1.0, 0.0),
        ];

        // Delaunator should handle duplicates (may produce fewer triangles)
        let triangles = triangulate_points(&positions);
        // Result depends on delaunator's handling of duplicates
    }

    #[test]
    fn test_best_projection_plane_xy() {
        // Points spread mostly in XY plane (small Z extent)
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(0.0, 10.0, 0.0),
            Vec3::new(10.0, 10.0, 0.0),
            Vec3::new(5.0, 5.0, 1.0), // Small Z variation
        ];

        let plane = best_projection_plane(&positions);
        assert_eq!(plane, ProjectionPlane::XY);
    }

    #[test]
    fn test_best_projection_plane_xz() {
        // Points spread mostly in XZ plane (small Y extent)
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 10.0),
            Vec3::new(10.0, 0.0, 10.0),
            Vec3::new(5.0, 1.0, 5.0), // Small Y variation
        ];

        let plane = best_projection_plane(&positions);
        assert_eq!(plane, ProjectionPlane::XZ);
    }

    #[test]
    fn test_best_projection_plane_empty() {
        let positions = vec![];
        let plane = best_projection_plane(&positions);
        // Should default to XY
        assert_eq!(plane, ProjectionPlane::XY);
    }

    #[test]
    fn test_best_projection_plane_single_point() {
        let positions = vec![Vec3::new(1.0, 2.0, 3.0)];
        let plane = best_projection_plane(&positions);
        // Should default to XY
        assert_eq!(plane, ProjectionPlane::XY);
    }

    #[test]
    fn test_triangulate_large_point_set() {
        // Test with a larger point set to verify delaunator performance
        let mut positions = Vec::new();
        for i in 0..100 {
            let angle = (i as f32) * 2.0 * std::f32::consts::PI / 100.0;
            positions.push(Vec3::new(angle.cos() * 10.0, angle.sin() * 10.0, 0.0));
        }

        let triangles = triangulate_points(&positions);
        assert!(!triangles.is_empty());

        // Circular arrangement should produce many triangles
        assert!(triangles.len() >= 90);
    }

    #[test]
    fn test_triangulate_3d_points() {
        // Test with truly 3D points (not coplanar)
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
        ];

        let triangles = triangulate_points(&positions);
        assert!(!triangles.is_empty());

        // Verify triangle indices are valid
        for triangle in &triangles {
            for &idx in triangle {
                assert!(idx < positions.len());
            }
        }
    }

    #[test]
    fn test_triangulate_negative_coordinates() {
        // Test with negative coordinates
        let positions = vec![
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-0.5, 0.0, 0.0),
        ];

        let triangles = triangulate_points(&positions);
        assert!(!triangles.is_empty());
    }

    #[test]
    fn test_triangulate_very_small_coordinates() {
        // Test with very small coordinates (precision edge case)
        let positions = vec![
            Vec3::new(0.0001, 0.0001, 0.0),
            Vec3::new(0.0002, 0.0001, 0.0),
            Vec3::new(0.00015, 0.0002, 0.0),
        ];

        let triangles = triangulate_points(&positions);
        // Should still produce at least one triangle
        assert!(!triangles.is_empty());
    }
}
