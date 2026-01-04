//! Core data types for point clouds, gaussians, and triangles.
//!
//! These are CPU-side representations used throughout the triad ecosystem.
//! GPU-specific types with bytemuck derive live in triad-gpu.

use glam::Vec3;

/// A simple colored point in 3D space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    /// Position in world space.
    pub position: Vec3,
    /// RGB color (linear, 0-1 range).
    pub color: Vec3,
}

impl Point {
    /// Create a new point with position and color.
    pub fn new(position: Vec3, color: Vec3) -> Self {
        Self { position, color }
    }

    /// Create a white point at the given position.
    pub fn white(position: Vec3) -> Self {
        Self {
            position,
            color: Vec3::ONE,
        }
    }
}

impl Default for Point {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            color: Vec3::splat(0.8),
        }
    }
}

/// A 3D Gaussian splat (CPU representation).
///
/// This represents an anisotropic Gaussian with position, orientation, scale, color, and opacity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gaussian {
    /// Center position in world space.
    pub position: Vec3,
    /// Rotation quaternion (x, y, z, w).
    pub rotation: [f32; 4],
    /// Per-axis scale (x, y, z).
    pub scale: Vec3,
    /// RGB color (linear, 0-1 range).
    pub color: Vec3,
    /// Opacity (0-1).
    pub opacity: f32,
}

impl Gaussian {
    /// Create a new Gaussian with all parameters.
    pub fn new(position: Vec3, rotation: [f32; 4], scale: Vec3, color: Vec3, opacity: f32) -> Self {
        Self {
            position,
            rotation,
            scale,
            color,
            opacity,
        }
    }

    /// Create a Gaussian from a point with default scale and rotation.
    pub fn from_point(point: &Point, scale: f32) -> Self {
        Self {
            position: point.position,
            rotation: [0.0, 0.0, 0.0, 1.0], // Identity quaternion
            scale: Vec3::splat(scale),
            color: point.color,
            opacity: 1.0,
        }
    }

    /// Create a spherical Gaussian (uniform scale).
    pub fn spherical(position: Vec3, radius: f32, color: Vec3, opacity: f32) -> Self {
        Self {
            position,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: Vec3::splat(radius),
            color,
            opacity,
        }
    }
}

impl Default for Gaussian {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: Vec3::splat(0.01),
            color: Vec3::splat(0.8),
            opacity: 1.0,
        }
    }
}

impl From<&Point> for Gaussian {
    fn from(point: &Point) -> Self {
        Self::from_point(point, 0.01)
    }
}

/// A triangle primitive with three vertices and a color.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Triangle {
    /// First vertex position.
    pub v0: Vec3,
    /// Second vertex position.
    pub v1: Vec3,
    /// Third vertex position.
    pub v2: Vec3,
    /// RGB color (linear, 0-1 range).
    pub color: Vec3,
    /// Opacity (0-1).
    pub opacity: f32,
}

impl Triangle {
    /// Create a new triangle from three vertices, color, and opacity.
    pub fn new(v0: Vec3, v1: Vec3, v2: Vec3, color: Vec3, opacity: f32) -> Self {
        Self {
            v0,
            v1,
            v2,
            color,
            opacity,
        }
    }

    /// Compute the center (centroid) of the triangle.
    pub fn center(&self) -> Vec3 {
        (self.v0 + self.v1 + self.v2) / 3.0
    }

    /// Compute the normal of the triangle (not normalized).
    pub fn normal(&self) -> Vec3 {
        let e1 = self.v1 - self.v0;
        let e2 = self.v2 - self.v0;
        e1.cross(e2)
    }

    /// Compute the normalized normal of the triangle.
    pub fn unit_normal(&self) -> Vec3 {
        self.normal().normalize_or_zero()
    }

    /// Compute the area of the triangle.
    pub fn area(&self) -> f32 {
        self.normal().length() * 0.5
    }
}

impl Default for Triangle {
    fn default() -> Self {
        Self {
            v0: Vec3::ZERO,
            v1: Vec3::X,
            v2: Vec3::Y,
            color: Vec3::splat(0.8),
            opacity: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_creation() {
        let p = Point::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(p.position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(p.color, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn test_gaussian_from_point() {
        let p = Point::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.5, 0.5, 0.5));
        let g = Gaussian::from_point(&p, 0.1);
        assert_eq!(g.position, p.position);
        assert_eq!(g.color, p.color);
        assert_eq!(g.scale, Vec3::splat(0.1));
    }

    #[test]
    fn test_triangle_center() {
        let t = Triangle::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
            Vec3::ONE,
            1.0,
        );
        assert_eq!(t.center(), Vec3::new(1.0, 1.0, 0.0));
    }

    #[test]
    fn test_triangle_normal() {
        let t = Triangle::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::ONE,
            1.0,
        );
        let normal = t.unit_normal();
        assert!((normal - Vec3::Z).length() < 0.001);
    }
}
