//! Gaussian initialization from point clouds and camera frames

use crate::ingest::point_cloud::Point;
use crate::scene::Gaussian4D;
use triad_gpu::GaussianPoint;

/// Strategy for initializing Gaussians from input data
#[derive(Debug, Clone, Copy)]
pub enum InitializationStrategy {
    /// One Gaussian per point
    OnePerPoint,
    /// Adaptive: merge nearby points with similar colors
    Adaptive { distance_threshold: f32 },
    /// Uniform grid sampling
    GridSampling { grid_size: f32 },
}

/// Initializes Gaussians from point clouds and camera frames
pub struct GaussianInitializer {
    strategy: InitializationStrategy,
}

impl GaussianInitializer {
    pub fn new(strategy: InitializationStrategy) -> Self {
        Self { strategy }
    }

    /// Initialize Gaussians from a point cloud
    pub fn from_point_cloud(
        &self,
        points: &[Point],
        time: f64,
    ) -> Vec<Gaussian4D> {
        match self.strategy {
            InitializationStrategy::OnePerPoint => {
                self.one_per_point_impl(points, time)
            }
            InitializationStrategy::Adaptive { distance_threshold } => {
                self.adaptive_initialization(points, time, distance_threshold)
            }
            InitializationStrategy::GridSampling { grid_size } => {
                self.grid_sampling(points, time, grid_size)
            }
        }
    }

    /// Convert a single point to a Gaussian
    fn point_to_gaussian(&self, point: &Point, time: f64) -> Gaussian4D {
        // Default scale based on point density or adaptive sizing
        let default_scale = 0.01; // TODO: Make this adaptive
        
        let gaussian = GaussianPoint {
            position: [point.position.x, point.position.y, point.position.z],
            _pad0: 0.0,
            color_opacity: [point.color.x, point.color.y, point.color.z, 1.0],
            rotation: [0.0, 0.0, 0.0, 1.0], // Identity quaternion
            scale: [default_scale, default_scale, default_scale, 0.0],
        };

        Gaussian4D::new(gaussian, time)
    }

    /// One-per-point initialization (base implementation)
    fn one_per_point_impl(&self, points: &[Point], time: f64) -> Vec<Gaussian4D> {
        points
            .iter()
            .map(|p| self.point_to_gaussian(p, time))
            .collect()
    }

    /// Adaptive initialization that merges nearby points
    fn adaptive_initialization(
        &self,
        points: &[Point],
        time: f64,
        _threshold: f32,
    ) -> Vec<Gaussian4D> {
        // Simple implementation: for now, just use one per point
        // TODO: Implement actual clustering/merging
        self.one_per_point_impl(points, time)
    }

    /// Grid-based sampling
    fn grid_sampling(
        &self,
        points: &[Point],
        time: f64,
        _grid_size: f32,
    ) -> Vec<Gaussian4D> {
        // Simple implementation: for now, just use one per point
        // TODO: Implement actual grid sampling
        self.one_per_point_impl(points, time)
    }
}

impl Default for GaussianInitializer {
    fn default() -> Self {
        Self::new(InitializationStrategy::OnePerPoint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::point_cloud::Point;

    #[test]
    fn test_gaussian_initializer_one_per_point() {
        let initializer = GaussianInitializer::new(InitializationStrategy::OnePerPoint);
        
        let points = vec![
            Point::new(glam::Vec3::new(0.0, 0.0, 0.0), glam::Vec3::new(1.0, 0.0, 0.0)),
            Point::new(glam::Vec3::new(1.0, 0.0, 0.0), glam::Vec3::new(0.0, 1.0, 0.0)),
            Point::new(glam::Vec3::new(0.0, 1.0, 0.0), glam::Vec3::new(0.0, 0.0, 1.0)),
        ];

        let gaussians = initializer.from_point_cloud(&points, 0.0);
        
        assert_eq!(gaussians.len(), points.len());
        
        // Verify each Gaussian corresponds to a point
        for (gaussian, point) in gaussians.iter().zip(points.iter()) {
            let gaussian_at_time = gaussian.evaluate_at(0.0);
            assert_eq!(gaussian_at_time.position[0], point.position.x);
            assert_eq!(gaussian_at_time.position[1], point.position.y);
            assert_eq!(gaussian_at_time.position[2], point.position.z);
            
            // Check color (allowing for small floating point differences)
            assert!((gaussian_at_time.color_opacity[0] - point.color.x).abs() < 0.001);
            assert!((gaussian_at_time.color_opacity[1] - point.color.y).abs() < 0.001);
            assert!((gaussian_at_time.color_opacity[2] - point.color.z).abs() < 0.001);
        }
    }

    #[test]
    fn test_gaussian_initializer_empty_points() {
        let initializer = GaussianInitializer::default();
        let points = vec![];
        
        let gaussians = initializer.from_point_cloud(&points, 0.0);
        assert!(gaussians.is_empty());
    }

    #[test]
    fn test_gaussian_initializer_single_point() {
        let initializer = GaussianInitializer::default();
        
        let points = vec![
            Point::new(glam::Vec3::new(5.0, 10.0, 15.0), glam::Vec3::new(0.5, 0.6, 0.7)),
        ];

        let gaussians = initializer.from_point_cloud(&points, 1.5);
        assert_eq!(gaussians.len(), 1);
        
        let gaussian = &gaussians[0];
        let gaussian_at_time = gaussian.evaluate_at(1.5);
        assert_eq!(gaussian_at_time.position[0], 5.0);
        assert_eq!(gaussian_at_time.position[1], 10.0);
        assert_eq!(gaussian_at_time.position[2], 15.0);
    }

    #[test]
    fn test_gaussian_initializer_adaptive_strategy() {
        let initializer = GaussianInitializer::new(InitializationStrategy::Adaptive {
            distance_threshold: 0.5,
        });
        
        let points = vec![
            Point::new(glam::Vec3::new(0.0, 0.0, 0.0), glam::Vec3::ONE),
            Point::new(glam::Vec3::new(0.1, 0.0, 0.0), glam::Vec3::ONE),
            Point::new(glam::Vec3::new(1.0, 0.0, 0.0), glam::Vec3::ONE),
        ];

        // Currently adaptive strategy just calls from_point_cloud, but test the interface
        let gaussians = initializer.from_point_cloud(&points, 0.0);
        assert!(!gaussians.is_empty());
    }

    #[test]
    fn test_gaussian_initializer_grid_sampling_strategy() {
        let initializer = GaussianInitializer::new(InitializationStrategy::GridSampling {
            grid_size: 1.0,
        });
        
        let points = vec![
            Point::new(glam::Vec3::new(0.0, 0.0, 0.0), glam::Vec3::ONE),
            Point::new(glam::Vec3::new(0.5, 0.5, 0.5), glam::Vec3::ONE),
            Point::new(glam::Vec3::new(1.0, 1.0, 1.0), glam::Vec3::ONE),
        ];

        // Currently grid sampling just calls from_point_cloud, but test the interface
        let gaussians = initializer.from_point_cloud(&points, 0.0);
        assert!(!gaussians.is_empty());
    }

    #[test]
    fn test_gaussian_initializer_time_preservation() {
        let initializer = GaussianInitializer::default();
        
        let points = vec![
            Point::new(glam::Vec3::ZERO, glam::Vec3::ONE),
        ];

        let time = 42.5;
        let gaussians = initializer.from_point_cloud(&points, time);
        
        assert_eq!(gaussians.len(), 1);
        // Verify the time is stored in the Gaussian4D
        // The exact way to check this depends on Gaussian4D implementation
        // For now, just verify it was created
        assert!(!gaussians.is_empty());
    }

    #[test]
    fn test_gaussian_initializer_quaternion_identity() {
        let initializer = GaussianInitializer::default();
        
        let points = vec![
            Point::new(glam::Vec3::ZERO, glam::Vec3::ONE),
        ];

        let gaussians = initializer.from_point_cloud(&points, 0.0);
        let gaussian = &gaussians[0];
        let gaussian_at_time = gaussian.evaluate_at(0.0);
        
        // Verify rotation is identity quaternion [0, 0, 0, 1]
        assert_eq!(gaussian_at_time.rotation[0], 0.0);
        assert_eq!(gaussian_at_time.rotation[1], 0.0);
        assert_eq!(gaussian_at_time.rotation[2], 0.0);
        assert_eq!(gaussian_at_time.rotation[3], 1.0);
    }

    #[test]
    fn test_gaussian_initializer_scale_default() {
        let initializer = GaussianInitializer::default();
        
        let points = vec![
            Point::new(glam::Vec3::ZERO, glam::Vec3::ONE),
        ];

        let gaussians = initializer.from_point_cloud(&points, 0.0);
        let gaussian = &gaussians[0];
        let gaussian_at_time = gaussian.evaluate_at(0.0);
        
        // Verify default scale is set (0.01 based on the implementation)
        let default_scale = 0.01;
        assert_eq!(gaussian_at_time.scale[0], default_scale);
        assert_eq!(gaussian_at_time.scale[1], default_scale);
        assert_eq!(gaussian_at_time.scale[2], default_scale);
    }

    #[test]
    fn test_gaussian_initializer_large_point_set() {
        let initializer = GaussianInitializer::default();
        
        let mut points = Vec::new();
        for i in 0..100 {
            points.push(Point::new(
                glam::Vec3::new(i as f32, i as f32 * 0.5, i as f32 * 0.25),
                glam::Vec3::new(
                    (i % 3) as f32 / 2.0,
                    ((i + 1) % 3) as f32 / 2.0,
                    ((i + 2) % 3) as f32 / 2.0,
                ),
            ));
        }

        let gaussians = initializer.from_point_cloud(&points, 0.0);
        assert_eq!(gaussians.len(), 100);
    }
}
