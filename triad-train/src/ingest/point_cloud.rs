//! Point cloud ingestion and processing

use glam::Vec3;

/// A single point in a point cloud
#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub position: Vec3,
    pub color: Vec3,
    pub normal: Option<Vec3>,
}

impl Point {
    pub fn new(position: Vec3, color: Vec3) -> Self {
        Self {
            position,
            color,
            normal: None,
        }
    }

    pub fn with_normal(position: Vec3, color: Vec3, normal: Vec3) -> Self {
        Self {
            position,
            color,
            normal: Some(normal),
        }
    }
}

/// A point cloud frame with optional temporal information
#[derive(Debug, Clone)]
pub struct PointCloud {
    pub points: Vec<Point>,
    pub timestamp: Option<f64>,
}

impl PointCloud {
    pub fn new(points: Vec<Point>) -> Self {
        Self {
            points,
            timestamp: None,
        }
    }

    pub fn with_timestamp(points: Vec<Point>, timestamp: f64) -> Self {
        Self {
            points,
            timestamp: Some(timestamp),
        }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

/// A point cloud frame for streaming scenarios
#[derive(Debug, Clone)]
pub struct PointCloudFrame {
    pub cloud: PointCloud,
    pub camera_pose: Option<CameraPose>,
}

/// Camera pose information
#[derive(Debug, Clone, Copy)]
pub struct CameraPose {
    pub position: Vec3,
    pub rotation: [f32; 4], // quaternion (x, y, z, w)
}

impl PointCloudFrame {
    pub fn new(cloud: PointCloud) -> Self {
        Self {
            cloud,
            camera_pose: None,
        }
    }

    pub fn with_pose(cloud: PointCloud, camera_pose: CameraPose) -> Self {
        Self {
            cloud,
            camera_pose: Some(camera_pose),
        }
    }
}
