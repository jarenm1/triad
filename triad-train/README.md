# Triad Train

Training infrastructure and data ingest pipeline for 4D Gaussian Splatting with real-time scene reconstruction.

## Overview

This crate provides:
- **Data Ingest**: Interfaces for camera streams, point clouds, and PLY files
- **Scene Representation**: 4D Gaussian data structures for time-varying scenes
- **Real-Time Reconstruction**: Algorithms for incrementally building scenes from streaming data
- **Training Infrastructure**: (Future) Loss functions, optimizers, and training loops

## Quick Start

### Basic Usage

```rust
use triad_train::{
    ingest::{Point, PointCloud},
    reconstruction::{GaussianInitializer, SceneUpdater, UpdateStrategy},
    scene::SceneGraph,
};

// Create a scene graph
let mut scene = SceneGraph::new();

// Initialize Gaussians from a point cloud
let points = vec![
    Point::new(glam::Vec3::ZERO, glam::Vec3::ONE),
    // ... more points
];
let point_cloud = PointCloud::with_timestamp(points, 0.0);

let initializer = GaussianInitializer::default();
let updater = SceneUpdater::new(initializer, UpdateStrategy::Append);

// Update the scene
updater.update_from_point_cloud(&mut scene, &point_cloud, 0.0);

// Query Gaussians at a specific time
let gaussians = scene.gaussians_at(0.0);
```

### 4D Gaussian Evaluation

```rust
use triad_train::scene::Gaussian4D;
use triad_gpu::GaussianPoint;

// Create a 4D Gaussian
let base_gaussian = GaussianPoint { /* ... */ };
let gaussian_4d = Gaussian4D::new(base_gaussian, 0.0);

// Evaluate at a different time
let gaussian_at_t1 = gaussian_4d.evaluate_at(1.0);
```

## Architecture

See [`../docs/crate_structure.md`](../docs/crate_structure.md) for detailed architecture documentation.

## Modules

- **`ingest`**: Camera streams, point clouds, PLY file streaming
- **`scene`**: 4D Gaussian representation, scene graph, temporal management
- **`reconstruction`**: Gaussian initialization and incremental scene updates

## Future Work

- ML framework integration for training
- Advanced temporal interpolation
- Streaming PLY support for large files
- Camera stream implementations (OpenCV, etc.)
