# Triad Data

GPU-agnostic data loading and geometric processing utilities.

## Purpose

This crate provides data loading and processing functionality that is independent of GPU rendering. It focuses on:

- PLY file parsing and loading
- Point cloud data structures
- Delaunay triangulation algorithms
- Geometric utilities

## Usage

### Loading PLY Files

```rust
use triad_data::load_vertices_from_ply;

let vertices = load_vertices_from_ply("path/to/file.ply")?;
```

### Triangulation

```rust
use triad_data::triangulate_points;
use glam::Vec3;

let positions = vec![
    Vec3::new(0.0, 0.0, 0.0),
    Vec3::new(1.0, 0.0, 0.0),
    Vec3::new(0.5, 1.0, 0.0),
];

let triangles = triangulate_points(&positions);
```

## Design Philosophy

This crate is intentionally GPU-agnostic. It provides:
- Raw data structures (`PlyVertex`)
- Geometric algorithms (triangulation)
- File I/O utilities

GPU-specific conversions (e.g., to `GaussianPoint` or `TrianglePrimitive`) are provided by `triad-gpu`.

## Dependencies

- `glam`: Math types
- `serde-ply`: PLY file parsing
- `delaunator`: Delaunay triangulation
- `tracing`: Logging
