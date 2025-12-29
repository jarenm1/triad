# Separation of Concerns Improvements

## Overview

This document describes the refactoring to improve separation of concerns in the Triad project, moving data loading and processing out of GPU-specific crates.

## Changes Made

### 1. New Crate: `triad-data`

Created a new crate dedicated to data loading and geometric processing that is GPU-agnostic.

**Responsibilities:**
- PLY file parsing and loading
- Point cloud data structures (`PlyVertex`)
- Delaunay triangulation algorithms
- Geometric utilities

**Location:** `triad-data/`

**Modules:**
- `ply/`: PLY file loading (`loader.rs`, `vertex.rs`)
- `triangulation.rs`: Delaunay triangulation algorithms

### 2. Refactored `triad-gpu`

**Removed:**
- Direct PLY parsing (moved to `triad-data`)
- Triangulation algorithms (moved to `triad-data`)

**Kept:**
- GPU-specific type conversions (`load_gaussians_from_ply`, `load_triangles_from_ply`)
- GPU type definitions (`GaussianPoint`, `TrianglePrimitive`, `CameraUniforms`)
- GPU-specific triangulation utilities (`build_triangles_from_vertices`)

**Rationale:**
- GPU types (`GaussianPoint`, `TrianglePrimitive`) are tightly coupled to shader layouts and GPU memory layout
- Conversion functions that produce GPU types belong in the GPU crate
- The actual parsing and geometric algorithms are GPU-agnostic and can be reused

### 3. Updated Dependencies

**`triad-gpu`:**
- Now depends on `triad-data` for PLY parsing and triangulation
- Re-exports `PlyVertex` from `triad-data` for convenience
- Re-exports triangulation functions from `triad-data`

**`triad-train`:**
- Now depends on `triad-data` for data loading
- Can use PLY loading without depending on GPU types

**Examples:**
- Updated to use `triad-data` directly where appropriate
- Still use `triad-gpu` for GPU-specific conversions

## Architecture Benefits

### 1. Clear Separation
- **Data Loading**: `triad-data` handles all file I/O and parsing
- **GPU Types**: `triad-gpu` handles GPU-specific data structures
- **Training**: `triad-train` can work with data without GPU dependencies

### 2. Reusability
- `triad-data` can be used by any crate that needs PLY loading or triangulation
- No need to pull in GPU dependencies for data processing
- Easier to test data loading independently

### 3. Maintainability
- Changes to PLY parsing don't affect GPU code
- Changes to GPU types don't affect data loading
- Clear boundaries make it easier to understand dependencies

## Other Separation Opportunities Considered

### Types (`GaussianPoint`, `TrianglePrimitive`)

**Decision:** Keep in `triad-gpu`

**Rationale:**
- These types are GPU-specific (bytemuck Pod/Zeroable, match shader layouts)
- Tightly coupled to rendering pipeline
- Used primarily for GPU buffer creation

**Alternative Considered:**
- Create `triad-types` crate
- Rejected because types are too GPU-specific

### Frame Graph

**Decision:** Keep in `triad-gpu`

**Rationale:**
- Frame graph is tightly coupled to wgpu
- Manages GPU resource lifecycle
- Not reusable outside GPU context

### Resource Registry

**Decision:** Keep in `triad-gpu`

**Rationale:**
- Manages GPU resource handles
- Type-safe wrapper around wgpu resources
- Not useful outside GPU context

## Future Separation Opportunities

### 1. Shader Management
- Could extract shader compilation/loading to separate crate
- Would allow shader hot-reloading without GPU crate changes

### 2. Camera/Controls
- `triad-window` contains camera and controls
- Could be moved to separate `triad-camera` crate if needed by other projects

### 3. Math Utilities
- Currently using `glam` directly everywhere
- Could create `triad-math` wrapper if custom math operations are needed

## Migration Guide

### For Code Using PLY Loading

**Before:**
```rust
use triad_gpu::ply_loader;
let vertices = ply_loader::load_vertices_from_ply(path)?;
```

**After:**
```rust
use triad_data::load_vertices_from_ply;
let vertices = load_vertices_from_ply(path)?;
```

### For Code Using Triangulation

**Before:**
```rust
use triad_gpu::triangulation;
let triangles = triangulation::triangulate_points(&positions);
```

**After:**
```rust
use triad_data::triangulate_points;
let triangles = triangulate_points(&positions);
```

### For Code Converting to GPU Types

**No change needed:**
```rust
use triad_gpu::ply_loader;
let gaussians = ply_loader::load_gaussians_from_ply(path)?; // Still works
```

## Summary

The main improvement is the creation of `triad-data` as a GPU-agnostic data loading crate. This allows:
- Better separation between data processing and GPU rendering
- Reusability of data loading code
- Easier testing and maintenance
- Clearer dependency graph

The GPU-specific types remain in `triad-gpu` where they belong, as they are tightly coupled to the rendering pipeline.
