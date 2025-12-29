# Triad Crate Structure and Responsibilities

## Current Crates

### `triad-gpu`
**Purpose**: Low-level GPU rendering infrastructure
**Responsibilities**:
- wgpu device/queue management
- Frame graph system for render passes
- Resource registry for GPU resource lifecycle
- Shader compilation and pipeline building
- Basic rendering primitives (GaussianPoint, TrianglePrimitive)
- PLY file loading utilities

**Dependencies**: wgpu, glam, bytemuck, serde-ply

### `triad-window`
**Purpose**: Window management and application framework
**Responsibilities**:
- Window creation and event handling (winit)
- Camera controls and input handling
- Render delegate system for pluggable rendering
- Application lifecycle management
- Scene bounds computation

**Dependencies**: triad-gpu, winit, glam

## New Crate: `triad-train`

### Purpose
Training infrastructure and data ingest pipeline for 4D Gaussian Splatting with real-time scene reconstruction.

### Responsibilities

#### 1. Data Ingest (`ingest` module)
- Camera stream interfaces (RGB, RGBD)
- Point cloud ingestion and processing
- PLY file streaming support
- Frame synchronization
- Pose estimation integration points

#### 2. Scene Representation (`scene` module)
- 4D Gaussian data structures (time-varying Gaussians)
- Scene graph for dynamic scenes
- Temporal keyframe management
- Streaming buffer management
- Incremental scene updates

#### 3. Training Infrastructure (`train` module - future)
- Loss function computation
- Gradient-based optimization
- Parameter update strategies
- Checkpointing and model serialization
- Training loop orchestration

#### 4. Real-Time Reconstruction (`reconstruction` module)
- Online Gaussian initialization
- Incremental point cloud fusion
- Adaptive Gaussian management (splitting/pruning)
- Memory-efficient streaming structures

### Module Structure
```
triad-train/
├── src/
│   ├── lib.rs
│   ├── ingest/
│   │   ├── mod.rs
│   │   ├── camera.rs          # Camera stream interfaces
│   │   ├── point_cloud.rs     # Point cloud ingestion
│   │   └── ply_stream.rs       # Streaming PLY support
│   ├── scene/
│   │   ├── mod.rs
│   │   ├── gaussian_4d.rs      # 4D Gaussian representation
│   │   ├── scene_graph.rs      # Scene organization
│   │   └── temporal.rs         # Time-based operations
│   ├── reconstruction/
│   │   ├── mod.rs
│   │   ├── initializer.rs     # Gaussian initialization
│   │   └── updater.rs          # Incremental updates
│   └── train/                  # Future training module
│       ├── mod.rs
│       ├── loss.rs
│       └── optimizer.rs
```

### Dependencies
- `triad-gpu`: For GaussianPoint types and GPU resource access
- `glam`: Math types
- `bytemuck`: Data serialization
- `serde`: Serialization (for checkpoints)
- `tokio` or `async-std`: Async runtime for streaming (optional)
- `image`: Image processing for camera streams

### Integration Points

1. **With `triad-gpu`**:
   - Extends `GaussianPoint` to `Gaussian4D` with temporal data
   - Provides streaming buffer updates to GPU resources
   - Integrates with resource registry for dynamic updates

2. **With `triad-window`**:
   - Provides data ingest callbacks
   - Real-time scene updates during rendering
   - Training visualization hooks

3. **Future ML Framework Integration**:
   - Abstract optimizer interface
   - Loss computation on GPU/CPU
   - Gradient computation and backpropagation

## Design Principles

1. **Separation of Concerns**: Training crate is independent of rendering
2. **Real-Time First**: Designed for streaming, incremental updates
3. **Extensibility**: Easy to add new data sources and training strategies
4. **Performance**: Efficient data structures for real-time operation
5. **Modularity**: Each module can be used independently
