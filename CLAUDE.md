# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Triad is a triangular splatting rendering application for real-time SLAM and simulation applications, focused on 4D Gaussian Splatting. The project aims to build cutting-edge rendering technology combined with end-to-end ML autonomy for robotics.

**Vision**: Bleeding-edge 4D Gaussian Splatting renderer + Full autonomy stack (RGB-only → Pose/Depth/Reconstruction → Motion Planning)

## Build Commands

### Basic Commands
```bash
# Build all crates
cargo build

# Build with release optimizations
cargo build --release

# Run the main application
cargo run --bin triad

# Run with a specific PLY file and mode
cargo run --bin triad -- --ply path/to/file.ply --mode gaussians

# Run tests
cargo test

# Run tests for a specific crate
cargo test -p triad-gpu
cargo test -p triad-data
cargo test -p triad-window
cargo test -p triad-train

# Check code without building
cargo check
```

### Tracy Profiling
The project has optional Tracy integration for performance profiling:

```bash
# Build with Tracy support
cargo build --features tracy

# Run with Tracy enabled
cargo run --bin triad --features tracy
```

Tracy is configured via the `LoggingConfig` in `triad-app/src/app.rs` and is conditional on the `tracy` feature flag.

### Application CLI Options
```bash
# Launch with specific window size
cargo run --bin triad -- --width 1920 --height 1080

# Available modes: points, gaussians, triangles
cargo run --bin triad -- --mode triangles
```

## Crate Architecture

The workspace is organized into specialized crates with clear separation of concerns:

### Core Crates

**`triad-gpu`** - Low-level GPU rendering infrastructure
- Explicit GPU abstraction over wgpu with frame graph system
- Builder pattern APIs: `BufferBuilder`, `BindGroupBuilder`, `RenderPipelineBuilder`
- Handle-based resource management via `ResourceRegistry`
- Frame graph for explicit render pass ordering and dependencies
- Key types: `Renderer`, `FrameGraph`, `ExecutableFrameGraph`, `PassBuilder`
- Located: `triad-gpu/src/`

**`triad-data`** - GPU-agnostic data loading and geometric processing
- PLY file loading and parsing
- Point cloud triangulation (Delaunay triangulation)
- Core geometric types: `Point`, `Gaussian`, `Triangle`
- No GPU dependencies - pure data processing
- Located: `triad-data/src/`

**`triad-window`** - Window management and application framework
- Built on winit and egui
- Camera system with orbit, pan, and zoom controls
- Input handling abstraction (`Controls`, `MouseController`, `CameraController`)
- `RendererManager` trait for rendering integration
- Located: `triad-window/src/`

**`triad-train`** - Training infrastructure for 4D Gaussian Splatting
- Data ingest pipelines (cameras, point clouds, PLY streams)
- 4D Gaussian scene representation (`Gaussian4D`, `SceneGraph`)
- Temporal management for time-varying scenes
- Reconstruction algorithms (initializer, updater)
- Note: ML training module not yet implemented (see roadmap Phase 4)
- Located: `triad-train/src/`

**`triad-app`** - Main application binary
- Integrates all crates into the viewer application
- Multi-layer renderer supporting Points, Gaussians, and Triangles
- Menu system for file loading and camera connection
- Uses frame graph to compose multiple rendering layers
- Located: `triad-app/src/`

**`triad-capture`** - Camera capture (webcam support)
- Real-time camera input for reconstruction
- Located: `triad-capture/src/`

## Architecture Patterns

### Frame Graph System
The rendering pipeline uses an explicit frame graph for composable, order-independent pass definition:

1. **Create passes** using `FrameGraph::add_compute_pass()` or `add_render_pass()`
2. **Declare dependencies** via `.depends_on()` to establish execution order
3. **Specify resources** with `.read_buffer()`, `.write_buffer()`, `.render_to()`
4. **Compile** the graph into an `ExecutableFrameGraph`
5. **Execute** each frame with cached topological sort

Key files:
- `triad-gpu/src/frame_graph/mod.rs` - Core frame graph implementation
- `triad-gpu/src/frame_graph/execution.rs` - Execution engine
- `triad-app/src/renderer_manager/frame_graph_builder.rs` - Multi-layer composition

### Resource Management
Resources are managed via handles and a central registry:

- `Handle<T>` - Type-safe resource handle (e.g., `Handle<wgpu::Buffer>`)
- `ResourceRegistry` - Central storage for GPU resources
- Handles are passed to the frame graph and dereferenced during execution
- Resources must be registered before use in passes

### Multi-Layer Rendering
The app supports three rendering modes, composited via blend passes:

1. **Points Layer** - Direct point cloud rendering
2. **Gaussians Layer** - 3D Gaussian splatting with compute shader sorting
3. **Triangles Layer** - Delaunay triangulation of point clouds

Each layer has dedicated resources (`LayerResources`) and can be toggled/opacity-controlled independently. The frame graph blends active layers into the final output.

Key file: `triad-app/src/renderer_manager/mod.rs`

### Builder Pattern APIs
All GPU resource creation uses builder patterns for ergonomic, explicit control:

```rust
// Buffer creation
let buffer = renderer.create_buffer()
    .label("my_buffer")
    .size(1024)
    .usage(BufferUsage::Uniform | BufferUsage::CopyDst)
    .build(&mut registry)?;

// Bind group creation
let bind_group = renderer.create_bind_group(&registry)
    .bind_buffer(0, buffer_handle, ShaderStage::Vertex)
    .build(&mut registry)?;
```

## Performance Profiling and Monitoring

### Logging Configuration
The project uses `tracing` for structured logging across all crates:

- **Setup location**: `triad-app/src/app.rs` in `init_logging()`
- **Default level**: `info` (configurable via `LoggingConfig`)
- **Environment variable**: `RUST_LOG` can override the log level

### Tracy Profiling
Tracy integration is optional and feature-gated for deep performance analysis:

- **Feature flag**: `tracy` (defined in `triad-gpu/Cargo.toml` and `triad-window/Cargo.toml`)
- **Dependency**: `tracing-tracy = "0.11"`
- **Enable at runtime**: Set `LoggingConfig.enable_tracy = true` (currently hardcoded in `main.rs:102`)
- **Conditional compilation**: `#[cfg(feature = "tracy")]` guards Tracy-specific code
- **Build with Tracy**: `cargo build --features tracy`
- **Run with Tracy**: `cargo run --bin triad --features tracy`

**Tracy Instrumentation Coverage:**
- Frame graph execution (`frame_graph_execute` span)
- Individual pass execution (`pass` span with index)
- Queue submission (`queue_submit` span)
- Buffer creation and writes (`create_buffer`, `write_buffer` spans)
- All render passes (via `info_span!` in layer factories and passes)
- PLY loading pipeline (`#[tracing::instrument]` on load functions)

**Adding Tracy Instrumentation:**
```rust
// Function-level instrumentation
#[tracing::instrument(skip(renderer, registry))]
fn my_function(renderer: &Renderer, registry: &mut ResourceRegistry) {
    // Tracy captures entire function execution
}

// Manual spans for specific regions
let _span = tracing::info_span!("expensive_operation").entered();
// ... code to profile ...
```

### FPS and Performance Metrics
Real-time performance monitoring is available via the in-app UI:

- **Location**: Performance window in egui UI
- **Frame Stats**:
  - Current FPS (rolling average over 120 frames)
  - Average frame time in milliseconds
  - Min/Max frame times
  - Total frame count
- **Implementation**: `PerformanceMetrics` struct in `triad-window/src/app.rs`
- **Sampling**: Last 120 frames retained for statistics

### GPU Memory Tracking
GPU memory usage is tracked via the ResourceRegistry:

- **Methods**:
  - `total_buffer_memory()` - Sum of all buffer sizes in bytes
  - `total_texture_memory()` - Approximate texture memory (size × format × mips)
  - `buffer_count()` - Number of registered buffers
  - `texture_count()` - Number of registered textures
- **Display**: Performance window shows memory usage in MB
- **Location**: `triad-gpu/src/resource_registry.rs`

### wgpu-profiler Integration
The project includes wgpu-profiler for detailed GPU timing analysis:

- **Dependency**: `wgpu-profiler = "0.19"` in `triad-gpu` and `triad-window`
- **Feature flag**: `profiling` (optional, distinct from Tracy)
- **Usage**: Can be integrated into frame graph passes to measure GPU execution time
- **Build with profiling**: `cargo build --features profiling`

**Note**: wgpu-profiler provides GPU-side timing (actual shader execution time), complementing Tracy's CPU-side profiling.

### Performance Monitoring Best Practices

1. **Use Tracy for CPU profiling**: Capture frame graph build/execute, resource creation, and CPU-side bottlenecks
2. **Use Performance UI for runtime monitoring**: Track FPS and memory during development
3. **Use wgpu-profiler for GPU profiling**: Measure shader execution time and GPU workload
4. **Log memory growth**: Monitor buffer/texture counts to detect leaks
5. **Profile with realistic data**: Use large PLY files to stress-test rendering pipeline

### Profiling Hot Paths
The following areas have comprehensive instrumentation:

- **Frame loop**: `frame` span captures entire render cycle
- **Frame graph execution**: `frame_graph_execute` with per-pass spans
- **Render passes**: All passes in `triad-app/src/renderer_manager/passes.rs` have spans
- **PLY loading**: `load_vertices_from_ply` and triangulation have instrumentation
- **Resource creation**: Buffer/bind group creation in layer factories
- **Queue operations**: Submit and buffer writes are tracked

## Rendering Implementation Status

### Current State (Per Roadmap)
- **Phase 1**: GPU API stabilization - IN PROGRESS
  - Frame graph system is functional
  - Builder APIs are established but not finalized
  - Multi-layer rendering works (Points, Gaussians, Triangles)

- **Phase 2**: 3D Gaussian rendering - NEEDS MAJOR FIXES
  - **Critical issue**: Current Gaussian rendering uses circular falloff, NOT proper 2D Gaussian projection
  - See ROADMAP.md Phase 2.1 for required math corrections:
    - Implement proper 2D covariance projection from 3D covariance
    - Fix fragment shader to evaluate proper Gaussian equation
    - Verify quaternion rotation math
  - GPU sorting is NOT implemented (renders without depth sorting)
  - Roadmap Phase 2.2 calls for radix sort implementation

- **Phase 3**: 4D Gaussian Splatting - NOT STARTED
  - `triad-train` has placeholder structures but needs complete rewrite
  - See ROADMAP.md Phase 3 for architecture decisions

### Shader Locations
- **Gaussian shaders**: `triad-gpu/shaders/gaussian_*.wgsl`
- **Triangle shaders**: `triad-gpu/shaders/triangle_*.wgsl`
- **Point shaders**: Inline in `triad-app/src/renderer_manager/layer_factory.rs`

## Key Implementation Notes

### Gaussian Rendering MUST BE FIXED
The current Gaussian implementation is mathematically incorrect and a placeholder. Before productionizing:
1. Study the original 3DGS paper for proper covariance projection
2. Implement vertex shader 3D → 2D covariance transformation
3. Rewrite fragment shader with correct 2D Gaussian evaluation
4. Add GPU-based depth sorting (radix sort or alternative)

Reference: ROADMAP.md Phase 2.1 - "Fix Core Gaussian Math"

### Frame Graph Caching
The frame graph uses execution order caching to avoid redundant topological sorts:
- Cache key is a bitmask of enabled layers
- Invalidated when layers change
- Located in `triad-app/src/renderer_manager/frame_graph_builder.rs`

### PLY File Loading
PLY loading supports both immediate and runtime loading:
- Initial load via `--ply` CLI argument
- Runtime load via File → Import PLY menu
- Channel-based communication from menu to renderer delegate
- Supports both vertex-only and triangle-face PLY files

## Development Workflow

### Adding a New Rendering Pass
1. Create resources via `renderer.create_buffer()` builders
2. Define pass in frame graph using `FrameGraph::add_compute_pass()` or `add_render_pass()`
3. Specify resource dependencies (`.read_buffer()`, `.write_buffer()`)
4. Add tracing instrumentation for performance profiling
5. Update frame graph builder to integrate the new pass
6. Ensure proper resource lifecycle management

### Adding Tracing to Code
```rust
// Function-level instrumentation
#[tracing::instrument(skip(renderer, registry))]
fn my_function(renderer: &Renderer, registry: &mut ResourceRegistry) {
    // Tracy will capture this function's span
}

// Manual spans for specific regions
use tracing::info_span;
let _span = info_span!("expensive_operation").entered();
// ... expensive code ...
```

### Working with the Roadmap
The ROADMAP.md file is the authoritative development plan. Before implementing features:
1. Check the current phase and task status
2. Review research links for papers and techniques
3. Follow the dependency graph (some phases can run in parallel)
4. Update checkboxes as tasks complete
5. Reference specific roadmap sections in commit messages

## Common Pitfalls

1. **Don't assume Gaussian rendering is correct** - It's a known placeholder needing fundamental fixes
2. **Resource handles must be registered** - Passing unregistered handles to frame graph will cause panics
3. **Frame graph dependencies are required** - Omitting `.depends_on()` can cause incorrect execution order
4. **Tracy requires feature flag** - Don't use `tracing-tracy` APIs without `#[cfg(feature = "tracy")]`
5. **Logging != stdout** - Use `tracing` macros, not `println!` for production logging
6. **Shader compilation errors are runtime** - Test shader changes thoroughly

## Research and References

### Key Papers (See ROADMAP.md Phase 0.1 for full list)
- **3DGS**: "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (arXiv:2308.04079)
- **4DGS**: "4D Gaussian Splatting for Real-Time Dynamic Scene Rendering"
- **Triangle Splatting+**: arXiv:2509.25122

### Future Directions
The project roadmap extends beyond rendering into ML autonomy:
- Phase 4: RGB-only pose & depth estimation (DUSt3R, MASt3R, DepthAnything)
- Phase 5: Gaussian reconstruction from RGB
- Phase 6: Synthetic data generation for self-supervised learning
- Phase 7: Motion planning and robotics integration

When making architectural decisions, consider the end-goal of real-time robotics autonomy.
