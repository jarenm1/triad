# TRIAD Development Roadmap

**Vision**: Cutting-edge 4D Gaussian Splatting renderer + End-to-end ML autonomy for robotics

**Last Updated**: 2026-01-03

---

## Executive Summary

### Project Goals

1. **Stable, explicit GPU API** - Clean abstraction over wgpu for Gaussian/primitive rendering (game/sim ready)
2. **Bleeding-edge rendering** - 4DGS + latest research, exceeding 3DGS quality
3. **Full autonomy stack** - RGB-only → Pose/Depth/Reconstruction → Motion Planning
4. **Modular ML architecture** - Swappable models for different robot tasks
5. **Synthetic data pipeline** - Self-supervised learning from rendered data

### Philosophy

- Real-time performance (robotics-grade)
- Explicit control over complexity (no magic abstractions)
- Research-driven (follow latest arXiv developments)

### Development Phases

```
Phase 0: Research & Architecture
Phase 1: GPU API Stabilization
Phase 2: 3D Gaussian Rendering
Phase 3: 4D Gaussian Splatting
Phase 4: ML - Pose & Depth
Phase 5: Gaussian Reconstruction
Phase 6: Synthetic Data Generation
Phase 7: Motion Planning & Autonomy
```

---

## Phase 0: Research & Architecture

**Goal**: Survey cutting-edge techniques and finalize technical architecture

### 0.1 Literature Review

#### 4D Gaussian Splatting Research
- [ ] **4D Gaussian Splatting for Real-Time Dynamic Scene Rendering** (original 4DGS paper)
- [ ] **SC-GS: Sparse-Controlled Gaussian Splatting** (efficiency improvements)
- [ ] **Deformable 3D Gaussians** (neural deformation networks)
- [ ] **Real-time Photorealistic Dynamic Scene Representation** (temporal consistency)
- [ ] **Spacetime Gaussian Feature Splatting** (latest as of 2024)

#### Advanced Rendering Techniques
- [ ] **2D Gaussian Splatting** (2DGS) - often better quality/speed than 3DGS
- [ ] **Mip-Splatting** - proper anti-aliasing for splats
- [ ] **Scaffold-GS** - hierarchical anchors for massive scenes
- [ ] **AbsGS** - absolute pose regression for unbounded scenes
- [ ] **DreamGaussian** - generative models with Gaussians

#### SLAM + Gaussian Splatting
- [ ] **GS-SLAM** - real-time monocular SLAM with Gaussians
- [ ] **Photo-SLAM** - photometric bundle adjustment with Gaussians
- [ ] **SplaTAM** - dense RGB-only SLAM
- [ ] **Gaussian-SLAM** (multiple papers, find latest)

#### RGB-only Depth/Pose Estimation
- [ ] **DUSt3R** - Dense Uncalibrated Stereoscopic Matching (SOTA 2024)
- [ ] **MASt3R** - Multi-view Aggregation for 3D Reconstruction
- [ ] **MonoGS** - Monocular Gaussian SLAM
- [ ] **DepthAnything V2** - monocular depth estimation
- [ ] **UniDepth** - universal monocular depth

**Deliverable**: Research summary document (`docs/research-summary.md`) with:
- Top 3-5 techniques to implement
- Technical feasibility assessment
- Integration strategy for each technique
- Prioritized implementation order

### 0.2 API Architecture Design

**Design the triad-gpu public API**

Key decisions to document:
- [ ] **Resource ownership model** - who owns buffers/textures/pipelines?
- [ ] **Frame graph exposure** - how much do we expose vs hide?
- [ ] **Builder vs declarative** - balance ergonomics vs explicitness
- [ ] **Error handling strategy** - Results vs panics, error types
- [ ] **Async/sync model** - blocking render() or async?
- [ ] **Multi-surface support** - multiple windows, headless rendering

**Example API sketch to validate**:
```rust
// Explicit control, but ergonomic
let mut renderer = Renderer::new()?;

// Explicit resource creation
let gaussian_buffer = renderer.create_buffer()
    .with_usage(BufferUsage::STORAGE | BufferUsage::COPY_DST)
    .with_data(&gaussians)
    .build();

// Explicit frame graph construction
let mut graph = FrameGraph::new();
let sort_pass = graph.add_compute_pass("gaussian_sort")
    .read_buffer(&gaussian_buffer)
    .write_buffer(&sort_buffer)
    .build();
    
let render_pass = graph.add_render_pass("gaussian_render")
    .depends_on(sort_pass)
    .read_buffer(&sorted_indices)
    .render_to(&surface)
    .build();

let executable = graph.compile()?;

// Render loop - explicit execute
loop {
    executable.execute(&mut renderer)?;
}
```

**Deliverable**: API design document (`docs/api-design.md`) with:
- Complete API surface specification
- Usage examples for common scenarios
- Performance characteristics (zero-cost abstractions)
- Breaking vs non-breaking change policy

### 0.3 ML Architecture Planning

**Design the modular ML pipeline**

Components to architect:
- [ ] **Pose estimation module** - RGB → camera pose
- [ ] **Depth estimation module** - RGB → dense depth map
- [ ] **Gaussian reconstruction module** - RGB+Pose+Depth → Gaussians
- [ ] **Scene understanding module** - semantic/instance segmentation (optional)
- [ ] **Motion planning module** - Scene → trajectory
- [ ] **Model interfaces** - how modules communicate

**Key architectural questions**:
- How do modules share intermediate representations?
- Which modules can run in parallel?
- Where do we cache/store temporal information?
- How do we handle model updates (online learning)?
- GPU vs CPU for inference?

**Deliverable**: ML architecture diagram (`docs/ml-architecture.md`) showing:
- Data flow through the pipeline
- Module interfaces and APIs
- Shared state management
- Performance budget per module

---

## Phase 1: GPU API Stabilization

**Goal**: Production-ready triad-gpu API with comprehensive documentation

### 1.1 Core API Implementation

**Public API surface**:
- [ ] Finalize `Renderer` public interface
- [ ] Stabilize `FrameGraph` and `PassBuilder` APIs
- [ ] Lock down `Handle<T>` and `ResourceRegistry` types
- [ ] Define `BufferBuilder`, `BindGroupBuilder` error types
- [ ] Document resource lifetime semantics

**Resource management improvements**:
- [ ] Add resource deallocation/cleanup APIs
- [ ] Implement reference counting for shared resources
- [ ] Add resource usage tracking (debug mode)
- [ ] Memory pool allocator for buffers (optional)

**Builder pattern refinements**:
- [ ] Consistent error messages across all builders
- [ ] Validation in builders (catch errors early)
- [ ] Sensible defaults for common use cases
- [ ] Type-state pattern for compile-time correctness

### 1.2 Documentation & Examples

**Documentation**:
- [ ] Crate-level README with quick start
- [ ] API reference docs for all public types
- [ ] Architecture guide (how frame graph works)
- [ ] Performance guide (best practices)
- [ ] Migration guide (if breaking changes)

**Examples**:
- [ ] `basic_rendering.rs` - minimal example
- [ ] `frame_graph.rs` - explicit frame graph construction
- [ ] `multi_layer.rs` - compositing multiple layers
- [ ] `compute_pipeline.rs` - compute shader usage
- [ ] `resource_management.rs` - buffer updates, cleanup
- [ ] `headless.rs` - offscreen rendering

### 1.3 Testing & Validation

**Test suite**:
- [ ] Unit tests for builders and handle system
- [ ] Integration tests for frame graph execution
- [ ] Render tests with image comparison
- [ ] Performance benchmarks (baseline metrics)
- [ ] Memory leak detection tests

**Validation**:
- [ ] API review (get external feedback if possible)
- [ ] Benchmark against raw wgpu (overhead measurement)
- [ ] Stress test (large scenes, many resources)

**Success Criteria**:
- ✅ Can render 1M+ Gaussians at 60fps (RTX 3070S baseline)
- ✅ API is documented with 5+ examples
- ✅ Zero-cost abstractions (< 5% overhead vs raw wgpu)

**Deliverable**: triad-gpu 0.1.0 release with stable API

---

## Phase 2: Correct 3D Gaussian Rendering

**Goal**: Mathematically correct, production-quality 3DGS rendering

### 2.1 Fix Core Gaussian Math

**Current issues to fix**:
- [ ] **Implement proper 2D covariance projection**
  - Project 3D covariance matrix to screen space
  - Compute 2D Gaussian parameters from projected covariance
  - Pass covariance to fragment shader

- [ ] **Correct fragment shader**
  - Replace circular falloff with proper 2D Gaussian
  - Evaluate: `exp(-0.5 * (pos - mean)^T * Σ^-1 * (pos - mean))`
  - Proper normalization constants

- [ ] **Verify quaternion math**
  - Check rotation matrix conversion
  - Validate against reference implementation

### 2.2 Implement GPU Sorting

**Radix sort implementation**:
- [ ] Research GPU radix sort algorithms (read papers)
- [ ] Implement multi-pass radix sort compute shader
- [ ] Optimize for workgroup size and memory access
- [ ] Handle variable-length arrays

**Integration**:
- [ ] Use sorted indices for rendering
- [ ] Implement indirect draw or reorder index buffer
- [ ] Add option to disable sorting (for debugging)
- [ ] Benchmark sorting overhead

**Alternative approaches to evaluate**:
- [ ] Bitonic sort (simpler but O(n log² n))
- [ ] Bucket sort (if depth quantization acceptable)
- [ ] Hybrid CPU/GPU sort

### 2.3 Rendering Optimizations

**Performance improvements**:
- [ ] Frustum culling compute pass
- [ ] Screen-space size culling (skip tiny Gaussians)
- [ ] Tile-based rendering (optional, research first)
- [ ] Packed data formats (fp16 where possible)

**Quality improvements**:
- [ ] Proper depth testing strategy
- [ ] Anti-aliasing (explore Mip-Splatting)
- [ ] HDR rendering support
- [ ] Tonemapping pass

### 2.4 Validation & Benchmarking

**Validation**:
- [ ] Compare against reference 3DGS implementation
- [ ] Visual quality comparison (same dataset)
- [ ] PSNR/SSIM metrics on standard test scenes
- [ ] Ensure back-to-front rendering works correctly

**Benchmarking**:
- [ ] FPS vs number of Gaussians
- [ ] Memory usage profiling
- [ ] Sort performance analysis
- [ ] Comparison with other renderers (if available)

**Success Criteria**:
- ✅ Visual quality matches original 3DGS paper
- ✅ PSNR within 1dB of reference on standard scenes
- ✅ Correct back-to-front rendering with GPU sorting

**Deliverable**: Production-quality 3DGS renderer matching or exceeding original paper

---

## Phase 3: 4D Gaussian Splatting

**Goal**: Real-time dynamic scene reconstruction with temporal coherence

### 3.1 Research & Architecture

**Research latest 4DGS techniques**:
- [ ] Compare different 4DGS formulations
- [ ] Analyze temporal representation strategies
- [ ] Study deformation network approaches
- [ ] Investigate neural vs explicit temporal models

**Design decisions**:
- [ ] Explicit keyframe interpolation vs neural deformation?
- [ ] Temporal consistency loss functions?
- [ ] Memory management for temporal data?
- [ ] Streaming vs in-memory for long sequences?

### 3.2 Rewrite 4D Data Structures

**Current code assessment**:
- [ ] Review existing `Gaussian4D` implementation (triad-train/src/reconstruction/gaussian4d.rs)
- [ ] Identify what to keep vs rewrite
- [ ] Design new temporal representation

**New architecture**:
- [ ] `Gaussian4D` - compact temporal Gaussian representation
- [ ] `TemporalKeyframe` - efficient keyframe storage
- [ ] `SceneGraph` - optimized temporal queries
- [ ] Interpolation strategies (linear, Bezier, neural?)

**GPU data structures**:
- [ ] Temporal buffer layout (SoA vs AoS)
- [ ] Efficient GPU interpolation
- [ ] Time-based culling

### 3.3 Temporal Rendering

**Shaders**:
- [ ] Temporal interpolation compute shader
- [ ] Dynamic Gaussian vertex shader (motion blur?)
- [ ] Temporal anti-aliasing

**Rendering pipeline**:
- [ ] Frame graph integration for 4D rendering
- [ ] Temporal consistency handling
- [ ] Motion vector generation (for temporal effects)

**Features**:
- [ ] Scrubbing through time (playback controls)
- [ ] Real-time reconstruction updates
- [ ] Temporal compression (reduce keyframes)

### 3.4 Temporal Reconstruction

**Incremental scene updates**:
- [ ] Rewrite `SceneUpdater` for efficiency (triad-train/src/reconstruction/updater.rs)
- [ ] Smart keyframe placement strategy
- [ ] Gaussian lifecycle management (birth/death)
- [ ] Temporal regularization

**Optimization**:
- [ ] Adaptive timestep selection
- [ ] Prune redundant Gaussians
- [ ] Compress temporal data

**Success Criteria**:
- ✅ Real-time dynamic scene playback (30fps+)
- ✅ Temporal consistency (no flickering)
- ✅ Incremental reconstruction works

**Deliverable**: Working 4DGS system with real-time dynamic reconstruction

---

## Phase 4: ML Foundation - Monocular Pose & Depth

**Goal**: RGB-only camera pose and depth estimation for reconstruction

### 4.1 ML Framework Integration

**Choose and integrate ML framework**

**Option A: PyTorch (tch-rs)**
- ✅ Access to pretrained models
- ✅ Huge ecosystem
- ✅ Easy prototyping
- ❌ C++ libtorch dependency
- ❌ Harder to deploy

**Option B: Candle**
- ✅ Pure Rust
- ✅ Good GPU support (CUDA, Metal)
- ✅ Growing ecosystem
- ❌ Fewer pretrained models
- ❌ Newer, less mature

**Option C: Burn**
- ✅ Pure Rust, very flexible
- ✅ Backend agnostic
- ❌ Fewer pretrained models
- ❌ More manual implementation

**Decision criteria**:
- [ ] Benchmark inference performance
- [ ] Evaluate model availability
- [ ] Test deployment complexity
- [ ] Assess long-term maintenance

**Implementation**:
- [ ] Integrate chosen framework
- [ ] Create `triad-ml` crate
- [ ] Setup model loading/inference APIs
- [ ] GPU memory management with rendering

### 4.2 Monocular Depth Estimation

**Model selection**:
- [ ] Evaluate DepthAnything V2
- [ ] Evaluate UniDepth
- [ ] Evaluate MiDaS v3.1
- [ ] Compare metric vs relative depth

**Integration**:
- [ ] Load pretrained model
- [ ] Implement inference pipeline
- [ ] Optimize for real-time (TensorRT, quantization, etc.)
- [ ] Handle resolution scaling

**Validation**:
- [ ] Test on standard datasets (NYUv2, KITTI)
- [ ] Measure inference time
- [ ] Assess depth quality for reconstruction

### 4.3 Visual Odometry / SLAM

**Implementation w/ Gaussian SLAM**:
- [ ] Camera pose tracking
- [ ] Temporal pose smoothing
- [ ] Relocalization support
- [ ] Map management

### 4.4 End-to-End Testing

**Pipeline integration**:
- [ ] RGB → Depth → Pose → Point Cloud
- [ ] Validate on existing datasets (TUM RGB-D, etc.)
- [ ] Measure end-to-end latency
- [ ] Assess reconstruction quality

**Benchmarks**:
- [ ] Pose estimation accuracy (ATE, RPE)
- [ ] Depth estimation error (AbsRel, RMSE)
- [ ] FPS on target hardware
- [ ] Memory consumption

**Success Criteria**:
- ✅ Depth estimation runs at 30fps+ (720p)
- ✅ Pose tracking works on TUM RGB-D dataset
- ✅ End-to-end latency < 100ms

**Deliverable**: Working RGB-only pose & depth estimation pipeline

---

## Phase 5: Gaussian Reconstruction from RGB

**Goal**: End-to-end RGB → 4D Gaussians reconstruction

### 5.1 Point Cloud to Gaussian Initialization

**Improve existing initializer**:
- [ ] Rewrite `GaussianInitializer` with better strategies (triad-train/src/reconstruction/initializer.rs)
- [ ] Implement adaptive initialization (clustering)
- [ ] Proper scale estimation from point cloud density
- [ ] Orientation estimation from local surface normals

**Neural initialization (optional)**:
- [ ] Train small network for Gaussian parameter prediction
- [ ] Learn optimal scale/rotation from local geometry
- [ ] Regularization for stable Gaussians

### 5.2 Gaussian Optimization

**Photometric loss**:
- [ ] Implement differentiable Gaussian rendering
- [ ] L1 + SSIM loss (standard 3DGS loss)
- [ ] Perceptual loss (VGG-based, optional)

**Regularization**:
- [ ] Gaussian scale regularization
- [ ] Opacity regularization (prevent opacity collapse)
- [ ] Spatial smoothness (for temporal consistency)
- [ ] Depth supervision (if depth available)

**Densification & pruning**:
- [ ] Adaptive Gaussian splitting (high-gradient regions)
- [ ] Gaussian cloning (insufficient coverage)
- [ ] Pruning (low opacity, large Gaussians)
- [ ] Periodic re-optimization

### 5.3 Temporal Consistency

**4D reconstruction**:
- [ ] Integrate pose + depth + optimization over time
- [ ] Keyframe selection strategy
- [ ] Temporal interpolation constraints
- [ ] Loop closure for 4D (if using SLAM)

**Online reconstruction**:
- [ ] Incremental Gaussian updates
- [ ] Real-time optimization budget
- [ ] Sliding window optimization
- [ ] Efficient GPU/CPU workload distribution

### 5.4 Validation

**Reconstruction quality**:
- [ ] PSNR/SSIM on held-out views
- [ ] Temporal consistency metrics
- [ ] Geometric accuracy (if ground truth available)

**Performance**:
- [ ] Reconstruction speed (seconds per frame)
- [ ] Memory scalability (scene size limits)
- [ ] Real-time performance demonstration

**Success Criteria**:
- ✅ RGB → Gaussians produces recognizable scene
- ✅ PSNR > 25dB on novel views
- ✅ Works on real-world captured data

**Deliverable**: Full RGB → 4D Gaussian reconstruction pipeline

---

## Phase 6: Synthetic Data Generation

**Goal**: Self-supervised learning via synthetic data from renderer

### 6.1 Procedural Scene Generation

**Scene synthesis**:
- [ ] Procedural geometry generators (rooms, corridors, outdoor)
- [ ] Texture synthesis or asset library
- [ ] Lighting randomization
- [ ] Camera trajectory generation

**Gaussian scene creation**:
- [ ] Convert meshes → Gaussians
- [ ] Noise injection for realism
- [ ] Material property variation

### 6.2 Rendering Pipeline

**Synthetic data rendering**:
- [ ] RGB rendering (primary)
- [ ] Depth ground truth
- [ ] Pose ground truth
- [ ] Semantic segmentation masks (optional)
- [ ] Instance IDs (optional)

**Augmentation**:
- [ ] Camera parameters randomization
- [ ] Lighting variations
- [ ] Weather effects (if outdoor)
- [ ] Sensor noise simulation

### 6.3 Data Pipeline

**Dataset generation**:
- [ ] Batch rendering infrastructure
- [ ] Dataset format specification (HDF5, TFRecord, etc.)
- [ ] Metadata management (poses, intrinsics)
- [ ] Storage and streaming

**Integration with training**:
- [ ] Data loader for ML models
- [ ] Online data generation during training
- [ ] Curriculum learning (simple → complex scenes)

### 6.4 Validation

**Sim-to-real transfer**:
- [ ] Test models trained on synthetic data on real data
- [ ] Domain randomization strategies
- [ ] Identify and fix domain gap issues

**Deliverable**: Synthetic data generation pipeline for self-supervised learning

---

## Phase 7: Motion Planning & Autonomy

**Goal**: Close the loop - perception to action

### 7.1 Motion Planning Module

**Architecture**:
- [ ] Define motion planning interface
- [ ] Choose planning algorithm (RRT, MPC, learning-based)
- [ ] Collision avoidance using Gaussian scene

**Approaches to explore**

**Option A: Classical planning (RRT*, MPC)**
- [ ] Use Gaussian scene for collision checking
- [ ] Path optimization
- [ ] Real-time replanning

**Option B: Learning-based (IL, RL)**
- [ ] Imitation learning from expert demonstrations
- [ ] Reinforcement learning in Gaussian sim
- [ ] Sim-to-real transfer

**Option C: Hybrid**
- [ ] Learned cost maps + classical planner
- [ ] Neural guidance for sampling-based planning

### 7.2 Robot Integration

**Decide on robot platform**:
- [ ] Simulation only (Mujoco, Isaac Gym)?
- [ ] Real hardware (arm, mobile robot)?
- [ ] Both (sim-to-real workflow)?

**Integration tasks**:
- [ ] Robot state representation
- [ ] Action space definition
- [ ] Sensor integration (camera mounting)
- [ ] Control loop timing

### 7.3 End-to-End System

**Full pipeline**:
- [ ] RGB camera → Pose/Depth → Gaussians → Plan → Action
- [ ] Real-time loop closure
- [ ] Error recovery mechanisms

**Demonstrations**:
- [ ] Navigation task (reach goal, avoid obstacles)
- [ ] Manipulation task (reach, grasp)
- [ ] Combined task (fetch object)

### 7.4 Evaluation

**Performance metrics**:
- [ ] Task success rate
- [ ] Time to completion
- [ ] Safety (collision rate)
- [ ] Generalization (new scenes)

**Success Criteria**:
- ✅ Robot completes task in simulation
- ✅ 80%+ success rate on task
- ✅ Safe operation (no collisions)

**Deliverable**: Working autonomous robot demonstrating perception + planning

---

**Questions or suggestions?** Open an issue or discussion on GitHub.

