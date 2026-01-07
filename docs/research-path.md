# Gaussian Splatting Research Path

**Last Updated**: 2026-01-07

This guide provides a structured path through the Gaussian Splatting literature, from foundational concepts to cutting-edge research. Papers are organized by difficulty and dependencies to optimize learning.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Math Requirements](#math-requirements)
3. [Reading Order](#reading-order)
4. [Paper Index](#paper-index)
5. [Additional Resources](#additional-resources)

---

## Prerequisites

### Required Background Knowledge

Before diving into Gaussian Splatting papers, you should be familiar with:

#### 1. **Computer Graphics Fundamentals**
- Rendering pipeline basics
- Rasterization vs ray tracing
- Camera models (pinhole, perspective projection)
- Homogeneous coordinates and transformations
- Texture mapping and interpolation

**Resources**:
- *Real-Time Rendering* by Akenine-Möller et al. (Chapters 1-4)
- *Fundamentals of Computer Graphics* by Marschner & Shirley

#### 2. **3D Computer Vision**
- Structure from Motion (SfM)
- Multi-view geometry
- Camera calibration (intrinsic/extrinsic parameters)
- Epipolar geometry
- Bundle adjustment
- Point cloud processing

**Resources**:
- *Multiple View Geometry in Computer Vision* by Hartley & Zisserman (Chapters 1-6, 9-10)
- *Computer Vision: Algorithms and Applications* by Szeliski (Chapters 2, 7, 11)

#### 3. **Deep Learning Basics**
- Neural network fundamentals (MLPs, backpropagation)
- Convolutional Neural Networks (CNNs)
- Transformers (for recent methods like DUSt3R/MASt3R)
- Optimization (Adam, SGD, learning rate schedules)
- Loss functions (L1, L2, SSIM, perceptual losses)

**Resources**:
- *Deep Learning* by Goodfellow, Bengio, and Courville (Chapters 6-8)
- "Attention Is All You Need" (Transformers) - arXiv:1706.03762

#### 4. **Differential Rendering**
- Differentiable rendering pipelines
- Gradient computation through rendering
- Implicit vs explicit representations

**Resources**:
- Survey: "Advances in Neural Rendering" - arXiv:2111.05849

---

## Math Requirements

### Essential Mathematics

#### 1. **Linear Algebra** (Critical)
- Matrix operations (multiplication, inversion, decomposition)
- Eigenvalues and eigenvectors
- Singular Value Decomposition (SVD)
- **Covariance matrices** ⭐ (central to Gaussian Splatting)
- Rotation matrices and quaternions
- Homogeneous coordinates

#### 2. **Multivariate Calculus** (Critical)
- Gradients and Jacobians
- Chain rule (for backpropagation)
- Partial derivatives
- Taylor series expansion

#### 3. **Probability & Statistics** (Critical)
- **Gaussian (normal) distributions** ⭐ (foundation of the method)
- Multivariate Gaussians
- Covariance and correlation
- Maximum likelihood estimation
- Expectation and variance

#### 4. **3D Geometry** (Critical)
- Coordinate transformations (world, camera, screen space)
- **Quaternions for rotations** ⭐ (used in Gaussian orientation)
- Rodrigues' rotation formula
- Projective geometry
- Homogeneous coordinates

#### 5. **Optimization** (Important)
- Gradient descent variants (Adam, AdaGrad)
- Regularization (L1, L2)
- Constrained optimization
- Levenberg-Marquardt (for bundle adjustment)

#### 6. **Signal Processing** (Helpful)
- Sampling theory (Nyquist theorem)
- Aliasing and anti-aliasing
- Frequency domain analysis
- Mipmap theory (for Mip-Splatting)

### Key Mathematical Concepts in Gaussian Splatting

1. **3D Gaussian Definition**:
   ```
   G(x) = exp(-1/2 * (x - μ)ᵀ Σ⁻¹ (x - μ))
   ```
   Where:
   - μ = mean (center position)
   - Σ = 3×3 covariance matrix (shape/orientation)

2. **Covariance Matrix from Scale & Rotation**:
   ```
   Σ = R S Sᵀ Rᵀ
   ```
   Where:
   - R = rotation matrix (from quaternion)
   - S = scaling matrix (diagonal, from 3D scale vector)

3. **2D Projection** (critical for rendering):
   - Project 3D covariance to 2D screen space
   - Compute 2D Gaussian parameters for fragment shader
   - Jacobian of perspective projection

---

## Reading Order

### Phase 0: Foundation (Start Here)

**Goal**: Understand the predecessor (NeRF) and the core Gaussian Splatting method

#### 0.1 - Neural Radiance Fields Background (Optional but Recommended)

**📄 NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis**
- **arXiv**: [2003.08934](https://arxiv.org/abs/2003.08934)
- **Authors**: Mildenhall et al.
- **Published**: ECCV 2020
- **Why read**: Understand the implicit representation that 3DGS improves upon
- **Key takeaways**: Volume rendering, neural implicit representations, positional encoding
- **Estimated reading time**: 4-6 hours
- **Difficulty**: ⭐⭐⭐ Medium

**Prerequisites**: Basic neural networks, rendering equation
**Skip if**: You're already familiar with NeRF or want to dive straight into explicit methods

---

#### 0.2 - Core 3D Gaussian Splatting (MUST READ)

**📄 3D Gaussian Splatting for Real-Time Radiance Field Rendering**
- **arXiv**: [2308.04079](https://arxiv.org/abs/2308.04079)
- **Authors**: Kerbl, Kopanas, Leimkühler, Drettakis (INRIA)
- **Published**: ACM SIGGRAPH 2023 / TOG
- **Project**: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **Code**: https://github.com/graphdeco-inria/gaussian-splatting
- **Why read**: This is THE foundational paper - everything else builds on this
- **Key concepts**:
  - 3D Gaussian primitives with position, covariance, opacity, color (SH)
  - Differentiable splatting rasterization
  - Adaptive density control (split/clone/prune)
  - Fast tile-based rendering
- **Estimated reading time**: 8-12 hours (read carefully, multiple times)
- **Difficulty**: ⭐⭐⭐⭐ Hard

**Prerequisites**:
- Multivariate Gaussians
- Computer graphics rasterization
- Gradient-based optimization
- Spherical harmonics (for view-dependent color)

**Implementation notes**:
- Study the CUDA rasterizer implementation
- Understand the sorting requirement (back-to-front for alpha blending)
- Pay special attention to Section 3.2 (Optimization with Adaptive Density Control)

---

### Phase 1: Core Improvements (Quality & Anti-Aliasing)

**Goal**: Understand how to fix 3DGS artifacts and improve quality

#### 1.1 - 2D Gaussian Splatting (HIGH PRIORITY)

**📄 2D Gaussian Splatting for Geometrically Accurate Radiance Fields**
- **arXiv**: [2403.17888](https://arxiv.org/abs/2403.17888)
- **Authors**: Huang, Yu, Chen, Geiger, Gao
- **Published**: ACM SIGGRAPH 2024
- **Why read**: Roadmap marked as "often better quality/speed than 3DGS"
- **Key innovation**: Use 2D Gaussian disks instead of 3D ellipsoids
- **Benefits**: Better geometry reconstruction, view-consistent surfaces
- **Trade-offs**: Less flexible for volumetric effects (smoke, clouds)
- **Estimated reading time**: 6-8 hours
- **Difficulty**: ⭐⭐⭐⭐ Hard (builds on 3DGS)

**Prerequisites**:
- Must read 3DGS paper first
- Understanding of surface representations vs volumetric

**Read after**: 3DGS foundation

---

#### 1.2 - Mip-Splatting (Anti-Aliasing)

**📄 Mip-Splatting: Alias-free 3D Gaussian Splatting**
- **arXiv**: [2311.16493](https://arxiv.org/abs/2311.16493)
- **Authors**: Yu, Chen, Huang, Sattler, Geiger
- **Published**: CVPR 2024 (🏆 Best Student Paper)
- **Project**: https://niujinshuchong.github.io/mip-splatting/
- **Code**: https://github.com/autonomousvision/mip-splatting
- **Why read**: Fixes aliasing artifacts when zooming/changing focal length
- **Key innovations**:
  - 3D smoothing filter (constrains Gaussian size based on sampling frequency)
  - 2D Mip filter (replaces 2D dilation, simulates box filter)
- **Estimated reading time**: 4-6 hours
- **Difficulty**: ⭐⭐⭐ Medium-Hard

**Prerequisites**:
- 3DGS paper
- Signal processing (Nyquist theorem, aliasing)
- Mipmap theory

**Read after**: 3DGS foundation

---

### Phase 2: Dynamic Scenes & 4D Reconstruction

**Goal**: Extend Gaussian Splatting to time-varying scenes

#### 2.1 - Deformable 3D Gaussians

**📄 Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction**
- **arXiv**: [2309.13101](https://arxiv.org/abs/2309.13101)
- **Published**: 2023
- **Why read**: Foundational approach to dynamic scenes using deformation networks
- **Key concept**: Learn Gaussians in canonical space + MLP deformation field
- **Architecture**: 3D Gaussians + time-conditioned deformation MLP
- **Estimated reading time**: 5-7 hours
- **Difficulty**: ⭐⭐⭐⭐ Hard

**Prerequisites**:
- 3DGS
- Neural networks (MLPs)
- Deformation field concept

**Read after**: 3DGS

---

#### 2.2 - 4D Gaussian Splatting (Original)

**📄 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering**
- **arXiv**: [2310.08528](https://arxiv.org/abs/2310.08528)
- **Authors**: Wu et al.
- **Published**: CVPR 2024
- **Project**: https://guanjunwu.github.io/4dgs/
- **Code**: https://github.com/hustvl/4DGaussians
- **Why read**: Holistic 4D representation (not per-frame 3DGS)
- **Key innovation**: 4D neural voxels + 3D Gaussians
- **Performance**: 82 FPS @ 800×800 on RTX 3090
- **Estimated reading time**: 6-8 hours
- **Difficulty**: ⭐⭐⭐⭐ Hard

**Prerequisites**:
- 3DGS
- Deformable Gaussians
- Voxel grids

**Read after**: 3DGS, Deformable Gaussians

---

#### 2.3 - Alternative: Real-time Photorealistic 4DGS

**📄 Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting**
- **arXiv**: [2310.10642](https://arxiv.org/abs/2310.10642)
- **Authors**: Yang et al.
- **Published**: 2023
- **Key difference**: 4D Gaussian primitives (anisotropic ellipses in spacetime)
- **View-dependent appearance**: Time-evolved + view-dependent color
- **Estimated reading time**: 5-7 hours
- **Difficulty**: ⭐⭐⭐⭐ Hard

**Read after**: 4DGS (2310.08528) for comparison

---

#### 2.4 - Spacetime Gaussian Feature Splatting

**📄 Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis**
- **arXiv**: [2312.16812](https://arxiv.org/abs/2312.16812)
- **Authors**: Li, Chen, Li, Xu (OPPO Research)
- **Published**: CVPR 2024
- **Code**: https://github.com/oppo-us-research/SpacetimeGaussians
- **Why read**: Latest 4DGS variant (as of roadmap), state-of-the-art quality
- **Key innovations**:
  - Temporal opacity + parametric motion/rotation
  - Neural features replace spherical harmonics (view/time-dependent)
  - Guided sampling strategy
- **Performance**: 60 FPS @ 8K on RTX 4090
- **Estimated reading time**: 6-8 hours
- **Difficulty**: ⭐⭐⭐⭐⭐ Very Hard

**Prerequisites**:
- 3DGS
- 4DGS variants
- Neural rendering features

**Read after**: 3DGS, at least one 4DGS paper

---

### Phase 3: Structured & Efficient Representations

**Goal**: Learn advanced techniques for large-scale and editable scenes

#### 3.1 - SC-GS (Sparse-Controlled)

**📄 SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes**
- **arXiv**: [2312.14937](https://arxiv.org/abs/2312.14937)
- **Authors**: Huang, Sun, Yang et al.
- **Published**: CVPR 2024
- **Code**: https://github.com/CVMI-Lab/SC-GS
- **Why read**: Enables motion editing via sparse control points
- **Key innovation**: Decouple motion (sparse control points) from appearance (dense Gaussians)
- **Use case**: User-controlled editing while retaining quality
- **Estimated reading time**: 5-7 hours
- **Difficulty**: ⭐⭐⭐⭐ Hard

**Prerequisites**:
- 3DGS
- Deformable Gaussians
- Deformation field interpolation

**Read after**: Deformable Gaussians

---

#### 3.2 - Scaffold-GS (Hierarchical)

**📄 Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering**
- **arXiv**: [2312.00109](https://arxiv.org/abs/2312.00109)
- **Authors**: Lu, Yu, Xu et al.
- **Published**: CVPR 2024
- **Why read**: Hierarchical anchor-based representation for large scenes
- **Key innovation**: Anchor points distribute local Gaussians with learnable offsets
- **Benefits**: Structural coherence, view-adaptive rendering, compression-friendly
- **Estimated reading time**: 5-7 hours
- **Difficulty**: ⭐⭐⭐⭐ Hard

**Prerequisites**:
- 3DGS
- Hierarchical representations
- Feature-based rendering

**Read after**: 3DGS

---

### Phase 4: SLAM Integration (RGB-only Reconstruction)

**Goal**: Real-time camera tracking and mapping with Gaussian Splatting

#### 4.1 - GS-SLAM (Dense Visual SLAM)

**📄 GS-SLAM: Dense Visual SLAM with 3D Gaussian Splatting**
- **arXiv**: [2311.11700](https://arxiv.org/abs/2311.11700)
- **Authors**: Yan, Qu, Xu et al.
- **Published**: CVPR 2024
- **Project**: https://gs-slam.github.io/
- **Why read**: First Gaussian Splatting-based SLAM system
- **Key features**:
  - Real-time tracking, mapping, rendering (386 FPS average)
  - Adaptive expansion strategy (add/delete Gaussians)
  - RGB-D input
- **Estimated reading time**: 6-8 hours
- **Difficulty**: ⭐⭐⭐⭐ Hard

**Prerequisites**:
- 3DGS
- SLAM fundamentals (tracking & mapping)
- Pose optimization

**Additional background**:
- ORB-SLAM2 or similar traditional SLAM system
- Bundle adjustment basics

**Read after**: 3DGS + SLAM background

---

#### 4.2 - SplaTAM (Dense RGB-D SLAM)

**📄 SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM**
- **arXiv**: [2312.02126](https://arxiv.org/abs/2312.02126)
- **Authors**: Keetha et al.
- **Published**: CVPR 2024
- **Project**: https://spla-tam.github.io/
- **Why read**: Superior performance in pose estimation and reconstruction
- **Performance**: 400 FPS rendering, 2× better than alternatives
- **Key innovation**: Tailored tracking/mapping for Gaussian representation
- **Estimated reading time**: 6-8 hours
- **Difficulty**: ⭐⭐⭐⭐ Hard

**Prerequisites**:
- 3DGS
- SLAM fundamentals

**Read after**: GS-SLAM (for comparison)

---

#### 4.3 - MonoGS (Monocular SLAM)

**📄 Gaussian Splatting SLAM (MonoGS)**
- **arXiv**: [2312.06741](https://arxiv.org/abs/2312.06741)
- **Authors**: Matsuki, Murai, Kelly, Davison
- **Published**: CVPR 2024 (🏆 Highlight & Best Demo Award)
- **Code**: https://github.com/muskie82/MonoGS
- **Why read**: Monocular-only (no depth) Gaussian SLAM
- **Key challenge**: Depth uncertainty from monocular input
- **Features**: Handles tiny and transparent objects
- **Performance**: Runs live at 3 FPS
- **Estimated reading time**: 6-8 hours
- **Difficulty**: ⭐⭐⭐⭐⭐ Very Hard

**Prerequisites**:
- 3DGS
- Monocular depth estimation concepts
- Visual odometry

**Read after**: GS-SLAM, SplaTAM

---

### Phase 5: Monocular Depth & Pose Estimation

**Goal**: RGB-only camera pose and dense depth for reconstruction

#### 5.1 - DUSt3R (Foundation for Uncalibrated Reconstruction)

**📄 DUSt3R: Geometric 3D Vision Made Easy**
- **arXiv**: [2312.14132](https://arxiv.org/abs/2312.14132)
- **Authors**: Wang, Leroy, Cabon et al. (Naver Labs)
- **Published**: CVPR 2024
- **Project**: https://europe.naverlabs.com/research/publications/dust3r-geometric-3d-vision-made-easy/
- **Why read**: Dense Uncalibrated Stereo - no camera calibration needed
- **Key innovation**: Regress pointmaps directly (relaxes camera model constraints)
- **Use case**: 3D reconstruction from arbitrary image collections
- **Performance**: SOTA on monocular/multi-view depth + relative pose
- **Estimated reading time**: 6-8 hours
- **Difficulty**: ⭐⭐⭐⭐ Hard

**Prerequisites**:
- Multi-view geometry
- Transformers (architecture uses ViT)
- Depth estimation basics

**Read after**: 3DGS (to understand where this fits in pipeline)

---

#### 5.2 - MASt3R (Matching and Stereo)

**📄 Grounding Image Matching in 3D with MASt3R**
- **arXiv**: [2406.09756](https://arxiv.org/abs/2406.09756)
- **Authors**: Leroy, Cabon, Revaud (Naver Labs)
- **Published**: ECCV 2024
- **Why read**: Extends DUSt3R with dense local features for matching
- **Key innovation**: Casts matching as 3D task, adds matching head to DUSt3R
- **Performance**: 30% improvement over best methods on Map-free localization
- **Estimated reading time**: 5-7 hours
- **Difficulty**: ⭐⭐⭐⭐ Hard

**Prerequisites**:
- DUSt3R (directly builds on it)
- Image matching fundamentals
- Feature descriptors

**Read after**: DUSt3R

---

#### 5.3 - Depth Anything V2 (Monocular Depth Foundation Model)

**📄 Depth Anything V2**
- **arXiv**: [2406.09414](https://arxiv.org/abs/2406.09414)
- **Authors**: Yang et al.
- **Published**: NeurIPS 2024
- **Code**: https://github.com/DepthAnything/Depth-Anything-V2
- **Project**: https://depth-anything-v2.github.io/
- **Why read**: Strong monocular depth baseline, useful for initialization
- **Key features**:
  - Fast inference (10× faster than Stable Diffusion-based)
  - Multiple model sizes (25M to 1.3B params)
  - Fine and robust predictions
- **Use case**: Quick depth estimation for Gaussian initialization
- **Estimated reading time**: 3-5 hours
- **Difficulty**: ⭐⭐⭐ Medium

**Prerequisites**:
- Monocular depth estimation basics
- CNNs / Vision Transformers

**Read after**: Can read standalone or after DUSt3R

---

#### 5.4 - UniDepth (Universal Metric Depth)

**📄 UniDepth: Universal Monocular Metric Depth Estimation**
- **arXiv**: [2403.18913](https://arxiv.org/abs/2403.18913)
- **Authors**: Piccinelli, Yang, Sakaridis et al.
- **Published**: CVPR 2024
- **Code**: https://github.com/lpiccinelli-eth/UniDepth
- **Why read**: Zero-shot metric 3D reconstruction from single images
- **Key innovation**: Self-promptable camera module + pseudo-spherical output
- **Benefit**: No camera parameters needed at test time
- **Estimated reading time**: 5-7 hours
- **Difficulty**: ⭐⭐⭐⭐ Hard

**Prerequisites**:
- Camera geometry
- Monocular depth estimation

**Read after**: Depth Anything V2 (simpler baseline)

---

## Recommended Reading Paths

### Path A: Fast Track to 3DGS Implementation
**Goal**: Get a working 3DGS renderer ASAP

1. 3D Gaussian Splatting (2308.04079) - Read deeply
2. Start implementing while reading
3. Mip-Splatting (2311.16493) - Fix aliasing issues
4. 2D Gaussian Splatting (2403.17888) - Improve quality

**Time estimate**: 4-6 weeks

---

### Path B: Full Stack - 4D Reconstruction from RGB
**Goal**: Build end-to-end RGB → 4D Gaussians system

1. **Foundation**:
   - NeRF (2003.08934) - Optional background
   - 3D Gaussian Splatting (2308.04079) - Core

2. **Dynamic Scenes**:
   - Deformable 3D Gaussians (2309.13101)
   - 4D Gaussian Splatting (2310.08528)
   - Spacetime Gaussian Feature Splatting (2312.16812)

3. **RGB-only Pipeline**:
   - Depth Anything V2 (2406.09414) - Monocular depth
   - DUSt3R (2312.14132) - Uncalibrated reconstruction
   - MASt3R (2406.09756) - Matching & pose

4. **SLAM Integration**:
   - GS-SLAM (2311.11700) - Foundation
   - SplaTAM (2312.02126) - Performance comparison
   - MonoGS (2312.06741) - Monocular variant

**Time estimate**: 3-6 months

---

### Path C: Robotics-Focused (Your TRIAD Project)
**Goal**: Real-time perception for robot autonomy

**Phase 1 - Core Rendering** (Weeks 1-4):
1. 3D Gaussian Splatting (2308.04079)
2. 2D Gaussian Splatting (2403.17888)
3. Mip-Splatting (2311.16493)

**Phase 2 - Dynamic Scenes** (Weeks 5-10):
4. Deformable 3D Gaussians (2309.13101)
5. 4D Gaussian Splatting (2310.08528)
6. SC-GS (2312.14937) - For editable scenes

**Phase 3 - Real-time Perception** (Weeks 11-16):
7. Depth Anything V2 (2406.09414) - Fast depth
8. DUSt3R (2312.14132) + MASt3R (2406.09756) - Pose estimation
9. MonoGS (2312.06741) - Monocular SLAM

**Phase 4 - Integration** (Weeks 17-20):
10. Combine depth + pose + Gaussian reconstruction
11. Optimize for real-time performance

**Time estimate**: 5-6 months

---

## Paper Index

### By Topic

#### Core 3DGS Methods
- [2308.04079] 3D Gaussian Splatting (Original)
- [2403.17888] 2D Gaussian Splatting
- [2311.16493] Mip-Splatting (Anti-aliasing)

#### Dynamic Scenes / 4D
- [2309.13101] Deformable 3D Gaussians
- [2310.08528] 4D Gaussian Splatting
- [2310.10642] Real-time Photorealistic 4DGS
- [2312.16812] Spacetime Gaussian Feature Splatting
- [2312.14937] SC-GS (Sparse-Controlled, Editable)

#### Structured Representations
- [2312.00109] Scaffold-GS (Hierarchical)

#### SLAM / Real-time Reconstruction
- [2311.11700] GS-SLAM (Dense Visual SLAM)
- [2312.02126] SplaTAM (RGB-D SLAM)
- [2312.06741] MonoGS (Monocular SLAM)

#### Depth & Pose Estimation
- [2312.14132] DUSt3R (Uncalibrated 3D Vision)
- [2406.09756] MASt3R (Matching & Stereo)
- [2406.09414] Depth Anything V2 (Monocular Depth)
- [2403.18913] UniDepth (Universal Metric Depth)

#### Background (Optional)
- [2003.08934] NeRF (Original)

---

## Additional Resources

### Implementation Codebases

#### Official Implementations
- **3DGS**: https://github.com/graphdeco-inria/gaussian-splatting (CUDA, C++)
- **2DGS**: Check paper for official code
- **Mip-Splatting**: https://github.com/autonomousvision/mip-splatting
- **4DGS**: https://github.com/hustvl/4DGaussians
- **Spacetime Gaussians**: https://github.com/oppo-us-research/SpacetimeGaussians
- **SC-GS**: https://github.com/CVMI-Lab/SC-GS
- **GS-SLAM**: https://gs-slam.github.io/
- **MonoGS**: https://github.com/muskie82/MonoGS
- **SplaTAM**: https://spla-tam.github.io/
- **DUSt3R**: https://europe.naverlabs.com/research/dust3r/
- **Depth Anything V2**: https://github.com/DepthAnything/Depth-Anything-V2
- **UniDepth**: https://github.com/lpiccinelli-eth/UniDepth

#### Community Resources
- **Awesome 3DGS**: https://github.com/qqqqqqy0227/awesome-3DGS
- **Awesome NeRF & 3DGS SLAM**: https://github.com/3D-Vision-World/awesome-NeRF-and-3DGS-SLAM
- **2024 Gaussian Splatting Papers**: https://github.com/Lee-JaeWon/2024-Arxiv-Paper-List-Gaussian-Splatting

### Online Courses & Tutorials

- **Multiple View Geometry** (Cyrill Stachniss): YouTube series on 3D vision fundamentals
- **Neural Fields** (Matthew Tancik): Stanford course covering NeRF and related methods
- **Real-Time Rendering** (SIGGRAPH Courses): Annual updates on rendering techniques

### Mathematical Prerequisites Resources

- **Linear Algebra**: *Linear Algebra and Its Applications* by Gilbert Strang
- **Multivariate Calculus**: Khan Academy, MIT OCW 18.02
- **Probability**: *All of Statistics* by Larry Wasserman (Chapters 1-5)
- **3D Math**: *3D Math Primer for Graphics and Game Development* by Dunn & Parberry
- **Quaternions**: "Understanding Quaternions" tutorial by Ken Shoemake

---

## Learning Tips

### For Papers

1. **Three-Pass Approach**:
   - **Pass 1** (15-30 min): Read abstract, intro, conclusion, figures - get the big picture
   - **Pass 2** (1-2 hours): Read full paper, skip complex math proofs, understand method
   - **Pass 3** (3-6 hours): Deep dive, work through math, understand implementation details

2. **Active Reading**:
   - Take notes, draw diagrams
   - Implement core equations in code
   - Compare with related work

3. **Group Study**:
   - Discuss papers with others
   - Present papers to reinforce understanding

### For Implementation

1. **Start Simple**:
   - Implement 3DGS first before variants
   - Use toy datasets (synthetic scenes)
   - Verify each component independently

2. **Debug Visually**:
   - Render intermediate outputs
   - Visualize Gaussians (as points, ellipsoids)
   - Check covariance matrices (positive semi-definite)

3. **Performance Later**:
   - Correctness before speed
   - Profile before optimizing
   - Start with CPU/PyTorch, move to CUDA when correct

---

## Sources

All papers and resources were found via arXiv search and academic databases. Key sources:

- [3D Gaussian Splatting Original](https://arxiv.org/abs/2308.04079)
- [2D Gaussian Splatting](https://arxiv.org/abs/2403.17888)
- [4D Gaussian Splatting](https://arxiv.org/abs/2310.08528)
- [Mip-Splatting](https://arxiv.org/abs/2311.16493)
- [SC-GS](https://arxiv.org/abs/2312.14937)
- [Scaffold-GS](https://ar5iv.labs.arxiv.org/html/2312.00109)
- [GS-SLAM](https://arxiv.org/abs/2311.11700)
- [SplaTAM](https://arxiv.org/abs/2312.02126)
- [MonoGS](https://arxiv.org/abs/2312.06741)
- [DUSt3R](https://arxiv.org/abs/2312.14132)
- [MASt3R](https://arxiv.org/abs/2406.09756)
- [Depth Anything V2](https://arxiv.org/abs/2406.09414)
- [UniDepth](https://arxiv.org/abs/2403.18913)
- [Deformable 3D Gaussians](https://ar5iv.labs.arxiv.org/html/2309.13101)
- [Spacetime Gaussian Feature Splatting](https://arxiv.org/abs/2312.16812)
- [NeRF Original](https://arxiv.org/abs/2003.08934)

---

**Next Steps**: Choose a reading path above and start with the 3D Gaussian Splatting paper. Good luck!
