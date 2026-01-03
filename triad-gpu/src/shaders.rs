//! Shader source code embedded at compile time.
//!
//! This module exposes WGSL shader sources for use in pipelines. The shaders
//! are embedded using `include_str!` at compile time, so they're available as
//! static string constants.

/// Point cloud vertex shader - renders points as billboarded quads.
pub const POINT_VERTEX: &str = include_str!("../shaders/point_vertex.wgsl");

/// Point cloud fragment shader - simple solid color output.
pub const POINT_FRAGMENT: &str = include_str!("../shaders/point_fragment.wgsl");

/// Gaussian splat vertex shader - 3D Gaussian splatting with anisotropic Gaussians.
pub const GAUSSIAN_VERTEX: &str = include_str!("../shaders/gaussian_vertex.wgsl");

/// Gaussian splat fragment shader - renders Gaussians with alpha blending.
pub const GAUSSIAN_FRAGMENT: &str = include_str!("../shaders/gaussian_fragment.wgsl");

/// Triangle mesh vertex shader - simple triangle rendering.
pub const TRIANGLE_VERTEX: &str = include_str!("../shaders/triangle_vertex.wgsl");

/// Triangle mesh fragment shader - renders triangles with premultiplied alpha.
pub const TRIANGLE_FRAGMENT: &str = include_str!("../shaders/triangle_fragment.wgsl");

/// Gaussian sorting compute shader - sorts gaussians by depth.
pub const GAUSSIAN_SORT: &str = include_str!("../shaders/gaussian_sort.wgsl");

/// Layer blending shader - composites multiple layer textures with opacity.
pub const LAYER_BLEND: &str = include_str!("../shaders/layer_blend.wgsl");

