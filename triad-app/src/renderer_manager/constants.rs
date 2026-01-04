//! Constants used throughout the renderer manager.

use triad_gpu::wgpu;

/// Number of render layers (Points, Gaussians, Triangles).
pub const LAYER_COUNT: usize = 3;

/// Default opacity for all layers.
pub const DEFAULT_OPACITY: [f32; LAYER_COUNT] = [1.0, 1.0, 1.0];

/// Depth format used for depth buffers.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
