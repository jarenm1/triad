//! Error types for triad-gpu.
//!
//! This module provides a unified error handling system following modern Rust practices:
//! - All error types are concrete and exported
//! - A unified `GpuError` type wraps all errors for easy `?` propagation
//! - `#[non_exhaustive]` ensures forward compatibility
//! - Errors include context where helpful

use thiserror::Error;

/// Unified error type for all triad-gpu operations.
///
/// This enum wraps all specific error types, allowing easy error propagation
/// with the `?` operator while still permitting granular error matching.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GpuError {
    /// Error during renderer initialization
    #[error(transparent)]
    Renderer(#[from] RendererError),

    /// Error during buffer operations
    #[error(transparent)]
    Buffer(#[from] BufferError),

    /// Error during bind group operations
    #[error(transparent)]
    BindGroup(#[from] BindGroupError),

    /// Error during shader operations
    #[error(transparent)]
    Shader(#[from] ShaderError),

    /// Error during pipeline operations
    #[error(transparent)]
    Pipeline(#[from] PipelineError),

    /// Error during compute pass construction
    #[error(transparent)]
    ComputePass(#[from] ComputePassError),

    /// Error during copy pass construction
    #[error(transparent)]
    CopyPass(#[from] CopyPassError),

    /// Error during render pass construction
    #[error(transparent)]
    RenderPass(#[from] RenderPassError),

    /// Error during frame graph operations
    #[error(transparent)]
    FrameGraph(#[from] FrameGraphError),

    /// Error during CPU readback from GPU buffers
    #[error(transparent)]
    Readback(#[from] ReadbackError),
}

/// Errors that occur during renderer initialization and surface management.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RendererError {
    /// Failed to request a GPU adapter
    #[error("failed to request GPU adapter: {0}")]
    RequestAdapter(#[from] wgpu::RequestAdapterError),

    /// Failed to request a GPU device
    #[error("failed to request GPU device: {0}")]
    RequestDevice(#[from] wgpu::RequestDeviceError),

    /// Surface operation failed
    #[error("surface error: {0}")]
    Surface(#[from] wgpu::SurfaceError),

    /// Invalid surface dimensions
    #[error("invalid surface dimensions: {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },

    /// No supported surface formats available
    #[error("no supported surface formats available")]
    NoSupportedFormats,

    /// No supported present modes available
    #[error("no supported present modes available")]
    NoSupportedPresentModes,

    /// No supported alpha modes available
    #[error("no supported alpha modes available")]
    NoSupportedAlphaModes,
}

/// Errors that occur during buffer operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum BufferError {
    /// Buffer requires either a size or initial data
    #[error("buffer must have either size or data specified")]
    MissingSizeOrData,

    /// Buffer handle not found in registry
    #[error("buffer not found in registry")]
    NotFound,

    /// Buffer write would exceed buffer bounds
    #[error(
        "invalid buffer offset: offset {offset} + data size {data_size} exceeds buffer size {buffer_size}"
    )]
    InvalidOffset {
        offset: u64,
        data_size: u64,
        buffer_size: u64,
    },

    /// DynamicBuffer capacity exceeded
    #[error("buffer capacity exceeded: requested {requested} elements but capacity is {capacity}")]
    CapacityExceeded { requested: usize, capacity: usize },
}

/// Errors that occur during bind group operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum BindGroupError {
    /// A resource handle was not found in the registry
    #[error("resource not found in registry at binding {binding}")]
    ResourceNotFound { binding: u32 },

    /// Bind group has no entries
    #[error("bind group must have at least one entry")]
    NoEntries,

    /// Bind group layout was not found after creation (internal error)
    #[error("bind group layout not found in registry")]
    LayoutNotFound,
}

/// Errors that occur during shader module operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ShaderError {
    /// Shader module builder requires an in-memory source payload.
    #[error("shader module must have source specified")]
    MissingSource,
}

/// Errors that occur during render pipeline operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PipelineError {
    /// Vertex shader is required for render pipelines
    #[error("vertex shader is required")]
    MissingVertexShader,

    /// Compute shader is required for compute pipelines
    #[error("compute shader is required")]
    MissingComputeShader,

    /// Shader module not found in registry
    #[error("shader module not found in registry")]
    ShaderNotFound,
}

/// Errors that occur during compute pass construction.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ComputePassError {
    /// Compute pass requires a pipeline handle.
    #[error("compute pass requires a compute pipeline")]
    MissingPipeline,

    /// Compute pass requires a dispatch configuration.
    #[error("compute pass requires a dispatch configuration")]
    MissingDispatch,
}

/// Errors that occur during copy pass construction.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CopyPassError {
    /// Copy pass requires at least one copy command.
    #[error("copy pass requires at least one copy command")]
    MissingCopy,
}

/// Errors that occur during render pass construction.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RenderPassError {
    /// Render pass requires a render pipeline handle.
    #[error("render pass requires a render pipeline")]
    MissingPipeline,

    /// Render pass requires at least one color attachment.
    #[error("render pass requires at least one color attachment")]
    MissingColorAttachment,

    /// Render pass requires a draw configuration.
    #[error("render pass requires a draw configuration")]
    MissingDraw,

    /// Indexed draw requires `with_index_buffer` before `build`.
    #[error("indexed draw requires an index buffer")]
    MissingIndexBuffer,

    /// Non-indexed draw must not set an index buffer.
    #[error("non-indexed draw cannot use an index buffer")]
    UnexpectedIndexBuffer,
}

/// Errors that occur during frame graph operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FrameGraphError {
    /// Circular dependency detected between passes
    #[error("circular dependency detected in frame graph")]
    CircularDependency,
}

/// Errors that occur while reading back GPU buffer contents to the CPU.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ReadbackError {
    /// Buffer handle not found in registry
    #[error("buffer not found in registry")]
    BufferNotFound,

    /// Buffer size is not a whole number of requested elements
    #[error("buffer size {buffer_size} is not aligned to element size {element_size}")]
    BufferSizeNotAligned {
        buffer_size: u64,
        element_size: usize,
    },

    /// Waiting for buffer mapping callback failed
    #[error("buffer map callback channel closed before completion")]
    MapChannelClosed,

    /// Device polling failed
    #[error("device poll failed during readback: {0}")]
    Poll(#[from] wgpu::PollError),

    /// Buffer mapping failed
    #[error("buffer map failed during readback: {0}")]
    Map(#[from] wgpu::BufferAsyncError),
}

/// Result type alias using the unified `GpuError`.
pub type Result<T> = std::result::Result<T, GpuError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = BufferError::MissingSizeOrData;
        assert_eq!(
            err.to_string(),
            "buffer must have either size or data specified"
        );

        let err = BindGroupError::ResourceNotFound { binding: 2 };
        assert_eq!(
            err.to_string(),
            "resource not found in registry at binding 2"
        );

        let err = ShaderError::MissingSource;
        assert_eq!(err.to_string(), "shader module must have source specified");
    }

    #[test]
    fn test_error_conversion_to_gpu_error() {
        let buffer_err = BufferError::NotFound;
        let gpu_err: GpuError = buffer_err.into();
        assert!(matches!(gpu_err, GpuError::Buffer(BufferError::NotFound)));

        let pipeline_err = PipelineError::MissingVertexShader;
        let gpu_err: GpuError = pipeline_err.into();
        assert!(matches!(
            gpu_err,
            GpuError::Pipeline(PipelineError::MissingVertexShader)
        ));

        let shader_err = ShaderError::MissingSource;
        let gpu_err: GpuError = shader_err.into();
        assert!(matches!(
            gpu_err,
            GpuError::Shader(ShaderError::MissingSource)
        ));

        let compute_pass_err = ComputePassError::MissingDispatch;
        let gpu_err: GpuError = compute_pass_err.into();
        assert!(matches!(
            gpu_err,
            GpuError::ComputePass(ComputePassError::MissingDispatch)
        ));

        let render_pass_err = RenderPassError::MissingDraw;
        let gpu_err: GpuError = render_pass_err.into();
        assert!(matches!(
            gpu_err,
            GpuError::RenderPass(RenderPassError::MissingDraw)
        ));

        let copy_pass_err = CopyPassError::MissingCopy;
        let gpu_err: GpuError = copy_pass_err.into();
        assert!(matches!(
            gpu_err,
            GpuError::CopyPass(CopyPassError::MissingCopy)
        ));
    }
}
