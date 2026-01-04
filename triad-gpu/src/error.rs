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

    /// Error during pipeline operations
    #[error(transparent)]
    Pipeline(#[from] PipelineError),

    /// Error during frame graph operations
    #[error(transparent)]
    FrameGraph(#[from] FrameGraphError),

    /// Error during PLY file loading
    #[error(transparent)]
    Ply(#[from] PlyError),
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
    #[error("invalid buffer offset: offset {offset} + data size {data_size} exceeds buffer size {buffer_size}")]
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

/// Errors that occur during render pipeline operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PipelineError {
    /// Vertex shader is required for render pipelines
    #[error("vertex shader is required")]
    MissingVertexShader,

    /// Shader module not found in registry
    #[error("shader module not found in registry")]
    ShaderNotFound,
}

/// Errors that occur during frame graph operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FrameGraphError {
    /// Circular dependency detected between passes
    #[error("circular dependency detected in frame graph")]
    CircularDependency,
}

/// Errors that occur during PLY file loading.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PlyError {
    /// Failed to open the PLY file
    #[error("failed to open PLY file: {0}")]
    Io(#[from] std::io::Error),

    /// Failed to parse the PLY header
    #[error("failed to parse PLY header: {message}")]
    HeaderParse { message: String },

    /// Failed to parse a PLY element
    #[error("failed to parse PLY element '{element}': {message}")]
    ElementParse { element: String, message: String },

    /// Required vertex property is missing
    #[error("missing required property '{property}' at vertex {vertex_index}")]
    MissingProperty {
        property: &'static str,
        vertex_index: usize,
    },

    /// PLY file contains no face data when faces were expected
    #[error("PLY file contains no face data; use triangulation for point clouds")]
    NoFaceData,

    /// Face references an out-of-bounds vertex index
    #[error("face {face_index} references invalid vertex index {vertex_index} (max: {max_index})")]
    InvalidVertexIndex {
        face_index: usize,
        vertex_index: usize,
        max_index: usize,
    },
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

        let err = PlyError::MissingProperty {
            property: "x",
            vertex_index: 42,
        };
        assert_eq!(
            err.to_string(),
            "missing required property 'x' at vertex 42"
        );
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
    }
}
