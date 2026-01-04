//! Error types for renderer manager operations.

use thiserror::Error;

/// Errors that can occur in the renderer manager.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RendererManagerError {
    /// PLY file loading error
    #[error(transparent)]
    Ply(#[from] triad_gpu::PlyError),

    /// Resource creation error
    #[error("resource creation error: {0}")]
    Resource(String),

    /// Invalid layer index
    #[error("invalid layer index: {0}")]
    InvalidLayerIndex(u8),

    /// Buffer operation error
    #[error(transparent)]
    Buffer(#[from] triad_gpu::BufferError),

    /// Bind group operation error
    #[error(transparent)]
    BindGroup(#[from] triad_gpu::BindGroupError),

    /// Pipeline operation error
    #[error(transparent)]
    Pipeline(#[from] triad_gpu::PipelineError),

    /// Frame graph error
    #[error(transparent)]
    FrameGraph(#[from] triad_gpu::FrameGraphError),

    /// IO error
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
