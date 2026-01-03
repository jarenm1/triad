//! Error types for renderer manager operations.

use thiserror::Error;

/// Errors that can occur in the renderer manager.
#[derive(Debug, Error)]
pub enum RendererManagerError {
    #[error("PLY loading error: {0}")]
    PlyError(String),

    #[error("Resource creation error: {0}")]
    ResourceError(String),

    #[error("Invalid layer index: {0}")]
    InvalidLayerIndex(u8),

    #[error("Buffer build error: {0}")]
    BufferBuildError(String),

    #[error("Bind group build error: {0}")]
    BindGroupBuildError(String),

    #[error("Pipeline build error: {0}")]
    PipelineBuildError(#[from] triad_gpu::PipelineBuildError),

    #[error("Frame graph error: {0}")]
    FrameGraphError(#[from] triad_gpu::FrameGraphError),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

// Note: Builder errors are not publicly exported, so we handle them via .map_err() at call sites

// Conversion from Box<dyn Error> for PLY loading errors
impl From<Box<dyn std::error::Error>> for RendererManagerError {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        RendererManagerError::PlyError(err.to_string())
    }
}
