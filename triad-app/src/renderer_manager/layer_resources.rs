//! Layer resource structures and types.

use std::sync::Arc;
use triad_gpu::{wgpu, Handle};

/// Holds GPU resources for a single layer.
pub struct LayerResources {
    pub pipeline: Handle<wgpu::RenderPipeline>,
    pub bind_group: Handle<wgpu::BindGroup>,
    pub bind_group_layout: Handle<wgpu::BindGroupLayout>,
    pub data_buffer: Handle<wgpu::Buffer>,
    pub index_buffer: Option<Handle<wgpu::Buffer>>,
    pub vertex_count: u32,
    pub index_count: u32,
    pub uses_indices: bool,
    // Intermediate render target
    pub texture: Handle<wgpu::Texture>,
    pub texture_view: Arc<wgpu::TextureView>,
}

impl LayerResources {
    /// Create a new empty layer resources structure.
    pub fn new(
        pipeline: Handle<wgpu::RenderPipeline>,
        bind_group: Handle<wgpu::BindGroup>,
        bind_group_layout: Handle<wgpu::BindGroupLayout>,
        data_buffer: Handle<wgpu::Buffer>,
        index_buffer: Option<Handle<wgpu::Buffer>>,
        vertex_count: u32,
        index_count: u32,
        uses_indices: bool,
        texture: Handle<wgpu::Texture>,
        texture_view: Arc<wgpu::TextureView>,
    ) -> Self {
        Self {
            pipeline,
            bind_group,
            bind_group_layout,
            data_buffer,
            index_buffer,
            vertex_count,
            index_count,
            uses_indices,
            texture,
            texture_view,
        }
    }
}
