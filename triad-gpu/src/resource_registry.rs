use crate::frame_graph::resource::{Handle, ResourceType};
use std::any::{Any, TypeId};
use std::collections::HashMap;

/// Registry mapping handles to actual wgpu resources
/// Provides type-safe resource lookup and management
pub struct ResourceRegistry {
    buffers: HashMap<u64, wgpu::Buffer>,
    textures: HashMap<u64, wgpu::Texture>,
    samplers: HashMap<u64, wgpu::Sampler>,
    bind_groups: HashMap<u64, wgpu::BindGroup>,
    render_pipelines: HashMap<u64, wgpu::RenderPipeline>,
    compute_pipelines: HashMap<u64, wgpu::ComputePipeline>,
    shader_modules: HashMap<u64, wgpu::ShaderModule>,
}

impl ResourceRegistry {
    pub fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            textures: HashMap::new(),
            samplers: HashMap::new(),
            bind_groups: HashMap::new(),
            render_pipelines: HashMap::new(),
            compute_pipelines: HashMap::new(),
            shader_modules: HashMap::new(),
        }
    }

    /// Register a buffer resource
    pub fn register_buffer(&mut self, handle: Handle<wgpu::Buffer>, buffer: wgpu::Buffer) {
        let id = handle.id; // Extract id to avoid move
        self.buffers.insert(id, buffer);
    }

    /// Get a buffer by handle
    pub fn get_buffer(&self, handle: Handle<wgpu::Buffer>) -> Option<&wgpu::Buffer> {
        self.buffers.get(&handle.id)
    }

    /// Register a texture resource
    pub fn register_texture(&mut self, handle: Handle<wgpu::Texture>, texture: wgpu::Texture) {
        let id = handle.id;
        self.textures.insert(id, texture);
    }

    /// Get a texture by handle
    pub fn get_texture(&self, handle: Handle<wgpu::Texture>) -> Option<&wgpu::Texture> {
        self.textures.get(&handle.id)
    }

    /// Register a sampler resource
    pub fn register_sampler(&mut self, handle: Handle<wgpu::Sampler>, sampler: wgpu::Sampler) {
        let id = handle.id;
        self.samplers.insert(id, sampler);
    }

    /// Get a sampler by handle
    pub fn get_sampler(&self, handle: Handle<wgpu::Sampler>) -> Option<&wgpu::Sampler> {
        self.samplers.get(&handle.id)
    }

    /// Register a bind group resource
    pub fn register_bind_group(
        &mut self,
        handle: Handle<wgpu::BindGroup>,
        bind_group: wgpu::BindGroup,
    ) {
        let id = handle.id;
        self.bind_groups.insert(id, bind_group);
    }

    /// Get a bind group by handle
    pub fn get_bind_group(&self, handle: Handle<wgpu::BindGroup>) -> Option<&wgpu::BindGroup> {
        self.bind_groups.get(&handle.id)
    }

    /// Register a render pipeline resource
    pub fn register_render_pipeline(
        &mut self,
        handle: Handle<wgpu::RenderPipeline>,
        pipeline: wgpu::RenderPipeline,
    ) {
        let id = handle.id;
        self.render_pipelines.insert(id, pipeline);
    }

    /// Get a render pipeline by handle
    pub fn get_render_pipeline(
        &self,
        handle: Handle<wgpu::RenderPipeline>,
    ) -> Option<&wgpu::RenderPipeline> {
        self.render_pipelines.get(&handle.id)
    }

    /// Register a compute pipeline resource
    pub fn register_compute_pipeline(
        &mut self,
        handle: Handle<wgpu::ComputePipeline>,
        pipeline: wgpu::ComputePipeline,
    ) {
        let id = handle.id;
        self.compute_pipelines.insert(id, pipeline);
    }

    /// Get a compute pipeline by handle
    pub fn get_compute_pipeline(
        &self,
        handle: Handle<wgpu::ComputePipeline>,
    ) -> Option<&wgpu::ComputePipeline> {
        self.compute_pipelines.get(&handle.id)
    }

    /// Register a shader module resource
    pub fn register_shader_module(
        &mut self,
        handle: Handle<wgpu::ShaderModule>,
        shader: wgpu::ShaderModule,
    ) {
        let id = handle.id;
        self.shader_modules.insert(id, shader);
    }

    /// Get a shader module by handle
    pub fn get_shader_module(
        &self,
        handle: Handle<wgpu::ShaderModule>,
    ) -> Option<&wgpu::ShaderModule> {
        self.shader_modules.get(&handle.id)
    }
}

impl Default for ResourceRegistry {
    fn default() -> Self {
        Self::new()
    }
}
