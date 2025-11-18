use crate::frame_graph::resource::Handle;
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

    /// Register a buffer resource.
    /// If a buffer with the same handle ID already exists, it will be silently replaced.
    /// The old buffer will be dropped, preventing memory leaks.
    pub fn register_buffer(&mut self, handle: Handle<wgpu::Buffer>, buffer: wgpu::Buffer) {
        let id = handle.id(); // Extract id to avoid move
        // HashMap::insert replaces existing entries, dropping the old value
        self.buffers.insert(id, buffer);
    }

    /// Get a buffer by handle
    pub fn get_buffer(&self, handle: Handle<wgpu::Buffer>) -> Option<&wgpu::Buffer> {
        self.buffers.get(&handle.id())
    }

    /// Remove a buffer resource by handle.
    /// Returns the removed buffer if it existed, or None otherwise.
    pub fn remove_buffer(&mut self, handle: Handle<wgpu::Buffer>) -> Option<wgpu::Buffer> {
        self.buffers.remove(&handle.id())
    }

    /// Register a texture resource.
    /// If a texture with the same handle ID already exists, it will be silently replaced.
    /// The old texture will be dropped, preventing memory leaks.
    pub fn register_texture(&mut self, handle: Handle<wgpu::Texture>, texture: wgpu::Texture) {
        let id = handle.id();
        // HashMap::insert replaces existing entries, dropping the old value
        self.textures.insert(id, texture);
    }

    /// Get a texture by handle
    pub fn get_texture(&self, handle: Handle<wgpu::Texture>) -> Option<&wgpu::Texture> {
        self.textures.get(&handle.id())
    }

    /// Remove a texture resource by handle.
    /// Returns the removed texture if it existed, or None otherwise.
    pub fn remove_texture(&mut self, handle: Handle<wgpu::Texture>) -> Option<wgpu::Texture> {
        self.textures.remove(&handle.id())
    }

    /// Register a sampler resource.
    /// If a sampler with the same handle ID already exists, it will be silently replaced.
    /// The old sampler will be dropped, preventing memory leaks.
    pub fn register_sampler(&mut self, handle: Handle<wgpu::Sampler>, sampler: wgpu::Sampler) {
        let id = handle.id();
        // HashMap::insert replaces existing entries, dropping the old value
        self.samplers.insert(id, sampler);
    }

    /// Get a sampler by handle
    pub fn get_sampler(&self, handle: Handle<wgpu::Sampler>) -> Option<&wgpu::Sampler> {
        self.samplers.get(&handle.id())
    }

    /// Remove a sampler resource by handle.
    /// Returns the removed sampler if it existed, or None otherwise.
    pub fn remove_sampler(&mut self, handle: Handle<wgpu::Sampler>) -> Option<wgpu::Sampler> {
        self.samplers.remove(&handle.id())
    }

    /// Register a bind group resource.
    /// If a bind group with the same handle ID already exists, it will be silently replaced.
    /// The old bind group will be dropped, preventing memory leaks.
    pub fn register_bind_group(
        &mut self,
        handle: Handle<wgpu::BindGroup>,
        bind_group: wgpu::BindGroup,
    ) {
        let id = handle.id();
        // HashMap::insert replaces existing entries, dropping the old value
        self.bind_groups.insert(id, bind_group);
    }

    /// Get a bind group by handle
    pub fn get_bind_group(&self, handle: Handle<wgpu::BindGroup>) -> Option<&wgpu::BindGroup> {
        self.bind_groups.get(&handle.id())
    }

    /// Remove a bind group resource by handle.
    /// Returns the removed bind group if it existed, or None otherwise.
    pub fn remove_bind_group(
        &mut self,
        handle: Handle<wgpu::BindGroup>,
    ) -> Option<wgpu::BindGroup> {
        self.bind_groups.remove(&handle.id())
    }

    /// Register a render pipeline resource.
    /// If a render pipeline with the same handle ID already exists, it will be silently replaced.
    /// The old pipeline will be dropped, preventing memory leaks.
    pub fn register_render_pipeline(
        &mut self,
        handle: Handle<wgpu::RenderPipeline>,
        pipeline: wgpu::RenderPipeline,
    ) {
        let id = handle.id();
        // HashMap::insert replaces existing entries, dropping the old value
        self.render_pipelines.insert(id, pipeline);
    }

    /// Get a render pipeline by handle
    pub fn get_render_pipeline(
        &self,
        handle: Handle<wgpu::RenderPipeline>,
    ) -> Option<&wgpu::RenderPipeline> {
        self.render_pipelines.get(&handle.id())
    }

    /// Remove a render pipeline resource by handle.
    /// Returns the removed pipeline if it existed, or None otherwise.
    pub fn remove_render_pipeline(
        &mut self,
        handle: Handle<wgpu::RenderPipeline>,
    ) -> Option<wgpu::RenderPipeline> {
        self.render_pipelines.remove(&handle.id())
    }

    /// Register a compute pipeline resource.
    /// If a compute pipeline with the same handle ID already exists, it will be silently replaced.
    /// The old pipeline will be dropped, preventing memory leaks.
    pub fn register_compute_pipeline(
        &mut self,
        handle: Handle<wgpu::ComputePipeline>,
        pipeline: wgpu::ComputePipeline,
    ) {
        let id = handle.id();
        // HashMap::insert replaces existing entries, dropping the old value
        self.compute_pipelines.insert(id, pipeline);
    }

    /// Get a compute pipeline by handle
    pub fn get_compute_pipeline(
        &self,
        handle: Handle<wgpu::ComputePipeline>,
    ) -> Option<&wgpu::ComputePipeline> {
        self.compute_pipelines.get(&handle.id())
    }

    /// Remove a compute pipeline resource by handle.
    /// Returns the removed pipeline if it existed, or None otherwise.
    pub fn remove_compute_pipeline(
        &mut self,
        handle: Handle<wgpu::ComputePipeline>,
    ) -> Option<wgpu::ComputePipeline> {
        self.compute_pipelines.remove(&handle.id())
    }

    /// Register a shader module resource.
    /// If a shader module with the same handle ID already exists, it will be silently replaced.
    /// The old shader module will be dropped, preventing memory leaks.
    pub fn register_shader_module(
        &mut self,
        handle: Handle<wgpu::ShaderModule>,
        shader: wgpu::ShaderModule,
    ) {
        let id = handle.id();
        // HashMap::insert replaces existing entries, dropping the old value
        self.shader_modules.insert(id, shader);
    }

    /// Get a shader module by handle
    pub fn get_shader_module(
        &self,
        handle: Handle<wgpu::ShaderModule>,
    ) -> Option<&wgpu::ShaderModule> {
        self.shader_modules.get(&handle.id())
    }

    /// Remove a shader module resource by handle.
    /// Returns the removed shader module if it existed, or None otherwise.
    pub fn remove_shader_module(
        &mut self,
        handle: Handle<wgpu::ShaderModule>,
    ) -> Option<wgpu::ShaderModule> {
        self.shader_modules.remove(&handle.id())
    }

    /// Check if a resource ID corresponds to a buffer
    pub fn is_buffer(&self, id: u64) -> bool {
        self.buffers.contains_key(&id)
    }

    /// Check if a resource ID corresponds to a texture
    pub fn is_texture(&self, id: u64) -> bool {
        self.textures.contains_key(&id)
    }

    /// Get a buffer by ID (for barrier insertion)
    pub fn get_buffer_by_id(&self, id: u64) -> Option<&wgpu::Buffer> {
        self.buffers.get(&id)
    }

    /// Get a texture by ID (for barrier insertion)
    pub fn get_texture_by_id(&self, id: u64) -> Option<&wgpu::Texture> {
        self.textures.get(&id)
    }

    /// Remove a resource by ID, checking all resource types.
    /// Returns true if a resource was found and removed, false otherwise.
    /// This is useful when you don't know the resource type but have the ID.
    pub fn remove_by_id(&mut self, id: u64) -> bool {
        // Try removing from each resource type
        // Order doesn't matter since IDs should be unique across types
        self.buffers.remove(&id).is_some()
            || self.textures.remove(&id).is_some()
            || self.samplers.remove(&id).is_some()
            || self.bind_groups.remove(&id).is_some()
            || self.render_pipelines.remove(&id).is_some()
            || self.compute_pipelines.remove(&id).is_some()
            || self.shader_modules.remove(&id).is_some()
    }

    /// Clear all buffers from the registry.
    /// All buffers will be dropped, freeing their GPU memory.
    pub fn clear_buffers(&mut self) {
        self.buffers.clear();
    }

    /// Clear all textures from the registry.
    /// All textures will be dropped, freeing their GPU memory.
    pub fn clear_textures(&mut self) {
        self.textures.clear();
    }

    /// Clear all samplers from the registry.
    pub fn clear_samplers(&mut self) {
        self.samplers.clear();
    }

    /// Clear all bind groups from the registry.
    pub fn clear_bind_groups(&mut self) {
        self.bind_groups.clear();
    }

    /// Clear all render pipelines from the registry.
    pub fn clear_render_pipelines(&mut self) {
        self.render_pipelines.clear();
    }

    /// Clear all compute pipelines from the registry.
    pub fn clear_compute_pipelines(&mut self) {
        self.compute_pipelines.clear();
    }

    /// Clear all shader modules from the registry.
    pub fn clear_shader_modules(&mut self) {
        self.shader_modules.clear();
    }

    /// Clear all resources from the registry.
    /// All resources will be dropped, freeing their GPU memory.
    /// This is useful for cleanup when shutting down or resetting the renderer.
    pub fn clear_all(&mut self) {
        self.buffers.clear();
        self.textures.clear();
        self.samplers.clear();
        self.bind_groups.clear();
        self.render_pipelines.clear();
        self.compute_pipelines.clear();
        self.shader_modules.clear();
    }

    /// Get the total number of resources currently registered.
    pub fn resource_count(&self) -> usize {
        self.buffers.len()
            + self.textures.len()
            + self.samplers.len()
            + self.bind_groups.len()
            + self.render_pipelines.len()
            + self.compute_pipelines.len()
            + self.shader_modules.len()
    }
}

impl Default for ResourceRegistry {
    fn default() -> Self {
        Self::new()
    }
}
