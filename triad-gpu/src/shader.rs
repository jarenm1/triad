use crate::frame_graph::resource::{Handle, ResourceType, next_handle_id};
use crate::resource_registry::ResourceRegistry;

/// Shader manager for creating and tracking shader modules
pub struct ShaderManager {
    registry: ResourceRegistry,
}

impl ShaderManager {
    pub fn new() -> Self {
        Self {
            registry: ResourceRegistry::new(),
        }
    }

    /// Create a shader module from WGSL source code
    pub fn create_shader(
        &mut self,
        device: &wgpu::Device,
        label: Option<&str>,
        source: &str,
    ) -> Handle<wgpu::ShaderModule> {
        let handle = Handle::new(next_handle_id());
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label,
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        // Handle implements Copy, but we clone explicitly to avoid move issues
        // For Copy types, clone() is just a copy operation (cheap)
        self.registry
            .register_shader_module(handle.clone(), shader_module);
        handle
    }

    /// Get a shader module by handle
    pub fn get_shader(&self, handle: Handle<wgpu::ShaderModule>) -> Option<&wgpu::ShaderModule> {
        self.registry.get_shader_module(handle)
    }

    /// Get mutable access to the resource registry (for advanced use cases)
    pub fn registry_mut(&mut self) -> &mut ResourceRegistry {
        &mut self.registry
    }

    /// Get immutable access to the resource registry
    pub fn registry(&self) -> &ResourceRegistry {
        &self.registry
    }
}

impl Default for ShaderManager {
    fn default() -> Self {
        Self::new()
    }
}
