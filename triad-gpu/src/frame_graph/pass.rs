use crate::frame_graph::resource::{Handle, ResourceState, ResourceType};
use crate::resource_registry::ResourceRegistry;
use std::collections::HashSet;

/// Pass execution context - provides access to device, encoder creation, and resources
pub struct PassContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub resources: &'a ResourceRegistry,
}

impl<'a> PassContext<'a> {
    /// Create a new command encoder for recording commands
    pub fn create_command_encoder(&self, label: Option<&str>) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label })
    }

    /// Get a buffer resource by handle
    pub fn get_buffer(&self, handle: Handle<wgpu::Buffer>) -> Option<&wgpu::Buffer> {
        self.resources.get_buffer(handle)
    }

    /// Get a texture resource by handle
    pub fn get_texture(&self, handle: Handle<wgpu::Texture>) -> Option<&wgpu::Texture> {
        self.resources.get_texture(handle)
    }

    /// Get a render pipeline by handle
    pub fn get_render_pipeline(
        &self,
        handle: Handle<wgpu::RenderPipeline>,
    ) -> Option<&wgpu::RenderPipeline> {
        self.resources.get_render_pipeline(handle)
    }

    /// Get a compute pipeline by handle
    pub fn get_compute_pipeline(
        &self,
        handle: Handle<wgpu::ComputePipeline>,
    ) -> Option<&wgpu::ComputePipeline> {
        self.resources.get_compute_pipeline(handle)
    }

    /// Get a bind group by handle
    pub fn get_bind_group(&self, handle: Handle<wgpu::BindGroup>) -> Option<&wgpu::BindGroup> {
        self.resources.get_bind_group(handle)
    }
}

/// Trait for frame graph passes
/// Passes return command buffers for optimal batching and parallel execution
pub trait Pass: Send + Sync {
    fn name(&self) -> &str;
    /// Execute the pass and return a command buffer
    /// The command buffer will be submitted by the frame graph executor
    fn execute(&self, ctx: &PassContext) -> wgpu::CommandBuffer;
}

/// Resource access declaration
#[derive(Debug, Clone)]
pub struct ResourceAccess {
    pub handle_id: u64,
    pub state: ResourceState,
}

/// Pass builder for declarative pass construction
pub struct PassBuilder {
    name: String,
    reads: Vec<ResourceAccess>,
    writes: Vec<ResourceAccess>,
    pass: Option<Box<dyn Pass>>,
}

impl PassBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            reads: Vec::new(),
            writes: Vec::new(),
            pass: None,
        }
    }

    pub fn read<T: ResourceType>(&mut self, handle: Handle<T>) -> &mut Self {
        self.reads.push(ResourceAccess {
            handle_id: handle.id(),
            state: ResourceState::Read,
        });
        self
    }

    pub fn write<T: ResourceType>(&mut self, handle: Handle<T>) -> &mut Self {
        self.writes.push(ResourceAccess {
            handle_id: handle.id(),
            state: ResourceState::Write,
        });
        self
    }

    pub fn read_write<T: ResourceType>(&mut self, handle: Handle<T>) -> &mut Self {
        self.reads.push(ResourceAccess {
            handle_id: handle.id(),
            state: ResourceState::Read,
        });
        self.writes.push(ResourceAccess {
            handle_id: handle.id(),
            state: ResourceState::ReadWrite,
        });
        self
    }

    pub fn with_pass(mut self, pass: Box<dyn Pass>) -> Self {
        self.pass = Some(pass);
        self
    }

    pub fn build(self) -> PassNode {
        PassNode {
            name: self.name,
            reads: self.reads.into_iter().map(|a| a.handle_id).collect(),
            writes: self.writes.into_iter().map(|a| a.handle_id).collect(),
            pass: self.pass.expect("Pass must be set"),
        }
    }
}

/// Internal pass node representation
pub struct PassNode {
    name: String,
    reads: HashSet<u64>,
    writes: HashSet<u64>,
    pass: Box<dyn Pass>,
}

impl PassNode {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn reads(&self) -> &HashSet<u64> {
        &self.reads
    }

    pub fn writes(&self) -> &HashSet<u64> {
        &self.writes
    }

    pub fn pass(&self) -> &dyn Pass {
        self.pass.as_ref()
    }

    pub fn dependencies(&self, other: &PassNode) -> bool {
        !self.writes.is_disjoint(&other.reads)
            || !self.writes.is_disjoint(&other.writes)
            || !self.reads.is_disjoint(&other.writes)
    }
}
