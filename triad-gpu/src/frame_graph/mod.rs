mod execution;
pub mod pass;
pub mod resource;

use crate::frame_graph::pass::PassNode;
use crate::frame_graph::resource::{ResourceInfo, ResourceState};
use crate::resource_registry::ResourceRegistry;
use std::collections::HashMap;

pub use pass::{Pass, PassBuilder, PassContext};
pub use resource::{Handle, ResourceType, next_handle_id};

/// Frame graph builder
pub struct FrameGraph {
    passes: Vec<PassNode>,
    resource_info: HashMap<u64, ResourceInfo>,
    surface_handles: Vec<u64>, // Surface handle IDs (surfaces tracked separately)
}

impl FrameGraph {
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            resource_info: HashMap::new(),
            surface_handles: Vec::new(),
        }
    }

    /// Register a resource in the frame graph
    pub fn register_resource<T: ResourceType>(&mut self, handle: Handle<T>) -> &mut Self {
        self.resource_info
            .entry(handle.id())
            .or_insert_with(ResourceInfo::new);
        self
    }

    /// Add a pass to the frame graph
    pub fn add_pass(&mut self, builder: PassBuilder) -> &mut Self {
        let pass = builder.build();

        // Update resource tracking
        for read_id in &pass.reads {
            let info = self
                .resource_info
                .entry(*read_id)
                .or_insert_with(ResourceInfo::new);
            info.state = ResourceState::Read;
        }

        for write_id in &pass.writes {
            let info = self
                .resource_info
                .entry(*write_id)
                .or_insert_with(ResourceInfo::new);
            info.state = ResourceState::Write;
        }

        self.passes.push(pass);
        self
    }

    /// Register a surface for multi-surface rendering
    /// Surfaces are tracked separately as they have special lifetime requirements
    pub fn register_surface(&mut self, surface_id: u64) -> &mut Self {
        self.surface_handles.push(surface_id);
        self
    }

    /// Build and validate the frame graph
    pub fn build(mut self) -> Result<ExecutableFrameGraph, FrameGraphError> {
        // Topological sort for execution order
        let execution_order = execution::topological_sort(&self.passes)?;

        // Update resource usage tracking
        for (idx, &pass_idx) in execution_order.iter().enumerate() {
            let pass = &self.passes[pass_idx];
            for read_id in &pass.reads {
                if let Some(info) = self.resource_info.get_mut(read_id) {
                    info.first_used_pass = info.first_used_pass.min(idx);
                    info.last_used_pass = info.last_used_pass.max(idx);
                }
            }
            for write_id in &pass.writes {
                if let Some(info) = self.resource_info.get_mut(write_id) {
                    info.first_used_pass = info.first_used_pass.min(idx);
                    info.last_used_pass = info.last_used_pass.max(idx);
                }
            }
        }

        Ok(ExecutableFrameGraph {
            passes: self.passes,
            execution_order,
            resource_info: self.resource_info,
            surface_handles: self.surface_handles,
        })
    }
}

/// Executable frame graph ready for execution
pub struct ExecutableFrameGraph {
    passes: Vec<PassNode>,
    execution_order: Vec<usize>,
    resource_info: HashMap<u64, ResourceInfo>,
    surface_handles: Vec<u64>, // Surface handle IDs (surfaces tracked separately)
}

impl ExecutableFrameGraph {
    /// Execute the frame graph
    /// Collects command buffers from passes and submits them in optimal order
    pub fn execute(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resources: &ResourceRegistry,
    ) {
        let ctx = pass::PassContext {
            device,
            queue,
            resources,
        };
        let mut command_buffers = Vec::new();

        // Execute passes in dependency order and collect command buffers
        for &pass_idx in &self.execution_order {
            // TODO: Insert barriers based on resource state transitions
            let pass = &self.passes[pass_idx];
            let command_buffer = pass.pass.execute(&ctx);
            command_buffers.push(command_buffer);
        }

        // Submit all command buffers in a single batch for optimal performance
        queue.submit(command_buffers);
    }

    pub fn surface_count(&self) -> usize {
        self.surface_handles.len()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum FrameGraphError {
    #[error("Circular dependency detected in frame graph")]
    CircularDependency,
}
