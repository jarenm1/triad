mod execution;
pub mod pass;
pub mod resource;

use crate::frame_graph::pass::PassNode;
use crate::frame_graph::resource::{ResourceInfo, ResourceState};
use crate::resource_registry::ResourceRegistry;
use std::collections::HashMap;

pub use pass::{Pass, PassBuilder, PassContext};
pub use resource::{Handle, HandleId, ResourceType};

pub type SurfaceId = u64;

#[derive(Default)]
pub struct FrameGraph {
    passes: Vec<PassNode>,
    resource_info: HashMap<HandleId, ResourceInfo>,
    surface_handles: Vec<SurfaceId>, // Surface handle IDs (surfaces tracked separately)
}

impl FrameGraph {
    pub fn register_resource<T: ResourceType>(&mut self, handle: Handle<T>) -> &mut Self {
        self.resource_info
            .entry(handle.id())
            .or_insert_with(ResourceInfo::new);
        self
    }

    pub fn add_pass(&mut self, builder: PassBuilder) -> &mut Self {
        let pass = builder.build();

        // Update resource tracking - accumulate access patterns rather than overwriting
        for read_id in pass.reads() {
            let info = self
                .resource_info
                .entry(*read_id)
                .or_insert_with(ResourceInfo::new);
            info.set_state(info.state().merge_with(ResourceState::Read));
        }

        for write_id in pass.writes() {
            let info = self
                .resource_info
                .entry(*write_id)
                .or_insert_with(ResourceInfo::new);
            info.set_state(info.state().merge_with(ResourceState::Write));
        }

        self.passes.push(pass);
        self
    }

    /// Register a surface for multi-surface rendering
    /// Surfaces are tracked separately as they have special lifetime requirements
    pub fn register_surface(&mut self, surface_id: u64) -> &mut Self {
        if !self.surface_handles.contains(&surface_id) {
            self.surface_handles.push(surface_id);
        }
        self
    }
    pub fn build(mut self) -> Result<ExecutableFrameGraph, FrameGraphError> {
        let execution_order = execution::topological_sort(&self.passes)?;

        for (idx, &pass_idx) in execution_order.iter().enumerate() {
            let pass = &self.passes[pass_idx];
            for read_id in pass.reads() {
                if let Some(info) = self.resource_info.get_mut(read_id) {
                    info.set_first_used_pass(info.first_used_pass().min(idx));
                    info.set_last_used_pass(info.last_used_pass().max(idx));
                }
            }
            for write_id in pass.writes() {
                if let Some(info) = self.resource_info.get_mut(write_id) {
                    info.set_first_used_pass(info.first_used_pass().min(idx));
                    info.set_last_used_pass(info.last_used_pass().max(idx));
                }
            }
        }

        Ok(ExecutableFrameGraph {
            passes: self.passes,
            execution_order,
            surface_handles: self.surface_handles,
        })
    }
}

/// Executable frame graph ready for execution
pub struct ExecutableFrameGraph {
    passes: Vec<PassNode>,
    execution_order: Vec<usize>,
    surface_handles: Vec<u64>, // Surface handle IDs (surfaces tracked separately)
}

impl ExecutableFrameGraph {
    /// Track resource state transitions and validate barrier requirements
    /// wgpu handles most barriers automatically, but we track transitions for validation
    /// and potential future explicit barrier insertion
    fn track_state_transitions(
        &self,
        current_states: &HashMap<u64, ResourceState>,
        pass: &PassNode,
    ) -> Vec<(u64, ResourceState, ResourceState)> {
        let mut transitions = Vec::new();

        // Check all resources accessed by this pass
        let mut resources_to_check = std::collections::HashSet::new();
        resources_to_check.extend(pass.reads());
        resources_to_check.extend(pass.writes());

        for resource_id in resources_to_check {
            let current_state = current_states
                .get(resource_id)
                .copied()
                .unwrap_or(ResourceState::Undefined);

            // Determine what state this resource needs to be in for this pass
            let needs_read = pass.reads().contains(resource_id);
            let needs_write = pass.writes().contains(resource_id);
            let required_state = match (needs_read, needs_write) {
                (true, true) => ResourceState::ReadWrite,
                (true, false) => ResourceState::Read,
                (false, true) => ResourceState::Write,
                (false, false) => continue, // Resource not actually used in this pass
            };

            // Track state transitions (wgpu handles barriers automatically, but we track for validation)
            if current_state != required_state && current_state != ResourceState::Undefined {
                transitions.push((*resource_id, current_state, required_state));
            }
        }

        transitions
    }

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

        // Track current runtime state of each resource
        let mut current_states: HashMap<u64, ResourceState> = HashMap::new();

        // Execute passes in dependency order and collect command buffers
        for &pass_idx in &self.execution_order {
            let pass = &self.passes[pass_idx];

            // Track state transitions (wgpu handles barriers automatically based on usage flags,
            // but we track transitions for validation and documentation)
            let _transitions = self.track_state_transitions(&current_states, pass);

            // Note: wgpu automatically inserts barriers based on resource usage flags.
            // For buffers and textures, the driver handles synchronization automatically.
            // We track transitions here for:
            // 1. Validation (can detect invalid transitions)
            // 2. Future explicit barrier insertion if needed
            // 3. Debugging and optimization opportunities
            // The transitions vector can be used to log warnings, validate state changes,
            // or insert explicit barriers if needed in the future.

            // Execute the pass
            let command_buffer = pass.pass().execute(&ctx);
            command_buffers.push(command_buffer);

            // Update current states after pass execution
            for read_id in pass.reads() {
                let current = current_states
                    .get(read_id)
                    .copied()
                    .unwrap_or(ResourceState::Undefined);
                current_states.insert(*read_id, current.merge_with(ResourceState::Read));
            }
            for write_id in pass.writes() {
                let current = current_states
                    .get(write_id)
                    .copied()
                    .unwrap_or(ResourceState::Undefined);
                current_states.insert(*write_id, current.merge_with(ResourceState::Write));
            }
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
