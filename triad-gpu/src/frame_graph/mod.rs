mod execution;
pub mod pass;
pub mod resource;

use crate::error::FrameGraphError;
use crate::frame_graph::pass::PassNode;
use crate::frame_graph::resource::{ResourceInfo, ResourceState};
use crate::resource_registry::ResourceRegistry;
use std::collections::HashMap;
use tracing::{debug_span, instrument};

pub use pass::{Pass, PassBuilder, PassContext};
pub use resource::{Handle, HandleId, ResourceType, TransientBufferDesc};

pub type SurfaceId = u64;

#[derive(Default)]
pub struct FrameGraph {
    passes: Vec<PassNode>,
    resource_info: HashMap<HandleId, ResourceInfo>,
    transient_buffers: HashMap<HandleId, TransientBufferDesc>,
    surface_handles: Vec<SurfaceId>, // Surface handle IDs (surfaces tracked separately)
}

#[cfg(test)]
impl FrameGraph {
    // Test helper to access resource_info for testing
    pub(crate) fn resource_info(&self) -> &HashMap<HandleId, ResourceInfo> {
        &self.resource_info
    }
}

impl FrameGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_resource<T: ResourceType>(&mut self, handle: Handle<T>) -> &mut Self {
        self.resource_info
            .entry(handle.id())
            .or_insert_with(ResourceInfo::new);
        self
    }

    pub fn create_transient_buffer(&mut self, desc: TransientBufferDesc) -> Handle<wgpu::Buffer> {
        let handle = Handle::next();
        self.resource_info
            .entry(handle.id())
            .or_insert_with(ResourceInfo::new);
        self.transient_buffers.insert(handle.id(), desc);
        handle
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
    pub fn build(self) -> Result<ExecutableFrameGraph, FrameGraphError> {
        self.build_with_cached_order(None)
    }

    /// Build the frame graph, optionally using a cached execution order.
    /// If the cached order is provided and the number of passes matches, it will be reused
    /// to avoid the expensive topological sort.
    pub fn build_with_cached_order(
        mut self,
        cached_execution_order: Option<&[usize]>,
    ) -> Result<ExecutableFrameGraph, FrameGraphError> {
        // Use cached execution order if provided and valid
        let execution_order = if let Some(cached) = cached_execution_order {
            // Validate cached order: must have same length as passes
            if cached.len() == self.passes.len() {
                cached.to_vec()
            } else {
                // Cached order is invalid (structure changed), recompute
                execution::topological_sort(&self.passes)?
            }
        } else {
            // No cache, compute execution order
            execution::topological_sort(&self.passes)?
        };

        // Update resource info with first/last used pass indices
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
            transient_buffers: self.transient_buffers,
            surface_handles: self.surface_handles,
        })
    }
}

/// Executable frame graph ready for execution
pub struct ExecutableFrameGraph {
    passes: Vec<PassNode>,
    execution_order: Vec<usize>,
    transient_buffers: HashMap<HandleId, TransientBufferDesc>,
    surface_handles: Vec<u64>, // Surface handle IDs (surfaces tracked separately)
}

impl ExecutableFrameGraph {
    /// Execute the frame graph and return command buffers (without submitting)
    /// This allows batching with other command buffers (e.g., UI) for a single submission
    /// Note: Passes may use the queue for immediate operations, but final submission is deferred
    #[instrument(skip(self, device, queue, resources), name = "fg_execute", fields(pass_count = self.execution_order.len()))]
    pub fn execute_no_submit(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resources: &ResourceRegistry,
    ) -> Vec<wgpu::CommandBuffer> {
        self.execute_internal(device, queue, resources, false)
    }

    /// Execute the frame graph
    /// Collects command buffers from passes and submits them in optimal order
    #[instrument(skip(self, device, queue, resources), name = "fg_execute", fields(pass_count = self.execution_order.len()))]
    pub fn execute(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resources: &ResourceRegistry,
    ) {
        let command_buffers = self.execute_internal(device, queue, resources, true);

        // Submit all command buffers in a single batch for optimal performance
        {
            let _span = debug_span!("queue_submit", count = command_buffers.len()).entered();
            queue.submit(command_buffers);
        }
    }

    fn execute_internal(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resources: &ResourceRegistry,
        _submit: bool,
    ) -> Vec<wgpu::CommandBuffer> {
        let mut command_buffers = Vec::with_capacity(self.execution_order.len());
        let transient_buffers = self.create_transient_buffers(device);
        let ctx = pass::PassContext {
            device,
            queue,
            resources,
            transient_buffers: &transient_buffers,
        };

        // Track current runtime state of each resource
        let mut current_states: HashMap<u64, ResourceState> =
            HashMap::with_capacity(self.execution_order.len());

        // Execute passes in dependency order and collect command buffers
        for &pass_idx in &self.execution_order {
            let pass = &self.passes[pass_idx];

            // Execute the pass
            let command_buffer = {
                let _span = debug_span!("pass", name = pass.name(), idx = pass_idx).entered();
                pass.pass().execute(&ctx)
            };
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

        command_buffers
    }

    pub fn surface_count(&self) -> usize {
        self.surface_handles.len()
    }

    /// Get the execution order of passes.
    /// This can be cached and reused when the frame graph structure hasn't changed.
    pub fn execution_order(&self) -> &[usize] {
        &self.execution_order
    }

    fn create_transient_buffers(&self, device: &wgpu::Device) -> HashMap<HandleId, wgpu::Buffer> {
        let mut buffers = HashMap::with_capacity(self.transient_buffers.len());
        for (&handle_id, desc) in &self.transient_buffers {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: desc.label(),
                size: desc.size(),
                usage: desc.usage(),
                mapped_at_creation: desc.is_mapped_at_creation(),
            });
            buffers.insert(handle_id, buffer);
        }
        buffers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_graph::pass::Pass;
    use crate::resource_registry::ResourceRegistry;
    use crate::test_util::create_test_device;
    use pollster::FutureExt;

    // Mock pass for testing
    struct MockPass {
        name: String,
        reads: Vec<u64>,
        writes: Vec<u64>,
    }

    impl Pass for MockPass {
        fn name(&self) -> &str {
            &self.name
        }

        fn execute(&self, ctx: &PassContext) -> wgpu::CommandBuffer {
            let encoder = ctx.create_command_encoder(Some(&self.name));
            encoder.finish()
        }
    }

    #[test]
    fn test_frame_graph_resource_registration() {
        let mut frame_graph = FrameGraph::default();
        let buffer_handle = Handle::<wgpu::Buffer>::next();
        let texture_handle = Handle::<wgpu::Texture>::next();

        frame_graph
            .register_resource(buffer_handle)
            .register_resource(texture_handle);

        // Resources should be registered
        assert!(frame_graph.resource_info.contains_key(&buffer_handle.id()));
        assert!(frame_graph.resource_info.contains_key(&texture_handle.id()));
    }

    #[test]
    fn test_frame_graph_transient_buffer_registration() {
        let mut frame_graph = FrameGraph::default();
        let handle = frame_graph.create_transient_buffer(
            TransientBufferDesc::new(256, wgpu::BufferUsages::STORAGE)
                .with_label("transient_scratch"),
        );

        assert!(frame_graph.resource_info.contains_key(&handle.id()));
        let desc = frame_graph
            .transient_buffers
            .get(&handle.id())
            .expect("transient descriptor should be registered");
        assert_eq!(desc.size(), 256);
        assert_eq!(desc.label(), Some("transient_scratch"));
    }

    #[test]
    fn test_frame_graph_preserves_declaration_order_for_conflicting_passes() {
        let (device, _queue) = create_test_device().block_on();
        let mut registry = ResourceRegistry::default();

        let buffer_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buffer_a"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        let handle_a = registry.insert(buffer_a);

        let buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buffer_b"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        let handle_b = registry.insert(buffer_b);

        let mut frame_graph = FrameGraph::default();

        // Pass 1 conflicts with Pass 2, but declaration order should win.
        let mut pass1_builder = PassBuilder::new("Pass1");
        pass1_builder.write(handle_a);
        pass1_builder.read(handle_b);
        let pass1 = pass1_builder.with_pass(Box::new(MockPass {
            name: "Pass1".to_string(),
            reads: vec![handle_b.id()],
            writes: vec![handle_a.id()],
        }));

        // Pass 2 also conflicts with Pass 1.
        let mut pass2_builder = PassBuilder::new("Pass2");
        pass2_builder.write(handle_b);
        pass2_builder.read(handle_a);
        let pass2 = pass2_builder.with_pass(Box::new(MockPass {
            name: "Pass2".to_string(),
            reads: vec![handle_a.id()],
            writes: vec![handle_b.id()],
        }));

        frame_graph
            .register_resource(handle_a)
            .register_resource(handle_b)
            .add_pass(pass1)
            .add_pass(pass2);

        let executable = frame_graph.build().expect("frame graph should build");
        assert_eq!(executable.execution_order(), &[0, 1]);
    }

    #[test]
    fn test_frame_graph_cached_execution_order() {
        let (device, _queue) = create_test_device().block_on();
        let mut registry = ResourceRegistry::default();

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buffer"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        let handle = registry.insert(buffer);

        // Build frame graph first time
        let mut frame_graph1 = FrameGraph::default();
        let mut pass1_builder = PassBuilder::new("Pass1");
        pass1_builder.write(handle);
        let pass1 = pass1_builder.with_pass(Box::new(MockPass {
            name: "Pass1".to_string(),
            reads: vec![],
            writes: vec![handle.id()],
        }));
        frame_graph1.register_resource(handle).add_pass(pass1);
        let executable1 = frame_graph1.build().expect("Failed to build");
        let execution_order1 = executable1.execution_order().to_vec();

        // Build frame graph second time with same structure
        let mut frame_graph2 = FrameGraph::default();
        let mut pass2_builder = PassBuilder::new("Pass1");
        pass2_builder.write(handle);
        let pass2 = pass2_builder.with_pass(Box::new(MockPass {
            name: "Pass1".to_string(),
            reads: vec![],
            writes: vec![handle.id()],
        }));
        frame_graph2.register_resource(handle).add_pass(pass2);

        // Use cached execution order
        let executable2 = frame_graph2
            .build_with_cached_order(Some(&execution_order1))
            .expect("Failed to build with cache");

        // Execution orders should match
        assert_eq!(executable1.execution_order(), executable2.execution_order());
    }

    #[test]
    fn test_frame_graph_cached_order_invalid_length() {
        let (device, _queue) = create_test_device().block_on();
        let mut registry = ResourceRegistry::default();

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buffer"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        let handle = registry.insert(buffer);

        // Create invalid cached order (wrong length)
        let invalid_cache = vec![0, 1, 2]; // 3 passes, but we'll only have 1

        let mut frame_graph = FrameGraph::default();
        let mut pass_builder = PassBuilder::new("Pass1");
        pass_builder.write(handle);
        let pass = pass_builder.with_pass(Box::new(MockPass {
            name: "Pass1".to_string(),
            reads: vec![],
            writes: vec![handle.id()],
        }));
        frame_graph.register_resource(handle).add_pass(pass);

        // Should ignore invalid cache and recompute
        let executable = frame_graph
            .build_with_cached_order(Some(&invalid_cache))
            .expect("Should build successfully even with invalid cache");

        // Should have correct execution order (1 pass, so order is [0])
        assert_eq!(executable.execution_order().len(), 1);
        assert_eq!(executable.execution_order()[0], 0);
    }

    #[test]
    fn test_frame_graph_independent_passes() {
        let (device, _queue) = create_test_device().block_on();
        let mut registry = ResourceRegistry::default();

        let buffer_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buffer_a"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        let handle_a = registry.insert(buffer_a);

        let buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buffer_b"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        let handle_b = registry.insert(buffer_b);

        let mut frame_graph = FrameGraph::default();

        // Two independent passes (no dependencies)
        let mut pass1_builder = PassBuilder::new("Pass1");
        pass1_builder.write(handle_a);
        let pass1 = pass1_builder.with_pass(Box::new(MockPass {
            name: "Pass1".to_string(),
            reads: vec![],
            writes: vec![handle_a.id()],
        }));

        let mut pass2_builder = PassBuilder::new("Pass2");
        pass2_builder.write(handle_b);
        let pass2 = pass2_builder.with_pass(Box::new(MockPass {
            name: "Pass2".to_string(),
            reads: vec![],
            writes: vec![handle_b.id()],
        }));

        frame_graph
            .register_resource(handle_a)
            .register_resource(handle_b)
            .add_pass(pass1)
            .add_pass(pass2);

        let executable = frame_graph.build().expect("Failed to build frame graph");
        let execution_order = executable.execution_order();

        // Both passes should be in execution order (order doesn't matter for independent passes)
        assert_eq!(execution_order.len(), 2);

        // All indices should be unique and in range [0, 1]
        let mut seen = std::collections::HashSet::new();
        for &idx in execution_order {
            assert!(idx < 2, "Execution order index out of range");
            assert!(seen.insert(idx), "Duplicate index in execution order");
        }
    }

    #[test]
    fn test_frame_graph_execution_order_access() {
        let (device, _queue) = create_test_device().block_on();
        let mut registry = ResourceRegistry::default();

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buffer"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        let handle = registry.insert(buffer);

        let mut frame_graph = FrameGraph::default();
        let mut pass_builder = PassBuilder::new("Pass1");
        pass_builder.write(handle);
        let pass = pass_builder.with_pass(Box::new(MockPass {
            name: "Pass1".to_string(),
            reads: vec![],
            writes: vec![handle.id()],
        }));
        frame_graph.register_resource(handle).add_pass(pass);

        let executable = frame_graph.build().expect("Failed to build frame graph");

        // Test that execution_order() method works
        let order = executable.execution_order();
        assert_eq!(order.len(), 1);
        assert_eq!(order[0], 0);
    }

    struct TransientBufferPass {
        name: String,
        buffer: Handle<wgpu::Buffer>,
        expected_size: u64,
    }

    impl Pass for TransientBufferPass {
        fn name(&self) -> &str {
            &self.name
        }

        fn execute(&self, ctx: &PassContext) -> wgpu::CommandBuffer {
            let buffer = ctx
                .get_buffer(self.buffer)
                .expect("transient buffer should be materialized during execution");
            assert_eq!(buffer.size(), self.expected_size);

            let mut encoder = ctx.create_command_encoder(Some(&self.name));
            encoder.clear_buffer(buffer, 0, None);
            encoder.finish()
        }
    }

    #[test]
    fn test_frame_graph_executes_with_transient_buffer() {
        let renderer = match crate::Renderer::new().block_on() {
            Ok(renderer) => renderer,
            Err(err) => {
                eprintln!("skipping transient frame graph execution test: {err}");
                return;
            }
        };
        let registry = ResourceRegistry::default();
        let mut frame_graph = FrameGraph::default();
        let transient = frame_graph.create_transient_buffer(
            TransientBufferDesc::new(
                128,
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            )
            .with_label("transient_exec"),
        );

        let mut pass_builder = PassBuilder::new("TransientPass");
        pass_builder.write(transient);
        let pass = pass_builder.with_pass(Box::new(TransientBufferPass {
            name: "TransientPass".to_string(),
            buffer: transient,
            expected_size: 128,
        }));

        frame_graph.add_pass(pass);
        let mut executable = frame_graph.build().expect("frame graph should build");
        let command_buffers =
            executable.execute_no_submit(renderer.device(), renderer.queue(), &registry);

        assert_eq!(command_buffers.len(), 1);
    }
}
