//! Core GPU infrastructure for Triad.
//!
//! `triad-gpu` intentionally stays below application concerns:
//! it owns device/surface setup, resource registration, frame-graph execution,
//! and pipeline/shader construction from in-memory inputs.
//! Asset loading, file watching, and user-facing shader workflows belong in
//! higher-level crates.

use wgpu::{Instance, SurfaceConfiguration};
mod builder;
mod compute;
mod copy;
pub mod error;
mod frame_graph;
mod frame_slot;
mod indirect;
mod pipeline;
#[cfg(test)]
mod reference_pipeline;
mod render;
mod resource_registry;
mod surface;
mod spatial_grid;
#[cfg(test)]
mod test_util;
mod type_map;

// Re-export all error types at crate root for convenience
pub use error::{
    BindGroupError, BufferError, ComputePassError, CopyPassError, FrameGraphError, GpuError,
    PipelineError, ReadbackError, RenderPassError, RendererError, Result, ShaderError,
};

pub use builder::{
    BindGroupBuilder, BindingType, BufferBuilder, BufferUsage, ComputePipelineBuilder,
    DynamicBuffer, DynamicBufferBuilder, GpuBuffer, GpuBufferBuilder, ShaderModuleBuilder,
    ShaderSource, ShaderStage,
};
pub use compute::{ComputeDispatch, ComputePassBuilder};
pub use copy::{BufferCopy, CopyPassBuilder};
pub use frame_graph::{
    ExecutableFrameGraph, FrameGraph, Handle, Pass, PassBuilder, PassContext, ResourceType,
    TransientBufferDesc,
};
pub use frame_slot::{FrameBufferHandle, FrameTextureView};
pub use indirect::{DispatchIndirectArgs, DrawIndexedIndirectArgs, DrawIndirectArgs};
pub use pipeline::RenderPipelineBuilder;
pub use render::{ColorLoadOp, DepthLoadOp, RenderDraw, RenderPassBuilder};
pub use resource_registry::ResourceRegistry;
pub use surface::SurfaceWrapper;
pub use spatial_grid::{
    EntityPosition, SpatialGridConfig, SpatialGridError, SpatialGridGpu, SpatialGridParams,
    SpatialGridResult, total_cells,
};
pub use wgpu;

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
}

impl Renderer {
    pub async fn new() -> std::result::Result<Self, RendererError> {
        #[cfg(test)]
        let _gpu_test_guard = crate::test_util::gpu_test_lock();

        let instance = Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Renderer"),
                ..Default::default()
            })
            .await?;

        Ok(Self {
            device,
            queue,
            instance,
            adapter,
        })
    }

    #[cfg(test)]
    pub(crate) fn into_device_queue(self) -> (wgpu::Device, wgpu::Queue) {
        let Self { device, queue, .. } = self;
        (device, queue)
    }

    /// Get a reference to the device
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get a reference to the queue
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Get a reference to the instance
    pub fn instance(&self) -> &wgpu::Instance {
        &self.instance
    }

    /// Create a buffer builder for constructing GPU buffers
    pub fn create_buffer(&self) -> BufferBuilder<'_> {
        BufferBuilder::new(&self.device)
    }

    /// Create a shader module builder for constructing shader modules from source.
    pub fn create_shader_module(&self) -> ShaderModuleBuilder<'_> {
        ShaderModuleBuilder::new(&self.device)
    }

    /// Create a typed GPU buffer builder for storage, uniforms, indirect args, and readback.
    pub fn create_gpu_buffer<T: bytemuck::Pod>(&self) -> GpuBufferBuilder<'_, T> {
        GpuBufferBuilder::new(&self.device)
    }

    /// Create a bind group builder for constructing bind groups
    pub fn create_bind_group(&self) -> BindGroupBuilder<'_> {
        BindGroupBuilder::new(&self.device)
    }

    /// Create a render pipeline builder.
    pub fn create_render_pipeline(&self) -> RenderPipelineBuilder<'_> {
        RenderPipelineBuilder::new(&self.device)
    }

    /// Create a compute pipeline builder.
    pub fn create_compute_pipeline(&self) -> ComputePipelineBuilder<'_> {
        ComputePipelineBuilder::new(&self.device)
    }

    /// Create a DynamicBuffer for incremental updates
    ///
    /// DynamicBuffer pre-allocates capacity and supports efficient partial updates
    /// without recreating bind groups.
    pub fn create_dynamic_buffer<T: bytemuck::Pod>(&self) -> builder::DynamicBufferBuilder<'_, T> {
        builder::DynamicBufferBuilder::new(&self.device)
    }

    /// Start a fresh frame graph.
    pub fn create_frame_graph(&self) -> FrameGraph {
        FrameGraph::new()
    }

    /// Create a reusable compute pass builder for frame graph integration.
    pub fn create_compute_pass(&self, name: impl Into<String>) -> ComputePassBuilder {
        ComputePassBuilder::new(name)
    }

    /// Create a reusable render pass builder for frame graph integration.
    pub fn create_render_pass(&self, name: impl Into<String>) -> RenderPassBuilder {
        RenderPassBuilder::new(name)
    }

    /// Create a reusable copy pass builder for frame graph integration.
    pub fn create_copy_pass(&self, name: impl Into<String>) -> CopyPassBuilder {
        CopyPassBuilder::new(name)
    }

    /// Write data to a buffer
    pub fn write_buffer<T: bytemuck::Pod>(
        &self,
        buffer: Handle<wgpu::Buffer>,
        data: &[T],
        registry: &ResourceRegistry,
    ) -> std::result::Result<(), BufferError> {
        let buffer_ref = registry.get(buffer).ok_or(BufferError::NotFound)?;
        self.queue
            .write_buffer(buffer_ref, 0, bytemuck::cast_slice(data));
        Ok(())
    }

    /// Write data to a buffer at a specified byte offset.
    ///
    /// This enables incremental buffer updates without recreating the entire buffer.
    ///
    /// # Errors
    /// - `BufferError::NotFound` if buffer handle is invalid
    /// - `BufferError::InvalidOffset` if offset + data size exceeds buffer size
    pub fn write_buffer_offset<T: bytemuck::Pod>(
        &self,
        buffer: Handle<wgpu::Buffer>,
        offset: u64,
        data: &[T],
        registry: &ResourceRegistry,
    ) -> std::result::Result<(), BufferError> {
        let buffer_ref = registry.get(buffer).ok_or(BufferError::NotFound)?;
        let data_bytes = bytemuck::cast_slice::<T, u8>(data);
        let data_size = data_bytes.len() as u64;

        if offset + data_size > buffer_ref.size() {
            return Err(BufferError::InvalidOffset {
                offset,
                data_size,
                buffer_size: buffer_ref.size(),
            });
        }

        self.queue.write_buffer(buffer_ref, offset, data_bytes);
        Ok(())
    }

    /// Read the full contents of a MAP_READ buffer back to the CPU as typed data.
    pub fn read_buffer<T: bytemuck::Pod>(
        &self,
        buffer: Handle<wgpu::Buffer>,
        registry: &ResourceRegistry,
    ) -> std::result::Result<Vec<T>, ReadbackError> {
        use std::sync::mpsc;

        let buffer_ref = registry.get(buffer).ok_or(ReadbackError::BufferNotFound)?;
        let buffer_size = buffer_ref.size();
        let element_size = std::mem::size_of::<T>();

        if buffer_size as usize % element_size != 0 {
            return Err(ReadbackError::BufferSizeNotAligned {
                buffer_size,
                element_size,
            });
        }

        let slice = buffer_ref.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        let map_result = rx.recv().map_err(|_| ReadbackError::MapChannelClosed)?;
        map_result?;

        let data = {
            let mapped = slice.get_mapped_range();
            bytemuck::cast_slice::<u8, T>(&mapped).to_vec()
        };
        buffer_ref.unmap();

        Ok(data)
    }

    pub fn create_surface(
        &self,
        surface: wgpu::Surface<'static>,
        width: u32,
        height: u32,
    ) -> std::result::Result<SurfaceWrapper, RendererError> {
        // Validate width and height are non-zero
        if width == 0 || height == 0 {
            return Err(RendererError::InvalidDimensions { width, height });
        }

        let caps = surface.get_capabilities(&self.adapter);

        // Check if formats array is empty
        if caps.formats.is_empty() {
            return Err(RendererError::NoSupportedFormats);
        }

        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        // Check if present_modes array is empty
        if caps.present_modes.is_empty() {
            return Err(RendererError::NoSupportedPresentModes);
        }

        // Check if alpha_modes array is empty
        if caps.alpha_modes.is_empty() {
            return Err(RendererError::NoSupportedAlphaModes);
        }

        let config = SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: caps.present_modes[0],
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&self.device, &config);
        Ok(SurfaceWrapper::new(surface, config))
    }

    pub fn create_surface_with_mode(
        &self,
        surface: wgpu::Surface<'static>,
        width: u32,
        height: u32,
        present_mode: wgpu::PresentMode,
    ) -> std::result::Result<SurfaceWrapper, RendererError> {
        // Validate width and height are non-zero
        if width == 0 || height == 0 {
            return Err(RendererError::InvalidDimensions { width, height });
        }

        let caps = surface.get_capabilities(&self.adapter);

        // Check if formats array is empty
        if caps.formats.is_empty() {
            return Err(RendererError::NoSupportedFormats);
        }

        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        // Check if alpha_modes array is empty
        if caps.alpha_modes.is_empty() {
            return Err(RendererError::NoSupportedAlphaModes);
        }

        let config = SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&self.device, &config);
        Ok(SurfaceWrapper::new(surface, config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pollster::FutureExt;

    #[test]
    fn test_renderer_creation() {
        let renderer = Renderer::new().block_on();
        assert!(renderer.is_ok());
    }

    #[test]
    fn test_renderer_device_access() {
        let renderer = Renderer::new()
            .block_on()
            .expect("Failed to create renderer");
        let device = renderer.device();
        assert!(device.limits().max_buffer_size > 0);
    }

    #[test]
    fn test_renderer_queue_access() {
        let renderer = Renderer::new()
            .block_on()
            .expect("Failed to create renderer");
        let _queue = renderer.queue();
        // Queue is accessible, test passes if no panic
    }

    #[test]
    fn test_renderer_shader_builder_access() {
        let renderer = Renderer::new()
            .block_on()
            .expect("Failed to create renderer");
        let _builder = renderer.create_shader_module();
    }

    #[test]
    fn test_renderer_gpu_buffer_builder_access() {
        let renderer = Renderer::new()
            .block_on()
            .expect("Failed to create renderer");
        let _builder = renderer.create_gpu_buffer::<u32>();
    }

    #[test]
    fn test_renderer_compute_pass_builder_access() {
        let renderer = Renderer::new()
            .block_on()
            .expect("Failed to create renderer");
        let _builder = renderer.create_compute_pass("simulate");
    }

    #[test]
    fn test_renderer_create_buffer() {
        let renderer = Renderer::new()
            .block_on()
            .expect("Failed to create renderer");
        let builder = renderer.create_buffer();
        // Builder created successfully
        assert!(std::mem::size_of_val(&builder) > 0);
    }

    #[test]
    fn test_renderer_create_bind_group() {
        let renderer = Renderer::new()
            .block_on()
            .expect("Failed to create renderer");
        let builder = renderer.create_bind_group();
        // Builder created successfully
        assert!(std::mem::size_of_val(&builder) > 0);
    }

    #[test]
    fn test_renderer_write_buffer() {
        let renderer = Renderer::new()
            .block_on()
            .expect("Failed to create renderer");
        let mut registry = ResourceRegistry::default();

        // Create a buffer
        let buffer_handle = renderer
            .create_buffer()
            .size(256)
            .usage(BufferUsage::Uniform)
            .build(&mut registry)
            .expect("Failed to create buffer");

        // Write data to buffer
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct TestData {
            value: f32,
        }

        let data = vec![TestData { value: 42.0 }];
        let result = renderer.write_buffer(buffer_handle, &data, &registry);
        assert!(result.is_ok());
    }

    #[test]
    fn test_renderer_write_buffer_not_found() {
        let renderer = Renderer::new()
            .block_on()
            .expect("Failed to create renderer");
        let registry = ResourceRegistry::default();

        // Try to write to a non-existent buffer
        let fake_handle = crate::frame_graph::resource::Handle::<wgpu::Buffer>::next();

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct TestData {
            value: f32,
        }

        let data = vec![TestData { value: 42.0 }];
        let result = renderer.write_buffer(fake_handle, &data, &registry);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), BufferError::NotFound));
    }

    #[test]
    fn test_renderer_integration_buffer_build_and_write() {
        let renderer = Renderer::new()
            .block_on()
            .expect("Failed to create renderer");
        let mut registry = ResourceRegistry::default();

        // Create buffer with data
        // Note: write_buffer requires COPY_DST usage. Uniform buffers include COPY_DST by default.
        // Storage buffers are typically written via compute shaders, not queue.write_buffer.
        let initial_data: [u8; 64] = [1; 64];
        let buffer_handle = renderer
            .create_buffer()
            .label("test_buffer")
            .with_data(&initial_data)
            .usage(BufferUsage::Uniform)
            .build(&mut registry)
            .expect("Failed to create buffer");

        // Verify buffer exists
        let buffer = registry.get(buffer_handle).expect("Buffer not found");
        assert_eq!(buffer.size(), 64);

        // Write new data
        let new_data: [u8; 64] = [2; 64];
        renderer
            .write_buffer(buffer_handle, &new_data, &registry)
            .expect("Failed to write buffer");
    }

    #[test]
    fn test_write_buffer_offset_success() {
        let renderer = Renderer::new()
            .block_on()
            .expect("Failed to create renderer");
        let mut registry = ResourceRegistry::default();

        // Create buffer with COPY_DST (Uniform includes it)
        let buffer_handle = renderer
            .create_buffer()
            .label("test_offset_buffer")
            .size(64)
            .usage(BufferUsage::Uniform)
            .build(&mut registry)
            .expect("Failed to create buffer");

        // Write at offset 0
        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let result = renderer.write_buffer_offset(buffer_handle, 0, &data, &registry);
        assert!(result.is_ok());

        // Write at offset 16 (after the first 4 floats)
        let result = renderer.write_buffer_offset(buffer_handle, 16, &data, &registry);
        assert!(result.is_ok());
    }

    #[test]
    fn test_write_buffer_offset_invalid() {
        let renderer = Renderer::new()
            .block_on()
            .expect("Failed to create renderer");
        let mut registry = ResourceRegistry::default();

        let buffer_handle = renderer
            .create_buffer()
            .size(64)
            .usage(BufferUsage::Uniform)
            .build(&mut registry)
            .expect("Failed to create buffer");

        // Try to write past buffer end: offset 60 + 16 bytes (4 floats) = 76 > 64
        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let result = renderer.write_buffer_offset(buffer_handle, 60, &data, &registry);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            BufferError::InvalidOffset { .. }
        ));
    }

    #[test]
    fn test_write_buffer_offset_not_found() {
        let renderer = Renderer::new()
            .block_on()
            .expect("Failed to create renderer");
        let registry = ResourceRegistry::default();

        let fake_handle = crate::frame_graph::resource::Handle::<wgpu::Buffer>::next();
        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let result = renderer.write_buffer_offset(fake_handle, 0, &data, &registry);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), BufferError::NotFound));
    }

    #[test]
    fn test_dynamic_buffer_creation_with_capacity() {
        let renderer = Renderer::new()
            .block_on()
            .expect("Failed to create renderer");
        let mut registry = ResourceRegistry::default();

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
        struct TestElement {
            value: f32,
            _pad: [f32; 3],
        }

        let buf: DynamicBuffer<TestElement> = renderer
            .create_dynamic_buffer()
            .label("test_dynamic")
            .capacity(100)
            .build(&mut registry)
            .expect("Failed to create dynamic buffer");

        assert_eq!(buf.len(), 0);
        assert_eq!(buf.capacity(), 100);
        assert!(buf.is_empty());

        // Verify underlying buffer exists
        let buffer = registry.get(buf.buffer()).expect("Buffer not found");
        assert_eq!(
            buffer.size(),
            (100 * std::mem::size_of::<TestElement>()) as u64
        );
    }

    #[test]
    fn test_dynamic_buffer_creation_with_data() {
        let renderer = Renderer::new()
            .block_on()
            .expect("Failed to create renderer");
        let mut registry = ResourceRegistry::default();

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
        struct TestElement {
            value: f32,
            _pad: [f32; 3],
        }

        let initial_data = vec![TestElement::default(); 10];
        let buf: DynamicBuffer<TestElement> = renderer
            .create_dynamic_buffer()
            .label("test_dynamic_with_data")
            .with_data(&initial_data)
            .build(&mut registry)
            .expect("Failed to create dynamic buffer");

        assert_eq!(buf.len(), 10);
        assert_eq!(buf.capacity(), 10);
    }

    #[test]
    fn test_dynamic_buffer_creation_with_data_and_extra_capacity() {
        let renderer = Renderer::new()
            .block_on()
            .expect("Failed to create renderer");
        let mut registry = ResourceRegistry::default();

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
        struct TestElement {
            value: f32,
            _pad: [f32; 3],
        }

        let initial_data = vec![TestElement::default(); 10];
        let buf: DynamicBuffer<TestElement> = renderer
            .create_dynamic_buffer()
            .label("test_dynamic_extra_cap")
            .capacity(50) // Extra capacity beyond initial data
            .with_data(&initial_data)
            .build(&mut registry)
            .expect("Failed to create dynamic buffer");

        assert_eq!(buf.len(), 10);
        assert_eq!(buf.capacity(), 50);
    }

    #[test]
    fn test_dynamic_buffer_push() {
        let renderer = Renderer::new()
            .block_on()
            .expect("Failed to create renderer");
        let mut registry = ResourceRegistry::default();

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
        struct TestElement {
            value: f32,
            _pad: [f32; 3],
        }

        let mut buf: DynamicBuffer<TestElement> = renderer
            .create_dynamic_buffer()
            .label("test_push")
            .capacity(100)
            .build(&mut registry)
            .expect("Failed to create dynamic buffer");

        let elements = vec![
            TestElement {
                value: 1.0,
                _pad: [0.0; 3]
            };
            10
        ];
        let idx = buf
            .push(&renderer, &registry, &elements)
            .expect("Failed to push");

        assert_eq!(idx, 0);
        assert_eq!(buf.len(), 10);

        // Push more elements
        let more_elements = vec![
            TestElement {
                value: 2.0,
                _pad: [0.0; 3]
            };
            5
        ];
        let idx = buf
            .push(&renderer, &registry, &more_elements)
            .expect("Failed to push");

        assert_eq!(idx, 10);
        assert_eq!(buf.len(), 15);
    }

    #[test]
    fn test_dynamic_buffer_update_at() {
        let renderer = Renderer::new()
            .block_on()
            .expect("Failed to create renderer");
        let mut registry = ResourceRegistry::default();

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
        struct TestElement {
            value: f32,
            _pad: [f32; 3],
        }

        let initial = vec![TestElement::default(); 10];
        let mut buf: DynamicBuffer<TestElement> = renderer
            .create_dynamic_buffer()
            .with_data(&initial)
            .build(&mut registry)
            .expect("Failed to create dynamic buffer");

        let updated = TestElement {
            value: 42.0,
            _pad: [0.0; 3],
        };
        let result = buf.update_at(&renderer, &registry, 5, &updated);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dynamic_buffer_update_at_out_of_bounds() {
        let renderer = Renderer::new()
            .block_on()
            .expect("Failed to create renderer");
        let mut registry = ResourceRegistry::default();

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
        struct TestElement {
            value: f32,
            _pad: [f32; 3],
        }

        let initial = vec![TestElement::default(); 10];
        let buf: DynamicBuffer<TestElement> = renderer
            .create_dynamic_buffer()
            .with_data(&initial)
            .build(&mut registry)
            .expect("Failed to create dynamic buffer");

        let updated = TestElement {
            value: 42.0,
            _pad: [0.0; 3],
        };
        // Index 10 is out of bounds (len is 10, indices 0-9 valid)
        let result = buf.update_at(&renderer, &registry, 10, &updated);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            BufferError::CapacityExceeded { .. }
        ));
    }

    #[test]
    fn test_dynamic_buffer_capacity_exceeded() {
        let renderer = Renderer::new()
            .block_on()
            .expect("Failed to create renderer");
        let mut registry = ResourceRegistry::default();

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
        struct TestElement {
            value: f32,
            _pad: [f32; 3],
        }

        let mut buf: DynamicBuffer<TestElement> = renderer
            .create_dynamic_buffer()
            .capacity(10)
            .build(&mut registry)
            .expect("Failed to create dynamic buffer");

        // Try to push more than capacity
        let elements = vec![TestElement::default(); 20];
        let result = buf.push(&renderer, &registry, &elements);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            BufferError::CapacityExceeded {
                requested: 20,
                capacity: 10
            }
        ));
    }

    #[test]
    fn test_dynamic_buffer_clear() {
        let renderer = Renderer::new()
            .block_on()
            .expect("Failed to create renderer");
        let mut registry = ResourceRegistry::default();

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
        struct TestElement {
            value: f32,
            _pad: [f32; 3],
        }

        let initial = vec![TestElement::default(); 10];
        let mut buf: DynamicBuffer<TestElement> = renderer
            .create_dynamic_buffer()
            .with_data(&initial)
            .build(&mut registry)
            .expect("Failed to create dynamic buffer");

        assert_eq!(buf.len(), 10);
        buf.clear();
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
        // Capacity unchanged
        assert_eq!(buf.capacity(), 10);
    }

    #[test]
    fn test_dynamic_buffer_set_len() {
        let renderer = Renderer::new()
            .block_on()
            .expect("Failed to create renderer");
        let mut registry = ResourceRegistry::default();

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
        struct TestElement {
            value: f32,
            _pad: [f32; 3],
        }

        let mut buf: DynamicBuffer<TestElement> = renderer
            .create_dynamic_buffer()
            .capacity(100)
            .build(&mut registry)
            .expect("Failed to create dynamic buffer");

        // Set length (e.g., after GPU compute shader writes data)
        let result = buf.set_len(50);
        assert!(result.is_ok());
        assert_eq!(buf.len(), 50);

        // Try to set beyond capacity
        let result = buf.set_len(200);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            BufferError::CapacityExceeded { .. }
        ));
    }

    #[test]
    fn test_dynamic_buffer_missing_size_or_data() {
        let renderer = Renderer::new()
            .block_on()
            .expect("Failed to create renderer");
        let mut registry = ResourceRegistry::default();

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default, Debug)]
        struct TestElement {
            value: f32,
        }

        // Neither capacity nor data specified
        let result: std::result::Result<DynamicBuffer<TestElement>, BufferError> = renderer
            .create_dynamic_buffer()
            .label("test")
            .build(&mut registry);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            BufferError::MissingSizeOrData
        ));
    }

    #[test]
    fn test_storage_writable_usage() {
        let renderer = Renderer::new()
            .block_on()
            .expect("Failed to create renderer");
        let mut registry = ResourceRegistry::default();

        // Create a StorageWritable buffer
        let buffer_handle = renderer
            .create_buffer()
            .label("storage_writable")
            .size(256)
            .usage(BufferUsage::StorageWritable)
            .build(&mut registry)
            .expect("Failed to create buffer");

        // Should be able to write to it
        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let result = renderer.write_buffer_offset(buffer_handle, 0, &data, &registry);
        assert!(result.is_ok());
    }

    #[test]
    fn test_copy_pass_and_typed_readback() {
        let renderer = match Renderer::new().block_on() {
            Ok(renderer) => renderer,
            Err(err) => {
                eprintln!("skipping copy/readback test: {err}");
                return;
            }
        };
        let mut registry = ResourceRegistry::default();

        let source_data: [u32; 4] = [7, 11, 13, 17];
        let source = renderer
            .create_gpu_buffer::<u32>()
            .label("copy_source")
            .with_data(&source_data)
            .usage(BufferUsage::CopySrc)
            .build(&mut registry)
            .expect("source buffer");

        let readback = renderer
            .create_gpu_buffer::<u32>()
            .label("copy_readback")
            .capacity(source_data.len())
            .usage(BufferUsage::Readback)
            .build(&mut registry)
            .expect("readback buffer");

        let pass = renderer
            .create_copy_pass("CopyToReadback")
            .copy_buffer(
                source.handle(),
                readback.handle(),
                std::mem::size_of_val(&source_data) as u64,
            )
            .build()
            .expect("copy pass");

        let mut graph = FrameGraph::new();
        graph.add_pass(pass);
        let mut executable = graph.build().expect("frame graph");
        let command_buffers =
            executable.execute_no_submit(renderer.device(), renderer.queue(), &registry);
        renderer.queue().submit(command_buffers);

        let copied = renderer
            .read_buffer::<u32>(readback.handle(), &registry)
            .expect("typed readback");
        assert_eq!(copied, source_data);
    }
}
