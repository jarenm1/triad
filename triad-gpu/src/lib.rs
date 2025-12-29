use wgpu::{Instance, SurfaceConfiguration};
mod builder;
mod frame_graph;
mod pipeline;
pub mod ply_loader;
mod resource_registry;
mod surface;
mod type_map;
mod types;

pub use builder::{BindGroupBuilder, BindingType, BufferBuilder, BufferUsage, ShaderStage};
pub use frame_graph::{FrameGraph, Handle, Pass, PassBuilder, PassContext, ResourceType};
pub use pipeline::{PipelineBuildError, RenderPipelineBuilder};
pub use resource_registry::ResourceRegistry;
pub use surface::SurfaceWrapper;
pub use types::{CameraUniforms, GaussianPoint, TrianglePrimitive};
pub use wgpu;

#[derive(Debug, thiserror::Error)]
pub enum RendererError {
    #[error("Request Adapter Error: {0}")]
    RequestAdapterError(#[from] wgpu::RequestAdapterError),
    #[error("Request Device Error: {0}")]
    RequestDeviceError(#[from] wgpu::RequestDeviceError),
    #[error("Surface Error: {0}")]
    SurfaceError(#[from] wgpu::SurfaceError),
    #[error("Surface Configuration Error: {0}")]
    SurfaceConfigurationError(String),
}

#[derive(Debug, thiserror::Error)]
pub enum BufferWriteError {
    #[error("Buffer not found in registry")]
    BufferNotFound,
}

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
}

impl Renderer {
    pub async fn new() -> Result<Self, RendererError> {
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
    pub fn create_buffer(&self) -> BufferBuilder {
        BufferBuilder::new(&self.device)
    }

    /// Create a bind group builder for constructing bind groups
    pub fn create_bind_group<'a>(&'a self, registry: &'a ResourceRegistry) -> BindGroupBuilder<'a> {
        BindGroupBuilder::new(&self.device, registry)
    }

    /// Write data to a buffer
    pub fn write_buffer<T: bytemuck::Pod>(
        &self,
        buffer: Handle<wgpu::Buffer>,
        data: &[T],
        registry: &ResourceRegistry,
    ) -> Result<(), BufferWriteError> {
        let buffer_ref = registry
            .get(buffer)
            .ok_or(BufferWriteError::BufferNotFound)?;
        self.queue
            .write_buffer(buffer_ref, 0, bytemuck::cast_slice(data));
        Ok(())
    }

    pub fn create_surface(
        &self,
        surface: wgpu::Surface<'static>,
        width: u32,
        height: u32,
    ) -> Result<SurfaceWrapper, RendererError> {
        // Validate width and height are non-zero
        if width == 0 || height == 0 {
            return Err(RendererError::SurfaceConfigurationError(format!(
                "Invalid surface dimensions: {}x{}",
                width, height
            )));
        }

        let caps = surface.get_capabilities(&self.adapter);

        // Check if formats array is empty
        if caps.formats.is_empty() {
            return Err(RendererError::SurfaceConfigurationError(
                "No supported surface formats available".to_string(),
            ));
        }

        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        // Check if present_modes array is empty
        if caps.present_modes.is_empty() {
            return Err(RendererError::SurfaceConfigurationError(
                "No supported present modes available".to_string(),
            ));
        }

        // Check if alpha_modes array is empty
        if caps.alpha_modes.is_empty() {
            return Err(RendererError::SurfaceConfigurationError(
                "No supported alpha modes available".to_string(),
            ));
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
        let renderer = Renderer::new().block_on().expect("Failed to create renderer");
        let device = renderer.device();
        assert!(device.limits().max_buffer_size > 0);
    }

    #[test]
    fn test_renderer_queue_access() {
        let renderer = Renderer::new().block_on().expect("Failed to create renderer");
        let _queue = renderer.queue();
        // Queue is accessible, test passes if no panic
    }

    #[test]
    fn test_renderer_create_buffer() {
        let renderer = Renderer::new().block_on().expect("Failed to create renderer");
        let builder = renderer.create_buffer();
        // Builder created successfully
        assert!(std::mem::size_of_val(&builder) > 0);
    }

    #[test]
    fn test_renderer_create_bind_group() {
        let renderer = Renderer::new().block_on().expect("Failed to create renderer");
        let registry = ResourceRegistry::default();
        let builder = renderer.create_bind_group(&registry);
        // Builder created successfully
        assert!(std::mem::size_of_val(&builder) > 0);
    }

    #[test]
    fn test_renderer_write_buffer() {
        let renderer = Renderer::new().block_on().expect("Failed to create renderer");
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
        let renderer = Renderer::new().block_on().expect("Failed to create renderer");
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
        match result.unwrap_err() {
            BufferWriteError::BufferNotFound => {}
        }
    }

    #[test]
    fn test_renderer_integration_buffer_build_and_write() {
        let renderer = Renderer::new().block_on().expect("Failed to create renderer");
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
}
