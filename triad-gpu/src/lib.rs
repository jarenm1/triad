use wgpu::{Instance, SurfaceConfiguration};
mod frame_graph;
mod pipeline;
pub mod ply_loader;
mod resource_registry;
mod surface;
pub mod triangulation;
mod type_map;
mod types;

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
