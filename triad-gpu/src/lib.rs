use wgpu::{Instance, SurfaceConfiguration};
mod frame_graph;
mod pipeline;
mod resource_registry;
mod shader;
mod surface;

pub use frame_graph::{FrameGraph, Handle, Pass, PassBuilder, PassContext, ResourceType};
pub use pipeline::{PipelineBuildError, RenderPipelineBuilder};
pub use resource_registry::ResourceRegistry;
pub use shader::ShaderManager;
pub use surface::SurfaceWrapper;

#[derive(Debug, thiserror::Error)]
pub enum RendererError {
    #[error("Request Adapter Error: {0}")]
    RequestAdapterError(#[from] wgpu::RequestAdapterError),
    #[error("Request Device Error: {0}")]
    RequestDeviceError(#[from] wgpu::RequestDeviceError),
    #[error("Surface Error: {0}")]
    SurfaceError(#[from] wgpu::SurfaceError),
}

pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub instance: wgpu::Instance,
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

    pub fn create_surface(
        &self,
        surface: wgpu::Surface<'static>,
        width: u32,
        height: u32,
    ) -> Result<SurfaceWrapper, RendererError> {
        let caps = surface.get_capabilities(&self.adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

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
