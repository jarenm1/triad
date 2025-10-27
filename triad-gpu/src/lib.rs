use tracing::instrument;
use triad_window::Window;
use wgpu::{Backends, Instance, RenderPassDescriptor, wgt::AccelerationStructureGeometryFlags};

#[derive(Debug, thiserror::Error)]
enum RendererError {
    #[error("Request Adapter Error: {0}")]
    RequestAdapterError(#[from] wgpu::RequestAdapterError),
    #[error("Request Device Error: {0}")]
    RequestDeviceError(#[from] wgpu::RequestDeviceError),
    #[error("Surface Error: {0}")]
    RequestSurfaceError(#[from] wgpu::SurfaceError),
    #[error("Create surface error: {0}")]
    CreateSurfaceError(#[from] wgpu::CreateSurfaceError),
}

struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: Option<wgpu::Surface<'static>>,
}

impl Renderer {
    #[instrument(level = "info", skip_all)]
    pub async fn new(window: Option<&'static Window>) -> Result<Self, RendererError> {
        let instance = Instance::new(&wgpu::InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let surface = match window {
            Some(window) => Some(&instance.create_surface(&window)?),
            None => None,
        };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: surface,
                ..Default::default()
            })
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
            surface,
        })
    }

    #[instrument(level = "info", skip_all)]
    fn render(&self) -> Result<(), RendererError> {
        let output = self.surface.unwrap().get_current_texture()?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("RenderPass"),
            color_attachments: &[None],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        {
            render_pass.set_pipeline(pipeline);
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            render_pass.draw(0..3, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        output.present();
        Ok(())
    }
}
