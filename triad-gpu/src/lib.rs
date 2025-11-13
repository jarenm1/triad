use wgpu::Instance;

#[derive(Debug, thiserror::Error)]
pub enum RendererError {
    #[error("Request Adapter Error: {0}")]
    RequestAdapterError(#[from] wgpu::RequestAdapterError),
    #[error("Request Device Error: {0}")]
    RequestDeviceError(#[from] wgpu::RequestDeviceError),
}

pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub render_pipeline: wgpu::RenderPipeline,
}

impl Renderer {
    pub async fn new() -> Result<Self, RendererError> {
        let instance = Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: None,
                ..Default::default()
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Renderer"),
                ..Default::default()
            })
            .await?;

        let fragment_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fragment Shader Layout"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fragment.wgsl").into()),
        });
        let vertex_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vertex Shader Layout"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/vertex.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Front),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &fragment_shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            render_pipeline,
        })
    }
    fn render(&self) {
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        self.queue.submit(Some(encoder.finish()));
    }
}
