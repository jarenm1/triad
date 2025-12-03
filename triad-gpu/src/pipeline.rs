use crate::frame_graph::resource::Handle;
use crate::resource_registry::ResourceRegistry;

/// Builder for creating render pipelines
pub struct RenderPipelineBuilder<'a> {
    device: &'a wgpu::Device,
    vertex_shader: Option<Handle<wgpu::ShaderModule>>,
    fragment_shader: Option<Handle<wgpu::ShaderModule>>,
    label: Option<String>,
    layout: Option<wgpu::PipelineLayout>,
    vertex_buffers: Vec<wgpu::VertexBufferLayout<'static>>,
    primitive: Option<wgpu::PrimitiveState>,
    depth_stencil: Option<wgpu::DepthStencilState>,
    multisample: Option<wgpu::MultisampleState>,
    fragment_targets: Vec<Option<wgpu::ColorTargetState>>,
}

impl<'a> RenderPipelineBuilder<'a> {
    pub fn new(device: &'a wgpu::Device) -> Self {
        Self {
            device,
            vertex_shader: None,
            fragment_shader: None,
            label: None,
            layout: None,
            vertex_buffers: Vec::new(),
            primitive: None,
            depth_stencil: None,
            multisample: None,
            fragment_targets: Vec::new(),
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn with_vertex_shader(mut self, shader: Handle<wgpu::ShaderModule>) -> Self {
        self.vertex_shader = Some(shader);
        self
    }

    pub fn with_fragment_shader(mut self, shader: Handle<wgpu::ShaderModule>) -> Self {
        self.fragment_shader = Some(shader);
        self
    }

    pub fn with_layout(mut self, layout: wgpu::PipelineLayout) -> Self {
        self.layout = Some(layout);
        self
    }

    pub fn with_vertex_buffer(mut self, buffer: wgpu::VertexBufferLayout<'static>) -> Self {
        self.vertex_buffers.push(buffer);
        self
    }

    pub fn with_primitive(mut self, primitive: wgpu::PrimitiveState) -> Self {
        self.primitive = Some(primitive);
        self
    }

    pub fn with_depth_stencil(mut self, depth_stencil: wgpu::DepthStencilState) -> Self {
        self.depth_stencil = Some(depth_stencil);
        self
    }

    pub fn with_multisample(mut self, multisample: wgpu::MultisampleState) -> Self {
        self.multisample = Some(multisample);
        self
    }

    pub fn with_fragment_target(mut self, target: Option<wgpu::ColorTargetState>) -> Self {
        self.fragment_targets.push(target);
        self
    }

    /// Build the render pipeline and register it in the registry
    pub fn build(
        self,
        registry: &mut ResourceRegistry,
    ) -> Result<Handle<wgpu::RenderPipeline>, PipelineBuildError> {
        let vertex_handle = self
            .vertex_shader
            .ok_or(PipelineBuildError::MissingVertexShader)?;
        let vertex_shader = registry
            .get(vertex_handle)
            .ok_or(PipelineBuildError::ShaderNotFound)?;

        let fragment_shader = if let Some(h) = self.fragment_shader {
            Some(registry.get(h).ok_or(PipelineBuildError::ShaderNotFound)?)
        } else {
            None
        };

        let pipeline_layout = self.layout.unwrap_or_else(|| {
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[],
                    push_constant_ranges: &[],
                })
        });

        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: self.label.as_deref(),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: vertex_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &self.vertex_buffers,
                },
                primitive: self.primitive.unwrap_or_else(|| wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                }),
                depth_stencil: self.depth_stencil,
                multisample: self.multisample.unwrap_or_else(|| wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                }),
                fragment: fragment_shader.map(|shader| wgpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &self.fragment_targets,
                }),
                multiview: None,
                cache: None,
            });

        let handle = registry.insert(pipeline);
        Ok(handle)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PipelineBuildError {
    #[error("Vertex shader is required")]
    MissingVertexShader,
    #[error("Shader module not found in registry")]
    ShaderNotFound,
}
