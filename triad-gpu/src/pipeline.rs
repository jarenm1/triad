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

#[cfg(test)]
mod tests {
    use super::*;
    use pollster::FutureExt;

    async fn create_test_device() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("Failed to get adapter");
        adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("Failed to get device")
    }

    fn create_test_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("test_shader"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
                @vertex
                fn vs_main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
                    return vec4<f32>(pos, 1.0);
                }

                @fragment
                fn fs_main() -> @location(0) vec4<f32> {
                    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
                }
                "#
                .into(),
            ),
        })
    }

    #[test]
    fn test_render_pipeline_builder_missing_vertex_shader() {
        let (device, _queue) = create_test_device().block_on();
        let mut registry = ResourceRegistry::default();

        let result = RenderPipelineBuilder::new(&device)
            .build(&mut registry);

        assert!(result.is_err());
        match result.unwrap_err() {
            PipelineBuildError::MissingVertexShader => {}
            _ => panic!("Expected MissingVertexShader error"),
        }
    }

    #[test]
    fn test_render_pipeline_builder_with_shaders() {
        let (device, _queue) = create_test_device().block_on();
        let mut registry = ResourceRegistry::default();

        // Create shader modules
        let vertex_shader = create_test_shader_module(&device);
        let vertex_handle = registry.insert(vertex_shader);

        let fragment_shader = create_test_shader_module(&device);
        let fragment_handle = registry.insert(fragment_shader);

        // Build pipeline
        let pipeline_handle = RenderPipelineBuilder::new(&device)
            .with_label("test_pipeline")
            .with_vertex_shader(vertex_handle)
            .with_fragment_shader(fragment_handle)
            .with_vertex_buffer(wgpu::VertexBufferLayout {
                array_stride: 12,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                }],
            })
            .with_fragment_target(Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            }))
            .build(&mut registry)
            .expect("Failed to build pipeline");

        // Verify pipeline exists
        assert!(registry.get(pipeline_handle).is_some());
    }

    #[test]
    fn test_render_pipeline_builder_shader_not_found() {
        let (device, _queue) = create_test_device().block_on();
        let mut registry = ResourceRegistry::default();

        // Create a fake handle that doesn't exist
        let fake_handle = Handle::<wgpu::ShaderModule>::next();

        let result = RenderPipelineBuilder::new(&device)
            .with_vertex_shader(fake_handle)
            .build(&mut registry);

        assert!(result.is_err());
        match result.unwrap_err() {
            PipelineBuildError::ShaderNotFound => {}
            _ => panic!("Expected ShaderNotFound error"),
        }
    }

    #[test]
    fn test_render_pipeline_builder_with_custom_layout() {
        let (device, _queue) = create_test_device().block_on();
        let mut registry = ResourceRegistry::default();

        let vertex_shader = create_test_shader_module(&device);
        let vertex_handle = registry.insert(vertex_shader);

        let fragment_shader = create_test_shader_module(&device);
        let fragment_handle = registry.insert(fragment_shader);

        // Create a custom pipeline layout
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("custom_layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        // Note: WGPU requires at least one color attachment or depth-stencil attachment
        let pipeline_handle = RenderPipelineBuilder::new(&device)
            .with_vertex_shader(vertex_handle)
            .with_fragment_shader(fragment_handle)
            .with_layout(layout)
            .with_vertex_buffer(wgpu::VertexBufferLayout {
                array_stride: 12,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                }],
            })
            .with_fragment_target(Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            }))
            .build(&mut registry)
            .expect("Failed to build pipeline");

        assert!(registry.get(pipeline_handle).is_some());
    }

    #[test]
    fn test_render_pipeline_builder_with_depth_stencil() {
        let (device, _queue) = create_test_device().block_on();
        let mut registry = ResourceRegistry::default();

        let vertex_shader = create_test_shader_module(&device);
        let vertex_handle = registry.insert(vertex_shader);

        let pipeline_handle = RenderPipelineBuilder::new(&device)
            .with_vertex_shader(vertex_handle)
            .with_depth_stencil(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            })
            .with_vertex_buffer(wgpu::VertexBufferLayout {
                array_stride: 12,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                }],
            })
            .build(&mut registry)
            .expect("Failed to build pipeline");

        assert!(registry.get(pipeline_handle).is_some());
    }

    #[test]
    fn test_render_pipeline_builder_defaults() {
        let (device, _queue) = create_test_device().block_on();
        let mut registry = ResourceRegistry::default();

        let vertex_shader = create_test_shader_module(&device);
        let vertex_handle = registry.insert(vertex_shader);

        let fragment_shader = create_test_shader_module(&device);
        let fragment_handle = registry.insert(fragment_shader);

        // Build with minimal configuration (should use defaults)
        // Note: WGPU requires at least one color attachment or depth-stencil attachment
        let pipeline_handle = RenderPipelineBuilder::new(&device)
            .with_vertex_shader(vertex_handle)
            .with_fragment_shader(fragment_handle)
            .with_vertex_buffer(wgpu::VertexBufferLayout {
                array_stride: 12,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                }],
            })
            .with_fragment_target(Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            }))
            .build(&mut registry)
            .expect("Failed to build pipeline");

        // Should use default primitive state, multisample, etc.
        assert!(registry.get(pipeline_handle).is_some());
    }
}
