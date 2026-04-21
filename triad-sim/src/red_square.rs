use triad_gpu::{
    BufferUsage, ColorLoadOp, CopyPassBuilder, DepthLoadOp, ExecutableFrameGraph, FrameGraph,
    FrameTextureView, GpuError, RenderPassBuilder, RenderPipelineBuilder, Renderer,
    ResourceRegistry, Result, TextureBufferCopy, wgpu,
};

const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const PIXEL_BYTES: u32 = 4;
const EXPECTED_RGBA: [u8; 4] = [255, 0, 0, 255];

const SHADER: &str = r#"
struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VsOut {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(3.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );

    var out: VsOut;
    out.clip_pos = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
"#;

#[derive(Debug, Clone, Copy)]
pub struct SimConfig {
    pub width: u32,
    pub height: u32,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            width: 64,
            height: 64,
        }
    }
}

/// Minimal shared sim core that can render to either persistent offscreen textures
/// or per-frame window attachments. This is the first slice of the GPU-first runner split:
/// the sim crate owns resources and frame-graph construction, while runners own process/UI concerns.
pub struct RedSquareSim {
    config: SimConfig,
    color_texture: triad_gpu::Handle<wgpu::Texture>,
    depth_texture: triad_gpu::Handle<wgpu::Texture>,
    readback: triad_gpu::GpuBuffer<u8>,
    pipeline: triad_gpu::Handle<wgpu::RenderPipeline>,
}

impl RedSquareSim {
    pub fn new(
        renderer: &Renderer,
        registry: &mut ResourceRegistry,
        config: SimConfig,
    ) -> Result<Self> {
        let color_texture = renderer
            .create_texture()
            .label("sim color")
            .size_2d(config.width, config.height)
            .format(COLOR_FORMAT)
            .usage_flags(
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
            )
            .build(registry)?;

        let depth_texture = renderer
            .create_texture()
            .label("sim depth")
            .size_2d(config.width, config.height)
            .format(DEPTH_FORMAT)
            .usage_flags(
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            )
            .build(registry)?;

        let readback = renderer
            .create_gpu_buffer::<u8>()
            .label("sim color readback")
            .capacity((config.width * config.height * PIXEL_BYTES) as usize)
            .usage(BufferUsage::Readback)
            .build(registry)?;

        let shader = renderer
            .create_shader_module()
            .label("sim fullscreen triangle")
            .with_wgsl_source(SHADER)
            .build(registry)?;

        let pipeline = RenderPipelineBuilder::new(renderer.device())
            .with_label("sim offscreen pipeline")
            .with_vertex_shader(shader)
            .with_fragment_shader(shader)
            .with_fragment_target(Some(wgpu::ColorTargetState {
                format: COLOR_FORMAT,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            }))
            .with_depth_stencil(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            })
            .build(registry)?;

        Ok(Self {
            config,
            color_texture,
            depth_texture,
            readback,
            pipeline,
        })
    }

    pub fn config(&self) -> SimConfig {
        self.config
    }

    pub fn color_texture(&self) -> triad_gpu::Handle<wgpu::Texture> {
        self.color_texture
    }

    pub fn readback_handle(&self) -> triad_gpu::Handle<wgpu::Buffer> {
        self.readback.handle()
    }

    pub fn build_offscreen_graph(&self) -> Result<ExecutableFrameGraph> {
        let mut graph = FrameGraph::new();
        graph.add_pass(self.offscreen_render_pass()?);
        graph.add_pass(self.readback_copy_pass()?);
        Ok(graph.build()?)
    }

    /// Builds the same scene against frame-local attachments so a windowed runner can visualize
    /// the exact same sim state without duplicating rendering code.
    pub fn build_window_graph(
        &self,
        frame_target: triad_gpu::Handle<FrameTextureView>,
        depth_target: triad_gpu::Handle<FrameTextureView>,
    ) -> Result<ExecutableFrameGraph> {
        let render_pass = RenderPassBuilder::new("SimWindowRender")
            .with_pipeline(self.pipeline)
            .with_frame_color_attachment(
                frame_target,
                ColorLoadOp::Clear(wgpu::Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                }),
            )
            .with_frame_depth_stencil_attachment(
                depth_target,
                DepthLoadOp::Clear(1.0),
                wgpu::StoreOp::Store,
                None,
            )
            .draw(3, 1)
            .build()?;

        let mut graph = FrameGraph::new();
        graph.add_pass(render_pass);
        Ok(graph.build()?)
    }

    pub fn validate_red_image(&self, bytes: &[u8]) -> std::result::Result<(), String> {
        let row_bytes = (self.config.width * PIXEL_BYTES) as usize;
        let expected_len = row_bytes * self.config.height as usize;
        if bytes.len() != expected_len {
            return Err(format!(
                "unexpected readback size: got {}, expected {}",
                bytes.len(),
                expected_len
            ));
        }

        for &pixel_index in &[
            0usize,
            (self.config.width as usize * self.config.height as usize) / 2,
            (self.config.width as usize * self.config.height as usize) - 1,
        ] {
            let start = pixel_index * PIXEL_BYTES as usize;
            let rgba = &bytes[start..start + PIXEL_BYTES as usize];
            if rgba != EXPECTED_RGBA {
                return Err(format!(
                    "unexpected pixel at index {}: got {:?}, expected {:?}",
                    pixel_index, rgba, EXPECTED_RGBA
                ));
            }
        }

        Ok(())
    }

    fn offscreen_render_pass(&self) -> std::result::Result<triad_gpu::PassBuilder, GpuError> {
        Ok(RenderPassBuilder::new("SimOffscreenRender")
            .with_pipeline(self.pipeline)
            .with_color_texture_attachment(
                self.color_texture,
                ColorLoadOp::Clear(wgpu::Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                }),
            )
            .with_depth_stencil_texture(
                self.depth_texture,
                DepthLoadOp::Clear(1.0),
                wgpu::StoreOp::Store,
                None,
            )
            .draw(3, 1)
            .build()?)
    }

    fn readback_copy_pass(&self) -> std::result::Result<triad_gpu::PassBuilder, GpuError> {
        Ok(CopyPassBuilder::new("SimReadbackColor")
            .copy_texture_to_buffer_region(
                TextureBufferCopy::texture_to_buffer(
                    self.color_texture,
                    self.readback.handle(),
                    wgpu::Extent3d {
                        width: self.config.width,
                        height: self.config.height,
                        depth_or_array_layers: 1,
                    },
                )
                .with_layout(
                    Some(self.config.width * PIXEL_BYTES),
                    Some(self.config.height),
                ),
            )
            .build()?)
    }
}
