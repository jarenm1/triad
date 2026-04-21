use std::error::Error;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

use tracing::info;
use triad_gpu::{
    BufferUsage, ColorLoadOp, CopyPassBuilder, DepthLoadOp, RenderPassBuilder,
    RenderPipelineBuilder, Renderer, ResourceRegistry, TextureBufferCopy, wgpu,
};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;
const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const PIXEL_BYTES: u32 = 4;
const EXPECTED_RGBA: [u8; 4] = [255, 0, 0, 255];
const DEFAULT_OUTPUT_PATH: &str = "triad-headless-output.png";

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

fn init_logging() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "triad_headless=info,triad_gpu=info".into()),
        )
        .with_target(false)
        .compact()
        .try_init();
}

fn main() -> Result<(), Box<dyn Error>> {
    init_logging();
    run_demo()
}

fn run_demo() -> Result<(), Box<dyn Error>> {
    info!(
        width = WIDTH,
        height = HEIGHT,
        "starting headless offscreen demo"
    );

    let renderer = pollster::block_on(Renderer::new())?;
    let mut registry = ResourceRegistry::default();

    let color_texture = renderer
        .create_texture()
        .label("headless color")
        .size_2d(WIDTH, HEIGHT)
        .format(COLOR_FORMAT)
        .usage_flags(
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
        )
        .build(&mut registry)?;

    let depth_texture = renderer
        .create_texture()
        .label("headless depth")
        .size_2d(WIDTH, HEIGHT)
        .format(DEPTH_FORMAT)
        .usage_flags(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING)
        .build(&mut registry)?;

    let readback_buffer_size = (WIDTH * HEIGHT * PIXEL_BYTES) as usize;
    let readback = renderer
        .create_gpu_buffer::<u8>()
        .label("headless color readback")
        .capacity(readback_buffer_size)
        .usage(BufferUsage::Readback)
        .build(&mut registry)?;

    let shader = renderer
        .create_shader_module()
        .label("headless fullscreen triangle")
        .with_wgsl_source(SHADER)
        .build(&mut registry)?;

    let pipeline = RenderPipelineBuilder::new(renderer.device())
        .with_label("headless offscreen pipeline")
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
        .build(&mut registry)?;

    let render_pass = RenderPassBuilder::new("HeadlessRender")
        .with_pipeline(pipeline)
        .with_color_texture_attachment(
            color_texture,
            ColorLoadOp::Clear(wgpu::Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            }),
        )
        .with_depth_stencil_texture(
            depth_texture,
            DepthLoadOp::Clear(1.0),
            wgpu::StoreOp::Store,
            None,
        )
        .draw(3, 1)
        .build()?;

    let copy_pass = CopyPassBuilder::new("ReadbackColor")
        .copy_texture_to_buffer_region(
            TextureBufferCopy::texture_to_buffer(
                color_texture,
                readback.handle(),
                wgpu::Extent3d {
                    width: WIDTH,
                    height: HEIGHT,
                    depth_or_array_layers: 1,
                },
            )
            .with_layout(Some(WIDTH * PIXEL_BYTES), Some(HEIGHT)),
        )
        .build()?;

    let mut graph = renderer.create_frame_graph();
    graph.add_pass(render_pass);
    graph.add_pass(copy_pass);

    let mut executable = graph.build()?;
    executable.execute(renderer.device(), renderer.queue(), &registry);

    let bytes = renderer.read_buffer::<u8>(readback.handle(), &registry)?;
    validate_pixels(&bytes)?;
    let output_path = output_path_from_env();
    write_png(&bytes, &output_path)?;

    info!(
        width = WIDTH,
        height = HEIGHT,
        path = %output_path.display(),
        "headless offscreen render, copy, and readback completed"
    );
    Ok(())
}

fn output_path_from_env() -> PathBuf {
    std::env::var("TRIAD_HEADLESS_PNG")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_OUTPUT_PATH))
}

fn validate_pixels(bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    let row_bytes = (WIDTH * PIXEL_BYTES) as usize;
    let expected_len = row_bytes * HEIGHT as usize;
    if bytes.len() != expected_len {
        return Err(format!(
            "unexpected readback size: got {}, expected {}",
            bytes.len(),
            expected_len
        )
        .into());
    }

    for &pixel_index in &[
        0usize,
        (WIDTH as usize * HEIGHT as usize) / 2,
        (WIDTH as usize * HEIGHT as usize) - 1,
    ] {
        let start = pixel_index * PIXEL_BYTES as usize;
        let rgba = &bytes[start..start + PIXEL_BYTES as usize];
        if rgba != EXPECTED_RGBA {
            return Err(format!(
                "unexpected pixel at index {}: got {:?}, expected {:?}",
                pixel_index, rgba, EXPECTED_RGBA
            )
            .into());
        }
    }

    Ok(())
}

fn write_png(bytes: &[u8], path: &PathBuf) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, WIDTH, HEIGHT);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut png_writer = encoder.write_header()?;
    png_writer.write_image_data(bytes)?;
    Ok(())
}
