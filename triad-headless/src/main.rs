use std::error::Error;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

use tracing::info;
use triad_gpu::{Renderer, ResourceRegistry};
use triad_sim::{RedSquareSim, SimConfig};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;
const DEFAULT_OUTPUT_PATH: &str = "triad-headless-output.png";

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
    let config = SimConfig {
        width: WIDTH,
        height: HEIGHT,
    };
    info!(
        width = config.width,
        height = config.height,
        "starting headless offscreen demo"
    );

    let renderer = pollster::block_on(Renderer::new())?;
    let mut registry = ResourceRegistry::default();
    let sim = RedSquareSim::new(&renderer, &mut registry, config)?;
    let mut executable = sim.build_offscreen_graph()?;
    executable.execute(renderer.device(), renderer.queue(), &registry);

    let bytes = renderer.read_buffer::<u8>(sim.readback_handle(), &registry)?;
    sim.validate_red_image(&bytes)
        .map_err(|err| std::io::Error::other(err))?;
    let output_path = output_path_from_env();
    write_png(&bytes, sim.config(), &output_path)?;

    info!(
        width = sim.config().width,
        height = sim.config().height,
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

fn write_png(bytes: &[u8], config: SimConfig, path: &PathBuf) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, config.width, config.height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut png_writer = encoder.write_header()?;
    png_writer.write_image_data(bytes)?;
    Ok(())
}
