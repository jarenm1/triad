use triad_gpu::*;

fn main() {
    let renderer = pollster::block_on(triad_gpu::Renderer::new()).unwrap();
}
