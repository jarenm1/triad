use std::error::Error;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

use tracing::{error, info};
use triad_gpu::{
    BindingType, BufferUsage, ComputePassBuilder, CopyPassBuilder, DepthLoadOp, DispatchIndirectArgs,
    DrawIndirectArgs, ExecutableFrameGraph, FrameBufferHandle, FrameGraph, FrameGraphError,
    FrameTextureView, Handle, Pass, PassBuilder, PassContext, RenderPassBuilder, Renderer,
    ResourceRegistry, ShaderStage,
    SpatialGridConfig, SpatialGridGpu, SpatialGridParams, total_cells, wgpu,
};
use triad_window::{CameraUniforms, RendererManager, WindowConfig, egui, run_with_renderer_config};

const DEFAULT_PARTICLE_COUNT: usize = 4_096;
const MIN_PARTICLE_COUNT: usize = 256;
const MAX_PARTICLE_COUNT: usize = 2_097_152;
const WORKGROUP_SIZE: u32 = 64;
/// World-space disk radius: rendered circle and collision use the same value (xy domain ~[-1, 1]).
const PARTICLE_RADIUS: f32 = 0.00875;
/// Jacobi-style separation passes per frame (same spatial hash each pass).
/// Lower this if GPU-bound; 2 is usually enough with the merged neighbor loop.
const COLLISION_ITERATIONS: u32 = 2;

/// Uniform 2D grid over simulation xy ∈ [-1, 1]; single z slab. Keeps `total_cells` moderate for the scan.
const SPATIAL_GRID_DIMS: [u32; 3] = [64, 64, 1];

/// After [`SpatialGridGpu`] rebuild, compute max over particles of (others in same + 8 neighbor
/// cells). Validates `counts_linear` and cell indexing against live positions.
const CLEAR_GRID_NEIGHBOR_STATS_SHADER: &str = r#"
struct GridNeighborStats {
    max_others_in_3x3_cells: atomic<u32>,
}

@group(0) @binding(0) var<storage, read_write> stats: GridNeighborStats;

@compute @workgroup_size(1)
fn cs_main() {
    atomicStore(&stats.max_others_in_3x3_cells, 0u);
}
"#;

const GRID_NEIGHBOR_MAX_SHADER: &str = r#"
struct Params {
    world_origin: vec3<f32>,
    cell_size: f32,
    grid_dims: vec3<u32>,
    entity_count: u32,
}

struct ParticleState {
    position: vec2<f32>,
    velocity: vec2<f32>,
    alive: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct ParticleBuffer {
    particles: array<ParticleState>,
}

struct GridNeighborStats {
    max_others_in_3x3_cells: atomic<u32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> particles: ParticleBuffer;
@group(0) @binding(2) var<storage, read> counts_linear: array<u32>;
@group(0) @binding(3) var<storage, read_write> stats: GridNeighborStats;

fn cell_index(world_pos: vec3<f32>) -> u32 {
    let g = params.grid_dims;
    let o = params.world_origin;
    let inv = 1.0 / params.cell_size;
    let f = floor((world_pos - o) * inv);
    let ix = clamp(i32(f.x), 0, i32(g.x) - 1);
    let iy = clamp(i32(f.y), 0, i32(g.y) - 1);
    let iz = clamp(i32(f.z), 0, i32(g.z) - 1);
    let ux = u32(ix);
    let uy = u32(iy);
    let uz = u32(iz);
    return ux + g.x * (uy + g.y * uz);
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.entity_count) {
        return;
    }
    if (particles.particles[i].alive == 0u) {
        return;
    }
    let p = particles.particles[i].position;
    let c = cell_index(vec3<f32>(p.x, p.y, 0.0));
    let gx = params.grid_dims.x;
    let gy = params.grid_dims.y;
    let cx = c % gx;
    let cy = (c / gx) % gy;

    var tot: u32 = 0u;
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let nx = i32(cx) + dx;
            let ny = i32(cy) + dy;
            if (nx < 0 || ny < 0 || nx >= i32(gx) || ny >= i32(gy)) {
                continue;
            }
            let nc = u32(nx) + gx * u32(ny);
            tot = tot + counts_linear[nc];
        }
    }
    let others = tot - 1u;
    atomicMax(&stats.max_others_in_3x3_cells, others);
}
"#;

/// 2D disk–disk response: full separation on the lower-index body per pair (`j > i`), elastic
/// normal impulse on both bodies (`j > i` gets position + velocity; `j < i` velocity only).
/// Single 3×3 sweep (no duplicate neighbor iteration). Uses r² to skip `sqrt` when separated.
/// Runs after [`SpatialGridGpu`] rebuild; multiple dispatches per frame approximate Jacobi.
const PARTICLE_COLLISION_SHADER: &str = r#"
struct GridParams {
    world_origin: vec3<f32>,
    cell_size: f32,
    grid_dims: vec3<u32>,
    entity_count: u32,
}

struct SimParams {
    dt_seconds: f32,
    _reserved0: f32,
    particle_radius: f32,
    _reserved1: f32,
}

struct ParticleState {
    position: vec2<f32>,
    velocity: vec2<f32>,
    alive: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct ParticleBuffer {
    particles: array<ParticleState>,
}

@group(0) @binding(0) var<uniform> grid_params: GridParams;
@group(0) @binding(1) var<uniform> sim_u: SimParams;
@group(0) @binding(2) var<storage, read_write> particles: ParticleBuffer;
@group(0) @binding(3) var<storage, read> cell_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> sorted_entity_ids: array<u32>;

fn cell_index(world_pos: vec3<f32>) -> u32 {
    let g = grid_params.grid_dims;
    let o = grid_params.world_origin;
    let inv = 1.0 / grid_params.cell_size;
    let f = floor((world_pos - o) * inv);
    let ix = clamp(i32(f.x), 0, i32(g.x) - 1);
    let iy = clamp(i32(f.y), 0, i32(g.y) - 1);
    let iz = clamp(i32(f.z), 0, i32(g.z) - 1);
    let ux = u32(ix);
    let uy = u32(iy);
    let uz = u32(iz);
    return ux + g.x * (uy + g.y * uz);
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= grid_params.entity_count) {
        return;
    }
    if (particles.particles[i].alive == 0u) {
        return;
    }

    let r = sim_u.particle_radius;
    let min_d = 2.0 * r;
    let min_d_sq = min_d * min_d;
    var pi = particles.particles[i].position;
    var vi = particles.particles[i].velocity;
    let c = cell_index(vec3<f32>(pi.x, pi.y, 0.0));
    let gx = grid_params.grid_dims.x;
    let gy = grid_params.grid_dims.y;
    let cx = c % gx;
    let cy = (c / gx) % gy;

    var delta = vec2<f32>(0.0, 0.0);

    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let nx = i32(cx) + dx;
            let ny = i32(cy) + dy;
            if (nx < 0 || ny < 0 || nx >= i32(gx) || ny >= i32(gy)) {
                continue;
            }
            let nc = u32(nx) + gx * u32(ny);
            let start = cell_offsets[nc];
            let end = cell_offsets[nc + 1u];
            for (var k = start; k < end; k = k + 1u) {
                let j = sorted_entity_ids[k];
                if (j == i) {
                    continue;
                }
                if (particles.particles[j].alive == 0u) {
                    continue;
                }
                let pj = particles.particles[j].position;
                let vj = particles.particles[j].velocity;
                var diff = pi - pj;
                let dist_sq = dot(diff, diff);
                if (dist_sq >= min_d_sq) {
                    continue;
                }
                var dist = sqrt(dist_sq);
                if (dist < 1e-8) {
                    diff = vec2<f32>(1.0, 0.3) * r * 0.02;
                    dist = length(diff);
                }
                let n = diff / dist;
                let vn = dot(vi - vj, n);
                if (vn < 0.0) {
                    vi = vi - vn * n;
                }
                if (j > i) {
                    delta = delta + n * (min_d - dist);
                }
            }
        }
    }

    pi = pi + delta;
    pi.x = clamp(pi.x, -0.999, 0.999);
    pi.y = clamp(pi.y, -0.999, 0.999);
    particles.particles[i].position = pi;
    particles.particles[i].velocity = vi;
}
"#;

const PARTICLES_TO_GRID_SHADER: &str = r#"
struct ParticleState {
    position: vec2<f32>,
    velocity: vec2<f32>,
    alive: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct ParticleBuffer {
    particles: array<ParticleState>,
};

@group(0) @binding(0) var<storage, read> particles: ParticleBuffer;
@group(0) @binding(1) var<storage, read_write> grid_positions: array<vec4<f32>>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&grid_positions)) {
        return;
    }
    if (i >= arrayLength(&particles.particles)) {
        return;
    }
    let p = particles.particles[i].position;
    grid_positions[i] = vec4<f32>(p.x, p.y, 0.0, 0.0);
}
"#;

fn particle_count_from_env() -> usize {
    let raw = std::env::var("TRIAD_PARTICLE_COUNT")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());
    let n = raw.unwrap_or(DEFAULT_PARTICLE_COUNT);
    let n = n.clamp(MIN_PARTICLE_COUNT, MAX_PARTICLE_COUNT);
    if raw.is_some_and(|r| r != n) {
        tracing::warn!(
            requested = raw,
            clamped = n,
            min = MIN_PARTICLE_COUNT,
            max = MAX_PARTICLE_COUNT,
            "TRIAD_PARTICLE_COUNT clamped for safety"
        );
    }
    n
}
const READBACK_INTERVAL_FRAMES: u64 = 15;
const READBACK_RING_SIZE: usize = 3;
const READBACK_VALIDATE_INTERVAL_FRAMES: u64 = 240;

fn grid_neighbor_validate_from_env() -> bool {
    std::env::var("TRIAD_GRID_NEIGHBOR_VALIDATE")
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

const RESET_DRAW_ARGS_SHADER: &str = r#"
struct DrawArgs {
    vertex_count: u32,
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0) var<storage, read_write> draw_args: DrawArgs;

@compute @workgroup_size(1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x > 0u) {
        return;
    }

    draw_args.vertex_count = 6u;
    atomicStore(&draw_args.instance_count, 0u);
    draw_args.first_vertex = 0u;
    draw_args.first_instance = 0u;
}
"#;

const SIMULATE_SHADER: &str = r#"
struct ParticleState {
    position: vec2<f32>,
    velocity: vec2<f32>,
    alive: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct ParticleBuffer {
    particles: array<ParticleState>,
};

struct SimParams {
    dt_seconds: f32,
    _reserved0: f32,
    particle_radius: f32,
    _reserved1: f32,
};

@group(0) @binding(0) var<storage, read_write> particles: ParticleBuffer;
@group(0) @binding(1) var<uniform> sim_params: SimParams;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= arrayLength(&particles.particles)) {
        return;
    }

    var particle = particles.particles[index];
    if (particle.alive == 0u) {
        return;
    }

    // Constant-velocity motion: v is set at spawn only.
    particle.position += particle.velocity * sim_params.dt_seconds;

    if (particle.position.x <= -0.98 || particle.position.x >= 0.98) {
        particle.velocity.x = -particle.velocity.x;
        particle.position.x = clamp(particle.position.x, -0.98, 0.98);
    }

    if (particle.position.y <= -0.98 || particle.position.y >= 0.98) {
        particle.velocity.y = -particle.velocity.y;
        particle.position.y = clamp(particle.position.y, -0.98, 0.98);
    }

    particles.particles[index] = particle;
}
"#;

const COMPACT_SHADER: &str = r#"
struct ParticleState {
    position: vec2<f32>,
    velocity: vec2<f32>,
    alive: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct ParticleBuffer {
    particles: array<ParticleState>,
};

struct VisibleIds {
    ids: array<u32>,
};

struct DrawArgs {
    vertex_count: u32,
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0) var<storage, read> particles: ParticleBuffer;
@group(0) @binding(1) var<storage, read_write> visible: VisibleIds;
@group(0) @binding(2) var<storage, read_write> draw_args: DrawArgs;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= arrayLength(&particles.particles)) {
        return;
    }

    let particle = particles.particles[index];
    if (particle.alive == 0u) {
        return;
    }

    let write_index = atomicAdd(&draw_args.instance_count, 1u);
    visible.ids[write_index] = index;
}
"#;

const PARTICLE_RENDER_SHADER: &str = r#"
struct ParticleState {
    position: vec2<f32>,
    velocity: vec2<f32>,
    alive: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct ParticleBuffer {
    particles: array<ParticleState>,
};

struct VisibleIds {
    ids: array<u32>,
};

/// width / height; maps simulation xy to NDC so one sim unit is isotropic in pixels (letterboxed).
struct SimViewParams {
    aspect: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<storage, read> particles: ParticleBuffer;
@group(0) @binding(1) var<storage, read> visible: VisibleIds;
@group(0) @binding(2) var<uniform> sim_view: SimViewParams;

// Must match [`PARTICLE_RADIUS`] in Rust (same world units).
const QUAD_HALF: f32 = 0.00875;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) corner: vec2<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VsOut {
    let particle_id = visible.ids[instance_index];
    let particle = particles.particles[particle_id];

    let n = max(arrayLength(&particles.particles), 1u);
    let z = 0.05 + 0.9 * (f32(particle_id) / f32(n));

    let corners = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
        vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0)
    );
    let q = corners[vertex_index];
    let world = particle.position + q * QUAD_HALF;

    let a = sim_view.aspect;
    var ndc_xy: vec2<f32>;
    if (a >= 1.0) {
        ndc_xy = vec2<f32>(world.x / a, world.y);
    } else {
        ndc_xy = vec2<f32>(world.x, world.y * a);
    }

    var out: VsOut;
    out.clip_pos = vec4<f32>(ndc_xy, z, 1.0);
    out.corner = q;
    return out;
}

@fragment
fn fs_main(@location(0) corner: vec2<f32>) -> @location(0) vec4<f32> {
    if (length(corner) > 1.0) {
        discard;
    }
    return vec4<f32>(0.95, 0.75, 0.2, 1.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ParticleState {
    position: [f32; 2],
    velocity: [f32; 2],
    alive: u32,
    _pad: [u32; 3],
}

impl ParticleState {
    fn from_index(index: usize, particle_count: usize) -> Self {
        let grid_width = ((particle_count as f32).sqrt().ceil() as usize).clamp(16, 256);
        let x = (index % grid_width) as f32;
        let y = (index / grid_width) as f32;
        let rows = (particle_count / grid_width).max(1);
        let px = (x / grid_width as f32) * 1.8 - 0.9;
        let py = (y / rows as f32) * 1.8 - 0.9;

        // World units per second (−1..1 domain); no runtime speed multiplier — only integration `v * dt`.
        let angle = (index as f32) * 2.583_005;
        let speed = 0.28 + ((index % 19) as f32) * 0.035;

        Self {
            position: [px, py],
            velocity: [angle.cos() * speed, angle.sin() * speed],
            alive: 1,
            _pad: [0; 3],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SimParams {
    dt_seconds: f32,
    _reserved0: f32,
    particle_radius: f32,
    _reserved1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SimViewParams {
    /// `viewport_width / viewport_height` — keeps simulation units isotropic on screen.
    aspect: f32,
    _pad: [f32; 3],
}

fn sim_view_params_from_viewport(viewport_w: u32, viewport_h: u32) -> SimViewParams {
    let w = viewport_w.max(1) as f32;
    let h = viewport_h.max(1) as f32;
    SimViewParams {
        aspect: w / h,
        _pad: [0.0; 3],
    }
}

#[derive(Clone, Debug)]
struct DemoStats {
    particle_count: usize,
    workgroup_count: u32,
    spatial_grid_cells: u32,
    gpu_visible_count: u32,
    gpu_visible_count_sync: u32,
    dt_ms: f32,
    update_cpu_ms: f32,
    graph_build_cpu_ms: f32,
    readback_cpu_ms: f32,
    readback_sync_cpu_ms: f32,
    readback_mismatch: bool,
    cached_order_len: usize,
    /// Max (over alive particles) of neighbors in the 3×3 cell block (same + 8 adjacent), excluding self.
    /// Read back from GPU; validates spatial grid `counts_linear` against positions.
    grid_max_others_in_3x3: u32,
    /// When false, grid-neighbor GPU validation passes are skipped (default; better FPS at high N).
    grid_neighbor_validate: bool,
}

impl DemoStats {
    fn new(
        particle_count: usize,
        workgroup_count: u32,
        spatial_grid_cells: u32,
        grid_neighbor_validate: bool,
    ) -> Self {
        Self {
            particle_count,
            workgroup_count,
            spatial_grid_cells,
            gpu_visible_count: 0,
            gpu_visible_count_sync: 0,
            dt_ms: 0.0,
            update_cpu_ms: 0.0,
            graph_build_cpu_ms: 0.0,
            readback_cpu_ms: 0.0,
            readback_sync_cpu_ms: 0.0,
            readback_mismatch: false,
            cached_order_len: 0,
            grid_max_others_in_3x3: 0,
            grid_neighbor_validate,
        }
    }
}

/// Frame-graph pass that records [`SpatialGridGpu::encode_rebuild`] (many compute passes, one encoder).
struct SpatialGridRebuildPass {
    name: String,
    grid: Arc<SpatialGridGpu>,
}

impl Pass for SpatialGridRebuildPass {
    fn name(&self) -> &str {
        &self.name
    }

    fn execute(&self, ctx: &PassContext) -> wgpu::CommandBuffer {
        let mut encoder = ctx.create_command_encoder(Some(&self.name));
        self.grid.encode_rebuild(&mut encoder, ctx.resources);
        encoder.finish()
    }
}

enum ReadbackState {
    Idle,
    Mapping(Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>>),
}

struct ReadbackSlot {
    handle: Handle<wgpu::Buffer>,
    state: ReadbackState,
}

struct ParticleRendererManager {
    particle_buffer: Handle<wgpu::Buffer>,
    visible_ids: Handle<wgpu::Buffer>,
    dispatch_args: Handle<wgpu::Buffer>,
    draw_args: Handle<wgpu::Buffer>,
    draw_args_sync_readback: Handle<wgpu::Buffer>,
    draw_args_readback_slot: Handle<FrameBufferHandle>,
    readback_ring: Vec<ReadbackSlot>,
    current_readback_index: usize,
    sim_params_buffer: Handle<wgpu::Buffer>,
    reset_bind_group: Handle<wgpu::BindGroup>,
    simulate_bind_group: Handle<wgpu::BindGroup>,
    compact_bind_group: Handle<wgpu::BindGroup>,
    render_bind_group: Handle<wgpu::BindGroup>,
    reset_pipeline: Handle<wgpu::ComputePipeline>,
    simulate_pipeline: Handle<wgpu::ComputePipeline>,
    compact_pipeline: Handle<wgpu::ComputePipeline>,
    render_pipeline: Handle<wgpu::RenderPipeline>,
    cached_execution_order: Option<Vec<usize>>,
    last_update: Instant,
    frame_index: u64,
    stats: Arc<Mutex<DemoStats>>,
    frame_target: Handle<FrameTextureView>,
    depth_frame: Handle<FrameTextureView>,
    spatial_grid: Arc<SpatialGridGpu>,
    particles_to_grid_pipeline: Handle<wgpu::ComputePipeline>,
    particles_to_grid_bind_group: Handle<wgpu::BindGroup>,
    particles_to_grid_dispatch_x: u32,
    grid_neighbor_stats: Handle<wgpu::Buffer>,
    grid_neighbor_readback: Handle<wgpu::Buffer>,
    clear_grid_neighbor_pipeline: Handle<wgpu::ComputePipeline>,
    grid_neighbor_max_pipeline: Handle<wgpu::ComputePipeline>,
    clear_grid_neighbor_bind_group: Handle<wgpu::BindGroup>,
    grid_neighbor_max_bind_group: Handle<wgpu::BindGroup>,
    collision_pipeline: Handle<wgpu::ComputePipeline>,
    collision_bind_group: Handle<wgpu::BindGroup>,
    /// When true, runs clear + max passes and copies stats for CPU readback (expensive at high N).
    grid_neighbor_validate: bool,
    viewport_w: u32,
    viewport_h: u32,
    sim_view_buffer: Handle<wgpu::Buffer>,
}

impl ParticleRendererManager {
    fn new(
        renderer: &Renderer,
        registry: &mut ResourceRegistry,
        surface_format: wgpu::TextureFormat,
        stats: Arc<Mutex<DemoStats>>,
        particle_count: usize,
        grid_neighbor_validate: bool,
        viewport_w: u32,
        viewport_h: u32,
    ) -> Result<Self, Box<dyn Error>> {
        let viewport_w = viewport_w.max(1);
        let viewport_h = viewport_h.max(1);
        let particles: Vec<ParticleState> = (0..particle_count)
            .map(|i| ParticleState::from_index(i, particle_count))
            .collect();

        let dispatch_count = (particle_count as u32).div_ceil(WORKGROUP_SIZE);

        let particle_buffer = renderer
            .create_gpu_buffer::<ParticleState>()
            .label("particles")
            .with_data(&particles)
            .build(registry)?;
        let visible_ids = renderer
            .create_gpu_buffer::<u32>()
            .label("visible ids")
            .capacity(particle_count)
            .build(registry)?;
        let dispatch_args = renderer
            .create_gpu_buffer::<DispatchIndirectArgs>()
            .label("particle dispatch args")
            .with_data(&[DispatchIndirectArgs::new(dispatch_count, 1, 1)])
            .usage(BufferUsage::Indirect)
            .build(registry)?;
        let draw_args = renderer
            .create_gpu_buffer::<DrawIndirectArgs>()
            .label("particle draw args")
            .with_data(&[DrawIndirectArgs::new(6, 0, 0, 0)])
            .usage(BufferUsage::Indirect)
            .add_usage(wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC)
            .build(registry)?;
        let mut readback_ring = Vec::with_capacity(READBACK_RING_SIZE);
        for index in 0..READBACK_RING_SIZE {
            let readback = renderer
                .create_gpu_buffer::<DrawIndirectArgs>()
                .label(format!("particle draw args readback {index}"))
                .capacity(1)
                .usage(BufferUsage::Readback)
                .build(registry)?;
            readback_ring.push(ReadbackSlot {
                handle: readback.handle(),
                state: ReadbackState::Idle,
            });
        }
        let draw_args_sync_readback = renderer
            .create_gpu_buffer::<DrawIndirectArgs>()
            .label("particle draw args sync readback")
            .capacity(1)
            .usage(BufferUsage::Readback)
            .build(registry)?;
        let draw_args_readback_slot = registry.insert(FrameBufferHandle::new());
        registry
            .get(draw_args_readback_slot)
            .expect("frame buffer readback slot should exist")
            .set(readback_ring[0].handle);
        let frame_target = registry.insert(FrameTextureView::new());
        let depth_frame = registry.insert(FrameTextureView::new());
        let sim_params_buffer = renderer
            .create_gpu_buffer::<SimParams>()
            .label("simulation params")
            .with_data(&[SimParams {
                dt_seconds: 1.0 / 60.0,
                _reserved0: 0.0,
                particle_radius: PARTICLE_RADIUS,
                _reserved1: 0.0,
            }])
            .usage(BufferUsage::Uniform)
            .build(registry)?;
        let sim_view_buffer = renderer
            .create_gpu_buffer::<SimViewParams>()
            .label("sim view NDC (aspect)")
            .with_data(&[sim_view_params_from_viewport(viewport_w, viewport_h)])
            .usage(BufferUsage::Uniform)
            .build(registry)?;

        let reset_shader = renderer
            .create_shader_module()
            .label("reset draw args")
            .with_wgsl_source(RESET_DRAW_ARGS_SHADER)
            .build(registry)?;
        let simulate_shader = renderer
            .create_shader_module()
            .label("simulate particles")
            .with_wgsl_source(SIMULATE_SHADER)
            .build(registry)?;
        let compact_shader = renderer
            .create_shader_module()
            .label("compact particles")
            .with_wgsl_source(COMPACT_SHADER)
            .build(registry)?;
        let render_shader = renderer
            .create_shader_module()
            .label("particle render")
            .with_wgsl_source(PARTICLE_RENDER_SHADER)
            .build(registry)?;

        let cell_size = 2.0 / SPATIAL_GRID_DIMS[0] as f32;
        let grid_params = SpatialGridParams::new(
            [-1.0, -1.0, 0.0],
            cell_size,
            SPATIAL_GRID_DIMS,
            particle_count as u32,
        );
        let spatial_grid = Arc::new(SpatialGridGpu::new(
            renderer,
            registry,
            SpatialGridConfig {
                params: grid_params,
                max_entities: particle_count as u32,
            },
        )?);

        let particles_to_grid_shader = renderer
            .create_shader_module()
            .label("particles to grid positions")
            .with_wgsl_source(PARTICLES_TO_GRID_SHADER)
            .build(registry)?;
        let (particles_to_grid_layout, particles_to_grid_bind_group) = renderer
            .create_bind_group()
            .label("particles to grid positions")
            .buffer_stage(
                0,
                ShaderStage::Compute,
                particle_buffer.handle(),
                BindingType::StorageRead,
            )
            .buffer_stage(
                1,
                ShaderStage::Compute,
                spatial_grid.positions,
                BindingType::StorageWrite,
            )
            .build(registry)?;
        let particles_to_grid_pl =
            renderer
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("particles to grid layout"),
                    bind_group_layouts: &[registry
                        .get(particles_to_grid_layout)
                        .expect("particles_to_grid layout")],
                    push_constant_ranges: &[],
                });
        let particles_to_grid_pipeline = renderer
            .create_compute_pipeline()
            .with_label("particles to grid pipeline")
            .with_compute_shader(particles_to_grid_shader)
            .with_layout(particles_to_grid_pl)
            .build(registry)?;
        let particles_to_grid_dispatch_x = (particle_count as u32).div_ceil(WORKGROUP_SIZE);

        let grid_neighbor_stats = renderer
            .create_buffer()
            .label("grid neighbor stats (atomic u32)")
            .size(4)
            .usage(BufferUsage::Storage { read_only: false })
            .add_usage(wgpu::BufferUsages::COPY_SRC)
            .build(registry)?;
        let grid_neighbor_readback_buf = renderer
            .create_gpu_buffer::<u32>()
            .label("grid neighbor max readback")
            .capacity(1)
            .usage(BufferUsage::Readback)
            .build(registry)?;
        let grid_neighbor_readback = grid_neighbor_readback_buf.handle();

        let clear_grid_neighbor_shader = renderer
            .create_shader_module()
            .label("clear grid neighbor stats")
            .with_wgsl_source(CLEAR_GRID_NEIGHBOR_STATS_SHADER)
            .build(registry)?;
        let grid_neighbor_max_shader = renderer
            .create_shader_module()
            .label("grid neighbor max")
            .with_wgsl_source(GRID_NEIGHBOR_MAX_SHADER)
            .build(registry)?;

        let (clear_grid_neighbor_layout, clear_grid_neighbor_bind_group) = renderer
            .create_bind_group()
            .label("clear grid neighbor stats")
            .buffer_stage(
                0,
                ShaderStage::Compute,
                grid_neighbor_stats,
                BindingType::StorageWrite,
            )
            .build(registry)?;

        let (grid_neighbor_max_layout, grid_neighbor_max_bind_group) = renderer
            .create_bind_group()
            .label("grid neighbor max")
            .buffer_stage(
                0,
                ShaderStage::Compute,
                spatial_grid.params,
                BindingType::Uniform,
            )
            .buffer_stage(
                1,
                ShaderStage::Compute,
                particle_buffer.handle(),
                BindingType::StorageRead,
            )
            .buffer_stage(
                2,
                ShaderStage::Compute,
                spatial_grid.counts_linear,
                BindingType::StorageRead,
            )
            .buffer_stage(
                3,
                ShaderStage::Compute,
                grid_neighbor_stats,
                BindingType::StorageWrite,
            )
            .build(registry)?;

        let clear_grid_neighbor_pl =
            renderer
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("clear grid neighbor layout"),
                    bind_group_layouts: &[registry
                        .get(clear_grid_neighbor_layout)
                        .expect("clear grid neighbor layout")],
                    push_constant_ranges: &[],
                });
        let grid_neighbor_max_pl = renderer
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("grid neighbor max layout"),
                bind_group_layouts: &[registry
                    .get(grid_neighbor_max_layout)
                    .expect("grid neighbor max layout")],
                push_constant_ranges: &[],
            });

        let clear_grid_neighbor_pipeline = renderer
            .create_compute_pipeline()
            .with_label("clear grid neighbor stats")
            .with_compute_shader(clear_grid_neighbor_shader)
            .with_layout(clear_grid_neighbor_pl)
            .build(registry)?;
        let grid_neighbor_max_pipeline = renderer
            .create_compute_pipeline()
            .with_label("grid neighbor max")
            .with_compute_shader(grid_neighbor_max_shader)
            .with_layout(grid_neighbor_max_pl)
            .build(registry)?;

        let collision_shader = renderer
            .create_shader_module()
            .label("particle collision")
            .with_wgsl_source(PARTICLE_COLLISION_SHADER)
            .build(registry)?;
        let (collision_layout, collision_bind_group) = renderer
            .create_bind_group()
            .label("particle collision")
            .buffer_stage(
                0,
                ShaderStage::Compute,
                spatial_grid.params,
                BindingType::Uniform,
            )
            .buffer_stage(
                1,
                ShaderStage::Compute,
                sim_params_buffer.handle(),
                BindingType::Uniform,
            )
            .buffer_stage(
                2,
                ShaderStage::Compute,
                particle_buffer.handle(),
                BindingType::StorageWrite,
            )
            .buffer_stage(
                3,
                ShaderStage::Compute,
                spatial_grid.cell_offsets,
                BindingType::StorageRead,
            )
            .buffer_stage(
                4,
                ShaderStage::Compute,
                spatial_grid.sorted_entity_ids,
                BindingType::StorageRead,
            )
            .build(registry)?;
        let collision_pl = renderer
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("particle collision layout"),
                bind_group_layouts: &[registry
                    .get(collision_layout)
                    .expect("collision layout")],
                push_constant_ranges: &[],
            });
        let collision_pipeline = renderer
            .create_compute_pipeline()
            .with_label("particle collision")
            .with_compute_shader(collision_shader)
            .with_layout(collision_pl)
            .build(registry)?;

        let (reset_layout, reset_bind_group) = renderer
            .create_bind_group()
            .label("reset draw args")
            .buffer_stage(
                0,
                ShaderStage::Compute,
                draw_args.handle(),
                BindingType::StorageWrite,
            )
            .build(registry)?;
        let (simulate_layout, simulate_bind_group) = renderer
            .create_bind_group()
            .label("simulate particles")
            .buffer_stage(
                0,
                ShaderStage::Compute,
                particle_buffer.handle(),
                BindingType::StorageWrite,
            )
            .buffer_stage(
                1,
                ShaderStage::Compute,
                sim_params_buffer.handle(),
                BindingType::Uniform,
            )
            .build(registry)?;
        let (compact_layout, compact_bind_group) = renderer
            .create_bind_group()
            .label("compact particles")
            .buffer_stage(
                0,
                ShaderStage::Compute,
                particle_buffer.handle(),
                BindingType::StorageRead,
            )
            .buffer_stage(
                1,
                ShaderStage::Compute,
                visible_ids.handle(),
                BindingType::StorageWrite,
            )
            .buffer_stage(
                2,
                ShaderStage::Compute,
                draw_args.handle(),
                BindingType::StorageWrite,
            )
            .build(registry)?;
        let (render_layout, render_bind_group) = renderer
            .create_bind_group()
            .label("render particles")
            .buffer_stage(
                0,
                ShaderStage::Vertex,
                particle_buffer.handle(),
                BindingType::StorageRead,
            )
            .buffer_stage(
                1,
                ShaderStage::Vertex,
                visible_ids.handle(),
                BindingType::StorageRead,
            )
            .buffer_stage(
                2,
                ShaderStage::Vertex,
                sim_view_buffer.handle(),
                BindingType::Uniform,
            )
            .build(registry)?;

        let reset_pipeline_layout =
            renderer
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("reset draw args layout"),
                    bind_group_layouts: &[registry
                        .get(reset_layout)
                        .expect("reset bind group layout should exist")],
                    push_constant_ranges: &[],
                });
        let simulate_pipeline_layout =
            renderer
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("simulate particles layout"),
                    bind_group_layouts: &[registry
                        .get(simulate_layout)
                        .expect("simulate bind group layout should exist")],
                    push_constant_ranges: &[],
                });
        let compact_pipeline_layout =
            renderer
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("compact particles layout"),
                    bind_group_layouts: &[registry
                        .get(compact_layout)
                        .expect("compact bind group layout should exist")],
                    push_constant_ranges: &[],
                });
        let render_pipeline_layout =
            renderer
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("particle render layout"),
                    bind_group_layouts: &[registry
                        .get(render_layout)
                        .expect("render bind group layout should exist")],
                    push_constant_ranges: &[],
                });

        let reset_pipeline = renderer
            .create_compute_pipeline()
            .with_label("reset draw args pipeline")
            .with_compute_shader(reset_shader)
            .with_layout(reset_pipeline_layout)
            .build(registry)?;
        let simulate_pipeline = renderer
            .create_compute_pipeline()
            .with_label("simulate particles pipeline")
            .with_compute_shader(simulate_shader)
            .with_layout(simulate_pipeline_layout)
            .build(registry)?;
        let compact_pipeline = renderer
            .create_compute_pipeline()
            .with_label("compact particles pipeline")
            .with_compute_shader(compact_shader)
            .with_layout(compact_pipeline_layout)
            .build(registry)?;
        let render_pipeline = renderer
            .create_render_pipeline()
            .with_label("particle render pipeline")
            .with_vertex_shader(render_shader)
            .with_fragment_shader(render_shader)
            .with_layout(render_pipeline_layout)
            .with_primitive(wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            })
            .with_fragment_target(Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            }))
            .with_depth_stencil(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            })
            .build(registry)?;

        Ok(Self {
            particle_buffer: particle_buffer.handle(),
            visible_ids: visible_ids.handle(),
            dispatch_args: dispatch_args.handle(),
            draw_args: draw_args.handle(),
            draw_args_sync_readback: draw_args_sync_readback.handle(),
            draw_args_readback_slot,
            readback_ring,
            current_readback_index: 0,
            sim_params_buffer: sim_params_buffer.handle(),
            reset_bind_group,
            simulate_bind_group,
            compact_bind_group,
            render_bind_group,
            reset_pipeline,
            simulate_pipeline,
            compact_pipeline,
            render_pipeline,
            cached_execution_order: None,
            last_update: Instant::now(),
            frame_index: 0,
            stats,
            frame_target,
            depth_frame,
            spatial_grid,
            particles_to_grid_pipeline,
            particles_to_grid_bind_group,
            particles_to_grid_dispatch_x,
            grid_neighbor_stats,
            grid_neighbor_readback,
            clear_grid_neighbor_pipeline,
            grid_neighbor_max_pipeline,
            clear_grid_neighbor_bind_group,
            grid_neighbor_max_bind_group,
            collision_pipeline,
            collision_bind_group,
            grid_neighbor_validate,
            viewport_w,
            viewport_h,
            sim_view_buffer: sim_view_buffer.handle(),
        })
    }
}

impl RendererManager for ParticleRendererManager {
    fn update(
        &mut self,
        renderer: &Renderer,
        registry: &mut ResourceRegistry,
        _camera: &CameraUniforms,
    ) -> Result<(), Box<dyn Error>> {
        let update_start = Instant::now();
        let now = Instant::now();
        let dt_seconds = (now - self.last_update)
            .as_secs_f32()
            .clamp(1.0 / 240.0, 1.0 / 10.0);
        self.last_update = now;

        let params = [SimParams {
            dt_seconds,
            _reserved0: 0.0,
            particle_radius: PARTICLE_RADIUS,
            _reserved1: 0.0,
        }];
        renderer.write_buffer(self.sim_params_buffer, &params, registry)?;
        let sim_view = [sim_view_params_from_viewport(self.viewport_w, self.viewport_h)];
        renderer.write_buffer(self.sim_view_buffer, &sim_view, registry)?;

        let mut gpu_visible_count = None;
        let mut gpu_visible_count_sync = None;
        let mut readback_cpu_ms = 0.0;
        let mut readback_sync_cpu_ms = 0.0;
        let mut readback_mismatch = None;
        renderer.device().poll(wgpu::PollType::Poll)?;
        for slot in &mut self.readback_ring {
            let completed = match &slot.state {
                ReadbackState::Idle => None,
                ReadbackState::Mapping(status) => {
                    status.lock().ok().and_then(|mut state| state.take())
                }
            };

            if let Some(result) = completed {
                match result {
                    Ok(()) => {
                        let readback_start = Instant::now();
                        let buffer = registry
                            .get(slot.handle)
                            .expect("readback buffer handle should exist");
                        {
                            let mapped = buffer.slice(..).get_mapped_range();
                            let values = bytemuck::cast_slice::<u8, DrawIndirectArgs>(&mapped);
                            gpu_visible_count = values.first().map(|args| args.instance_count);
                        }
                        buffer.unmap();
                        readback_cpu_ms = readback_start.elapsed().as_secs_f32() * 1000.0;
                    }
                    Err(err) => {
                        error!(error = %err, "failed to map particle draw args readback buffer");
                        if let Some(buffer) = registry.get(slot.handle) {
                            buffer.unmap();
                        }
                    }
                }
                slot.state = ReadbackState::Idle;
            }
        }

        self.frame_index = self.frame_index.wrapping_add(1);
        if self.frame_index % READBACK_INTERVAL_FRAMES == 0 {
            if let Some(next_index) = self.next_idle_readback_index() {
                let current_handle = self.readback_ring[self.current_readback_index].handle;
                let buffer = registry
                    .get(current_handle)
                    .expect("current readback buffer should exist");
                let slice = buffer.slice(..);
                let status = Arc::new(Mutex::new(None));
                let status_for_callback = Arc::clone(&status);
                slice.map_async(wgpu::MapMode::Read, move |result| {
                    if let Ok(mut slot) = status_for_callback.lock() {
                        *slot = Some(result);
                    }
                });
                self.readback_ring[self.current_readback_index].state =
                    ReadbackState::Mapping(status);
                self.current_readback_index = next_index;
                registry
                    .get(self.draw_args_readback_slot)
                    .expect("frame buffer readback slot should exist")
                    .set(self.readback_ring[self.current_readback_index].handle);
            } else {
                error!("no idle readback buffer available; skipping GPU stats rotation");
            }
        }

        if self.frame_index % READBACK_VALIDATE_INTERVAL_FRAMES == 0 {
            let readback_start = Instant::now();
            match renderer.read_buffer::<DrawIndirectArgs>(self.draw_args_sync_readback, registry) {
                Ok(values) => {
                    gpu_visible_count_sync = values.first().map(|args| args.instance_count);
                    readback_sync_cpu_ms = readback_start.elapsed().as_secs_f32() * 1000.0;
                    let async_count = gpu_visible_count
                        .or_else(|| self.stats.lock().ok().map(|stats| stats.gpu_visible_count));
                    if let (Some(async_count), Some(sync_count)) =
                        (async_count, gpu_visible_count_sync)
                    {
                        let mismatch = async_count != sync_count;
                        readback_mismatch = Some(mismatch);
                        if mismatch {
                            error!(
                                async_count,
                                sync_count, "particle draw args async/sync readback mismatch"
                            );
                        }
                    }
                }
                Err(err) => {
                    error!(error = %err, "failed to synchronously read back particle draw args");
                }
            }
        }

        if let Ok(mut stats) = self.stats.lock() {
            stats.dt_ms = dt_seconds * 1000.0;
            stats.update_cpu_ms = update_start.elapsed().as_secs_f32() * 1000.0;
            if let Some(count) = gpu_visible_count {
                stats.gpu_visible_count = count;
                stats.readback_cpu_ms = readback_cpu_ms;
            }
            if let Some(count) = gpu_visible_count_sync {
                stats.gpu_visible_count_sync = count;
                stats.readback_sync_cpu_ms = readback_sync_cpu_ms;
            }
            if let Some(mismatch) = readback_mismatch {
                stats.readback_mismatch = mismatch;
            }
        }

        if self.grid_neighbor_validate && self.frame_index % 120 == 0 {
            match renderer.read_buffer::<u32>(self.grid_neighbor_readback, registry) {
                Ok(v) => {
                    if let Some(m) = v.first() {
                        if let Ok(mut stats) = self.stats.lock() {
                            stats.grid_max_others_in_3x3 = *m;
                        }
                    }
                }
                Err(err) => {
                    error!(error = %err, "grid neighbor max readback failed");
                }
            }
        }

        Ok(())
    }

    fn prepare_frame(
        &mut self,
        registry: &mut ResourceRegistry,
        final_view: Arc<wgpu::TextureView>,
        depth_view: Option<Arc<wgpu::TextureView>>,
    ) -> Result<bool, Box<dyn Error>> {
        let slot = registry
            .get(self.frame_target)
            .expect("frame target slot should exist");
        slot.set(final_view);
        if let Some(depth) = depth_view {
            registry
                .get(self.depth_frame)
                .expect("depth frame slot should exist")
                .set(depth);
        }
        Ok(false)
    }

    fn build_frame_graph(&mut self) -> Result<ExecutableFrameGraph, FrameGraphError> {
        let build_start = Instant::now();
        let reset_pass = ComputePassBuilder::new("ResetDrawArgs")
            .read_write(self.draw_args)
            .with_pipeline(self.reset_pipeline)
            .with_bind_group(0, self.reset_bind_group)
            .dispatch(1, 1, 1)
            .build()
            .expect("reset pass should build");

        let simulate_pass = ComputePassBuilder::new("SimulateParticles")
            .read(self.dispatch_args)
            .read_write(self.particle_buffer)
            .with_pipeline(self.simulate_pipeline)
            .with_bind_group(0, self.simulate_bind_group)
            .dispatch_indirect(self.dispatch_args, 0)
            .build()
            .expect("simulate pass should build");

        let particles_to_grid_pass = ComputePassBuilder::new("ParticlesToGridPositions")
            .read(self.particle_buffer)
            .write(self.spatial_grid.positions)
            .with_pipeline(self.particles_to_grid_pipeline)
            .with_bind_group(0, self.particles_to_grid_bind_group)
            .dispatch(self.particles_to_grid_dispatch_x, 1, 1)
            .build()
            .expect("particles to grid pass should build");

        let mut spatial_grid_pb = PassBuilder::new("SpatialGridRebuild");
        {
            let g = &*self.spatial_grid;
            spatial_grid_pb.read(g.params);
            spatial_grid_pb.read(g.positions);
            spatial_grid_pb.read_write(g.cell_atomics);
            spatial_grid_pb.read_write(g.counts_linear);
            spatial_grid_pb.read_write(g.cell_offsets);
            spatial_grid_pb.read_write(g.write_heads);
            spatial_grid_pb.read_write(g.sorted_entity_ids);
        }
        let spatial_grid_pass = spatial_grid_pb.with_pass(Box::new(SpatialGridRebuildPass {
            name: "SpatialGridRebuild".to_string(),
            grid: Arc::clone(&self.spatial_grid),
        }));

        let mut collision_passes: Vec<PassBuilder> = Vec::with_capacity(COLLISION_ITERATIONS as usize);
        for iter in 0..COLLISION_ITERATIONS {
            let p = ComputePassBuilder::new(format!("ParticleCollision_{iter}"))
                .read(self.spatial_grid.params)
                .read(self.sim_params_buffer)
                .read_write(self.particle_buffer)
                .read(self.spatial_grid.cell_offsets)
                .read(self.spatial_grid.sorted_entity_ids)
                .with_pipeline(self.collision_pipeline)
                .with_bind_group(0, self.collision_bind_group)
                .dispatch(self.particles_to_grid_dispatch_x, 1, 1)
                .build()
                .unwrap_or_else(|_| {
                    panic!("collision pass {iter} should build");
                });
            collision_passes.push(p);
        }

        let clear_grid_neighbor_pass = ComputePassBuilder::new("ClearGridNeighborStats")
            .read_write(self.grid_neighbor_stats)
            .with_pipeline(self.clear_grid_neighbor_pipeline)
            .with_bind_group(0, self.clear_grid_neighbor_bind_group)
            .dispatch(1, 1, 1)
            .build()
            .expect("clear grid neighbor pass should build");

        let grid_neighbor_pass = ComputePassBuilder::new("GridNeighborMax")
            .read(self.spatial_grid.params)
            .read(self.particle_buffer)
            .read(self.spatial_grid.counts_linear)
            .read_write(self.grid_neighbor_stats)
            .with_pipeline(self.grid_neighbor_max_pipeline)
            .with_bind_group(0, self.grid_neighbor_max_bind_group)
            .dispatch(self.particles_to_grid_dispatch_x, 1, 1)
            .build()
            .expect("grid neighbor max pass should build");

        let compact_pass = ComputePassBuilder::new("CompactParticles")
            .read(self.dispatch_args)
            .read(self.particle_buffer)
            .write(self.visible_ids)
            .read_write(self.draw_args)
            .with_pipeline(self.compact_pipeline)
            .with_bind_group(0, self.compact_bind_group)
            .dispatch_indirect(self.dispatch_args, 0)
            .build()
            .expect("compact pass should build");

        let mut copy_readback = CopyPassBuilder::new("CopyDrawArgsReadback")
            .copy_buffer_to_frame_slot(
                self.draw_args,
                self.draw_args_readback_slot,
                std::mem::size_of::<DrawIndirectArgs>() as u64,
            )
            .copy_buffer(
                self.draw_args,
                self.draw_args_sync_readback,
                std::mem::size_of::<DrawIndirectArgs>() as u64,
            );
        if self.grid_neighbor_validate {
            copy_readback = copy_readback.copy_buffer(
                self.grid_neighbor_stats,
                self.grid_neighbor_readback,
                4,
            );
        }
        let copy_readback_pass = copy_readback
            .build()
            .expect("copy readback pass should build");

        let render_pass = RenderPassBuilder::new("RenderParticles")
            .read(self.particle_buffer)
            .read(self.visible_ids)
            .with_pipeline(self.render_pipeline)
            .with_bind_group(0, self.render_bind_group)
            .with_frame_color_attachment(
                self.frame_target,
                triad_gpu::ColorLoadOp::Clear(wgpu::Color {
                    r: 0.04,
                    g: 0.05,
                    b: 0.06,
                    a: 1.0,
                }),
            )
            .with_frame_depth_stencil_attachment(
                self.depth_frame,
                DepthLoadOp::Clear(1.0),
                wgpu::StoreOp::Store,
                None,
            )
            .draw_indirect(self.draw_args, 0)
            .build()
            .expect("render pass should build");

        let mut graph = FrameGraph::new();
        graph.add_pass(reset_pass);
        graph.add_pass(simulate_pass);
        graph.add_pass(particles_to_grid_pass);
        graph.add_pass(spatial_grid_pass);
        for p in collision_passes {
            graph.add_pass(p);
        }
        if self.grid_neighbor_validate {
            graph.add_pass(clear_grid_neighbor_pass);
            graph.add_pass(grid_neighbor_pass);
        }
        graph.add_pass(compact_pass);
        graph.add_pass(copy_readback_pass);
        graph.add_pass(render_pass);

        let executable = graph.build_with_cached_order(self.cached_execution_order.as_deref())?;
        self.cached_execution_order = Some(executable.execution_order().to_vec());
        if let Ok(mut stats) = self.stats.lock() {
            stats.graph_build_cpu_ms = build_start.elapsed().as_secs_f32() * 1000.0;
            stats.cached_order_len = self.cached_execution_order.as_ref().map_or(0, Vec::len);
        }
        Ok(executable)
    }

    fn resize(
        &mut self,
        _device: &wgpu::Device,
        _registry: &mut ResourceRegistry,
        width: u32,
        height: u32,
    ) -> Result<(), Box<dyn Error>> {
        self.viewport_w = width.max(1);
        self.viewport_h = height.max(1);
        Ok(())
    }
}

impl ParticleRendererManager {
    fn next_idle_readback_index(&self) -> Option<usize> {
        (1..=self.readback_ring.len())
            .map(|offset| (self.current_readback_index + offset) % self.readback_ring.len())
            .find(|&index| matches!(self.readback_ring[index].state, ReadbackState::Idle))
    }
}

fn init_logging() {
    let filter = std::env::var("RUST_LOG").unwrap_or_else(|_| "info,triad_window=info".to_string());
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .compact()
        .try_init();
}

fn main() -> Result<(), Box<dyn Error>> {
    init_logging();
    info!("starting triad app");
    let particle_count = particle_count_from_env();
    let grid_neighbor_validate = grid_neighbor_validate_from_env();
    info!(
        particle_count,
        grid_neighbor_validate,
        "particle count (override with TRIAD_PARTICLE_COUNT); grid-neighbor GPU validate (TRIAD_GRID_NEIGHBOR_VALIDATE)"
    );
    info!(
        display = std::env::var("DISPLAY").unwrap_or_else(|_| "<unset>".to_string()),
        wayland = std::env::var("WAYLAND_DISPLAY").unwrap_or_else(|_| "<unset>".to_string()),
        xdg_session = std::env::var("XDG_SESSION_TYPE").unwrap_or_else(|_| "<unset>".to_string()),
        "environment"
    );

    let stats = Arc::new(Mutex::new(DemoStats::new(
        particle_count,
        (particle_count as u32).div_ceil(WORKGROUP_SIZE),
        total_cells(SPATIAL_GRID_DIMS),
        grid_neighbor_validate,
    )));
    let ui_stats = Arc::clone(&stats);
    let manager_stats = Arc::clone(&stats);

    let result = run_with_renderer_config(
        "Triad",
        WindowConfig {
            present_mode: wgpu::PresentMode::Fifo,
        },
        |controls| {
            let ui_stats = Arc::clone(&ui_stats);
            controls.on_ui(move |ctx| {
                let Ok(stats) = ui_stats.lock() else {
                    return;
                };
                egui::Window::new("Triad")
                    .default_pos(egui::pos2(16.0, 96.0))
                    .resizable(false)
                    .show(ctx, |ui| {
                        ui.label("Compute-driven particle demo");
                        ui.label(format!(
                            "Stress: set TRIAD_PARTICLE_COUNT ({}..={})",
                            MIN_PARTICLE_COUNT, MAX_PARTICLE_COUNT
                        ));
                        ui.separator();
                        ui.label(format!("Particles: {}", stats.particle_count));
                        ui.label(format!("Workgroups: {}", stats.workgroup_count));
                        ui.label(format!(
                            "Spatial grid: {} cells ({}×{}×{})",
                            stats.spatial_grid_cells,
                            SPATIAL_GRID_DIMS[0],
                            SPATIAL_GRID_DIMS[1],
                            SPATIAL_GRID_DIMS[2]
                        ));
                        if stats.grid_neighbor_validate {
                            ui.label(format!(
                                "Grid max others (3×3 cells, excl. self): {}",
                                stats.grid_max_others_in_3x3
                            ));
                        } else {
                            ui.label(
                                "Grid max others (3×3): disabled (set TRIAD_GRID_NEIGHBOR_VALIDATE=1; costs GPU time)",
                            );
                        }
                        ui.label(format!(
                            "Disk radius (draw + collision): {:.4} world units; {} collision passes/frame",
                            PARTICLE_RADIUS, COLLISION_ITERATIONS
                        ));
                        ui.label("Motion: x += v·dt; walls reflect v; disk contacts apply separation + normal impulse.");
                        ui.label(format!("GPU visible count: {}", stats.gpu_visible_count));
                        ui.label(format!(
                            "GPU visible count sync: {}",
                            stats.gpu_visible_count_sync
                        ));
                        ui.label(format!("Sim dt: {:.3} ms", stats.dt_ms));
                        ui.label(format!("Update CPU: {:.3} ms", stats.update_cpu_ms));
                        ui.label(format!(
                            "Graph build CPU: {:.3} ms",
                            stats.graph_build_cpu_ms
                        ));
                        ui.label(format!("Readback CPU: {:.3} ms", stats.readback_cpu_ms));
                        ui.label(format!(
                            "Readback sync CPU: {:.3} ms",
                            stats.readback_sync_cpu_ms
                        ));
                        ui.label(format!("Readback mismatch: {}", stats.readback_mismatch));
                        ui.label(format!("Cached pass order len: {}", stats.cached_order_len));
                        ui.separator();
                        ui.label(format!(
                            "Frame: reset → simulate → to-grid → spatial rebuild → collision×{} → grid-neighbor max → …",
                            COLLISION_ITERATIONS
                        ));
                        ui.label(
                            "… compact → readback → draw",
                        );
                        ui.label("Depth: Less + Depth32Float (per-instance z from id).");
                        ui.label("Simulation is time-based; GPU resources are persistent.");
                    });
            });
        },
        move |renderer, registry, surface_format, width, height| {
            info!(?surface_format, "creating particle renderer manager");
            Ok(Box::new(ParticleRendererManager::new(
                renderer,
                registry,
                surface_format,
                Arc::clone(&manager_stats),
                particle_count,
                grid_neighbor_validate,
                width,
                height,
            )?))
        },
    );

    if let Err(err) = &result {
        error!(error = %err, "triad app exited with error");
    } else {
        info!("triad app exited cleanly");
    }

    result
}
