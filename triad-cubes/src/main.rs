//! Instanced unit cubes with per-pixel lighting and depth testing.
//! Draw path: `draw_indexed_indirect` (`DrawIndexedIndirectArgs`) so the GPU reads draw parameters
//! from a buffer (updated each frame when pulse mode is enabled).

use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

use tracing::{error, info};
use triad_gpu::{
    BindingType, BufferUsage, DepthLoadOp, DrawIndexedIndirectArgs, ExecutableFrameGraph, FrameGraph,
    FrameGraphError, FrameTextureView, Handle, RenderPassBuilder, Renderer, ResourceRegistry, ShaderStage,
    wgpu,
};

/// `'static` vertex layouts for [`wgpu::RenderPipelineDescriptor`] (no temporary `vertex_attr_array!`).
static CUBE_MESH_VERTEX_ATTRS: [wgpu::VertexAttribute; 2] = [
    wgpu::VertexAttribute {
        offset: 0,
        shader_location: 0,
        format: wgpu::VertexFormat::Float32x3,
    },
    wgpu::VertexAttribute {
        offset: 12,
        shader_location: 1,
        format: wgpu::VertexFormat::Float32x3,
    },
];

static CUBE_INSTANCE_ATTRS: [wgpu::VertexAttribute; 2] = [
    wgpu::VertexAttribute {
        offset: 0,
        shader_location: 2,
        format: wgpu::VertexFormat::Float32x3,
    },
    wgpu::VertexAttribute {
        offset: 16,
        shader_location: 3,
        format: wgpu::VertexFormat::Float32x3,
    },
];
use triad_window::{CameraUniforms, RendererManager, WindowConfig, egui, run_with_renderer_config};

const WGSL: &str = r#"
struct CameraUniforms {
    view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    view_pos: vec3<f32>,
    _padding: f32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) inst_offset: vec3<f32>,
    @location(3) inst_color: vec3<f32>,
}

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) color: vec3<f32>,
}

@vertex
fn vs_main(v: VsIn) -> VsOut {
    let world_pos = v.pos + v.inst_offset;
    var out: VsOut;
    out.clip_pos = camera.proj_matrix * camera.view_matrix * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.world_n = normalize(v.normal);
    out.color = v.inst_color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.35, 0.85, 0.4));
    let n = normalize(in.world_n);
    let ndl = max(dot(n, light_dir), 0.0);
    let ambient = 0.18;
    let rgb = in.color * (ambient + (1.0 - ambient) * ndl);
    return vec4<f32>(rgb, 1.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MeshVertex {
    position: [f32; 3],
    normal: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CubeInstance {
    offset: [f32; 3],
    _pad0: f32,
    color: [f32; 3],
    _pad1: f32,
}

fn push_quad(
    verts: &mut Vec<MeshVertex>,
    indices: &mut Vec<u16>,
    corners: [[f32; 3]; 4],
    normal: [f32; 3],
) {
    let base = verts.len() as u16;
    for p in corners {
        verts.push(MeshVertex {
            position: p,
            normal,
        });
    }
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

/// Axis-aligned cube centered at origin, half-extent `h`. Winding is CCW on each face for back-face culling.
fn build_cube_mesh(h: f32) -> (Vec<MeshVertex>, Vec<u16>) {
    let mut verts = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    let p = h;
    let n = -h;
    // +Z
    push_quad(
        &mut verts,
        &mut indices,
        [
            [n, n, p],
            [p, n, p],
            [p, p, p],
            [n, p, p],
        ],
        [0.0, 0.0, 1.0],
    );
    // -Z
    push_quad(
        &mut verts,
        &mut indices,
        [
            [p, n, n],
            [n, n, n],
            [n, p, n],
            [p, p, n],
        ],
        [0.0, 0.0, -1.0],
    );
    // +X — CCW when viewed from +X so (v1−v0)×(v2−v0) points along +X (matches `FrontFace::Ccw` + back cull).
    push_quad(
        &mut verts,
        &mut indices,
        [
            [p, n, n],
            [p, p, n],
            [p, p, p],
            [p, n, p],
        ],
        [1.0, 0.0, 0.0],
    );
    // -X
    push_quad(
        &mut verts,
        &mut indices,
        [
            [n, n, p],
            [n, p, p],
            [n, p, n],
            [n, n, n],
        ],
        [-1.0, 0.0, 0.0],
    );
    // +Y
    push_quad(
        &mut verts,
        &mut indices,
        [
            [n, p, n],
            [n, p, p],
            [p, p, p],
            [p, p, n],
        ],
        [0.0, 1.0, 0.0],
    );
    // -Y
    push_quad(
        &mut verts,
        &mut indices,
        [
            [n, n, n],
            [p, n, n],
            [p, n, p],
            [n, n, p],
        ],
        [0.0, -1.0, 0.0],
    );
    (verts, indices)
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> [f32; 3] {
    let h = h.rem_euclid(1.0);
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let hue_to_rgb = |t: f32| {
        let t = t.rem_euclid(1.0);
        if t < 1.0 / 6.0 {
            p + (q - p) * 6.0 * t
        } else if t < 0.5 {
            q
        } else if t < 2.0 / 3.0 {
            p + (q - p) * (2.0 / 3.0 - t) * 6.0
        } else {
            p
        }
    };
    [
        hue_to_rgb(h + 1.0 / 3.0),
        hue_to_rgb(h),
        hue_to_rgb(h - 1.0 / 3.0),
    ]
}

fn build_instances(nx: u32, ny: u32, nz: u32, spacing: f32) -> Vec<CubeInstance> {
    let mut out = Vec::with_capacity((nx * ny * nz) as usize);
    let fx = (nx.saturating_sub(1)) as f32 * 0.5;
    let fy = (ny.saturating_sub(1)) as f32 * 0.5;
    let fz = (nz.saturating_sub(1)) as f32 * 0.5;
    let mut i = 0u32;
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let ox = (ix as f32 - fx) * spacing;
                let oy = (iy as f32 - fy) * spacing;
                let oz = (iz as f32 - fz) * spacing * 1.15;
                let t = (i as f32) * 0.127_347;
                let c = hsl_to_rgb(t, 0.55, 0.52);
                out.push(CubeInstance {
                    offset: [ox, oy, oz],
                    _pad0: 0.0,
                    color: c,
                    _pad1: 0.0,
                });
                i += 1;
            }
        }
    }
    out
}

fn grid_from_env() -> (u32, u32, u32) {
    let raw = std::env::var("TRIAD_CUBE_GRID").unwrap_or_else(|_| "14,14,5".to_string());
    let parts: Vec<u32> = raw
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if parts.len() == 3 {
        let x = parts[0].clamp(1, 256);
        let y = parts[1].clamp(1, 256);
        let z = parts[2].clamp(1, 256);
        let total = (x as u64) * (y as u64) * (z as u64);
        if total > 2_000_000 {
            error!(x, y, z, total, "TRIAD_CUBE_GRID product too large; clamping to 64³");
            return (64, 64, 64);
        }
        (x, y, z)
    } else {
        error!(raw, "TRIAD_CUBE_GRID must be three comma-separated integers (e.g. 12,10,6); using default");
        (14, 14, 5)
    }
}

fn pulse_from_env() -> bool {
    matches!(
        std::env::var("TRIAD_CUBE_PULSE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true")),
        Ok(true)
    )
}

struct CubesManager {
    mesh_vertices: Handle<wgpu::Buffer>,
    mesh_indices: Handle<wgpu::Buffer>,
    instance_buffer: Handle<wgpu::Buffer>,
    camera_buffer: Handle<wgpu::Buffer>,
    indirect_buffer: Handle<wgpu::Buffer>,
    render_pipeline: Handle<wgpu::RenderPipeline>,
    render_bind_group: Handle<wgpu::BindGroup>,
    frame_target: Handle<FrameTextureView>,
    depth_frame: Handle<FrameTextureView>,
    index_count: u32,
    active_instances: u32,
    pulse_indirect: bool,
    start: Instant,
}

impl CubesManager {
    fn new(
        renderer: &Renderer,
        registry: &mut ResourceRegistry,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Result<Self, Box<dyn Error>> {
        let _ = (width, height);
        let (nx, ny, nz) = grid_from_env();
        let instances = build_instances(nx, ny, nz, 0.82);
        let active_instances = instances.len() as u32;
        let (mesh_v, mesh_i) = build_cube_mesh(0.38);
        let index_count = mesh_i.len() as u32;

        let mesh_vertices = renderer
            .create_gpu_buffer::<MeshVertex>()
            .label("cube mesh vertices")
            .with_data(&mesh_v)
            .usage(BufferUsage::Vertex)
            .build(registry)?;
        let mesh_indices = renderer
            .create_gpu_buffer::<u16>()
            .label("cube mesh indices")
            .with_data(&mesh_i)
            .usage(BufferUsage::Index)
            .build(registry)?;
        let instance_buffer = renderer
            .create_gpu_buffer::<CubeInstance>()
            .label("cube instances")
            .with_data(&instances)
            .usage(BufferUsage::Vertex)
            .build(registry)?;

        let camera_buffer = renderer
            .create_gpu_buffer::<CameraUniforms>()
            .label("cubes camera uniform")
            .capacity(1)
            .usage(BufferUsage::Uniform)
            .build(registry)?;

        let indirect_init = [DrawIndexedIndirectArgs::new(index_count, active_instances, 0, 0, 0)];
        let indirect_buffer = renderer
            .create_gpu_buffer::<DrawIndexedIndirectArgs>()
            .label("cube draw indexed indirect")
            .with_data(&indirect_init)
            .usage(BufferUsage::Indirect)
            .add_usage(wgpu::BufferUsages::COPY_DST)
            .build(registry)?;

        let shader = renderer
            .create_shader_module()
            .label("cubes shader")
            .with_wgsl_source(WGSL)
            .build(registry)?;

        let (camera_layout, camera_bind_group) = renderer
            .create_bind_group()
            .label("cubes camera")
            .buffer_stage(0, ShaderStage::Vertex, camera_buffer.handle(), BindingType::Uniform)
            .build(registry)?;

        let pipeline_layout = renderer.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cubes pipeline layout"),
            bind_group_layouts: &[registry.get(camera_layout).expect("bind group layout")],
            push_constant_ranges: &[],
        });

        let mesh_stride = std::mem::size_of::<MeshVertex>() as wgpu::BufferAddress;
        let inst_stride = std::mem::size_of::<CubeInstance>() as wgpu::BufferAddress;

        let render_pipeline = renderer
            .create_render_pipeline()
            .with_label("cubes render pipeline")
            .with_vertex_shader(shader)
            .with_fragment_shader(shader)
            .with_layout(pipeline_layout)
            .with_vertex_buffer(wgpu::VertexBufferLayout {
                array_stride: mesh_stride,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &CUBE_MESH_VERTEX_ATTRS,
            })
            .with_vertex_buffer(wgpu::VertexBufferLayout {
                array_stride: inst_stride,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &CUBE_INSTANCE_ATTRS,
            })
            .with_primitive(wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
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

        let frame_target = registry.insert(FrameTextureView::new());
        let depth_frame = registry.insert(FrameTextureView::new());

        Ok(Self {
            mesh_vertices: mesh_vertices.handle(),
            mesh_indices: mesh_indices.handle(),
            instance_buffer: instance_buffer.handle(),
            camera_buffer: camera_buffer.handle(),
            indirect_buffer: indirect_buffer.handle(),
            render_pipeline,
            render_bind_group: camera_bind_group,
            frame_target,
            depth_frame,
            index_count,
            active_instances,
            pulse_indirect: pulse_from_env(),
            start: Instant::now(),
        })
    }
}

impl RendererManager for CubesManager {
    fn update(
        &mut self,
        renderer: &Renderer,
        registry: &mut ResourceRegistry,
        camera: &CameraUniforms,
    ) -> Result<(), Box<dyn Error>> {
        renderer.write_buffer(self.camera_buffer, &[*camera], registry)?;

        let mut count = self.active_instances;
        if self.pulse_indirect {
            let t = self.start.elapsed().as_secs_f32();
            let wave = (t * 0.85).sin() * 0.5 + 0.5;
            count = ((self.active_instances as f32) * (0.2 + 0.8 * wave))
                .round()
                .clamp(1.0, self.active_instances as f32) as u32;
        }
        let args = [DrawIndexedIndirectArgs::new(self.index_count, count, 0, 0, 0)];
        renderer.write_buffer(self.indirect_buffer, &args, registry)?;
        Ok(())
    }

    fn prepare_frame(
        &mut self,
        registry: &mut ResourceRegistry,
        final_view: Arc<wgpu::TextureView>,
        depth_view: Option<Arc<wgpu::TextureView>>,
    ) -> Result<bool, Box<dyn Error>> {
        registry
            .get(self.frame_target)
            .expect("frame slot")
            .set(final_view);
        if let Some(d) = depth_view {
            registry.get(self.depth_frame).expect("depth slot").set(d);
        }
        Ok(false)
    }

    fn build_frame_graph(&mut self) -> Result<ExecutableFrameGraph, FrameGraphError> {
        let render_pass = RenderPassBuilder::new("DrawCubesIndexedIndirect")
            .read(self.mesh_vertices)
            .read(self.instance_buffer)
            .read(self.camera_buffer)
            .with_pipeline(self.render_pipeline)
            .with_bind_group(0, self.render_bind_group)
            .with_vertex_buffer(0, self.mesh_vertices)
            .with_vertex_buffer(1, self.instance_buffer)
            .with_index_buffer(self.mesh_indices, wgpu::IndexFormat::Uint16)
            .with_frame_color_attachment(
                self.frame_target,
                triad_gpu::ColorLoadOp::Clear(wgpu::Color {
                    r: 0.02,
                    g: 0.025,
                    b: 0.04,
                    a: 1.0,
                }),
            )
            .with_frame_depth_stencil_attachment(
                self.depth_frame,
                DepthLoadOp::Clear(1.0),
                wgpu::StoreOp::Store,
                None,
            )
            .draw_indexed_indirect(self.indirect_buffer, 0)
            .build()
            .expect("cubes render pass should build");

        let mut graph = FrameGraph::new();
        graph.add_pass(render_pass);
        graph.build()
    }

    fn resize(
        &mut self,
        _device: &wgpu::Device,
        _registry: &mut ResourceRegistry,
        _width: u32,
        _height: u32,
    ) -> Result<(), Box<dyn Error>> {
        Ok(())
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
    let (nx, ny, nz) = grid_from_env();
    let pulse = pulse_from_env();
    info!(
        nx,
        ny,
        nz,
        total = nx * ny * nz,
        pulse,
        "triad-cubes (override grid with TRIAD_CUBE_GRID=nx,ny,nz; TRIAD_CUBE_PULSE=1 animates indirect instance_count)"
    );

    let grid_label = format!("{nx}×{ny}×{nz}");
    run_with_renderer_config(
        "Triad cubes",
        WindowConfig {
            present_mode: wgpu::PresentMode::Fifo,
        },
        move |controls| {
            let grid_label = grid_label.clone();
            controls.on_ui(move |ctx| {
                egui::Window::new("Triad cubes")
                    .default_pos(egui::pos2(16.0, 96.0))
                    .resizable(false)
                    .show(ctx, |ui| {
                        ui.label("Instanced cubes · depth test · draw_indexed_indirect");
                        ui.label(format!("Grid: {grid_label}"));
                        ui.label(format!(
                            "Indirect pulse: {} (TRIAD_CUBE_PULSE=1)",
                            if pulse { "on" } else { "off" }
                        ));
                        ui.separator();
                        ui.label("Orbit / pan / zoom with mouse; depth shows cubes behind cubes.");
                    });
            });
        },
        move |renderer, registry, surface_format, width, height| {
            info!(?surface_format, "creating cubes manager");
            Ok(Box::new(CubesManager::new(
                renderer,
                registry,
                surface_format,
                width,
                height,
            )?))
        },
    )?;
    Ok(())
}
