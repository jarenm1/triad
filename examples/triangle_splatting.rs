//! Triangle Splatting+ example
//!
//! This example demonstrates Triangle Splatting+ rendering using the triad-window
//! render delegate system. It loads PLY files and renders them as triangles with
//! soft edge falloff.
//!
//! Usage:
//!   cargo run --example triangle_splatting -- <path_to_ply>
//!
//! If the PLY file contains face data, those faces are used directly.
//! Otherwise, Delaunay triangulation is applied to generate triangles from the point cloud.

use glam::{Mat4, Vec3};
use std::error::Error;
use std::path::{Path, PathBuf};
use tracing::info;
use triad_data::triangulation;
use triad_gpu::{
    CameraUniforms, Handle, RenderPipelineBuilder, Renderer, ResourceRegistry, TrianglePrimitive,
    ply_loader,
};
use triad_window::{
    CameraControl, CameraIntent, CameraPose, FrameUpdate, InputState, IntentMode, KeyCode,
    MouseButton, PhysicalKey, RenderContext, RenderDelegate, SceneBounds,
};

const DEPTH_FORMAT: triad_gpu::wgpu::TextureFormat = triad_gpu::wgpu::TextureFormat::Depth32Float;

/// Initialization data for triangle splatting.
pub struct TriangleInitData {
    pub ply_path: PathBuf,
}

impl TriangleInitData {
    pub fn new(ply_path: impl AsRef<Path>) -> Self {
        Self {
            ply_path: ply_path.as_ref().to_path_buf(),
        }
    }
}

/// Triangle splatting render delegate.
pub struct TriangleDelegate {
    bounds: SceneBounds,
    index_count: u32,
    camera_buffer_handle: Handle<triad_gpu::wgpu::Buffer>,
    index_buffer_handle: Handle<triad_gpu::wgpu::Buffer>,
    bind_group_handle: Handle<triad_gpu::wgpu::BindGroup>,
    pipeline_handle: Handle<triad_gpu::wgpu::RenderPipeline>,
}

impl RenderDelegate for TriangleDelegate {
    type InitData = TriangleInitData;

    fn create(
        renderer: &Renderer,
        registry: &mut ResourceRegistry,
        surface_format: triad_gpu::wgpu::TextureFormat,
        init_data: Self::InitData,
    ) -> Result<Self, Box<dyn Error>> {
        use triad_gpu::wgpu::util::DeviceExt;
        let device = renderer.device();

        let ply_path_str = init_data
            .ply_path
            .to_str()
            .ok_or_else(|| format!("PLY path {:?} is not valid UTF-8", init_data.ply_path))?;

        info!("Loading triangles from {}", ply_path_str);

        // Try to load triangles from face data first, fall back to Delaunay triangulation
        let triangles: Vec<TrianglePrimitive> =
            if ply_loader::ply_has_faces(ply_path_str).unwrap_or(false) {
                info!("PLY file has face data, loading triangles directly");
                ply_loader::load_triangles_from_ply(ply_path_str)?
            } else {
                info!("PLY file has no faces, using Delaunay triangulation");
                let vertices = ply_loader::load_vertices_from_ply(ply_path_str)?;

                // Extract positions for triangulation
                let positions: Vec<Vec3> = vertices.iter().map(|v| v.position).collect();

                // Triangulate using triad-data
                let triangle_indices = triangulation::triangulate_points(&positions);

                // Build triangle primitives
                let mut triangles = Vec::with_capacity(triangle_indices.len());
                for [i0, i1, i2] in triangle_indices {
                    let v0 = &vertices[i0];
                    let v1 = &vertices[i1];
                    let v2 = &vertices[i2];

                    // Average vertex colors and opacities
                    let avg_color = (v0.color + v1.color + v2.color) / 3.0;
                    let avg_opacity = (v0.opacity + v1.opacity + v2.opacity) / 3.0;

                    triangles.push(TrianglePrimitive::new(
                        v0.position,
                        v1.position,
                        v2.position,
                        avg_color,
                        avg_opacity,
                    ));
                }
                info!(
                    "Built {} triangle primitives from vertices",
                    triangles.len()
                );
                triangles
            };

        info!("Loaded {} triangles", triangles.len());

        if triangles.is_empty() {
            return Err("No triangles could be generated from the PLY file".into());
        }

        // Compute scene bounds from triangle vertices
        let bounds = SceneBounds::from_positions(
            triangles
                .iter()
                .flat_map(|t| [t.vertex0(), t.vertex1(), t.vertex2()]),
        );

        // Create triangle buffer
        let triangle_data = bytemuck::cast_slice(&triangles);
        let triangle_buffer =
            device.create_buffer_init(&triad_gpu::wgpu::util::BufferInitDescriptor {
                label: Some("Triangle Buffer"),
                contents: triangle_data,
                usage: triad_gpu::wgpu::BufferUsages::STORAGE,
            });
        let triangle_buffer_handle = registry.insert(triangle_buffer);

        // Create camera buffer
        let camera_buffer =
            device.create_buffer_init(&triad_gpu::wgpu::util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: bytemuck::cast_slice(&[CameraUniforms {
                    view_matrix: Mat4::IDENTITY.to_cols_array_2d(),
                    proj_matrix: Mat4::IDENTITY.to_cols_array_2d(),
                    view_pos: [0.0, 0.0, 0.0],
                    _padding: 0.0,
                }]),
                usage: triad_gpu::wgpu::BufferUsages::UNIFORM
                    | triad_gpu::wgpu::BufferUsages::COPY_DST,
            });
        let camera_buffer_handle = registry.insert(camera_buffer);

        // Create index buffer (3 indices per triangle)
        let triangle_count = triangles.len() as u32;
        let mut indices = Vec::with_capacity((triangle_count * 3) as usize);
        for i in 0..triangle_count {
            let base = i * 3;
            indices.push(base);
            indices.push(base + 1);
            indices.push(base + 2);
        }
        let index_buffer =
            device.create_buffer_init(&triad_gpu::wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: triad_gpu::wgpu::BufferUsages::INDEX,
            });
        let index_buffer_handle = registry.insert(index_buffer);

        // Create bind group layout
        let bind_group_layout =
            device.create_bind_group_layout(&triad_gpu::wgpu::BindGroupLayoutDescriptor {
                label: Some("Triangle Bind Group Layout"),
                entries: &[
                    triad_gpu::wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: triad_gpu::wgpu::ShaderStages::VERTEX,
                        ty: triad_gpu::wgpu::BindingType::Buffer {
                            ty: triad_gpu::wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    triad_gpu::wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: triad_gpu::wgpu::ShaderStages::VERTEX,
                        ty: triad_gpu::wgpu::BindingType::Buffer {
                            ty: triad_gpu::wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: Some(
                                std::num::NonZeroU64::new(
                                    std::mem::size_of::<CameraUniforms>() as u64
                                )
                                .unwrap(),
                            ),
                        },
                        count: None,
                    },
                ],
            });

        // Create bind group
        let bind_group = device.create_bind_group(&triad_gpu::wgpu::BindGroupDescriptor {
            label: Some("Triangle Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                triad_gpu::wgpu::BindGroupEntry {
                    binding: 0,
                    resource: registry
                        .get(triangle_buffer_handle)
                        .unwrap()
                        .as_entire_binding(),
                },
                triad_gpu::wgpu::BindGroupEntry {
                    binding: 1,
                    resource: registry
                        .get(camera_buffer_handle)
                        .unwrap()
                        .as_entire_binding(),
                },
            ],
        });
        let bind_group_handle = registry.insert(bind_group);

        // Load shaders
        let vertex_shader_source = include_str!("../triad-gpu/shaders/triangle_vertex.wgsl");
        let fragment_shader_source = include_str!("../triad-gpu/shaders/triangle_fragment.wgsl");

        let vertex_shader = device.create_shader_module(triad_gpu::wgpu::ShaderModuleDescriptor {
            label: Some("triangle_vs"),
            source: triad_gpu::wgpu::ShaderSource::Wgsl(vertex_shader_source.into()),
        });
        let fragment_shader =
            device.create_shader_module(triad_gpu::wgpu::ShaderModuleDescriptor {
                label: Some("triangle_fs"),
                source: triad_gpu::wgpu::ShaderSource::Wgsl(fragment_shader_source.into()),
            });

        // Create pipeline layout
        let pipeline_layout =
            device.create_pipeline_layout(&triad_gpu::wgpu::PipelineLayoutDescriptor {
                label: Some("Triangle Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        // Build render pipeline with depth testing
        let pipeline_handle = RenderPipelineBuilder::new(device)
            .with_label("Triangle Splatting Pipeline")
            .with_vertex_shader(registry.insert(vertex_shader))
            .with_fragment_shader(registry.insert(fragment_shader))
            .with_layout(pipeline_layout)
            .with_primitive(triad_gpu::wgpu::PrimitiveState {
                topology: triad_gpu::wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: triad_gpu::wgpu::FrontFace::Ccw,
                cull_mode: None, // No culling for proper rendering from all angles
                unclipped_depth: false,
                polygon_mode: triad_gpu::wgpu::PolygonMode::Fill,
                conservative: false,
            })
            .with_fragment_target(Some(triad_gpu::wgpu::ColorTargetState {
                format: surface_format,
                // Premultiplied alpha blending for smooth triangle edge blending
                blend: Some(triad_gpu::wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: triad_gpu::wgpu::ColorWrites::ALL,
            }))
            .with_depth_stencil(triad_gpu::wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: triad_gpu::wgpu::CompareFunction::Less,
                stencil: triad_gpu::wgpu::StencilState::default(),
                bias: triad_gpu::wgpu::DepthBiasState::default(),
            })
            .build(registry)?;

        Ok(Self {
            bounds,
            index_count: triangle_count * 3,
            camera_buffer_handle,
            index_buffer_handle,
            bind_group_handle,
            pipeline_handle,
        })
    }

    fn bounds(&self) -> &SceneBounds {
        &self.bounds
    }

    fn depth_format(&self) -> Option<triad_gpu::wgpu::TextureFormat> {
        Some(DEPTH_FORMAT)
    }

    fn update(
        &mut self,
        queue: &triad_gpu::wgpu::Queue,
        registry: &ResourceRegistry,
        camera: &CameraUniforms,
    ) {
        let camera_buffer = registry
            .get(self.camera_buffer_handle)
            .expect("camera buffer");
        queue.write_buffer(camera_buffer, 0, bytemuck::bytes_of(camera));
    }

    fn render(
        &self,
        encoder: &mut triad_gpu::wgpu::CommandEncoder,
        ctx: RenderContext,
        registry: &ResourceRegistry,
    ) {
        let pipeline = registry.get(self.pipeline_handle).expect("pipeline");
        let bind_group = registry.get(self.bind_group_handle).expect("bind group");
        let index_buffer = registry
            .get(self.index_buffer_handle)
            .expect("index buffer");

        // Set up depth stencil attachment if depth view is available
        let depth_stencil_attachment =
            ctx.depth_view.map(
                |depth_view| triad_gpu::wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(triad_gpu::wgpu::Operations {
                        load: triad_gpu::wgpu::LoadOp::Clear(1.0),
                        store: triad_gpu::wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                },
            );

        let mut render_pass = encoder.begin_render_pass(&triad_gpu::wgpu::RenderPassDescriptor {
            label: Some("Triangle Splatting Pass"),
            color_attachments: &[Some(triad_gpu::wgpu::RenderPassColorAttachment {
                view: ctx.color_view,
                resolve_target: None,
                ops: triad_gpu::wgpu::Operations {
                    load: triad_gpu::wgpu::LoadOp::Clear(triad_gpu::wgpu::Color {
                        r: 0.02,
                        g: 0.02,
                        b: 0.025,
                        a: 1.0,
                    }),
                    store: triad_gpu::wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.set_index_buffer(index_buffer.slice(..), triad_gpu::wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.index_count, 0, 0..1);
    }
}

/// Blender-like orbit controller: orbit around a focus point, wheel to zoom, shift+drag to pan focus.
struct OrbitCam {
    focus: Vec3,
    yaw: f32,
    pitch: f32,
    radius: f32,
    rotation_sensitivity: f32,
    pan_sensitivity: f32,
    zoom_sensitivity: f32,
}

impl OrbitCam {
    fn new(focus: Vec3, radius: f32) -> Self {
        Self {
            focus,
            yaw: 0.0,
            pitch: 0.0,
            radius,
            rotation_sensitivity: 0.005,
            pan_sensitivity: 0.0025,
            zoom_sensitivity: 0.5,
        }
    }

    fn pose_from_state(&self) -> CameraPose {
        let dir = glam::Quat::from_euler(glam::EulerRot::YXZ, self.yaw, self.pitch, 0.0) * -Vec3::Z;
        let position = self.focus - dir.normalize() * self.radius;
        let forward = (self.focus - position).normalize_or_zero();
        let yaw = forward.x.atan2(-forward.z);
        let pitch = forward
            .y
            .asin()
            .clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
        CameraPose {
            position,
            yaw,
            pitch,
            roll: 0.0,
        }
    }
}

impl CameraControl for OrbitCam {
    fn update(
        &mut self,
        _dt: f32,
        input: &InputState,
        _current: &CameraPose,
    ) -> Option<CameraIntent> {
        let delta = input.mouse_delta();
        let shift = input.key_down(PhysicalKey::Code(KeyCode::ShiftLeft))
            || input.key_down(PhysicalKey::Code(KeyCode::ShiftRight));

        if input.mouse_button_down(MouseButton::Left) && !shift {
            self.yaw -= delta.x * self.rotation_sensitivity;
            self.pitch = (self.pitch - delta.y * self.rotation_sensitivity).clamp(
                -std::f32::consts::FRAC_PI_2 + 0.01,
                std::f32::consts::FRAC_PI_2 - 0.01,
            );
        } else if input.mouse_button_down(MouseButton::Left) && shift {
            // Pan the focus point in view plane.
            let right = glam::Quat::from_euler(glam::EulerRot::YXZ, self.yaw, 0.0, 0.0) * Vec3::X;
            let up = Vec3::Y;
            let pan = (-delta.x * self.pan_sensitivity * self.radius) * right
                + (delta.y * self.pan_sensitivity * self.radius) * up;
            self.focus += pan;
        }

        if input.scroll_delta() != 0.0 {
            let factor = 1.0 - input.scroll_delta() * 0.1 * self.zoom_sensitivity;
            self.radius = (self.radius * factor).clamp(0.2, 500.0);
        }

        Some(CameraIntent {
            pose: self.pose_from_state(),
            mode: IntentMode::Override,
        })
    }

    fn on_reset(&mut self, _pose: &CameraPose) {
        self.focus = Vec3::ZERO;
        self.yaw = 0.0;
        self.pitch = 0.0;
        self.radius = 10.0;
    }
}

fn compute_scene_center(ply_path: &Path) -> Result<Vec3, Box<dyn Error>> {
    // Reuse the same triangle loading logic as the delegate to derive bounds.
    let ply_path_str = ply_path
        .to_str()
        .ok_or_else(|| format!("PLY path {:?} is not valid UTF-8", ply_path))?;

    let triangles: Vec<TrianglePrimitive> =
        if ply_loader::ply_has_faces(ply_path_str).unwrap_or(false) {
            ply_loader::load_triangles_from_ply(ply_path_str)?
        } else {
            let vertices = ply_loader::load_vertices_from_ply(ply_path_str)?;

            // Extract positions for triangulation
            let positions: Vec<Vec3> = vertices.iter().map(|v| v.position).collect();

            // Triangulate using triad-data
            let triangle_indices = triangulation::triangulate_points(&positions);

            // Build triangle primitives
            let mut triangles = Vec::with_capacity(triangle_indices.len());
            for [i0, i1, i2] in triangle_indices {
                let v0 = &vertices[i0];
                let v1 = &vertices[i1];
                let v2 = &vertices[i2];

                // Average vertex colors and opacities
                let avg_color = (v0.color + v1.color + v2.color) / 3.0;
                let avg_opacity = (v0.opacity + v1.opacity + v2.opacity) / 3.0;

                triangles.push(TrianglePrimitive::new(
                    v0.position,
                    v1.position,
                    v2.position,
                    avg_color,
                    avg_opacity,
                ));
            }
            triangles
        };

    if triangles.is_empty() {
        return Err("No triangles could be generated from the PLY file".into());
    }

    let bounds = SceneBounds::from_positions(
        triangles
            .iter()
            .flat_map(|t| [t.vertex0(), t.vertex1(), t.vertex2()]),
    );
    Ok(bounds.center)
}

fn main() {
    let ply_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .expect("failed to find ply file, or not provided.");

    let scene_center = compute_scene_center(&ply_path).unwrap_or(Vec3::ZERO);

    if let Err(err) = triad_window::run_with_delegate_config::<TriangleDelegate, _>(
        "Triad Triangle Splatting+ Viewer",
        TriangleInitData::new(&ply_path),
        move |controls| {
            let target = scene_center;
            // Orbit around a focus point (default origin); shift+drag pans focus, wheel zooms.
            controls
                .clear_controllers()
                .single_active(true)
                .add_controller_with_priority(Box::new(OrbitCam::new(target, 10.0)), 0)
                .on_frame(move |ctx: FrameUpdate| {
                    if ctx.input.just_pressed(PhysicalKey::Code(KeyCode::KeyR)) {
                        let mut pose = *ctx.pose;
                        pose.position = target + Vec3::new(0.0, 0.0, 10.0);
                        let forward = (target - pose.position).normalize_or_zero();
                        pose.yaw = forward.x.atan2(-forward.z);
                        pose.pitch = forward
                            .y
                            .asin()
                            .clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
                        pose.roll = 0.0;
                        *ctx.reset_request = Some(pose);
                    }
                });
        },
    ) {
        eprintln!("triangle_splatting failed: {err}");
    }
}
