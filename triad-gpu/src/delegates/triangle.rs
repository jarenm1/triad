//! Triangle splatting render delegate.

use crate::{
    BufferUsage, CameraUniforms, Handle, RenderContext, RenderDelegate, RenderPipelineBuilder,
    Renderer, ResourceRegistry, SceneBounds, TrianglePrimitive, ply_loader,
};
use glam::{Mat4, Vec3};
use std::error::Error;
use std::path::{Path, PathBuf};
use tracing::info;
use triad_data::triangulation;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

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
    camera_buffer_handle: Handle<wgpu::Buffer>,
    index_buffer_handle: Handle<wgpu::Buffer>,
    bind_group_handle: Handle<wgpu::BindGroup>,
    pipeline_handle: Handle<wgpu::RenderPipeline>,
}

impl RenderDelegate for TriangleDelegate {
    type InitData = TriangleInitData;

    fn create(
        renderer: &Renderer,
        registry: &mut ResourceRegistry,
        surface_format: wgpu::TextureFormat,
        init_data: Self::InitData,
    ) -> Result<Self, Box<dyn Error>> {
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

        // Create triangle buffer using new API
        let triangle_buffer_handle = renderer
            .create_buffer()
            .label("Triangle Buffer")
            .with_pod_data(&triangles)
            .usage(BufferUsage::Storage { read_only: true })
            .build(registry)?;

        // Create camera buffer using new API
        let camera_buffer_handle = renderer
            .create_buffer()
            .label("Camera Buffer")
            .with_pod_data(&[CameraUniforms {
                view_matrix: Mat4::IDENTITY.to_cols_array_2d(),
                proj_matrix: Mat4::IDENTITY.to_cols_array_2d(),
                view_pos: [0.0, 0.0, 0.0],
                _padding: 0.0,
            }])
            .usage(BufferUsage::Uniform)
            .build(registry)?;

        // Create index buffer (3 indices per triangle) using new API
        let triangle_count = triangles.len() as u32;
        let mut indices = Vec::with_capacity((triangle_count * 3) as usize);
        for i in 0..triangle_count {
            let base = i * 3;
            indices.push(base);
            indices.push(base + 1);
            indices.push(base + 2);
        }
        let index_buffer_handle = renderer
            .create_buffer()
            .label("Index Buffer")
            .with_pod_data(&indices)
            .usage(BufferUsage::Index)
            .build(registry)?;

        // Create bind group layout and bind group
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Triangle Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: Some(
                                std::num::NonZeroU64::new(
                                    std::mem::size_of::<CameraUniforms>() as u64,
                                )
                                .unwrap(),
                            ),
                        },
                        count: None,
                    },
                ],
            });
        let bind_group_layout_handle = registry.insert(bind_group_layout);

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Triangle Bind Group"),
            layout: registry
                .get(bind_group_layout_handle)
                .expect("bind group layout"),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: registry
                        .get(triangle_buffer_handle)
                        .expect("triangle buffer")
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: registry
                        .get(camera_buffer_handle)
                        .expect("camera buffer")
                        .as_entire_binding(),
                },
            ],
        });
        let bind_group_handle = registry.insert(bind_group);

        // Load shaders
        let vertex_shader_source = include_str!("../../shaders/triangle_vertex.wgsl");
        let fragment_shader_source = include_str!("../../shaders/triangle_fragment.wgsl");

        let vertex_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("triangle_vs"),
            source: wgpu::ShaderSource::Wgsl(vertex_shader_source.into()),
        });
        let fragment_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("triangle_fs"),
            source: wgpu::ShaderSource::Wgsl(fragment_shader_source.into()),
        });

        // Create pipeline layout
        let bind_group_layout = registry
            .get(bind_group_layout_handle)
            .expect("bind group layout");
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Triangle Pipeline Layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        // Build render pipeline with depth testing
        let pipeline_handle = RenderPipelineBuilder::new(device)
            .with_label("Triangle Splatting Pipeline")
            .with_vertex_shader(registry.insert(vertex_shader))
            .with_fragment_shader(registry.insert(fragment_shader))
            .with_layout(pipeline_layout)
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
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
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

    fn depth_format(&self) -> Option<wgpu::TextureFormat> {
        Some(DEPTH_FORMAT)
    }

    fn update(&mut self, queue: &wgpu::Queue, registry: &ResourceRegistry, camera: &CameraUniforms) {
        let camera_buffer = registry
            .get(self.camera_buffer_handle)
            .expect("camera buffer");
        queue.write_buffer(camera_buffer, 0, bytemuck::bytes_of(camera));
    }

    fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: RenderContext,
        registry: &ResourceRegistry,
    ) {
        let pipeline = registry.get(self.pipeline_handle).expect("pipeline");
        let bind_group = registry.get(self.bind_group_handle).expect("bind group");
        let index_buffer = registry
            .get(self.index_buffer_handle)
            .expect("index buffer");

        let depth_stencil_attachment =
            ctx.depth_view
                .map(|depth_view| wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Triangle Splatting Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.02,
                        g: 0.02,
                        b: 0.025,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.index_count, 0, 0..1);
    }
}

