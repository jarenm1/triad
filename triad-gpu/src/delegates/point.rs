//! Point cloud render delegate.

use crate::{
    BufferUsage, CameraUniforms, Handle, RenderContext, RenderDelegate, RenderPipelineBuilder,
    Renderer, ResourceRegistry, SceneBounds, PointPrimitive,
};
use glam::Mat4;
use std::error::Error;
use std::path::{Path, PathBuf};
use tracing::info;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Initialization data for point cloud rendering.
pub struct PointInitData {
    /// Path to PLY file, or None if providing points directly.
    pub ply_path: Option<PathBuf>,
    /// Direct point data (used if ply_path is None).
    pub points: Option<Vec<PointPrimitive>>,
    /// Point size in world units.
    pub point_size: f32,
}

impl PointInitData {
    /// Create init data from a PLY file path.
    pub fn from_ply(ply_path: impl AsRef<Path>) -> Self {
        Self {
            ply_path: Some(ply_path.as_ref().to_path_buf()),
            points: None,
            point_size: 0.01,
        }
    }

    /// Create init data from direct point data.
    pub fn from_points(points: Vec<PointPrimitive>) -> Self {
        Self {
            ply_path: None,
            points: Some(points),
            point_size: 0.01,
        }
    }

    /// Set the point size.
    pub fn with_point_size(mut self, size: f32) -> Self {
        self.point_size = size;
        self
    }
}

/// Point cloud render delegate.
pub struct PointDelegate {
    bounds: SceneBounds,
    point_count: u32,
    camera_buffer_handle: Handle<wgpu::Buffer>,
    bind_group_handle: Handle<wgpu::BindGroup>,
    pipeline_handle: Handle<wgpu::RenderPipeline>,
}

impl RenderDelegate for PointDelegate {
    type InitData = PointInitData;

    fn create(
        renderer: &Renderer,
        registry: &mut ResourceRegistry,
        surface_format: wgpu::TextureFormat,
        init_data: Self::InitData,
    ) -> Result<Self, Box<dyn Error>> {
        let device = renderer.device();

        // Load points either from PLY or direct data
        let points: Vec<PointPrimitive> = if let Some(ref ply_path) = init_data.ply_path {
            let ply_path_str = ply_path
                .to_str()
                .ok_or_else(|| format!("PLY path {:?} is not valid UTF-8", ply_path))?;

            info!("Loading points from {}", ply_path_str);
            let vertices = crate::ply_loader::load_vertices_from_ply(ply_path_str)?;

            vertices
                .iter()
                .map(|v| PointPrimitive::new(v.position, init_data.point_size, v.color, v.opacity))
                .collect()
        } else if let Some(points) = init_data.points {
            points
        } else {
            return Err("PointInitData must have either ply_path or points".into());
        };

        info!("Loaded {} points", points.len());

        if points.is_empty() {
            return Err("No points to render".into());
        }

        // Compute bounds
        let bounds = SceneBounds::from_positions(points.iter().map(|p| p.position()));

        // Create point buffer
        let point_buffer_handle = renderer
            .create_buffer()
            .label("Point Buffer")
            .with_pod_data(&points)
            .usage(BufferUsage::Storage { read_only: true })
            .build(registry)?;

        // Create camera buffer
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

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Point Bind Group Layout"),
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
                            std::num::NonZeroU64::new(std::mem::size_of::<CameraUniforms>() as u64)
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
            label: Some("Point Bind Group"),
            layout: registry
                .get(bind_group_layout_handle)
                .expect("bind group layout"),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: registry
                        .get(point_buffer_handle)
                        .expect("point buffer")
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
        let vertex_shader_source = include_str!("../../shaders/point_vertex.wgsl");
        let fragment_shader_source = include_str!("../../shaders/point_fragment.wgsl");

        let vertex_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("point_vs"),
            source: wgpu::ShaderSource::Wgsl(vertex_shader_source.into()),
        });
        let fragment_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("point_fs"),
            source: wgpu::ShaderSource::Wgsl(fragment_shader_source.into()),
        });

        // Create pipeline layout
        let bind_group_layout = registry
            .get(bind_group_layout_handle)
            .expect("bind group layout");
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Point Pipeline Layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        // Build render pipeline
        let pipeline_handle = RenderPipelineBuilder::new(device)
            .with_label("Point Cloud Pipeline")
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
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
            point_count: points.len() as u32,
            camera_buffer_handle,
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
            label: Some("Point Cloud Pass"),
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
        // 3 vertices per point (triangle covering quad)
        render_pass.draw(0..self.point_count * 3, 0..1);
    }
}

