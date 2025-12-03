//! Gaussian splatting render delegate implementation.

use crate::{RenderContext, RenderDelegate, SceneBounds};
use glam::{Mat4, Vec3};
use std::error::Error;
use std::path::{Path, PathBuf};
use triad_gpu::{
    CameraUniforms, Handle, RenderPipelineBuilder, Renderer, ResourceRegistry,
    ply_loader,
};
use tracing::info;

/// Initialization data for Gaussian splatting.
pub struct GaussianInitData {
    pub ply_path: PathBuf,
}

impl GaussianInitData {
    pub fn new(ply_path: impl AsRef<Path>) -> Self {
        Self {
            ply_path: ply_path.as_ref().to_path_buf(),
        }
    }
}

/// Gaussian splatting render delegate.
pub struct GaussianDelegate {
    bounds: SceneBounds,
    index_count: u32,
    camera_buffer_handle: Handle<triad_gpu::wgpu::Buffer>,
    index_buffer_handle: Handle<triad_gpu::wgpu::Buffer>,
    bind_group_handle: Handle<triad_gpu::wgpu::BindGroup>,
    pipeline_handle: Handle<triad_gpu::wgpu::RenderPipeline>,
}

impl RenderDelegate for GaussianDelegate {
    type InitData = GaussianInitData;

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
        info!("Loading gaussians from {}", ply_path_str);
        let gaussians = ply_loader::load_gaussians_from_ply(ply_path_str)?;
        info!("Loaded {} gaussians", gaussians.len());

        let bounds = SceneBounds::from_positions(
            gaussians
                .iter()
                .map(|g| Vec3::new(g.position[0], g.position[1], g.position[2])),
        );

        let gaussian_data = bytemuck::cast_slice(&gaussians);
        let gaussian_buffer =
            device.create_buffer_init(&triad_gpu::wgpu::util::BufferInitDescriptor {
                label: Some("Gaussian Buffer"),
                contents: gaussian_data,
                usage: triad_gpu::wgpu::BufferUsages::STORAGE,
            });
        let gaussian_buffer_handle = registry.insert(gaussian_buffer);

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

        let mut indices = Vec::with_capacity(gaussians.len() * 3);
        for i in 0..gaussians.len() as u32 {
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

        let bind_group_layout =
            device.create_bind_group_layout(&triad_gpu::wgpu::BindGroupLayoutDescriptor {
                label: Some("Gaussian Bind Group Layout"),
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

        let bind_group = device.create_bind_group(&triad_gpu::wgpu::BindGroupDescriptor {
            label: Some("Gaussian Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                triad_gpu::wgpu::BindGroupEntry {
                    binding: 0,
                    resource: registry
                        .get(gaussian_buffer_handle)
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

        let vertex_shader_source = include_str!("../../triad-gpu/shaders/gaussian_vertex.wgsl");
        let fragment_shader_source = include_str!("../../triad-gpu/shaders/gaussian_fragment.wgsl");
        let vertex_shader = device.create_shader_module(triad_gpu::wgpu::ShaderModuleDescriptor {
            label: Some("gaussian_vs"),
            source: triad_gpu::wgpu::ShaderSource::Wgsl(vertex_shader_source.into()),
        });
        let fragment_shader = device.create_shader_module(triad_gpu::wgpu::ShaderModuleDescriptor {
            label: Some("gaussian_fs"),
            source: triad_gpu::wgpu::ShaderSource::Wgsl(fragment_shader_source.into()),
        });

        let pipeline_layout =
            device.create_pipeline_layout(&triad_gpu::wgpu::PipelineLayoutDescriptor {
                label: Some("Gaussian Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline_handle = RenderPipelineBuilder::new(device)
            .with_label("Gaussian Pipeline")
            .with_vertex_shader(registry.insert(vertex_shader))
            .with_fragment_shader(registry.insert(fragment_shader))
            .with_layout(pipeline_layout)
            .with_primitive(triad_gpu::wgpu::PrimitiveState {
                topology: triad_gpu::wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: triad_gpu::wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: triad_gpu::wgpu::PolygonMode::Fill,
                conservative: false,
            })
            .with_fragment_target(Some(triad_gpu::wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(triad_gpu::wgpu::BlendState::ALPHA_BLENDING),
                write_mask: triad_gpu::wgpu::ColorWrites::ALL,
            }))
            .build(registry)?;

        Ok(Self {
            bounds,
            index_count: (gaussians.len() as u32) * 3,
            camera_buffer_handle,
            index_buffer_handle,
            bind_group_handle,
            pipeline_handle,
        })
    }

    fn bounds(&self) -> &SceneBounds {
        &self.bounds
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
        let pipeline = registry
            .get(self.pipeline_handle)
            .expect("pipeline");
        let bind_group = registry
            .get(self.bind_group_handle)
            .expect("bind group");
        let index_buffer = registry
            .get(self.index_buffer_handle)
            .expect("index buffer");

        let mut render_pass = encoder.begin_render_pass(&triad_gpu::wgpu::RenderPassDescriptor {
            label: Some("Gaussian Pass"),
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
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.set_index_buffer(index_buffer.slice(..), triad_gpu::wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.index_count, 0, 0..1);
    }
}

/// Run the Gaussian splatting viewer with a PLY file.
pub fn run(ply_path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
    crate::run_with_delegate::<GaussianDelegate>(
        "Triad Gaussian Viewer",
        GaussianInitData::new(ply_path),
    )
}

