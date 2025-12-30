//! Multi-delegate renderer that supports hot-swapping between visualization modes.

use crate::app::ModeSignal;
use crate::layers::LayerMode;
use glam::{Mat4, Vec3};
use std::error::Error;
use std::path::PathBuf;
use tracing::info;
use triad_gpu::{
    BufferUsage, CameraUniforms, Handle, PointPrimitive, RenderContext,
    RenderDelegate, RenderPipelineBuilder, Renderer, ResourceRegistry, SceneBounds,
    TrianglePrimitive, ply_loader,
};
use triad_data::triangulation;

const DEPTH_FORMAT: triad_gpu::wgpu::TextureFormat = triad_gpu::wgpu::TextureFormat::Depth32Float;

/// Initialization data for the multi-delegate renderer.
pub struct MultiInitData {
    pub ply_path: PathBuf,
    pub initial_mode: LayerMode,
    pub point_size: f32,
    pub mode_signal: ModeSignal,
}

impl MultiInitData {
    pub fn new(ply_path: PathBuf, initial_mode: LayerMode, mode_signal: ModeSignal) -> Self {
        Self {
            ply_path,
            initial_mode,
            point_size: 0.01,
            mode_signal,
        }
    }
}

/// Holds GPU resources for a single visualization mode.
struct ModeResources {
    bind_group: Handle<triad_gpu::wgpu::BindGroup>,
    pipeline: Handle<triad_gpu::wgpu::RenderPipeline>,
    index_buffer: Option<Handle<triad_gpu::wgpu::Buffer>>,
    vertex_count: u32,
    index_count: u32,
    uses_indices: bool,
}

/// Multi-delegate renderer that can switch between Points, Gaussians, and Triangles.
pub struct MultiDelegate {
    bounds: SceneBounds,
    current_mode: LayerMode,
    
    // Mode signal for external control (read during update)
    mode_signal: ModeSignal,
    
    // Shared camera buffer
    camera_buffer: Handle<triad_gpu::wgpu::Buffer>,
    
    // Per-mode resources
    point_resources: ModeResources,
    gaussian_resources: ModeResources,
    triangle_resources: ModeResources,
}

impl MultiDelegate {
    fn current_resources(&self) -> &ModeResources {
        match self.current_mode {
            LayerMode::Points => &self.point_resources,
            LayerMode::Gaussians => &self.gaussian_resources,
            LayerMode::Triangles => &self.triangle_resources,
        }
    }
}

impl RenderDelegate for MultiDelegate {
    type InitData = MultiInitData;

    fn create(
        renderer: &Renderer,
        registry: &mut ResourceRegistry,
        surface_format: triad_gpu::wgpu::TextureFormat,
        init_data: Self::InitData,
    ) -> Result<Self, Box<dyn Error>> {
        let device = renderer.device();
        let ply_path_str = init_data.ply_path.to_str()
            .ok_or_else(|| format!("Invalid PLY path: {:?}", init_data.ply_path))?;

        info!("Loading PLY data from {}", ply_path_str);

        // Load vertices once - this is the base data
        let vertices = ply_loader::load_vertices_from_ply(ply_path_str)?;
        info!("Loaded {} vertices", vertices.len());

        if vertices.is_empty() {
            return Err("No vertices in PLY file".into());
        }

        // Compute bounds from vertices
        let bounds = SceneBounds::from_positions(vertices.iter().map(|v| v.position));

        // Create shared camera buffer
        let camera_buffer = renderer
            .create_buffer()
            .label("Shared Camera Buffer")
            .with_pod_data(&[CameraUniforms {
                view_matrix: Mat4::IDENTITY.to_cols_array_2d(),
                proj_matrix: Mat4::IDENTITY.to_cols_array_2d(),
                view_pos: [0.0, 0.0, 0.0],
                _padding: 0.0,
            }])
            .usage(BufferUsage::Uniform)
            .build(registry)?;

        // === Create Point Resources ===
        let point_resources = {
            let points: Vec<PointPrimitive> = vertices
                .iter()
                .map(|v| PointPrimitive::new(v.position, init_data.point_size, v.color, v.opacity))
                .collect();

            let point_buffer = renderer
                .create_buffer()
                .label("Point Buffer")
                .with_pod_data(&points)
                .usage(BufferUsage::Storage { read_only: true })
                .build(registry)?;

            let bind_group_layout = device.create_bind_group_layout(&triad_gpu::wgpu::BindGroupLayoutDescriptor {
                label: Some("Point Bind Group Layout"),
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
                            min_binding_size: Some(std::num::NonZeroU64::new(std::mem::size_of::<CameraUniforms>() as u64).unwrap()),
                        },
                        count: None,
                    },
                ],
            });

            let bind_group = device.create_bind_group(&triad_gpu::wgpu::BindGroupDescriptor {
                label: Some("Point Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    triad_gpu::wgpu::BindGroupEntry {
                        binding: 0,
                        resource: registry.get(point_buffer).unwrap().as_entire_binding(),
                    },
                    triad_gpu::wgpu::BindGroupEntry {
                        binding: 1,
                        resource: registry.get(camera_buffer).unwrap().as_entire_binding(),
                    },
                ],
            });

            let vertex_shader = device.create_shader_module(triad_gpu::wgpu::ShaderModuleDescriptor {
                label: Some("point_vs"),
                source: triad_gpu::wgpu::ShaderSource::Wgsl(triad_gpu::shaders::POINT_VERTEX.into()),
            });
            let fragment_shader = device.create_shader_module(triad_gpu::wgpu::ShaderModuleDescriptor {
                label: Some("point_fs"),
                source: triad_gpu::wgpu::ShaderSource::Wgsl(triad_gpu::shaders::POINT_FRAGMENT.into()),
            });

            let pipeline_layout = device.create_pipeline_layout(&triad_gpu::wgpu::PipelineLayoutDescriptor {
                label: Some("Point Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

            let pipeline = RenderPipelineBuilder::new(device)
                .with_label("Point Pipeline")
                .with_vertex_shader(registry.insert(vertex_shader))
                .with_fragment_shader(registry.insert(fragment_shader))
                .with_layout(pipeline_layout)
                .with_primitive(triad_gpu::wgpu::PrimitiveState {
                    topology: triad_gpu::wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                })
                .with_fragment_target(Some(triad_gpu::wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(triad_gpu::wgpu::BlendState::ALPHA_BLENDING),
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

            ModeResources {
                bind_group: registry.insert(bind_group),
                pipeline,
                index_buffer: None,
                vertex_count: points.len() as u32 * 3,
                index_count: 0,
                uses_indices: false,
            }
        };

        // === Create Gaussian Resources ===
        let gaussian_resources = {
            let gaussians = ply_loader::load_gaussians_from_ply(ply_path_str)?;

            let gaussian_buffer = renderer
                .create_buffer()
                .label("Gaussian Buffer")
                .with_pod_data(&gaussians)
                .usage(BufferUsage::Storage { read_only: true })
                .build(registry)?;

            let mut indices = Vec::with_capacity(gaussians.len() * 3);
            for i in 0..gaussians.len() as u32 {
                let base = i * 3;
                indices.push(base);
                indices.push(base + 1);
                indices.push(base + 2);
            }
            let index_buffer = renderer
                .create_buffer()
                .label("Gaussian Index Buffer")
                .with_pod_data(&indices)
                .usage(BufferUsage::Index)
                .build(registry)?;

            let bind_group_layout = device.create_bind_group_layout(&triad_gpu::wgpu::BindGroupLayoutDescriptor {
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
                            min_binding_size: Some(std::num::NonZeroU64::new(std::mem::size_of::<CameraUniforms>() as u64).unwrap()),
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
                        resource: registry.get(gaussian_buffer).unwrap().as_entire_binding(),
                    },
                    triad_gpu::wgpu::BindGroupEntry {
                        binding: 1,
                        resource: registry.get(camera_buffer).unwrap().as_entire_binding(),
                    },
                ],
            });

            let vertex_shader = device.create_shader_module(triad_gpu::wgpu::ShaderModuleDescriptor {
                label: Some("gaussian_vs"),
                source: triad_gpu::wgpu::ShaderSource::Wgsl(triad_gpu::shaders::GAUSSIAN_VERTEX.into()),
            });
            let fragment_shader = device.create_shader_module(triad_gpu::wgpu::ShaderModuleDescriptor {
                label: Some("gaussian_fs"),
                source: triad_gpu::wgpu::ShaderSource::Wgsl(triad_gpu::shaders::GAUSSIAN_FRAGMENT.into()),
            });

            let pipeline_layout = device.create_pipeline_layout(&triad_gpu::wgpu::PipelineLayoutDescriptor {
                label: Some("Gaussian Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

            let pipeline = RenderPipelineBuilder::new(device)
                .with_label("Gaussian Pipeline")
                .with_vertex_shader(registry.insert(vertex_shader))
                .with_fragment_shader(registry.insert(fragment_shader))
                .with_layout(pipeline_layout)
                .with_primitive(triad_gpu::wgpu::PrimitiveState {
                    topology: triad_gpu::wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                })
                .with_fragment_target(Some(triad_gpu::wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(triad_gpu::wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: triad_gpu::wgpu::ColorWrites::ALL,
                }))
                .with_depth_stencil(triad_gpu::wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: false, // Gaussians use alpha blending, don't write depth
                    depth_compare: triad_gpu::wgpu::CompareFunction::Less,
                    stencil: triad_gpu::wgpu::StencilState::default(),
                    bias: triad_gpu::wgpu::DepthBiasState::default(),
                })
                .build(registry)?;

            ModeResources {
                bind_group: registry.insert(bind_group),
                pipeline,
                index_buffer: Some(index_buffer),
                vertex_count: 0,
                index_count: gaussians.len() as u32 * 3,
                uses_indices: true,
            }
        };

        // === Create Triangle Resources ===
        let triangle_resources = {
            let triangles: Vec<TrianglePrimitive> = if ply_loader::ply_has_faces(ply_path_str).unwrap_or(false) {
                info!("Using face data from PLY");
                ply_loader::load_triangles_from_ply(ply_path_str)?
            } else {
                info!("Triangulating point cloud");
                let positions: Vec<Vec3> = vertices.iter().map(|v| v.position).collect();
                let triangle_indices = triangulation::triangulate_points(&positions);
                
                triangle_indices.iter().map(|[i0, i1, i2]| {
                    let v0 = &vertices[*i0];
                    let v1 = &vertices[*i1];
                    let v2 = &vertices[*i2];
                    let avg_color = (v0.color + v1.color + v2.color) / 3.0;
                    let avg_opacity = (v0.opacity + v1.opacity + v2.opacity) / 3.0;
                    TrianglePrimitive::new(v0.position, v1.position, v2.position, avg_color, avg_opacity)
                }).collect()
            };

            info!("Created {} triangles", triangles.len());

            let triangle_buffer = renderer
                .create_buffer()
                .label("Triangle Buffer")
                .with_pod_data(&triangles)
                .usage(BufferUsage::Storage { read_only: true })
                .build(registry)?;

            let mut indices = Vec::with_capacity(triangles.len() * 3);
            for i in 0..triangles.len() as u32 {
                let base = i * 3;
                indices.push(base);
                indices.push(base + 1);
                indices.push(base + 2);
            }
            let index_buffer = renderer
                .create_buffer()
                .label("Triangle Index Buffer")
                .with_pod_data(&indices)
                .usage(BufferUsage::Index)
                .build(registry)?;

            let bind_group_layout = device.create_bind_group_layout(&triad_gpu::wgpu::BindGroupLayoutDescriptor {
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
                            min_binding_size: Some(std::num::NonZeroU64::new(std::mem::size_of::<CameraUniforms>() as u64).unwrap()),
                        },
                        count: None,
                    },
                ],
            });

            let bind_group = device.create_bind_group(&triad_gpu::wgpu::BindGroupDescriptor {
                label: Some("Triangle Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    triad_gpu::wgpu::BindGroupEntry {
                        binding: 0,
                        resource: registry.get(triangle_buffer).unwrap().as_entire_binding(),
                    },
                    triad_gpu::wgpu::BindGroupEntry {
                        binding: 1,
                        resource: registry.get(camera_buffer).unwrap().as_entire_binding(),
                    },
                ],
            });

            let vertex_shader = device.create_shader_module(triad_gpu::wgpu::ShaderModuleDescriptor {
                label: Some("triangle_vs"),
                source: triad_gpu::wgpu::ShaderSource::Wgsl(triad_gpu::shaders::TRIANGLE_VERTEX.into()),
            });
            let fragment_shader = device.create_shader_module(triad_gpu::wgpu::ShaderModuleDescriptor {
                label: Some("triangle_fs"),
                source: triad_gpu::wgpu::ShaderSource::Wgsl(triad_gpu::shaders::TRIANGLE_FRAGMENT.into()),
            });

            let pipeline_layout = device.create_pipeline_layout(&triad_gpu::wgpu::PipelineLayoutDescriptor {
                label: Some("Triangle Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

            let pipeline = RenderPipelineBuilder::new(device)
                .with_label("Triangle Pipeline")
                .with_vertex_shader(registry.insert(vertex_shader))
                .with_fragment_shader(registry.insert(fragment_shader))
                .with_layout(pipeline_layout)
                .with_primitive(triad_gpu::wgpu::PrimitiveState {
                    topology: triad_gpu::wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                })
                .with_fragment_target(Some(triad_gpu::wgpu::ColorTargetState {
                    format: surface_format,
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

            ModeResources {
                bind_group: registry.insert(bind_group),
                pipeline,
                index_buffer: Some(index_buffer),
                vertex_count: 0,
                index_count: triangles.len() as u32 * 3,
                uses_indices: true,
            }
        };

        info!("All visualization modes ready");

        Ok(Self {
            bounds,
            current_mode: init_data.initial_mode,
            mode_signal: init_data.mode_signal,
            camera_buffer,
            point_resources,
            gaussian_resources,
            triangle_resources,
        })
    }

    fn bounds(&self) -> &SceneBounds {
        &self.bounds
    }

    fn depth_format(&self) -> Option<triad_gpu::wgpu::TextureFormat> {
        Some(DEPTH_FORMAT)
    }

    fn update(&mut self, queue: &triad_gpu::wgpu::Queue, registry: &ResourceRegistry, camera: &CameraUniforms) {
        // Check for mode changes from the signal
        let new_mode = crate::app::read_mode(&self.mode_signal);
        if new_mode != self.current_mode {
            info!("Mode changed: {} -> {}", self.current_mode, new_mode);
            self.current_mode = new_mode;
        }

        let camera_buffer = registry.get(self.camera_buffer).expect("camera buffer");
        queue.write_buffer(camera_buffer, 0, bytemuck::bytes_of(camera));
    }

    fn render(
        &self,
        encoder: &mut triad_gpu::wgpu::CommandEncoder,
        ctx: RenderContext,
        registry: &ResourceRegistry,
    ) {
        let resources = self.current_resources();
        let pipeline = registry.get(resources.pipeline).expect("pipeline");
        let bind_group = registry.get(resources.bind_group).expect("bind group");

        let depth_stencil_attachment = ctx.depth_view.map(|dv| triad_gpu::wgpu::RenderPassDepthStencilAttachment {
            view: dv,
            depth_ops: Some(triad_gpu::wgpu::Operations {
                load: triad_gpu::wgpu::LoadOp::Clear(1.0),
                store: triad_gpu::wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        });

        let mut pass = encoder.begin_render_pass(&triad_gpu::wgpu::RenderPassDescriptor {
            label: Some("Multi-Delegate Pass"),
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

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);

        if resources.uses_indices {
            let index_buffer = registry.get(resources.index_buffer.unwrap()).expect("index buffer");
            pass.set_index_buffer(index_buffer.slice(..), triad_gpu::wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..resources.index_count, 0, 0..1);
        } else {
            pass.draw(0..resources.vertex_count, 0..1);
        }
    }
}

