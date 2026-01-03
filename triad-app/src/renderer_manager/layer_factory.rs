//! Factory for creating layer resources with common patterns.

use crate::renderer_manager::constants::DEPTH_FORMAT;
use crate::renderer_manager::errors::RendererManagerError;
use crate::renderer_manager::layer_resources::LayerResources;
use glam::Vec3;
use std::sync::Arc;
use triad_data::PlyVertex;
use triad_gpu::{
    wgpu, BufferUsage, Handle, PointPrimitive, Renderer,
    ResourceRegistry, RenderPipelineBuilder, TrianglePrimitive,
};

/// Configuration for creating a render layer.
pub struct LayerConfig<'a> {
    pub label: &'a str,
    pub vertex_shader: &'a str,
    pub fragment_shader: &'a str,
    pub depth_write_enabled: bool,
    pub blend_state: Option<wgpu::BlendState>,
}

/// Create a bind group for a layer (data buffer + camera uniform).
pub fn create_layer_bind_group(
    renderer: &Renderer,
    registry: &mut ResourceRegistry,
    label: &str,
    data_buffer: Handle<wgpu::Buffer>,
    camera_buffer: Handle<wgpu::Buffer>,
) -> Result<(Handle<wgpu::BindGroupLayout>, Handle<wgpu::BindGroup>), RendererManagerError> {
    let device = renderer.device();

    // Create bind group layout manually (builder has borrow issues)
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("{} Bind Group Layout", label)),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let layout_handle = registry.insert(bind_group_layout);

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("{} Bind Group", label)),
        layout: registry.get::<wgpu::BindGroupLayout>(layout_handle).unwrap(),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(
                    registry.get::<wgpu::Buffer>(data_buffer).unwrap().as_entire_buffer_binding(),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(
                    registry.get::<wgpu::Buffer>(camera_buffer).unwrap().as_entire_buffer_binding(),
                ),
            },
        ],
    });
    let bind_group_handle = registry.insert(bind_group);

    Ok((layout_handle, bind_group_handle))
}

/// Create a render pipeline for a layer.
pub fn create_layer_pipeline(
    device: &wgpu::Device,
    registry: &mut ResourceRegistry,
    config: &LayerConfig<'_>,
    surface_format: wgpu::TextureFormat,
    bind_group_layout: Handle<wgpu::BindGroupLayout>,
) -> Result<Handle<wgpu::RenderPipeline>, RendererManagerError> {
    let vertex_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&format!("{}_vs", config.label)),
        source: wgpu::ShaderSource::Wgsl(config.vertex_shader.into()),
    });

    let fragment_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&format!("{}_fs", config.label)),
        source: wgpu::ShaderSource::Wgsl(config.fragment_shader.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{} Pipeline Layout", config.label)),
        bind_group_layouts: &[registry.get::<wgpu::BindGroupLayout>(bind_group_layout).unwrap()],
        push_constant_ranges: &[],
    });

    let pipeline = RenderPipelineBuilder::new(device)
        .with_label(&format!("{} Pipeline", config.label))
        .with_vertex_shader(registry.insert(vertex_shader))
        .with_fragment_shader(registry.insert(fragment_shader))
        .with_layout(pipeline_layout)
        .with_primitive(wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        })
        .with_fragment_target(Some(wgpu::ColorTargetState {
            format: surface_format,
            blend: config.blend_state,
            write_mask: wgpu::ColorWrites::ALL,
        }))
        .with_depth_stencil(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: config.depth_write_enabled,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        })
        .build(registry)?;

    Ok(pipeline)
}

/// Create point layer resources.
pub fn create_point_resources(
    renderer: &Renderer,
    registry: &mut ResourceRegistry,
    vertices: &[PlyVertex],
    point_size: f32,
    surface_format: wgpu::TextureFormat,
    camera_buffer: Handle<wgpu::Buffer>,
    texture: Handle<wgpu::Texture>,
    texture_view: Arc<wgpu::TextureView>,
) -> Result<LayerResources, RendererManagerError> {
    let mut points: Vec<PointPrimitive> = vertices
        .iter()
        .map(|v| PointPrimitive::new(v.position, point_size, v.color, v.opacity))
        .collect();

    if points.is_empty() {
        points.push(PointPrimitive::new(Vec3::ZERO, point_size, Vec3::ZERO, 0.0));
    }

    let point_buffer = renderer
        .create_buffer()
        .label("Point Buffer")
        .with_pod_data(&points)
        .usage(BufferUsage::Storage { read_only: true })
        .build(registry)
        .map_err(|e| RendererManagerError::BufferBuildError(e.to_string()))?;

    let (bind_group_layout, bind_group) = create_layer_bind_group(
        renderer,
        registry,
        "Point",
        point_buffer,
        camera_buffer,
    )?;

    let config = LayerConfig {
        label: "Point",
        vertex_shader: triad_gpu::shaders::POINT_VERTEX,
        fragment_shader: triad_gpu::shaders::POINT_FRAGMENT,
        depth_write_enabled: true,
        blend_state: Some(wgpu::BlendState::ALPHA_BLENDING),
    };

    let pipeline = create_layer_pipeline(
        renderer.device(),
        registry,
        &config,
        surface_format,
        bind_group_layout,
    )?;

    let vertex_count = if points.len() == 1 && vertices.is_empty() {
        0
    } else {
        points.len() as u32 * 3
    };

    Ok(LayerResources::new(
        pipeline,
        bind_group,
        bind_group_layout,
        point_buffer,
        None,
        vertex_count,
        0,
        false,
        texture,
        texture_view,
    ))
}

/// Create triangle layer resources.
pub fn create_triangle_resources(
    renderer: &Renderer,
    registry: &mut ResourceRegistry,
    vertices: &[PlyVertex],
    ply_path: &Option<std::path::PathBuf>,
    surface_format: wgpu::TextureFormat,
    camera_buffer: Handle<wgpu::Buffer>,
    texture: Handle<wgpu::Texture>,
    texture_view: Arc<wgpu::TextureView>,
) -> Result<LayerResources, RendererManagerError> {
    use triad_data::triangulation;
    use triad_gpu::ply_loader;

    let mut triangles: Vec<TrianglePrimitive> = if vertices.is_empty() {
        Vec::new()
    } else if let Some(ply_path) = ply_path {
        let ply_path_str = ply_path
            .to_str()
            .ok_or_else(|| RendererManagerError::ResourceError(format!("Invalid PLY path: {:?}", ply_path)))?;
        if ply_loader::ply_has_faces(ply_path_str).unwrap_or(false) {
            tracing::info!("Using face data from PLY");
            ply_loader::load_triangles_from_ply(ply_path_str)?
        } else {
            tracing::info!("Triangulating point cloud");
            let positions: Vec<Vec3> = vertices.iter().map(|v| v.position).collect();
            let triangle_indices = triangulation::triangulate_points(&positions);

            triangle_indices
                .iter()
                .map(|[i0, i1, i2]| {
                    let v0 = &vertices[*i0];
                    let v1 = &vertices[*i1];
                    let v2 = &vertices[*i2];
                    let avg_color = (v0.color + v1.color + v2.color) / 3.0;
                    let avg_opacity = (v0.opacity + v1.opacity + v2.opacity) / 3.0;
                    TrianglePrimitive::new(
                        v0.position,
                        v1.position,
                        v2.position,
                        avg_color,
                        avg_opacity,
                    )
                })
                .collect()
        }
    } else {
        Vec::new()
    };

    if triangles.is_empty() {
        triangles.push(TrianglePrimitive::new(
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::ZERO,
            0.0,
        ));
    }

    let triangle_buffer = renderer
        .create_buffer()
        .label("Triangle Buffer")
        .with_pod_data(&triangles)
        .usage(BufferUsage::Storage { read_only: true })
        .build(registry)
        .map_err(|e| RendererManagerError::BufferBuildError(e.to_string()))?;

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
        .build(registry)
        .map_err(|e| RendererManagerError::BufferBuildError(e.to_string()))?;

    let (bind_group_layout, bind_group) = create_layer_bind_group(
        renderer,
        registry,
        "Triangle",
        triangle_buffer,
        camera_buffer,
    )?;

    let config = LayerConfig {
        label: "Triangle",
        vertex_shader: triad_gpu::shaders::TRIANGLE_VERTEX,
        fragment_shader: triad_gpu::shaders::TRIANGLE_FRAGMENT,
        depth_write_enabled: true,
        blend_state: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
    };

    let pipeline = create_layer_pipeline(
        renderer.device(),
        registry,
        &config,
        surface_format,
        bind_group_layout,
    )?;

    Ok(LayerResources::new(
        pipeline,
        bind_group,
        bind_group_layout,
        triangle_buffer,
        Some(index_buffer),
        0,
        triangles.len() as u32 * 3,
        true,
        texture,
        texture_view,
    ))
}

/// Gaussian compute resources.
pub struct GaussianComputeResources {
    pub sort_pipeline: Handle<wgpu::ComputePipeline>,
    pub sort_bind_group: Handle<wgpu::BindGroup>,
    pub sort_bind_group_layout: Handle<wgpu::BindGroupLayout>,
    pub sort_buffer: Handle<wgpu::Buffer>,
}

/// Create Gaussian layer resources (including compute pipeline).
pub fn create_gaussian_resources(
    renderer: &Renderer,
    registry: &mut ResourceRegistry,
    ply_path: &Option<std::path::PathBuf>,
    surface_format: wgpu::TextureFormat,
    camera_buffer: Handle<wgpu::Buffer>,
    texture: Handle<wgpu::Texture>,
    texture_view: Arc<wgpu::TextureView>,
) -> Result<(LayerResources, GaussianComputeResources), RendererManagerError> {
    use std::path::PathBuf;
    use triad_gpu::{GaussianPoint, ply_loader};

    let mut gaussians = if let Some(ply_path) = ply_path {
        let ply_path_str = ply_path
            .to_str()
            .ok_or_else(|| RendererManagerError::ResourceError(format!("Invalid PLY path: {:?}", ply_path)))?;
        ply_loader::load_gaussians_from_ply(ply_path_str)?
    } else {
        Vec::new()
    };

    if gaussians.is_empty() {
        gaussians.push(GaussianPoint::new(
            Vec3::ZERO,
            Vec3::ZERO,
            0.0,
            [0.0, 0.0, 0.0, 1.0],
            Vec3::ONE,
        ));
    }

    let gaussian_buffer = renderer
        .create_buffer()
        .label("Gaussian Buffer")
        .with_pod_data(&gaussians)
        .usage(BufferUsage::Storage { read_only: true })
        .build(registry)
        .map_err(|e| RendererManagerError::BufferBuildError(e.to_string()))?;

    // Create sort buffer using builder
    let sort_data_size = gaussians.len() * std::mem::size_of::<(f32, u32)>();
    let sort_buffer_handle = renderer
        .create_buffer()
        .label("Sort Buffer")
        .size(sort_data_size as u64)
        .usage(BufferUsage::Storage { read_only: false })
        .build(registry)
        .map_err(|e| RendererManagerError::BufferBuildError(e.to_string()))?;

    // Create compute bind group layout and bind group manually (builder has borrow issues)
    let device = renderer.device();
    let compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Gaussian Sort Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let compute_layout = registry.insert(compute_bind_group_layout);

    let compute_bind_group_inner = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Gaussian Sort Bind Group"),
        layout: registry.get::<wgpu::BindGroupLayout>(compute_layout).unwrap(),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(
                    registry.get::<wgpu::Buffer>(gaussian_buffer).unwrap().as_entire_buffer_binding(),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(
                    registry.get::<wgpu::Buffer>(sort_buffer_handle).unwrap().as_entire_buffer_binding(),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(
                    registry.get::<wgpu::Buffer>(camera_buffer).unwrap().as_entire_buffer_binding(),
                ),
            },
        ],
    });
    let compute_bind_group = registry.insert(compute_bind_group_inner);

    // Create compute shader and pipeline
    let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gaussian_sort_cs"),
        source: wgpu::ShaderSource::Wgsl(triad_gpu::shaders::GAUSSIAN_SORT.into()),
    });

    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Gaussian Sort Pipeline Layout"),
        bind_group_layouts: &[registry.get::<wgpu::BindGroupLayout>(compute_layout).unwrap()],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Gaussian Sort Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &compute_shader,
        entry_point: Some("cs_main"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });

    // Create index buffer
    let mut indices = Vec::with_capacity(gaussians.len() * 3);
    for i in 0..gaussians.len() as u32 {
        let base = i * 3;
        indices.push(base);
        indices.push(base + 1);
        indices.push(base + 2);
    }
    if indices.is_empty() {
        indices.push(0);
        indices.push(1);
        indices.push(2);
    }
    let index_buffer = renderer
        .create_buffer()
        .label("Gaussian Index Buffer")
        .with_pod_data(&indices)
        .usage(BufferUsage::Index)
        .build(registry)
        .map_err(|e| RendererManagerError::BufferBuildError(e.to_string()))?;

    // Create render pipeline
    let (bind_group_layout, bind_group) = create_layer_bind_group(
        renderer,
        registry,
        "Gaussian",
        gaussian_buffer,
        camera_buffer,
    )?;

    let config = LayerConfig {
        label: "Gaussian",
        vertex_shader: triad_gpu::shaders::GAUSSIAN_VERTEX,
        fragment_shader: triad_gpu::shaders::GAUSSIAN_FRAGMENT,
        depth_write_enabled: false,
        blend_state: Some(wgpu::BlendState::ALPHA_BLENDING),
    };

    let pipeline = create_layer_pipeline(
        renderer.device(),
        registry,
        &config,
        surface_format,
        bind_group_layout,
    )?;

    let index_count = if gaussians.len() == 1 && ply_path.is_none() {
        0
    } else {
        gaussians.len() as u32 * 3
    };

    let layer_resources = LayerResources::new(
        pipeline,
        bind_group,
        bind_group_layout,
        gaussian_buffer,
        Some(index_buffer),
        0,
        index_count,
        true,
        texture,
        texture_view,
    );

    let compute_resources = GaussianComputeResources {
        sort_pipeline: registry.insert(compute_pipeline),
        sort_bind_group: compute_bind_group,
        sort_bind_group_layout: compute_layout,
        sort_buffer: sort_buffer_handle,
    };

    Ok((layer_resources, compute_resources))
}

/// Update layer resources when data changes.
pub fn update_layer_bind_group(
    renderer: &Renderer,
    registry: &mut ResourceRegistry,
    label: &str,
    bind_group_layout: Handle<wgpu::BindGroupLayout>,
    data_buffer: Handle<wgpu::Buffer>,
    camera_buffer: Handle<wgpu::Buffer>,
) -> Result<Handle<wgpu::BindGroup>, RendererManagerError> {
    let (_, bind_group) = create_layer_bind_group(renderer, registry, label, data_buffer, camera_buffer)?;
    Ok(bind_group)
}
