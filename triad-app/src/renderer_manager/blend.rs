//! Blend pass resources for compositing layers.

use crate::renderer_manager::errors::RendererManagerError;
use std::sync::Arc;
use triad_gpu::{
    wgpu, Handle, Renderer, ResourceRegistry, RenderPipelineBuilder,
};

/// Blend pass resources.
pub struct BlendResources {
    pub pipeline: Handle<wgpu::RenderPipeline>,
    pub bind_group: Handle<wgpu::BindGroup>,
    pub bind_group_layout: Handle<wgpu::BindGroupLayout>,
    pub opacity_buffer: Handle<wgpu::Buffer>,
    pub sampler: Handle<wgpu::Sampler>,
}

/// Layer opacity uniforms.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LayerUniforms {
    pub opacity: [f32; 3],
    pub _padding: f32,
}

/// Create blend pass resources.
pub fn create_blend_resources(
    renderer: &Renderer,
    registry: &mut ResourceRegistry,
    surface_format: wgpu::TextureFormat,
    point_texture_view: &Arc<wgpu::TextureView>,
    gaussian_texture_view: &Arc<wgpu::TextureView>,
    triangle_texture_view: &Arc<wgpu::TextureView>,
) -> Result<BlendResources, RendererManagerError> {
    // Create opacity uniform buffer using builder
    let opacity_buffer_handle = renderer
        .create_buffer()
        .label("Layer Opacity Buffer")
        .with_pod_data(&[LayerUniforms {
            opacity: [1.0, 1.0, 1.0],
            _padding: 0.0,
        }])
        .usage(triad_gpu::BufferUsage::Uniform)
        .build(registry)
        .map_err(|e| RendererManagerError::BufferBuildError(e.to_string()))?;

    let device = renderer.device();

    // Create sampler (no builder API available, keep manual)
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Layer Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let sampler_handle = registry.insert(sampler);

    // Create bind group layout and bind group manually (Arc<TextureView> can't use builder)
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Blend Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let bind_group_layout_handle = registry.insert(bind_group_layout);

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Blend Bind Group"),
        layout: registry.get::<wgpu::BindGroupLayout>(bind_group_layout_handle).unwrap(),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(point_texture_view.as_ref()),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(gaussian_texture_view.as_ref()),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(triangle_texture_view.as_ref()),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(registry.get::<wgpu::Sampler>(sampler_handle).unwrap()),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Buffer(
                    registry.get::<wgpu::Buffer>(opacity_buffer_handle).unwrap().as_entire_buffer_binding(),
                ),
            },
        ],
    });
    let bind_group_handle = registry.insert(bind_group);

    // Create blend shader
    let vertex_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("blend_vs"),
        source: wgpu::ShaderSource::Wgsl(triad_gpu::shaders::LAYER_BLEND.into()),
    });
    let fragment_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("blend_fs"),
        source: wgpu::ShaderSource::Wgsl(triad_gpu::shaders::LAYER_BLEND.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Blend Pipeline Layout"),
        bind_group_layouts: &[registry.get::<wgpu::BindGroupLayout>(bind_group_layout_handle).unwrap()],
        push_constant_ranges: &[],
    });

    let pipeline = RenderPipelineBuilder::new(device)
        .with_label("Blend Pipeline")
        .with_vertex_shader(registry.insert(vertex_shader))
        .with_fragment_shader(registry.insert(fragment_shader))
        .with_layout(pipeline_layout)
        .with_primitive(wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        })
        .with_fragment_target(Some(wgpu::ColorTargetState {
            format: surface_format,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING), // Alpha blend so background shows through when transparent
            write_mask: wgpu::ColorWrites::ALL,
        }))
        .build(registry)?;

    Ok(BlendResources {
        pipeline,
        bind_group: bind_group_handle,
        bind_group_layout: bind_group_layout_handle,
        opacity_buffer: opacity_buffer_handle,
        sampler: sampler_handle,
    })
}

/// Recreate blend bind group with new texture views.
pub fn recreate_blend_bind_group(
    device: &wgpu::Device,
    registry: &mut ResourceRegistry,
    bind_group_layout: Handle<wgpu::BindGroupLayout>,
    point_texture_view: &Arc<wgpu::TextureView>,
    gaussian_texture_view: &Arc<wgpu::TextureView>,
    triangle_texture_view: &Arc<wgpu::TextureView>,
    sampler: Handle<wgpu::Sampler>,
    opacity_buffer: Handle<wgpu::Buffer>,
) -> Result<Handle<wgpu::BindGroup>, RendererManagerError> {
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Blend Bind Group"),
        layout: registry.get::<wgpu::BindGroupLayout>(bind_group_layout).unwrap(),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(point_texture_view.as_ref()),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(gaussian_texture_view.as_ref()),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(triangle_texture_view.as_ref()),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(registry.get::<wgpu::Sampler>(sampler).unwrap()),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Buffer(
                    registry.get::<wgpu::Buffer>(opacity_buffer).unwrap().as_entire_buffer_binding(),
                ),
            },
        ],
    });

    Ok(registry.insert(bind_group))
}
