//! Frame graph construction helpers.

use crate::layers::LayerMode;
use crate::renderer_manager::layer_resources::LayerResources;
use crate::renderer_manager::passes::{GaussianSortPass, GenericRenderPass, LayerBlendPass};
use std::sync::Arc;
use triad_gpu::{
    wgpu, ExecutableFrameGraph, FrameGraph, FrameGraphError, Handle, PassBuilder,
};

/// Add a Gaussian sort compute pass to the frame graph.
pub fn add_gaussian_sort_pass(
    frame_graph: &mut FrameGraph,
    gaussian_buffer: Handle<wgpu::Buffer>,
    sort_buffer: Handle<wgpu::Buffer>,
    sort_pipeline: Handle<wgpu::ComputePipeline>,
    sort_bind_group: Handle<wgpu::BindGroup>,
    camera_buffer: Handle<wgpu::Buffer>,
    gaussian_count: u32,
) {
    frame_graph
        .register_resource(gaussian_buffer)
        .register_resource(sort_buffer)
        .register_resource(sort_pipeline)
        .register_resource(sort_bind_group);

    let mut sort_pass_builder = PassBuilder::new("GaussianSort");
    sort_pass_builder.read(gaussian_buffer);
    sort_pass_builder.read(camera_buffer);
    sort_pass_builder.write(sort_buffer);
    let sort_pass = sort_pass_builder.with_pass(Box::new(GaussianSortPass::new(
        sort_pipeline,
        sort_bind_group,
        gaussian_count,
    )));
    frame_graph.add_pass(sort_pass);
}

/// Add a layer render pass to the frame graph.
pub fn add_layer_pass(
    frame_graph: &mut FrameGraph,
    layer_mode: LayerMode,
    resources: &LayerResources,
    camera_buffer: Handle<wgpu::Buffer>,
    depth_view: Option<Arc<wgpu::TextureView>>,
) {
    // Register layer resources
    frame_graph
        .register_resource(resources.pipeline)
        .register_resource(resources.bind_group)
        .register_resource(resources.data_buffer);

    if let Some(idx_buf) = resources.index_buffer {
        frame_graph.register_resource(idx_buf);
    }

    // Add layer render pass (renders to intermediate texture)
    let pass_name = format!("Layer{:?}", layer_mode);
    let mut pass_builder = PassBuilder::new(pass_name);
    pass_builder.read(resources.pipeline);
    pass_builder.read(resources.bind_group);
    pass_builder.read(resources.data_buffer);
    pass_builder.read(camera_buffer);

    if let Some(idx_buf) = resources.index_buffer {
        pass_builder.read(idx_buf);
    }

    frame_graph.add_pass(
        pass_builder.with_pass(Box::new(GenericRenderPass::new(
            resources.pipeline,
            resources.bind_group,
            resources.index_buffer,
            resources.index_count,
            resources.vertex_count,
            resources.texture_view.clone(),
            depth_view,
        ))),
    );
}

/// Add blend pass to composite all layers.
pub fn add_blend_pass(
    frame_graph: &mut FrameGraph,
    blend_pipeline: Handle<wgpu::RenderPipeline>,
    blend_bind_group: Handle<wgpu::BindGroup>,
    blend_opacity_buffer: Handle<wgpu::Buffer>,
    point_texture: Option<Handle<wgpu::Texture>>,
    gaussian_texture: Option<Handle<wgpu::Texture>>,
    triangle_texture: Option<Handle<wgpu::Texture>>,
    final_view: Arc<wgpu::TextureView>,
) {
    // Register blend resources
    frame_graph
        .register_resource(blend_pipeline)
        .register_resource(blend_bind_group)
        .register_resource(blend_opacity_buffer);

    // Register layer textures if they exist
    if let Some(texture) = point_texture {
        frame_graph.register_resource(texture);
    }
    if let Some(texture) = gaussian_texture {
        frame_graph.register_resource(texture);
    }
    if let Some(texture) = triangle_texture {
        frame_graph.register_resource(texture);
    }

    // Add blend pass
    let mut blend_pass_builder = PassBuilder::new("BlendLayers");
    blend_pass_builder.read(blend_pipeline);
    blend_pass_builder.read(blend_bind_group);
    blend_pass_builder.read(blend_opacity_buffer);
    let blend_pass = blend_pass_builder.with_pass(Box::new(LayerBlendPass::new(
        blend_pipeline,
        blend_bind_group,
        blend_opacity_buffer,
        final_view,
    )));
    frame_graph.add_pass(blend_pass);
}

/// Build a complete frame graph with all enabled layers.
pub fn build_frame_graph(
    camera_buffer: Handle<wgpu::Buffer>,
    enabled_layers: &[bool; 3],
    layer_opacity: &[f32; 3],
    point_resources: &LayerResources,
    gaussian_resources: &LayerResources,
    triangle_resources: &LayerResources,
    gaussian_sort_pipeline: Option<Handle<wgpu::ComputePipeline>>,
    gaussian_sort_bind_group: Option<Handle<wgpu::BindGroup>>,
    sort_buffer: Option<Handle<wgpu::Buffer>>,
    blend_pipeline: Handle<wgpu::RenderPipeline>,
    blend_bind_group: Handle<wgpu::BindGroup>,
    blend_opacity_buffer: Handle<wgpu::Buffer>,
    final_view: Arc<wgpu::TextureView>,
    depth_view: Option<Arc<wgpu::TextureView>>,
) -> Result<ExecutableFrameGraph, FrameGraphError> {
    let mut frame_graph = FrameGraph::default();

    // Register shared camera buffer
    frame_graph.register_resource(camera_buffer);

    // For each enabled layer, add render pass
    for (layer_idx, layer_mode) in [
        LayerMode::Points,
        LayerMode::Gaussians,
        LayerMode::Triangles,
    ]
    .iter()
    .enumerate()
    {
        if !enabled_layers[layer_idx] {
            continue;
        }

        let resources = match layer_mode {
            LayerMode::Points => point_resources,
            LayerMode::Gaussians => gaussian_resources,
            LayerMode::Triangles => triangle_resources,
        };

        // For Gaussians, add compute pass first
        if *layer_mode == LayerMode::Gaussians {
            if let (Some(sort_pipeline), Some(sort_bind_group), Some(sort_buf)) = (
                gaussian_sort_pipeline,
                gaussian_sort_bind_group,
                sort_buffer,
            ) {
                add_gaussian_sort_pass(
                    &mut frame_graph,
                    resources.data_buffer,
                    sort_buf,
                    sort_pipeline,
                    sort_bind_group,
                    camera_buffer,
                    resources.index_count / 3, // gaussian count
                );
            }
        }

        add_layer_pass(&mut frame_graph, *layer_mode, resources, camera_buffer, depth_view.clone());
    }

    // Add blend pass to composite all layers
    let has_enabled_layers = enabled_layers.iter().any(|&enabled| enabled);
    if has_enabled_layers {
        add_blend_pass(
            &mut frame_graph,
            blend_pipeline,
            blend_bind_group,
            blend_opacity_buffer,
            if enabled_layers[0] {
                Some(point_resources.texture)
            } else {
                None
            },
            if enabled_layers[1] {
                Some(gaussian_resources.texture)
            } else {
                None
            },
            if enabled_layers[2] {
                Some(triangle_resources.texture)
            } else {
                None
            },
            final_view,
        );
    }

    frame_graph.build()
}
