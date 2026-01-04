//! Frame graph passes for rendering layers.

use std::sync::Arc;
use triad_gpu::{Handle, Pass, PassContext, wgpu};

/// Generic render pass that renders to a texture view.
pub struct GenericRenderPass {
    render_pipeline: Handle<wgpu::RenderPipeline>,
    render_bind_group: Handle<wgpu::BindGroup>,
    index_buffer: Option<Handle<wgpu::Buffer>>,
    index_count: u32,
    vertex_count: u32,
    uses_indices: bool,
    color_view: Arc<wgpu::TextureView>,
    depth_view: Option<Arc<wgpu::TextureView>>,
}

impl GenericRenderPass {
    pub fn new(
        render_pipeline: Handle<wgpu::RenderPipeline>,
        render_bind_group: Handle<wgpu::BindGroup>,
        index_buffer: Option<Handle<wgpu::Buffer>>,
        index_count: u32,
        vertex_count: u32,
        color_view: Arc<wgpu::TextureView>,
        depth_view: Option<Arc<wgpu::TextureView>>,
    ) -> Self {
        Self {
            render_pipeline,
            render_bind_group,
            index_buffer,
            index_count,
            vertex_count,
            uses_indices: index_buffer.is_some(),
            color_view,
            depth_view,
        }
    }
}

impl Pass for GenericRenderPass {
    fn name(&self) -> &str {
        "GenericRender"
    }

    fn execute(&self, ctx: &PassContext) -> wgpu::CommandBuffer {
        let mut encoder = ctx.create_command_encoder(Some("Generic Render Encoder"));

        let pipeline = ctx
            .get_render_pipeline(self.render_pipeline)
            .expect("render pipeline");
        let bind_group = ctx
            .get_bind_group(self.render_bind_group)
            .expect("render bind group");

        let depth_stencil_attachment =
            self.depth_view
                .as_ref()
                .map(|depth_view| wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Generic Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
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

        if self.uses_indices {
            let index_buffer = ctx
                .get_buffer(self.index_buffer.unwrap())
                .expect("index buffer");
            render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.index_count, 0, 0..1);
        } else {
            render_pass.draw(0..self.vertex_count, 0..1);
        }

        drop(render_pass);
        encoder.finish()
    }
}

/// Compute pass for sorting Gaussians by depth.
pub struct GaussianSortPass {
    compute_pipeline: Handle<wgpu::ComputePipeline>,
    compute_bind_group: Handle<wgpu::BindGroup>,
    gaussian_count: u32,
}

impl GaussianSortPass {
    pub fn new(
        compute_pipeline: Handle<wgpu::ComputePipeline>,
        compute_bind_group: Handle<wgpu::BindGroup>,
        gaussian_count: u32,
    ) -> Self {
        Self {
            compute_pipeline,
            compute_bind_group,
            gaussian_count,
        }
    }
}

impl Pass for GaussianSortPass {
    fn name(&self) -> &str {
        "GaussianSort"
    }

    fn execute(&self, ctx: &PassContext) -> wgpu::CommandBuffer {
        let mut encoder = ctx.create_command_encoder(Some("Gaussian Sort Encoder"));

        let pipeline = ctx
            .get_compute_pipeline(self.compute_pipeline)
            .expect("compute pipeline");
        let bind_group = ctx
            .get_bind_group(self.compute_bind_group)
            .expect("compute bind group");

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Gaussian Sort Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, bind_group, &[]);

        // Dispatch one workgroup per 64 gaussians (workgroup_size is 64)
        let workgroup_count = (self.gaussian_count + 63) / 64;
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);

        drop(compute_pass);
        encoder.finish()
    }
}

/// Layer blend pass that composites multiple layer textures with opacity.
pub struct LayerBlendPass {
    blend_pipeline: Handle<wgpu::RenderPipeline>,
    blend_bind_group: Handle<wgpu::BindGroup>,
    blend_opacity_buffer: Handle<wgpu::Buffer>,
    final_view: Arc<wgpu::TextureView>,
}

impl LayerBlendPass {
    pub fn new(
        blend_pipeline: Handle<wgpu::RenderPipeline>,
        blend_bind_group: Handle<wgpu::BindGroup>,
        blend_opacity_buffer: Handle<wgpu::Buffer>,
        final_view: Arc<wgpu::TextureView>,
    ) -> Self {
        Self {
            blend_pipeline,
            blend_bind_group,
            blend_opacity_buffer,
            final_view,
        }
    }
}

impl Pass for LayerBlendPass {
    fn name(&self) -> &str {
        "LayerBlend"
    }

    fn execute(&self, ctx: &PassContext) -> wgpu::CommandBuffer {
        let mut encoder = ctx.create_command_encoder(Some("Layer Blend Encoder"));

        let pipeline = ctx
            .get_render_pipeline(self.blend_pipeline)
            .expect("blend pipeline");
        let bind_group = ctx
            .get_bind_group(self.blend_bind_group)
            .expect("blend bind group");

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Layer Blend Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.final_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.01,
                        g: 0.01,
                        b: 0.01,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        // Fullscreen quad - 3 vertices (triangle)
        render_pass.draw(0..3, 0..1);

        drop(render_pass);
        encoder.finish()
    }
}
