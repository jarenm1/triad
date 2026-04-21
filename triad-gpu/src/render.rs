use crate::FrameTextureView;
use crate::error::RenderPassError;
use crate::frame_graph::pass::{Pass, PassBuilder, PassContext};
use crate::frame_graph::{Handle, ResourceType};
use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
pub enum ColorLoadOp {
    Load,
    Clear(wgpu::Color),
}

#[derive(Debug, Clone)]
pub enum RenderDraw {
    Direct {
        vertices: std::ops::Range<u32>,
        instances: std::ops::Range<u32>,
    },
    Indirect {
        buffer: Handle<wgpu::Buffer>,
        offset: u64,
    },
}

impl RenderDraw {
    pub fn direct(vertex_count: u32, instance_count: u32) -> Self {
        Self::Direct {
            vertices: 0..vertex_count,
            instances: 0..instance_count,
        }
    }

    pub fn direct_ranges(vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>) -> Self {
        Self::Direct {
            vertices,
            instances,
        }
    }

    pub fn indirect(buffer: Handle<wgpu::Buffer>, offset: u64) -> Self {
        Self::Indirect { buffer, offset }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ColorAttachmentView {
    Static(Handle<wgpu::TextureView>),
    FrameSlot(Handle<FrameTextureView>),
}

#[derive(Debug, Clone, Copy)]
struct ColorAttachment {
    view: ColorAttachmentView,
    load: ColorLoadOp,
}

#[derive(Debug, Clone, Copy)]
struct VertexBufferBinding {
    slot: u32,
    buffer: Handle<wgpu::Buffer>,
    offset: u64,
    size: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
struct BoundBindGroup {
    index: u32,
    handle: Handle<wgpu::BindGroup>,
}

#[derive(Debug)]
struct RenderDispatchPass {
    name: String,
    pipeline: Handle<wgpu::RenderPipeline>,
    color_attachments: Vec<ColorAttachment>,
    vertex_buffers: Vec<VertexBufferBinding>,
    bind_groups: Vec<BoundBindGroup>,
    draw: RenderDraw,
}

enum ResolvedColorView {
    Static(Handle<wgpu::TextureView>),
    Frame(Arc<wgpu::TextureView>),
}

impl Pass for RenderDispatchPass {
    fn name(&self) -> &str {
        &self.name
    }

    fn execute(&self, ctx: &PassContext) -> wgpu::CommandBuffer {
        let pipeline = ctx
            .get_render_pipeline(self.pipeline)
            .expect("render pipeline handle missing from registry");

        let mut encoder = ctx.create_command_encoder(Some(&self.name));
        let resolved_views: Vec<_> = self
            .color_attachments
            .iter()
            .map(|attachment| match attachment.view {
                ColorAttachmentView::Static(handle) => ResolvedColorView::Static(handle),
                ColorAttachmentView::FrameSlot(handle) => {
                    let slot = ctx
                        .resources
                        .get(handle)
                        .expect("frame texture view slot missing from registry");
                    ResolvedColorView::Frame(
                        slot.get()
                            .expect("frame texture view slot should be populated before execution"),
                    )
                }
            })
            .collect();
        let mut color_views = Vec::with_capacity(self.color_attachments.len());
        for (attachment, resolved_view) in self.color_attachments.iter().zip(&resolved_views) {
            let view: &wgpu::TextureView = match resolved_view {
                ResolvedColorView::Static(handle) => ctx
                    .resources
                    .get(*handle)
                    .expect("color attachment view missing from registry"),
                ResolvedColorView::Frame(view) => view.as_ref(),
            };
            let load = match attachment.load {
                ColorLoadOp::Load => wgpu::LoadOp::Load,
                ColorLoadOp::Clear(color) => wgpu::LoadOp::Clear(color),
            };

            color_views.push(Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            }));
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&self.name),
                color_attachments: &color_views,
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            pass.set_pipeline(pipeline);

            for bind_group in &self.bind_groups {
                let resource = ctx
                    .get_bind_group(bind_group.handle)
                    .expect("bind group handle missing from registry");
                pass.set_bind_group(bind_group.index, resource, &[]);
            }

            for vertex_buffer in &self.vertex_buffers {
                let buffer = ctx
                    .get_buffer(vertex_buffer.buffer)
                    .expect("vertex buffer handle missing from registry");
                let slice = match vertex_buffer.size {
                    Some(size) => buffer.slice(vertex_buffer.offset..vertex_buffer.offset + size),
                    None => buffer.slice(vertex_buffer.offset..),
                };
                pass.set_vertex_buffer(vertex_buffer.slot, slice);
            }

            match self.draw {
                RenderDraw::Direct {
                    ref vertices,
                    ref instances,
                } => {
                    pass.draw(vertices.clone(), instances.clone());
                }
                RenderDraw::Indirect { buffer, offset } => {
                    let args = ctx
                        .get_buffer(buffer)
                        .expect("indirect draw buffer missing from registry");
                    pass.draw_indirect(args, offset);
                }
            }
        }

        encoder.finish()
    }
}

pub struct RenderPassBuilder {
    name: String,
    reads: Vec<u64>,
    writes: Vec<u64>,
    pipeline: Option<Handle<wgpu::RenderPipeline>>,
    color_attachments: Vec<ColorAttachment>,
    vertex_buffers: Vec<VertexBufferBinding>,
    bind_groups: Vec<BoundBindGroup>,
    draw: Option<RenderDraw>,
}

impl RenderPassBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            reads: Vec::new(),
            writes: Vec::new(),
            pipeline: None,
            color_attachments: Vec::new(),
            vertex_buffers: Vec::new(),
            bind_groups: Vec::new(),
            draw: None,
        }
    }

    pub fn read<T: ResourceType>(mut self, handle: Handle<T>) -> Self {
        self.reads.push(handle.id());
        self
    }

    pub fn write<T: ResourceType>(mut self, handle: Handle<T>) -> Self {
        self.writes.push(handle.id());
        self
    }

    pub fn read_write<T: ResourceType>(mut self, handle: Handle<T>) -> Self {
        self.reads.push(handle.id());
        self.writes.push(handle.id());
        self
    }

    pub fn with_pipeline(mut self, pipeline: Handle<wgpu::RenderPipeline>) -> Self {
        self.pipeline = Some(pipeline);
        self
    }

    pub fn with_bind_group(mut self, index: u32, bind_group: Handle<wgpu::BindGroup>) -> Self {
        self.bind_groups.push(BoundBindGroup {
            index,
            handle: bind_group,
        });
        self
    }

    pub fn with_vertex_buffer(mut self, slot: u32, buffer: Handle<wgpu::Buffer>) -> Self {
        self.vertex_buffers.push(VertexBufferBinding {
            slot,
            buffer,
            offset: 0,
            size: None,
        });
        self
    }

    pub fn with_vertex_buffer_slice(
        mut self,
        slot: u32,
        buffer: Handle<wgpu::Buffer>,
        offset: u64,
        size: u64,
    ) -> Self {
        self.vertex_buffers.push(VertexBufferBinding {
            slot,
            buffer,
            offset,
            size: Some(size),
        });
        self
    }

    pub fn with_color_attachment(
        mut self,
        view: Handle<wgpu::TextureView>,
        load: ColorLoadOp,
    ) -> Self {
        self.writes.push(view.id());
        self.color_attachments.push(ColorAttachment {
            view: ColorAttachmentView::Static(view),
            load,
        });
        self
    }

    pub fn with_frame_color_attachment(
        mut self,
        view: Handle<FrameTextureView>,
        load: ColorLoadOp,
    ) -> Self {
        self.writes.push(view.id());
        self.color_attachments.push(ColorAttachment {
            view: ColorAttachmentView::FrameSlot(view),
            load,
        });
        self
    }

    pub fn draw(mut self, vertex_count: u32, instance_count: u32) -> Self {
        self.draw = Some(RenderDraw::direct(vertex_count, instance_count));
        self
    }

    pub fn draw_ranges(
        mut self,
        vertices: std::ops::Range<u32>,
        instances: std::ops::Range<u32>,
    ) -> Self {
        self.draw = Some(RenderDraw::direct_ranges(vertices, instances));
        self
    }

    pub fn draw_indirect(mut self, buffer: Handle<wgpu::Buffer>, offset: u64) -> Self {
        self.reads.push(buffer.id());
        self.draw = Some(RenderDraw::indirect(buffer, offset));
        self
    }

    pub fn build(self) -> Result<PassBuilder, RenderPassError> {
        let pipeline = self.pipeline.ok_or(RenderPassError::MissingPipeline)?;
        let draw = self.draw.ok_or(RenderPassError::MissingDraw)?;
        if self.color_attachments.is_empty() {
            return Err(RenderPassError::MissingColorAttachment);
        }

        let mut builder = PassBuilder::new(self.name.clone());
        for read in self.reads {
            builder.read_handle_id(read);
        }
        for write in self.writes {
            builder.write_handle_id(write);
        }

        Ok(builder.with_pass(Box::new(RenderDispatchPass {
            name: self.name,
            pipeline,
            color_attachments: self.color_attachments,
            vertex_buffers: self.vertex_buffers,
            bind_groups: self.bind_groups,
            draw,
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_pass_builder_requires_pipeline() {
        let view = Handle::<wgpu::TextureView>::next();
        let err = RenderPassBuilder::new("render")
            .with_color_attachment(view, ColorLoadOp::Load)
            .draw(3, 1)
            .build()
            .err()
            .expect("builder should require a pipeline");

        assert!(matches!(err, RenderPassError::MissingPipeline));
    }

    #[test]
    fn test_render_pass_builder_requires_color_attachment() {
        let pipeline = Handle::<wgpu::RenderPipeline>::next();
        let err = RenderPassBuilder::new("render")
            .with_pipeline(pipeline)
            .draw(3, 1)
            .build()
            .err()
            .expect("builder should require a color attachment");

        assert!(matches!(err, RenderPassError::MissingColorAttachment));
    }

    #[test]
    fn test_render_pass_builder_requires_draw() {
        let pipeline = Handle::<wgpu::RenderPipeline>::next();
        let view = Handle::<wgpu::TextureView>::next();
        let err = RenderPassBuilder::new("render")
            .with_pipeline(pipeline)
            .with_color_attachment(view, ColorLoadOp::Load)
            .build()
            .err()
            .expect("builder should require a draw");

        assert!(matches!(err, RenderPassError::MissingDraw));
    }
}
