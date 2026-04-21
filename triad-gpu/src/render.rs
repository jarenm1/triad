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

/// How depth is loaded at the start of a render pass attachment.
#[derive(Debug, Clone, Copy)]
pub enum DepthLoadOp {
    Load,
    Clear(f32),
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
    DirectIndexed {
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    },
    IndirectIndexed {
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

    pub fn direct_indexed(index_count: u32, base_vertex: i32, instance_count: u32) -> Self {
        Self::DirectIndexed {
            indices: 0..index_count,
            base_vertex,
            instances: 0..instance_count,
        }
    }

    pub fn direct_indexed_ranges(
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    ) -> Self {
        Self::DirectIndexed {
            indices,
            base_vertex,
            instances,
        }
    }

    pub fn indirect_indexed(buffer: Handle<wgpu::Buffer>, offset: u64) -> Self {
        Self::IndirectIndexed { buffer, offset }
    }

    fn is_indexed(&self) -> bool {
        matches!(
            self,
            RenderDraw::DirectIndexed { .. } | RenderDraw::IndirectIndexed { .. }
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ColorAttachmentView {
    StaticView(Handle<wgpu::TextureView>),
    Texture(Handle<wgpu::Texture>),
    FrameSlot(Handle<FrameTextureView>),
}

#[derive(Debug, Clone, Copy)]
struct ColorAttachment {
    view: ColorAttachmentView,
    load: ColorLoadOp,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum DepthAttachmentView {
    StaticView(Handle<wgpu::TextureView>),
    Texture(Handle<wgpu::Texture>),
    FrameSlot(Handle<FrameTextureView>),
}

#[derive(Debug, Clone)]
struct DepthStencilAttachment {
    view: DepthAttachmentView,
    depth_ops: wgpu::Operations<f32>,
    stencil_ops: Option<wgpu::Operations<u32>>,
}

#[derive(Debug, Clone, Copy)]
struct VertexBufferBinding {
    slot: u32,
    buffer: Handle<wgpu::Buffer>,
    offset: u64,
    size: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
struct IndexBufferBinding {
    buffer: Handle<wgpu::Buffer>,
    offset: u64,
    size: Option<u64>,
    format: wgpu::IndexFormat,
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
    depth_stencil: Option<DepthStencilAttachment>,
    vertex_buffers: Vec<VertexBufferBinding>,
    index_buffer: Option<IndexBufferBinding>,
    bind_groups: Vec<BoundBindGroup>,
    draw: RenderDraw,
}

enum ResolvedColorView {
    StaticHandle(Handle<wgpu::TextureView>),
    Frame(Arc<wgpu::TextureView>),
    Owned(wgpu::TextureView),
}

enum ResolvedDepthView {
    StaticHandle(Handle<wgpu::TextureView>),
    Frame(Arc<wgpu::TextureView>),
    Owned(wgpu::TextureView),
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
                ColorAttachmentView::StaticView(handle) => ResolvedColorView::StaticHandle(handle),
                ColorAttachmentView::Texture(handle) => {
                    let texture = ctx
                        .get_texture(handle)
                        .expect("color attachment texture missing from registry or transients");
                    ResolvedColorView::Owned(
                        texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    )
                }
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
                ResolvedColorView::StaticHandle(handle) => ctx
                    .resources
                    .get(*handle)
                    .expect("color attachment view missing from registry"),
                ResolvedColorView::Frame(view) => view.as_ref(),
                ResolvedColorView::Owned(view) => view,
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

        let resolved_depth_view: Option<ResolvedDepthView> =
            self.depth_stencil.as_ref().map(|ds| match ds.view {
                DepthAttachmentView::StaticView(handle) => ResolvedDepthView::StaticHandle(handle),
                DepthAttachmentView::Texture(handle) => {
                    let texture = ctx
                        .get_texture(handle)
                        .expect("depth attachment texture missing from registry or transients");
                    ResolvedDepthView::Owned(
                        texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    )
                }
                DepthAttachmentView::FrameSlot(slot_h) => {
                    let slot = ctx
                        .resources
                        .get(slot_h)
                        .expect("frame texture view slot missing from registry");
                    ResolvedDepthView::Frame(
                        slot.get()
                            .expect("frame texture view slot should be populated before execution"),
                    )
                }
            });

        let depth_stencil_for_pass: Option<wgpu::RenderPassDepthStencilAttachment<'_>> =
            self.depth_stencil.as_ref().and_then(|ds| {
                let view: &wgpu::TextureView = match resolved_depth_view.as_ref()? {
                    ResolvedDepthView::StaticHandle(handle) => ctx.resources.get(*handle)?,
                    ResolvedDepthView::Frame(view) => view.as_ref(),
                    ResolvedDepthView::Owned(view) => view,
                };
                Some(wgpu::RenderPassDepthStencilAttachment {
                    view,
                    depth_ops: Some(ds.depth_ops),
                    stencil_ops: ds.stencil_ops,
                })
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&self.name),
                color_attachments: &color_views,
                depth_stencil_attachment: depth_stencil_for_pass,
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

            if let Some(index_buffer) = self.index_buffer {
                let buffer = ctx
                    .get_buffer(index_buffer.buffer)
                    .expect("index buffer handle missing from registry");
                let slice = match index_buffer.size {
                    Some(size) => buffer.slice(index_buffer.offset..index_buffer.offset + size),
                    None => buffer.slice(index_buffer.offset..),
                };
                pass.set_index_buffer(slice, index_buffer.format);
            }

            match &self.draw {
                RenderDraw::Direct {
                    vertices,
                    instances,
                } => {
                    pass.draw(vertices.clone(), instances.clone());
                }
                RenderDraw::Indirect { buffer, offset } => {
                    let args = ctx
                        .get_buffer(*buffer)
                        .expect("indirect draw buffer missing from registry");
                    pass.draw_indirect(args, *offset);
                }
                RenderDraw::DirectIndexed {
                    indices,
                    base_vertex,
                    instances,
                } => {
                    pass.draw_indexed(indices.clone(), *base_vertex, instances.clone());
                }
                RenderDraw::IndirectIndexed { buffer, offset } => {
                    let args = ctx
                        .get_buffer(*buffer)
                        .expect("indirect indexed draw buffer missing from registry");
                    pass.draw_indexed_indirect(args, *offset);
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
    depth_stencil: Option<DepthStencilAttachment>,
    vertex_buffers: Vec<VertexBufferBinding>,
    index_buffer: Option<IndexBufferBinding>,
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
            depth_stencil: None,
            vertex_buffers: Vec::new(),
            index_buffer: None,
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

    /// Binds an index buffer for [`Self::draw_indexed`](Self::draw_indexed) /
    /// [`Self::draw_indexed_ranges`](Self::draw_indexed_ranges) /
    /// [`Self::draw_indexed_indirect`](Self::draw_indexed_indirect).
    pub fn with_index_buffer(
        mut self,
        buffer: Handle<wgpu::Buffer>,
        format: wgpu::IndexFormat,
    ) -> Self {
        self.reads.push(buffer.id());
        self.index_buffer = Some(IndexBufferBinding {
            buffer,
            offset: 0,
            size: None,
            format,
        });
        self
    }

    pub fn with_index_buffer_slice(
        mut self,
        buffer: Handle<wgpu::Buffer>,
        format: wgpu::IndexFormat,
        offset: u64,
        size: u64,
    ) -> Self {
        self.reads.push(buffer.id());
        self.index_buffer = Some(IndexBufferBinding {
            buffer,
            offset,
            size: Some(size),
            format,
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
            view: ColorAttachmentView::StaticView(view),
            load,
        });
        self
    }

    pub fn with_color_texture_attachment(
        mut self,
        texture: Handle<wgpu::Texture>,
        load: ColorLoadOp,
    ) -> Self {
        self.writes.push(texture.id());
        self.color_attachments.push(ColorAttachment {
            view: ColorAttachmentView::Texture(texture),
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

    /// Depth (and optional stencil) attachment. Stencil ops default to `None` (depth-only targets).
    pub fn with_depth_stencil_attachment(
        mut self,
        view: Handle<wgpu::TextureView>,
        depth_load: DepthLoadOp,
        depth_store: wgpu::StoreOp,
        stencil_ops: Option<wgpu::Operations<u32>>,
    ) -> Self {
        let id = view.id();
        self.reads.push(id);
        self.writes.push(id);
        let depth_load_op = match depth_load {
            DepthLoadOp::Load => wgpu::LoadOp::Load,
            DepthLoadOp::Clear(v) => wgpu::LoadOp::Clear(v),
        };
        self.depth_stencil = Some(DepthStencilAttachment {
            view: DepthAttachmentView::StaticView(view),
            depth_ops: wgpu::Operations {
                load: depth_load_op,
                store: depth_store,
            },
            stencil_ops,
        });
        self
    }

    /// Depth (and optional stencil) attachment backed directly by a texture handle. A default
    /// view is created at execution time, which lets transient textures participate without
    /// pre-registering persistent texture views.
    pub fn with_depth_stencil_texture(
        mut self,
        texture: Handle<wgpu::Texture>,
        depth_load: DepthLoadOp,
        depth_store: wgpu::StoreOp,
        stencil_ops: Option<wgpu::Operations<u32>>,
    ) -> Self {
        let id = texture.id();
        self.reads.push(id);
        self.writes.push(id);
        let depth_load_op = match depth_load {
            DepthLoadOp::Load => wgpu::LoadOp::Load,
            DepthLoadOp::Clear(v) => wgpu::LoadOp::Clear(v),
        };
        self.depth_stencil = Some(DepthStencilAttachment {
            view: DepthAttachmentView::Texture(texture),
            depth_ops: wgpu::Operations {
                load: depth_load_op,
                store: depth_store,
            },
            stencil_ops,
        });
        self
    }

    pub fn with_frame_depth_stencil_attachment(
        mut self,
        view: Handle<FrameTextureView>,
        depth_load: DepthLoadOp,
        depth_store: wgpu::StoreOp,
        stencil_ops: Option<wgpu::Operations<u32>>,
    ) -> Self {
        let id = view.id();
        self.reads.push(id);
        self.writes.push(id);
        let depth_load_op = match depth_load {
            DepthLoadOp::Load => wgpu::LoadOp::Load,
            DepthLoadOp::Clear(v) => wgpu::LoadOp::Clear(v),
        };
        self.depth_stencil = Some(DepthStencilAttachment {
            view: DepthAttachmentView::FrameSlot(view),
            depth_ops: wgpu::Operations {
                load: depth_load_op,
                store: depth_store,
            },
            stencil_ops,
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

    pub fn draw_indexed(mut self, index_count: u32, base_vertex: i32, instance_count: u32) -> Self {
        self.draw = Some(RenderDraw::direct_indexed(
            index_count,
            base_vertex,
            instance_count,
        ));
        self
    }

    pub fn draw_indexed_ranges(
        mut self,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    ) -> Self {
        self.draw = Some(RenderDraw::direct_indexed_ranges(
            indices,
            base_vertex,
            instances,
        ));
        self
    }

    pub fn draw_indexed_indirect(mut self, buffer: Handle<wgpu::Buffer>, offset: u64) -> Self {
        self.reads.push(buffer.id());
        self.draw = Some(RenderDraw::indirect_indexed(buffer, offset));
        self
    }

    pub fn build(self) -> Result<PassBuilder, RenderPassError> {
        let pipeline = self.pipeline.ok_or(RenderPassError::MissingPipeline)?;
        let draw = self.draw.as_ref().ok_or(RenderPassError::MissingDraw)?;
        if self.color_attachments.is_empty() {
            return Err(RenderPassError::MissingColorAttachment);
        }

        let indexed = draw.is_indexed();
        match (&self.index_buffer, indexed) {
            (None, true) => return Err(RenderPassError::MissingIndexBuffer),
            (Some(_), false) => return Err(RenderPassError::UnexpectedIndexBuffer),
            _ => {}
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
            depth_stencil: self.depth_stencil,
            vertex_buffers: self.vertex_buffers,
            index_buffer: self.index_buffer,
            bind_groups: self.bind_groups,
            draw: self.draw.expect("draw checked above"),
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
    fn test_render_pass_builder_accepts_texture_color_attachment() {
        let pipeline = Handle::<wgpu::RenderPipeline>::next();
        let texture = Handle::<wgpu::Texture>::next();

        let pass = RenderPassBuilder::new("render")
            .with_pipeline(pipeline)
            .with_color_texture_attachment(texture, ColorLoadOp::Load)
            .draw(3, 1)
            .build();

        assert!(pass.is_ok());
    }

    #[test]
    fn test_render_pass_builder_accepts_texture_depth_attachment() {
        let pipeline = Handle::<wgpu::RenderPipeline>::next();
        let color = Handle::<wgpu::Texture>::next();
        let depth = Handle::<wgpu::Texture>::next();

        let pass = RenderPassBuilder::new("render")
            .with_pipeline(pipeline)
            .with_color_texture_attachment(color, ColorLoadOp::Load)
            .with_depth_stencil_texture(depth, DepthLoadOp::Clear(1.0), wgpu::StoreOp::Store, None)
            .draw(3, 1)
            .build();

        assert!(pass.is_ok());
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

    #[test]
    fn test_indexed_draw_requires_index_buffer() {
        let pipeline = Handle::<wgpu::RenderPipeline>::next();
        let view = Handle::<wgpu::TextureView>::next();
        let err = RenderPassBuilder::new("render")
            .with_pipeline(pipeline)
            .with_color_attachment(view, ColorLoadOp::Load)
            .draw_indexed(6, 0, 1)
            .build()
            .err()
            .expect("indexed draw should require index buffer");

        assert!(matches!(err, RenderPassError::MissingIndexBuffer));
    }

    #[test]
    fn test_vertex_draw_rejects_index_buffer() {
        let pipeline = Handle::<wgpu::RenderPipeline>::next();
        let color = Handle::<wgpu::TextureView>::next();
        let index_buf = Handle::<wgpu::Buffer>::next();
        let err = RenderPassBuilder::new("render")
            .with_pipeline(pipeline)
            .with_color_attachment(color, ColorLoadOp::Load)
            .with_index_buffer(index_buf, wgpu::IndexFormat::Uint32)
            .draw(3, 1)
            .build()
            .err()
            .expect("vertex draw should not allow index buffer");

        assert!(matches!(err, RenderPassError::UnexpectedIndexBuffer));
    }
}
