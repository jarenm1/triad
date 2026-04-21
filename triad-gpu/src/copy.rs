use crate::FrameBufferHandle;
use crate::error::CopyPassError;
use crate::frame_graph::Handle;
use crate::frame_graph::pass::{Pass, PassBuilder, PassContext};

#[derive(Debug, Clone, Copy)]
enum BufferCopyEndpoint {
    Static(Handle<wgpu::Buffer>),
    FrameSlot(Handle<FrameBufferHandle>),
}

#[derive(Debug, Clone, Copy)]
pub struct BufferCopy {
    src: BufferCopyEndpoint,
    dst: BufferCopyEndpoint,
    src_offset: u64,
    dst_offset: u64,
    size: u64,
}

impl BufferCopy {
    pub fn new(src: Handle<wgpu::Buffer>, dst: Handle<wgpu::Buffer>, size: u64) -> Self {
        Self {
            src: BufferCopyEndpoint::Static(src),
            dst: BufferCopyEndpoint::Static(dst),
            src_offset: 0,
            dst_offset: 0,
            size,
        }
    }

    pub fn to_frame_slot(
        src: Handle<wgpu::Buffer>,
        dst: Handle<FrameBufferHandle>,
        size: u64,
    ) -> Self {
        Self {
            src: BufferCopyEndpoint::Static(src),
            dst: BufferCopyEndpoint::FrameSlot(dst),
            src_offset: 0,
            dst_offset: 0,
            size,
        }
    }

    pub fn with_offsets(mut self, src_offset: u64, dst_offset: u64) -> Self {
        self.src_offset = src_offset;
        self.dst_offset = dst_offset;
        self
    }
}

#[derive(Debug)]
struct CopyBufferPass {
    name: String,
    copies: Vec<BufferCopy>,
}

impl Pass for CopyBufferPass {
    fn name(&self) -> &str {
        &self.name
    }

    fn execute(&self, ctx: &PassContext) -> wgpu::CommandBuffer {
        let mut encoder = ctx.create_command_encoder(Some(&self.name));

        for copy in &self.copies {
            let src = resolve_buffer_endpoint(ctx, copy.src)
                .expect("copy source buffer missing from registry or transient graph resources");
            let dst = resolve_buffer_endpoint(ctx, copy.dst).expect(
                "copy destination buffer missing from registry, transient graph resources, or frame slot",
            );
            encoder.copy_buffer_to_buffer(src, copy.src_offset, dst, copy.dst_offset, copy.size);
        }

        encoder.finish()
    }
}

pub struct CopyPassBuilder {
    name: String,
    copies: Vec<BufferCopy>,
}

impl CopyPassBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            copies: Vec::new(),
        }
    }

    pub fn copy_buffer(
        mut self,
        src: Handle<wgpu::Buffer>,
        dst: Handle<wgpu::Buffer>,
        size: u64,
    ) -> Self {
        self.copies.push(BufferCopy::new(src, dst, size));
        self
    }

    pub fn copy_buffer_to_frame_slot(
        mut self,
        src: Handle<wgpu::Buffer>,
        dst: Handle<FrameBufferHandle>,
        size: u64,
    ) -> Self {
        self.copies.push(BufferCopy::to_frame_slot(src, dst, size));
        self
    }

    pub fn copy_buffer_region(mut self, copy: BufferCopy) -> Self {
        self.copies.push(copy);
        self
    }

    pub fn build(self) -> Result<PassBuilder, CopyPassError> {
        if self.copies.is_empty() {
            return Err(CopyPassError::MissingCopy);
        }

        let mut builder = PassBuilder::new(self.name.clone());
        for copy in &self.copies {
            match copy.src {
                BufferCopyEndpoint::Static(handle) => {
                    builder.read(handle);
                }
                BufferCopyEndpoint::FrameSlot(handle) => {
                    builder.read(handle);
                }
            }
            match copy.dst {
                BufferCopyEndpoint::Static(handle) => {
                    builder.write(handle);
                }
                BufferCopyEndpoint::FrameSlot(handle) => {
                    builder.write(handle);
                }
            }
        }

        Ok(builder.with_pass(Box::new(CopyBufferPass {
            name: self.name,
            copies: self.copies,
        })))
    }
}

fn resolve_buffer_endpoint<'a>(
    ctx: &'a PassContext,
    endpoint: BufferCopyEndpoint,
) -> Option<&'a wgpu::Buffer> {
    match endpoint {
        BufferCopyEndpoint::Static(handle) => ctx.get_buffer(handle),
        BufferCopyEndpoint::FrameSlot(slot_handle) => {
            let slot = ctx.resources.get(slot_handle)?;
            let buffer_handle = slot.get()?;
            ctx.get_buffer(buffer_handle)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_copy_pass_builder_requires_copy() {
        let err = CopyPassBuilder::new("copy")
            .build()
            .err()
            .expect("builder should require a copy command");

        assert!(matches!(err, CopyPassError::MissingCopy));
    }
}
