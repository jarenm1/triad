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

#[derive(Debug, Clone, Copy)]
pub struct TextureCopy {
    src: Handle<wgpu::Texture>,
    dst: Handle<wgpu::Texture>,
    src_mip_level: u32,
    dst_mip_level: u32,
    src_origin: wgpu::Origin3d,
    dst_origin: wgpu::Origin3d,
    src_aspect: wgpu::TextureAspect,
    dst_aspect: wgpu::TextureAspect,
    extent: wgpu::Extent3d,
}

impl TextureCopy {
    pub fn new(
        src: Handle<wgpu::Texture>,
        dst: Handle<wgpu::Texture>,
        extent: wgpu::Extent3d,
    ) -> Self {
        Self {
            src,
            dst,
            src_mip_level: 0,
            dst_mip_level: 0,
            src_origin: wgpu::Origin3d::ZERO,
            dst_origin: wgpu::Origin3d::ZERO,
            src_aspect: wgpu::TextureAspect::All,
            dst_aspect: wgpu::TextureAspect::All,
            extent,
        }
    }

    pub fn with_src_mip_level(mut self, mip_level: u32) -> Self {
        self.src_mip_level = mip_level;
        self
    }

    pub fn with_dst_mip_level(mut self, mip_level: u32) -> Self {
        self.dst_mip_level = mip_level;
        self
    }

    pub fn with_src_origin(mut self, origin: wgpu::Origin3d) -> Self {
        self.src_origin = origin;
        self
    }

    pub fn with_dst_origin(mut self, origin: wgpu::Origin3d) -> Self {
        self.dst_origin = origin;
        self
    }

    pub fn with_src_aspect(mut self, aspect: wgpu::TextureAspect) -> Self {
        self.src_aspect = aspect;
        self
    }

    pub fn with_dst_aspect(mut self, aspect: wgpu::TextureAspect) -> Self {
        self.dst_aspect = aspect;
        self
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TextureBufferCopy {
    texture: Handle<wgpu::Texture>,
    buffer: BufferCopyEndpoint,
    texture_mip_level: u32,
    texture_origin: wgpu::Origin3d,
    texture_aspect: wgpu::TextureAspect,
    buffer_offset: u64,
    bytes_per_row: Option<u32>,
    rows_per_image: Option<u32>,
    extent: wgpu::Extent3d,
}

impl TextureBufferCopy {
    pub fn texture_to_buffer(
        texture: Handle<wgpu::Texture>,
        buffer: Handle<wgpu::Buffer>,
        extent: wgpu::Extent3d,
    ) -> Self {
        Self {
            texture,
            buffer: BufferCopyEndpoint::Static(buffer),
            texture_mip_level: 0,
            texture_origin: wgpu::Origin3d::ZERO,
            texture_aspect: wgpu::TextureAspect::All,
            buffer_offset: 0,
            bytes_per_row: None,
            rows_per_image: None,
            extent,
        }
    }

    pub fn texture_to_frame_slot(
        texture: Handle<wgpu::Texture>,
        buffer: Handle<FrameBufferHandle>,
        extent: wgpu::Extent3d,
    ) -> Self {
        Self {
            texture,
            buffer: BufferCopyEndpoint::FrameSlot(buffer),
            texture_mip_level: 0,
            texture_origin: wgpu::Origin3d::ZERO,
            texture_aspect: wgpu::TextureAspect::All,
            buffer_offset: 0,
            bytes_per_row: None,
            rows_per_image: None,
            extent,
        }
    }

    pub fn buffer_to_texture(
        buffer: Handle<wgpu::Buffer>,
        texture: Handle<wgpu::Texture>,
        extent: wgpu::Extent3d,
    ) -> Self {
        Self {
            texture,
            buffer: BufferCopyEndpoint::Static(buffer),
            texture_mip_level: 0,
            texture_origin: wgpu::Origin3d::ZERO,
            texture_aspect: wgpu::TextureAspect::All,
            buffer_offset: 0,
            bytes_per_row: None,
            rows_per_image: None,
            extent,
        }
    }

    pub fn with_texture_mip_level(mut self, mip_level: u32) -> Self {
        self.texture_mip_level = mip_level;
        self
    }

    pub fn with_texture_origin(mut self, origin: wgpu::Origin3d) -> Self {
        self.texture_origin = origin;
        self
    }

    pub fn with_texture_aspect(mut self, aspect: wgpu::TextureAspect) -> Self {
        self.texture_aspect = aspect;
        self
    }

    pub fn with_buffer_offset(mut self, offset: u64) -> Self {
        self.buffer_offset = offset;
        self
    }

    pub fn with_layout(mut self, bytes_per_row: Option<u32>, rows_per_image: Option<u32>) -> Self {
        self.bytes_per_row = bytes_per_row;
        self.rows_per_image = rows_per_image;
        self
    }
}

#[derive(Debug, Clone, Copy)]
enum CopyCommand {
    Buffer(BufferCopy),
    Texture(TextureCopy),
    TextureToBuffer(TextureBufferCopy),
    BufferToTexture(TextureBufferCopy),
}

#[derive(Debug)]
struct CopyPass {
    name: String,
    copies: Vec<CopyCommand>,
}

impl Pass for CopyPass {
    fn name(&self) -> &str {
        &self.name
    }

    fn execute(&self, ctx: &PassContext) -> wgpu::CommandBuffer {
        let mut encoder = ctx.create_command_encoder(Some(&self.name));

        for copy in &self.copies {
            match copy {
                CopyCommand::Buffer(copy) => {
                    let src = resolve_buffer_endpoint(ctx, copy.src).expect(
                        "copy source buffer missing from registry or transient graph resources",
                    );
                    let dst = resolve_buffer_endpoint(ctx, copy.dst).expect(
                        "copy destination buffer missing from registry, transient graph resources, or frame slot",
                    );
                    encoder.copy_buffer_to_buffer(
                        src,
                        copy.src_offset,
                        dst,
                        copy.dst_offset,
                        copy.size,
                    );
                }
                CopyCommand::Texture(copy) => {
                    let src = ctx.get_texture(copy.src).expect(
                        "copy source texture missing from registry or transient graph resources",
                    );
                    let dst = ctx
                        .get_texture(copy.dst)
                        .expect("copy destination texture missing from registry or transient graph resources");
                    encoder.copy_texture_to_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: src,
                            mip_level: copy.src_mip_level,
                            origin: copy.src_origin,
                            aspect: copy.src_aspect,
                        },
                        wgpu::TexelCopyTextureInfo {
                            texture: dst,
                            mip_level: copy.dst_mip_level,
                            origin: copy.dst_origin,
                            aspect: copy.dst_aspect,
                        },
                        copy.extent,
                    );
                }
                CopyCommand::TextureToBuffer(copy) => {
                    let texture = ctx.get_texture(copy.texture).expect(
                        "copy source texture missing from registry or transient graph resources",
                    );
                    let buffer = resolve_buffer_endpoint(ctx, copy.buffer).expect(
                        "copy destination buffer missing from registry, transient graph resources, or frame slot",
                    );
                    encoder.copy_texture_to_buffer(
                        wgpu::TexelCopyTextureInfo {
                            texture,
                            mip_level: copy.texture_mip_level,
                            origin: copy.texture_origin,
                            aspect: copy.texture_aspect,
                        },
                        wgpu::TexelCopyBufferInfo {
                            buffer,
                            layout: wgpu::TexelCopyBufferLayout {
                                offset: copy.buffer_offset,
                                bytes_per_row: copy.bytes_per_row,
                                rows_per_image: copy.rows_per_image,
                            },
                        },
                        copy.extent,
                    );
                }
                CopyCommand::BufferToTexture(copy) => {
                    let buffer = resolve_buffer_endpoint(ctx, copy.buffer).expect(
                        "copy source buffer missing from registry, transient graph resources, or frame slot",
                    );
                    let texture = ctx
                        .get_texture(copy.texture)
                        .expect("copy destination texture missing from registry or transient graph resources");
                    encoder.copy_buffer_to_texture(
                        wgpu::TexelCopyBufferInfo {
                            buffer,
                            layout: wgpu::TexelCopyBufferLayout {
                                offset: copy.buffer_offset,
                                bytes_per_row: copy.bytes_per_row,
                                rows_per_image: copy.rows_per_image,
                            },
                        },
                        wgpu::TexelCopyTextureInfo {
                            texture,
                            mip_level: copy.texture_mip_level,
                            origin: copy.texture_origin,
                            aspect: copy.texture_aspect,
                        },
                        copy.extent,
                    );
                }
            }
        }

        encoder.finish()
    }
}

pub struct CopyPassBuilder {
    name: String,
    copies: Vec<CopyCommand>,
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
        self.copies
            .push(CopyCommand::Buffer(BufferCopy::new(src, dst, size)));
        self
    }

    pub fn copy_buffer_to_frame_slot(
        mut self,
        src: Handle<wgpu::Buffer>,
        dst: Handle<FrameBufferHandle>,
        size: u64,
    ) -> Self {
        self.copies
            .push(CopyCommand::Buffer(BufferCopy::to_frame_slot(
                src, dst, size,
            )));
        self
    }

    pub fn copy_texture(
        mut self,
        src: Handle<wgpu::Texture>,
        dst: Handle<wgpu::Texture>,
        extent: wgpu::Extent3d,
    ) -> Self {
        self.copies
            .push(CopyCommand::Texture(TextureCopy::new(src, dst, extent)));
        self
    }

    pub fn copy_texture_region(mut self, copy: TextureCopy) -> Self {
        self.copies.push(CopyCommand::Texture(copy));
        self
    }

    pub fn copy_texture_to_buffer(
        mut self,
        texture: Handle<wgpu::Texture>,
        buffer: Handle<wgpu::Buffer>,
        extent: wgpu::Extent3d,
    ) -> Self {
        self.copies.push(CopyCommand::TextureToBuffer(
            TextureBufferCopy::texture_to_buffer(texture, buffer, extent),
        ));
        self
    }

    pub fn copy_texture_to_frame_slot(
        mut self,
        texture: Handle<wgpu::Texture>,
        buffer: Handle<FrameBufferHandle>,
        extent: wgpu::Extent3d,
    ) -> Self {
        self.copies.push(CopyCommand::TextureToBuffer(
            TextureBufferCopy::texture_to_frame_slot(texture, buffer, extent),
        ));
        self
    }

    pub fn copy_texture_to_buffer_region(mut self, copy: TextureBufferCopy) -> Self {
        self.copies.push(CopyCommand::TextureToBuffer(copy));
        self
    }

    pub fn copy_buffer_to_texture(
        mut self,
        buffer: Handle<wgpu::Buffer>,
        texture: Handle<wgpu::Texture>,
        extent: wgpu::Extent3d,
    ) -> Self {
        self.copies.push(CopyCommand::BufferToTexture(
            TextureBufferCopy::buffer_to_texture(buffer, texture, extent),
        ));
        self
    }

    pub fn copy_buffer_to_texture_region(mut self, copy: TextureBufferCopy) -> Self {
        self.copies.push(CopyCommand::BufferToTexture(copy));
        self
    }

    pub fn copy_buffer_region(mut self, copy: BufferCopy) -> Self {
        self.copies.push(CopyCommand::Buffer(copy));
        self
    }

    pub fn build(self) -> Result<PassBuilder, CopyPassError> {
        if self.copies.is_empty() {
            return Err(CopyPassError::MissingCopy);
        }

        let mut builder = PassBuilder::new(self.name.clone());
        for copy in &self.copies {
            match copy {
                CopyCommand::Buffer(copy) => {
                    register_buffer_endpoint_read(&mut builder, copy.src);
                    register_buffer_endpoint_write(&mut builder, copy.dst);
                }
                CopyCommand::Texture(copy) => {
                    builder.read(copy.src);
                    builder.write(copy.dst);
                }
                CopyCommand::TextureToBuffer(copy) => {
                    builder.read(copy.texture);
                    register_buffer_endpoint_write(&mut builder, copy.buffer);
                }
                CopyCommand::BufferToTexture(copy) => {
                    register_buffer_endpoint_read(&mut builder, copy.buffer);
                    builder.write(copy.texture);
                }
            }
        }

        Ok(builder.with_pass(Box::new(CopyPass {
            name: self.name,
            copies: self.copies,
        })))
    }
}

fn register_buffer_endpoint_read(builder: &mut PassBuilder, endpoint: BufferCopyEndpoint) {
    match endpoint {
        BufferCopyEndpoint::Static(handle) => {
            builder.read(handle);
        }
        BufferCopyEndpoint::FrameSlot(handle) => {
            builder.read(handle);
        }
    }
}

fn register_buffer_endpoint_write(builder: &mut PassBuilder, endpoint: BufferCopyEndpoint) {
    match endpoint {
        BufferCopyEndpoint::Static(handle) => {
            builder.write(handle);
        }
        BufferCopyEndpoint::FrameSlot(handle) => {
            builder.write(handle);
        }
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

    #[test]
    fn test_copy_pass_builder_accepts_texture_to_buffer_copy() {
        let texture = Handle::<wgpu::Texture>::next();
        let buffer = Handle::<wgpu::Buffer>::next();
        let pass = CopyPassBuilder::new("copy")
            .copy_texture_to_buffer(
                texture,
                buffer,
                wgpu::Extent3d {
                    width: 64,
                    height: 64,
                    depth_or_array_layers: 1,
                },
            )
            .build();

        assert!(pass.is_ok());
    }

    #[test]
    fn test_copy_pass_builder_accepts_texture_to_texture_copy() {
        let src = Handle::<wgpu::Texture>::next();
        let dst = Handle::<wgpu::Texture>::next();
        let pass = CopyPassBuilder::new("copy")
            .copy_texture(
                src,
                dst,
                wgpu::Extent3d {
                    width: 32,
                    height: 32,
                    depth_or_array_layers: 1,
                },
            )
            .build();

        assert!(pass.is_ok());
    }

    #[test]
    fn test_copy_pass_builder_accepts_buffer_to_texture_copy() {
        let buffer = Handle::<wgpu::Buffer>::next();
        let texture = Handle::<wgpu::Texture>::next();
        let pass = CopyPassBuilder::new("copy")
            .copy_buffer_to_texture(
                buffer,
                texture,
                wgpu::Extent3d {
                    width: 16,
                    height: 16,
                    depth_or_array_layers: 1,
                },
            )
            .build();

        assert!(pass.is_ok());
    }
}
