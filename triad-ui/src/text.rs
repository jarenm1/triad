//! Text rendering using cosmic-text.

use crate::atlas::{GlyphAtlas, GlyphKey};
use crate::primitives::{Color, UiVertex};
use cosmic_text::{Attrs, Buffer, FontSystem, Metrics, Shaping, SwashCache};
use glam::Vec2;
use std::sync::Arc;

/// Style configuration for text rendering.
#[derive(Debug, Clone)]
pub struct TextStyle {
    /// Font size in pixels.
    pub font_size: f32,
    /// Line height multiplier.
    pub line_height: f32,
    /// Text color.
    pub color: Color,
}

impl Default for TextStyle {
    fn default() -> Self {
        Self {
            font_size: 16.0,
            line_height: 1.2,
            color: Color::WHITE,
        }
    }
}

impl TextStyle {
    /// Create a new text style with the given font size.
    pub fn new(font_size: f32) -> Self {
        Self {
            font_size,
            ..Default::default()
        }
    }

    /// Set the text color.
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Set the line height multiplier.
    pub fn with_line_height(mut self, line_height: f32) -> Self {
        self.line_height = line_height;
        self
    }
}

/// A section of text to render.
pub struct TextSection {
    /// The text content.
    pub text: String,
    /// Position in screen space.
    pub position: Vec2,
    /// Text style.
    pub style: TextStyle,
    /// Maximum width for wrapping (None = no wrap).
    pub max_width: Option<f32>,
}

impl TextSection {
    /// Create a new text section.
    pub fn new(text: impl Into<String>, position: Vec2) -> Self {
        Self {
            text: text.into(),
            position,
            style: TextStyle::default(),
            max_width: None,
        }
    }

    /// Set the text style.
    pub fn with_style(mut self, style: TextStyle) -> Self {
        self.style = style;
        self
    }

    /// Set the maximum width for text wrapping.
    pub fn with_max_width(mut self, width: f32) -> Self {
        self.max_width = Some(width);
        self
    }
}

/// Text renderer using cosmic-text for text shaping and layout.
pub struct TextRenderer {
    font_system: FontSystem,
    swash_cache: SwashCache,
    atlas: GlyphAtlas,
    /// Texture handle for the glyph atlas (managed externally).
    texture: Option<wgpu::Texture>,
    texture_view: Option<wgpu::TextureView>,
    bind_group: Option<wgpu::BindGroup>,
    bind_group_layout: Option<Arc<wgpu::BindGroupLayout>>,
    pipeline: Option<wgpu::RenderPipeline>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    /// Screen dimensions for coordinate conversion.
    screen_size: Vec2,
}

impl TextRenderer {
    /// Create a new text renderer.
    pub fn new() -> Self {
        let font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let atlas = GlyphAtlas::new(1024, 1024);

        Self {
            font_system,
            swash_cache,
            atlas,
            texture: None,
            texture_view: None,
            bind_group: None,
            bind_group_layout: None,
            pipeline: None,
            vertex_buffer: None,
            index_buffer: None,
            screen_size: Vec2::new(1280.0, 720.0),
        }
    }

    /// Initialize GPU resources.
    pub fn init(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) {
        let (width, height) = self.atlas.dimensions();

        // Create atlas texture
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Text Atlas"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create sampler
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Text Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Text Bind Group Layout"),
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bind_group_layout = Arc::new(bind_group_layout);

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Text Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Create shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Text Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/text.wgsl").into()),
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Text Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Text Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<UiVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: 8,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: 16,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Upload initial atlas data
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            self.atlas.data(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        self.texture = Some(texture);
        self.texture_view = Some(texture_view);
        self.bind_group = Some(bind_group);
        self.bind_group_layout = Some(bind_group_layout);
        self.pipeline = Some(pipeline);
    }

    /// Set the screen size for coordinate conversion.
    pub fn set_screen_size(&mut self, width: f32, height: f32) {
        self.screen_size = Vec2::new(width, height);
    }

    /// Prepare text sections for rendering.
    ///
    /// Returns vertices and indices for the text quads.
    pub fn prepare(
        &mut self,
        sections: &[TextSection],
    ) -> (Vec<UiVertex>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for section in sections {
            self.prepare_section(section, &mut vertices, &mut indices);
        }

        (vertices, indices)
    }

    fn prepare_section(
        &mut self,
        section: &TextSection,
        vertices: &mut Vec<UiVertex>,
        indices: &mut Vec<u32>,
    ) {
        let metrics = Metrics::new(section.style.font_size, section.style.font_size * section.style.line_height);
        let mut buffer = Buffer::new(&mut self.font_system, metrics);

        let width = section.max_width.unwrap_or(f32::MAX);
        buffer.set_size(&mut self.font_system, Some(width), None);
        buffer.set_text(&mut self.font_system, &section.text, Attrs::new(), Shaping::Advanced);
        buffer.shape_until_scroll(&mut self.font_system, false);

        let (atlas_width, atlas_height) = self.atlas.dimensions();

        for run in buffer.layout_runs() {
            for glyph in run.glyphs.iter() {
                let physical_glyph = glyph.physical((0.0, 0.0), 1.0);

                // Get or create glyph in atlas
                // Use a hash of the cache_key components as the font_id since font_id.0 is private
                let key = GlyphKey {
                    font_id: (physical_glyph.cache_key.glyph_id >> 8) as u16,
                    glyph_id: physical_glyph.cache_key.glyph_id,
                    size_px: (section.style.font_size * 10.0) as u16, // Quantize for caching
                };

                let atlas_rect = if let Some(rect) = self.atlas.get(&key) {
                    *rect
                } else {
                    // Rasterize glyph
                    if let Some(image) = self.swash_cache.get_image(&mut self.font_system, physical_glyph.cache_key) {
                        if image.placement.width > 0 && image.placement.height > 0 {
                            if let Some(rect) = self.atlas.insert(
                                key,
                                image.placement.width as u32,
                                image.placement.height as u32,
                                &image.data,
                            ) {
                                rect
                            } else {
                                continue;
                            }
                        } else {
                            continue;
                        }
                    } else {
                        continue;
                    }
                };

                // Calculate screen position
                let x = section.position.x + physical_glyph.x as f32;
                let y = section.position.y + run.line_y + physical_glyph.y as f32;

                let w = atlas_rect.width as f32;
                let h = atlas_rect.height as f32;

                // Convert to NDC
                let x0 = (x / self.screen_size.x) * 2.0 - 1.0;
                let y0 = 1.0 - (y / self.screen_size.y) * 2.0;
                let x1 = ((x + w) / self.screen_size.x) * 2.0 - 1.0;
                let y1 = 1.0 - ((y + h) / self.screen_size.y) * 2.0;

                // UV coordinates
                let uv = atlas_rect.uv(atlas_width, atlas_height);

                let base_idx = vertices.len() as u32;
                let color = section.style.color;

                vertices.push(UiVertex::new(Vec2::new(x0, y0), Vec2::new(uv[0], uv[1]), color));
                vertices.push(UiVertex::new(Vec2::new(x1, y0), Vec2::new(uv[2], uv[1]), color));
                vertices.push(UiVertex::new(Vec2::new(x1, y1), Vec2::new(uv[2], uv[3]), color));
                vertices.push(UiVertex::new(Vec2::new(x0, y1), Vec2::new(uv[0], uv[3]), color));

                indices.extend_from_slice(&[
                    base_idx,
                    base_idx + 1,
                    base_idx + 2,
                    base_idx,
                    base_idx + 2,
                    base_idx + 3,
                ]);
            }
        }
    }

    /// Upload prepared vertices to GPU and render.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        sections: &[TextSection],
    ) {
        let (vertices, indices) = self.prepare(sections);

        if vertices.is_empty() {
            return;
        }

        // Update atlas texture if dirty
        if self.atlas.is_dirty() {
            if let Some(ref texture) = self.texture {
                let (width, height) = self.atlas.dimensions();
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    self.atlas.data(),
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(width),
                        rows_per_image: Some(height),
                    },
                    wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                );
                self.atlas.mark_clean();
            }
        }

        // Create/update vertex buffer
        let vertex_data = bytemuck::cast_slice(&vertices);
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Text Vertex Buffer"),
            contents: vertex_data,
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Create/update index buffer
        let index_data = bytemuck::cast_slice(&indices);
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Text Index Buffer"),
            contents: index_data,
            usage: wgpu::BufferUsages::INDEX,
        });

        // Render
        if let (Some(pipeline), Some(bind_group)) = (&self.pipeline, &self.bind_group) {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Text Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
        }

        self.vertex_buffer = Some(vertex_buffer);
        self.index_buffer = Some(index_buffer);
    }
}

impl Default for TextRenderer {
    fn default() -> Self {
        Self::new()
    }
}

// Need to add wgpu::util for BufferInitDescriptor
use wgpu::util::DeviceExt;

