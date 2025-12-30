//! Glyph atlas management for efficient text rendering.

use std::collections::HashMap;
use tracing::debug;

/// A packed rectangle in the atlas.
#[derive(Debug, Clone, Copy)]
pub struct AtlasRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl AtlasRect {
    /// Get UV coordinates for this rect in a texture of the given size.
    pub fn uv(&self, atlas_width: u32, atlas_height: u32) -> [f32; 4] {
        let u0 = self.x as f32 / atlas_width as f32;
        let v0 = self.y as f32 / atlas_height as f32;
        let u1 = (self.x + self.width) as f32 / atlas_width as f32;
        let v1 = (self.y + self.height) as f32 / atlas_height as f32;
        [u0, v0, u1, v1]
    }
}

/// Key for looking up glyphs in the atlas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlyphKey {
    /// Font ID (index into font system).
    pub font_id: u16,
    /// Glyph ID within the font.
    pub glyph_id: u16,
    /// Font size in pixels (quantized for caching).
    pub size_px: u16,
}

/// A glyph atlas that packs glyphs into a texture.
pub struct GlyphAtlas {
    width: u32,
    height: u32,
    /// Packed glyph data (single channel alpha).
    data: Vec<u8>,
    /// Map from glyph key to atlas rect.
    glyphs: HashMap<GlyphKey, AtlasRect>,
    /// Current packing cursor.
    cursor_x: u32,
    cursor_y: u32,
    row_height: u32,
    /// Whether the atlas texture needs to be re-uploaded.
    dirty: bool,
}

impl GlyphAtlas {
    /// Create a new glyph atlas with the given dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![0; (width * height) as usize],
            glyphs: HashMap::new(),
            cursor_x: 0,
            cursor_y: 0,
            row_height: 0,
            dirty: false,
        }
    }

    /// Get the atlas dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Get the raw atlas data (single channel alpha).
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Check if the atlas has been modified since last upload.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Mark the atlas as clean (after uploading to GPU).
    pub fn mark_clean(&mut self) {
        self.dirty = false;
    }

    /// Look up a glyph in the atlas.
    pub fn get(&self, key: &GlyphKey) -> Option<&AtlasRect> {
        self.glyphs.get(key)
    }

    /// Insert a glyph into the atlas.
    ///
    /// Returns the atlas rect if successful, None if there's no space.
    pub fn insert(&mut self, key: GlyphKey, width: u32, height: u32, data: &[u8]) -> Option<AtlasRect> {
        // Check if already exists
        if let Some(rect) = self.glyphs.get(&key) {
            return Some(*rect);
        }

        // Try to fit in current row
        if self.cursor_x + width > self.width {
            // Move to next row
            self.cursor_x = 0;
            self.cursor_y += self.row_height;
            self.row_height = 0;
        }

        // Check if there's vertical space
        if self.cursor_y + height > self.height {
            debug!("Glyph atlas full, cannot insert glyph");
            return None;
        }

        let rect = AtlasRect {
            x: self.cursor_x,
            y: self.cursor_y,
            width,
            height,
        };

        // Copy glyph data into atlas
        for row in 0..height {
            let src_start = (row * width) as usize;
            let src_end = src_start + width as usize;
            let dst_start = ((self.cursor_y + row) * self.width + self.cursor_x) as usize;

            if src_end <= data.len() {
                self.data[dst_start..dst_start + width as usize]
                    .copy_from_slice(&data[src_start..src_end]);
            }
        }

        // Update cursor
        self.cursor_x += width;
        self.row_height = self.row_height.max(height);
        self.dirty = true;

        self.glyphs.insert(key, rect);
        Some(rect)
    }

    /// Clear the atlas and reset packing state.
    pub fn clear(&mut self) {
        self.data.fill(0);
        self.glyphs.clear();
        self.cursor_x = 0;
        self.cursor_y = 0;
        self.row_height = 0;
        self.dirty = true;
    }
}

impl Default for GlyphAtlas {
    fn default() -> Self {
        Self::new(1024, 1024)
    }
}

