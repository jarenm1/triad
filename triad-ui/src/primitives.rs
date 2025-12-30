//! Basic UI primitives for rendering shapes.

use glam::Vec2;

/// RGBA color with values in 0-1 range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const WHITE: Self = Self::rgb(1.0, 1.0, 1.0);
    pub const BLACK: Self = Self::rgb(0.0, 0.0, 0.0);
    pub const RED: Self = Self::rgb(1.0, 0.0, 0.0);
    pub const GREEN: Self = Self::rgb(0.0, 1.0, 0.0);
    pub const BLUE: Self = Self::rgb(0.0, 0.0, 1.0);
    pub const TRANSPARENT: Self = Self::rgba(0.0, 0.0, 0.0, 0.0);

    /// Create an opaque RGB color.
    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    /// Create an RGBA color.
    pub const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Create a color from a hex value (0xRRGGBB).
    pub fn from_hex(hex: u32) -> Self {
        let r = ((hex >> 16) & 0xFF) as f32 / 255.0;
        let g = ((hex >> 8) & 0xFF) as f32 / 255.0;
        let b = (hex & 0xFF) as f32 / 255.0;
        Self::rgb(r, g, b)
    }

    /// Create a color from a hex value with alpha (0xRRGGBBAA).
    pub fn from_hex_rgba(hex: u32) -> Self {
        let r = ((hex >> 24) & 0xFF) as f32 / 255.0;
        let g = ((hex >> 16) & 0xFF) as f32 / 255.0;
        let b = ((hex >> 8) & 0xFF) as f32 / 255.0;
        let a = (hex & 0xFF) as f32 / 255.0;
        Self::rgba(r, g, b, a)
    }

    /// Convert to an array [r, g, b, a].
    pub fn to_array(&self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }

    /// Multiply alpha (useful for blending).
    pub fn with_alpha(self, alpha: f32) -> Self {
        Self {
            a: self.a * alpha,
            ..self
        }
    }
}

impl Default for Color {
    fn default() -> Self {
        Self::WHITE
    }
}

impl From<[f32; 4]> for Color {
    fn from(arr: [f32; 4]) -> Self {
        Self::rgba(arr[0], arr[1], arr[2], arr[3])
    }
}

impl From<[f32; 3]> for Color {
    fn from(arr: [f32; 3]) -> Self {
        Self::rgb(arr[0], arr[1], arr[2])
    }
}

/// A rectangle defined by position and size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    /// Top-left position.
    pub pos: Vec2,
    /// Size (width, height).
    pub size: Vec2,
}

impl Rect {
    /// Create a new rectangle.
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            pos: Vec2::new(x, y),
            size: Vec2::new(width, height),
        }
    }

    /// Create a rectangle from position and size vectors.
    pub fn from_pos_size(pos: Vec2, size: Vec2) -> Self {
        Self { pos, size }
    }

    /// Create a rectangle from min and max points.
    pub fn from_min_max(min: Vec2, max: Vec2) -> Self {
        Self {
            pos: min,
            size: max - min,
        }
    }

    /// Get the left edge x coordinate.
    pub fn left(&self) -> f32 {
        self.pos.x
    }

    /// Get the right edge x coordinate.
    pub fn right(&self) -> f32 {
        self.pos.x + self.size.x
    }

    /// Get the top edge y coordinate.
    pub fn top(&self) -> f32 {
        self.pos.y
    }

    /// Get the bottom edge y coordinate.
    pub fn bottom(&self) -> f32 {
        self.pos.y + self.size.y
    }

    /// Get the center point.
    pub fn center(&self) -> Vec2 {
        self.pos + self.size * 0.5
    }

    /// Get width.
    pub fn width(&self) -> f32 {
        self.size.x
    }

    /// Get height.
    pub fn height(&self) -> f32 {
        self.size.y
    }

    /// Check if a point is inside the rectangle.
    pub fn contains(&self, point: Vec2) -> bool {
        point.x >= self.left()
            && point.x <= self.right()
            && point.y >= self.top()
            && point.y <= self.bottom()
    }

    /// Expand the rectangle by the given amount on all sides.
    pub fn expand(&self, amount: f32) -> Self {
        Self {
            pos: self.pos - Vec2::splat(amount),
            size: self.size + Vec2::splat(amount * 2.0),
        }
    }

    /// Shrink the rectangle by the given amount on all sides.
    pub fn shrink(&self, amount: f32) -> Self {
        self.expand(-amount)
    }
}

impl Default for Rect {
    fn default() -> Self {
        Self {
            pos: Vec2::ZERO,
            size: Vec2::ONE,
        }
    }
}

/// A rounded rectangle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoundedRect {
    /// The base rectangle.
    pub rect: Rect,
    /// Corner radius.
    pub radius: f32,
}

impl RoundedRect {
    /// Create a new rounded rectangle.
    pub fn new(rect: Rect, radius: f32) -> Self {
        Self { rect, radius }
    }

    /// Create from explicit values.
    pub fn from_values(x: f32, y: f32, width: f32, height: f32, radius: f32) -> Self {
        Self {
            rect: Rect::new(x, y, width, height),
            radius,
        }
    }
}

/// Vertex for UI rendering.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UiVertex {
    /// Position in screen space.
    pub position: [f32; 2],
    /// UV coordinates (for texture sampling).
    pub uv: [f32; 2],
    /// RGBA color.
    pub color: [f32; 4],
}

impl UiVertex {
    pub fn new(position: Vec2, uv: Vec2, color: Color) -> Self {
        Self {
            position: [position.x, position.y],
            uv: [uv.x, uv.y],
            color: color.to_array(),
        }
    }
}

