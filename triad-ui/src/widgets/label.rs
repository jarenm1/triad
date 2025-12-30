//! Text label widget.

use crate::{Color, TextSection, TextStyle};
use glam::Vec2;

/// A simple text label widget.
pub struct Label {
    text: String,
    position: Vec2,
    style: TextStyle,
}

impl Label {
    /// Create a new label.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            position: Vec2::ZERO,
            style: TextStyle::default(),
        }
    }

    /// Set the position.
    pub fn at(mut self, x: f32, y: f32) -> Self {
        self.position = Vec2::new(x, y);
        self
    }

    /// Set the font size.
    pub fn size(mut self, size: f32) -> Self {
        self.style.font_size = size;
        self
    }

    /// Set the text color.
    pub fn color(mut self, color: Color) -> Self {
        self.style.color = color;
        self
    }

    /// Convert to a TextSection for rendering.
    pub fn to_section(&self) -> TextSection {
        TextSection {
            text: self.text.clone(),
            position: self.position,
            style: self.style.clone(),
            max_width: None,
        }
    }
}

