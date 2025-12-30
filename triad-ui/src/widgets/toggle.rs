//! Toggle switch widget.

use crate::{Color, Rect};
use glam::Vec2;

/// A toggle switch widget.
pub struct Toggle {
    /// Current state.
    pub checked: bool,
    /// Position.
    pub position: Vec2,
    /// Size.
    pub size: Vec2,
    /// Label text.
    pub label: Option<String>,
    /// Color when checked.
    pub active_color: Color,
    /// Color when unchecked.
    pub inactive_color: Color,
}

impl Toggle {
    /// Create a new toggle.
    pub fn new(checked: bool) -> Self {
        Self {
            checked,
            position: Vec2::ZERO,
            size: Vec2::new(40.0, 20.0),
            label: None,
            active_color: Color::from_hex(0x4CAF50),
            inactive_color: Color::from_hex(0x757575),
        }
    }

    /// Set the position.
    pub fn at(mut self, x: f32, y: f32) -> Self {
        self.position = Vec2::new(x, y);
        self
    }

    /// Set a label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Get the bounding rect.
    pub fn bounds(&self) -> Rect {
        Rect::from_pos_size(self.position, self.size)
    }

    /// Check if a point is inside the toggle.
    pub fn contains(&self, point: Vec2) -> bool {
        self.bounds().contains(point)
    }

    /// Toggle the state.
    pub fn toggle(&mut self) {
        self.checked = !self.checked;
    }

    /// Get the current color based on state.
    pub fn current_color(&self) -> Color {
        if self.checked {
            self.active_color
        } else {
            self.inactive_color
        }
    }
}

impl Default for Toggle {
    fn default() -> Self {
        Self::new(false)
    }
}

