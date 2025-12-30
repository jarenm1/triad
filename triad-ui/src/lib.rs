//! Triad UI - Lightweight UI primitives for GPU rendering
//!
//! This crate provides text rendering via cosmic-text and basic UI primitives
//! for building debug overlays and simple interfaces.
//!
//! ## Features
//!
//! - Text rendering with [`TextRenderer`] using cosmic-text for shaping
//! - Glyph atlas management for efficient texture caching
//! - Basic shape primitives (rect, rounded rect, line)
//! - Simple widget building blocks

mod atlas;
mod primitives;
mod text;
pub mod widgets;

pub use atlas::GlyphAtlas;
pub use primitives::{Color, Rect, RoundedRect};
pub use text::{TextRenderer, TextSection, TextStyle};

