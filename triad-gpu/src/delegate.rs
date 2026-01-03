//! Render delegate trait and context types.
//!
//! This module provides the `RenderDelegate` trait for implementing different
//! rendering strategies (points, gaussians, triangles) and related types.

use crate::{CameraUniforms, Renderer, ResourceRegistry};
use glam::Vec3;
use std::error::Error;

/// Scene bounds computed from primitive positions.
#[derive(Debug, Clone)]
pub struct SceneBounds {
    pub min: Vec3,
    pub max: Vec3,
    pub center: Vec3,
    pub radius: f32,
}

impl SceneBounds {
    /// Compute bounds from an iterator of positions.
    pub fn from_positions<'a>(positions: impl Iterator<Item = Vec3>) -> Self {
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        let mut count = 0;

        for pos in positions {
            min = min.min(pos);
            max = max.max(pos);
            count += 1;
        }

        if count == 0 {
            return Self {
                min: Vec3::ZERO,
                max: Vec3::ZERO,
                center: Vec3::ZERO,
                radius: 1.0,
            };
        }

        let center = (min + max) * 0.5;
        let radius = (max - min).length().max(1.0);
        Self {
            min,
            max,
            center,
            radius,
        }
    }

    /// Create bounds from explicit min/max.
    pub fn from_min_max(min: Vec3, max: Vec3) -> Self {
        let center = (min + max) * 0.5;
        let radius = (max - min).length().max(1.0);
        Self {
            min,
            max,
            center,
            radius,
        }
    }
}

impl Default for SceneBounds {
    fn default() -> Self {
        Self {
            min: Vec3::ZERO,
            max: Vec3::ONE,
            center: Vec3::splat(0.5),
            radius: 1.0,
        }
    }
}

/// Context passed to the render delegate for rendering.
pub struct RenderContext<'a> {
    pub color_view: &'a wgpu::TextureView,
    pub depth_view: Option<&'a wgpu::TextureView>,
}

/// Trait for shader-agnostic rendering. Implement this to render different primitive types.
pub trait RenderDelegate: Sized {
    /// Data needed to construct the delegate (e.g., loaded primitives, file path).
    type InitData;

    /// Create GPU resources for rendering.
    fn create(
        renderer: &Renderer,
        registry: &mut ResourceRegistry,
        surface_format: wgpu::TextureFormat,
        init_data: Self::InitData,
    ) -> Result<Self, Box<dyn Error>>;

    /// Get the scene bounds for camera positioning.
    fn bounds(&self) -> &SceneBounds;

    /// Return depth format if depth testing is needed. Default is None (no depth).
    fn depth_format(&self) -> Option<wgpu::TextureFormat> {
        None
    }

    /// Update GPU resources (e.g., camera uniforms).
    fn update(
        &mut self,
        queue: &wgpu::Queue,
        registry: &ResourceRegistry,
        camera: &CameraUniforms,
    );

    /// Record render commands.
    fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: RenderContext,
        registry: &ResourceRegistry,
    );

    /// Check if there's a pending PLY reload and handle it.
    /// Returns true if a reload was performed.
    /// Default implementation does nothing.
    fn handle_pending_ply_reload(
        &mut self,
        _renderer: &Renderer,
        _registry: &mut ResourceRegistry,
    ) -> Result<bool, Box<dyn Error>> {
        Ok(false)
    }
}

