//! Layer management for visualization modes.

use std::fmt;

/// Available visualization modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerMode {
    /// Render as point cloud.
    Points,
    /// Render as Gaussian splats.
    Gaussians,
    /// Render as triangulated mesh.
    Triangles,
}

impl fmt::Display for LayerMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayerMode::Points => write!(f, "Points"),
            LayerMode::Gaussians => write!(f, "Gaussians"),
            LayerMode::Triangles => write!(f, "Triangles"),
        }
    }
}

impl LayerMode {
    /// Get all available modes.
    pub fn all() -> &'static [LayerMode] {
        &[LayerMode::Points, LayerMode::Gaussians, LayerMode::Triangles]
    }

    /// Cycle to the next mode.
    pub fn next(self) -> Self {
        match self {
            LayerMode::Points => LayerMode::Gaussians,
            LayerMode::Gaussians => LayerMode::Triangles,
            LayerMode::Triangles => LayerMode::Points,
        }
    }

    /// Cycle to the previous mode.
    pub fn prev(self) -> Self {
        match self {
            LayerMode::Points => LayerMode::Triangles,
            LayerMode::Gaussians => LayerMode::Points,
            LayerMode::Triangles => LayerMode::Gaussians,
        }
    }
}

/// Manages multiple visualization layers.
///
/// In the future, this will support hot-swapping between render delegates
/// at runtime. For now, it tracks the current mode.
pub struct LayerManager {
    current_mode: LayerMode,
    enabled: [bool; 3],
}

impl LayerManager {
    /// Create a new layer manager.
    pub fn new(initial_mode: LayerMode) -> Self {
        let mut enabled = [false; 3];
        enabled[initial_mode as usize] = true;

        Self {
            current_mode: initial_mode,
            enabled,
        }
    }

    /// Get the current active mode.
    pub fn current_mode(&self) -> LayerMode {
        self.current_mode
    }

    /// Set the current mode.
    pub fn set_mode(&mut self, mode: LayerMode) {
        // Disable current
        self.enabled[self.current_mode as usize] = false;
        // Enable new
        self.current_mode = mode;
        self.enabled[mode as usize] = true;
    }

    /// Check if a mode is enabled.
    pub fn is_enabled(&self, mode: LayerMode) -> bool {
        self.enabled[mode as usize]
    }

    /// Toggle a specific layer.
    pub fn toggle(&mut self, mode: LayerMode) {
        self.enabled[mode as usize] = !self.enabled[mode as usize];
        
        // If we disabled the current mode, switch to the first enabled one
        if !self.is_enabled(self.current_mode) {
            for mode in LayerMode::all() {
                if self.is_enabled(*mode) {
                    self.current_mode = *mode;
                    break;
                }
            }
        }
    }

    /// Cycle to the next mode.
    pub fn next_mode(&mut self) {
        self.set_mode(self.current_mode.next());
    }

    /// Cycle to the previous mode.
    pub fn prev_mode(&mut self) {
        self.set_mode(self.current_mode.prev());
    }
}

impl Default for LayerManager {
    fn default() -> Self {
        Self::new(LayerMode::Points)
    }
}

