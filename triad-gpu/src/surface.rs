use wgpu::{Surface, SurfaceConfiguration, SurfaceTexture, TextureFormat};

/// Wrapper around wgpu::Surface with configuration management
pub struct SurfaceWrapper {
    pub surface: Surface<'static>,
    pub config: SurfaceConfiguration,
}

impl SurfaceWrapper {
    pub fn new(surface: Surface<'static>, config: SurfaceConfiguration) -> Self {
        Self { surface, config }
    }

    pub fn format(&self) -> TextureFormat {
        self.config.format
    }

    pub fn width(&self) -> u32 {
        self.config.width
    }

    pub fn height(&self) -> u32 {
        self.config.height
    }

    pub fn reconfigure(&mut self, device: &wgpu::Device, config: SurfaceConfiguration) {
        self.config = config;
        self.surface.configure(device, &self.config);
    }

    /// Get the current surface texture for rendering
    /// Returns None if the surface is lost or needs to be recreated
    pub fn get_current_texture(&self) -> Result<SurfaceTexture, wgpu::SurfaceError> {
        self.surface.get_current_texture()
    }
}
