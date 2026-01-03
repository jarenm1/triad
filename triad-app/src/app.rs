//! Application state and main run loop with builder pattern.

use crate::layers::LayerMode;
use crate::renderer_manager::RendererInitData;
use glam::Vec3;
use std::error::Error;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use triad_window::{
    CameraControl, Controls, FrameUpdate, InputState, KeyCode, MouseButton, PhysicalKey, egui,
    run_with_renderer_config,
};

/// Thread-safe signal for mode switching.
pub type ModeSignal = Arc<AtomicU8>;

/// Read the current mode from the signal.
pub fn read_mode(signal: &ModeSignal) -> LayerMode {
    match signal.load(Ordering::Relaxed) {
        0 => LayerMode::Points,
        1 => LayerMode::Gaussians,
        2 => LayerMode::Triangles,
        _ => LayerMode::Points,
    }
}

/// Write a mode to the signal.
pub fn write_mode(signal: &ModeSignal, mode: LayerMode) {
    let value = match mode {
        LayerMode::Points => 0,
        LayerMode::Gaussians => 1,
        LayerMode::Triangles => 2,
    };
    signal.store(value, Ordering::Relaxed);
}

/// Configuration for RendererManager.
pub struct RendererConfig {
    pub initial_mode: LayerMode,
    pub point_size: f32,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            initial_mode: LayerMode::Points,
            point_size: 0.01,
        }
    }
}

/// Configuration for camera setup.
pub struct CameraConfig {
    pub initial_position: Option<Vec3>,
    pub initial_center: Option<Vec3>,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            initial_position: None,
            initial_center: None,
        }
    }
}

/// Logging configuration.
pub struct LoggingConfig {
    pub level: String,
    pub enable_tracy: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            enable_tracy: false,
        }
    }
}

// Removed DelegateConfig - now using RendererConfig directly

/// Builder for configuring and running the application.
pub struct AppBuilder {
    title: String,
    window_size: (u32, u32),
    renderer_config: Option<RendererConfig>,
    controls_config: Option<Box<dyn FnOnce(&mut Controls) + Send>>,
    camera_config: CameraConfig,
    logging: LoggingConfig,
    menu_config: Option<Box<dyn FnMut(&egui::Context) + Send>>,
    ply_path: Option<PathBuf>,
    mode_signal: Option<ModeSignal>,
    ply_receiver: Option<std::sync::mpsc::Receiver<PathBuf>>,
}

impl AppBuilder {
    /// Create a new AppBuilder with default settings.
    pub fn new() -> Self {
        Self {
            title: "Triad Viewer".to_string(),
            window_size: (1280, 720),
            renderer_config: None,
            controls_config: None,
            camera_config: CameraConfig::default(),
            logging: LoggingConfig::default(),
            menu_config: None,
            ply_path: None,
            mode_signal: None,
            ply_receiver: None,
        }
    }

    /// Set the PLY receiver for runtime loading.
    pub fn with_ply_receiver(mut self, receiver: std::sync::mpsc::Receiver<PathBuf>) -> Self {
        self.ply_receiver = Some(receiver);
        self
    }

    /// Set the mode signal (for menu integration).
    pub fn with_mode_signal(mut self, signal: ModeSignal) -> Self {
        self.mode_signal = Some(signal);
        self
    }

    /// Set the window title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Set the window size.
    pub fn with_window_size(mut self, width: u32, height: u32) -> Self {
        self.window_size = (width, height);
        self
    }

    /// Configure with renderer settings.
    pub fn with_renderer_config(mut self, config: RendererConfig) -> Self {
        self.renderer_config = Some(config);
        self
    }

    /// Configure with MultiDelegate (deprecated - use with_renderer_config).
    pub fn with_multi_delegate(mut self, config: RendererConfig) -> Self {
        self.renderer_config = Some(config);
        self
    }

    /// Set PLY file path for runtime loading.
    pub fn with_ply_path(mut self, path: PathBuf) -> Self {
        self.ply_path = Some(path);
        self
    }

    /// Configure camera settings.
    pub fn with_camera_config(mut self, config: CameraConfig) -> Self {
        self.camera_config = config;
        self
    }

    /// Configure controls with a closure.
    pub fn with_controls<F>(mut self, configure: F) -> Self
    where
        F: FnOnce(&mut Controls) + Send + 'static,
    {
        self.controls_config = Some(Box::new(configure));
        self
    }

    /// Configure camera setup with a closure (for scriptable camera).
    pub fn with_camera_setup<F>(self, _setup: F) -> Self
    where
        F: FnOnce(&mut crate::camera::CameraScriptConfig) + Send + 'static,
    {
        // This will be used by the camera scripting system
        self
    }

    /// Configure egui menu with a closure.
    pub fn with_menu<F>(mut self, menu: F) -> Self
    where
        F: FnMut(&egui::Context) + Send + 'static,
    {
        self.menu_config = Some(Box::new(menu));
        self
    }

    /// Configure logging.
    pub fn with_logging(mut self, config: LoggingConfig) -> Self {
        self.logging = config;
        self
    }

    /// Run the application.
    pub fn run(mut self) -> Result<(), Box<dyn Error>> {
        // Initialize logging
        self.init_logging();

        // Get renderer config or use default
        let config = self.renderer_config.unwrap_or_default();

        let init_data = RendererInitData {
            ply_path: self.ply_path,
            initial_mode: config.initial_mode,
            point_size: config.point_size,
            ply_receiver: self.ply_receiver.take().map(|r| {
                // Wrap the receiver in Arc<Mutex<>> for thread safety
                Arc::new(Mutex::new(r))
            }),
        };

        let mut controls_config = self.controls_config;
        let mut menu_config = self.menu_config;

        // Convert RendererInitData from triad-app to triad-window format
        let window_init_data = triad_window::RendererInitData {
            ply_path: init_data.ply_path,
            initial_mode: match init_data.initial_mode {
                LayerMode::Points => 0,
                LayerMode::Gaussians => 1,
                LayerMode::Triangles => 2,
            },
            point_size: init_data.point_size,
            ply_receiver: init_data.ply_receiver.and_then(|r| {
                // Unwrap from Arc<Mutex<>> to get the receiver
                Arc::try_unwrap(r).ok()
                    .and_then(|m| m.into_inner().ok())
            }),
        };

        run_with_renderer_config(
            &self.title,
            window_init_data,
            move |controls| {
                // Apply controls configuration
                if let Some(config_fn) = controls_config.take() {
                    config_fn(controls);
                }

                // Add menu UI hook
                if let Some(mut menu_fn) = menu_config.take() {
                    controls.on_ui(move |ctx| {
                        menu_fn(ctx);
                    });
                }
            },
            move |window_init_data,
                  renderer,
                  registry,
                  surface_format: triad_gpu::wgpu::TextureFormat,
                  width,
                  height| {
                // Convert back to triad-app format
                let app_init_data = crate::renderer_manager::RendererInitData {
                    ply_path: window_init_data.ply_path,
                    initial_mode: match window_init_data.initial_mode {
                        0 => LayerMode::Points,
                        1 => LayerMode::Gaussians,
                        2 => LayerMode::Triangles,
                        _ => LayerMode::Points,
                    },
                    point_size: window_init_data.point_size,
                    ply_receiver: window_init_data.ply_receiver.map(|r| Arc::new(Mutex::new(r))),
                };

                // Create the renderer manager
                let manager = crate::renderer_manager::RendererManager::create(
                    renderer,
                    registry,
                    surface_format,
                    width,
                    height,
                    app_init_data,
                )?;

                Ok(Box::new(manager))
            },
        )
    }

    fn init_logging(&self) {
        #[cfg(feature = "tracy")]
        {
            if self.logging.enable_tracy {
                use tracing_subscriber::Layer;
                use tracing_subscriber::layer::SubscriberExt;
                use tracing_subscriber::util::SubscriberInitExt;
                tracing_subscriber::registry()
                    .with(tracing_tracy::TracyLayer::default())
                    .with(
                        tracing_subscriber::fmt::layer().with_filter(
                            tracing_subscriber::EnvFilter::try_from_default_env()
                                .unwrap_or_else(|_| self.logging.level.clone().into()),
                        ),
                    )
                    .init();
                return;
            }
        }

        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&self.logging.level)),
            )
            .with_target(false)
            .init();
    }
}

impl Default for AppBuilder {
    fn default() -> Self {
        Self::new()
    }
}
