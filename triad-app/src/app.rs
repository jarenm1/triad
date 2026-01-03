//! Application state and main run loop with builder pattern.

use crate::layers::LayerMode;
use crate::multi_delegate::{MultiDelegate, MultiInitData};
use glam::Vec3;
use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use triad_gpu::SceneBounds;
use triad_window::{
    CameraControl, Controls, FrameUpdate, GaussianDelegate, GaussianInitData, InputState, KeyCode,
    MouseButton, PhysicalKey, PointDelegate, PointInitData, TriangleDelegate, TriangleInitData,
    egui, run_with_delegate_config,
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

/// Configuration for MultiDelegate.
pub struct MultiDelegateConfig {
    pub initial_mode: LayerMode,
    pub point_size: f32,
}

impl Default for MultiDelegateConfig {
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

/// Delegate configuration enum.
pub enum DelegateConfig {
    Point(PointInitData),
    Gaussian(GaussianInitData),
    Triangle(TriangleInitData),
    Multi(MultiDelegateConfig),
}

/// Builder for configuring and running the application.
pub struct AppBuilder {
    title: String,
    window_size: (u32, u32),
    delegate_config: Option<DelegateConfig>,
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
            delegate_config: None,
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

    /// Configure with Point delegate.
    pub fn with_point_delegate(mut self, init_data: PointInitData) -> Self {
        self.delegate_config = Some(DelegateConfig::Point(init_data));
        self
    }

    /// Configure with Gaussian delegate.
    pub fn with_gaussian_delegate(mut self, init_data: GaussianInitData) -> Self {
        self.delegate_config = Some(DelegateConfig::Gaussian(init_data));
        self
    }

    /// Configure with Triangle delegate.
    pub fn with_triangle_delegate(mut self, init_data: TriangleInitData) -> Self {
        self.delegate_config = Some(DelegateConfig::Triangle(init_data));
        self
    }

    /// Configure with MultiDelegate.
    pub fn with_multi_delegate(mut self, config: MultiDelegateConfig) -> Self {
        self.delegate_config = Some(DelegateConfig::Multi(config));
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

        // Determine delegate and run
        match self.delegate_config {
            Some(DelegateConfig::Multi(config)) => {
                let mode_signal: ModeSignal = self.mode_signal.unwrap_or_else(|| {
                    Arc::new(AtomicU8::new(match config.initial_mode {
                        LayerMode::Points => 0,
                        LayerMode::Gaussians => 1,
                        LayerMode::Triangles => 2,
                    }))
                });

                let init_data = MultiInitData {
                    ply_path: self.ply_path.clone(),
                    initial_mode: config.initial_mode,
                    point_size: config.point_size,
                    mode_signal: mode_signal.clone(),
                    ply_receiver: self.ply_receiver.take(),
                };

                let mut controls_config = self.controls_config;
                let mut menu_config = self.menu_config;

                run_with_delegate_config::<MultiDelegate, _>(
                    &self.title,
                    init_data,
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
                )
            }
            Some(DelegateConfig::Point(init_data)) => {
                let mut controls_config = self.controls_config;
                let mut menu_config = self.menu_config;

                run_with_delegate_config::<PointDelegate, _>(
                    &self.title,
                    init_data,
                    move |controls| {
                        if let Some(config_fn) = controls_config.take() {
                            config_fn(controls);
                        }
                        if let Some(mut menu_fn) = menu_config.take() {
                            controls.on_ui(move |ctx| {
                                menu_fn(ctx);
                            });
                        }
                    },
                )
            }
            Some(DelegateConfig::Gaussian(init_data)) => {
                let mut controls_config = self.controls_config;
                let mut menu_config = self.menu_config;

                run_with_delegate_config::<GaussianDelegate, _>(
                    &self.title,
                    init_data,
                    move |controls| {
                        if let Some(config_fn) = controls_config.take() {
                            config_fn(controls);
                        }
                        if let Some(mut menu_fn) = menu_config.take() {
                            controls.on_ui(move |ctx| {
                                menu_fn(ctx);
                            });
                        }
                    },
                )
            }
            Some(DelegateConfig::Triangle(init_data)) => {
                let mut controls_config = self.controls_config;
                let mut menu_config = self.menu_config;

                run_with_delegate_config::<TriangleDelegate, _>(
                    &self.title,
                    init_data,
                    move |controls| {
                        if let Some(config_fn) = controls_config.take() {
                            config_fn(controls);
                        }
                        if let Some(mut menu_fn) = menu_config.take() {
                            controls.on_ui(move |ctx| {
                                menu_fn(ctx);
                            });
                        }
                    },
                )
            }
            None => Err("No delegate configured. Use one of: with_point_delegate, with_gaussian_delegate, with_triangle_delegate, or with_multi_delegate".into()),
        }
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
