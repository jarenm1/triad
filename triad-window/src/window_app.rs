use crate::camera::{Camera, Projection};
use crate::camera_uniforms::CameraUniforms;
use crate::controls::Controls;
use glam::Vec3;
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tracing::{debug_span, error, info, instrument};
use triad_gpu::wgpu;
use triad_gpu::{
    ExecutableFrameGraph, FrameGraphError, Renderer, ResourceRegistry, SurfaceWrapper,
};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{EventLoop, EventLoopProxy};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowAttributes, WindowId};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::{EventLoopExtWebSys, WindowAttributesExtWebSys};

pub use egui;

const FPS_HISTORY_SIZE: usize = 60;

type CreateManager = Box<
    dyn FnOnce(
            &Renderer,
            &mut ResourceRegistry,
            wgpu::TextureFormat,
            u32,
            u32,
        ) -> Result<Box<dyn RendererManager>, Box<dyn Error>>
        + Send,
>;

type InitResultSlot = Arc<Mutex<Option<Result<ViewerState, String>>>>;

#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    #[error("Surface error: {0}")]
    Surface(#[from] triad_gpu::wgpu::SurfaceError),
    #[error("Frame graph error: {0}")]
    FrameGraph(#[from] FrameGraphError),
    #[error("Renderer manager error: {0}")]
    RendererManager(String),
}

#[derive(Debug, Clone, Copy)]
pub struct WindowConfig {
    pub present_mode: wgpu::PresentMode,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            present_mode: wgpu::PresentMode::AutoVsync,
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[derive(Debug, Clone)]
pub struct WebWindowConfig {
    pub canvas_id: String,
}

#[cfg(target_arch = "wasm32")]
impl WebWindowConfig {
    pub fn new(canvas_id: impl Into<String>) -> Self {
        Self {
            canvas_id: canvas_id.into(),
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[derive(Debug, Clone, Copy)]
enum AppUserEvent {
    InitializationComplete,
}

#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
enum StartupMode {
    Native,
    #[cfg(target_arch = "wasm32")]
    Web(WebWindowConfig),
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run_with_renderer_config<F, M>(
    title: &str,
    config: WindowConfig,
    configure_controls: F,
    create_manager: M,
) -> Result<(), Box<dyn Error>>
where
    F: FnOnce(&mut Controls),
    M: FnOnce(
            &Renderer,
            &mut ResourceRegistry,
            triad_gpu::wgpu::TextureFormat,
            u32,
            u32,
        ) -> Result<Box<dyn RendererManager>, Box<dyn Error>>
        + Send
        + 'static,
{
    info!(title, ?config.present_mode, "creating event loop");
    let event_loop = EventLoop::<AppUserEvent>::with_user_event()
        .build()
        .map_err(|e| format!("Failed to create event loop: {e}"))?;
    let proxy = event_loop.create_proxy();
    let mut controls = Controls::default();
    configure_controls(&mut controls);

    let mut app = App::new(
        title.to_string(),
        config,
        controls,
        create_manager,
        StartupMode::Native,
        proxy,
    );
    info!("starting winit app loop");
    let run_result = event_loop.run_app(&mut app);
    info!("winit app loop returned");
    let app_result = app.finish();
    run_result?;
    app_result
}

#[cfg(target_arch = "wasm32")]
pub fn run_with_renderer_config<F, M>(
    _title: &str,
    _config: WindowConfig,
    _configure_controls: F,
    _create_manager: M,
) -> Result<(), Box<dyn Error>>
where
    F: FnOnce(&mut Controls),
    M: FnOnce(
            &Renderer,
            &mut ResourceRegistry,
            triad_gpu::wgpu::TextureFormat,
            u32,
            u32,
        ) -> Result<Box<dyn RendererManager>, Box<dyn Error>>
        + Send
        + 'static,
{
    Err("use run_with_renderer_config_web on wasm32".into())
}

#[cfg(target_arch = "wasm32")]
pub fn run_with_renderer_config_web<F, M>(
    title: &str,
    config: WindowConfig,
    web_config: WebWindowConfig,
    configure_controls: F,
    create_manager: M,
) -> Result<(), Box<dyn Error>>
where
    F: FnOnce(&mut Controls),
    M: FnOnce(
            &Renderer,
            &mut ResourceRegistry,
            triad_gpu::wgpu::TextureFormat,
            u32,
            u32,
        ) -> Result<Box<dyn RendererManager>, Box<dyn Error>>
        + Send
        + 'static,
{
    info!(
        title,
        ?config.present_mode,
        canvas_id = web_config.canvas_id,
        "creating web event loop"
    );
    let event_loop = EventLoop::<AppUserEvent>::with_user_event()
        .build()
        .map_err(|e| format!("Failed to create event loop: {e}"))?;
    let proxy = event_loop.create_proxy();
    let mut controls = Controls::default();
    configure_controls(&mut controls);

    let app = App::new(
        title.to_string(),
        config,
        controls,
        create_manager,
        StartupMode::Web(web_config),
        proxy,
    );
    info!("starting winit web app loop");
    event_loop.spawn_app(app);
    Ok(())
}

struct App {
    title: String,
    config: Option<WindowConfig>,
    controls: Option<Controls>,
    create_manager: Option<CreateManager>,
    state: AppState,
    startup_mode: Option<StartupMode>,
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    event_loop_proxy: EventLoopProxy<AppUserEvent>,
    init_result: Option<InitResultSlot>,
    error: Option<String>,
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
enum AppState {
    Uninitialized,
    Initializing(Arc<Window>),
    Ready(ViewerState),
}

impl App {
    fn new<M>(
        title: String,
        config: WindowConfig,
        controls: Controls,
        create_manager: M,
        startup_mode: StartupMode,
        event_loop_proxy: EventLoopProxy<AppUserEvent>,
    ) -> Self
    where
        M: FnOnce(
                &Renderer,
                &mut ResourceRegistry,
                wgpu::TextureFormat,
                u32,
                u32,
            ) -> Result<Box<dyn RendererManager>, Box<dyn Error>>
            + Send
            + 'static,
    {
        Self {
            title,
            config: Some(config),
            controls: Some(controls),
            create_manager: Some(Box::new(create_manager)),
            state: AppState::Uninitialized,
            startup_mode: Some(startup_mode),
            event_loop_proxy,
            init_result: None,
            error: None,
        }
    }

    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    fn finish(self) -> Result<(), Box<dyn Error>> {
        if let Some(err) = self.error {
            Err(err.into())
        } else {
            Ok(())
        }
    }

    fn initialize(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
    ) -> Result<(), Box<dyn Error>> {
        let config = self.config.take().expect("config already consumed");
        let controls = self.controls.take().expect("controls already consumed");
        let create_manager = self
            .create_manager
            .take()
            .expect("create_manager already consumed");
        let startup_mode = self
            .startup_mode
            .take()
            .expect("startup mode already consumed");

        match startup_mode {
            StartupMode::Native => {
                let state = ViewerState::new_blocking(
                    event_loop,
                    &self.title,
                    config,
                    controls,
                    create_manager,
                )?;
                info!("viewer state initialized");
                self.state = AppState::Ready(state);
            }
            #[cfg(target_arch = "wasm32")]
            StartupMode::Web(web_config) => {
                let window = create_web_window(event_loop, &self.title, &web_config)?;
                let init_result = Arc::new(Mutex::new(None));
                let init_result_slot = Arc::clone(&init_result);
                let proxy = self.event_loop_proxy.clone();
                let window_for_init = Arc::clone(&window);

                info!(
                    canvas_id = web_config.canvas_id,
                    "spawning async web viewer initialization"
                );
                wasm_bindgen_futures::spawn_local(async move {
                    let result =
                        ViewerState::new_async(window_for_init, config, controls, create_manager)
                            .await
                            .map_err(|err| err.to_string());
                    if let Ok(mut slot) = init_result_slot.lock() {
                        *slot = Some(result);
                    }
                    let _ = proxy.send_event(AppUserEvent::InitializationComplete);
                });
                self.init_result = Some(init_result);
                self.state = AppState::Initializing(window);
            }
        }
        Ok(())
    }

    fn finish_web_initialization(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let Some(init_result) = self.init_result.as_ref() else {
            return;
        };
        let Some(result) = init_result.lock().ok().and_then(|mut slot| slot.take()) else {
            return;
        };
        match result {
            Ok(state) => {
                info!("viewer state initialized");
                self.state = AppState::Ready(state);
            }
            Err(err) => {
                error!("Failed to initialize viewer: {err}");
                self.error = Some(err);
                event_loop.exit();
            }
        }
    }
}

impl ApplicationHandler<AppUserEvent> for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if !matches!(self.state, AppState::Uninitialized) || self.error.is_some() {
            return;
        }
        info!("application resumed; initializing viewer state");

        if let Err(err) = self.initialize(event_loop) {
            error!("Failed to initialize viewer: {err}");
            self.error = Some(err.to_string());
            event_loop.exit();
        }
    }

    fn user_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _event: AppUserEvent,
    ) {
        self.finish_web_initialization(event_loop);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        match &mut self.state {
            AppState::Ready(state) => {
                if state.window.id() != window_id {
                    return;
                }

                let egui_consumed = state.handle_egui_event(&event);

                if !egui_consumed && state.handle_window_event(event_loop, &event) {
                    return;
                }

                match event {
                    WindowEvent::CloseRequested => event_loop.exit(),
                    WindowEvent::Resized(size) => state.resize(size),
                    WindowEvent::RedrawRequested => {
                        let _frame_span = tracing::info_span!("frame").entered();
                        match state.render() {
                            Ok(()) => {}
                            Err(
                                RenderError::Surface(triad_gpu::wgpu::SurfaceError::Lost)
                                | RenderError::Surface(triad_gpu::wgpu::SurfaceError::Outdated),
                            ) => {
                                let size = state.window.inner_size();
                                state.resize(size);
                            }
                            Err(RenderError::Surface(
                                triad_gpu::wgpu::SurfaceError::OutOfMemory,
                            )) => {
                                error!("GPU Out of Memory - exiting");
                                event_loop.exit();
                            }
                            Err(e) => error!("Render error: {e}"),
                        }
                    }
                    _ => {}
                }
            }
            AppState::Initializing(window) => {
                if window.id() != window_id {
                    return;
                }
                if matches!(event, WindowEvent::CloseRequested) {
                    event_loop.exit();
                }
            }
            AppState::Uninitialized => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let AppState::Ready(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

struct ViewerState {
    window: Arc<Window>,
    renderer: Renderer,
    surface: SurfaceWrapper,
    registry: ResourceRegistry,
    renderer_manager: Box<dyn RendererManager>,
    cached_frame_graph: Option<ExecutableFrameGraph>,
    camera: Camera,
    controls: Controls,
    projection: Projection,
    last_frame: Instant,
    depth_texture: Option<triad_gpu::wgpu::Texture>,
    depth_view: Option<Arc<triad_gpu::wgpu::TextureView>>,
    egui_ctx: egui::Context,
    egui_winit: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    fps_history: Vec<f32>,
    fps_history_index: usize,
    fps_display: f32,
    frame_graph_rebuilt_last_frame: bool,
    frame_graph_command_buffers_last_frame: usize,
    current_present_mode: wgpu::PresentMode,
    pending_present_mode: Option<wgpu::PresentMode>,
    pending_resize: Option<PhysicalSize<u32>>,
    show_ui: bool,
}

pub trait RendererManager: Send + Sync {
    fn update(
        &mut self,
        renderer: &Renderer,
        registry: &mut ResourceRegistry,
        camera: &CameraUniforms,
    ) -> Result<(), Box<dyn Error>>;

    fn prepare_frame(
        &mut self,
        registry: &mut ResourceRegistry,
        final_view: Arc<wgpu::TextureView>,
        depth_view: Option<Arc<wgpu::TextureView>>,
    ) -> Result<bool, Box<dyn Error>>;

    fn build_frame_graph(&mut self) -> Result<ExecutableFrameGraph, FrameGraphError>;

    fn resize(
        &mut self,
        device: &wgpu::Device,
        registry: &mut ResourceRegistry,
        width: u32,
        height: u32,
    ) -> Result<(), Box<dyn Error>>;
}

impl ViewerState {
    #[cfg(not(target_arch = "wasm32"))]
    fn new_blocking(
        event_loop: &winit::event_loop::ActiveEventLoop,
        title: &str,
        config: WindowConfig,
        controls: Controls,
        create_manager: CreateManager,
    ) -> Result<Self, Box<dyn Error>> {
        let window = create_native_window(event_loop, title)?;
        pollster::block_on(Self::new_async(window, config, controls, create_manager))
    }

    #[cfg(target_arch = "wasm32")]
    fn new_blocking(
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _title: &str,
        _config: WindowConfig,
        _controls: Controls,
        _create_manager: CreateManager,
    ) -> Result<Self, Box<dyn Error>> {
        Err("native viewer initialization is unavailable on wasm32".into())
    }

    async fn new_async(
        window: Arc<Window>,
        config: WindowConfig,
        controls: Controls,
        create_manager: CreateManager,
    ) -> Result<Self, Box<dyn Error>> {
        info!(window_id = ?window.id(), "window created");

        info!("requesting renderer");
        let renderer = Renderer::new().await?;
        info!("renderer created");
        let size = window.inner_size();
        info!(
            width = size.width,
            height = size.height,
            "window size acquired"
        );

        info!("creating render surface");
        let surface = renderer.instance().create_surface(window.clone())?;
        let surface = renderer.create_surface_with_mode(
            surface,
            size.width.max(1),
            size.height.max(1),
            config.present_mode,
        )?;
        info!(format = ?surface.format(), ?config.present_mode, "surface configured");

        let mut registry = ResourceRegistry::default();
        let camera = Camera::new(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO);
        let projection = Projection::new(
            size.width.max(1),
            size.height.max(1),
            std::f32::consts::FRAC_PI_3,
            0.01,
            10000.0,
        );

        let (depth_texture, depth_view) = Self::create_depth_texture(
            renderer.device(),
            size.width.max(1),
            size.height.max(1),
            triad_gpu::wgpu::TextureFormat::Depth32Float,
        );

        let egui_ctx = egui::Context::default();
        let egui_winit = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            renderer.device(),
            surface.format(),
            egui_wgpu::RendererOptions::default(),
        );

        info!("creating renderer manager");
        let renderer_manager = create_manager(
            &renderer,
            &mut registry,
            surface.format(),
            size.width.max(1),
            size.height.max(1),
        )?;
        info!("renderer manager created");

        Ok(Self {
            window,
            renderer,
            surface,
            registry,
            renderer_manager,
            cached_frame_graph: None,
            camera,
            controls,
            projection,
            last_frame: Instant::now(),
            depth_texture: Some(depth_texture),
            depth_view: Some(Arc::new(depth_view)),
            egui_ctx,
            egui_winit,
            egui_renderer,
            fps_history: vec![60.0; FPS_HISTORY_SIZE],
            fps_history_index: 0,
            fps_display: 60.0,
            frame_graph_rebuilt_last_frame: true,
            frame_graph_command_buffers_last_frame: 0,
            current_present_mode: config.present_mode,
            pending_present_mode: None,
            pending_resize: None,
            show_ui: true,
        })
    }

    fn create_depth_texture(
        device: &triad_gpu::wgpu::Device,
        width: u32,
        height: u32,
        format: triad_gpu::wgpu::TextureFormat,
    ) -> (triad_gpu::wgpu::Texture, triad_gpu::wgpu::TextureView) {
        let texture = device.create_texture(&triad_gpu::wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: triad_gpu::wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: triad_gpu::wgpu::TextureDimension::D2,
            format,
            usage: triad_gpu::wgpu::TextureUsages::RENDER_ATTACHMENT
                | triad_gpu::wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&triad_gpu::wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn handle_egui_event(&mut self, event: &WindowEvent) -> bool {
        let response = self.egui_winit.on_window_event(&self.window, event);
        response.consumed
    }

    #[cfg_attr(target_arch = "wasm32", allow(unused_variables))]
    fn handle_window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        event: &WindowEvent,
    ) -> bool {
        if let WindowEvent::KeyboardInput {
            event:
                KeyEvent {
                    state: ElementState::Pressed,
                    physical_key: PhysicalKey::Code(key_code),
                    ..
                },
            ..
        } = event
        {
            match key_code {
                #[cfg(not(target_arch = "wasm32"))]
                KeyCode::Escape => {
                    event_loop.exit();
                    return true;
                }
                KeyCode::F1 => {
                    self.show_ui = !self.show_ui;
                    tracing::info!("UI visibility: {}", self.show_ui);
                    return true;
                }
                _ => {}
            }
        }

        self.controls.handle_event(event)
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.pending_resize = Some(new_size);
    }

    fn apply_resize(&mut self, new_size: PhysicalSize<u32>) {
        let mut config = self.surface.config().clone();
        config.width = new_size.width;
        config.height = new_size.height;
        self.surface.reconfigure(self.renderer.device(), config);
        self.projection.update_size(new_size.width, new_size.height);

        if let Err(e) = self.renderer_manager.resize(
            self.renderer.device(),
            &mut self.registry,
            new_size.width,
            new_size.height,
        ) {
            error!("Failed to resize renderer resources: {}", e);
        }
        self.cached_frame_graph = None;

        let (tex, view) = Self::create_depth_texture(
            self.renderer.device(),
            new_size.width,
            new_size.height,
            triad_gpu::wgpu::TextureFormat::Depth32Float,
        );
        self.depth_texture = Some(tex);
        self.depth_view = Some(Arc::new(view));
    }

    fn set_present_mode(&mut self, present_mode: wgpu::PresentMode) {
        if self.current_present_mode == present_mode {
            return;
        }
        tracing::debug!(
            "Requesting present mode change to {:?} (will apply next frame)",
            present_mode
        );
        self.pending_present_mode = Some(present_mode);
    }

    fn apply_present_mode(&mut self, present_mode: wgpu::PresentMode) {
        tracing::info!(
            "Changing present mode from {:?} to {:?}",
            self.current_present_mode,
            present_mode
        );
        self.current_present_mode = present_mode;
        let mut config = self.surface.config().clone();
        config.present_mode = present_mode;
        self.surface.reconfigure(self.renderer.device(), config);

        let actual_mode = self.surface.config().present_mode;
        tracing::info!("Surface reconfigured - actual present mode: {:?}", actual_mode);
        if actual_mode != present_mode {
            tracing::warn!(
                "Present mode mismatch! Requested {:?} but got {:?}",
                present_mode,
                actual_mode
            );
        }
    }

    #[instrument(skip(self), name = "render")]
    fn render(&mut self) -> Result<(), RenderError> {
        if let Some(present_mode) = self.pending_present_mode.take() {
            self.apply_present_mode(present_mode);
        }
        if let Some(new_size) = self.pending_resize.take() {
            self.apply_resize(new_size);
        }

        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        let current_fps = if dt > 0.0 { 1.0 / dt } else { 0.0 };
        self.fps_history[self.fps_history_index] = current_fps;
        self.fps_history_index = (self.fps_history_index + 1) % FPS_HISTORY_SIZE;
        self.fps_display = self.fps_history.iter().sum::<f32>() / FPS_HISTORY_SIZE as f32;

        {
            let _span = debug_span!("camera_update").entered();
            self.controls.update(dt, &mut self.camera);

            let view = self.camera.view_matrix();
            let proj = self.projection.matrix();
            let uniforms = CameraUniforms::from_matrices(view, proj, self.camera.position());

            self.renderer_manager
                .update(&self.renderer, &mut self.registry, &uniforms)
                .map_err(|e| RenderError::RendererManager(e.to_string()))?;
        }

        let (surface_texture, surface_view) = {
            let _span = debug_span!("surface_acquire").entered();
            let surface_texture = self.surface.get_current_texture()?;
            let surface_view = Arc::new(
                surface_texture
                    .texture
                    .create_view(&triad_gpu::wgpu::TextureViewDescriptor::default()),
            );
            (surface_texture, surface_view)
        };

        let needs_rebuild = self
            .renderer_manager
            .prepare_frame(
                &mut self.registry,
                surface_view.clone(),
                self.depth_view.clone(),
            )
            .map_err(|e| RenderError::RendererManager(e.to_string()))?;

        let rebuilt_frame_graph = needs_rebuild || self.cached_frame_graph.is_none();
        if rebuilt_frame_graph {
            let frame_graph = {
                let _span = debug_span!("frame_graph_build").entered();
                self.renderer_manager.build_frame_graph()?
            };
            self.cached_frame_graph = Some(frame_graph);
        }

        let frame_graph = self
            .cached_frame_graph
            .as_mut()
            .expect("cached frame graph should be available");

        let mut command_buffers = {
            let _span = debug_span!("frame_graph_execute").entered();
            frame_graph.execute_no_submit(
                self.renderer.device(),
                self.renderer.queue(),
                &self.registry,
            )
        };
        self.frame_graph_rebuilt_last_frame = rebuilt_frame_graph;
        self.frame_graph_command_buffers_last_frame = command_buffers.len();

        let (full_output, new_present_mode) = if self.show_ui {
            let _span = debug_span!("egui_run").entered();
            let raw_input = self.egui_winit.take_egui_input(&self.window);
            #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
            let mut new_mode = None;
            let output = self.egui_ctx.run(raw_input, |ctx| {
                ctx.request_repaint_after(std::time::Duration::from_millis(100));

                self.controls.run_ui(ctx);

                egui::Window::new("Performance")
                    .default_pos(egui::pos2(10.0, 10.0))
                    .resizable(false)
                    .collapsible(false)
                    .title_bar(false)
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(format!("FPS: {:.1}", self.fps_display))
                                    .size(16.0)
                                    .color(if self.fps_display >= 50.0 {
                                        egui::Color32::from_rgb(100, 255, 100)
                                    } else if self.fps_display >= 30.0 {
                                        egui::Color32::from_rgb(255, 255, 100)
                                    } else {
                                        egui::Color32::from_rgb(255, 100, 100)
                                    }),
                            );
                        });

                        ui.horizontal(|ui| {
                            let frame_time_ms = 1000.0 / self.fps_display.max(0.01);
                            ui.label(
                                egui::RichText::new(format!("{:.2}ms", frame_time_ms))
                                    .size(12.0)
                                    .color(egui::Color32::GRAY),
                            );
                        });

                        ui.separator();

                        ui.horizontal(|ui| {
                            ui.label("Graph:");
                            ui.label(if self.frame_graph_rebuilt_last_frame {
                                "rebuilt"
                            } else {
                                "cached"
                            });
                        });

                        ui.horizontal(|ui| {
                            ui.label("Cmd buffers:");
                            ui.label(self.frame_graph_command_buffers_last_frame.to_string());
                        });

                        #[cfg(not(target_arch = "wasm32"))]
                        ui.horizontal(|ui| {
                            ui.label("Present:");

                            if ui
                                .selectable_label(
                                    matches!(
                                        self.current_present_mode,
                                        wgpu::PresentMode::AutoVsync
                                    ),
                                    "Auto vsync",
                                )
                                .clicked()
                            {
                                new_mode = Some(wgpu::PresentMode::AutoVsync);
                            }

                            if ui
                                .selectable_label(
                                    matches!(self.current_present_mode, wgpu::PresentMode::Fifo),
                                    "FIFO",
                                )
                                .clicked()
                            {
                                new_mode = Some(wgpu::PresentMode::Fifo);
                            }

                            if ui
                                .selectable_label(
                                    matches!(self.current_present_mode, wgpu::PresentMode::Mailbox),
                                    "Mailbox",
                                )
                                .clicked()
                            {
                                new_mode = Some(wgpu::PresentMode::Mailbox);
                            }

                            if ui
                                .selectable_label(
                                    matches!(
                                        self.current_present_mode,
                                        wgpu::PresentMode::AutoNoVsync
                                    ),
                                    "Auto no vsync",
                                )
                                .clicked()
                            {
                                new_mode = Some(wgpu::PresentMode::AutoNoVsync);
                            }
                        });
                    });
            });
            (output, Some(new_mode))
        } else {
            let raw_input = self.egui_winit.take_egui_input(&self.window);
            let output = self.egui_ctx.run(raw_input, |_ctx| {});
            (output, None)
        };

        let new_present_mode = new_present_mode.flatten();

        if let Some(mode) = new_present_mode {
            self.set_present_mode(mode);
        }

        self.egui_winit
            .handle_platform_output(&self.window, full_output.platform_output);

        let (screen_descriptor, tris) = {
            let _span = debug_span!("egui_tessellate").entered();
            let screen_descriptor = egui_wgpu::ScreenDescriptor {
                size_in_pixels: [self.surface.config().width, self.surface.config().height],
                pixels_per_point: self.window.scale_factor() as f32,
            };

            let tris = self
                .egui_ctx
                .tessellate(full_output.shapes, full_output.pixels_per_point);

            (screen_descriptor, tris)
        };

        {
            let _span = debug_span!("egui_render").entered();

            {
                let _span = debug_span!("egui_texture_update").entered();
                for (id, image_delta) in &full_output.textures_delta.set {
                    self.egui_renderer.update_texture(
                        self.renderer.device(),
                        self.renderer.queue(),
                        *id,
                        image_delta,
                    );
                }
            }

            let mut egui_encoder = {
                let _span = debug_span!("egui_encoder_create").entered();
                self.renderer.device().create_command_encoder(
                    &triad_gpu::wgpu::CommandEncoderDescriptor {
                        label: Some("egui Encoder"),
                    },
                )
            };

            {
                let _span = debug_span!("egui_update_buffers").entered();
                self.egui_renderer.update_buffers(
                    self.renderer.device(),
                    self.renderer.queue(),
                    &mut egui_encoder,
                    &tris,
                    &screen_descriptor,
                );
            }

            {
                let _span = debug_span!("egui_render_pass").entered();
                render_egui(
                    &self.egui_renderer,
                    &mut egui_encoder,
                    &surface_view,
                    &tris,
                    &screen_descriptor,
                );
            }

            {
                let _span = debug_span!("egui_texture_free").entered();
                for id in &full_output.textures_delta.free {
                    self.egui_renderer.free_texture(id);
                }
            }

            command_buffers.push(egui_encoder.finish());

            {
                let _span = debug_span!("queue_submit_all", count = command_buffers.len()).entered();
                self.renderer.queue().submit(command_buffers);
            }

            {
                let _span = debug_span!("surface_present").entered();
                surface_texture.present();
            }
        }

        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn create_native_window(
    event_loop: &winit::event_loop::ActiveEventLoop,
    title: &str,
) -> Result<Arc<Window>, Box<dyn Error>> {
    info!(title, "creating native window");
    let window_attributes = default_window_attributes(title);
    Ok(Arc::new(event_loop.create_window(window_attributes)?))
}

#[cfg(target_arch = "wasm32")]
fn create_web_window(
    event_loop: &winit::event_loop::ActiveEventLoop,
    title: &str,
    web_config: &WebWindowConfig,
) -> Result<Arc<Window>, Box<dyn Error>> {
    let browser_window =
        web_sys::window().ok_or_else(|| "browser window is unavailable".to_string())?;
    let document = browser_window
        .document()
        .ok_or_else(|| "browser document is unavailable".to_string())?;
    let element = document.get_element_by_id(&web_config.canvas_id).ok_or_else(|| {
        format!("canvas element with id '{}' was not found", web_config.canvas_id)
    })?;
    let canvas = element
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .map_err(|_| format!("element '{}' is not a canvas", web_config.canvas_id))?;
    let width = canvas.width().max(1280);
    let height = canvas.height().max(720);
    let window_attributes = default_window_attributes(title)
        .with_canvas(Some(canvas))
        .with_inner_size(PhysicalSize::new(width, height));
    Ok(Arc::new(event_loop.create_window(window_attributes)?))
}

fn default_window_attributes(title: &str) -> WindowAttributes {
    Window::default_attributes()
        .with_title(title)
        .with_inner_size(PhysicalSize::new(1280, 720))
}

fn render_egui(
    renderer: &egui_wgpu::Renderer,
    encoder: &mut triad_gpu::wgpu::CommandEncoder,
    view: &triad_gpu::wgpu::TextureView,
    paint_jobs: &[egui::ClippedPrimitive],
    screen_descriptor: &egui_wgpu::ScreenDescriptor,
) {
    let render_pass = encoder.begin_render_pass(&triad_gpu::wgpu::RenderPassDescriptor {
        label: Some("egui Render Pass"),
        color_attachments: &[Some(triad_gpu::wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: triad_gpu::wgpu::Operations {
                load: triad_gpu::wgpu::LoadOp::Load,
                store: triad_gpu::wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    let mut render_pass = render_pass.forget_lifetime();

    renderer.render(&mut render_pass, paint_jobs, screen_descriptor);
}
