use crate::camera::{Camera, CameraController, Projection};
use glam::Vec3;
use std::error::Error;
use std::sync::Arc;
use std::time::Instant;
use tracing::{error, info};
use triad_gpu::{CameraUniforms, Renderer, ResourceRegistry, SurfaceWrapper};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// Scene bounds computed from primitive positions.
#[derive(Debug, Clone)]
pub struct SceneBounds {
    pub min: Vec3,
    pub max: Vec3,
    pub center: Vec3,
    pub radius: f32,
}

impl SceneBounds {
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
}

/// Context passed to the render delegate for rendering.
pub struct RenderContext<'a> {
    pub color_view: &'a triad_gpu::wgpu::TextureView,
    pub depth_view: Option<&'a triad_gpu::wgpu::TextureView>,
}

/// Trait for shader-agnostic rendering. Implement this to render different primitive types.
pub trait RenderDelegate: Sized {
    /// Data needed to construct the delegate (e.g., loaded primitives, file path).
    type InitData;

    /// Create GPU resources for rendering.
    fn create(
        renderer: &Renderer,
        registry: &mut ResourceRegistry,
        surface_format: triad_gpu::wgpu::TextureFormat,
        init_data: Self::InitData,
    ) -> Result<Self, Box<dyn Error>>;

    /// Get the scene bounds for camera positioning.
    fn bounds(&self) -> &SceneBounds;

    /// Return depth format if depth testing is needed. Default is None (no depth).
    fn depth_format(&self) -> Option<triad_gpu::wgpu::TextureFormat> {
        None
    }

    /// Update GPU resources (e.g., camera uniforms).
    fn update(
        &mut self,
        queue: &triad_gpu::wgpu::Queue,
        registry: &ResourceRegistry,
        camera: &CameraUniforms,
    );

    /// Record render commands.
    fn render(
        &self,
        encoder: &mut triad_gpu::wgpu::CommandEncoder,
        ctx: RenderContext,
        registry: &ResourceRegistry,
    );
}

/// Run the viewer with a custom render delegate.
pub fn run_with_delegate<D: RenderDelegate + 'static>(
    title: &str,
    init_data: D::InitData,
) -> Result<(), Box<dyn Error>>
where
    D::InitData: 'static,
{
    #[cfg(feature = "tracy")]
    {
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;
        use tracing_subscriber::Layer;
        tracing_subscriber::registry()
            .with(tracing_tracy::TracyLayer::default())
            .with(
                tracing_subscriber::fmt::layer().with_filter(
                    tracing_subscriber::EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| "info".into()),
                )
            )
            .init();
    }

    #[cfg(not(feature = "tracy"))]
    {
        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
            )
            .with_target(false)
            .init();
    }

    let event_loop = EventLoop::new().map_err(|e| format!("Failed to create event loop: {e}"))?;
    let mut app = App::<D>::new(title.to_string(), init_data);
    let run_result = event_loop.run_app(&mut app);
    let app_result = app.finish();
    run_result?;
    app_result
}

struct App<D: RenderDelegate> {
    title: String,
    init_data: Option<D::InitData>,
    state: Option<ViewerState<D>>,
    error: Option<String>,
}

impl<D: RenderDelegate> App<D> {
    fn new(title: String, init_data: D::InitData) -> Self {
        Self {
            title,
            init_data: Some(init_data),
            state: None,
            error: None,
        }
    }

    fn finish(self) -> Result<(), Box<dyn Error>> {
        if let Some(err) = self.error {
            Err(err.into())
        } else {
            Ok(())
        }
    }
}

impl<D: RenderDelegate + 'static> ApplicationHandler for App<D> {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.state.is_some() || self.error.is_some() {
            return;
        }

        let init_data = self.init_data.take().expect("init_data already consumed");

        match ViewerState::<D>::new(event_loop, &self.title, init_data) {
            Ok(state) => self.state = Some(state),
            Err(err) => {
                error!("Failed to initialize viewer: {err}");
                self.error = Some(err.to_string());
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        if state.window.id() != window_id {
            return;
        }

        if state.handle_window_event(event_loop, &event) {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size),
            WindowEvent::RedrawRequested => {
                let _frame_span = tracing::info_span!("frame").entered();
                let now = Instant::now();
                state.update_frame_time(now);
                match state.render() {
                    Ok(()) => {}
                    Err(
                        triad_gpu::wgpu::SurfaceError::Lost
                        | triad_gpu::wgpu::SurfaceError::Outdated,
                    ) => {
                        let size = state.window.inner_size();
                        state.resize(size);
                    }
                    Err(triad_gpu::wgpu::SurfaceError::OutOfMemory) => {
                        error!("GPU Out of Memory - exiting");
                        event_loop.exit();
                    }
                    Err(e) => error!("Render error: {:?}", e),
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(state) = self.state.as_ref() {
            state.window.request_redraw();
        }
    }
}

struct ViewerState<D: RenderDelegate> {
    window: Arc<Window>,
    renderer: Renderer,
    surface: SurfaceWrapper,
    registry: ResourceRegistry,
    delegate: D,
    camera: Camera,
    controller: CameraController,
    projection: Projection,
    last_frame: Instant,
    depth_texture: Option<triad_gpu::wgpu::Texture>,
    depth_view: Option<triad_gpu::wgpu::TextureView>,
}

impl<D: RenderDelegate> ViewerState<D> {
    fn new(
        event_loop: &winit::event_loop::ActiveEventLoop,
        title: &str,
        init_data: D::InitData,
    ) -> Result<Self, Box<dyn Error>> {
        let window_attributes = Window::default_attributes()
            .with_title(title)
            .with_inner_size(PhysicalSize::new(1280, 720));
        let window = Arc::new(event_loop.create_window(window_attributes)?);

        let renderer = pollster::block_on(Renderer::new())?;
        let size = window.inner_size();

        let surface = renderer.instance().create_surface(window.clone())?;
        let surface = renderer.create_surface(surface, size.width.max(1), size.height.max(1))?;

        let mut registry = ResourceRegistry::default();

        let delegate = D::create(&renderer, &mut registry, surface.format(), init_data)?;

        let bounds = delegate.bounds();
        info!(
            "Scene bounds: center={:?}, radius={}",
            bounds.center, bounds.radius
        );

        let camera_pos = bounds.center + Vec3::new(0.0, 0.0, bounds.radius * 2.5);
        let camera = Camera::new(camera_pos, bounds.center);
        let projection = Projection::new(
            size.width.max(1),
            size.height.max(1),
            std::f32::consts::FRAC_PI_3,
            0.01,
            bounds.radius * 10.0,
        );

        // Create depth texture if delegate needs one
        let (depth_texture, depth_view) = if let Some(depth_format) = delegate.depth_format() {
            let (tex, view) = Self::create_depth_texture(
                renderer.device(),
                size.width.max(1),
                size.height.max(1),
                depth_format,
            );
            (Some(tex), Some(view))
        } else {
            (None, None)
        };

        Ok(Self {
            window,
            renderer,
            surface,
            registry,
            delegate,
            camera,
            controller: CameraController::new(),
            projection,
            last_frame: Instant::now(),
            depth_texture,
            depth_view,
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

    fn handle_window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        event: &WindowEvent,
    ) -> bool {
        if let WindowEvent::KeyboardInput {
            event:
                KeyEvent {
                    state: ElementState::Pressed,
                    physical_key: PhysicalKey::Code(KeyCode::Escape),
                    ..
                },
            ..
        } = event
        {
            event_loop.exit();
            return true;
        }

        self.controller.process_event(event, &mut self.camera)
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        let mut config = self.surface.config().clone();
        config.width = new_size.width;
        config.height = new_size.height;
        self.surface.reconfigure(self.renderer.device(), config);
        self.projection.update_size(new_size.width, new_size.height);

        // Recreate depth texture if needed
        if let Some(depth_format) = self.delegate.depth_format() {
            let (tex, view) = Self::create_depth_texture(
                self.renderer.device(),
                new_size.width,
                new_size.height,
                depth_format,
            );
            self.depth_texture = Some(tex);
            self.depth_view = Some(view);
        }
    }

    fn update_frame_time(&mut self, now: Instant) {
        self.last_frame = now;
    }

    fn render(&mut self) -> Result<(), triad_gpu::wgpu::SurfaceError> {
        let view = self.camera.view_matrix();
        let proj = self.projection.matrix();
        let uniforms = CameraUniforms::from_matrices(view, proj, self.camera.position());

        self.delegate
            .update(self.renderer.queue(), &self.registry, &uniforms);

        let surface_texture = self.surface.get_current_texture()?;
        let surface_view = surface_texture
            .texture
            .create_view(&triad_gpu::wgpu::TextureViewDescriptor::default());

        let device = self.renderer.device();
        let mut encoder =
            device.create_command_encoder(&triad_gpu::wgpu::CommandEncoderDescriptor {
                label: Some("Frame Encoder"),
            });

        let ctx = RenderContext {
            color_view: &surface_view,
            depth_view: self.depth_view.as_ref(),
        };

        self.delegate.render(&mut encoder, ctx, &self.registry);

        self.renderer.queue().submit(Some(encoder.finish()));
        surface_texture.present();
        Ok(())
    }
}
