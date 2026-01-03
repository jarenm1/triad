use crate::camera::{Camera, Projection};
use crate::controls::Controls;
use glam::Vec3;
use std::error::Error;
use std::sync::Arc;
use std::time::Instant;
use tracing::{error, info};
use triad_gpu::{
    CameraUniforms, RenderContext, RenderDelegate, Renderer, ResourceRegistry, SurfaceWrapper,
};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

// Re-export egui for use by applications
pub use egui;

/// Run the viewer with a custom render delegate.
pub fn run_with_delegate<D: RenderDelegate + 'static>(
    title: &str,
    init_data: D::InitData,
) -> Result<(), Box<dyn Error>>
where
    D::InitData: 'static,
{
    run_with_delegate_config::<D, _>(title, init_data, |_| {})
}

/// Run the viewer with a custom render delegate and control configuration.
pub fn run_with_delegate_config<D: RenderDelegate + 'static, F>(
    title: &str,
    init_data: D::InitData,
    configure_controls: F,
) -> Result<(), Box<dyn Error>>
where
    D::InitData: 'static,
    F: FnOnce(&mut Controls),
{
    let event_loop = EventLoop::new().map_err(|e| format!("Failed to create event loop: {e}"))?;
    let mut controls = Controls::default();
    configure_controls(&mut controls);

    let mut app = App::<D>::new(title.to_string(), init_data, controls);
    let run_result = event_loop.run_app(&mut app);
    let app_result = app.finish();
    run_result?;
    app_result
}

struct App<D: RenderDelegate> {
    title: String,
    init_data: Option<D::InitData>,
    controls: Option<Controls>,
    state: Option<ViewerState<D>>,
    error: Option<String>,
}

impl<D: RenderDelegate> App<D> {
    fn new(title: String, init_data: D::InitData, controls: Controls) -> Self {
        Self {
            title,
            init_data: Some(init_data),
            controls: Some(controls),
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
        let controls = self.controls.take().expect("controls already consumed");

        match ViewerState::<D>::new(event_loop, &self.title, init_data, controls) {
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

        // Pass events to egui first
        let egui_consumed = state.handle_egui_event(&event);

        // Only pass to other handlers if egui didn't want the event
        if !egui_consumed {
            if state.handle_window_event(event_loop, &event) {
                return;
            }
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size),
            WindowEvent::RedrawRequested => {
                let _frame_span = tracing::info_span!("frame").entered();
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
    controls: Controls,
    projection: Projection,
    last_frame: Instant,
    depth_texture: Option<triad_gpu::wgpu::Texture>,
    depth_view: Option<triad_gpu::wgpu::TextureView>,
    // egui integration
    egui_ctx: egui::Context,
    egui_winit: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
}

impl<D: RenderDelegate> ViewerState<D> {
    fn new(
        event_loop: &winit::event_loop::ActiveEventLoop,
        title: &str,
        init_data: D::InitData,
        controls: Controls,
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
        // Use a larger multiplier and ensure minimum far plane for better view range
        let far_plane = (bounds.radius * 50.0).max(100.0).min(10000.0);
        let projection = Projection::new(
            size.width.max(1),
            size.height.max(1),
            std::f32::consts::FRAC_PI_3,
            0.01,
            far_plane,
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

        // Initialize egui
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

        Ok(Self {
            window,
            renderer,
            surface,
            registry,
            delegate,
            camera,
            controls,
            projection,
            last_frame: Instant::now(),
            depth_texture,
            depth_view,
            egui_ctx,
            egui_winit,
            egui_renderer,
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

        self.controls.handle_event(event)
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

    fn render(&mut self) -> Result<(), triad_gpu::wgpu::SurfaceError> {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        self.controls.update(dt, &mut self.camera);

        let view = self.camera.view_matrix();
        let proj = self.projection.matrix();
        let uniforms = CameraUniforms::from_matrices(view, proj, self.camera.position());

        self.delegate
            .update(self.renderer.queue(), &self.registry, &uniforms);

        // Check for pending PLY reload
        if let Err(e) = self
            .delegate
            .handle_pending_ply_reload(&self.renderer, &mut self.registry)
        {
            tracing::error!("Failed to handle pending PLY reload: {}", e);
        }

        // Update projection far plane if bounds changed (e.g., after PLY reload)
        let bounds = self.delegate.bounds();
        let new_far = (bounds.radius * 50.0).max(100.0).min(10000.0);
        if (self.projection.far() - new_far).abs() > 0.1 {
            self.projection.set_far(new_far);
            info!(
                "Updated projection far plane to {} (bounds radius: {})",
                new_far, bounds.radius
            );
        }

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

        // Render the 3D scene first
        self.delegate.render(&mut encoder, ctx, &self.registry);

        // Run egui frame
        let raw_input = self.egui_winit.take_egui_input(&self.window);
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            self.controls.run_ui(ctx);
        });

        // Handle egui platform output (clipboard, cursor, etc.)
        self.egui_winit
            .handle_platform_output(&self.window, full_output.platform_output);

        // Render egui
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.surface.config().width, self.surface.config().height],
            pixels_per_point: self.window.scale_factor() as f32,
        };

        let tris = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);

        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(device, self.renderer.queue(), *id, image_delta);
        }

        // Submit 3D scene commands
        self.renderer.queue().submit(Some(encoder.finish()));

        // Create separate encoder for egui to work around lifetime requirements
        let mut egui_encoder =
            device.create_command_encoder(&triad_gpu::wgpu::CommandEncoderDescriptor {
                label: Some("egui Encoder"),
            });

        self.egui_renderer.update_buffers(
            device,
            self.renderer.queue(),
            &mut egui_encoder,
            &tris,
            &screen_descriptor,
        );

        // Render egui - the render_pass must be dropped before encoder.finish()
        render_egui(
            &self.egui_renderer,
            &mut egui_encoder,
            &surface_view,
            &tris,
            &screen_descriptor,
        );

        // Free textures that are no longer needed
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.renderer.queue().submit(Some(egui_encoder.finish()));
        surface_texture.present();
        Ok(())
    }
}

/// Helper function to render egui, encapsulating the render pass lifetime
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
