use crate::camera::{Camera, Projection};
use crate::controls::Controls;
use glam::Vec3;
use std::error::Error;
use std::sync::Arc;
use std::time::Instant;
use tracing::error;
use triad_gpu::{
    CameraUniforms, ExecutableFrameGraph, FrameGraphError, Renderer, ResourceRegistry, SurfaceWrapper,
};
use triad_gpu::wgpu;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

// Re-export egui for use by applications
pub use egui;

/// Error type for rendering operations.
#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    #[error("Surface error: {0}")]
    Surface(#[from] triad_gpu::wgpu::SurfaceError),
    #[error("Frame graph error: {0}")]
    FrameGraph(#[from] FrameGraphError),
}

/// Run the viewer with renderer initialization data and factory.
pub fn run_with_renderer_config<F, M>(
    title: &str,
    init_data: RendererInitData,
    configure_controls: F,
    create_manager: M,
) -> Result<(), Box<dyn Error>>
where
    F: FnOnce(&mut Controls),
    M: FnOnce(RendererInitData, &Renderer, &mut ResourceRegistry, triad_gpu::wgpu::TextureFormat, u32, u32) -> Result<Box<dyn RendererManagerTrait>, Box<dyn Error>> + Send + 'static,
{
    let event_loop = EventLoop::new().map_err(|e| format!("Failed to create event loop: {e}"))?;
    let mut controls = Controls::default();
    configure_controls(&mut controls);

    let mut app = App::new(title.to_string(), init_data, controls, create_manager);
    let run_result = event_loop.run_app(&mut app);
    let app_result = app.finish();
    run_result?;
    app_result
}

/// Initialization data for renderer.
pub struct RendererInitData {
    pub ply_path: Option<std::path::PathBuf>,
    pub initial_mode: u8, // 0=Points, 1=Gaussians, 2=Triangles
    pub point_size: f32,
    pub ply_receiver: Option<std::sync::mpsc::Receiver<std::path::PathBuf>>,
}

struct App {
    title: String,
    init_data: Option<RendererInitData>,
    controls: Option<Controls>,
    create_manager: Option<Box<dyn FnOnce(RendererInitData, &Renderer, &mut ResourceRegistry, wgpu::TextureFormat, u32, u32) -> Result<Box<dyn RendererManagerTrait>, Box<dyn Error>> + Send>>,
    state: Option<ViewerState>,
    error: Option<String>,
}

impl App {
    fn new<M>(title: String, init_data: RendererInitData, controls: Controls, create_manager: M) -> Self
    where
        M: FnOnce(RendererInitData, &Renderer, &mut ResourceRegistry, wgpu::TextureFormat, u32, u32) -> Result<Box<dyn RendererManagerTrait>, Box<dyn Error>> + Send + 'static,
    {
        Self {
            title,
            init_data: Some(init_data),
            controls: Some(controls),
            create_manager: Some(Box::new(create_manager)),
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

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.state.is_some() || self.error.is_some() {
            return;
        }

        let init_data = self.init_data.take().expect("init_data already consumed");
        let controls = self.controls.take().expect("controls already consumed");
        let create_manager = self.create_manager.take().expect("create_manager already consumed");

        match ViewerState::new(event_loop, &self.title, init_data, controls, create_manager) {
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
                        RenderError::Surface(triad_gpu::wgpu::SurfaceError::Lost)
                        | RenderError::Surface(triad_gpu::wgpu::SurfaceError::Outdated),
                    ) => {
                        let size = state.window.inner_size();
                        state.resize(size);
                    }
                    Err(RenderError::Surface(triad_gpu::wgpu::SurfaceError::OutOfMemory)) => {
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

struct ViewerState {
    window: Arc<Window>,
    renderer: Renderer,
    surface: SurfaceWrapper,
    registry: ResourceRegistry,
    renderer_manager: Box<dyn RendererManagerTrait>,
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

/// Trait for renderer managers that can build frame graphs.
pub trait RendererManager: Send + Sync {
    fn update_camera(&self, queue: &wgpu::Queue, registry: &ResourceRegistry, camera: &CameraUniforms);
    fn update_opacity_buffer(&self, queue: &wgpu::Queue, registry: &ResourceRegistry);
    fn check_pending_ply(&mut self) -> Option<std::path::PathBuf>;
    fn load_ply(&mut self, renderer: &Renderer, registry: &mut ResourceRegistry, ply_path: &std::path::PathBuf) -> Result<(), Box<dyn Error>>;
    fn build_frame_graph(&self, final_view: Arc<wgpu::TextureView>, depth_view: Option<Arc<wgpu::TextureView>>) -> Result<triad_gpu::ExecutableFrameGraph, triad_gpu::FrameGraphError>;
    fn resize_textures(&mut self, device: &wgpu::Device, registry: &mut ResourceRegistry, width: u32, height: u32) -> Result<(), Box<dyn Error>>;
    fn set_layer_opacity(&mut self, layer: u8, opacity: f32);
    fn get_layer_opacity(&self, layer: u8) -> f32;
    fn set_layer_enabled(&mut self, layer: u8, enabled: bool);
    fn is_layer_enabled(&self, layer: u8) -> bool;
}

/// Trait object for renderer manager
pub trait RendererManagerTrait: Send + Sync {
    fn update_camera(&self, queue: &wgpu::Queue, registry: &ResourceRegistry, camera: &CameraUniforms);
    fn update_opacity_buffer(&self, queue: &wgpu::Queue, registry: &ResourceRegistry);
    fn check_pending_ply(&mut self) -> Option<std::path::PathBuf>;
    fn load_ply(&mut self, renderer: &Renderer, registry: &mut ResourceRegistry, ply_path: &std::path::PathBuf) -> Result<(), Box<dyn Error>>;
    fn build_frame_graph(&self, final_view: Arc<wgpu::TextureView>, depth_view: Option<Arc<wgpu::TextureView>>) -> Result<triad_gpu::ExecutableFrameGraph, triad_gpu::FrameGraphError>;
    fn resize_textures(&mut self, device: &wgpu::Device, registry: &mut ResourceRegistry, width: u32, height: u32) -> Result<(), Box<dyn Error>>;
    fn set_layer_opacity(&mut self, layer: u8, opacity: f32);
    fn get_layer_opacity(&self, layer: u8) -> f32;
    fn set_layer_enabled(&mut self, layer: u8, enabled: bool);
    fn is_layer_enabled(&self, layer: u8) -> bool;
}

impl<M: RendererManager> RendererManagerTrait for M {
    fn update_camera(&self, queue: &wgpu::Queue, registry: &ResourceRegistry, camera: &CameraUniforms) {
        RendererManager::update_camera(self, queue, registry, camera);
    }
    fn update_opacity_buffer(&self, queue: &wgpu::Queue, registry: &ResourceRegistry) {
        RendererManager::update_opacity_buffer(self, queue, registry);
    }
    fn check_pending_ply(&mut self) -> Option<std::path::PathBuf> {
        RendererManager::check_pending_ply(self)
    }
    fn load_ply(&mut self, renderer: &Renderer, registry: &mut ResourceRegistry, ply_path: &std::path::PathBuf) -> Result<(), Box<dyn Error>> {
        RendererManager::load_ply(self, renderer, registry, ply_path)
    }
    fn build_frame_graph(&self, final_view: Arc<wgpu::TextureView>, depth_view: Option<Arc<wgpu::TextureView>>) -> Result<ExecutableFrameGraph, FrameGraphError> {
        RendererManager::build_frame_graph(self, final_view, depth_view)
    }
    fn resize_textures(&mut self, device: &wgpu::Device, registry: &mut ResourceRegistry, width: u32, height: u32) -> Result<(), Box<dyn Error>> {
        RendererManager::resize_textures(self, device, registry, width, height)
    }
    fn set_layer_opacity(&mut self, layer: u8, opacity: f32) {
        RendererManager::set_layer_opacity(self, layer, opacity);
    }
    fn get_layer_opacity(&self, layer: u8) -> f32 {
        RendererManager::get_layer_opacity(self, layer)
    }
    fn set_layer_enabled(&mut self, layer: u8, enabled: bool) {
        RendererManager::set_layer_enabled(self, layer, enabled);
    }
    fn is_layer_enabled(&self, layer: u8) -> bool {
        RendererManager::is_layer_enabled(self, layer)
    }
}

impl ViewerState {
    fn new(
        event_loop: &winit::event_loop::ActiveEventLoop,
        title: &str,
        init_data: RendererInitData,
        controls: Controls,
        create_manager: Box<dyn FnOnce(RendererInitData, &Renderer, &mut ResourceRegistry, triad_gpu::wgpu::TextureFormat, u32, u32) -> Result<Box<dyn RendererManagerTrait>, Box<dyn Error>> + Send>,
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

        // Initialize camera - just use default position (camera renders everything)
        let camera = Camera::new(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::ZERO,
        );
        let projection = Projection::new(
            size.width.max(1),
            size.height.max(1),
            std::f32::consts::FRAC_PI_3,
            0.01,
            10000.0, // Fixed far plane
        );

        // Create depth texture
        let (depth_texture, depth_view) = Self::create_depth_texture(
            renderer.device(),
            size.width.max(1),
            size.height.max(1),
            triad_gpu::wgpu::TextureFormat::Depth32Float,
        );

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

        // Create renderer manager using factory function
        let renderer_manager = create_manager(
            init_data,
            &renderer,
            &mut registry,
            surface.format(),
            size.width.max(1),
            size.height.max(1),
        )?;

        Ok(Self {
            window,
            renderer,
            surface,
            registry,
            renderer_manager,
            camera,
            controls,
            projection,
            last_frame: Instant::now(),
            depth_texture: Some(depth_texture),
            depth_view: Some(depth_view),
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

        // Resize layer textures
        if let Err(e) = self.renderer_manager.resize_textures(
            self.renderer.device(),
            &mut self.registry,
            new_size.width,
            new_size.height,
        ) {
            error!("Failed to resize layer textures: {}", e);
        }

        // Recreate depth texture
        let (tex, view) = Self::create_depth_texture(
            self.renderer.device(),
            new_size.width,
            new_size.height,
            triad_gpu::wgpu::TextureFormat::Depth32Float,
        );
        self.depth_texture = Some(tex);
        self.depth_view = Some(view);
    }

    fn render(&mut self) -> Result<(), RenderError> {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        self.controls.update(dt, &mut self.camera);

        let view = self.camera.view_matrix();
        let proj = self.projection.matrix();
        let uniforms = CameraUniforms::from_matrices(view, proj, self.camera.position());

        // Update camera
        self.renderer_manager.update_camera(
            self.renderer.queue(),
            &self.registry,
            &uniforms,
        );
        
        // Update opacity buffer
        self.renderer_manager.update_opacity_buffer(
            self.renderer.queue(),
            &self.registry,
        );

        // Check for pending PLY reload
        if let Some(ply_path) = self.renderer_manager.check_pending_ply() {
            if let Err(e) = self.renderer_manager.load_ply(
                &self.renderer,
                &mut self.registry,
                &ply_path,
            ) {
                error!("Failed to load PLY: {}", e);
            }
        }

        let surface_texture = self.surface.get_current_texture()?;
        let surface_view = Arc::new(
            surface_texture
                .texture
                .create_view(&triad_gpu::wgpu::TextureViewDescriptor::default())
        );

        // Build and execute frame graph
        // Create Arc for depth view - we need to create a new view from the texture
        let depth_view_arc = self.depth_texture.as_ref().map(|tex| {
            Arc::new(tex.create_view(&triad_gpu::wgpu::TextureViewDescriptor::default()))
        });
        let mut frame_graph = self.renderer_manager.build_frame_graph(
            surface_view.clone(),
            depth_view_arc,
        )?;
        
        frame_graph.execute(
            self.renderer.device(),
            self.renderer.queue(),
            &self.registry,
        );

        // Run egui frame
        let raw_input = self.egui_winit.take_egui_input(&self.window);
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            self.controls.run_ui(ctx);
            
            // Add layer controls UI
            egui::Window::new("Layers").show(ctx, |ui| {
                for layer_idx in 0..3 {
                    ui.horizontal(|ui| {
                        let mut enabled = self.renderer_manager.is_layer_enabled(layer_idx);
                        let layer_name = match layer_idx {
                            0 => "Points",
                            1 => "Gaussians",
                            2 => "Triangles",
                            _ => "Unknown",
                        };
                        if ui.checkbox(&mut enabled, layer_name).changed() {
                            self.renderer_manager.set_layer_enabled(layer_idx, enabled);
                        }
                        
                        if enabled {
                            let mut opacity = self.renderer_manager.get_layer_opacity(layer_idx);
                            ui.add(egui::Slider::new(&mut opacity, 0.0..=1.0)
                                .text("Opacity"));
                            if (opacity - self.renderer_manager.get_layer_opacity(layer_idx)).abs() > 0.001 {
                                self.renderer_manager.set_layer_opacity(layer_idx, opacity);
                            }
                        }
                    });
                }
            });
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
                .update_texture(self.renderer.device(), self.renderer.queue(), *id, image_delta);
        }

        // Create separate encoder for egui to work around lifetime requirements
        let mut egui_encoder =
            self.renderer.device().create_command_encoder(&triad_gpu::wgpu::CommandEncoderDescriptor {
                label: Some("egui Encoder"),
            });

        self.egui_renderer.update_buffers(
            self.renderer.device(),
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
