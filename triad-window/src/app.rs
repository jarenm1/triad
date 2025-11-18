use crate::camera::{Camera, CameraController, Projection};
use glam::{Mat4, Vec3};
use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::sync::Arc;
use tracing::{error, info};
use triad_gpu::{
    CameraUniforms, GaussianPoint, Handle, RenderPipelineBuilder, Renderer, ResourceRegistry,
    ShaderManager, SurfaceWrapper, ply_loader,
};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

pub fn run(ply_path: &Path) -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    let event_loop = EventLoop::new().map_err(|e| format!("Failed to create event loop: {e}"))?;
    let mut app = App::new(ply_path.to_path_buf());
    let run_result = event_loop.run_app(&mut app);
    let app_result = app.finish();
    run_result?;
    app_result
}

struct App {
    ply_path: PathBuf,
    state: Option<ViewerState>,
    error: Option<String>,
}

impl App {
    fn new(ply_path: PathBuf) -> Self {
        Self {
            ply_path,
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
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop ) {
        if self.state.is_some() || self.error.is_some() {
            return;
        }

        match ViewerState::new(event_loop, &self.ply_path) {
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

struct ViewerState {
    window: Arc<Window>,
    renderer: Renderer,
    surface: SurfaceWrapper,
    registry: ResourceRegistry,
    shader_manager: ShaderManager,
    resources: GaussianResources,
    camera: Camera,
    controller: CameraController,
    projection: Projection,
    last_frame: Instant,
}

impl ViewerState {
    fn new(event_loop: &winit::event_loop::ActiveEventLoop, ply_path: &Path) -> Result<Self, Box<dyn Error>> {
        let window_attributes = Window::default_attributes()
            .with_title("Triad Gaussian Viewer")
            .with_inner_size(PhysicalSize::new(1280, 720));
        let window = Arc::new(event_loop.create_window(window_attributes)?);

        let renderer = pollster::block_on(Renderer::new())?;
        let size = window.inner_size();

        let surface = unsafe { renderer.instance().create_surface(window.clone()) }?;
        let surface = renderer.create_surface(surface, size.width.max(1), size.height.max(1))?;

        let mut registry = ResourceRegistry::new();
        let mut shader_manager = ShaderManager::new();

        let ply_path_str = ply_path
            .to_str()
            .ok_or_else(|| format!("PLY path {:?} is not valid UTF-8", ply_path))?;
        info!("Loading gaussians from {}", ply_path_str);
        let gaussians = ply_loader::load_gaussians_from_ply(ply_path_str)?;
        info!("Loaded {} gaussians", gaussians.len());

        let bounds = SceneBounds::from_gaussians(&gaussians);
        let camera_pos = bounds.center + Vec3::new(0.0, 0.0, bounds.radius * 2.5);
        let camera = Camera::new(camera_pos, bounds.center);
        let projection = Projection::new(
            size.width.max(1),
            size.height.max(1),
            std::f32::consts::FRAC_PI_3,
            0.01,
            bounds.radius * 10.0,
        );

        let resources = GaussianResources::new(
            &renderer,
            &mut registry,
            &mut shader_manager,
            surface.format(),
            &gaussians,
        )?;

        Ok(Self {
            window,
            renderer,
            surface,
            registry,
            shader_manager,
            resources,
            camera,
            controller: CameraController::new(),
            projection,
            last_frame: Instant::now(),
        })
    }

    fn handle_window_event(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, event: &WindowEvent) -> bool {
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
    }

    fn update_frame_time(&mut self, now: Instant) {
        self.last_frame = now;
    }

    fn render(&mut self) -> Result<(), triad_gpu::wgpu::SurfaceError> {
        let view = self.camera.view_matrix();
        let proj = self.projection.matrix();
        let uniforms = CameraUniforms::from_matrices(view, proj, self.camera.position());
        let queue = self.renderer.queue();
        let camera_buffer = self
            .registry
            .get_buffer(self.resources.camera_buffer_handle.clone())
            .expect("camera buffer");
        queue.write_buffer(camera_buffer, 0, bytemuck::bytes_of(&uniforms));

        let surface_texture = self.surface.get_current_texture()?;
        let surface_view = surface_texture
            .texture
            .create_view(&triad_gpu::wgpu::TextureViewDescriptor::default());

        let device = self.renderer.device();
        let mut encoder =
            device.create_command_encoder(&triad_gpu::wgpu::CommandEncoderDescriptor {
                label: Some("Gaussian Frame Encoder"),
            });

        {
            let pipeline = self
                .registry
                .get_render_pipeline(self.resources.pipeline_handle.clone())
                .expect("pipeline");
            let bind_group = self
                .registry
                .get_bind_group(self.resources.bind_group_handle.clone())
                .expect("bind group");
            let index_buffer = self
                .registry
                .get_buffer(self.resources.index_buffer_handle.clone())
                .expect("index buffer");

            let mut render_pass =
                encoder.begin_render_pass(&triad_gpu::wgpu::RenderPassDescriptor {
                    label: Some("Gaussian Pass"),
                    color_attachments: &[Some(triad_gpu::wgpu::RenderPassColorAttachment {
                        view: &surface_view,
                        resolve_target: None,
                        ops: triad_gpu::wgpu::Operations {
                            load: triad_gpu::wgpu::LoadOp::Clear(triad_gpu::wgpu::Color {
                                r: 0.02,
                                g: 0.02,
                                b: 0.025,
                                a: 1.0,
                            }),
                            store: triad_gpu::wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });

            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(0, bind_group, &[]);
            render_pass
                .set_index_buffer(index_buffer.slice(..), triad_gpu::wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.resources.index_count, 0, 0..1);
        }

        queue.submit(Some(encoder.finish()));
        surface_texture.present();
        Ok(())
    }
}

struct GaussianResources {
    gaussian_count: u32,
    index_count: u32,
    gaussian_buffer_handle: Handle<triad_gpu::wgpu::Buffer>,
    camera_buffer_handle: Handle<triad_gpu::wgpu::Buffer>,
    index_buffer_handle: Handle<triad_gpu::wgpu::Buffer>,
    bind_group_handle: Handle<triad_gpu::wgpu::BindGroup>,
    pipeline_handle: Handle<triad_gpu::wgpu::RenderPipeline>,
}

impl GaussianResources {
    fn new(
        renderer: &Renderer,
        registry: &mut ResourceRegistry,
        shader_manager: &mut ShaderManager,
        color_format: triad_gpu::wgpu::TextureFormat,
        gaussians: &[GaussianPoint],
    ) -> Result<Self, Box<dyn Error>> {
        use triad_gpu::wgpu::util::DeviceExt;
        let device = renderer.device();

        let gaussian_buffer_handle = Handle::next();
        let gaussian_data = bytemuck::cast_slice(gaussians);
        let gaussian_buffer =
            device.create_buffer_init(&triad_gpu::wgpu::util::BufferInitDescriptor {
                label: Some("Gaussian Buffer"),
                contents: gaussian_data,
                usage: triad_gpu::wgpu::BufferUsages::STORAGE,
            });
        registry.register_buffer(gaussian_buffer_handle.clone(), gaussian_buffer);

        let camera_buffer_handle = Handle::next();
        let camera_buffer =
            device.create_buffer_init(&triad_gpu::wgpu::util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: bytemuck::cast_slice(&[CameraUniforms {
                    view_matrix: Mat4::IDENTITY.to_cols_array_2d(),
                    proj_matrix: Mat4::IDENTITY.to_cols_array_2d(),
                    view_pos: [0.0, 0.0, 0.0],
                    _padding: 0.0,
                }]),
                usage: triad_gpu::wgpu::BufferUsages::UNIFORM
                    | triad_gpu::wgpu::BufferUsages::COPY_DST,
            });
        registry.register_buffer(camera_buffer_handle.clone(), camera_buffer);

        let mut indices = Vec::with_capacity(gaussians.len() * 3);
        for i in 0..gaussians.len() as u32 {
            let base = i * 3;
            indices.push(base);
            indices.push(base + 1);
            indices.push(base + 2);
        }
        let index_buffer_handle = Handle::next();
        let index_buffer =
            device.create_buffer_init(&triad_gpu::wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: triad_gpu::wgpu::BufferUsages::INDEX,
            });
        registry.register_buffer(index_buffer_handle.clone(), index_buffer);

        let bind_group_layout =
            device.create_bind_group_layout(&triad_gpu::wgpu::BindGroupLayoutDescriptor {
                label: Some("Gaussian Bind Group Layout"),
                entries: &[
                    triad_gpu::wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: triad_gpu::wgpu::ShaderStages::VERTEX,
                        ty: triad_gpu::wgpu::BindingType::Buffer {
                            ty: triad_gpu::wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    triad_gpu::wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: triad_gpu::wgpu::ShaderStages::VERTEX,
                        ty: triad_gpu::wgpu::BindingType::Buffer {
                            ty: triad_gpu::wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: Some(
                                std::num::NonZeroU64::new(
                                    std::mem::size_of::<CameraUniforms>() as u64
                                )
                                .unwrap(),
                            ),
                        },
                        count: None,
                    },
                ],
            });

        let bind_group_handle = Handle::next();
        let bind_group = device.create_bind_group(&triad_gpu::wgpu::BindGroupDescriptor {
            label: Some("Gaussian Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                triad_gpu::wgpu::BindGroupEntry {
                    binding: 0,
                    resource: registry
                        .get_buffer(gaussian_buffer_handle.clone())
                        .unwrap()
                        .as_entire_binding(),
                },
                triad_gpu::wgpu::BindGroupEntry {
                    binding: 1,
                    resource: registry
                        .get_buffer(camera_buffer_handle.clone())
                        .unwrap()
                        .as_entire_binding(),
                },
            ],
        });
        registry.register_bind_group(bind_group_handle.clone(), bind_group);

        let vertex_shader_source = include_str!("../../triad-gpu/shaders/gaussian_vertex.wgsl");
        let fragment_shader_source = include_str!("../../triad-gpu/shaders/gaussian_fragment.wgsl");
        let vertex_shader =
            shader_manager.create_shader(device, Some("gaussian_vs"), vertex_shader_source);
        let fragment_shader =
            shader_manager.create_shader(device, Some("gaussian_fs"), fragment_shader_source);

        let pipeline_layout =
            device.create_pipeline_layout(&triad_gpu::wgpu::PipelineLayoutDescriptor {
                label: Some("Gaussian Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline_handle = RenderPipelineBuilder::new(device, shader_manager)
            .with_label("Gaussian Pipeline")
            .with_vertex_shader(vertex_shader)
            .with_fragment_shader(fragment_shader)
            .with_layout(pipeline_layout)
            .with_primitive(triad_gpu::wgpu::PrimitiveState {
                topology: triad_gpu::wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: triad_gpu::wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: triad_gpu::wgpu::PolygonMode::Fill,
                conservative: false,
            })
            .with_fragment_target(Some(triad_gpu::wgpu::ColorTargetState {
                format: color_format,
                blend: Some(triad_gpu::wgpu::BlendState::ALPHA_BLENDING),
                write_mask: triad_gpu::wgpu::ColorWrites::ALL,
            }))
            .build(registry)?;

        Ok(Self {
            gaussian_count: gaussians.len() as u32,
            index_count: (gaussians.len() as u32) * 3,
            gaussian_buffer_handle,
            camera_buffer_handle,
            index_buffer_handle,
            bind_group_handle,
            pipeline_handle,
        })
    }
}

#[derive(Debug)]
struct SceneBounds {
    min: Vec3,
    max: Vec3,
    center: Vec3,
    radius: f32,
}

impl SceneBounds {
    fn from_gaussians(gaussians: &[GaussianPoint]) -> Self {
        if gaussians.is_empty() {
            return Self {
                min: Vec3::ZERO,
                max: Vec3::ZERO,
                center: Vec3::ZERO,
                radius: 1.0,
            };
        }

        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        for g in gaussians {
            let pos = Vec3::new(
                g.position_radius[0],
                g.position_radius[1],
                g.position_radius[2],
            );
            min = min.min(pos);
            max = max.max(pos);
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
