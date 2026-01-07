//! Renderer manager that uses frame graph for multi-layer rendering.

mod blend;
mod constants;
mod errors;
mod frame_graph_builder;
mod layer_factory;
mod layer_resources;
mod passes;

use crate::layers::LayerMode;
use blend::{BlendResources, LayerUniforms, create_blend_resources, recreate_blend_bind_group};
use constants::{DEFAULT_OPACITY, LAYER_COUNT};
use frame_graph_builder::build_frame_graph_with_cache;
use glam::{Mat4, Vec3};
use layer_factory::{
    GaussianComputeResources, create_gaussian_resources, create_point_resources,
    create_triangle_resources, update_layer_bind_group,
};
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use tracing::{debug_span, error, info, info_span};
use triad_data::triangulation;
use triad_gpu::{
    BufferUsage, CameraUniforms, ExecutableFrameGraph, FrameGraphError, GaussianPoint, Handle,
    PointPrimitive, Renderer, ResourceRegistry, TrianglePrimitive, ply_loader, wgpu,
};
use triad_window::RendererManager as RendererManagerTrait;

/// Parsed PLY data ready for GPU upload.
/// This is computed on a background thread to avoid blocking the render loop.
pub struct ParsedPlyData {
    pub points: Vec<PointPrimitive>,
    pub gaussians: Vec<GaussianPoint>,
    pub triangles: Vec<TrianglePrimitive>,
}

// Re-export public types
pub use errors::RendererManagerError;
pub use layer_resources::LayerResources;

/// Initialization data for the renderer manager.
pub struct RendererInitData {
    pub ply_path: Option<PathBuf>,
    pub initial_mode: LayerMode,
    pub point_size: f32,
    pub present_mode: wgpu::PresentMode,
    pub ply_receiver: Option<Arc<Mutex<mpsc::Receiver<PathBuf>>>>,
}

impl RendererInitData {
    pub fn new(ply_path: Option<PathBuf>, initial_mode: LayerMode, point_size: f32) -> Self {
        Self {
            ply_path,
            initial_mode,
            point_size,
            present_mode: wgpu::PresentMode::AutoVsync,
            ply_receiver: None,
        }
    }
}

/// Renderer manager that handles all layers and builds frame graphs.
pub struct RendererManager {
    // Shared resources
    camera_buffer: Handle<wgpu::Buffer>,

    // Per-layer resources
    point_resources: layer_resources::LayerResources,
    gaussian_resources: layer_resources::LayerResources,
    triangle_resources: layer_resources::LayerResources,

    // Gaussian compute resources
    gaussian_compute: GaussianComputeResources,

    // Blend resources
    blend_resources: BlendResources,

    // Layer state
    layer_opacity: [f32; LAYER_COUNT],
    enabled_layers: [bool; LAYER_COUNT],

    // Configuration
    point_size: f32,
    surface_format: wgpu::TextureFormat,
    surface_width: u32,
    surface_height: u32,

    // PLY loading - path receiver from UI
    ply_receiver: Option<Arc<Mutex<mpsc::Receiver<PathBuf>>>>,
    // Parsed data receiver from background thread (wrapped in Mutex for Sync)
    parsed_data_receiver: Option<Mutex<mpsc::Receiver<ParsedPlyData>>>,
    // Flag indicating loading is in progress
    loading_in_progress: bool,

    // Frame graph cache - stores cache key and execution order
    // Execution order is cached to avoid expensive topological sort when structure unchanged
    frame_graph_cache: Option<(u8, Vec<usize>)>,
}

impl RendererManager {
    /// Create a new renderer manager with all resources initialized.
    pub fn create(
        renderer: &Renderer,
        registry: &mut ResourceRegistry,
        surface_format: wgpu::TextureFormat,
        surface_width: u32,
        surface_height: u32,
        init_data: RendererInitData,
    ) -> Result<Self, RendererManagerError> {
        let device = renderer.device();

        // Initialize layer state
        let mut enabled_layers = [false; LAYER_COUNT];
        enabled_layers[init_data.initial_mode as usize] = true;
        let layer_opacity = DEFAULT_OPACITY;

        // Load vertices if PLY path is provided
        let vertices = if let Some(ref ply_path) = init_data.ply_path {
            let ply_path_str = ply_path.to_str().ok_or_else(|| {
                errors::RendererManagerError::Resource(format!("Invalid PLY path: {:?}", ply_path))
            })?;
            info!("Loading PLY data from {}", ply_path_str);
            ply_loader::load_vertices_from_ply(ply_path_str)?
        } else {
            info!("No PLY path provided - creating empty renderer (data can be loaded at runtime)");
            Vec::new()
        };

        // Create shared camera buffer
        let camera_buffer = renderer
            .create_buffer()
            .label("Shared Camera Buffer")
            .with_pod_data(&[CameraUniforms {
                view_matrix: Mat4::IDENTITY.to_cols_array_2d(),
                proj_matrix: Mat4::IDENTITY.to_cols_array_2d(),
                view_pos: [0.0, 0.0, 0.0],
                _padding: 0.0,
            }])
            .usage(BufferUsage::Uniform)
            .build(registry)?;

        // Create layer textures
        let point_texture =
            create_layer_texture(device, surface_width, surface_height, surface_format);
        let point_texture_view =
            Arc::new(point_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        let point_texture_handle = registry.insert(point_texture);

        let gaussian_texture =
            create_layer_texture(device, surface_width, surface_height, surface_format);
        let gaussian_texture_view =
            Arc::new(gaussian_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        let gaussian_texture_handle = registry.insert(gaussian_texture);

        let triangle_texture =
            create_layer_texture(device, surface_width, surface_height, surface_format);
        let triangle_texture_view =
            Arc::new(triangle_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        let triangle_texture_handle = registry.insert(triangle_texture);

        // Create point resources
        let point_resources = create_point_resources(
            renderer,
            registry,
            &vertices,
            init_data.point_size,
            surface_format,
            camera_buffer,
            point_texture_handle,
            point_texture_view.clone(),
        )?;

        // Create Gaussian resources (including compute)
        let (gaussian_resources, gaussian_compute) = create_gaussian_resources(
            renderer,
            registry,
            &init_data.ply_path,
            surface_format,
            camera_buffer,
            gaussian_texture_handle,
            gaussian_texture_view.clone(),
        )?;

        // Create triangle resources
        let triangle_resources = create_triangle_resources(
            renderer,
            registry,
            &vertices,
            &init_data.ply_path,
            surface_format,
            camera_buffer,
            triangle_texture_handle,
            triangle_texture_view.clone(),
        )?;

        // Create blend resources
        let blend_resources = create_blend_resources(
            renderer,
            registry,
            surface_format,
            &point_texture_view,
            &gaussian_texture_view,
            &triangle_texture_view,
        )?;

        Ok(Self {
            camera_buffer,
            point_resources,
            gaussian_resources,
            triangle_resources,
            gaussian_compute,
            blend_resources,
            layer_opacity,
            enabled_layers,
            point_size: init_data.point_size,
            surface_format,
            surface_width,
            surface_height,
            ply_receiver: init_data.ply_receiver,
            parsed_data_receiver: None,
            loading_in_progress: false,
            frame_graph_cache: None,
        })
    }

    /// Create a layer texture.
    fn create_layer_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Result<wgpu::Texture, RendererManagerError> {
        Ok(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Layer Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }))
    }

    /// Compute cache key from enabled layers state.
    fn compute_cache_key(&self) -> u8 {
        let mut key = 0u8;
        if self.enabled_layers[0] {
            key |= 1 << 0;
        }
        if self.enabled_layers[1] {
            key |= 1 << 1;
        }
        if self.enabled_layers[2] {
            key |= 1 << 2;
        }
        key
    }

    /// Build frame graph with all enabled layers.
    /// The frame graph structure is cached - the expensive topological sort is skipped
    /// when layer enable/disable state hasn't changed.
    /// Note: Views change every frame, so Pass objects are rebuilt each frame,
    /// but the expensive dependency analysis is cached and reused.
    pub fn build_frame_graph(
        &mut self,
        final_view: Arc<wgpu::TextureView>,
        depth_view: Option<Arc<wgpu::TextureView>>,
    ) -> Result<ExecutableFrameGraph, FrameGraphError> {
        let cache_key = self.compute_cache_key();

        // Get cached execution order if structure hasn't changed
        let cached_order = if let Some((cached_key, cached_order)) = &self.frame_graph_cache {
            if *cached_key == cache_key {
                Some(cached_order.as_slice())
            } else {
                // Structure changed, clear cache
                self.frame_graph_cache = None;
                None
            }
        } else {
            None
        };

        // Build frame graph (will use cached execution order if available)
        let frame_graph = build_frame_graph_with_cache(
            self.camera_buffer,
            &self.enabled_layers,
            &self.layer_opacity,
            &self.point_resources,
            &self.gaussian_resources,
            &self.triangle_resources,
            Some(self.gaussian_compute.sort_pipeline),
            Some(self.gaussian_compute.sort_bind_group),
            Some(self.gaussian_compute.sort_buffer),
            self.blend_resources.pipeline,
            self.blend_resources.bind_group,
            self.blend_resources.opacity_buffer,
            final_view,
            depth_view,
            cached_order,
        )?;

        // Cache the execution order for next frame
        self.frame_graph_cache = Some((cache_key, frame_graph.execution_order().to_vec()));

        Ok(frame_graph)
    }

    /// Update camera uniforms.
    pub fn update_camera(
        &self,
        queue: &wgpu::Queue,
        registry: &ResourceRegistry,
        camera: &CameraUniforms,
    ) {
        let camera_buffer = registry.get(self.camera_buffer).expect("camera buffer");
        queue.write_buffer(camera_buffer, 0, bytemuck::bytes_of(camera));
    }

    /// Update layer opacity buffer.
    pub fn update_opacity_buffer(&self, queue: &wgpu::Queue, registry: &ResourceRegistry) {
        let opacity_buffer = registry
            .get(self.blend_resources.opacity_buffer)
            .expect("opacity buffer");
        let uniforms = LayerUniforms {
            opacity: self.layer_opacity,
            _padding: 0.0,
        };
        queue.write_buffer(opacity_buffer, 0, bytemuck::cast_slice(&[uniforms]));
    }

    /// Set layer opacity.
    pub fn set_layer_opacity(&mut self, layer: LayerMode, opacity: f32) {
        self.layer_opacity[layer as usize] = opacity.clamp(0.0, 1.0);
    }

    /// Get layer opacity.
    pub fn get_layer_opacity(&self, layer: LayerMode) -> f32 {
        self.layer_opacity[layer as usize]
    }

    /// Set layer enabled state.
    /// This invalidates the frame graph cache since the structure changes.
    pub fn set_layer_enabled(&mut self, layer: LayerMode, enabled: bool) {
        self.enabled_layers[layer as usize] = enabled;
        // Invalidate cache when layer state changes
        self.frame_graph_cache = None;
    }

    /// Check if layer is enabled.
    pub fn is_layer_enabled(&self, layer: LayerMode) -> bool {
        self.enabled_layers[layer as usize]
    }

    /// Load PLY file and update all layer buffers.
    pub fn load_ply(
        &mut self,
        renderer: &Renderer,
        registry: &mut ResourceRegistry,
        ply_path: &PathBuf,
    ) -> Result<(), errors::RendererManagerError> {
        let ply_path_str = ply_path.to_str().ok_or_else(|| {
            errors::RendererManagerError::Resource(format!("Invalid PLY path: {:?}", ply_path))
        })?;

        info!("Loading PLY data from {}", ply_path_str);
        let vertices = ply_loader::load_vertices_from_ply(ply_path_str)?;
        info!("Loaded {} vertices", vertices.len());

        // Update point buffer
        let mut points: Vec<triad_gpu::PointPrimitive> = vertices
            .iter()
            .map(|v| {
                triad_gpu::PointPrimitive::new(v.position, self.point_size, v.color, v.opacity)
            })
            .collect();

        if points.is_empty() {
            points.push(triad_gpu::PointPrimitive::new(
                Vec3::ZERO,
                self.point_size,
                Vec3::ZERO,
                0.0,
            ));
        }

        let point_buffer = renderer
            .create_buffer()
            .label("Point Buffer")
            .with_pod_data(&points)
            .usage(BufferUsage::Storage { read_only: true })
            .build(registry)?;

        // Recreate bind group with new buffer
        let new_bind_group = update_layer_bind_group(
            renderer,
            registry,
            "Point",
            self.point_resources.bind_group_layout,
            point_buffer,
            self.camera_buffer,
        )?;

        self.point_resources.data_buffer = point_buffer;
        self.point_resources.bind_group = new_bind_group;
        self.point_resources.vertex_count = if points.len() == 1 && vertices.is_empty() {
            0
        } else {
            points.len() as u32 * 3
        };

        // Update Gaussian buffer
        let mut gaussians = ply_loader::load_gaussians_from_ply(ply_path_str)?;
        if gaussians.is_empty() {
            use triad_gpu::GaussianPoint;
            gaussians.push(GaussianPoint::new(
                Vec3::ZERO,
                Vec3::ZERO,
                0.0,
                [0.0, 0.0, 0.0, 1.0],
                Vec3::ONE,
            ));
        }

        let gaussian_buffer = renderer
            .create_buffer()
            .label("Gaussian Buffer")
            .with_pod_data(&gaussians)
            .usage(BufferUsage::Storage { read_only: true })
            .build(registry)?;

        // Update sort buffer size if needed
        let sort_data_size = gaussians.len() * std::mem::size_of::<(f32, u32)>();
        let sort_buffer_handle = renderer
            .create_buffer()
            .label("Sort Buffer")
            .size(sort_data_size as u64)
            .usage(BufferUsage::Storage { read_only: false })
            .build(registry)?;

        // Recreate compute bind group manually (builder has borrow issues)
        let device = renderer.device();
        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Gaussian Sort Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let compute_layout = registry.insert(compute_bind_group_layout);

        let compute_bind_group_inner = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gaussian Sort Bind Group"),
            layout: registry
                .get::<wgpu::BindGroupLayout>(compute_layout)
                .unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(
                        registry
                            .get::<wgpu::Buffer>(gaussian_buffer)
                            .unwrap()
                            .as_entire_buffer_binding(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(
                        registry
                            .get::<wgpu::Buffer>(sort_buffer_handle)
                            .unwrap()
                            .as_entire_buffer_binding(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(
                        registry
                            .get::<wgpu::Buffer>(self.camera_buffer)
                            .unwrap()
                            .as_entire_buffer_binding(),
                    ),
                },
            ],
        });
        let compute_bind_group = registry.insert(compute_bind_group_inner);

        let mut indices = Vec::with_capacity(gaussians.len() * 3);
        for i in 0..gaussians.len() as u32 {
            let base = i * 3;
            indices.push(base);
            indices.push(base + 1);
            indices.push(base + 2);
        }
        if indices.is_empty() {
            indices.push(0);
            indices.push(1);
            indices.push(2);
        }
        let index_buffer = renderer
            .create_buffer()
            .label("Gaussian Index Buffer")
            .with_pod_data(&indices)
            .usage(BufferUsage::Index)
            .build(registry)?;

        // Recreate render bind group
        let new_gaussian_bind_group = update_layer_bind_group(
            renderer,
            registry,
            "Gaussian",
            self.gaussian_resources.bind_group_layout,
            gaussian_buffer,
            self.camera_buffer,
        )?;

        self.gaussian_resources.data_buffer = gaussian_buffer;
        self.gaussian_resources.index_buffer = Some(index_buffer);
        self.gaussian_resources.index_count = if gaussians.len() == 1 {
            0
        } else {
            gaussians.len() as u32 * 3
        };
        self.gaussian_resources.bind_group = new_gaussian_bind_group;

        // Update compute resources
        self.gaussian_compute.sort_buffer = sort_buffer_handle;
        self.gaussian_compute.sort_bind_group = compute_bind_group;
        self.gaussian_compute.sort_bind_group_layout = compute_layout;

        // Update triangle buffer
        let mut triangles: Vec<triad_gpu::TrianglePrimitive> =
            if ply_loader::ply_has_faces(ply_path_str).unwrap_or(false) {
                ply_loader::load_triangles_from_ply(ply_path_str)?
            } else {
                let positions: Vec<Vec3> = vertices.iter().map(|v| v.position).collect();
                let triangle_indices = triangulation::triangulate_points(&positions);

                triangle_indices
                    .iter()
                    .map(|[i0, i1, i2]| {
                        let v0 = &vertices[*i0];
                        let v1 = &vertices[*i1];
                        let v2 = &vertices[*i2];
                        let avg_color = (v0.color + v1.color + v2.color) / 3.0;
                        let avg_opacity = (v0.opacity + v1.opacity + v2.opacity) / 3.0;
                        triad_gpu::TrianglePrimitive::new(
                            v0.position,
                            v1.position,
                            v2.position,
                            avg_color,
                            avg_opacity,
                        )
                    })
                    .collect()
            };

        if triangles.is_empty() {
            triangles.push(triad_gpu::TrianglePrimitive::new(
                Vec3::ZERO,
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::ZERO,
                0.0,
            ));
        }

        let triangle_buffer = renderer
            .create_buffer()
            .label("Triangle Buffer")
            .with_pod_data(&triangles)
            .usage(BufferUsage::Storage { read_only: true })
            .build(registry)?;

        let mut indices = Vec::with_capacity(triangles.len() * 3);
        for i in 0..triangles.len() as u32 {
            let base = i * 3;
            indices.push(base);
            indices.push(base + 1);
            indices.push(base + 2);
        }
        let index_buffer = renderer
            .create_buffer()
            .label("Triangle Index Buffer")
            .with_pod_data(&indices)
            .usage(BufferUsage::Index)
            .build(registry)?;

        // Recreate triangle bind group
        let new_triangle_bind_group = update_layer_bind_group(
            renderer,
            registry,
            "Triangle",
            self.triangle_resources.bind_group_layout,
            triangle_buffer,
            self.camera_buffer,
        )?;

        self.triangle_resources.data_buffer = triangle_buffer;
        self.triangle_resources.index_buffer = Some(index_buffer);
        self.triangle_resources.index_count = triangles.len() as u32 * 3;
        self.triangle_resources.bind_group = new_triangle_bind_group;

        Ok(())
    }

    /// Check for pending PLY path requests and spawn background loading.
    /// This does NOT block the render thread - loading happens in background.
    pub fn check_and_start_ply_loading(&mut self) {
        // Don't start new loading if already in progress
        if self.loading_in_progress {
            return;
        }

        // Check for new PLY path requests
        let mut new_path = None;
        if let Some(ref receiver) = self.ply_receiver {
            if let Ok(receiver) = receiver.lock() {
                while let Ok(ply_path) = receiver.try_recv() {
                    info!("Received PLY import request: {:?}", ply_path);
                    new_path = Some(ply_path);
                }
            }
        }

        // Start background loading if we have a new path
        if let Some(ply_path) = new_path {
            self.start_background_loading(ply_path);
        }
    }

    /// Spawn background thread to parse PLY file.
    fn start_background_loading(&mut self, ply_path: PathBuf) {
        let (tx, rx) = mpsc::channel();
        let point_size = self.point_size;

        self.parsed_data_receiver = Some(Mutex::new(rx));
        self.loading_in_progress = true;

        thread::spawn(move || {
            // Create a span for the entire background operation
            let _thread_span = info_span!("ply_background_load", path = ?ply_path).entered();

            info!("Background: Starting PLY parse for {:?}", ply_path);

            let ply_path_str = match ply_path.to_str() {
                Some(s) => s.to_string(),
                None => {
                    error!("Invalid PLY path: {:?}", ply_path);
                    return;
                }
            };

            // Load vertices (used for points and triangulation)
            let vertices = {
                let _span = debug_span!("load_vertices").entered();
                match ply_loader::load_vertices_from_ply(&ply_path_str) {
                    Ok(v) => v,
                    Err(e) => {
                        error!("Failed to load vertices: {}", e);
                        return;
                    }
                }
            };
            info!("Background: Loaded {} vertices", vertices.len());

            // Convert to points
            let points: Vec<PointPrimitive> = {
                let _span = debug_span!("convert_to_points", count = vertices.len()).entered();
                if vertices.is_empty() {
                    vec![PointPrimitive::new(Vec3::ZERO, point_size, Vec3::ZERO, 0.0)]
                } else {
                    vertices
                        .iter()
                        .map(|v| PointPrimitive::new(v.position, point_size, v.color, v.opacity))
                        .collect()
                }
            };

            // Load gaussians
            let gaussians: Vec<GaussianPoint> = {
                let _span = debug_span!("load_gaussians").entered();
                match ply_loader::load_gaussians_from_ply(&ply_path_str) {
                    Ok(g) if !g.is_empty() => g,
                    _ => vec![GaussianPoint::new(
                        Vec3::ZERO,
                        Vec3::ZERO,
                        0.0,
                        [0.0, 0.0, 0.0, 1.0],
                        Vec3::ONE,
                    )],
                }
            };
            info!("Background: Loaded {} gaussians", gaussians.len());

            // Load or generate triangles
            let triangles: Vec<TrianglePrimitive> =
                if ply_loader::ply_has_faces(&ply_path_str).unwrap_or(false) {
                    let _span = debug_span!("load_triangles_from_faces").entered();
                    match ply_loader::load_triangles_from_ply(&ply_path_str) {
                        Ok(t) if !t.is_empty() => t,
                        _ => vec![TrianglePrimitive::new(
                            Vec3::ZERO,
                            Vec3::new(1.0, 0.0, 0.0),
                            Vec3::new(0.0, 1.0, 0.0),
                            Vec3::ZERO,
                            0.0,
                        )],
                    }
                } else if !vertices.is_empty() {
                    info!("Background: Starting triangulation...");
                    let _span = debug_span!("triangulate", vertex_count = vertices.len()).entered();
                    let positions: Vec<Vec3> = vertices.iter().map(|v| v.position).collect();
                    let triangle_indices = triangulation::triangulate_points(&positions);
                    info!(
                        "Background: Triangulation complete, {} triangles",
                        triangle_indices.len()
                    );

                    triangle_indices
                        .iter()
                        .map(|[i0, i1, i2]| {
                            let v0 = &vertices[*i0];
                            let v1 = &vertices[*i1];
                            let v2 = &vertices[*i2];
                            let avg_color = (v0.color + v1.color + v2.color) / 3.0;
                            let avg_opacity = (v0.opacity + v1.opacity + v2.opacity) / 3.0;
                            TrianglePrimitive::new(
                                v0.position, v1.position, v2.position, avg_color, avg_opacity,
                            )
                        })
                        .collect()
                } else {
                    vec![TrianglePrimitive::new(
                        Vec3::ZERO,
                        Vec3::new(1.0, 0.0, 0.0),
                        Vec3::new(0.0, 1.0, 0.0),
                        Vec3::ZERO,
                        0.0,
                    )]
                };

            info!("Background: PLY parse complete, sending data to main thread");
            let _ = tx.send(ParsedPlyData {
                points,
                gaussians,
                triangles,
            });
        });
    }

    /// Check for parsed PLY data from background thread and apply to GPU buffers.
    /// Returns true if data was applied (caller may want to log this).
    pub fn apply_parsed_ply_data(
        &mut self,
        renderer: &Renderer,
        registry: &mut ResourceRegistry,
    ) -> bool {
        // Check if we have parsed data ready
        let parsed_data = if let Some(ref receiver) = self.parsed_data_receiver {
            let receiver = receiver.lock().unwrap();
            match receiver.try_recv() {
                Ok(data) => Some(data),
                Err(mpsc::TryRecvError::Empty) => None,
                Err(mpsc::TryRecvError::Disconnected) => {
                    // Thread finished but didn't send data (error case)
                    drop(receiver); // Release lock before modifying self
                    self.loading_in_progress = false;
                    self.parsed_data_receiver = None;
                    return false;
                }
            }
        } else {
            None
        };

        // Apply parsed data if available
        if let Some(data) = parsed_data {
            info!("Applying parsed PLY data to GPU buffers");
            if let Err(e) = self.apply_parsed_data(renderer, registry, data) {
                error!("Failed to apply parsed PLY data: {}", e);
            }
            self.loading_in_progress = false;
            self.parsed_data_receiver = None;
            return true;
        }

        false
    }

    /// Apply parsed PLY data to GPU buffers (fast - just buffer creation).
    fn apply_parsed_data(
        &mut self,
        renderer: &Renderer,
        registry: &mut ResourceRegistry,
        data: ParsedPlyData,
    ) -> Result<(), errors::RendererManagerError> {
        let _span = debug_span!("apply_parsed_data",
            point_count = data.points.len(),
            gaussian_count = data.gaussians.len(),
            triangle_count = data.triangles.len()
        ).entered();

        // Update point buffer
        {
            let _span = debug_span!("update_point_buffer").entered();
            let point_buffer = renderer
                .create_buffer()
                .label("Point Buffer")
                .with_pod_data(&data.points)
                .usage(BufferUsage::Storage { read_only: true })
                .build(registry)?;

            let new_point_bind_group = update_layer_bind_group(
                renderer,
                registry,
                "Point",
                self.point_resources.bind_group_layout,
                point_buffer,
                self.camera_buffer,
            )?;

            self.point_resources.data_buffer = point_buffer;
            self.point_resources.bind_group = new_point_bind_group;
            self.point_resources.vertex_count = data.points.len() as u32 * 3;
        }

        // Update gaussian buffer
        {
            let _span = debug_span!("update_gaussian_buffer").entered();
            let gaussian_buffer = renderer
            .create_buffer()
            .label("Gaussian Buffer")
            .with_pod_data(&data.gaussians)
            .usage(BufferUsage::Storage { read_only: true })
            .build(registry)?;

        let sort_data_size = data.gaussians.len() * std::mem::size_of::<(f32, u32)>();
        let sort_buffer_handle = renderer
            .create_buffer()
            .label("Sort Buffer")
            .size(sort_data_size as u64)
            .usage(BufferUsage::Storage { read_only: false })
            .build(registry)?;

        // Recreate compute bind group
        let device = renderer.device();
        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Gaussian Sort Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let gaussian_buf = registry.get(gaussian_buffer).ok_or_else(|| {
            errors::RendererManagerError::Resource("Gaussian buffer not found".to_string())
        })?;
        let sort_buf = registry.get(sort_buffer_handle).ok_or_else(|| {
            errors::RendererManagerError::Resource("Sort buffer not found".to_string())
        })?;
        let camera_buf = registry.get(self.camera_buffer).ok_or_else(|| {
            errors::RendererManagerError::Resource("Camera buffer not found".to_string())
        })?;

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gaussian Sort Bind Group"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gaussian_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sort_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: camera_buf.as_entire_binding(),
                },
            ],
        });
        let compute_layout = registry.insert(compute_bind_group_layout);
        let compute_bind_group = registry.insert(compute_bind_group);

        // Update gaussian render bind group
        let new_gaussian_bind_group = update_layer_bind_group(
            renderer,
            registry,
            "Gaussian",
            self.gaussian_resources.bind_group_layout,
            gaussian_buffer,
            self.camera_buffer,
        )?;

        // Generate index buffer for gaussians
        let mut indices = Vec::with_capacity(data.gaussians.len() * 3);
        for i in 0..data.gaussians.len() as u32 {
            let base = i * 3;
            indices.push(base);
            indices.push(base + 1);
            indices.push(base + 2);
        }
        let index_buffer = renderer
            .create_buffer()
            .label("Gaussian Index Buffer")
            .with_pod_data(&indices)
            .usage(BufferUsage::Index)
            .build(registry)?;

            self.gaussian_resources.data_buffer = gaussian_buffer;
            self.gaussian_resources.index_buffer = Some(index_buffer);
            self.gaussian_resources.index_count = data.gaussians.len() as u32 * 3;
            self.gaussian_resources.bind_group = new_gaussian_bind_group;

            self.gaussian_compute.sort_buffer = sort_buffer_handle;
            self.gaussian_compute.sort_bind_group = compute_bind_group;
            self.gaussian_compute.sort_bind_group_layout = compute_layout;
        }

        // Update triangle buffer
        {
            let _span = debug_span!("update_triangle_buffer").entered();
        let triangle_buffer = renderer
            .create_buffer()
            .label("Triangle Buffer")
            .with_pod_data(&data.triangles)
            .usage(BufferUsage::Storage { read_only: true })
            .build(registry)?;

        let mut tri_indices = Vec::with_capacity(data.triangles.len() * 3);
        for i in 0..data.triangles.len() as u32 {
            let base = i * 3;
            tri_indices.push(base);
            tri_indices.push(base + 1);
            tri_indices.push(base + 2);
        }
        let tri_index_buffer = renderer
            .create_buffer()
            .label("Triangle Index Buffer")
            .with_pod_data(&tri_indices)
            .usage(BufferUsage::Index)
            .build(registry)?;

        let new_triangle_bind_group = update_layer_bind_group(
            renderer,
            registry,
            "Triangle",
            self.triangle_resources.bind_group_layout,
            triangle_buffer,
            self.camera_buffer,
        )?;

            self.triangle_resources.data_buffer = triangle_buffer;
            self.triangle_resources.index_buffer = Some(tri_index_buffer);
            self.triangle_resources.index_count = data.triangles.len() as u32 * 3;
            self.triangle_resources.bind_group = new_triangle_bind_group;
        }

        Ok(())
    }

    /// Returns true if PLY loading is currently in progress.
    pub fn is_loading(&self) -> bool {
        self.loading_in_progress
    }

    /// Resize layer textures when surface is resized.
    pub fn resize_textures(
        &mut self,
        device: &wgpu::Device,
        registry: &mut ResourceRegistry,
        width: u32,
        height: u32,
    ) -> Result<(), errors::RendererManagerError> {
        self.surface_width = width;
        self.surface_height = height;

        // Recreate textures
        let point_texture = create_layer_texture(device, width, height, self.surface_format);
        let point_texture_view =
            Arc::new(point_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.point_resources.texture = registry.insert(point_texture);
        self.point_resources.texture_view = point_texture_view.clone();

        let gaussian_texture = create_layer_texture(device, width, height, self.surface_format);
        let gaussian_texture_view =
            Arc::new(gaussian_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.gaussian_resources.texture = registry.insert(gaussian_texture);
        self.gaussian_resources.texture_view = gaussian_texture_view.clone();

        let triangle_texture = create_layer_texture(device, width, height, self.surface_format);
        let triangle_texture_view =
            Arc::new(triangle_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.triangle_resources.texture = registry.insert(triangle_texture);
        self.triangle_resources.texture_view = triangle_texture_view.clone();

        // Recreate blend bind group with new texture views
        let new_blend_bind_group = recreate_blend_bind_group(
            device,
            registry,
            self.blend_resources.bind_group_layout,
            &point_texture_view,
            &gaussian_texture_view,
            &triangle_texture_view,
            self.blend_resources.sampler,
            self.blend_resources.opacity_buffer,
        )?;
        self.blend_resources.bind_group = new_blend_bind_group;

        Ok(())
    }
}

/// Create a layer texture.
fn create_layer_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Layer Texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

// Implement the RendererManager trait from triad-window
impl RendererManagerTrait for RendererManager {
    fn update_camera(
        &self,
        queue: &wgpu::Queue,
        registry: &ResourceRegistry,
        camera: &CameraUniforms,
    ) {
        self.update_camera(queue, registry, camera);
    }

    fn update_opacity_buffer(&self, queue: &wgpu::Queue, registry: &ResourceRegistry) {
        self.update_opacity_buffer(queue, registry);
    }

    fn check_and_start_ply_loading(&mut self) {
        self.check_and_start_ply_loading();
    }

    fn apply_parsed_ply_data(
        &mut self,
        renderer: &Renderer,
        registry: &mut ResourceRegistry,
    ) -> bool {
        self.apply_parsed_ply_data(renderer, registry)
    }

    fn build_frame_graph(
        &mut self,
        final_view: Arc<wgpu::TextureView>,
        depth_view: Option<Arc<wgpu::TextureView>>,
    ) -> Result<ExecutableFrameGraph, FrameGraphError> {
        self.build_frame_graph(final_view, depth_view)
    }

    fn resize_textures(
        &mut self,
        device: &wgpu::Device,
        registry: &mut ResourceRegistry,
        width: u32,
        height: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.resize_textures(device, registry, width, height)
            .map_err(|e| e.into())
    }

    fn set_layer_opacity(&mut self, layer: u8, opacity: f32) {
        let layer_mode = u8_to_layer_mode(layer);
        self.set_layer_opacity(layer_mode, opacity);
    }

    fn get_layer_opacity(&self, layer: u8) -> f32 {
        let layer_mode = u8_to_layer_mode(layer);
        self.get_layer_opacity(layer_mode)
    }

    fn set_layer_enabled(&mut self, layer: u8, enabled: bool) {
        let layer_mode = u8_to_layer_mode(layer);
        self.set_layer_enabled(layer_mode, enabled);
    }

    fn is_layer_enabled(&self, layer: u8) -> bool {
        let layer_mode = u8_to_layer_mode(layer);
        self.is_layer_enabled(layer_mode)
    }
}

/// Convert u8 layer index to LayerMode enum.
fn u8_to_layer_mode(layer: u8) -> LayerMode {
    match layer {
        0 => LayerMode::Points,
        1 => LayerMode::Gaussians,
        2 => LayerMode::Triangles,
        _ => {
            tracing::warn!("Invalid layer index: {}, defaulting to Points", layer);
            LayerMode::Points
        }
    }
}
