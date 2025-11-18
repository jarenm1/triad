use glam::{Mat4, Vec3};
use tracing::{Level, debug, error, info, instrument, span, trace, warn};
use triad_gpu::{
    CameraUniforms, FrameGraph, GaussianPoint, Handle, Pass, PassBuilder, PassContext,
    RenderPipelineBuilder, Renderer, ResourceRegistry, ShaderManager, ply_loader,
};
use wgpu::util::DeviceExt;

// Gaussian splatting render pass
struct GaussianRenderPass {
    pipeline_handle: Handle<wgpu::RenderPipeline>,
    bind_group_handle: Handle<wgpu::BindGroup>,
    index_buffer_handle: Handle<wgpu::Buffer>,
    gaussian_count: u32,
    output_texture_handle: Handle<wgpu::Texture>,
}

impl Pass for GaussianRenderPass {
    fn name(&self) -> &str {
        "gaussian_splatting"
    }

    #[instrument(skip(self, ctx), fields(pass = self.name(), gaussian_count = self.gaussian_count))]
    fn execute(&self, ctx: &PassContext) -> wgpu::CommandBuffer {
        debug!("Executing GaussianRenderPass");

        let mut encoder = ctx.create_command_encoder(Some(self.name()));

        // Get resources
        let pipeline = ctx
            .get_render_pipeline(self.pipeline_handle.clone())
            .expect("Pipeline not found");
        trace!("Pipeline retrieved");

        let bind_group = ctx
            .get_bind_group(self.bind_group_handle.clone())
            .expect("Bind group not found");
        trace!("Bind group retrieved");

        let output_texture = ctx
            .get_texture(self.output_texture_handle.clone())
            .expect("Output texture not found");
        debug!(
            "Output texture: {}x{}",
            output_texture.width(),
            output_texture.height()
        );

        // Create render pass
        let view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Gaussian Splatting Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 1.0,
                        g: 1.0,
                        b: 1.0,
                        a: 1.0,
                    }), // White background to check for lighting issues
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        // Render Gaussians
        let index_buffer = ctx
            .get_buffer(self.index_buffer_handle.clone())
            .expect("Index buffer not found");
        trace!("Index buffer retrieved");

        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);

        // Each Gaussian is a single triangle (3 indices)
        let index_count = self.gaussian_count * 3;
        debug!(
            "Drawing {} indices ({} triangles, {} Gaussians)",
            index_count, self.gaussian_count, self.gaussian_count
        );

        // Validate index count before drawing
        let index_buffer_size = index_buffer.size();
        let max_indices = (index_buffer_size / 4) as u32; // Each index is u32 (4 bytes)

        debug!("Index buffer validation:");
        debug!("  Buffer size: {} bytes", index_buffer_size);
        debug!("  Max indices: {}", max_indices);
        debug!("  Requested indices: {}", index_count);
        debug!("  Requested Gaussians: {}", self.gaussian_count);

        if index_count > max_indices {
            error!(
                "Index count {} exceeds buffer size {} (max {} indices). This will cause rendering failures!",
                index_count, index_buffer_size, max_indices
            );
            // Don't draw if we know it will fail
            drop(render_pass);
            let command_buffer = encoder.finish();
            return command_buffer;
        }

        // Additional validation: check if we're trying to access beyond array bounds
        // The shader checks gaussian_idx >= arrayLength(&gaussians), but we should validate here too
        let max_gaussians_by_indices = max_indices / 3; // 3 indices per Gaussian
        if self.gaussian_count > max_gaussians_by_indices {
            warn!(
                "Gaussian count {} exceeds max {} based on index buffer ({} indices / 3)",
                self.gaussian_count, max_gaussians_by_indices, max_indices
            );
        }

        render_pass.draw_indexed(0..index_count, 0, 0..1);

        drop(render_pass);

        let command_buffer = encoder.finish();
        trace!("Command buffer created");
        command_buffer
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing subscriber
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("debug")),
        )
        .with_target(false)
        .with_thread_ids(true)
        .with_thread_names(true)
        .init();

    info!("Initializing renderer...");
    let renderer = pollster::block_on(Renderer::new())?;

    let device = renderer.device();
    let queue = renderer.queue();

    // Set up GPU error callback to capture validation errors and out-of-bounds access
    use std::sync::Arc;
    device.on_uncaptured_error(Arc::new(|err: wgpu::Error| {
        error!("GPU Error: {:?}", err);
        // Log detailed error information
        let error_str = format!("{:?}", err);
        if error_str.contains("validation") || error_str.contains("Validation") {
            error!(
                "GPU Validation Error detected - this may indicate buffer overruns or invalid state"
            );
        }
        if error_str.contains("memory") || error_str.contains("Memory") {
            error!("GPU Memory Error detected - buffer may be too large");
        }
    }));

    debug!("Device and queue obtained");

    // Log GPU limits for debugging
    let limits = device.limits();
    info!("GPU Limits:");
    info!(
        "  Max storage buffer binding size: {} bytes ({:.2} MB)",
        limits.max_storage_buffer_binding_size,
        limits.max_storage_buffer_binding_size as f64 / (1024.0 * 1024.0)
    );
    info!(
        "  Max buffer size: {} bytes ({:.2} MB)",
        limits.max_buffer_size,
        limits.max_buffer_size as f64 / (1024.0 * 1024.0)
    );
    info!(
        "  Max compute workgroup storage size: {} bytes",
        limits.max_compute_workgroup_storage_size
    );
    info!(
        "  Max storage buffers per shader stage: {}",
        limits.max_storage_buffers_per_shader_stage
    );

    // Create resource registry
    let mut registry = ResourceRegistry::new();
    let mut shader_manager = ShaderManager::new();

    // Load Gaussian data
    let ply_path = std::env::args()
        .nth(1)
        .ok_or("Please provide a PLY file path as argument")?;
    info!("Loading Gaussians from PLY file: {}", ply_path);
    let gaussians: Vec<GaussianPoint> = ply_loader::load_gaussians_from_ply(&ply_path)?;
    let gaussian_count = gaussians.len() as u32;
    info!("Loaded {} Gaussians", gaussian_count);

    // Calculate bounding box to position camera correctly
    let (bbox_min, bbox_max, bbox_center, bbox_size) = calculate_bounding_box(&gaussians);
    info!("Bounding box: min={:?}, max={:?}", bbox_min, bbox_max);
    info!("Center: {:?}, Size: {:?}", bbox_center, bbox_size);

    for (i, g) in gaussians.iter().take(5).enumerate() {
        info!(
            "Gaussian {}: pos=({:.3}, {:.3}, {:.3}), radius={:.4}, color=({:.3}, {:.3}, {:.3}), opacity={:.3}",
            i,
            g.position_radius[0],
            g.position_radius[1],
            g.position_radius[2],
            g.position_radius[3],
            g.color_opacity[0],
            g.color_opacity[1],
            g.color_opacity[2],
            g.color_opacity[3]
        );
    }

    // Create Gaussian buffer
    let gaussian_buffer_handle = Handle::next();
    let gaussian_data = bytemuck::cast_slice(&gaussians);
    let gaussian_buffer_size = gaussian_data.len() as u64;

    info!(
        "Gaussian buffer size: {} bytes ({} Gaussians)",
        gaussian_buffer_size, gaussian_count
    );

    // Validate buffer size against GPU limits
    let limits = device.limits();
    if gaussian_buffer_size > limits.max_storage_buffer_binding_size as u64 {
        return Err(format!(
            "Gaussian buffer size {} exceeds GPU limit {} bytes. Try reducing the number of Gaussians or split into multiple buffers.",
            gaussian_buffer_size, limits.max_storage_buffer_binding_size
        ).into());
    }

    if gaussian_buffer_size > limits.max_buffer_size as u64 {
        return Err(format!(
            "Gaussian buffer size {} exceeds GPU max buffer size {} bytes",
            gaussian_buffer_size, limits.max_buffer_size
        )
        .into());
    }

    // Validate array length (storage buffers have array length limits)
    // Check actual size and alignment of GaussianPoint
    let gaussian_size = std::mem::size_of::<GaussianPoint>() as u64;
    let gaussian_align = std::mem::align_of::<GaussianPoint>() as u64;
    info!(
        "GaussianPoint size: {} bytes, alignment: {} bytes",
        gaussian_size, gaussian_align
    );

    // Verify the buffer size matches expected
    let expected_buffer_size = gaussian_count as u64 * gaussian_size;
    if gaussian_buffer_size != expected_buffer_size {
        warn!(
            "Buffer size mismatch: expected {} bytes ({} * {}), got {} bytes",
            expected_buffer_size, gaussian_count, gaussian_size, gaussian_buffer_size
        );
    }

    let max_gaussians_by_size = (limits.max_storage_buffer_binding_size as u64) / gaussian_size;
    info!(
        "Max Gaussians by buffer size: {} (Gaussian size: {} bytes)",
        max_gaussians_by_size, gaussian_size
    );

    if gaussian_count as u64 > max_gaussians_by_size {
        warn!(
            "Gaussian count {} exceeds estimated max {} based on buffer size limits",
            gaussian_count, max_gaussians_by_size
        );
    }

    // Log sample data to verify colors are correct
    if gaussian_count > 0 {
        let sample_idx = (gaussian_count / 2).min(100); // Sample middle or first 100
        let sample = &gaussians[sample_idx as usize];
        info!(
            "Sample Gaussian [{}]: pos=({:.3}, {:.3}, {:.3}), radius={:.4}, color=({:.3}, {:.3}, {:.3}), opacity={:.3}",
            sample_idx,
            sample.position_radius[0],
            sample.position_radius[1],
            sample.position_radius[2],
            sample.position_radius[3],
            sample.color_opacity[0],
            sample.color_opacity[1],
            sample.color_opacity[2],
            sample.color_opacity[3]
        );
    }

    let gaussian_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Gaussian Buffer"),
        contents: gaussian_data,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Verify the buffer was created correctly
    let actual_buffer_size = gaussian_buffer.size();
    info!(
        "Gaussian buffer created: {} bytes (expected {} bytes)",
        actual_buffer_size, gaussian_buffer_size
    );

    if actual_buffer_size < gaussian_buffer_size {
        warn!(
            "Buffer size mismatch: expected {} bytes, got {} bytes",
            gaussian_buffer_size, actual_buffer_size
        );
    }

    registry.register_buffer(gaussian_buffer_handle.clone(), gaussian_buffer);

    info!("Gaussian buffer registered successfully");

    // Create index buffer for triangle list rendering
    // Each Gaussian is rendered as a single triangle (3 vertices, 3 indices)
    // For Gaussian i: base = i * 3, indices = [base, base+1, base+2]
    let mut indices = Vec::with_capacity((gaussian_count * 3) as usize);
    for i in 0..gaussian_count {
        let base = i * 3;
        // Single triangle: 0, 1, 2
        indices.push(base);
        indices.push(base + 1);
        indices.push(base + 2);
    }

    let index_buffer_handle = Handle::next();
    let index_data = bytemuck::cast_slice(&indices);
    let index_buffer_size = index_data.len() as u64;

    info!(
        "Index buffer size: {} bytes ({} indices)",
        index_buffer_size,
        indices.len()
    );

    // Validate index buffer size
    let limits = device.limits();
    if index_buffer_size > limits.max_buffer_size as u64 {
        return Err(format!(
            "Index buffer size {} exceeds GPU max buffer size {} bytes",
            index_buffer_size, limits.max_buffer_size
        )
        .into());
    }

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Index Buffer"),
        contents: index_data,
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
    });
    registry.register_buffer(index_buffer_handle.clone(), index_buffer);

    // Verify actual buffer size matches expected
    let actual_index_buffer_size = registry
        .get_buffer(index_buffer_handle.clone())
        .unwrap()
        .size();
    info!(
        "Created index buffer: {} bytes (expected {} bytes)",
        actual_index_buffer_size, index_buffer_size
    );

    if actual_index_buffer_size < index_buffer_size {
        warn!(
            "Index buffer size mismatch: expected {} bytes, got {} bytes",
            index_buffer_size, actual_index_buffer_size
        );
    }

    // Create camera uniforms
    info!("Setting up camera...");
    let camera_uniforms = create_camera_uniforms_from_bbox(bbox_center, bbox_size);
    let camera_buffer_handle = Handle::next();
    let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Camera Buffer"),
        contents: bytemuck::cast_slice(&[camera_uniforms]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    registry.register_buffer(camera_buffer_handle.clone(), camera_buffer);

    // Create output texture (headless rendering)
    println!("Creating output texture...");
    let width = 1920u32;
    let height = 1080u32;
    let output_texture_handle = Handle::next();
    let output_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Output Texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    registry.register_texture(output_texture_handle.clone(), output_texture);

    // Load shaders
    let shader_span = span!(Level::INFO, "shader_loading");
    let _shader_guard = shader_span.enter();
    info!("Loading shaders...");

    let vertex_shader_source = include_str!("../shaders/gaussian_vertex.wgsl");
    let fragment_shader_source = include_str!("../shaders/gaussian_fragment.wgsl");

    debug!("Vertex shader length: {} bytes", vertex_shader_source.len());
    debug!(
        "Fragment shader length: {} bytes",
        fragment_shader_source.len()
    );

    let vertex_shader =
        shader_manager.create_shader(device, Some("gaussian_vs"), vertex_shader_source);
    trace!("Vertex shader created: handle id {}", vertex_shader.id());

    let fragment_shader =
        shader_manager.create_shader(device, Some("gaussian_fs"), fragment_shader_source);
    trace!(
        "Fragment shader created: handle id {}",
        fragment_shader.id()
    );

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Gaussian Bind Group Layout"),
        entries: &[
            // Gaussian storage buffer
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Camera uniform buffer
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        std::num::NonZero::new(std::mem::size_of::<CameraUniforms>() as u64)
                            .unwrap(),
                    ),
                },
                count: None,
            },
        ],
    });

    // Create bind group
    let bind_group_handle = Handle::next();
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Gaussian Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: registry
                    .get_buffer(gaussian_buffer_handle.clone())
                    .unwrap()
                    .as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: registry
                    .get_buffer(camera_buffer_handle.clone())
                    .unwrap()
                    .as_entire_binding(),
            },
        ],
    });
    registry.register_bind_group(bind_group_handle.clone(), bind_group);

    // Create pipeline layout
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Gaussian Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // Create render pipeline
    let pipeline_span = span!(Level::INFO, "pipeline_creation");
    let _pipeline_guard = pipeline_span.enter();
    info!("Creating render pipeline...");
    let pipeline_handle = RenderPipelineBuilder::new(device, &shader_manager)
        .with_label("Gaussian Splatting Pipeline")
        .with_vertex_shader(vertex_shader)
        .with_fragment_shader(fragment_shader)
        .with_layout(pipeline_layout)
        .with_primitive(wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None, // No culling for Gaussian splatting
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        })
        .with_fragment_target(Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba8Unorm,
            // Standard alpha blending for Gaussian splatting
            // Formula: src_color * src_alpha + dst_color * (1 - src_alpha)
            blend: Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
            write_mask: wgpu::ColorWrites::ALL,
        }))
        .build(&mut registry)?;
    debug!(
        "Render pipeline created: handle id {}",
        pipeline_handle.id()
    );

    // Create frame graph
    info!("Building frame graph...");
    let mut frame_graph = FrameGraph::default();

    // Register resources
    frame_graph.register_resource(gaussian_buffer_handle.clone());
    frame_graph.register_resource(camera_buffer_handle.clone());
    frame_graph.register_resource(output_texture_handle.clone());
    frame_graph.register_resource(pipeline_handle.clone());
    frame_graph.register_resource(bind_group_handle.clone());
    frame_graph.register_resource(index_buffer_handle.clone());

    // Add render pass
    let mut pass_builder = PassBuilder::new("gaussian_splatting");
    pass_builder
        .read(gaussian_buffer_handle.clone())
        .read(camera_buffer_handle.clone())
        .read(pipeline_handle.clone())
        .read(bind_group_handle.clone())
        .read(index_buffer_handle.clone())
        .write(output_texture_handle.clone());

    let render_pass = GaussianRenderPass {
        pipeline_handle: pipeline_handle.clone(),
        bind_group_handle: bind_group_handle.clone(),
        index_buffer_handle: index_buffer_handle.clone(),
        gaussian_count,
        output_texture_handle: output_texture_handle.clone(),
    };

    frame_graph.add_pass(pass_builder.with_pass(Box::new(render_pass)));

    // Build and execute
    let exec_span = span!(Level::INFO, "frame_graph_execution");
    let _exec_guard = exec_span.enter();
    info!("Executing frame graph...");

    let mut executable = frame_graph.build()?;
    debug!("Frame graph built successfully");

    executable.execute(device, queue, &registry);

    // Wait for GPU to finish - use a more reliable approach
    debug!("Waiting for GPU to finish...");
    // Create a simple fence/staging buffer to ensure GPU work completes
    // Submit an empty command buffer and wait for it to ensure previous work is done
    let sync_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Sync Encoder"),
    });
    let sync_buffer = sync_encoder.finish();
    let sync_index = queue.submit(Some(sync_buffer));

    // Now wait for this submission to complete, which ensures all previous work is done
    // The sync submission acts as a fence - when it completes, all previous work is done
    // Wait indefinitely until the sync submission completes
    loop {
        match device.poll(wgpu::PollType::Wait {
            submission_index: Some(sync_index.clone()),
            timeout: Some(std::time::Duration::from_millis(100)),
        }) {
            Ok(_) => break,     // Work is complete
            Err(_) => continue, // Keep waiting
        }
    }

    // Read back texture and save as PNG
    info!("Reading back rendered image...");
    save_texture_to_png(
        device,
        queue,
        &registry,
        output_texture_handle.clone(),
        width,
        height,
    )?;

    info!("Rendering complete! Check output.png");

    Ok(())
}

fn calculate_bounding_box(gaussians: &[GaussianPoint]) -> (Vec3, Vec3, Vec3, Vec3) {
    if gaussians.is_empty() {
        return (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
    }

    let mut min = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
    let mut max = Vec3::new(f32::MIN, f32::MIN, f32::MIN);

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
    let size = max - min;

    (min, max, center, size)
}

#[instrument]
fn create_camera_uniforms() -> CameraUniforms {
    // Camera positioned in front of Gaussians, looking at them
    // Gaussians are at z=-5, so camera should be at z=0 looking at z=-5
    let eye = Vec3::new(0.0, 0.0, 0.0);
    let target = Vec3::new(0.0, 0.0, -5.0);
    let up = Vec3::new(0.0, 1.0, 0.0);

    let view_matrix = Mat4::look_at_rh(eye, target, up);

    let aspect = 1920.0 / 1080.0;
    let fov = std::f32::consts::PI / 4.0; // 45 degrees
    let proj_matrix = Mat4::perspective_rh(fov, aspect, 0.1, 100.0);

    CameraUniforms {
        view_matrix: view_matrix.to_cols_array_2d(),
        proj_matrix: proj_matrix.to_cols_array_2d(),
        view_pos: [eye.x, eye.y, eye.z],
        _padding: 0.0,
    }
}

#[instrument]
fn create_camera_uniforms_from_bbox(center: Vec3, size: Vec3) -> CameraUniforms {
    // Calculate distance to view the entire bounding box
    // We want to position the camera so the bounding box fits in view
    let max_dim = size.x.max(size.y).max(size.z);

    // Position camera at a distance that shows the entire object
    // Use a distance of about 2.5x the max dimension to ensure nothing is cut off
    let distance = max_dim * 2.5;

    // Position camera looking at the center, offset along +Z axis
    let eye = center + Vec3::new(0.0, 0.0, distance);
    let target = center;
    let up = Vec3::new(0.0, 1.0, 0.0);

    let view_matrix = Mat4::look_at_rh(eye, target, up);

    let aspect = 1920.0 / 1080.0;
    let fov = std::f32::consts::PI / 4.0; // 45 degrees

    // Adjust near/far planes based on bounding box
    // Make sure we can see the entire model - use larger margins
    let near = 0.01; // Fixed near plane
    let far = distance * 5.0; // Far enough to see everything with margin

    let proj_matrix = Mat4::perspective_rh(fov, aspect, near, far);

    info!(
        "Camera: eye={:?}, target={:?}, distance={:.2}, near={:.2}, far={:.2}",
        eye, target, distance, near, far
    );

    CameraUniforms {
        view_matrix: view_matrix.to_cols_array_2d(),
        proj_matrix: proj_matrix.to_cols_array_2d(),
        view_pos: [eye.x, eye.y, eye.z],
        _padding: 0.0,
    }
}

#[instrument(skip(device, queue, registry), fields(width, height))]
fn save_texture_to_png(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    registry: &ResourceRegistry,
    texture_handle: Handle<wgpu::Texture>,
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let texture = registry.get_texture(texture_handle).unwrap();

    // Create a buffer to read the texture into
    let buffer_size = (width * height * 4) as u64; // RGBA
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Copy texture to buffer
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Copy Encoder"),
    });

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(Some(encoder.finish()));

    // Map buffer and read data using pollster-style polling
    let buffer_slice = output_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });

    // Poll device until buffer is ready (using wgpu's polling mechanism)
    // Use Wait instead of Poll to ensure we actually wait for completion
    loop {
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_millis(100)),
        });
        match receiver.try_recv() {
            Ok(result) => {
                result?;
                break;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                // Continue polling
            }
            Err(e) => return Err(Box::new(e)),
        }
    }

    // Read pixel data
    let data = buffer_slice.get_mapped_range();
    let pixels: Vec<u8> = data.iter().copied().collect();
    drop(data);
    output_buffer.unmap();

    // Convert to image and save directly - Rgba8Unorm is already in the correct format
    let img = image::RgbaImage::from_raw(width, height, pixels)
        .ok_or("Failed to create image from pixel data")?;
    img.save("output.png")?;

    Ok(())
}
