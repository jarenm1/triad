//! Example demonstrating two sequential compute passes using triad-gpu.
//!
//! This example:
//! 1. Creates input data (array of floats)
//! 2. Runs first compute pass: doubles each value
//! 3. Runs second compute pass: squares each value
//! 4. Reads back results and prints them
//!
//! Run with: `cargo run --example compute`

use pollster::FutureExt;
use triad_gpu::{
    BufferUsage, FrameGraph, Handle, Pass, PassBuilder, PassContext, ResourceRegistry, Renderer,
};
use triad_gpu::wgpu;

// Inline compute shader that doubles values
const DOUBLE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= arrayLength(&input) || idx >= arrayLength(&output)) {
        return;
    }
    
    // Double the input value
    output[idx] = input[idx] * 2.0;
}
"#;

// Inline compute shader that squares values
const SQUARE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= arrayLength(&input) || idx >= arrayLength(&output)) {
        return;
    }
    
    // Square the input value
    output[idx] = input[idx] * input[idx];
}
"#;

/// First compute pass: doubles values
struct DoubleComputePass {
    compute_pipeline: Handle<wgpu::ComputePipeline>,
    bind_group: Handle<wgpu::BindGroup>,
    element_count: u32,
}

impl DoubleComputePass {
    fn new(
        compute_pipeline: Handle<wgpu::ComputePipeline>,
        bind_group: Handle<wgpu::BindGroup>,
        element_count: u32,
    ) -> Self {
        Self {
            compute_pipeline,
            bind_group,
            element_count,
        }
    }
}

impl Pass for DoubleComputePass {
    fn name(&self) -> &str {
        "DoubleComputePass"
    }

    fn execute(&self, ctx: &PassContext) -> wgpu::CommandBuffer {
        let mut encoder = ctx.create_command_encoder(Some("Double Compute Encoder"));

        let pipeline = ctx
            .get_compute_pipeline(self.compute_pipeline)
            .expect("compute pipeline");
        let bind_group = ctx.get_bind_group(self.bind_group).expect("bind group");

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Double Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, bind_group, &[]);

        // Dispatch one workgroup per 64 elements
        let workgroup_count = (self.element_count + 63) / 64;
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);

        drop(compute_pass);
        encoder.finish()
    }
}

/// Second compute pass: squares values
struct SquareComputePass {
    compute_pipeline: Handle<wgpu::ComputePipeline>,
    bind_group: Handle<wgpu::BindGroup>,
    element_count: u32,
}

impl SquareComputePass {
    fn new(
        compute_pipeline: Handle<wgpu::ComputePipeline>,
        bind_group: Handle<wgpu::BindGroup>,
        element_count: u32,
    ) -> Self {
        Self {
            compute_pipeline,
            bind_group,
            element_count,
        }
    }
}

impl Pass for SquareComputePass {
    fn name(&self) -> &str {
        "SquareComputePass"
    }

    fn execute(&self, ctx: &PassContext) -> wgpu::CommandBuffer {
        let mut encoder = ctx.create_command_encoder(Some("Square Compute Encoder"));

        let pipeline = ctx
            .get_compute_pipeline(self.compute_pipeline)
            .expect("compute pipeline");
        let bind_group = ctx.get_bind_group(self.bind_group).expect("bind group");

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Square Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, bind_group, &[]);

        // Dispatch one workgroup per 64 elements
        let workgroup_count = (self.element_count + 63) / 64;
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);

        drop(compute_pass);
        encoder.finish()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("Creating renderer...");
    let renderer = Renderer::new().block_on()?;
    let device = renderer.device();
    let queue = renderer.queue();
    let mut registry = ResourceRegistry::default();

    // Create input data: [1.0, 2.0, 3.0, 4.0, 5.0]
    const ELEMENT_COUNT: usize = 5;
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    println!("Input data: {:?}", input_data);

    // Create buffers
    println!("Creating buffers...");
    let input_buffer = renderer
        .create_buffer()
        .label("Input Buffer")
        .with_pod_data(&input_data)
        .usage(BufferUsage::Storage { read_only: true })
        .build(&mut registry)?;

    let intermediate_buffer = renderer
        .create_buffer()
        .label("Intermediate Buffer")
        .size((ELEMENT_COUNT * std::mem::size_of::<f32>()) as u64)
        .usage(BufferUsage::Storage { read_only: false })
        .build(&mut registry)?;

    let output_buffer = renderer
        .create_buffer()
        .label("Output Buffer")
        .size((ELEMENT_COUNT * std::mem::size_of::<f32>()) as u64)
        .usage(BufferUsage::Storage { read_only: false })
        .build(&mut registry)?;

    // Create compute shaders
    println!("Creating compute shaders...");
    let double_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("double_shader"),
        source: wgpu::ShaderSource::Wgsl(DOUBLE_SHADER.into()),
    });

    let square_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("square_shader"),
        source: wgpu::ShaderSource::Wgsl(SQUARE_SHADER.into()),
    });

    // Create bind group layouts
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Compute Bind Group Layout"),
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
        ],
    });

    // Create bind groups
    let double_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Double Compute Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: registry.get(input_buffer).unwrap().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: registry.get(intermediate_buffer).unwrap().as_entire_binding(),
            },
        ],
    });

    let square_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Square Compute Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: registry.get(intermediate_buffer).unwrap().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: registry.get(output_buffer).unwrap().as_entire_binding(),
            },
        ],
    });

    // Create compute pipelines
    println!("Creating compute pipelines...");
    let double_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Double Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let double_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Double Compute Pipeline"),
        layout: Some(&double_pipeline_layout),
        module: &double_shader,
        entry_point: Some("cs_main"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });

    let square_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Square Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let square_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Square Compute Pipeline"),
        layout: Some(&square_pipeline_layout),
        module: &square_shader,
        entry_point: Some("cs_main"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });

    // Register resources in registry
    let double_pipeline_handle: Handle<wgpu::ComputePipeline> = registry.insert(double_pipeline);
    let square_pipeline_handle: Handle<wgpu::ComputePipeline> = registry.insert(square_pipeline);
    let double_bind_group_handle: Handle<wgpu::BindGroup> = registry.insert(double_bind_group);
    let square_bind_group_handle: Handle<wgpu::BindGroup> = registry.insert(square_bind_group);

    // Build frame graph with two sequential compute passes
    println!("Building frame graph...");
    let mut frame_graph = FrameGraph::default();

    // Register all resources (only buffers/textures need registration, not pipelines/bind groups)
    frame_graph
        .register_resource(input_buffer)
        .register_resource(intermediate_buffer)
        .register_resource(output_buffer);

    // Add first compute pass: input -> intermediate (doubles values)
    let mut double_pass_builder = PassBuilder::new("DoubleComputePass");
    double_pass_builder
        .read(input_buffer)
        .write(intermediate_buffer);
    let double_pass = double_pass_builder.with_pass(Box::new(DoubleComputePass::new(
        double_pipeline_handle,
        double_bind_group_handle,
        ELEMENT_COUNT as u32,
    )));
    frame_graph.add_pass(double_pass);

    // Add second compute pass: intermediate -> output (squares values)
    let mut square_pass_builder = PassBuilder::new("SquareComputePass");
    square_pass_builder
        .read(intermediate_buffer)
        .write(output_buffer);
    let square_pass = square_pass_builder.with_pass(Box::new(SquareComputePass::new(
        square_pipeline_handle,
        square_bind_group_handle,
        ELEMENT_COUNT as u32,
    )));
    frame_graph.add_pass(square_pass);

    // Build and execute
    println!("Executing compute passes...");
    let mut executable_graph = frame_graph.build()?;
    executable_graph.execute(device, queue, &registry);

    // Read back results using a staging buffer
    println!("Reading back results...");
    let output_buffer_ref = registry.get(output_buffer).unwrap();
    let buffer_size = output_buffer_ref.size();
    
    // Create a staging buffer to read back results
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Copy from output buffer to staging buffer
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Copy Encoder"),
    });
    encoder.copy_buffer_to_buffer(
        output_buffer_ref,
        0,
        &staging_buffer,
        0,
        buffer_size,
    );
    queue.submit(Some(encoder.finish()));

    // Map the staging buffer and read data
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });

    // Poll device until mapping is complete
    // Note: On native platforms, we need to poll the device to process callbacks
    // On web, this is handled automatically
    // For wgpu 27, we use a simple polling loop
    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(5);
    loop {
        // Process any pending callbacks - device.poll() processes callbacks
        let _ = device.poll(wgpu::PollType::Poll);
        if let Ok(result) = receiver.try_recv() {
            result?;
            break;
        }
        if start.elapsed() > timeout {
            return Err("Timeout waiting for buffer mapping".into());
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    let output_data = buffer_slice.get_mapped_range();
    let output_values: &[f32] = bytemuck::cast_slice(&output_data);
    let output_vec: Vec<f32> = output_values.to_vec();
    drop(output_data);
    staging_buffer.unmap();

    println!("\nResults:");
    println!("  Input:        {:?}", input_data);
    println!("  After double: {:?}", 
        input_data.iter().map(|x| x * 2.0).collect::<Vec<_>>());
    println!("  After square: {:?}", output_vec);
    println!("\nExpected: Each value should be (input * 2)^2");
    println!("  [1.0, 2.0, 3.0, 4.0, 5.0] -> [4.0, 16.0, 36.0, 64.0, 100.0]");

    Ok(())
}
