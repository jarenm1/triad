//! High-level builder APIs for creating GPU resources
//!
//! These builders provide a simpler, more ergonomic API compared to
//! directly using wgpu descriptors.

use std::borrow::Cow;
use std::marker::PhantomData;

use crate::Renderer;
use crate::error::{BindGroupError, BufferError, PipelineError, ShaderError};
use crate::frame_graph::resource::Handle;
use crate::resource_registry::ResourceRegistry;

/// Buffer usage flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferUsage {
    /// Vertex buffer
    Vertex,
    /// Index buffer
    Index,
    /// Uniform buffer
    Uniform,
    /// Storage buffer (read-only or read-write)
    Storage { read_only: bool },
    /// Storage buffer with CPU write access (STORAGE | COPY_DST)
    /// Use this for buffers that need incremental updates via write_buffer_offset
    StorageWritable,
    /// Copy source
    CopySrc,
    /// Copy destination
    CopyDst,
    /// Indirect draw/dispatch arguments
    Indirect,
    /// CPU-readable staging buffer
    Readback,
    /// CPU-writable staging buffer
    Upload,
}

impl BufferUsage {
    fn to_wgpu(&self) -> wgpu::BufferUsages {
        match self {
            BufferUsage::Vertex => wgpu::BufferUsages::VERTEX,
            BufferUsage::Index => wgpu::BufferUsages::INDEX,
            BufferUsage::Uniform => wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            BufferUsage::Storage { read_only: _ } => wgpu::BufferUsages::STORAGE,
            BufferUsage::StorageWritable => {
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
            }
            BufferUsage::CopySrc => wgpu::BufferUsages::COPY_SRC,
            BufferUsage::CopyDst => wgpu::BufferUsages::COPY_DST,
            BufferUsage::Indirect => wgpu::BufferUsages::INDIRECT,
            BufferUsage::Readback => wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            BufferUsage::Upload => wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
        }
    }
}

/// Builder for creating GPU buffers
pub struct BufferBuilder<'a> {
    device: &'a wgpu::Device,
    label: Option<String>,
    size: Option<u64>,
    data: Option<&'a [u8]>,
    usage: wgpu::BufferUsages,
}

impl<'a> BufferBuilder<'a> {
    pub(crate) fn new(device: &'a wgpu::Device) -> Self {
        Self {
            device,
            label: None,
            size: None,
            data: None,
            usage: BufferUsage::Vertex.to_wgpu(),
        }
    }

    /// Set the buffer label
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set buffer size (for empty buffers)
    pub fn size(mut self, size: u64) -> Self {
        self.size = Some(size);
        self
    }

    /// Set buffer data (for initialized buffers)
    pub fn with_data(mut self, data: &'a [u8]) -> Self {
        self.data = Some(data);
        self
    }

    /// Set buffer data from a slice of Pod types
    pub fn with_pod_data<T: bytemuck::Pod>(mut self, data: &'a [T]) -> Self {
        self.data = Some(bytemuck::cast_slice(data));
        self
    }

    /// Set buffer usage
    pub fn usage(mut self, usage: BufferUsage) -> Self {
        self.usage = usage.to_wgpu();
        self
    }

    /// Replace the raw usage flags.
    pub fn usage_flags(mut self, usage: wgpu::BufferUsages) -> Self {
        self.usage = usage;
        self
    }

    /// Add an extra usage flag without replacing the existing set.
    pub fn add_usage(mut self, usage: wgpu::BufferUsages) -> Self {
        self.usage |= usage;
        self
    }

    /// Build the buffer and register it in the registry
    pub fn build(
        self,
        registry: &mut ResourceRegistry,
    ) -> Result<Handle<wgpu::Buffer>, BufferError> {
        use wgpu::util::DeviceExt;

        let buffer = if let Some(data) = self.data {
            // Initialize buffer with data
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: self.label.as_deref(),
                    contents: data,
                    usage: self.usage,
                })
        } else if let Some(size) = self.size {
            // Create empty buffer
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: self.label.as_deref(),
                size,
                usage: self.usage,
                mapped_at_creation: false,
            })
        } else {
            return Err(BufferError::MissingSizeOrData);
        };

        Ok(registry.insert(buffer))
    }
}

/// Supported shader source payloads.
pub enum ShaderSource<'a> {
    Wgsl(Cow<'a, str>),
}

/// Builder for creating shader modules from in-memory source.
pub struct ShaderModuleBuilder<'a> {
    device: &'a wgpu::Device,
    label: Option<String>,
    source: Option<ShaderSource<'a>>,
}

impl<'a> ShaderModuleBuilder<'a> {
    pub(crate) fn new(device: &'a wgpu::Device) -> Self {
        Self {
            device,
            label: None,
            source: None,
        }
    }

    /// Set the shader label.
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set WGSL source code for this shader module.
    pub fn with_wgsl_source(mut self, source: impl Into<Cow<'a, str>>) -> Self {
        self.source = Some(ShaderSource::Wgsl(source.into()));
        self
    }

    /// Build the shader module and register it in the registry.
    pub fn build(
        self,
        registry: &mut ResourceRegistry,
    ) -> Result<Handle<wgpu::ShaderModule>, ShaderError> {
        let source = self.source.ok_or(ShaderError::MissingSource)?;

        let module = match source {
            ShaderSource::Wgsl(source) => {
                self.device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: self.label.as_deref(),
                        source: wgpu::ShaderSource::Wgsl(source),
                    })
            }
        };

        Ok(registry.insert(module))
    }
}

/// Typed metadata for a GPU buffer owned by the registry.
#[derive(Debug)]
pub struct GpuBuffer<T: bytemuck::Pod> {
    handle: Handle<wgpu::Buffer>,
    len: usize,
    capacity: usize,
    usage: wgpu::BufferUsages,
    _marker: PhantomData<T>,
}

impl<T: bytemuck::Pod> GpuBuffer<T> {
    /// Returns the underlying buffer handle.
    pub fn handle(&self) -> Handle<wgpu::Buffer> {
        self.handle
    }

    /// Returns the current logical length in elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the total allocated capacity in elements.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns true when the buffer contains no initialized elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the raw wgpu usage flags used to create the buffer.
    pub fn usage(&self) -> wgpu::BufferUsages {
        self.usage
    }

    /// Returns the allocated size in bytes.
    pub fn size_bytes(&self) -> u64 {
        (self.capacity * std::mem::size_of::<T>()) as u64
    }
}

/// Builder for typed GPU buffers intended for storage, uniforms, indirect args, and readback.
pub struct GpuBufferBuilder<'a, T: bytemuck::Pod> {
    device: &'a wgpu::Device,
    label: Option<String>,
    capacity: Option<usize>,
    initial_data: Option<&'a [T]>,
    usage: wgpu::BufferUsages,
    _marker: PhantomData<T>,
}

impl<'a, T: bytemuck::Pod> GpuBufferBuilder<'a, T> {
    pub(crate) fn new(device: &'a wgpu::Device) -> Self {
        Self {
            device,
            label: None,
            capacity: None,
            initial_data: None,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            _marker: PhantomData,
        }
    }

    /// Set the buffer label.
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set total element capacity.
    pub fn capacity(mut self, capacity: usize) -> Self {
        self.capacity = Some(capacity);
        self
    }

    /// Initialize the buffer from a typed slice.
    pub fn with_data(mut self, data: &'a [T]) -> Self {
        self.initial_data = Some(data);
        self
    }

    /// Replace the usage flags with a predefined usage class.
    pub fn usage(mut self, usage: BufferUsage) -> Self {
        self.usage = usage.to_wgpu();
        self
    }

    /// Replace the raw usage flags.
    pub fn usage_flags(mut self, usage: wgpu::BufferUsages) -> Self {
        self.usage = usage;
        self
    }

    /// Add an extra usage flag without replacing the existing set.
    pub fn add_usage(mut self, usage: wgpu::BufferUsages) -> Self {
        self.usage |= usage;
        self
    }

    /// Build the typed buffer and register it.
    pub fn build(self, registry: &mut ResourceRegistry) -> Result<GpuBuffer<T>, BufferError> {
        use wgpu::util::DeviceExt;

        let element_size = std::mem::size_of::<T>();

        let (capacity, len) = match (self.capacity, self.initial_data) {
            (Some(capacity), Some(data)) => (capacity.max(data.len()), data.len()),
            (Some(capacity), None) => (capacity, 0),
            (None, Some(data)) => (data.len(), data.len()),
            (None, None) => return Err(BufferError::MissingSizeOrData),
        };

        let size_bytes = (capacity * element_size) as u64;
        let buffer = if let Some(data) = self.initial_data {
            let mut padded = bytemuck::cast_slice::<T, u8>(data).to_vec();
            padded.resize(size_bytes as usize, 0);

            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: self.label.as_deref(),
                    contents: &padded,
                    usage: self.usage,
                })
        } else {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: self.label.as_deref(),
                size: size_bytes,
                usage: self.usage,
                mapped_at_creation: false,
            })
        };

        Ok(GpuBuffer {
            handle: registry.insert(buffer),
            len,
            capacity,
            usage: self.usage,
            _marker: PhantomData,
        })
    }
}

/// A buffer that supports incremental updates with pre-allocated capacity.
///
/// DynamicBuffer wraps a GPU buffer with tracking for:
/// - Element count vs capacity
/// - Element size for offset calculations
/// - Efficient partial updates without bind group recreation
///
/// The underlying buffer handle remains constant, so bind groups don't need
/// to be recreated when data is updated.
#[derive(Debug)]
pub struct DynamicBuffer<T: bytemuck::Pod> {
    buffer: Handle<wgpu::Buffer>,
    capacity: usize,
    len: usize,
    element_size: usize,
    _marker: PhantomData<T>,
}

impl<T: bytemuck::Pod> DynamicBuffer<T> {
    /// Returns the underlying buffer handle (for bind groups)
    pub fn buffer(&self) -> Handle<wgpu::Buffer> {
        self.buffer
    }

    /// Current number of elements
    pub fn len(&self) -> usize {
        self.len
    }

    /// Maximum capacity in elements
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Update element at index
    pub fn update_at(
        &self,
        renderer: &Renderer,
        registry: &ResourceRegistry,
        index: usize,
        element: &T,
    ) -> Result<(), BufferError> {
        if index >= self.len {
            return Err(BufferError::CapacityExceeded {
                requested: index + 1,
                capacity: self.len,
            });
        }
        let offset = (index * self.element_size) as u64;
        renderer.write_buffer_offset(self.buffer, offset, std::slice::from_ref(element), registry)
    }

    /// Update range of elements starting at index
    pub fn update_range(
        &mut self,
        renderer: &Renderer,
        registry: &ResourceRegistry,
        start_index: usize,
        elements: &[T],
    ) -> Result<(), BufferError> {
        let end_index = start_index + elements.len();
        if end_index > self.capacity {
            return Err(BufferError::CapacityExceeded {
                requested: end_index,
                capacity: self.capacity,
            });
        }
        let offset = (start_index * self.element_size) as u64;
        renderer.write_buffer_offset(self.buffer, offset, elements, registry)?;
        // Extend len if we wrote past current end
        if end_index > self.len {
            self.len = end_index;
        }
        Ok(())
    }

    /// Push elements to the end, returns start index
    pub fn push(
        &mut self,
        renderer: &Renderer,
        registry: &ResourceRegistry,
        elements: &[T],
    ) -> Result<usize, BufferError> {
        let start_index = self.len;
        self.update_range(renderer, registry, start_index, elements)?;
        Ok(start_index)
    }

    /// Clear (logical only, doesn't zero memory)
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Set length (for cases where GPU compute modified data)
    pub fn set_len(&mut self, len: usize) -> Result<(), BufferError> {
        if len > self.capacity {
            return Err(BufferError::CapacityExceeded {
                requested: len,
                capacity: self.capacity,
            });
        }
        self.len = len;
        Ok(())
    }
}

/// Builder for creating DynamicBuffer instances
pub struct DynamicBufferBuilder<'a, T: bytemuck::Pod> {
    device: &'a wgpu::Device,
    label: Option<String>,
    capacity: Option<usize>,
    initial_data: Option<&'a [T]>,
    additional_usage: wgpu::BufferUsages,
    _marker: PhantomData<T>,
}

impl<'a, T: bytemuck::Pod> DynamicBufferBuilder<'a, T> {
    pub(crate) fn new(device: &'a wgpu::Device) -> Self {
        Self {
            device,
            label: None,
            capacity: None,
            initial_data: None,
            additional_usage: wgpu::BufferUsages::empty(),
            _marker: PhantomData,
        }
    }

    /// Set the buffer label
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set maximum capacity (number of elements)
    pub fn capacity(mut self, capacity: usize) -> Self {
        self.capacity = Some(capacity);
        self
    }

    /// Initialize with data (capacity defaults to data.len() if not set)
    pub fn with_data(mut self, data: &'a [T]) -> Self {
        self.initial_data = Some(data);
        self
    }

    /// Add extra usage flags on top of STORAGE | COPY_DST.
    pub fn add_usage(mut self, usage: wgpu::BufferUsages) -> Self {
        self.additional_usage |= usage;
        self
    }

    /// Build the DynamicBuffer and register it
    pub fn build(self, registry: &mut ResourceRegistry) -> Result<DynamicBuffer<T>, BufferError> {
        use wgpu::util::DeviceExt;

        let element_size = std::mem::size_of::<T>();

        let (capacity, initial_len) = match (self.capacity, self.initial_data) {
            (Some(cap), Some(data)) => (cap.max(data.len()), data.len()),
            (Some(cap), None) => (cap, 0),
            (None, Some(data)) => (data.len(), data.len()),
            (None, None) => return Err(BufferError::MissingSizeOrData),
        };

        let buffer_size = (capacity * element_size) as u64;

        let buffer = if let Some(data) = self.initial_data {
            // Create with initial data, but allocate full capacity
            let mut padded = bytemuck::cast_slice::<T, u8>(data).to_vec();
            padded.resize(buffer_size as usize, 0);

            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: self.label.as_deref(),
                    contents: &padded,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | self.additional_usage,
                })
        } else {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: self.label.as_deref(),
                size: buffer_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | self.additional_usage,
                mapped_at_creation: false,
            })
        };

        let handle = registry.insert(buffer);

        Ok(DynamicBuffer {
            buffer: handle,
            capacity,
            len: initial_len,
            element_size,
            _marker: PhantomData,
        })
    }
}

/// Builder for creating compute pipelines.
pub struct ComputePipelineBuilder<'a> {
    device: &'a wgpu::Device,
    compute_shader: Option<Handle<wgpu::ShaderModule>>,
    label: Option<String>,
    layout: Option<wgpu::PipelineLayout>,
}

impl<'a> ComputePipelineBuilder<'a> {
    pub(crate) fn new(device: &'a wgpu::Device) -> Self {
        Self {
            device,
            compute_shader: None,
            label: None,
            layout: None,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn with_compute_shader(mut self, shader: Handle<wgpu::ShaderModule>) -> Self {
        self.compute_shader = Some(shader);
        self
    }

    pub fn with_layout(mut self, layout: wgpu::PipelineLayout) -> Self {
        self.layout = Some(layout);
        self
    }

    pub fn build(
        self,
        registry: &mut ResourceRegistry,
    ) -> Result<Handle<wgpu::ComputePipeline>, PipelineError> {
        let compute_handle = self
            .compute_shader
            .ok_or(PipelineError::MissingComputeShader)?;
        let compute_shader = registry
            .get(compute_handle)
            .ok_or(PipelineError::ShaderNotFound)?;

        let pipeline_layout = self.layout.unwrap_or_else(|| {
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[],
                    push_constant_ranges: &[],
                })
        });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: self.label.as_deref(),
                layout: Some(&pipeline_layout),
                module: compute_shader,
                entry_point: Some("cs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Ok(registry.insert(pipeline))
    }
}

/// Binding type for bind groups
#[derive(Debug, Clone, Copy)]
pub enum BindingType {
    Uniform,
    StorageRead,
    StorageWrite,
    Texture {
        sample_type: wgpu::TextureSampleType,
        view_dimension: wgpu::TextureViewDimension,
        multisampled: bool,
    },
    Sampler {
        filtering: bool,
    },
}

impl BindingType {
    fn to_wgpu_binding_type(&self) -> wgpu::BindingType {
        match self {
            BindingType::Uniform => wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            BindingType::StorageRead => wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            BindingType::StorageWrite => wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            BindingType::Texture {
                sample_type,
                view_dimension,
                multisampled,
            } => wgpu::BindingType::Texture {
                sample_type: *sample_type,
                view_dimension: *view_dimension,
                multisampled: *multisampled,
            },
            BindingType::Sampler { filtering } => wgpu::BindingType::Sampler(if *filtering {
                wgpu::SamplerBindingType::Filtering
            } else {
                wgpu::SamplerBindingType::NonFiltering
            }),
        }
    }
}

/// Shader stage visibility
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
    VertexFragment,
    All,
}

impl ShaderStage {
    fn to_wgpu(&self) -> wgpu::ShaderStages {
        match self {
            ShaderStage::Vertex => wgpu::ShaderStages::VERTEX,
            ShaderStage::Fragment => wgpu::ShaderStages::FRAGMENT,
            ShaderStage::Compute => wgpu::ShaderStages::COMPUTE,
            ShaderStage::VertexFragment => {
                wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT
            }
            ShaderStage::All => wgpu::ShaderStages::all(),
        }
    }
}

/// Entry in a bind group layout
struct BindGroupLayoutEntry {
    binding: u32,
    visibility: ShaderStage,
    binding_type: BindingType,
}

/// Builder for creating bind groups
pub struct BindGroupBuilder<'a> {
    device: &'a wgpu::Device,
    label: Option<String>,
    entries: Vec<BindGroupLayoutEntry>,
    // Store handles instead of resources to avoid lifetime issues
    buffer_bindings: Vec<(u32, Handle<wgpu::Buffer>, BindingType)>,
    texture_bindings: Vec<(u32, Handle<wgpu::TextureView>, BindingType)>,
    sampler_bindings: Vec<(u32, Handle<wgpu::Sampler>)>,
}

impl<'a> BindGroupBuilder<'a> {
    pub(crate) fn new(device: &'a wgpu::Device) -> Self {
        Self {
            device,
            label: None,
            entries: Vec::new(),
            buffer_bindings: Vec::new(),
            texture_bindings: Vec::new(),
            sampler_bindings: Vec::new(),
        }
    }

    /// Set the bind group label
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Add a buffer binding
    pub fn buffer(
        mut self,
        binding: u32,
        buffer_handle: Handle<wgpu::Buffer>,
        binding_type: BindingType,
    ) -> Self {
        self.entries.push(BindGroupLayoutEntry {
            binding,
            visibility: ShaderStage::All,
            binding_type,
        });

        self.buffer_bindings
            .push((binding, buffer_handle, binding_type));
        self
    }

    /// Add a buffer binding with explicit shader-stage visibility.
    pub fn buffer_stage(
        mut self,
        binding: u32,
        visibility: ShaderStage,
        buffer_handle: Handle<wgpu::Buffer>,
        binding_type: BindingType,
    ) -> Self {
        self.entries.push(BindGroupLayoutEntry {
            binding,
            visibility,
            binding_type,
        });

        self.buffer_bindings
            .push((binding, buffer_handle, binding_type));
        self
    }

    /// Add a texture binding
    pub fn texture(
        mut self,
        binding: u32,
        texture_view_handle: Handle<wgpu::TextureView>,
        binding_type: BindingType,
    ) -> Self {
        self.entries.push(BindGroupLayoutEntry {
            binding,
            visibility: ShaderStage::All,
            binding_type,
        });

        self.texture_bindings
            .push((binding, texture_view_handle, binding_type));
        self
    }

    /// Add a texture binding with explicit shader-stage visibility.
    pub fn texture_stage(
        mut self,
        binding: u32,
        visibility: ShaderStage,
        texture_view_handle: Handle<wgpu::TextureView>,
        binding_type: BindingType,
    ) -> Self {
        self.entries.push(BindGroupLayoutEntry {
            binding,
            visibility,
            binding_type,
        });

        self.texture_bindings
            .push((binding, texture_view_handle, binding_type));
        self
    }

    /// Add a sampler binding
    pub fn sampler(mut self, binding: u32, sampler_handle: Handle<wgpu::Sampler>) -> Self {
        self.entries.push(BindGroupLayoutEntry {
            binding,
            visibility: ShaderStage::All,
            binding_type: BindingType::Sampler { filtering: true },
        });

        self.sampler_bindings.push((binding, sampler_handle));
        self
    }

    /// Add a sampler binding with explicit shader-stage visibility.
    pub fn sampler_stage(
        mut self,
        binding: u32,
        visibility: ShaderStage,
        sampler_handle: Handle<wgpu::Sampler>,
    ) -> Self {
        self.entries.push(BindGroupLayoutEntry {
            binding,
            visibility,
            binding_type: BindingType::Sampler { filtering: true },
        });

        self.sampler_bindings.push((binding, sampler_handle));
        self
    }

    /// Build the bind group layout and bind group
    pub fn build(
        self,
        registry: &mut ResourceRegistry,
    ) -> Result<(Handle<wgpu::BindGroupLayout>, Handle<wgpu::BindGroup>), BindGroupError> {
        if self.entries.is_empty() {
            return Err(BindGroupError::NoEntries);
        }

        // Create bind group layout
        let layout_entries: Vec<wgpu::BindGroupLayoutEntry> = self
            .entries
            .iter()
            .map(|e| wgpu::BindGroupLayoutEntry {
                binding: e.binding,
                visibility: e.visibility.to_wgpu(),
                ty: e.binding_type.to_wgpu_binding_type(),
                count: None,
            })
            .collect();

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: self
                        .label
                        .as_deref()
                        .map(|l| format!("{} Layout", l))
                        .as_deref(),
                    entries: &layout_entries,
                });
        let layout_handle = registry.insert(bind_group_layout);

        // Build bind group entries from stored handles
        let mut bind_group_entries = Vec::new();

        // Add buffer bindings
        for (binding, buffer_handle, _) in &self.buffer_bindings {
            let buffer = registry
                .get(*buffer_handle)
                .ok_or(BindGroupError::ResourceNotFound { binding: *binding })?;
            bind_group_entries.push(wgpu::BindGroupEntry {
                binding: *binding,
                resource: buffer.as_entire_binding(),
            });
        }

        // Add texture bindings
        for (binding, texture_view_handle, _) in &self.texture_bindings {
            let texture_view = registry
                .get(*texture_view_handle)
                .ok_or(BindGroupError::ResourceNotFound { binding: *binding })?;
            bind_group_entries.push(wgpu::BindGroupEntry {
                binding: *binding,
                resource: wgpu::BindingResource::TextureView(texture_view),
            });
        }

        // Add sampler bindings
        for (binding, sampler_handle) in &self.sampler_bindings {
            let sampler = registry
                .get(*sampler_handle)
                .ok_or(BindGroupError::ResourceNotFound { binding: *binding })?;
            bind_group_entries.push(wgpu::BindGroupEntry {
                binding: *binding,
                resource: wgpu::BindingResource::Sampler(sampler),
            });
        }

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: self.label.as_deref(),
            layout: registry
                .get(layout_handle)
                .ok_or(BindGroupError::LayoutNotFound)?,
            entries: &bind_group_entries,
        });
        let bind_group_handle = registry.insert(bind_group);

        Ok((layout_handle, bind_group_handle))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource_registry::ResourceRegistry;
    use pollster::FutureExt;

    async fn create_test_device() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("Failed to get adapter");
        adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("Failed to get device")
    }

    #[test]
    fn test_buffer_builder_with_data() {
        let (device, _queue) = create_test_device().block_on();
        let mut registry = ResourceRegistry::default();

        let data: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let handle = BufferBuilder::new(&device)
            .label("test_buffer")
            .with_data(&data)
            .usage(BufferUsage::Vertex)
            .build(&mut registry)
            .expect("Failed to build buffer");

        // Verify buffer exists in registry
        let buffer = registry.get(handle).expect("Buffer not found");
        assert_eq!(buffer.size(), 16);
    }

    #[test]
    fn test_buffer_builder_with_size() {
        let (device, _queue) = create_test_device().block_on();
        let mut registry = ResourceRegistry::default();

        let handle = BufferBuilder::new(&device)
            .label("empty_buffer")
            .size(1024)
            .usage(BufferUsage::Uniform)
            .build(&mut registry)
            .expect("Failed to build buffer");

        let buffer = registry.get(handle).expect("Buffer not found");
        assert_eq!(buffer.size(), 1024);
    }

    #[test]
    fn test_buffer_builder_with_pod_data() {
        let (device, _queue) = create_test_device().block_on();
        let mut registry = ResourceRegistry::default();

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct TestVertex {
            x: f32,
            y: f32,
            z: f32,
        }

        let vertices = vec![
            TestVertex {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            TestVertex {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            TestVertex {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
        ];

        let handle = BufferBuilder::new(&device)
            .with_pod_data(&vertices)
            .usage(BufferUsage::Vertex)
            .build(&mut registry)
            .expect("Failed to build buffer");

        let buffer = registry.get(handle).expect("Buffer not found");
        assert_eq!(
            buffer.size(),
            (std::mem::size_of::<TestVertex>() * 3) as u64
        );
    }

    #[test]
    fn test_buffer_builder_missing_size_or_data() {
        let (device, _queue) = create_test_device().block_on();
        let mut registry = ResourceRegistry::default();

        let result = BufferBuilder::new(&device)
            .usage(BufferUsage::Vertex)
            .build(&mut registry);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            BufferError::MissingSizeOrData
        ));
    }

    #[test]
    fn test_buffer_usage_conversion() {
        assert_eq!(BufferUsage::Vertex.to_wgpu(), wgpu::BufferUsages::VERTEX);
        assert_eq!(BufferUsage::Index.to_wgpu(), wgpu::BufferUsages::INDEX);
        assert!(
            BufferUsage::Uniform
                .to_wgpu()
                .contains(wgpu::BufferUsages::UNIFORM)
        );
        assert!(
            BufferUsage::Uniform
                .to_wgpu()
                .contains(wgpu::BufferUsages::COPY_DST)
        );
        assert_eq!(
            BufferUsage::Storage { read_only: true }.to_wgpu(),
            wgpu::BufferUsages::STORAGE
        );
        // StorageWritable should have both STORAGE and COPY_DST
        assert!(
            BufferUsage::StorageWritable
                .to_wgpu()
                .contains(wgpu::BufferUsages::STORAGE)
        );
        assert!(
            BufferUsage::StorageWritable
                .to_wgpu()
                .contains(wgpu::BufferUsages::COPY_DST)
        );
        assert_eq!(
            BufferUsage::Indirect.to_wgpu(),
            wgpu::BufferUsages::INDIRECT
        );
        assert_eq!(
            BufferUsage::Readback.to_wgpu(),
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ
        );
        assert_eq!(
            BufferUsage::Upload.to_wgpu(),
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE
        );
    }

    #[test]
    fn test_buffer_builder_usage_flags() {
        let (device, _queue) = create_test_device().block_on();
        let mut registry = ResourceRegistry::default();

        let handle = BufferBuilder::new(&device)
            .size(16)
            .usage_flags(wgpu::BufferUsages::STORAGE)
            .add_usage(wgpu::BufferUsages::INDIRECT)
            .build(&mut registry)
            .expect("buffer");

        let buffer = registry.get(handle).expect("stored buffer");
        assert_eq!(
            buffer.usage(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT
        );
    }

    #[test]
    fn test_typed_gpu_buffer_builder_with_data_and_capacity() {
        let (device, _queue) = create_test_device().block_on();
        let mut registry = ResourceRegistry::default();

        let data = [1u32, 2u32, 3u32];
        let buffer = GpuBufferBuilder::new(&device)
            .label("particles")
            .with_data(&data)
            .capacity(8)
            .add_usage(wgpu::BufferUsages::INDIRECT)
            .build(&mut registry)
            .expect("typed buffer");

        assert_eq!(buffer.len(), data.len());
        assert_eq!(buffer.capacity(), 8);
        assert_eq!(
            buffer.usage(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::INDIRECT
        );

        let raw = registry.get(buffer.handle()).expect("raw buffer");
        assert_eq!(raw.size(), buffer.size_bytes());
    }

    #[test]
    fn test_shader_module_builder_missing_source() {
        let (device, _queue) = create_test_device().block_on();
        let mut registry = ResourceRegistry::default();

        let result = ShaderModuleBuilder::new(&device).build(&mut registry);
        assert!(matches!(result, Err(ShaderError::MissingSource)));
    }

    #[test]
    fn test_shader_module_builder_with_wgsl() {
        let (device, _queue) = create_test_device().block_on();
        let mut registry = ResourceRegistry::default();

        let shader = ShaderModuleBuilder::new(&device)
            .label("test_shader")
            .with_wgsl_source(
                r#"
                @vertex
                fn vs_main() -> @builtin(position) vec4<f32> {
                    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
                }
                "#,
            )
            .build(&mut registry)
            .expect("shader module should build");

        assert!(registry.get(shader).is_some());
    }

    #[test]
    fn test_compute_pipeline_builder_missing_shader() {
        let (device, _queue) = create_test_device().block_on();
        let mut registry = ResourceRegistry::default();

        let result = ComputePipelineBuilder::new(&device).build(&mut registry);
        assert!(matches!(result, Err(PipelineError::MissingComputeShader)));
    }

    #[test]
    fn test_compute_pipeline_builder_with_shader() {
        let (device, _queue) = create_test_device().block_on();
        let mut registry = ResourceRegistry::default();

        let compute_shader = ShaderModuleBuilder::new(&device)
            .with_wgsl_source(
                r#"
                @compute @workgroup_size(1)
                fn cs_main() {}
                "#,
            )
            .build(&mut registry)
            .expect("compute shader should build");

        let pipeline = ComputePipelineBuilder::new(&device)
            .with_label("compute")
            .with_compute_shader(compute_shader)
            .build(&mut registry)
            .expect("compute pipeline should build");

        assert!(registry.get(pipeline).is_some());
    }

    #[test]
    fn test_bind_group_builder_no_entries() {
        let (device, _queue) = create_test_device().block_on();
        let mut registry_mut = ResourceRegistry::default();

        let result = BindGroupBuilder::new(&device).build(&mut registry_mut);

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), BindGroupError::NoEntries));
    }

    #[test]
    fn test_bind_group_builder_invalid_resource() {
        let (device, _queue) = create_test_device().block_on();
        let mut registry_mut = ResourceRegistry::default();

        // Create a handle that doesn't exist in the registry
        let fake_handle = crate::frame_graph::resource::Handle::<wgpu::Buffer>::next();

        let result = BindGroupBuilder::new(&device)
            .buffer(0, fake_handle, BindingType::Uniform)
            .build(&mut registry_mut);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            BindGroupError::ResourceNotFound { binding: 0 }
        ));
    }

    #[test]
    fn test_shader_stage_conversion() {
        assert_eq!(ShaderStage::Vertex.to_wgpu(), wgpu::ShaderStages::VERTEX);
        assert_eq!(
            ShaderStage::Fragment.to_wgpu(),
            wgpu::ShaderStages::FRAGMENT
        );
        assert_eq!(ShaderStage::Compute.to_wgpu(), wgpu::ShaderStages::COMPUTE);
        assert!(
            ShaderStage::VertexFragment
                .to_wgpu()
                .contains(wgpu::ShaderStages::VERTEX)
        );
        assert!(
            ShaderStage::VertexFragment
                .to_wgpu()
                .contains(wgpu::ShaderStages::FRAGMENT)
        );
    }

    #[test]
    fn test_binding_type_conversion() {
        // Test Uniform binding type
        let uniform_ty = BindingType::Uniform.to_wgpu_binding_type();
        match uniform_ty {
            wgpu::BindingType::Buffer { ty, .. } => {
                assert_eq!(ty, wgpu::BufferBindingType::Uniform);
            }
            _ => panic!("Expected Buffer binding type"),
        }

        // Test Storage binding types
        let storage_read_ty = BindingType::StorageRead.to_wgpu_binding_type();
        match storage_read_ty {
            wgpu::BindingType::Buffer { ty, .. } => match ty {
                wgpu::BufferBindingType::Storage { read_only } => assert!(read_only),
                _ => panic!("Expected Storage binding type"),
            },
            _ => panic!("Expected Buffer binding type"),
        }

        // Test Sampler binding type
        let sampler_ty = BindingType::Sampler { filtering: true }.to_wgpu_binding_type();
        match sampler_ty {
            wgpu::BindingType::Sampler(ty) => {
                assert_eq!(ty, wgpu::SamplerBindingType::Filtering);
            }
            _ => panic!("Expected Sampler binding type"),
        }
    }
}
