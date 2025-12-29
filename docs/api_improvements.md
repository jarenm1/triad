# Triad-GPU API Improvements

## Current API Issues

### 1. Excessive wgpu Exposure
**Problem:** Users must use `triad_gpu::wgpu::...` everywhere, exposing too much of the underlying API.

**Example:**
```rust
device.create_buffer_init(&triad_gpu::wgpu::util::BufferInitDescriptor {
    label: Some("Triangle Buffer"),
    contents: triangle_data,
    usage: triad_gpu::wgpu::BufferUsages::STORAGE,
});
```

**Issues:**
- Verbose and repetitive
- No abstraction over wgpu types
- Hard to change underlying implementation
- Users need deep wgpu knowledge

### 2. Low-Level Resource Creation
**Problem:** All resource creation requires manual wgpu calls with full descriptors.

**Example:**
```rust
let bind_group_layout = device.create_bind_group_layout(&triad_gpu::wgpu::BindGroupLayoutDescriptor {
    label: Some("Triangle Bind Group Layout"),
    entries: &[
        triad_gpu::wgpu::BindGroupLayoutEntry { /* ... */ },
        triad_gpu::wgpu::BindGroupLayoutEntry { /* ... */ },
    ],
});
```

**Issues:**
- Very verbose
- Error-prone (easy to misconfigure)
- No type safety for common patterns
- Repetitive boilerplate

### 3. Renderer is Too Minimal
**Problem:** `Renderer` only wraps device/queue/instance, doesn't provide helpers.

**Current:**
```rust
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
}
```

**Issues:**
- Users must access device/queue directly
- No convenience methods for common operations
- No resource management helpers

### 4. Resource Registry Verbosity
**Problem:** Handle-based system requires explicit type parameters everywhere.

**Issues:**
- Type annotations needed frequently
- Handle management is manual
- No automatic cleanup

### 5. Shader Management
**Problem:** Shader creation is manual and not integrated with resource registry.

**Current:**
```rust
let vertex_shader = device.create_shader_module(triad_gpu::wgpu::ShaderModuleDescriptor {
    label: Some("triangle_vs"),
    source: triad_gpu::wgpu::ShaderSource::Wgsl(vertex_shader_source.into()),
});
let vertex_shader_handle = registry.insert(vertex_shader);
```

**Issues:**
- Manual shader module creation
- No shader caching
- No hot-reloading support

## Proposed Improvements

### 1. High-Level Resource Builders

#### Buffer Builder
```rust
// Instead of:
device.create_buffer_init(&wgpu::util::BufferInitDescriptor { ... })

// Provide:
let buffer = renderer
    .create_buffer("Triangle Buffer")
    .with_data(triangle_data)
    .usage(BufferUsage::Storage)
    .build(&mut registry)?;
```

#### Bind Group Builder
```rust
// Instead of manual bind group layout + bind group creation

// Provide:
let bind_group = renderer
    .create_bind_group("Triangle Bind Group")
    .buffer(0, &triangle_buffer, BindingType::StorageRead)
    .buffer(1, &camera_buffer, BindingType::Uniform)
    .build(&mut registry)?;
```

#### Texture Builder
```rust
let texture = renderer
    .create_texture("Depth Texture")
    .size(width, height)
    .format(TextureFormat::Depth32Float)
    .usage(TextureUsage::RenderAttachment)
    .build(&mut registry)?;
```

### 2. Enhanced Renderer API

```rust
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    registry: ResourceRegistry, // Built-in registry
}

impl Renderer {
    // Existing methods...
    
    // New convenience methods:
    
    /// Create a buffer builder
    pub fn create_buffer(&self, label: impl Into<String>) -> BufferBuilder;
    
    /// Create a texture builder
    pub fn create_texture(&self, label: impl Into<String>) -> TextureBuilder;
    
    /// Create a bind group builder
    pub fn create_bind_group(&self, label: impl Into<String>) -> BindGroupBuilder;
    
    /// Create a shader from source
    pub fn create_shader(&self, source: &str, kind: ShaderKind) -> Handle<ShaderModule>;
    
    /// Write to a buffer
    pub fn write_buffer<T: bytemuck::Pod>(&self, buffer: Handle<Buffer>, data: &[T]);
    
    /// Submit command encoder
    pub fn submit(&self, encoder: CommandEncoder) -> SubmissionId;
}
```

### 3. Type-Safe Enums

Replace wgpu enums with our own for better API:

```rust
pub enum BufferUsage {
    Vertex,
    Index,
    Uniform,
    Storage { read_only: bool },
    CopySrc,
    CopyDst,
}

pub enum BindingType {
    Uniform,
    Storage { read_only: bool },
    Texture { sample_type: TextureSampleType },
    Sampler { filtering: bool },
}

pub enum ShaderKind {
    Vertex,
    Fragment,
    Compute,
}
```

### 4. Simplified Pipeline Building

```rust
// Instead of:
let pipeline = RenderPipelineBuilder::new(device)
    .with_vertex_shader(vertex_shader_handle)
    .with_fragment_shader(fragment_shader_handle)
    .with_layout(pipeline_layout)
    .with_primitive(...)
    .build(registry)?;

// Provide:
let pipeline = renderer
    .create_render_pipeline("Triangle Pipeline")
    .vertex_shader(&vertex_shader)
    .fragment_shader(&fragment_shader)
    .bind_group_layout(&bind_group_layout)
    .primitive(PrimitiveConfig::triangles())
    .depth_test(DepthTest::Less)
    .blend(BlendMode::PremultipliedAlpha)
    .build(&mut registry)?;
```

### 5. Command Encoder Wrapper

```rust
pub struct CommandEncoder {
    inner: wgpu::CommandEncoder,
    renderer: &Renderer,
}

impl CommandEncoder {
    pub fn begin_render_pass(&mut self, pass: RenderPassDescriptor) -> RenderPass;
    
    pub fn copy_buffer_to_buffer(&mut self, src: Handle<Buffer>, dst: Handle<Buffer>);
    
    pub fn finish(self) -> CommandBuffer;
}
```

### 6. Shader Management

```rust
pub struct ShaderManager {
    shaders: HashMap<String, Handle<ShaderModule>>,
    hot_reload: bool,
}

impl Renderer {
    pub fn load_shader(&mut self, name: &str, source: &str) -> Handle<ShaderModule>;
    
    pub fn reload_shader(&mut self, name: &str) -> Result<(), ShaderError>;
    
    pub fn get_shader(&self, name: &str) -> Option<Handle<ShaderModule>>;
}
```

### 7. Resource Cleanup

```rust
impl ResourceRegistry {
    /// Remove all resources of a type
    pub fn clear<T: ResourceType>(&mut self);
    
    /// Get resource count
    pub fn count<T: ResourceType>(&self) -> usize;
    
    /// Automatic cleanup on drop (if needed)
}
```

## Migration Strategy

### Phase 1: Add New APIs (Non-Breaking)
- Add builder patterns alongside existing APIs
- Mark old APIs as `#[deprecated]` with migration notes
- Keep full backward compatibility

### Phase 2: Update Examples
- Migrate examples to use new APIs
- Document patterns and best practices

### Phase 3: Deprecate Old APIs
- After sufficient migration period
- Provide clear migration guide

## Benefits

1. **Simpler API**: Less boilerplate, more readable code
2. **Type Safety**: Enums prevent invalid configurations
3. **Better Errors**: Builder pattern allows validation before creation
4. **Easier Testing**: Mockable builders and clearer interfaces
5. **Future-Proof**: Can change underlying implementation without breaking API
6. **Better Documentation**: Self-documenting builder APIs

## Example: Before vs After

### Before (Current)
```rust
use triad_gpu::wgpu::util::DeviceExt;

let triangle_buffer = device.create_buffer_init(&triad_gpu::wgpu::util::BufferInitDescriptor {
    label: Some("Triangle Buffer"),
    contents: triangle_data,
    usage: triad_gpu::wgpu::BufferUsages::STORAGE,
});
let triangle_buffer_handle = registry.insert(triangle_buffer);

let bind_group_layout = device.create_bind_group_layout(&triad_gpu::wgpu::BindGroupLayoutDescriptor {
    label: Some("Triangle Bind Group Layout"),
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
        // ... more entries
    ],
});

let bind_group = device.create_bind_group(&triad_gpu::wgpu::BindGroupDescriptor {
    label: Some("Triangle Bind Group"),
    layout: &bind_group_layout,
    entries: &[
        triad_gpu::wgpu::BuildGroupEntry {
            binding: 0,
            resource: registry.get(triangle_buffer_handle).unwrap().as_entire_binding(),
        },
        // ... more entries
    ],
});
let bind_group_handle = registry.insert(bind_group);
```

### After (Proposed)
```rust
let triangle_buffer = renderer
    .create_buffer("Triangle Buffer")
    .with_data(triangle_data)
    .usage(BufferUsage::Storage { read_only: true })
    .build(&mut registry)?;

let bind_group = renderer
    .create_bind_group("Triangle Bind Group")
    .buffer(0, &triangle_buffer, BindingType::StorageRead)
    .buffer(1, &camera_buffer, BindingType::Uniform)
    .build(&mut registry)?;
```

**Reduction:** ~40 lines → ~8 lines (80% reduction)
