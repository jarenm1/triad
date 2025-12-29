# API Usage Examples

## Before and After Comparison

### Creating Buffers

#### Before (Current API)
```rust
use triad_gpu::wgpu::util::DeviceExt;

let triangle_buffer = device.create_buffer_init(&triad_gpu::wgpu::util::BufferInitDescriptor {
    label: Some("Triangle Buffer"),
    contents: triangle_data,
    usage: triad_gpu::wgpu::BufferUsages::STORAGE,
});
let triangle_buffer_handle = registry.insert(triangle_buffer);
```

#### After (New Builder API)
```rust
let triangle_buffer = renderer
    .create_buffer()
    .label("Triangle Buffer")
    .with_pod_data(&triangles)
    .usage(BufferUsage::Storage { read_only: true })
    .build(&mut registry)?;
```

**Benefits:**
- 6 lines → 5 lines (slightly shorter)
- No need to import `DeviceExt`
- Type-safe `BufferUsage` enum
- Automatic bytemuck casting with `with_pod_data`

### Creating Bind Groups

#### Before (Current API)
```rust
// Create bind group layout
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
        triad_gpu::wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: triad_gpu::wgpu::ShaderStages::VERTEX,
            ty: triad_gpu::wgpu::BindingType::Buffer {
                ty: triad_gpu::wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(
                    std::num::NonZeroU64::new(
                        std::mem::size_of::<CameraUniforms>() as u64
                    ).unwrap(),
                ),
            },
            count: None,
        },
    ],
});

// Create bind group
let bind_group = device.create_bind_group(&triad_gpu::wgpu::BindGroupDescriptor {
    label: Some("Triangle Bind Group"),
    layout: &bind_group_layout,
    entries: &[
        triad_gpu::wgpu::BindGroupEntry {
            binding: 0,
            resource: registry.get(triangle_buffer_handle).unwrap().as_entire_binding(),
        },
        triad_gpu::wgpu::BindGroupEntry {
            binding: 1,
            resource: registry.get(camera_buffer_handle).unwrap().as_entire_binding(),
        },
    ],
});
let bind_group_handle = registry.insert(bind_group);
```

#### After (New Builder API)
```rust
let (layout_handle, bind_group_handle) = renderer
    .create_bind_group(&registry)
    .label("Triangle Bind Group")
    .buffer(0, triangle_buffer, BindingType::StorageRead)
    .buffer(1, camera_buffer, BindingType::Uniform)
    .build(&mut registry)?;
```

**Benefits:**
- ~50 lines → 6 lines (88% reduction!)
- Automatic layout creation
- Type-safe binding types
- No manual entry construction
- Clearer intent

### Writing to Buffers

#### Before (Current API)
```rust
let camera_buffer = registry.get(camera_buffer_handle).expect("camera buffer");
queue.write_buffer(camera_buffer, 0, bytemuck::bytes_of(camera));
```

#### After (New API)
```rust
renderer.write_buffer(camera_buffer_handle, &[camera], &registry)?;
```

**Benefits:**
- Automatic bytemuck conversion
- Error handling instead of expect
- Cleaner API

## Complete Example: Triangle Splatting Setup

### Before
```rust
// ~100 lines of boilerplate for buffer and bind group creation
```

### After
```rust
// Create buffers
let triangle_buffer = renderer
    .create_buffer()
    .label("Triangle Buffer")
    .with_pod_data(&triangles)
    .usage(BufferUsage::Storage { read_only: true })
    .build(&mut registry)?;

let camera_buffer = renderer
    .create_buffer()
    .label("Camera Buffer")
    .with_pod_data(&[CameraUniforms::default()])
    .usage(BufferUsage::Uniform)
    .build(&mut registry)?;

let index_buffer = renderer
    .create_buffer()
    .label("Index Buffer")
    .with_pod_data(&indices)
    .usage(BufferUsage::Index)
    .build(&mut registry)?;

// Create bind group
let (_layout, bind_group) = renderer
    .create_bind_group(&registry)
    .label("Triangle Bind Group")
    .buffer(0, triangle_buffer, BindingType::StorageRead)
    .buffer(1, camera_buffer, BindingType::Uniform)
    .build(&mut registry)?;
```

**Total reduction:** ~100 lines → ~25 lines (75% reduction)

## Migration Path

1. **Phase 1 (Current)**: New builders coexist with old API
2. **Phase 2**: Update examples to use new API
3. **Phase 3**: Mark old API as deprecated
4. **Phase 4**: Remove old API (after sufficient migration period)

## Future Enhancements

- Texture builder
- Sampler builder
- Shader manager with hot-reloading
- Command encoder wrapper
- Pipeline builder improvements
