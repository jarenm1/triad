use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

pub type HandleId = u64;

/// Resource state for tracking access patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResourceState {
    #[default]
    Undefined,
    Read,
    Write,
    ReadWrite,
}

impl ResourceState {
    /// Merge two resource states
    pub fn merge_with(self, other: ResourceState) -> ResourceState {
        match (self, other) {
            (ResourceState::Undefined, other) => other,
            (current, ResourceState::Undefined) => current,
            (ResourceState::Read, ResourceState::Read) => ResourceState::Read,
            (ResourceState::Write, ResourceState::Write) => ResourceState::Write,
            _ => ResourceState::ReadWrite,
        }
    }
}

/// Resource tracking information for frame graph lifetime management
#[derive(Debug, Clone)]
pub struct ResourceInfo {
    state: ResourceState,
    first_used_pass: usize,
    last_used_pass: usize,
}

impl ResourceInfo {
    pub fn new() -> Self {
        Self {
            state: ResourceState::Undefined,
            first_used_pass: usize::MAX,
            last_used_pass: 0,
        }
    }

    pub fn state(&self) -> ResourceState {
        self.state
    }

    pub fn set_state(&mut self, state: ResourceState) {
        self.state = state;
    }

    pub fn first_used_pass(&self) -> usize {
        self.first_used_pass
    }

    pub fn set_first_used_pass(&mut self, pass: usize) {
        self.first_used_pass = pass;
    }

    pub fn last_used_pass(&self) -> usize {
        self.last_used_pass
    }

    pub fn set_last_used_pass(&mut self, pass: usize) {
        self.last_used_pass = pass;
    }
}

impl Default for ResourceInfo {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct TransientBufferDesc {
    label: Option<String>,
    size: u64,
    usage: wgpu::BufferUsages,
    mapped_at_creation: bool,
}

impl TransientBufferDesc {
    pub fn new(size: u64, usage: wgpu::BufferUsages) -> Self {
        Self {
            label: None,
            size,
            usage,
            mapped_at_creation: false,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn mapped_at_creation(mut self, mapped_at_creation: bool) -> Self {
        self.mapped_at_creation = mapped_at_creation;
        self
    }

    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn usage(&self) -> wgpu::BufferUsages {
        self.usage
    }

    pub fn is_mapped_at_creation(&self) -> bool {
        self.mapped_at_creation
    }
}

#[derive(Debug, Clone)]
pub struct TransientTextureDesc {
    label: Option<String>,
    size: wgpu::Extent3d,
    mip_level_count: u32,
    sample_count: u32,
    dimension: wgpu::TextureDimension,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
    view_formats: Vec<wgpu::TextureFormat>,
}

impl TransientTextureDesc {
    pub fn new(
        size: wgpu::Extent3d,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> Self {
        Self {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: Vec::new(),
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn with_mip_level_count(mut self, mip_level_count: u32) -> Self {
        self.mip_level_count = mip_level_count;
        self
    }

    pub fn with_sample_count(mut self, sample_count: u32) -> Self {
        self.sample_count = sample_count;
        self
    }

    pub fn with_dimension(mut self, dimension: wgpu::TextureDimension) -> Self {
        self.dimension = dimension;
        self
    }

    pub fn with_view_formats(mut self, view_formats: impl Into<Vec<wgpu::TextureFormat>>) -> Self {
        self.view_formats = view_formats.into();
        self
    }

    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    pub fn size(&self) -> wgpu::Extent3d {
        self.size
    }

    pub fn mip_level_count(&self) -> u32 {
        self.mip_level_count
    }

    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    pub fn dimension(&self) -> wgpu::TextureDimension {
        self.dimension
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    pub fn usage(&self) -> wgpu::TextureUsages {
        self.usage
    }

    pub fn view_formats(&self) -> &[wgpu::TextureFormat] {
        &self.view_formats
    }
}

/// Type-safe resource handle
///
/// We manually implement Hash, Eq, PartialEq, Clone, Copy to avoid adding bounds on T.
/// The derive macros would add `T: Hash`, `T: Eq`, `T: Clone`, `T: Copy` etc. even though
/// we only hash/compare/copy the `id` field.
#[derive(Debug)]
pub struct Handle<T> {
    id: HandleId,
    _phantom: PhantomData<fn(T) -> T>,
}

// Manual implementations without bounds on T
impl<T> Hash for Handle<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T> Eq for Handle<T> {}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Handle<T> {}
impl<T: ResourceType> Handle<T> {
    #[allow(dead_code)]
    pub(crate) fn new(id: u64) -> Self {
        Self {
            id,
            _phantom: PhantomData,
        }
    }

    /// Creates a new handle with a unique ID generated internally.
    /// This is the safe public constructor that prevents ID collisions.
    pub fn next() -> Self {
        Self {
            id: next_handle_id(),
            _phantom: PhantomData,
        }
    }

    /// Get the handle's unique ID
    pub fn id(&self) -> HandleId {
        self.id
    }
}

/// Handle ID generator
static HANDLE_ID: AtomicU64 = AtomicU64::new(1);

pub fn next_handle_id() -> HandleId {
    HANDLE_ID.fetch_add(1, Ordering::Relaxed)
}

/// Resource type marker traits
pub trait ResourceType: 'static {}

impl ResourceType for wgpu::Buffer {}
impl ResourceType for wgpu::Texture {}
impl ResourceType for wgpu::TextureView {}
impl ResourceType for wgpu::Sampler {}
impl ResourceType for wgpu::BindGroup {}
impl ResourceType for wgpu::BindGroupLayout {}
impl ResourceType for wgpu::RenderPipeline {}
impl ResourceType for wgpu::ComputePipeline {}
impl ResourceType for wgpu::ShaderModule {}
impl ResourceType for crate::FrameTextureView {}
impl ResourceType for crate::FrameBufferHandle {}
