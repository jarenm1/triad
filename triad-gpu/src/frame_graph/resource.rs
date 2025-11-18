use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

/// Type-safe resource handle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Handle<T> {
    pub id: u64,
    _phantom: PhantomData<T>,
}

impl<T> Handle<T> {
    pub fn new(id: u64) -> Self {
        Self {
            id,
            _phantom: PhantomData,
        }
    }

    pub fn id(&self) -> u64 {
        self.id
    }
}

/// Resource type marker traits
pub trait ResourceType: 'static {}

impl ResourceType for wgpu::Buffer {}
impl ResourceType for wgpu::Texture {}
impl ResourceType for wgpu::Sampler {}
impl ResourceType for wgpu::BindGroup {}
impl ResourceType for wgpu::RenderPipeline {}
impl ResourceType for wgpu::ComputePipeline {}
impl ResourceType for wgpu::ShaderModule {}

/// Resource state tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceState {
    Undefined,
    Read,
    Write,
    ReadWrite,
}

/// Resource metadata
pub struct ResourceInfo {
    pub state: ResourceState,
    pub last_used_pass: usize,
    pub first_used_pass: usize,
}

impl ResourceInfo {
    pub fn new() -> Self {
        Self {
            state: ResourceState::Undefined,
            last_used_pass: 0,
            first_used_pass: usize::MAX,
        }
    }
}

/// Handle ID generator
static HANDLE_ID: AtomicU64 = AtomicU64::new(1);

pub fn next_handle_id() -> u64 {
    HANDLE_ID.fetch_add(1, Ordering::Relaxed)
}
