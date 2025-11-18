use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

/// Type-safe resource handle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Handle<T> {
    id: u64,
    _phantom: PhantomData<T>,
}
impl<T> Handle<T> {
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

impl ResourceState {
    /// Merge this state with a new access pattern.
    /// Accumulates access types rather than overwriting:
    /// - Read + Write → ReadWrite
    /// - Write + Read → ReadWrite
    /// - ReadWrite + anything → ReadWrite
    pub fn merge_with(self, new_access: ResourceState) -> ResourceState {
        match (self, new_access) {
            // If either is ReadWrite, result is ReadWrite
            (ResourceState::ReadWrite, _) | (_, ResourceState::ReadWrite) => {
                ResourceState::ReadWrite
            }
            // Read + Write or Write + Read → ReadWrite
            (ResourceState::Read, ResourceState::Write)
            | (ResourceState::Write, ResourceState::Read) => ResourceState::ReadWrite,
            // Same access type → keep it
            (a, b) if a == b => a,
            // Undefined + anything → the new access
            (ResourceState::Undefined, new) => new,
            (old, ResourceState::Undefined) => old,
            // This shouldn't happen, but handle gracefully
            _ => ResourceState::ReadWrite,
        }
    }
}

/// Resource metadata
pub struct ResourceInfo {
    state: ResourceState,
    last_used_pass: usize,
    first_used_pass: usize,
}

impl ResourceInfo {
    pub fn new() -> Self {
        Self {
            state: ResourceState::Undefined,
            last_used_pass: 0,
            first_used_pass: usize::MAX,
        }
    }

    pub fn state(&self) -> ResourceState {
        self.state
    }

    pub fn set_state(&mut self, state: ResourceState) {
        self.state = state;
    }

    pub fn last_used_pass(&self) -> usize {
        self.last_used_pass
    }

    pub fn set_last_used_pass(&mut self, pass: usize) {
        self.last_used_pass = pass;
    }

    pub fn first_used_pass(&self) -> usize {
        self.first_used_pass
    }

    pub fn set_first_used_pass(&mut self, pass: usize) {
        self.first_used_pass = pass;
    }
}

/// Handle ID generator
static HANDLE_ID: AtomicU64 = AtomicU64::new(1);

pub fn next_handle_id() -> u64 {
    HANDLE_ID.fetch_add(1, Ordering::Relaxed)
}
