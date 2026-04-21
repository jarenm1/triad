use crate::Handle;
use std::sync::{Arc, Mutex};

/// Frame-local texture view slot for render targets that change every frame,
/// such as the current surface view.
#[derive(Debug, Default)]
pub struct FrameTextureView {
    inner: Arc<Mutex<Option<Arc<wgpu::TextureView>>>>,
}

impl FrameTextureView {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&self, view: Arc<wgpu::TextureView>) {
        if let Ok(mut slot) = self.inner.lock() {
            *slot = Some(view);
        }
    }

    pub fn clear(&self) {
        if let Ok(mut slot) = self.inner.lock() {
            *slot = None;
        }
    }

    pub fn get(&self) -> Option<Arc<wgpu::TextureView>> {
        self.inner.lock().ok().and_then(|slot| slot.clone())
    }
}

/// Frame-local buffer slot for resources that should vary per frame without
/// forcing a frame-graph rebuild, such as rotating readback targets.
#[derive(Debug, Default)]
pub struct FrameBufferHandle {
    inner: Arc<Mutex<Option<Handle<wgpu::Buffer>>>>,
}

impl FrameBufferHandle {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&self, handle: Handle<wgpu::Buffer>) {
        if let Ok(mut slot) = self.inner.lock() {
            *slot = Some(handle);
        }
    }

    pub fn clear(&self) {
        if let Ok(mut slot) = self.inner.lock() {
            *slot = None;
        }
    }

    pub fn get(&self) -> Option<Handle<wgpu::Buffer>> {
        self.inner.lock().ok().and_then(|slot| *slot)
    }
}
