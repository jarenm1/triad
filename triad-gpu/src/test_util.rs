//! Helpers used only by unit tests.
//!
//! Parallel `cargo test` runs many tests that each create a [`wgpu::Instance`]. Some Vulkan
//! drivers crash (SIGSEGV) when multiple instances are initialized concurrently. We
//! serialize GPU setup in test builds via [`gpu_test_lock`] held for the duration of
//! [`crate::Renderer::new`].

use std::sync::Mutex;

static GPU_TEST_INIT: Mutex<()> = Mutex::new(());

pub(crate) fn gpu_test_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU_TEST_INIT.lock().unwrap_or_else(|e| e.into_inner())
}

/// Shared helper for tests that only need a [`wgpu::Device`] + [`wgpu::Queue`].
pub(crate) async fn create_test_device() -> (wgpu::Device, wgpu::Queue) {
    let renderer = crate::Renderer::new()
        .await
        .expect("Failed to create renderer");
    renderer.into_device_queue()
}
