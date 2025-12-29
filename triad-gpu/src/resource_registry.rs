use crate::frame_graph::resource::{Handle, ResourceType};
use crate::type_map::TypeMap;
use std::collections::HashMap;
use std::hash::Hash;

#[derive(Default)]
pub struct ResourceRegistry {
    /// Type map for storing resources by type
    storages: TypeMap,
}

impl ResourceRegistry {
    pub fn insert<T: ResourceType>(&mut self, resource: T) -> Handle<T>
    where
        Handle<T>: Hash + Eq,
    {
        let handle = Handle::next();

        self.storages
            .entry::<HashMap<Handle<T>, T>>()
            .or_default()
            .insert(handle, resource);

        handle
    }

    pub fn get<T: ResourceType>(&self, handle: Handle<T>) -> Option<&T>
    where
        Handle<T>: Hash + Eq,
    {
        self.storages
            .get::<HashMap<Handle<T>, T>>()
            .and_then(|map| map.get(&handle))
    }

    pub fn get_mut<T: ResourceType>(&mut self, handle: Handle<T>) -> Option<&mut T>
    where
        Handle<T>: Hash + Eq,
    {
        self.storages
            .get_mut::<HashMap<Handle<T>, T>>()
            .and_then(|map| map.get_mut(&handle))
    }

    pub fn remove<T: ResourceType>(&mut self, handle: Handle<T>) -> Option<T>
    where
        Handle<T>: Hash + Eq,
    {
        self.storages
            .get_mut::<HashMap<Handle<T>, T>>()
            .and_then(|map| map.remove(&handle))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_graph::resource::ResourceType;
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
    fn test_resource_registry_insert_and_get() {
        let (device, _queue) = create_test_device().block_on();
        let mut registry = ResourceRegistry::default();

        // Create a buffer and insert it
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_buffer"),
            size: 1024,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        let handle = registry.insert(buffer);
        
        // Verify we can retrieve it
        let retrieved = registry.get(handle);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().size(), 1024);
    }

    #[test]
    fn test_resource_registry_get_mut() {
        let (device, _queue) = create_test_device().block_on();
        let mut registry = ResourceRegistry::default();

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_buffer"),
            size: 512,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        let handle = registry.insert(buffer);
        
        // Verify we can get a mutable reference
        let buffer_mut = registry.get_mut(handle);
        assert!(buffer_mut.is_some());
    }

    #[test]
    fn test_resource_registry_remove() {
        let (device, _queue) = create_test_device().block_on();
        let mut registry = ResourceRegistry::default();

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_buffer"),
            size: 256,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        let handle = registry.insert(buffer);
        
        // Remove the resource
        let removed = registry.remove(handle);
        assert!(removed.is_some());
        
        // Verify it's gone
        assert!(registry.get(handle).is_none());
    }

    #[test]
    fn test_resource_registry_multiple_types() {
        let (device, _queue) = create_test_device().block_on();
        let mut registry = ResourceRegistry::default();

        // Insert different types of resources
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buffer"),
            size: 128,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        let buffer_handle = registry.insert(buffer);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
        let sampler_handle = registry.insert(sampler);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("texture"),
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let texture_handle = registry.insert(texture);

        // Verify all resources can be retrieved
        assert!(registry.get(buffer_handle).is_some());
        assert!(registry.get(sampler_handle).is_some());
        assert!(registry.get(texture_handle).is_some());
    }

    #[test]
    fn test_resource_registry_nonexistent_handle() {
        let registry = ResourceRegistry::default();
        
        // Try to get a handle that doesn't exist
        let fake_handle = crate::frame_graph::resource::Handle::<wgpu::Buffer>::next();
        assert!(registry.get(fake_handle).is_none());
    }

    #[test]
    fn test_resource_registry_handle_uniqueness() {
        let (device, _queue) = create_test_device().block_on();
        let mut registry = ResourceRegistry::default();

        let buffer1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buffer1"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        let handle1 = registry.insert(buffer1);

        let buffer2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buffer2"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        let handle2 = registry.insert(buffer2);

        // Handles should be unique
        assert_ne!(handle1.id(), handle2.id());
        
        // Both should be retrievable
        assert!(registry.get(handle1).is_some());
        assert!(registry.get(handle2).is_some());
    }
}
