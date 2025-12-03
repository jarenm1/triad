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
