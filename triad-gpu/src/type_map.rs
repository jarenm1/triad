use std::{
    any::{Any, TypeId},
    collections::{HashMap, hash_map::Entry},
};

#[derive(Default)]
pub struct TypeMap {
    inner: HashMap<TypeId, Box<dyn Any>>,
}

impl TypeMap {
    pub fn get<T: 'static>(&self) -> Option<&T> {
        self.inner
            .get(&TypeId::of::<T>())
            .and_then(|b| b.downcast_ref())
    }

    pub fn get_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.inner
            .get_mut(&TypeId::of::<T>())
            .and_then(|b| b.downcast_mut())
    }

    /// Get an entry for type T, allowing in-place initialization
    pub fn entry<T: 'static + Default>(&mut self) -> TypeMapEntry<'_, T> {
        let type_id = TypeId::of::<T>();
        match self.inner.entry(type_id) {
            Entry::Occupied(entry) => TypeMapEntry::Occupied(OccupiedEntry {
                entry,
                _marker: std::marker::PhantomData,
            }),
            Entry::Vacant(entry) => TypeMapEntry::Vacant(VacantEntry {
                entry,
                _marker: std::marker::PhantomData,
            }),
        }
    }
}

pub enum TypeMapEntry<'a, T: 'static> {
    Occupied(OccupiedEntry<'a, T>),
    Vacant(VacantEntry<'a, T>),
}

pub struct OccupiedEntry<'a, T: 'static> {
    entry: std::collections::hash_map::OccupiedEntry<'a, TypeId, Box<dyn Any>>,
    _marker: std::marker::PhantomData<T>,
}

pub struct VacantEntry<'a, T: 'static> {
    entry: std::collections::hash_map::VacantEntry<'a, TypeId, Box<dyn Any>>,
    _marker: std::marker::PhantomData<T>,
}

impl<'a, T: 'static + Default> TypeMapEntry<'a, T> {
    pub fn or_default(self) -> &'a mut T {
        match self {
            TypeMapEntry::Occupied(entry) => entry.entry.into_mut().downcast_mut().unwrap(),
            TypeMapEntry::Vacant(entry) => entry
                .entry
                .insert(Box::new(T::default()))
                .downcast_mut()
                .unwrap(),
        }
    }
}
