# Clone and Copy Analysis

## Investigation Results

### ✅ Good: Handle<T> Copy Implementation
**Location:** `triad-gpu/src/frame_graph/resource.rs:104-110`

```rust
impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self  // Just copies the ID, not the resource
    }
}
impl<T> Copy for Handle<T> {}
```

**Analysis:** ✅ **Correct** - Handles are just IDs (u64), so copying them is cheap and correct. The actual resources are stored in the registry.

### ✅ Good: ResourceRegistry Returns References
**Location:** `triad-gpu/src/resource_registry.rs:27-34`

```rust
pub fn get<T: ResourceType>(&self, handle: Handle<T>) -> Option<&T>
```

**Analysis:** ✅ **Correct** - Returns references, not clones. Resources are stored by value in HashMap, but wgpu types are internally Arc-like, so this is efficient.

### ⚠️ Issue: ResourceInfo Clone
**Location:** `triad-gpu/src/frame_graph/resource.rs:31`

```rust
#[derive(Debug, Clone)]
pub struct ResourceInfo {
    state: ResourceState,
    first_used_pass: usize,
    last_used_pass: usize,
}
```

**Analysis:** ⚠️ **Potentially Unnecessary** - `ResourceInfo` is cloned in frame graph tracking, but it's small (3 usize + enum), so impact is minimal. However, we could use references or `Rc` if needed.

### ⚠️ Issue: ResourceAccess Clone
**Location:** `triad-gpu/src/frame_graph/pass.rs:61`

```rust
#[derive(Debug, Clone)]
pub struct ResourceAccess {
    pub handle_id: u64,
    pub state: ResourceState,
}
```

**Analysis:** ⚠️ **Potentially Unnecessary** - Small struct (u64 + enum), cloned when building passes. Could be optimized but impact is minimal.

### ✅ Good: ResourceState Copy
**Location:** `triad-gpu/src/frame_graph/resource.rs:8`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResourceState { ... }
```

**Analysis:** ✅ **Correct** - Small enum, Copy is appropriate.

### ✅ Good: BindingType Copy
**Location:** `triad-gpu/src/builder.rs:127`

```rust
#[derive(Debug, Clone, Copy)]
pub enum BindingType { ... }
```

**Analysis:** ✅ **Correct** - Enum with Copy, appropriate.

### ✅ Good: BufferUsage Copy
**Location:** `triad-gpu/src/builder.rs:11`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferUsage { ... }
```

**Analysis:** ✅ **Correct** - Enum with Copy, appropriate.

## Potential Optimizations

### 1. ResourceInfo in Frame Graph
**Current:** Cloned when tracking resource states
**Impact:** Low - small struct, infrequent operations
**Recommendation:** Keep as-is unless profiling shows it's a bottleneck

### 2. ResourceAccess in Pass Builder
**Current:** Cloned when building passes
**Impact:** Low - small struct, only during setup
**Recommendation:** Keep as-is

### 3. Handle Storage in Builders
**Current:** Handles stored in vectors in builders
**Analysis:** ✅ **Correct** - Handles are Copy, so storing them is cheap

### 4. Resource Lookups
**Current:** Multiple `registry.get()` calls in bind group builder
**Example:** `triad-gpu/src/builder.rs:319-327`

```rust
for (binding, buffer_handle, _) in &self.buffer_bindings {
    let buffer = self
        .registry
        .get(*buffer_handle)  // Lookup each time
        .ok_or(BindGroupBuildError::ResourceNotFound)?;
    // ...
}
```

**Analysis:** ✅ **Correct** - Returns references, no cloning. Multiple lookups are fine since HashMap lookups are O(1).

## Wgpu Resource Internals

**Important:** wgpu resources (Buffer, Texture, etc.) are internally reference-counted (Arc-like), so:
- Storing them by value in HashMap is efficient
- Cloning a wgpu resource is cheap (just increments ref count)
- However, we're NOT cloning them - we're storing them once and returning references

## Summary

### ✅ What's Working Well
1. **Handles are Copy** - Cheap to copy, just IDs
2. **Registry returns references** - No unnecessary clones
3. **Resources stored efficiently** - wgpu types are Arc-like internally
4. **Enums use Copy** - Appropriate for small types

### ⚠️ Minor Optimizations (Low Priority)
1. **ResourceInfo Clone** - Could use references if needed, but current approach is fine
2. **ResourceAccess Clone** - Small struct, minimal impact

### 🎯 Key Finding
**The code is already efficiently sharing data through the resource registry.** Resources are stored once, accessed via references, and handles are cheap to copy. No significant clone/copy issues found.

## Recommendations

1. **Keep current approach** - The resource registry pattern is working well
2. **Monitor performance** - If profiling shows issues, consider:
   - Using `Rc<ResourceInfo>` instead of cloning
   - Caching resource lookups if same resource accessed multiple times per frame
3. **Document the pattern** - Make it clear that handles are cheap to copy and resources are shared
