//! Uniform 3D spatial grid on the GPU: clear → count → linearize → exclusive scan →
//! init per-cell write heads → scatter entity indices into cell-major order.
//!
//! Intended for broadphase / binning (neighbor queries, culling prep). The exclusive scan
//! pass is **single-threaded** over cells — keep [`total_cells`] modest for frame-time work.

use crate::error::{BindGroupError, BufferError, PipelineError, ShaderError};
use crate::frame_graph::Handle;
use crate::resource_registry::ResourceRegistry;
use crate::{BufferUsage, ComputePipelineBuilder, Renderer, ShaderModuleBuilder};
use thiserror::Error;

/// Matches WGSL `struct Params { world_origin: vec3<f32>, cell_size: f32, grid_dims: vec3<u32>, entity_count: u32 }` (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpatialGridParams {
    /// `xyz` = world origin of cell `(0,0,0)` corner; `w` = `cell_size`.
    pub world_origin_cell: [f32; 4],
    /// `xyz` = grid dimensions `(nx, ny, nz)`; `w` = active entity count this frame.
    pub grid_dims_entities: [u32; 4],
}

impl SpatialGridParams {
    #[inline]
    pub fn new(world_origin: [f32; 3], cell_size: f32, grid_dims: [u32; 3], entity_count: u32) -> Self {
        Self {
            world_origin_cell: [world_origin[0], world_origin[1], world_origin[2], cell_size],
            grid_dims_entities: [grid_dims[0], grid_dims[1], grid_dims[2], entity_count],
        }
    }

    #[inline]
    pub fn entity_count(&self) -> u32 {
        self.grid_dims_entities[3]
    }

    #[inline]
    pub fn grid_dims(&self) -> [u32; 3] {
        [
            self.grid_dims_entities[0],
            self.grid_dims_entities[1],
            self.grid_dims_entities[2],
        ]
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EntityPosition {
    pub position: [f32; 3],
    pub _pad: f32,
}

#[derive(Debug, Clone)]
pub struct SpatialGridConfig {
    pub params: SpatialGridParams,
    pub max_entities: u32,
}

#[derive(Debug, Error)]
pub enum SpatialGridError {
    #[error("grid has zero cells (check grid_dims)")]
    ZeroCells,
    #[error("entity_count {entity} exceeds max_entities {max}")]
    EntityOverflow { entity: u32, max: u32 },
    #[error(transparent)]
    Buffer(#[from] BufferError),
    #[error(transparent)]
    Shader(#[from] ShaderError),
    #[error(transparent)]
    Pipeline(#[from] PipelineError),
    #[error(transparent)]
    BindGroup(#[from] BindGroupError),
}

pub type SpatialGridResult<T> = Result<T, SpatialGridError>;

/// `nx * ny * nz` cells.
#[inline]
pub const fn total_cells(dims: [u32; 3]) -> u32 {
    dims[0].saturating_mul(dims[1]).saturating_mul(dims[2])
}

const WGSL_PARAMS: &str = r#"
struct Params {
    world_origin: vec3<f32>,
    cell_size: f32,
    grid_dims: vec3<u32>,
    entity_count: u32,
}

@group(0) @binding(0) var<uniform> params: Params;

fn total_cells() -> u32 {
    return params.grid_dims.x * params.grid_dims.y * params.grid_dims.z;
}

fn cell_index(world_pos: vec3<f32>) -> u32 {
    let g = params.grid_dims;
    let o = params.world_origin;
    let inv = 1.0 / params.cell_size;
    let f = floor((world_pos - o) * inv);
    let ix = clamp(i32(f.x), 0, i32(g.x) - 1);
    let iy = clamp(i32(f.y), 0, i32(g.y) - 1);
    let iz = clamp(i32(f.z), 0, i32(g.z) - 1);
    let ux = u32(ix);
    let uy = u32(iy);
    let uz = u32(iz);
    return ux + g.x * (uy + g.y * uz);
}
"#;

const WGSL_CLEAR: &str = r#"
@group(0) @binding(1) var<storage, read_write> cell_atomics: array<atomic<u32>>;

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = total_cells();
    if (i >= n) { return; }
    atomicStore(&cell_atomics[i], 0u);
}
"#;

const WGSL_COUNT: &str = r#"
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> cell_atomics: array<atomic<u32>>;

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let entity = gid.x;
    if (entity >= params.entity_count) { return; }
    let p = positions[entity].xyz;
    let c = cell_index(p);
    atomicAdd(&cell_atomics[c], 1u);
}
"#;

const WGSL_LINEARIZE: &str = r#"
@group(0) @binding(1) var<storage, read_write> cell_atomics: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> counts_linear: array<u32>;

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = total_cells();
    if (i >= n) { return; }
    counts_linear[i] = atomicLoad(&cell_atomics[i]);
}
"#;

const WGSL_SCAN: &str = r#"
@group(0) @binding(1) var<storage, read> counts_for_scan: array<u32>;
@group(0) @binding(2) var<storage, read_write> cell_offsets: array<u32>;

@compute @workgroup_size(1)
fn cs_main() {
    let n = total_cells();
    var sum = 0u;
    var i = 0u;
    loop {
        if (i >= n) { break; }
        cell_offsets[i] = sum;
        sum = sum + counts_for_scan[i];
        i = i + 1u;
    }
    cell_offsets[n] = sum;
}
"#;

const WGSL_INIT_HEADS: &str = r#"
@group(0) @binding(1) var<storage, read> offsets_init: array<u32>;
@group(0) @binding(2) var<storage, read_write> write_heads: array<atomic<u32>>;

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = total_cells();
    if (i >= n) { return; }
    atomicStore(&write_heads[i], offsets_init[i]);
}
"#;

const WGSL_SCATTER: &str = r#"
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> write_heads: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> sorted_entity_ids: array<u32>;

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let entity = gid.x;
    if (entity >= params.entity_count) { return; }
    let p = positions[entity].xyz;
    let c = cell_index(p);
    let slot = atomicAdd(&write_heads[c], 1u);
    sorted_entity_ids[slot] = entity;
}
"#;

fn concat_wgsl(prefix: &str, body: &str) -> String {
    let mut s = String::with_capacity(prefix.len() + body.len());
    s.push_str(prefix);
    s.push_str(body);
    s
}

/// GPU resources and compute pipelines for one spatial grid configuration.
pub struct SpatialGridGpu {
    pub params: Handle<wgpu::Buffer>,
    pub positions: Handle<wgpu::Buffer>,
    pub cell_atomics: Handle<wgpu::Buffer>,
    pub counts_linear: Handle<wgpu::Buffer>,
    pub cell_offsets: Handle<wgpu::Buffer>,
    pub write_heads: Handle<wgpu::Buffer>,
    pub sorted_entity_ids: Handle<wgpu::Buffer>,
    pub pipeline_clear: Handle<wgpu::ComputePipeline>,
    pub pipeline_count: Handle<wgpu::ComputePipeline>,
    pub pipeline_linearize: Handle<wgpu::ComputePipeline>,
    pub pipeline_scan: Handle<wgpu::ComputePipeline>,
    pub pipeline_init_heads: Handle<wgpu::ComputePipeline>,
    pub pipeline_scatter: Handle<wgpu::ComputePipeline>,
    pub bind_clear: Handle<wgpu::BindGroup>,
    pub bind_count: Handle<wgpu::BindGroup>,
    pub bind_linearize: Handle<wgpu::BindGroup>,
    pub bind_scan: Handle<wgpu::BindGroup>,
    pub bind_init_heads: Handle<wgpu::BindGroup>,
    pub bind_scatter: Handle<wgpu::BindGroup>,
    cells: u32,
    max_entities: u32,
}

impl SpatialGridGpu {
    pub fn new(
        renderer: &Renderer,
        registry: &mut ResourceRegistry,
        cfg: SpatialGridConfig,
    ) -> SpatialGridResult<Self> {
        let dims = cfg.params.grid_dims();
        let cells = total_cells(dims);
        if cells == 0 {
            return Err(SpatialGridError::ZeroCells);
        }
        if cfg.params.entity_count() > cfg.max_entities {
            return Err(SpatialGridError::EntityOverflow {
                entity: cfg.params.entity_count(),
                max: cfg.max_entities,
            });
        }

        let device = renderer.device();

        let params_buf = renderer
            .create_gpu_buffer::<SpatialGridParams>()
            .label("spatial_grid params")
            .with_data(&[cfg.params])
            .usage(BufferUsage::Uniform)
            .build(registry)?;

        let positions = renderer
            .create_gpu_buffer::<EntityPosition>()
            .label("spatial_grid positions")
            .capacity(cfg.max_entities as usize)
            // COPY_DST so CPU/tests can seed positions; shaders use read-only storage.
            .usage(BufferUsage::StorageWritable)
            .build(registry)?;

        let cell_atomics = renderer
            .create_buffer()
            .label("spatial_grid cell_atomics")
            .size((cells as u64) * 4)
            .usage_flags(wgpu::BufferUsages::STORAGE)
            .build(registry)?;

        let counts_linear = renderer
            .create_gpu_buffer::<u32>()
            .label("spatial_grid counts_linear")
            .capacity(cells as usize)
            .usage(BufferUsage::Storage { read_only: false })
            .add_usage(wgpu::BufferUsages::COPY_SRC)
            .build(registry)?;

        let offset_slots = (cells as usize) + 1;
        let cell_offsets = renderer
            .create_gpu_buffer::<u32>()
            .label("spatial_grid cell_offsets")
            .capacity(offset_slots)
            .usage(BufferUsage::Storage { read_only: false })
            .add_usage(wgpu::BufferUsages::COPY_SRC)
            .build(registry)?;

        let write_heads = renderer
            .create_buffer()
            .label("spatial_grid write_heads")
            .size((cells as u64) * 4)
            .usage_flags(wgpu::BufferUsages::STORAGE)
            .build(registry)?;

        let sorted_entity_ids = renderer
            .create_gpu_buffer::<u32>()
            .label("spatial_grid sorted_entity_ids")
            .capacity(cfg.max_entities as usize)
            .usage(BufferUsage::Storage { read_only: false })
            .add_usage(wgpu::BufferUsages::COPY_SRC)
            .build(registry)?;

        let shader_clear = ShaderModuleBuilder::new(device)
            .label("spatial_grid clear")
            .with_wgsl_source(concat_wgsl(WGSL_PARAMS, WGSL_CLEAR))
            .build(registry)?;
        let shader_count = ShaderModuleBuilder::new(device)
            .label("spatial_grid count")
            .with_wgsl_source(concat_wgsl(WGSL_PARAMS, WGSL_COUNT))
            .build(registry)?;
        let shader_linearize = ShaderModuleBuilder::new(device)
            .label("spatial_grid linearize")
            .with_wgsl_source(concat_wgsl(WGSL_PARAMS, WGSL_LINEARIZE))
            .build(registry)?;
        let shader_scan = ShaderModuleBuilder::new(device)
            .label("spatial_grid scan")
            .with_wgsl_source(concat_wgsl(WGSL_PARAMS, WGSL_SCAN))
            .build(registry)?;
        let shader_init_heads = ShaderModuleBuilder::new(device)
            .label("spatial_grid init_heads")
            .with_wgsl_source(concat_wgsl(WGSL_PARAMS, WGSL_INIT_HEADS))
            .build(registry)?;
        let shader_scatter = ShaderModuleBuilder::new(device)
            .label("spatial_grid scatter")
            .with_wgsl_source(concat_wgsl(WGSL_PARAMS, WGSL_SCATTER))
            .build(registry)?;

        let layout_clear = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("spatial_grid clear layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout_count = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("spatial_grid count layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout_linearize = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("spatial_grid linearize layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout_scan = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("spatial_grid scan layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout_init_heads = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("spatial_grid init_heads layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout_scatter = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("spatial_grid scatter layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pl_clear = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("spatial_grid clear pl"),
            bind_group_layouts: &[&layout_clear],
            push_constant_ranges: &[],
        });
        let pl_count = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("spatial_grid count pl"),
            bind_group_layouts: &[&layout_count],
            push_constant_ranges: &[],
        });
        let pl_linearize = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("spatial_grid linearize pl"),
            bind_group_layouts: &[&layout_linearize],
            push_constant_ranges: &[],
        });
        let pl_scan = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("spatial_grid scan pl"),
            bind_group_layouts: &[&layout_scan],
            push_constant_ranges: &[],
        });
        let pl_init_heads = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("spatial_grid init_heads pl"),
            bind_group_layouts: &[&layout_init_heads],
            push_constant_ranges: &[],
        });
        let pl_scatter = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("spatial_grid scatter pl"),
            bind_group_layouts: &[&layout_scatter],
            push_constant_ranges: &[],
        });

        let params_handle = params_buf.handle();

        let bind_clear = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("spatial_grid bind clear"),
            layout: &layout_clear,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: registry.get(params_handle).unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: registry.get(cell_atomics).unwrap().as_entire_binding(),
                },
            ],
        });
        let bind_clear = registry.insert(bind_clear);

        let bind_count = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("spatial_grid bind count"),
            layout: &layout_count,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: registry.get(params_handle).unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: registry.get(positions.handle()).unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: registry.get(cell_atomics).unwrap().as_entire_binding(),
                },
            ],
        });
        let bind_count = registry.insert(bind_count);

        let bind_linearize = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("spatial_grid bind linearize"),
            layout: &layout_linearize,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: registry.get(params_handle).unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: registry.get(cell_atomics).unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: registry.get(counts_linear.handle()).unwrap().as_entire_binding(),
                },
            ],
        });
        let bind_linearize = registry.insert(bind_linearize);

        let bind_scan = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("spatial_grid bind scan"),
            layout: &layout_scan,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: registry.get(params_handle).unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: registry.get(counts_linear.handle()).unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: registry.get(cell_offsets.handle()).unwrap().as_entire_binding(),
                },
            ],
        });
        let bind_scan = registry.insert(bind_scan);

        let bind_init_heads = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("spatial_grid bind init_heads"),
            layout: &layout_init_heads,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: registry.get(params_handle).unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: registry.get(cell_offsets.handle()).unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: registry.get(write_heads).unwrap().as_entire_binding(),
                },
            ],
        });
        let bind_init_heads = registry.insert(bind_init_heads);

        let bind_scatter = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("spatial_grid bind scatter"),
            layout: &layout_scatter,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: registry.get(params_handle).unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: registry.get(positions.handle()).unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: registry.get(write_heads).unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: registry
                        .get(sorted_entity_ids.handle())
                        .unwrap()
                        .as_entire_binding(),
                },
            ],
        });
        let bind_scatter = registry.insert(bind_scatter);

        let pipeline_clear = ComputePipelineBuilder::new(device)
            .with_label("spatial_grid clear")
            .with_compute_shader(shader_clear)
            .with_layout(pl_clear)
            .build(registry)?;

        let pipeline_count = ComputePipelineBuilder::new(device)
            .with_label("spatial_grid count")
            .with_compute_shader(shader_count)
            .with_layout(pl_count)
            .build(registry)?;

        let pipeline_linearize = ComputePipelineBuilder::new(device)
            .with_label("spatial_grid linearize")
            .with_compute_shader(shader_linearize)
            .with_layout(pl_linearize)
            .build(registry)?;

        let pipeline_scan = ComputePipelineBuilder::new(device)
            .with_label("spatial_grid scan")
            .with_compute_shader(shader_scan)
            .with_layout(pl_scan)
            .build(registry)?;

        let pipeline_init_heads = ComputePipelineBuilder::new(device)
            .with_label("spatial_grid init_heads")
            .with_compute_shader(shader_init_heads)
            .with_layout(pl_init_heads)
            .build(registry)?;

        let pipeline_scatter = ComputePipelineBuilder::new(device)
            .with_label("spatial_grid scatter")
            .with_compute_shader(shader_scatter)
            .with_layout(pl_scatter)
            .build(registry)?;

        Ok(Self {
            params: params_handle,
            positions: positions.handle(),
            cell_atomics,
            counts_linear: counts_linear.handle(),
            cell_offsets: cell_offsets.handle(),
            write_heads,
            sorted_entity_ids: sorted_entity_ids.handle(),
            pipeline_clear,
            pipeline_count,
            pipeline_linearize,
            pipeline_scan,
            pipeline_init_heads,
            pipeline_scatter,
            bind_clear,
            bind_count,
            bind_linearize,
            bind_scan,
            bind_init_heads,
            bind_scatter,
            cells,
            max_entities: cfg.max_entities,
        })
    }

    #[inline]
    pub fn cell_count(&self) -> u32 {
        self.cells
    }

    #[inline]
    pub fn max_entities(&self) -> u32 {
        self.max_entities
    }

    /// Workgroup dispatch size for 1D kernels with `workgroup_size` 256.
    #[inline]
    pub fn dispatch_1d_256(count: u32) -> u32 {
        count.div_ceil(256)
    }

    /// Encode the full grid rebuild: clear → count → linearize → scan → init_heads → scatter.
    pub fn encode_rebuild(&self, encoder: &mut wgpu::CommandEncoder, registry: &ResourceRegistry) {
        let cells = self.cells;
        let wg256 = Self::dispatch_1d_256(cells);
        let wg_ent = Self::dispatch_1d_256(self.max_entities);

        let p_clear = registry.get(self.pipeline_clear).expect("pipeline_clear");
        let p_count = registry.get(self.pipeline_count).expect("pipeline_count");
        let p_linearize = registry.get(self.pipeline_linearize).expect("pipeline_linearize");
        let p_scan = registry.get(self.pipeline_scan).expect("pipeline_scan");
        let p_init = registry.get(self.pipeline_init_heads).expect("pipeline_init_heads");
        let p_scatter = registry.get(self.pipeline_scatter).expect("pipeline_scatter");

        let g_clear = registry.get(self.bind_clear).expect("bind_clear");
        let g_count = registry.get(self.bind_count).expect("bind_count");
        let g_linearize = registry.get(self.bind_linearize).expect("bind_linearize");
        let g_scan = registry.get(self.bind_scan).expect("bind_scan");
        let g_init = registry.get(self.bind_init_heads).expect("bind_init_heads");
        let g_scatter = registry.get(self.bind_scatter).expect("bind_scatter");

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("spatial_grid clear"),
                timestamp_writes: None,
            });
            pass.set_pipeline(p_clear);
            pass.set_bind_group(0, g_clear, &[]);
            pass.dispatch_workgroups(wg256, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("spatial_grid count"),
                timestamp_writes: None,
            });
            pass.set_pipeline(p_count);
            pass.set_bind_group(0, g_count, &[]);
            pass.dispatch_workgroups(wg_ent, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("spatial_grid linearize"),
                timestamp_writes: None,
            });
            pass.set_pipeline(p_linearize);
            pass.set_bind_group(0, g_linearize, &[]);
            pass.dispatch_workgroups(wg256, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("spatial_grid scan"),
                timestamp_writes: None,
            });
            pass.set_pipeline(p_scan);
            pass.set_bind_group(0, g_scan, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("spatial_grid init_heads"),
                timestamp_writes: None,
            });
            pass.set_pipeline(p_init);
            pass.set_bind_group(0, g_init, &[]);
            pass.dispatch_workgroups(wg256, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("spatial_grid scatter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(p_scatter);
            pass.set_bind_group(0, g_scatter, &[]);
            pass.dispatch_workgroups(wg_ent, 1, 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BufferUsage;
    use crate::Renderer;
    use pollster::FutureExt;

    #[test]
    fn spatial_grid_bins_entities() {
        let renderer = match Renderer::new().block_on() {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skip spatial_grid test: {e}");
                return;
            }
        };
        let mut registry = ResourceRegistry::default();

        // 2×2×2 cells of size 1, origin at origin.
        let dims = [2u32, 2, 2];
        let n_ent = 4u32;
        let params = SpatialGridParams::new([0.0, 0.0, 0.0], 1.0, dims, n_ent);
        let grid = SpatialGridGpu::new(
            &renderer,
            &mut registry,
            SpatialGridConfig {
                params,
                max_entities: 16,
            },
        )
        .expect("SpatialGridGpu::new");

        let rb_counts = renderer
            .create_gpu_buffer::<u32>()
            .label("rb_counts")
            .capacity(8)
            .usage(BufferUsage::Readback)
            .build(&mut registry)
            .expect("rb_counts");
        let rb_offsets = renderer
            .create_gpu_buffer::<u32>()
            .label("rb_offsets")
            .capacity(9)
            .usage(BufferUsage::Readback)
            .build(&mut registry)
            .expect("rb_offsets");
        let rb_sorted = renderer
            .create_gpu_buffer::<u32>()
            .label("rb_sorted")
            .capacity(16)
            .usage(BufferUsage::Readback)
            .build(&mut registry)
            .expect("rb_sorted");

        let positions = [
            EntityPosition {
                position: [0.5, 0.5, 0.5],
                _pad: 0.0,
            },
            EntityPosition {
                position: [1.5, 0.5, 0.5],
                _pad: 0.0,
            },
            EntityPosition {
                position: [0.5, 1.5, 0.5],
                _pad: 0.0,
            },
            EntityPosition {
                position: [1.5, 1.5, 1.5],
                _pad: 0.0,
            },
        ];
        renderer
            .write_buffer(grid.positions, &positions, &registry)
            .expect("write positions");

        let mut encoder = renderer
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("spatial_grid test"),
            });
        grid.encode_rebuild(&mut encoder, &registry);

        let src_counts = registry.get(grid.counts_linear).expect("counts");
        let dst_counts = registry.get(rb_counts.handle()).expect("rb_counts");
        encoder.copy_buffer_to_buffer(src_counts, 0, dst_counts, 0, 8 * 4);

        let src_off = registry.get(grid.cell_offsets).expect("offsets");
        let dst_off = registry.get(rb_offsets.handle()).expect("rb_offsets");
        encoder.copy_buffer_to_buffer(src_off, 0, dst_off, 0, 9 * 4);

        let src_sorted = registry.get(grid.sorted_entity_ids).expect("sorted");
        let dst_sorted = registry.get(rb_sorted.handle()).expect("rb_sorted");
        encoder.copy_buffer_to_buffer(src_sorted, 0, dst_sorted, 0, 16 * 4);

        renderer.queue().submit([encoder.finish()]);
        renderer.device().poll(wgpu::PollType::wait_indefinitely()).unwrap();

        let counts = renderer
            .read_buffer::<u32>(rb_counts.handle(), &registry)
            .expect("read counts");
        assert_eq!(counts.len(), 8);
        assert_eq!(counts[0], 1);
        assert_eq!(counts[1], 1);
        assert_eq!(counts[2], 1);
        assert_eq!(counts[3], 0);
        assert_eq!(counts[4], 0);
        assert_eq!(counts[5], 0);
        assert_eq!(counts[6], 0);
        assert_eq!(counts[7], 1);

        let offsets = renderer
            .read_buffer::<u32>(rb_offsets.handle(), &registry)
            .expect("read offsets");
        assert_eq!(offsets.len(), 9);
        assert_eq!(offsets[8], 4);

        let sorted = renderer
            .read_buffer::<u32>(rb_sorted.handle(), &registry)
            .expect("read sorted");
        let mut first4: Vec<u32> = sorted[..4].to_vec();
        first4.sort_unstable();
        assert_eq!(first4, vec![0, 1, 2, 3]);
    }
}
