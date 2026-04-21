use crate::error::ComputePassError;
use crate::frame_graph::pass::{Pass, PassBuilder, PassContext};
use crate::frame_graph::{Handle, ResourceType};

#[derive(Debug, Clone, Copy)]
pub enum ComputeDispatch {
    Direct {
        x: u32,
        y: u32,
        z: u32,
    },
    Indirect {
        buffer: Handle<wgpu::Buffer>,
        offset: u64,
    },
}

impl ComputeDispatch {
    pub fn direct(x: u32, y: u32, z: u32) -> Self {
        Self::Direct { x, y, z }
    }

    pub fn indirect(buffer: Handle<wgpu::Buffer>, offset: u64) -> Self {
        Self::Indirect { buffer, offset }
    }
}

#[derive(Debug, Clone, Copy)]
struct BoundBindGroup {
    index: u32,
    handle: Handle<wgpu::BindGroup>,
}

#[derive(Debug)]
struct ComputeDispatchPass {
    name: String,
    pipeline: Handle<wgpu::ComputePipeline>,
    bind_groups: Vec<BoundBindGroup>,
    dispatch: ComputeDispatch,
}

impl Pass for ComputeDispatchPass {
    fn name(&self) -> &str {
        &self.name
    }

    fn execute(&self, ctx: &PassContext) -> wgpu::CommandBuffer {
        let pipeline = ctx
            .get_compute_pipeline(self.pipeline)
            .expect("compute pipeline handle missing from registry");

        let mut encoder = ctx.create_command_encoder(Some(&self.name));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&self.name),
                timestamp_writes: None,
            });

            pass.set_pipeline(pipeline);

            for bind_group in &self.bind_groups {
                let resource = ctx
                    .get_bind_group(bind_group.handle)
                    .expect("bind group handle missing from registry");
                pass.set_bind_group(bind_group.index, resource, &[]);
            }

            match self.dispatch {
                ComputeDispatch::Direct { x, y, z } => {
                    pass.dispatch_workgroups(x, y, z);
                }
                ComputeDispatch::Indirect { buffer, offset } => {
                    let args = ctx
                        .get_buffer(buffer)
                        .expect("indirect dispatch buffer missing from registry");
                    pass.dispatch_workgroups_indirect(args, offset);
                }
            }
        }

        encoder.finish()
    }
}

pub struct ComputePassBuilder {
    name: String,
    reads: Vec<u64>,
    writes: Vec<u64>,
    pipeline: Option<Handle<wgpu::ComputePipeline>>,
    bind_groups: Vec<BoundBindGroup>,
    dispatch: Option<ComputeDispatch>,
}

impl ComputePassBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            reads: Vec::new(),
            writes: Vec::new(),
            pipeline: None,
            bind_groups: Vec::new(),
            dispatch: None,
        }
    }

    pub fn read<T: ResourceType>(mut self, handle: Handle<T>) -> Self {
        self.reads.push(handle.id());
        self
    }

    pub fn write<T: ResourceType>(mut self, handle: Handle<T>) -> Self {
        self.writes.push(handle.id());
        self
    }

    pub fn read_write<T: ResourceType>(mut self, handle: Handle<T>) -> Self {
        self.reads.push(handle.id());
        self.writes.push(handle.id());
        self
    }

    pub fn with_pipeline(mut self, pipeline: Handle<wgpu::ComputePipeline>) -> Self {
        self.pipeline = Some(pipeline);
        self
    }

    pub fn with_bind_group(mut self, index: u32, bind_group: Handle<wgpu::BindGroup>) -> Self {
        self.bind_groups.push(BoundBindGroup {
            index,
            handle: bind_group,
        });
        self
    }

    pub fn dispatch(mut self, x: u32, y: u32, z: u32) -> Self {
        self.dispatch = Some(ComputeDispatch::direct(x, y, z));
        self
    }

    pub fn dispatch_indirect(mut self, buffer: Handle<wgpu::Buffer>, offset: u64) -> Self {
        self.dispatch = Some(ComputeDispatch::indirect(buffer, offset));
        self
    }

    pub fn build(self) -> Result<PassBuilder, ComputePassError> {
        let pipeline = self.pipeline.ok_or(ComputePassError::MissingPipeline)?;
        let dispatch = self.dispatch.ok_or(ComputePassError::MissingDispatch)?;

        let mut builder = PassBuilder::new(self.name.clone());
        for read in self.reads {
            builder.read_handle_id(read);
        }
        for write in self.writes {
            builder.write_handle_id(write);
        }

        Ok(builder.with_pass(Box::new(ComputeDispatchPass {
            name: self.name,
            pipeline,
            bind_groups: self.bind_groups,
            dispatch,
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FrameGraph, Renderer, ResourceRegistry};
    use pollster::FutureExt;

    const COMPUTE_SHADER: &str = r#"
        @compute @workgroup_size(1)
        fn cs_main() {}
    "#;

    #[test]
    fn test_compute_pass_builder_requires_pipeline() {
        let err = ComputePassBuilder::new("simulate")
            .dispatch(1, 1, 1)
            .build()
            .err()
            .expect("builder should require a pipeline");

        assert!(matches!(err, ComputePassError::MissingPipeline));
    }

    #[test]
    fn test_compute_pass_builder_requires_dispatch() {
        let pipeline = Handle::<wgpu::ComputePipeline>::next();
        let err = ComputePassBuilder::new("simulate")
            .with_pipeline(pipeline)
            .build()
            .err()
            .expect("builder should require a dispatch");

        assert!(matches!(err, ComputePassError::MissingDispatch));
    }

    #[test]
    fn test_compute_pass_executes_direct_dispatch() {
        let renderer = Renderer::new().block_on().expect("renderer");
        let mut registry = ResourceRegistry::default();

        let shader = renderer
            .create_shader_module()
            .label("compute")
            .with_wgsl_source(COMPUTE_SHADER)
            .build(&mut registry)
            .expect("shader");

        let pipeline = renderer
            .create_compute_pipeline()
            .with_label("compute pipeline")
            .with_compute_shader(shader)
            .build(&mut registry)
            .expect("pipeline");

        let pass = ComputePassBuilder::new("simulate")
            .with_pipeline(pipeline)
            .dispatch(1, 1, 1)
            .build()
            .expect("pass");

        let mut graph = FrameGraph::new();
        graph.add_pass(pass);
        let mut executable = graph.build().expect("graph");

        let command_buffers =
            executable.execute_no_submit(renderer.device(), renderer.queue(), &registry);
        assert_eq!(command_buffers.len(), 1);
    }
}
