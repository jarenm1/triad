# Visualizer Refactor

## Goal

`triad-visualizer` is a simulation inspection and course-authoring tool. It should not own PPO training, checkpoint loading, policy-server IPC, or task-specific curriculum logic.

The reusable RL environment should be built below the visualizer and exposed through Rust/Python APIs. The visualizer should consume the same course/procgen inputs as training, then display the resolved GPU simulation state.

## Performance Boundary

Keep the simulation hot path GPU-first:

- Per-frame/per-step work should remain GPU stepping, GPU buffer writes for actions/resets, and explicit readbacks already needed for UI inspection.
- Procgen and course editing may run on the CPU only when inputs change: file hot reload, editor save, or explicit parameter changes.
- After a course/procgen change, the CPU should compile the source spec into packed layout/gate/obstacle data once, upload it to GPU buffers, then let the GPU sim advance from those buffers.
- Do not add CPU geometry regeneration, CPU collision checks, or CPU reward evaluation to the render frame loop.

## Target Flow

```text
course file / procgen params / editor edits
        -> resolved course geometry
        -> packed GPU buffers
        -> GPU simulation
        -> visualizer readback for inspection only
```

This keeps hot reload responsive without compromising training throughput.

## Refactor Sequence

1. Keep `triad-visualizer` free of PPO and checkpoint dependencies.
2. Move course/procgen descriptions into a reusable crate or module with serialization.
3. Compile course/procgen specs into packed GPU layout buffers on reload or edit.
4. Move drone-specific curriculum, reward, and PPO task metadata into a drone RL environment layer.
5. Let Python training and the visualizer load the same course/procgen sources.
