# Triad

Triad is a GPU-first drone simulation workspace for running, visualizing, and training vectorized environments. The repo combines a shared Rust simulation core, windowed and headless runners, and Python bindings used for rollout collection and PPO-based training.

The rendering crates are still an important part of the workspace, but they now exist in service of the simulation stack rather than as a generic renderer refactor by itself.

## What Lives Here

- **`triad-sim`**: shared simulation core, environment state, observations, rewards, reset parameters, and course generation
- **`triad-visualizer`**: interactive windowed viewer for inspecting many simulation environments, replaying behavior, and loading PPO checkpoints
- **`triad-showcase`**: scripted drone showcase used for demo/site presentation flows
- **`triad-headless`**: offscreen GPU render/readback demo for headless execution paths
- **`triad-py`**: Rust C-ABI bindings plus Python wrappers for vector env access, rollout collection, curriculum helpers, and PPO training/evaluation
- **`triad-gpu`**: lower-level `wgpu` rendering and frame-graph infrastructure used by the apps and sim tooling
- **`triad-window`**: reusable window, input, camera, and `egui` integration for windowed tools
- **`triad-app`**: small placeholder app from the earlier workspace refactor

## Repository Focus

The main direction of the repo is:

- build a reusable GPU simulation core for drone racing/navigation tasks
- expose that core to Python for fast RL training loops
- provide native Rust tools for visualization, inspection, and demos
- keep the rendering/windowing layers reusable, but secondary to the sim and training workflow
