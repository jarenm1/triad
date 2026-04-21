# Triad

Triad is being refactored into a generic renderer and particle simulation workspace.

The current direction is:

- `triad-gpu` as the render core on top of `wgpu`
- `triad-window` as the reusable windowing, input, and viewer shell
- `triad-app` and the old splatting/reconstruction infrastructure to be removed or replaced

This repository is in transition. Some crates and files still reflect the older triangle/gaussian splatting project and should be treated as legacy code until they are either deleted or rewritten.

## Workspace

- **`triad-gpu`**: render core, frame graph, GPU resource management, shader/pipeline utilities
- **`triad-window`**: window lifecycle, input handling, camera controls, egui integration
- **`triad-app`**: legacy application crate slated for replacement
- **`triad-data`**: legacy data/import code slated for removal or extraction
- **`triad-train`**: legacy reconstruction/training code slated for removal
- **`triad-capture`**: legacy capture pipeline pending evaluation during the refactor

## Refactor Goal

The immediate goal is to simplify the workspace around a clean rendering core that can support:

- generic scene rendering
- particle simulation and visualization
- future app/tooling built on stable engine boundaries
