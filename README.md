# Triad

Triangular splatting rendering application for use in realtime SLAM and sim applications.

![Showcase image for triangle like splatting](./showcase.png "Triangle Like Splatting PLY file render")

## Crates

- **`triad-data`**: GPU-agnostic data loading and geometric processing (PLY files, triangulation)
- **`triad-gpu`**: Low-level GPU rendering infrastructure (wgpu, frame graph, resource management)
- **`triad-window`**: Window management and application framework
- **`triad-train`**: Training infrastructure and data ingest pipeline for 4D Gaussian Splatting

## Features

- Triangle Splatting+ rendering
- 3D Gaussian Splatting rendering
- 4D Gaussian Splatting infrastructure (time-varying scenes)
- Real-time scene reconstruction from point clouds
- Data ingest pipelines (cameras, point clouds, PLY files)
- Frame graph system for efficient rendering

## Examples

- `triangle_splatting`: Triangle Splatting+ viewer
- `realtime_reconstruction`: Real-time scene reconstruction example

## Resources
- [Triangle Splatting+](https://arxiv.org/abs/2509.25122)
- [3DGS Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079)
- [4D Gaussian Splatting Research](./docs/4d_gaussian_research.md)