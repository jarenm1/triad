//! Triangle Splatting+ example
//!
//! This example demonstrates Triangle Splatting+ rendering using the triad-window
//! render delegate system. It loads PLY files and renders them as triangles with
//! soft edge falloff.
//!
//! Usage:
//!   cargo run --example triangle_splatting -- <path_to_ply>
//!
//! If the PLY file contains face data, those faces are used directly.
//! Otherwise, Delaunay triangulation is applied to generate triangles from the point cloud.

use glam::Vec3;
use std::path::PathBuf;
use triad_gpu::{SceneBounds, TrianglePrimitive, ply_loader};
use triad_data::triangulation;
use triad_window::{
    CameraControl, CameraIntent, CameraPose, FrameUpdate, InputState, IntentMode, KeyCode,
    MouseButton, PhysicalKey, TriangleDelegate, TriangleInitData,
};

/// Blender-like orbit controller: orbit around a focus point, wheel to zoom, shift+drag to pan focus.
struct OrbitCam {
    focus: Vec3,
    yaw: f32,
    pitch: f32,
    radius: f32,
    rotation_sensitivity: f32,
    pan_sensitivity: f32,
    zoom_sensitivity: f32,
}

impl OrbitCam {
    fn new(focus: Vec3, radius: f32) -> Self {
        Self {
            focus,
            yaw: 0.0,
            pitch: 0.0,
            radius,
            rotation_sensitivity: 0.005,
            pan_sensitivity: 0.0025,
            zoom_sensitivity: 0.5,
        }
    }

    fn pose_from_state(&self) -> CameraPose {
        let dir = glam::Quat::from_euler(glam::EulerRot::YXZ, self.yaw, self.pitch, 0.0) * -Vec3::Z;
        let position = self.focus - dir.normalize() * self.radius;
        let forward = (self.focus - position).normalize_or_zero();
        let yaw = forward.x.atan2(-forward.z);
        let pitch = forward
            .y
            .asin()
            .clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
        CameraPose {
            position,
            yaw,
            pitch,
            roll: 0.0,
        }
    }
}

impl CameraControl for OrbitCam {
    fn update(
        &mut self,
        _dt: f32,
        input: &InputState,
        _current: &CameraPose,
    ) -> Option<CameraIntent> {
        let delta = input.mouse_delta();
        let shift = input.key_down(PhysicalKey::Code(KeyCode::ShiftLeft))
            || input.key_down(PhysicalKey::Code(KeyCode::ShiftRight));

        if input.mouse_button_down(MouseButton::Left) && !shift {
            self.yaw -= delta.x * self.rotation_sensitivity;
            self.pitch = (self.pitch - delta.y * self.rotation_sensitivity).clamp(
                -std::f32::consts::FRAC_PI_2 + 0.01,
                std::f32::consts::FRAC_PI_2 - 0.01,
            );
        } else if input.mouse_button_down(MouseButton::Left) && shift {
            // Pan the focus point in view plane.
            let right = glam::Quat::from_euler(glam::EulerRot::YXZ, self.yaw, 0.0, 0.0) * Vec3::X;
            let up = Vec3::Y;
            let pan = (-delta.x * self.pan_sensitivity * self.radius) * right
                + (delta.y * self.pan_sensitivity * self.radius) * up;
            self.focus += pan;
        }

        if input.scroll_delta() != 0.0 {
            let factor = 1.0 - input.scroll_delta() * 0.1 * self.zoom_sensitivity;
            self.radius = (self.radius * factor).clamp(0.2, 500.0);
        }

        Some(CameraIntent {
            pose: self.pose_from_state(),
            mode: IntentMode::Override,
        })
    }

    fn on_reset(&mut self, _pose: &CameraPose) {
        self.focus = Vec3::ZERO;
        self.yaw = 0.0;
        self.pitch = 0.0;
        self.radius = 10.0;
    }
}

fn compute_scene_center(ply_path: &std::path::Path) -> Result<Vec3, Box<dyn std::error::Error>> {
    let ply_path_str = ply_path
        .to_str()
        .ok_or_else(|| format!("PLY path {:?} is not valid UTF-8", ply_path))?;

    let triangles: Vec<TrianglePrimitive> =
        if ply_loader::ply_has_faces(ply_path_str).unwrap_or(false) {
            ply_loader::load_triangles_from_ply(ply_path_str)?
        } else {
            let vertices = ply_loader::load_vertices_from_ply(ply_path_str)?;
            let positions: Vec<Vec3> = vertices.iter().map(|v| v.position).collect();
            let triangle_indices = triangulation::triangulate_points(&positions);

            let mut triangles = Vec::with_capacity(triangle_indices.len());
            for [i0, i1, i2] in triangle_indices {
                let v0 = &vertices[i0];
                let v1 = &vertices[i1];
                let v2 = &vertices[i2];
                let avg_color = (v0.color + v1.color + v2.color) / 3.0;
                let avg_opacity = (v0.opacity + v1.opacity + v2.opacity) / 3.0;
                triangles.push(TrianglePrimitive::new(
                    v0.position, v1.position, v2.position, avg_color, avg_opacity,
                ));
            }
            triangles
        };

    if triangles.is_empty() {
        return Err("No triangles could be generated from the PLY file".into());
    }

    let bounds = SceneBounds::from_positions(
        triangles
            .iter()
            .flat_map(|t| [t.vertex0(), t.vertex1(), t.vertex2()]),
    );
    Ok(bounds.center)
}

fn main() {
    let ply_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .expect("failed to find ply file, or not provided.");

    let scene_center = compute_scene_center(&ply_path).unwrap_or(Vec3::ZERO);

    if let Err(err) = triad_window::run_with_delegate_config::<TriangleDelegate, _>(
        "Triad Triangle Splatting+ Viewer",
        TriangleInitData::new(&ply_path),
        move |controls| {
            let target = scene_center;
            controls
                .clear_controllers()
                .single_active(true)
                .add_controller_with_priority(Box::new(OrbitCam::new(target, 10.0)), 0)
                .on_frame(move |ctx: FrameUpdate| {
                    if ctx.input.just_pressed(PhysicalKey::Code(KeyCode::KeyR)) {
                        let mut pose = *ctx.pose;
                        pose.position = target + Vec3::new(0.0, 0.0, 10.0);
                        let forward = (target - pose.position).normalize_or_zero();
                        pose.yaw = forward.x.atan2(-forward.z);
                        pose.pitch = forward
                            .y
                            .asin()
                            .clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
                        pose.roll = 0.0;
                        *ctx.reset_request = Some(pose);
                    }
                });
        },
    ) {
        eprintln!("triangle_splatting failed: {err}");
    }
}
