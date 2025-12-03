use glam::{Mat4, Vec3};

/// CPU-side representation of a Gaussian splat instance.
/// Matches the layout used by `gaussian_vertex.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct GaussianPoint {
    /// xyz center position.
    pub position: [f32; 3],
    /// Padding to align color_opacity to 16 bytes.
    pub _pad0: f32,
    /// rgb color (linear 0-1) and opacity in w.
    pub color_opacity: [f32; 4],
    /// Quaternion (x, y, z, w) describing orientation.
    pub rotation: [f32; 4],
    /// Per-axis scale (x, y, z) + padding.
    pub scale: [f32; 4],
}

impl GaussianPoint {
    pub fn position(&self) -> Vec3 {
        Vec3::from_slice(&self.position)
    }

    pub fn color(&self) -> Vec3 {
        Vec3::new(
            self.color_opacity[0],
            self.color_opacity[1],
            self.color_opacity[2],
        )
    }

    pub fn opacity(&self) -> f32 {
        self.color_opacity[3]
    }

    pub fn rotation(&self) -> [f32; 4] {
        self.rotation
    }

    pub fn scale(&self) -> Vec3 {
        Vec3::new(self.scale[0], self.scale[1], self.scale[2])
    }
}

/// Shared camera uniform layout between host and shader.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniforms {
    pub view_matrix: [[f32; 4]; 4],
    pub proj_matrix: [[f32; 4]; 4],
    pub view_pos: [f32; 3],
    pub _padding: f32,
}

impl CameraUniforms {
    pub fn from_matrices(view: Mat4, proj: Mat4, eye: Vec3) -> Self {
        Self {
            view_matrix: view.to_cols_array_2d(),
            proj_matrix: proj.to_cols_array_2d(),
            view_pos: [eye.x, eye.y, eye.z],
            _padding: 0.0,
        }
    }
}
