use glam::{Mat4, Vec3};

/// CPU-side representation of a Gaussian splat instance.
/// Matches the layout used by `gaussian_vertex.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct GaussianPoint {
    /// xyz center and isotropic radius packed into w.
    pub position_radius: [f32; 4],
    /// rgb color (linear 0-1) and opacity in w.
    pub color_opacity: [f32; 4],
    /// Quaternion (x, y, z, w) describing orientation.
    pub rotation: [f32; 4],
    /// Per-axis scale; w reserved for padding/extra scalar.
    pub scale: [f32; 4],
}

impl GaussianPoint {
    pub fn position(&self) -> Vec3 {
        Vec3::from_slice(&self.position_radius[..3])
    }

    pub fn radius(&self) -> f32 {
        self.position_radius[3]
    }

    pub fn rotation(&self) -> [f32; 4] {
        self.rotation
    }

    pub fn scale(&self) -> [f32; 4] {
        self.scale
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

