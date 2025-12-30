use glam::{Mat4, Vec3};

/// CPU-side representation of a simple point for GPU rendering.
/// Matches the layout used by `point_vertex.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct PointPrimitive {
    /// xyz position.
    pub position: [f32; 3],
    /// Point size (radius in world units).
    pub size: f32,
    /// rgba color (linear 0-1).
    pub color: [f32; 4],
}

impl PointPrimitive {
    /// Create a new point primitive.
    pub fn new(position: Vec3, size: f32, color: Vec3, opacity: f32) -> Self {
        Self {
            position: [position.x, position.y, position.z],
            size,
            color: [color.x, color.y, color.z, opacity],
        }
    }

    /// Get position as Vec3.
    pub fn position(&self) -> Vec3 {
        Vec3::from_slice(&self.position)
    }

    /// Get color as Vec3.
    pub fn color_rgb(&self) -> Vec3 {
        Vec3::new(self.color[0], self.color[1], self.color[2])
    }

    /// Get opacity.
    pub fn opacity(&self) -> f32 {
        self.color[3]
    }
}

impl Default for PointPrimitive {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            size: 0.01,
            color: [0.8, 0.8, 0.8, 1.0],
        }
    }
}

impl From<&triad_data::Point> for PointPrimitive {
    fn from(point: &triad_data::Point) -> Self {
        Self::new(point.position, 0.01, point.color, 1.0)
    }
}

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
    pub fn new(
        position: Vec3,
        color: Vec3,
        opacity: f32,
        rotation: [f32; 4],
        scale: Vec3,
    ) -> Self {
        Self {
            position: [position.x, position.y, position.z],
            _pad0: 0.0,
            color_opacity: [color.x, color.y, color.z, opacity],
            rotation,
            scale: [scale.x, scale.y, scale.z, 0.0],
        }
    }

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

impl From<&triad_data::Gaussian> for GaussianPoint {
    fn from(gaussian: &triad_data::Gaussian) -> Self {
        Self::new(
            gaussian.position,
            gaussian.color,
            gaussian.opacity,
            gaussian.rotation,
            gaussian.scale,
        )
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

/// CPU-side representation of a triangle primitive for Triangle Splatting+.
/// Matches the layout used by `triangle_vertex.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct TrianglePrimitive {
    /// First vertex position (xyz) + padding.
    pub v0: [f32; 4],
    /// Second vertex position (xyz) + padding.
    pub v1: [f32; 4],
    /// Third vertex position (xyz) + padding.
    pub v2: [f32; 4],
    /// RGB color (linear 0-1) and opacity in w.
    pub color: [f32; 4],
}

impl TrianglePrimitive {
    /// Create a new triangle primitive from three vertices, color, and opacity.
    pub fn new(v0: Vec3, v1: Vec3, v2: Vec3, color: Vec3, opacity: f32) -> Self {
        Self {
            v0: [v0.x, v0.y, v0.z, 0.0],
            v1: [v1.x, v1.y, v1.z, 0.0],
            v2: [v2.x, v2.y, v2.z, 0.0],
            color: [color.x, color.y, color.z, opacity],
        }
    }

    /// Get vertex 0 as Vec3.
    pub fn vertex0(&self) -> Vec3 {
        Vec3::new(self.v0[0], self.v0[1], self.v0[2])
    }

    /// Get vertex 1 as Vec3.
    pub fn vertex1(&self) -> Vec3 {
        Vec3::new(self.v1[0], self.v1[1], self.v1[2])
    }

    /// Get vertex 2 as Vec3.
    pub fn vertex2(&self) -> Vec3 {
        Vec3::new(self.v2[0], self.v2[1], self.v2[2])
    }

    /// Compute the center of the triangle.
    pub fn center(&self) -> Vec3 {
        (self.vertex0() + self.vertex1() + self.vertex2()) / 3.0
    }

    /// Get the color as Vec3.
    pub fn color(&self) -> Vec3 {
        Vec3::new(self.color[0], self.color[1], self.color[2])
    }

    /// Get the opacity.
    pub fn opacity(&self) -> f32 {
        self.color[3]
    }

    /// Compute the normal of the triangle (not normalized).
    pub fn normal(&self) -> Vec3 {
        let e1 = self.vertex1() - self.vertex0();
        let e2 = self.vertex2() - self.vertex0();
        e1.cross(e2)
    }
}

impl From<&triad_data::Triangle> for TrianglePrimitive {
    fn from(triangle: &triad_data::Triangle) -> Self {
        Self::new(
            triangle.v0,
            triangle.v1,
            triangle.v2,
            triangle.color,
            triangle.opacity,
        )
    }
}
