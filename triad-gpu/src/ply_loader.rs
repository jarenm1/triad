use crate::GaussianPoint;
use glam::Vec3;
use serde::Deserialize;
use std::collections::HashMap;
use tracing::{debug, info, warn};

// Face structure for PLY files (we ignore this data for Gaussian splatting)
// Note: serde_ply handles list properties automatically
#[derive(Deserialize, Debug)]
struct PlyFace {
    vertex_indices: Vec<i32>,
}

// Use a map-based structure for PLY file deserialization
// serde_ply requires rows to be deserialized as maps for flexible PLY files
// Use serde_json::Value as a flexible value type
use serde_json::Value as JsonValue;

#[derive(Deserialize, Debug)]
struct PlyFile {
    #[serde(rename = "vertex")]
    vertex: Vec<HashMap<String, JsonValue>>,
    #[serde(default, rename = "face", skip_serializing_if = "Vec::is_empty")]
    face: Vec<PlyFace>, // Optional - many PLY files have face data we can ignore
}

/// Load Gaussian points from a PLY file
/// Supports ASCII and binary PLY formats via serde_ply
/// For Gaussian splatting, expects: x, y, z, scale_0/1/2, rot_0/1/2/3, red/green/blue, opacity
/// For regular point clouds, only requires: x, y, z, red/green/blue
pub fn load_gaussians_from_ply(
    path: &str,
) -> Result<Vec<GaussianPoint>, Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::BufReader;

    debug!("Loading PLY file: {}", path);
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    info!("Parsing PLY file (this may take a while for large files)...");
    let ply_data: PlyFile = serde_ply::from_reader(reader).map_err(|e| {
        warn!("Failed to parse PLY file: {}", e);
        let error_msg = format!("PLY parsing error: {}", e);
        // Provide helpful hints for common errors
        if error_msg.contains("missing field") {
            warn!("Hint: This PLY file may have a different structure than expected.");
            warn!("Supported fields: x, y, z, [red/green/blue or r/g/b], [scale_*], [rot_* or q*], [opacity/alpha]");
            warn!("Missing fields will use defaults (light gray color, adaptive scale, identity rotation)");
        }
        error_msg
    })?;

    info!(
        "PLY file parsed successfully: {} vertices, {} faces",
        ply_data.vertex.len(),
        ply_data.face.len()
    );
    debug!("PLY file has {} vertices", ply_data.vertex.len());

    // Calculate bounding box for adaptive scaling
    let mut bbox_min = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
    let mut bbox_max = Vec3::new(f32::MIN, f32::MIN, f32::MIN);

    // Helper function to extract f32 from JsonValue (used for bounding box calculation)
    fn get_f32_for_bbox(prop: Option<&JsonValue>) -> Option<f32> {
        prop.and_then(|v| match v {
            JsonValue::Number(n) => n.as_f64().map(|f| f as f32),
            _ => None,
        })
    }

    for vertex in &ply_data.vertex {
        if let (Some(x), Some(y), Some(z)) = (
            get_f32_for_bbox(vertex.get("x")),
            get_f32_for_bbox(vertex.get("y")),
            get_f32_for_bbox(vertex.get("z")),
        ) {
            let pos = Vec3::new(x, y, z);
            bbox_min = bbox_min.min(pos);
            bbox_max = bbox_max.max(pos);
        }
    }

    let bbox_size = bbox_max - bbox_min;
    let max_dim = bbox_size.x.max(bbox_size.y).max(bbox_size.z);

    // Adaptive scale: use a fraction of the bounding box size
    // For point clouds, we want Gaussians to overlap nicely
    // Try a medium scale - not too big (causes weird blending) but not too small (can't see points)
    let default_scale = if max_dim > 0.0 {
        max_dim / 5000.0 // Medium scale for point clouds
    } else {
        0.01 // Fallback for degenerate cases
    };
    info!(
        "Computed default scale: {:.6} (bbox size: {:.2})",
        default_scale, max_dim
    );

    let mut gaussians = Vec::new();

    // Helper functions to extract values from JsonValue
    fn get_f32(prop: Option<&JsonValue>) -> Option<f32> {
        prop.and_then(|v| match v {
            JsonValue::Number(n) => n.as_f64().map(|f| f as f32),
            _ => None,
        })
    }

    fn get_u8(prop: Option<&JsonValue>) -> Option<u8> {
        prop.and_then(|v| match v {
            JsonValue::Number(n) => {
                if let Some(u) = n.as_u64() {
                    Some(u as u8)
                } else if let Some(i) = n.as_i64() {
                    Some(i as u8)
                } else {
                    None
                }
            }
            _ => None,
        })
    }

    for (i, vertex) in ply_data.vertex.iter().enumerate() {
        // Extract position (required)
        let x = get_f32(vertex.get("x"))
            .ok_or_else(|| format!("Missing 'x' coordinate in PLY file at vertex {}", i))?;
        let y = get_f32(vertex.get("y"))
            .ok_or_else(|| format!("Missing 'y' coordinate in PLY file at vertex {}", i))?;
        let z = get_f32(vertex.get("z"))
            .ok_or_else(|| format!("Missing 'z' coordinate in PLY file at vertex {}", i))?;
        let position = Vec3::new(x, y, z);

        // Parse optional per-axis scale to derive a simple isotropic radius.
        // 3D Gaussian splatting datasets often store log-scales; regular point
        // clouds typically omit them, in which case we fall back to the adaptive
        // default computed from the scene bounds.
        let scale_vec = if let (Some(s0), Some(s1), Some(s2)) = (
            get_f32(vertex.get("scale_0")),
            get_f32(vertex.get("scale_1")),
            get_f32(vertex.get("scale_2")),
        ) {
            Vec3::new(s0.exp(), s1.exp(), s2.exp())
        } else if let (Some(sx), Some(sy), Some(sz)) = (
            get_f32(vertex.get("scale_x")),
            get_f32(vertex.get("scale_y")),
            get_f32(vertex.get("scale_z")),
        ) {
            Vec3::new(sx, sy, sz)
        } else if let Some(uniform_scale) = get_f32(vertex.get("scale")) {
            Vec3::splat(uniform_scale)
        } else {
            Vec3::splat(default_scale)
        };

        // Parse rotation quaternion.
        let rotation = if let (Some(r0), Some(r1), Some(r2), Some(r3)) = (
            get_f32(vertex.get("rot_0")),
            get_f32(vertex.get("rot_1")),
            get_f32(vertex.get("rot_2")),
            get_f32(vertex.get("rot_3")),
        ) {
            [r0, r1, r2, r3]
        } else if let (Some(rx), Some(ry), Some(rz), Some(rw)) = (
            get_f32(vertex.get("rot_x")),
            get_f32(vertex.get("rot_y")),
            get_f32(vertex.get("rot_z")),
            get_f32(vertex.get("rot_w")),
        ) {
            [rx, ry, rz, rw]
        } else if let (Some(qx), Some(qy), Some(qz), Some(qw)) = (
            get_f32(vertex.get("qx")),
            get_f32(vertex.get("qy")),
            get_f32(vertex.get("qz")),
            get_f32(vertex.get("qw")),
        ) {
            [qx, qy, qz, qw]
        } else {
            [0.0, 0.0, 0.0, 1.0]
        };

        // Parse color - PLY files may use uchar (0-255) for colors
        // Handle multiple color field name variations and missing colors
        let color = if let (Some(r), Some(g), Some(b)) = (
            get_u8(vertex.get("red")),
            get_u8(vertex.get("green")),
            get_u8(vertex.get("blue")),
        ) {
            // Standard red/green/blue fields (uchar 0-255)
            Vec3::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0)
        } else if let (Some(r), Some(g), Some(b)) = (
            get_u8(vertex.get("r")),
            get_u8(vertex.get("g")),
            get_u8(vertex.get("b")),
        ) {
            // Alternative r/g/b fields (uchar 0-255)
            Vec3::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0)
        } else {
            // No color information - use a default color (light gray)
            if i < 10 {
                warn!(
                    "No color information found for vertex {}, using default color",
                    i
                );
            }
            Vec3::new(0.8, 0.8, 0.8) // Light gray default
        };

        // Parse opacity. Treat values outside [0, 1] as logits (common for 3D GS).
        let raw_opacity = get_f32(vertex.get("opacity"))
            .or_else(|| get_f32(vertex.get("alpha")))
            .unwrap_or(1.0);
        let opacity = if (0.0..=1.0).contains(&raw_opacity) {
            raw_opacity
        } else {
            (1.0 / (1.0 + (-raw_opacity).exp())).clamp(0.0, 1.0)
        };

        gaussians.push(GaussianPoint {
            position: [position.x, position.y, position.z],
            color_opacity: [color.x, color.y, color.z, opacity],
            rotation,
            scale: [scale_vec.x, scale_vec.y, scale_vec.z],
        });

        // Debug: verify color is RGB not BGR
        if i < 3 {
            debug!(
                "Gaussian {} color check: R={:.3}, G={:.3}, B={:.3}",
                i, color.x, color.y, color.z
            );
        }

        // Log first few vertices for debugging
        if i < 5 {
            let raw_color_str = if let (Some(r), Some(g), Some(b)) = (
                get_u8(vertex.get("red")),
                get_u8(vertex.get("green")),
                get_u8(vertex.get("blue")),
            ) {
                format!("r:{}, g:{}, b:{}", r, g, b)
            } else if let (Some(r), Some(g), Some(b)) = (
                get_u8(vertex.get("r")),
                get_u8(vertex.get("g")),
                get_u8(vertex.get("b")),
            ) {
                format!("r:{}, g:{}, b:{}", r, g, b)
            } else {
                "default".to_string()
            };
            info!(
                "Vertex {}: pos=({:.3}, {:.3}, {:.3}), raw_color=({}), normalized_color=({:.3}, {:.3}, {:.3}), opacity={:.3}",
                i,
                position.x,
                position.y,
                position.z,
                raw_color_str,
                color.x,
                color.y,
                color.z,
                opacity
            );
        }
    }

    debug!("Loaded {} Gaussians from PLY file", gaussians.len());
    Ok(gaussians)
}
