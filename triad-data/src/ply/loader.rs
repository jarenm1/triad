//! PLY file loading functions

use crate::ply::PlyVertex;
use glam::Vec3;
use serde::Deserialize;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use tracing::{debug, info, warn};

// Face structure for PLY files
#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct PlyFace {
    vertex_indices: Vec<i32>,
}

// PLY file structure
#[derive(Deserialize, Debug)]
struct PlyFile {
    #[serde(rename = "vertex")]
    vertex: Vec<HashMap<String, JsonValue>>,
    #[serde(default, rename = "face", skip_serializing_if = "Vec::is_empty")]
    face: Vec<PlyFace>,
}

/// Load vertices from a PLY file without converting to GPU-specific format.
/// Returns raw vertex data that can be used for triangulation or other processing.
#[tracing::instrument(skip_all, fields(path = %path))]
pub fn load_vertices_from_ply(path: &str) -> Result<Vec<PlyVertex>, Box<dyn std::error::Error>> {
    debug!("Loading PLY vertices from: {}", path);
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let ply_data: PlyFile = serde_ply::from_reader(reader).map_err(|e| {
        warn!("Failed to parse PLY file: {}", e);
        format!("PLY parsing error: {}", e)
    })?;

    info!(
        "PLY file parsed: {} vertices, {} faces",
        ply_data.vertex.len(),
        ply_data.face.len()
    );

    fn get_f32(prop: Option<&JsonValue>) -> Option<f32> {
        prop.and_then(|v| match v {
            JsonValue::Number(n) => n.as_f64().map(|f| f as f32),
            _ => None,
        })
    }

    fn get_u8(prop: Option<&JsonValue>) -> Option<u8> {
        prop.and_then(|v| match v {
            JsonValue::Number(n) => n
                .as_u64()
                .map(|u| u as u8)
                .or_else(|| n.as_i64().map(|i| i as u8)),
            _ => None,
        })
    }

    let mut vertices = Vec::with_capacity(ply_data.vertex.len());

    for (i, vertex) in ply_data.vertex.iter().enumerate() {
        let x = get_f32(vertex.get("x"))
            .ok_or_else(|| format!("Missing 'x' at vertex {}", i))?;
        let y = get_f32(vertex.get("y"))
            .ok_or_else(|| format!("Missing 'y' at vertex {}", i))?;
        let z = get_f32(vertex.get("z"))
            .ok_or_else(|| format!("Missing 'z' at vertex {}", i))?;

        let color = if let (Some(r), Some(g), Some(b)) = (
            get_u8(vertex.get("red")),
            get_u8(vertex.get("green")),
            get_u8(vertex.get("blue")),
        ) {
            Vec3::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0)
        } else if let (Some(r), Some(g), Some(b)) = (
            get_u8(vertex.get("r")),
            get_u8(vertex.get("g")),
            get_u8(vertex.get("b")),
        ) {
            Vec3::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0)
        } else {
            Vec3::new(0.8, 0.8, 0.8)
        };

        let raw_opacity = get_f32(vertex.get("opacity"))
            .or_else(|| get_f32(vertex.get("alpha")))
            .unwrap_or(1.0);
        let opacity = if (0.0..=1.0).contains(&raw_opacity) {
            raw_opacity
        } else {
            (1.0 / (1.0 + (-raw_opacity).exp())).clamp(0.0, 1.0)
        };

        vertices.push(PlyVertex {
            position: Vec3::new(x, y, z),
            color,
            opacity,
        });
    }

    debug!("Loaded {} vertices from PLY file", vertices.len());
    Ok(vertices)
}

/// Check if a PLY file contains face data without fully parsing it.
pub fn ply_has_faces(path: &str) -> Result<bool, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let ply_data: PlyFile = serde_ply::from_reader(reader)?;
    Ok(!ply_data.face.is_empty())
}
