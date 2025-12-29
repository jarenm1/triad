//! Streaming PLY file support for incremental loading

use crate::ingest::point_cloud::Point;
use std::io::{BufReader, Read, Seek};
use std::path::Path;
use tracing::{debug, warn};

/// Streaming PLY reader that loads points incrementally
pub struct PlyStream {
    // For now, we'll implement a simple version that reads the whole file
    // Future: implement true streaming for large files
    points: Vec<Point>,
    current_index: usize,
}

impl PlyStream {
    /// Create a new PLY stream from a file path
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        use std::fs::File;
        let file = File::open(path)?;
        Self::from_reader(BufReader::new(file))
    }

    /// Create a new PLY stream from a reader
    pub fn from_reader<R: Read + Seek>(_reader: R) -> Result<Self, Box<dyn std::error::Error>> {
        // For now, use the existing PLY loader from triad-gpu
        // TODO: Implement incremental streaming for large files
        warn!("PlyStream currently loads entire file into memory. Streaming support coming soon.");
        
        // Use triad-data for PLY loading
        // For now, return an empty stream as a placeholder
        Ok(Self {
            points: Vec::new(),
            current_index: 0,
        })
    }

    /// Read the next batch of points (up to `max_points`)
    pub fn read_batch(&mut self, max_points: usize) -> Vec<Point> {
        let start = self.current_index;
        let end = (start + max_points).min(self.points.len());
        let batch = self.points[start..end].to_vec();
        self.current_index = end;
        debug!("Read batch: {} points (total: {})", batch.len(), self.points.len());
        batch
    }

    /// Check if there are more points to read
    pub fn has_more(&self) -> bool {
        self.current_index < self.points.len()
    }

    /// Reset the stream to the beginning
    pub fn reset(&mut self) {
        self.current_index = 0;
    }

    /// Get the total number of points
    pub fn total_points(&self) -> usize {
        self.points.len()
    }
}
