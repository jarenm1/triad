use crate::frame_graph::FrameGraphError;
use crate::frame_graph::pass::PassNode;
use std::collections::VecDeque;

/// Topological sort of passes based on resource dependencies
pub fn topological_sort(passes: &[PassNode]) -> Result<Vec<usize>, FrameGraphError> {
    let n = passes.len();
    let mut in_degree = vec![0; n];
    let mut graph: Vec<Vec<usize>> = vec![Vec::new(); n];

    // Build dependency graph
    for i in 0..n {
        for j in 0..n {
            if i != j && passes[i].dependencies(&passes[j]) {
                graph[i].push(j);
                in_degree[j] += 1;
            }
        }
    }

    // Kahn's algorithm
    let mut queue = VecDeque::new();
    for i in 0..n {
        if in_degree[i] == 0 {
            queue.push_back(i);
        }
    }

    let mut result = Vec::new();
    while let Some(u) = queue.pop_front() {
        result.push(u);

        for &v in &graph[u] {
            in_degree[v] -= 1;
            if in_degree[v] == 0 {
                queue.push_back(v);
            }
        }
    }

    if result.len() != n {
        return Err(FrameGraphError::CircularDependency);
    }

    Ok(result)
}
