//! Vertex reordering utilities (Original, DegreeSort, BfsOrder).
//! Reordering build times are timed and logged separately from BFS traversal.

use crate::variants::traits::VertexOrder;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct Permutation {
    pub orig_to_new: Vec<u32>,
    pub new_to_orig: Vec<u32>,
    pub order: VertexOrder,
}

impl Permutation {
    pub fn identity(num_nodes: usize) -> Self {
        let orig_to_new: Vec<u32> = (0..num_nodes as u32).collect();
        let new_to_orig = orig_to_new.clone();
        Self {
            orig_to_new,
            new_to_orig,
            order: VertexOrder::Original,
        }
    }
}

/// Computes vertex permutation according to the specified VertexOrder.
/// If `order == BfsOrder`, `bfs_seed` is used as the start node.
/// IMPORTANT: `bfs_seed` must be disjoint from any benchmark source node.
pub fn compute_permutation(
    adjacency: &[Vec<u32>],
    order: VertexOrder,
    bfs_seed: u32,
) -> Permutation {
    let n = adjacency.len();
    match order {
        VertexOrder::Original => Permutation::identity(n),
        VertexOrder::DegreeSort => {
            let mut node_degrees: Vec<(usize, u32)> = adjacency
                .iter()
                .enumerate()
                .map(|(i, edges)| (i, edges.len() as u32))
                .collect();

            // Sort descending by degree; tie-break on node index for stability
            node_degrees.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

            let mut orig_to_new = vec![0u32; n];
            let mut new_to_orig = vec![0u32; n];

            for (new_id, &(orig_id, _)) in node_degrees.iter().enumerate() {
                orig_to_new[orig_id] = new_id as u32;
                new_to_orig[new_id] = orig_id as u32;
            }

            Permutation {
                orig_to_new,
                new_to_orig,
                order,
            }
        }
        VertexOrder::BfsOrder => {
            let mut visited = vec![false; n];
            let mut new_to_orig = Vec::with_capacity(n);
            let mut orig_to_new = vec![u32::MAX; n];

            let seed = (bfs_seed as usize).min(if n > 0 { n - 1 } else { 0 });

            let mut queue = std::collections::VecDeque::with_capacity(n);

            // Start from seed
            if n > 0 {
                visited[seed] = true;
                queue.push_back(seed);
            }

            // Outer loop to handle disconnected components
            let mut unvisited_cursor = 0;
            loop {
                while let Some(u) = queue.pop_front() {
                    let new_id = new_to_orig.len() as u32;
                    new_to_orig.push(u as u32);
                    orig_to_new[u] = new_id;

                    for &v in &adjacency[u] {
                        let v_idx = v as usize;
                        if v_idx < n && !visited[v_idx] {
                            visited[v_idx] = true;
                            queue.push_back(v_idx);
                        }
                    }
                }

                if new_to_orig.len() == n {
                    break;
                }

                // Find next unvisited node
                while unvisited_cursor < n && visited[unvisited_cursor] {
                    unvisited_cursor += 1;
                }
                if unvisited_cursor < n {
                    visited[unvisited_cursor] = true;
                    queue.push_back(unvisited_cursor);
                    unvisited_cursor += 1;
                } else {
                    break;
                }
            }

            Permutation {
                orig_to_new,
                new_to_orig,
                order,
            }
        }
    }
}

/// Reorders an adjacency list using the given permutation and measures reorder elapsed time.
pub fn apply_permutation(
    adjacency: &[Vec<u32>],
    perm: &Permutation,
) -> (Vec<Vec<u32>>, Duration) {
    let start = Instant::now();
    let n = adjacency.len();
    let mut reordered = vec![Vec::new(); n];

    for (orig_u, neighbors) in adjacency.iter().enumerate() {
        let new_u = perm.orig_to_new[orig_u] as usize;
        let mut remapped: Vec<u32> = neighbors
            .iter()
            .map(|&v| {
                let v_idx = v as usize;
                if v_idx < n {
                    perm.orig_to_new[v_idx]
                } else {
                    v
                }
            })
            .collect();
        remapped.sort_unstable();
        reordered[new_u] = remapped;
    }

    let elapsed = start.elapsed();
    (reordered, elapsed)
}
