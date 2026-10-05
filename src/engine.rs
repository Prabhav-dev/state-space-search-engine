use crate::frontier::SharedCasFrontier;
use crate::hash::ZobristVisitedSet;
use crate::node::StateNode;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Reusable scratch space to isolate graph traversal execution time from memory allocation overhead.
pub struct EngineScratch {
    pub visited: Vec<u64>,
    pub cur: Vec<u32>,
    pub next: Vec<u32>,
}

impl EngineScratch {
    pub fn new(capacity: usize) -> Self {
        Self {
            visited: vec![0; (capacity + 63) / 64],
            cur: Vec::with_capacity(capacity),
            next: Vec::with_capacity(capacity),
        }
    }

    pub fn reset(&mut self) {
        self.visited.fill(0);
        self.cur.clear();
        self.next.clear();
    }
}

#[inline(always)]
fn test_and_set_bit(visited: &mut [u64], v: u32) -> bool {
    let w = (v >> 6) as usize;
    let m = 1u64 << (v & 63);
    let old = visited[w];
    visited[w] = old | m;
    (old & m) != 0
}

/// Core traversal engine for executing cache-resident state-space searches.
pub struct SearchEngine {
    nodes: Vec<StateNode>,
    overflow: Vec<u32>,
}

impl SearchEngine {
    /// Initializes the search engine with a flat arena of state nodes.
    pub fn new(nodes: Vec<StateNode>) -> Self {
        Self {
            nodes,
            overflow: Vec::new(),
        }
    }

    /// Initializes the search engine with state nodes and an overflow buffer (for degree > 4).
    pub fn with_overflow(nodes: Vec<StateNode>, overflow: Vec<u32>) -> Self {
        Self { nodes, overflow }
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Helper closure iterator over neighbors of a node.
    #[inline(always)]
    fn for_each_neighbor<F>(&self, node: &StateNode, mut f: F)
    where
        F: FnMut(u32),
    {
        let count = node.outbound_count as usize;
        if count <= 4 {
            for &v in &node.outbound_links[..count] {
                f(v);
            }
        } else {
            for &v in &node.outbound_links[..3] {
                f(v);
            }
            let overflow_offset = node.outbound_links[3] as usize;
            let remaining = count - 3;
            if overflow_offset + remaining <= self.overflow.len() {
                for &v in &self.overflow[overflow_offset..overflow_offset + remaining] {
                    f(v);
                }
            }
        }
    }

    /// Basic single-threaded BFS (allocates visited array internally).
    pub fn traverse_bfs(&self, start_node_id: u32) -> u64 {
        let mut scratch = EngineScratch::new(self.nodes.len());
        self.traverse_bfs_with_scratch(start_node_id, &mut scratch)
    }

    /// Optimized single-threaded BFS using pre-allocated scratch buffers (zero allocation inside timed region).
    pub fn traverse_bfs_with_scratch(&self, start_node_id: u32, scratch: &mut EngineScratch) -> u64 {
        scratch.reset();
        let EngineScratch { visited, cur, next } = scratch;

        let mut edges_traversed: u64 = 0;

        let start_idx = start_node_id as usize;
        if start_idx < self.nodes.len() {
            test_and_set_bit(visited, start_node_id);
            cur.push(start_node_id);
        }

        while !cur.is_empty() {
            for &node_id in cur.iter() {
                let node = &self.nodes[node_id as usize];

                self.for_each_neighbor(node, |neighbor_id| {
                    let neighbor_idx = neighbor_id as usize;
                    if neighbor_idx < self.nodes.len() {
                        edges_traversed += 1;
                        if !test_and_set_bit(visited, neighbor_id) {
                            next.push(neighbor_id);
                        }
                    }
                });
            }

            cur.clear();
            std::mem::swap(cur, next);
        }

        edges_traversed
    }

    /// Performs multi-threaded level-synchronous BFS using per-thread partitioned frontiers (H1 / Step 5).
    pub fn traverse_parallel_bfs(&self, start_node_id: u32, num_threads: usize) -> u64 {
        if self.nodes.is_empty() || start_node_id as usize >= self.nodes.len() {
            return 0;
        }

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        pool.install(|| {
            let visited: Vec<AtomicBool> = (0..self.nodes.len())
                .map(|_| AtomicBool::new(false))
                .collect();
            let total_edges = AtomicU64::new(0);

            visited[start_node_id as usize].store(true, Ordering::Relaxed);
            let mut current_frontier = vec![start_node_id];

            while !current_frontier.is_empty() {
                let chunk_size = (current_frontier.len() + num_threads - 1) / num_threads;

                let next_frontiers: Vec<Vec<u32>> = current_frontier
                    .par_chunks(chunk_size.max(1))
                    .map(|chunk| {
                        let mut local_next = Vec::new();
                        let mut local_edges = 0u64;

                        for &node_id in chunk {
                            let node = &self.nodes[node_id as usize];
                            self.for_each_neighbor(node, |neighbor_id| {
                                let neighbor_idx = neighbor_id as usize;
                                if neighbor_idx < self.nodes.len() {
                                    local_edges += 1;
                                    if !visited[neighbor_idx].swap(true, Ordering::AcqRel) {
                                        local_next.push(neighbor_id);
                                    }
                                }
                            });
                        }

                        total_edges.fetch_add(local_edges, Ordering::Relaxed);
                        local_next
                    })
                    .collect();

                current_frontier = next_frontiers.into_iter().flatten().collect();
            }

            total_edges.load(Ordering::Relaxed)
        })
    }

    /// Performs multi-threaded BFS using a Shared CAS Atomic Queue frontier to directly test H2 (Shared CAS vs Partitioned).
    pub fn traverse_parallel_shared_cas_bfs(&self, start_node_id: u32, num_threads: usize) -> u64 {
        if self.nodes.is_empty() || start_node_id as usize >= self.nodes.len() {
            return 0;
        }

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        pool.install(|| {
            let visited: Vec<AtomicBool> = (0..self.nodes.len())
                .map(|_| AtomicBool::new(false))
                .collect();
            let total_edges = AtomicU64::new(0);

            let cur_queue = SharedCasFrontier::new(self.nodes.len());
            let next_queue = SharedCasFrontier::new(self.nodes.len());

            visited[start_node_id as usize].store(true, Ordering::Relaxed);
            cur_queue.push(start_node_id);

            loop {
                // Execute level processing across threads reading from shared CAS frontier
                let active_threads: usize = (0..num_threads)
                    .into_par_iter()
                    .map(|_| {
                        let mut local_edges = 0u64;
                        let mut popped = false;
                        while let Some(node_id) = cur_queue.pop() {
                            popped = true;
                            let node = &self.nodes[node_id as usize];
                            self.for_each_neighbor(node, |neighbor_id| {
                                let neighbor_idx = neighbor_id as usize;
                                if neighbor_idx < self.nodes.len() {
                                    local_edges += 1;
                                    if !visited[neighbor_idx].swap(true, Ordering::AcqRel) {
                                        next_queue.push(neighbor_id);
                                    }
                                }
                            });
                        }
                        total_edges.fetch_add(local_edges, Ordering::Relaxed);
                        if popped { 1 } else { 0 }
                    })
                    .sum();

                if active_threads == 0 {
                    break;
                }

                cur_queue.clear();
                // Swap queues logic (copy remaining or re-push)
                while let Some(v) = next_queue.pop() {
                    cur_queue.push(v);
                }
                next_queue.clear();
            }

            total_edges.load(Ordering::Relaxed)
        })
    }

    /// Evaluates Zobrist bitstate search (H3):
    /// 1. `evaluate_zobrist_bitstate_coverage`: Runs actual bitstate search (skipping enqueuing on Zobrist bit set).
    ///    Returns (nodes_visited, exact_reachable_nodes, coverage_percentage, false_positives).
    pub fn evaluate_zobrist_bitstate_coverage(
        &self,
        start_node_id: u32,
        power_of_two_bits: usize,
    ) -> (usize, usize, f64, usize) {
        let mut zobrist_set = ZobristVisitedSet::new(power_of_two_bits);
        let mut exact_visited = vec![false; self.nodes.len()];

        let mut cur = Vec::new();
        let mut next = Vec::new();

        let mut bitstate_nodes_reached = 0usize;
        let mut false_positives = 0usize;

        let start_idx = start_node_id as usize;
        if start_idx < self.nodes.len() {
            cur.push(start_node_id);
            zobrist_set.test_and_set(start_node_id as u64);
            exact_visited[start_idx] = true;
            bitstate_nodes_reached += 1;
        }

        while !cur.is_empty() {
            for &node_id in &cur {
                let node = &self.nodes[node_id as usize];

                self.for_each_neighbor(node, |neighbor_id| {
                    let neighbor_idx = neighbor_id as usize;
                    if neighbor_idx < self.nodes.len() {
                        let zobrist_hit = zobrist_set.test_and_set(neighbor_id as u64);
                        let exact_already = exact_visited[neighbor_idx];

                        if zobrist_hit && !exact_already {
                            // False positive: Zobrist bit table hit, but state was NOT visited in exact search!
                            false_positives += 1;
                        } else if !zobrist_hit {
                            exact_visited[neighbor_idx] = true;
                            bitstate_nodes_reached += 1;
                            next.push(neighbor_id);
                        }
                    }
                });
            }
            cur.clear();
            std::mem::swap(&mut cur, &mut next);
        }

        // Measure exact reachability baseline
        let mut scratch = EngineScratch::new(self.nodes.len());
        self.traverse_bfs_with_scratch(start_node_id, &mut scratch);
        let exact_reachable = scratch.visited.iter().map(|w| w.count_ones() as usize).sum();

        let coverage_pct = if exact_reachable > 0 {
            100.0 * bitstate_nodes_reached as f64 / exact_reachable as f64
        } else {
            0.0
        };

        (
            bitstate_nodes_reached,
            exact_reachable,
            coverage_pct,
            false_positives,
        )
    }

    /// Target-directed search with learned context-keyed edge priorities (H4).
    /// Supports training on target sets $T_{train}$ and evaluating on held-out target sets $T_{test}$.
    pub fn search_with_learned_priorities(
        &self,
        start_node_id: u32,
        target_node_id: u32,
        edge_priorities: &mut [Vec<u32>],
        train_mode: bool,
    ) -> (u64, bool) {
        let mut visited = vec![false; self.nodes.len()];
        let mut queue = std::collections::VecDeque::new();
        let mut nodes_expanded = 0u64;

        let start_idx = start_node_id as usize;
        if start_idx >= self.nodes.len() {
            return (0, false);
        }

        queue.push_back(start_node_id);
        visited[start_idx] = true;

        let mut parent_map: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        let mut found = false;

        while let Some(curr) = queue.pop_front() {
            nodes_expanded += 1;
            if curr == target_node_id {
                found = true;
                break;
            }

            let curr_idx = curr as usize;
            let node = &self.nodes[curr_idx];

            let mut neighbor_order = Vec::new();
            self.for_each_neighbor(node, |v| neighbor_order.push(v));

            if curr_idx < edge_priorities.len() && !edge_priorities[curr_idx].is_empty() {
                let prio = &edge_priorities[curr_idx];
                neighbor_order.sort_by_key(|&v| {
                    let v_idx = v as usize;
                    if v_idx < prio.len() {
                        u32::MAX - prio[v_idx]
                    } else {
                        u32::MAX
                    }
                });
            }

            for &v in &neighbor_order {
                let v_idx = v as usize;
                if v_idx < self.nodes.len() && !visited[v_idx] {
                    visited[v_idx] = true;
                    parent_map.insert(v, curr);
                    queue.push_back(v);
                }
            }
        }

        if found && train_mode {
            let mut curr = target_node_id;
            while let Some(&p_node) = parent_map.get(&curr) {
                let p_idx = p_node as usize;
                if p_idx < edge_priorities.len() {
                    let v_idx = curr as usize;
                    if v_idx >= edge_priorities[p_idx].len() {
                        edge_priorities[p_idx].resize(v_idx + 1, 0);
                    }
                    edge_priorities[p_idx][v_idx] =
                        edge_priorities[p_idx][v_idx].saturating_add(10);
                }
                curr = p_node;
            }
        }

        (nodes_expanded, found)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_bfs_traversal() {
        let nodes = vec![
            StateNode {
                node_id: 0,
                structural_mask: 0,
                outbound_count: 2,
                metadata_flags: 0,
                outbound_links: [1, 2, u32::MAX, u32::MAX],
            },
            StateNode {
                node_id: 1,
                structural_mask: 0,
                outbound_count: 0,
                metadata_flags: 0,
                outbound_links: [u32::MAX; 4],
            },
            StateNode {
                node_id: 2,
                structural_mask: 0,
                outbound_count: 0,
                metadata_flags: 0,
                outbound_links: [u32::MAX; 4],
            },
        ];

        let engine = SearchEngine::new(nodes);
        let edges = engine.traverse_bfs(0);
        assert_eq!(edges, 2);
    }
}