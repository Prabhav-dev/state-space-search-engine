//! R-MAT / Kronecker power-law graph generator and graph analysis tools.
//! Implements Step 4 of L3 State Solver Working Draft v0.2.

use crate::node::StateNode;

pub struct RmatConfig {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
}

impl Default for RmatConfig {
    fn default() -> Self {
        // Standard R-MAT parameters for power-law social/scale-free networks (a=0.57, b=0.19, c=0.19, d=0.05)
        Self {
            a: 0.57,
            b: 0.19,
            c: 0.19,
            d: 0.05,
        }
    }
}

pub struct RmatGraphResult {
    pub nodes: Vec<StateNode>,
    pub overflow: Vec<u32>,
    pub reached_nodes: usize,
    pub max_depth: usize,
    pub total_edges: usize,
}

struct SimpleRng(u64);
impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_f64(&mut self) -> f64 {
        // SplitMix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let res = z ^ (z >> 31);
        (res >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Generates an R-MAT power-law graph.
/// Each node's out-edges are populated into `StateNode`. If degree > 4, the first 3 out-edges are inline,
/// and `outbound_links[3]` holds the offset into `overflow` buffer.
pub fn generate_rmat_graph(
    node_count: usize,
    edges_per_node: usize,
    config: RmatConfig,
    seed: u64,
) -> RmatGraphResult {
    let total_edges = node_count * edges_per_node;
    let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); node_count];
    let mut rng = SimpleRng::new(seed);

    let n_bits = (node_count as f64).log2().ceil() as u32;
    let scale = 1u64 << n_bits;

    let a = config.a;
    let ab = a + config.b;
    let abc = ab + config.c;

    for _ in 0..total_edges {
        let mut u: u64 = 0;
        let mut v: u64 = 0;
        let mut step = scale >> 1;

        while step > 0 {
            let r = rng.next_f64();
            if r < a {
                // Top-Left quadrant (u, v unchanged)
            } else if r < ab {
                // Top-Right quadrant
                v += step;
            } else if r < abc {
                // Bottom-Left quadrant
                u += step;
            } else {
                // Bottom-Right quadrant
                u += step;
                v += step;
            }
            step >>= 1;
        }

        let src = (u % node_count as u64) as usize;
        let dst = (v % node_count as u64) as u32;
        adjacency[src].push(dst);
    }

    // Build StateNodes and overflow buffer
    let mut nodes = Vec::with_capacity(node_count);
    let mut overflow = Vec::new();

    for (i, neighbors) in adjacency.iter().enumerate() {
        let deg = neighbors.len() as u32;
        let mut links = [u32::MAX; 4];

        if deg <= 4 {
            for (idx, &dst) in neighbors.iter().enumerate() {
                links[idx] = dst;
            }
        } else {
            // First 3 inline, 4th slot points to overflow start index
            for idx in 0..3 {
                links[idx] = neighbors[idx];
            }
            links[3] = overflow.len() as u32;
            overflow.extend_from_slice(&neighbors[3..]);
        }

        nodes.push(StateNode {
            node_id: i as u32,
            structural_mask: 0,
            outbound_count: deg,
            metadata_flags: 0,
            outbound_links: links,
        });
    }

    // Measure reachability and depth using BFS from node 0
    let (reached_nodes, max_depth) = analyze_graph_reachability(&nodes, &overflow, 0);

    RmatGraphResult {
        nodes,
        overflow,
        reached_nodes,
        max_depth,
        total_edges,
    }
}

/// Helper to analyze reachability and BFS level depth of a graph.
pub fn analyze_graph_reachability(
    nodes: &[StateNode],
    overflow: &[u32],
    start: u32,
) -> (usize, usize) {
    if nodes.is_empty() || start as usize >= nodes.len() {
        return (0, 0);
    }

    let mut visited = vec![false; nodes.len()];
    let mut cur_frontier = vec![start];
    let mut next_frontier = Vec::new();
    visited[start as usize] = true;

    let mut reached = 1usize;
    let mut depth = 0usize;

    while !cur_frontier.is_empty() {
        depth += 1;
        for &u in &cur_frontier {
            let node = &nodes[u as usize];
            let count = node.outbound_count as usize;

            if count <= 4 {
                for &v in &node.outbound_links[..count] {
                    let v_idx = v as usize;
                    if v_idx < nodes.len() && !visited[v_idx] {
                        visited[v_idx] = true;
                        reached += 1;
                        next_frontier.push(v);
                    }
                }
            } else {
                for &v in &node.outbound_links[..3] {
                    let v_idx = v as usize;
                    if v_idx < nodes.len() && !visited[v_idx] {
                        visited[v_idx] = true;
                        reached += 1;
                        next_frontier.push(v);
                    }
                }
                let overflow_offset = node.outbound_links[3] as usize;
                let remaining = count - 3;
                for &v in &overflow[overflow_offset..overflow_offset + remaining] {
                    let v_idx = v as usize;
                    if v_idx < nodes.len() && !visited[v_idx] {
                        visited[v_idx] = true;
                        reached += 1;
                        next_frontier.push(v);
                    }
                }
            }
        }
        cur_frontier.clear();
        std::mem::swap(&mut cur_frontier, &mut next_frontier);
    }

    (reached, depth)
}
