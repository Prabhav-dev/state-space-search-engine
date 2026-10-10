//! V2: Slim 16-byte arena record (u32 count + 3 slots; no node_id, mask, or flags).
//! When degree <= 3, all 3 are inline edges.
//! When degree > 3, edges[0] and edges[1] are inline, and edges[2] is the index into overflow buffer.
//! Uses separate shared visited bitset.

use crate::node::{StateNode, MAX_INLINE_EDGES};
use crate::variants::traits::{prefetch_ptr, GraphLayout};
use std::mem::size_of;

pub const V2_MAX_INLINE_EDGES: usize = 3;
pub const V2_OVERFLOW_INLINE_EDGES: usize = 2;

#[repr(C)]
#[repr(align(16))]
#[derive(Debug, Clone, Copy)]
pub struct SlimNode16 {
    pub count: u32,
    pub edges: [u32; 3], // If count <= 3: 3 inline edges. If count > 3: 2 inline + 1 overflow index.
}

impl SlimNode16 {
    pub const SIZE: usize = size_of::<Self>();
}

#[derive(Debug, Clone)]
pub struct V2SlimArenaGraph {
    pub nodes: Vec<SlimNode16>,
    pub overflow: Vec<u32>,
    pub num_nodes: usize,
    pub num_edges: usize,
}

impl V2SlimArenaGraph {
    pub fn from_adjacency(adjacency: &[Vec<u32>]) -> Self {
        let num_nodes = adjacency.len();
        let mut nodes = Vec::with_capacity(num_nodes);
        let mut overflow = Vec::new();
        let mut total_edges = 0;

        for neighbors in adjacency {
            let deg = neighbors.len() as u32;
            total_edges += deg as usize;
            let mut edges = [u32::MAX; 3];

            if deg as usize <= V2_MAX_INLINE_EDGES {
                for (idx, &dst) in neighbors.iter().enumerate() {
                    edges[idx] = dst;
                }
            } else {
                edges[0] = neighbors[0];
                edges[1] = neighbors[1];
                edges[2] = overflow.len() as u32;
                overflow.extend_from_slice(&neighbors[V2_OVERFLOW_INLINE_EDGES..]);
            }

            nodes.push(SlimNode16 { count: deg, edges });
        }

        Self {
            nodes,
            overflow,
            num_nodes,
            num_edges: total_edges,
        }
    }

    pub fn from_nodes_and_overflow(raw_nodes: &[StateNode], raw_overflow: &[u32]) -> Self {
        let num_nodes = raw_nodes.len();
        let mut nodes = Vec::with_capacity(num_nodes);
        let mut overflow = Vec::new();
        let mut total_edges = 0;

        for node in raw_nodes {
            let deg = node.outbound_count as usize;
            total_edges += deg;
            let mut edges = [u32::MAX; 3];

            if deg <= V2_MAX_INLINE_EDGES {
                for idx in 0..deg {
                    edges[idx] = node.outbound_links[idx];
                }
            } else {
                edges[0] = node.outbound_links[0];
                edges[1] = node.outbound_links[1];
                edges[2] = overflow.len() as u32;

                // 3rd inline edge from StateNode
                overflow.push(node.outbound_links[2]);

                // remaining edges from raw_overflow
                let off = node.outbound_links[3] as usize;
                let remaining = deg - MAX_INLINE_EDGES;
                if off + remaining <= raw_overflow.len() {
                    overflow.extend_from_slice(&raw_overflow[off..off + remaining]);
                }
            }

            nodes.push(SlimNode16 {
                count: deg as u32,
                edges,
            });
        }

        Self {
            nodes,
            overflow,
            num_nodes,
            num_edges: total_edges,
        }
    }
}

impl GraphLayout for V2SlimArenaGraph {
    #[inline(always)]
    fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    #[inline(always)]
    fn num_edges(&self) -> usize {
        self.num_edges
    }

    #[inline(always)]
    fn for_each_neighbor<F: FnMut(u32)>(&self, node_id: u32, mut f: F) {
        let u = node_id as usize;
        if u < self.num_nodes {
            let node = &self.nodes[u];
            let count = node.count as usize;
            if count <= V2_MAX_INLINE_EDGES {
                for idx in 0..count {
                    f(node.edges[idx]);
                }
            } else {
                f(node.edges[0]);
                f(node.edges[1]);
                let off = node.edges[2] as usize;
                let remaining = count - V2_OVERFLOW_INLINE_EDGES;
                let end = (off + remaining).min(self.overflow.len());
                for &v in &self.overflow[off..end] {
                    f(v);
                }
            }
        }
    }

    #[inline(always)]
    fn prefetch_stage1(&self, node_id: u32) {
        let u = node_id as usize;
        if u < self.num_nodes {
            prefetch_ptr(&self.nodes[u]);
        }
    }

    #[inline(always)]
    fn prefetch_stage2(&self, node_id: u32) {
        let u = node_id as usize;
        if u < self.num_nodes {
            let node = &self.nodes[u];
            if node.count as usize > V2_MAX_INLINE_EDGES {
                let off = node.edges[2] as usize;
                if off < self.overflow.len() {
                    prefetch_ptr(&self.overflow[off]);
                }
            }
        }
    }
}
