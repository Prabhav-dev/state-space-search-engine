//! V0: Native Compressed Sparse Row (CSR) Graph Layout.

use crate::node::{StateNode, MAX_INLINE_EDGES};
use crate::variants::traits::{prefetch_ptr, GraphLayout};

#[derive(Debug, Clone)]
pub struct V0CsrGraph {
    pub offsets: Vec<u32>,
    pub targets: Vec<u32>,
    pub num_nodes: usize,
    pub num_edges: usize,
}

impl V0CsrGraph {
    pub fn new(offsets: Vec<u32>, targets: Vec<u32>, num_nodes: usize) -> Self {
        let num_edges = targets.len();
        Self {
            offsets,
            targets,
            num_nodes,
            num_edges,
        }
    }

    pub fn from_adjacency(adjacency: &[Vec<u32>]) -> Self {
        let num_nodes = adjacency.len();
        let mut offsets = Vec::with_capacity(num_nodes + 1);
        let total_edges: usize = adjacency.iter().map(|n| n.len()).sum();
        let mut targets = Vec::with_capacity(total_edges);

        for neighbors in adjacency {
            offsets.push(targets.len() as u32);
            targets.extend_from_slice(neighbors);
        }
        offsets.push(targets.len() as u32);

        Self {
            offsets,
            targets,
            num_nodes,
            num_edges: total_edges,
        }
    }

    pub fn from_nodes_and_overflow(nodes: &[StateNode], overflow: &[u32]) -> Self {
        let mut offsets = Vec::with_capacity(nodes.len() + 1);
        let mut targets = Vec::new();

        for node in nodes {
            offsets.push(targets.len() as u32);
            let count = node.outbound_count as usize;
            if count <= MAX_INLINE_EDGES {
                targets.extend_from_slice(&node.outbound_links[..count]);
            } else {
                targets.extend_from_slice(&node.outbound_links[..MAX_INLINE_EDGES]);
                let off = node.outbound_links[3] as usize;
                let remaining = count - MAX_INLINE_EDGES;
                if off + remaining <= overflow.len() {
                    targets.extend_from_slice(&overflow[off..off + remaining]);
                }
            }
        }
        offsets.push(targets.len() as u32);
        let num_edges = targets.len();
        Self {
            offsets,
            targets,
            num_nodes: nodes.len(),
            num_edges,
        }
    }
}

impl GraphLayout for V0CsrGraph {
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
            let lo = self.offsets[u] as usize;
            let hi = self.offsets[u + 1] as usize;
            for &v in &self.targets[lo..hi] {
                f(v);
            }
        }
    }

    #[inline(always)]
    fn prefetch_stage1(&self, node_id: u32) {
        let u = node_id as usize;
        if u < self.num_nodes {
            prefetch_ptr(&self.offsets[u]);
        }
    }

    #[inline(always)]
    fn prefetch_stage2(&self, node_id: u32) {
        let u = node_id as usize;
        if u < self.num_nodes {
            let lo = self.offsets[u] as usize;
            if lo < self.targets.len() {
                prefetch_ptr(&self.targets[lo]);
            }
        }
    }
}
