//! V3: Slim 16-byte arena record with inline visited bit.
//! Uses AtomicU32 count_and_visited with fetch_or(0x8000_0000).
//! High bit (bit 31) stores visited; low 31 bits store degree count.
//! Record is exactly 16 bytes and 16-byte cache-aligned.

use crate::node::{StateNode, MAX_INLINE_EDGES};
use crate::variants::traits::{prefetch_ptr, GraphLayout};
use crate::variants::v2_slim_arena::{V2_MAX_INLINE_EDGES, V2_OVERFLOW_INLINE_EDGES};
use std::mem::size_of;
use std::sync::atomic::{AtomicU32, Ordering};

pub const VISITED_MASK: u32 = 0x8000_0000;
pub const COUNT_MASK: u32 = 0x7FFF_FFFF;

#[repr(C)]
#[repr(align(16))]
#[derive(Debug)]
pub struct SlimNode16Visited {
    pub count_and_visited: AtomicU32,
    pub edges: [u32; 3], // If count <= 3: 3 inline edges. If count > 3: 2 inline + 1 overflow index.
}

impl SlimNode16Visited {
    pub const SIZE: usize = size_of::<Self>();
}

#[derive(Debug)]
pub struct V3InlineVisitedGraph {
    pub nodes: Vec<SlimNode16Visited>,
    pub overflow: Vec<u32>,
    pub num_nodes: usize,
    pub num_edges: usize,
}

impl V3InlineVisitedGraph {
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

            nodes.push(SlimNode16Visited {
                count_and_visited: AtomicU32::new(deg & COUNT_MASK),
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

                overflow.push(node.outbound_links[2]);

                let off = node.outbound_links[3] as usize;
                let remaining = deg - MAX_INLINE_EDGES;
                if off + remaining <= raw_overflow.len() {
                    overflow.extend_from_slice(&raw_overflow[off..off + remaining]);
                }
            }

            nodes.push(SlimNode16Visited {
                count_and_visited: AtomicU32::new((deg as u32) & COUNT_MASK),
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

impl GraphLayout for V3InlineVisitedGraph {
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
            let count = (node.count_and_visited.load(Ordering::Relaxed) & COUNT_MASK) as usize;
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
            let count = (node.count_and_visited.load(Ordering::Relaxed) & COUNT_MASK) as usize;
            if count > V2_MAX_INLINE_EDGES {
                let off = node.edges[2] as usize;
                if off < self.overflow.len() {
                    prefetch_ptr(&self.overflow[off]);
                }
            }
        }
    }

    #[inline(always)]
    fn has_internal_visited(&self) -> bool {
        true
    }

    #[inline(always)]
    fn internal_test_and_set(&self, node_id: u32) -> bool {
        let u = node_id as usize;
        if u < self.num_nodes {
            let node = &self.nodes[u];
            let old = node.count_and_visited.load(Ordering::Relaxed);
            if (old & VISITED_MASK) != 0 {
                return true;
            }
            node.count_and_visited.store(old | VISITED_MASK, Ordering::Relaxed);
            false
        } else {
            true
        }
    }

    #[inline(always)]
    fn internal_test_and_set_parallel(&self, node_id: u32, test_before_atomic: bool) -> bool {
        let u = node_id as usize;
        if u < self.num_nodes {
            let node = &self.nodes[u];
            if test_before_atomic {
                let old = node.count_and_visited.load(Ordering::Relaxed);
                if (old & VISITED_MASK) != 0 {
                    return true;
                }
            }
            let prev = node.count_and_visited.fetch_or(VISITED_MASK, Ordering::AcqRel);
            (prev & VISITED_MASK) != 0
        } else {
            true
        }
    }

    #[inline(always)]
    fn internal_reset_visited(&self) {
        for node in &self.nodes {
            node.count_and_visited.fetch_and(COUNT_MASK, Ordering::Relaxed);
        }
    }

    #[inline(always)]
    fn internal_is_visited(&self, node_id: u32) -> bool {
        let u = node_id as usize;
        if u < self.num_nodes {
            (self.nodes[u].count_and_visited.load(Ordering::Relaxed) & VISITED_MASK) != 0
        } else {
            false
        }
    }
}
