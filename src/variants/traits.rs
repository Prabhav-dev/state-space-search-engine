//! Traits and shared types for the graph-layout BFS study.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexOrder {
    Original,
    DegreeSort,
    BfsOrder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchMode {
    None,
    KAhead(usize),
    TwoStage(usize),
    Batched(usize),
}

#[derive(Debug, Clone, Copy)]
pub struct TraversalConfig {
    pub num_threads: usize,
    pub test_before_atomic: bool,
    pub prefetch: PrefetchMode,
}

impl Default for TraversalConfig {
    fn default() -> Self {
        Self {
            num_threads: 1,
            test_before_atomic: false,
            prefetch: PrefetchMode::None,
        }
    }
}

/// Helper function to prefetch memory at a given pointer address into CPU L1/L2 cache.
#[inline(always)]
pub fn prefetch_ptr<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_mm_prefetch(ptr as *const i8, core::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = ptr;
    }
}

/// Shared visited interface to support both external visited buffers and in-record visited flags.
pub trait VisitedTracker: Send + Sync {
    fn reset(&self);
    fn test_and_set(&self, node_id: u32) -> bool;
    fn test_and_set_parallel(&self, node_id: u32, test_before_atomic: bool) -> bool;
    fn is_visited(&self, node_id: u32) -> bool;
}

/// Core interface implemented by all graph variants (V0 through V5).
pub trait GraphLayout: Send + Sync {
    fn num_nodes(&self) -> usize;
    fn num_edges(&self) -> usize;

    /// Iterate over all out-neighbors of a node.
    fn for_each_neighbor<F: FnMut(u32)>(&self, node_id: u32, f: F);

    /// Stage 1 prefetch: issue prefetch for node record / offset header.
    fn prefetch_stage1(&self, node_id: u32) {
        let _ = node_id;
    }

    /// Stage 2 prefetch: issue prefetch for targets / edge storage once offset is loaded.
    fn prefetch_stage2(&self, node_id: u32) {
        let _ = node_id;
    }

    /// Whether this layout provides its own internal visited storage (e.g. V3).
    /// If false, the shared visited tracker is used.
    fn has_internal_visited(&self) -> bool {
        false
    }

    /// Internal visited test-and-set for variants like V3.
    fn internal_test_and_set(&self, _node_id: u32) -> bool {
        false
    }

    /// Internal parallel visited test-and-set for variants like V3.
    fn internal_test_and_set_parallel(&self, _node_id: u32, _test_before_atomic: bool) -> bool {
        false
    }

    fn internal_reset_visited(&self) {}
    fn internal_is_visited(&self, _node_id: u32) -> bool {
        false
    }
}
