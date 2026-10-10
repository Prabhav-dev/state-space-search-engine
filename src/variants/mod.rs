pub mod traits;
pub mod visited;
pub mod reorder;
pub mod v0_csr;
pub mod v2_slim_arena;
pub mod v3_inline_visited;
pub mod bfs_driver;

pub use traits::{GraphLayout, PrefetchMode, TraversalConfig, VertexOrder};
pub use visited::{SharedParallelScratch, SharedSingleScratch};
pub use reorder::{compute_permutation, apply_permutation, Permutation};
pub use v0_csr::V0CsrGraph;
pub use v2_slim_arena::{SlimNode16, V2SlimArenaGraph};
pub use v3_inline_visited::{SlimNode16Visited, V3InlineVisitedGraph};
pub use bfs_driver::{
    prepare_scratch_parallel, prepare_scratch_parallel_for_threads, prepare_scratch_single, traverse_and_record_levels,
    traverse_bfs_parallel, traverse_bfs_parallel_prepared, traverse_bfs_single,
    traverse_bfs_single_prepared,
};
