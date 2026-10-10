use state_search_engine::graph::{generate_rmat_graph, RmatConfig};
use state_search_engine::variants::bfs_driver::{
    traverse_and_record_levels, traverse_bfs_parallel, traverse_bfs_single,
};
use state_search_engine::variants::reorder::{apply_permutation, compute_permutation};
use state_search_engine::variants::traits::{PrefetchMode, TraversalConfig, VertexOrder};
use state_search_engine::variants::v0_csr::V0CsrGraph;
use state_search_engine::variants::visited::{SharedParallelScratch, SharedSingleScratch};

#[test]
fn test_v0_basic_topologies() {
    // Line graph: 0 -> 1 -> 2 -> 3
    let adj = vec![vec![1], vec![2], vec![3], vec![]];
    let csr = V0CsrGraph::from_adjacency(&adj);

    let mut scratch = SharedSingleScratch::new(4);
    let cfg = TraversalConfig::default();
    let (edges, reached) = traverse_bfs_single(&csr, 0, &cfg, &mut scratch);
    assert_eq!(edges, 3);
    assert_eq!(reached, 4);

    let (lvl_edges, lvl_reached, levels) = traverse_and_record_levels(&csr, 0);
    assert_eq!(lvl_edges, 3);
    assert_eq!(lvl_reached, 4);
    assert_eq!(levels, vec![0, 1, 2, 3]);
}

#[test]
fn test_v0_cycle_and_disconnected() {
    // Cycle with branch: 0 -> 1, 1 -> 2, 2 -> 0, 1 -> 3
    // Disconnected: 4 -> 5
    let adj = vec![vec![1], vec![2, 3], vec![0], vec![], vec![5], vec![]];
    let csr = V0CsrGraph::from_adjacency(&adj);

    let mut scratch = SharedSingleScratch::new(6);
    let cfg = TraversalConfig::default();
    let (edges, reached) = traverse_bfs_single(&csr, 0, &cfg, &mut scratch);
    assert_eq!(edges, 4); // 0->1, 1->2, 1->3, 2->0
    assert_eq!(reached, 4); // 0, 1, 2, 3

    let (_, _, levels) = traverse_and_record_levels(&csr, 0);
    assert_eq!(levels[0], 0);
    assert_eq!(levels[1], 1);
    assert_eq!(levels[2], 2);
    assert_eq!(levels[3], 2);
    assert_eq!(levels[4], u32::MAX);
    assert_eq!(levels[5], u32::MAX);
}

#[test]
fn test_v0_reordering_levels_identity() {
    let n = 10_000usize;
    let rmat = generate_rmat_graph(n, 4, RmatConfig::default(), 0x1234_5678);
    let csr_orig = V0CsrGraph::from_nodes_and_overflow(&rmat.nodes, &rmat.overflow);

    let mut orig_adj = vec![Vec::new(); n];
    for u in 0..n {
        let lo = csr_orig.offsets[u] as usize;
        let hi = csr_orig.offsets[u + 1] as usize;
        orig_adj[u] = csr_orig.targets[lo..hi].to_vec();
    }

    let (_, reached_base, levels_base) = traverse_and_record_levels(&csr_orig, 0);

    for &order in &[VertexOrder::DegreeSort, VertexOrder::BfsOrder] {
        // Source node for BfsOrder is disjoint from benchmark source (n - 1 vs 0)
        let bfs_seed = (n - 1) as u32;
        let perm = compute_permutation(&orig_adj, order, bfs_seed);

        // Verify permutation is bijective
        let mut seen = vec![false; n];
        for &new_id in &perm.orig_to_new {
            assert!((new_id as usize) < n);
            assert!(!seen[new_id as usize]);
            seen[new_id as usize] = true;
        }

        let (reordered_adj, _) = apply_permutation(&orig_adj, &perm);
        let csr_reordered = V0CsrGraph::from_adjacency(&reordered_adj);

        let reordered_start = perm.orig_to_new[0];
        let (_, reached_reord, levels_reord) =
            traverse_and_record_levels(&csr_reordered, reordered_start);

        assert_eq!(reached_base, reached_reord);

        // Map levels back to original IDs and assert exact identity
        let mut mapped_levels = vec![u32::MAX; n];
        for orig_u in 0..n {
            let new_u = perm.orig_to_new[orig_u] as usize;
            mapped_levels[orig_u] = levels_reord[new_u];
        }

        assert_eq!(
            levels_base, mapped_levels,
            "BFS levels mapped from {:?} must match baseline exactly",
            order
        );
    }
}

#[test]
fn test_v0_switches_equivalence() {
    let n = 20_000usize;
    let rmat = generate_rmat_graph(n, 4, RmatConfig::default(), 0xABCD_EF01);
    let csr = V0CsrGraph::from_nodes_and_overflow(&rmat.nodes, &rmat.overflow);

    let mut single_scratch = SharedSingleScratch::new(n);
    let base_cfg = TraversalConfig::default();
    let (base_edges, base_reached) =
        traverse_bfs_single(&csr, 0, &base_cfg, &mut single_scratch);

    // Test all prefetch modes in single-thread
    let prefetch_modes = [
        PrefetchMode::None,
        PrefetchMode::KAhead(4),
        PrefetchMode::KAhead(8),
        PrefetchMode::KAhead(16),
        PrefetchMode::TwoStage(4),
        PrefetchMode::TwoStage(8),
        PrefetchMode::TwoStage(16),
        PrefetchMode::KAhead(32),
        PrefetchMode::Batched(8),
        PrefetchMode::Batched(16),
    ];

    for &pf in &prefetch_modes {
        let cfg = TraversalConfig {
            num_threads: 1,
            test_before_atomic: false,
            prefetch: pf,
        };
        let (e, r) = traverse_bfs_single(&csr, 0, &cfg, &mut single_scratch);
        assert_eq!(e, base_edges, "Single-thread edges mismatch for {:?}", pf);
        assert_eq!(r, base_reached, "Single-thread reached mismatch for {:?}", pf);
    }

    // Test parallel across threads and test_before_atomic
    let pool_4 = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    let mut par_scratch = SharedParallelScratch::new(n);

    for &test_before in &[false, true] {
        for &pf in &prefetch_modes {
            let cfg = TraversalConfig {
                num_threads: 4,
                test_before_atomic: test_before,
                prefetch: pf,
            };
            let (e, r) = traverse_bfs_parallel(&csr, 0, &pool_4, &cfg, &mut par_scratch);
            assert_eq!(e, base_edges, "Parallel edges mismatch for tba={}, pf={:?}", test_before, pf);
            assert_eq!(r, base_reached, "Parallel reached mismatch for tba={}, pf={:?}", test_before, pf);
        }
    }
}
