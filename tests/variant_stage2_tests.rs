use std::mem::{align_of, size_of};

use state_search_engine::graph::{generate_rmat_graph, RmatConfig};
use state_search_engine::variants::bfs_driver::{
    traverse_and_record_levels, traverse_bfs_parallel, traverse_bfs_single,
};
use state_search_engine::variants::reorder::{apply_permutation, compute_permutation};
use state_search_engine::variants::traits::{PrefetchMode, TraversalConfig, VertexOrder};
use state_search_engine::variants::v0_csr::V0CsrGraph;
use state_search_engine::variants::v2_slim_arena::{SlimNode16, V2SlimArenaGraph};
use state_search_engine::variants::v3_inline_visited::{SlimNode16Visited, V3InlineVisitedGraph};
use state_search_engine::variants::visited::{SharedParallelScratch, SharedSingleScratch};

#[test]
fn test_stage2_node_sizes_and_alignments() {
    assert_eq!(size_of::<SlimNode16>(), 16, "SlimNode16 must be exactly 16 bytes");
    assert_eq!(align_of::<SlimNode16>(), 16, "SlimNode16 must be 16-byte aligned");

    assert_eq!(size_of::<SlimNode16Visited>(), 16, "SlimNode16Visited must be exactly 16 bytes");
    assert_eq!(align_of::<SlimNode16Visited>(), 16, "SlimNode16Visited must be 16-byte aligned");
}

#[test]
fn test_stage2_v2_and_v3_correctness_against_v0_on_rmat() {
    let n = 20_000usize;
    let rmat = generate_rmat_graph(n, 4, RmatConfig::default(), 0xFACE_B00C);

    let v0 = V0CsrGraph::from_nodes_and_overflow(&rmat.nodes, &rmat.overflow);
    let v2 = V2SlimArenaGraph::from_nodes_and_overflow(&rmat.nodes, &rmat.overflow);
    let v3 = V3InlineVisitedGraph::from_nodes_and_overflow(&rmat.nodes, &rmat.overflow);

    // 1. Verify BFS levels match baseline V0 exactly
    let (v0_edges, v0_reached, v0_levels) = traverse_and_record_levels(&v0, 0);
    let (v2_edges, v2_reached, v2_levels) = traverse_and_record_levels(&v2, 0);
    let (v3_edges, v3_reached, v3_levels) = traverse_and_record_levels(&v3, 0);

    assert_eq!(v0_edges, v2_edges, "V2 edges mismatch with V0");
    assert_eq!(v0_edges, v3_edges, "V3 edges mismatch with V0");
    assert_eq!(v0_reached, v2_reached, "V2 reached mismatch with V0");
    assert_eq!(v0_reached, v3_reached, "V3 reached mismatch with V0");
    assert_eq!(v0_levels, v2_levels, "V2 levels mismatch with V0");
    assert_eq!(v0_levels, v3_levels, "V3 levels mismatch with V0");

    // 2. Single-threaded driver verification across prefetch modes
    let mut single_scratch = SharedSingleScratch::new(n);
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

        let (e2, r2) = traverse_bfs_single(&v2, 0, &cfg, &mut single_scratch);
        assert_eq!(e2, v0_edges);
        assert_eq!(r2, v0_reached);

        let (e3, r3) = traverse_bfs_single(&v3, 0, &cfg, &mut single_scratch);
        assert_eq!(e3, v0_edges);
        assert_eq!(r3, v0_reached);
    }

    // 3. Multi-threaded parallel driver verification
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    let mut par_scratch = SharedParallelScratch::new(n);

    for &test_before in &[false, true] {
        for &pf in &[
            PrefetchMode::None,
            PrefetchMode::KAhead(8),
            PrefetchMode::TwoStage(4),
            PrefetchMode::TwoStage(8),
            PrefetchMode::TwoStage(16),
            PrefetchMode::Batched(8),
        ] {
            let cfg = TraversalConfig {
                num_threads: 4,
                test_before_atomic: test_before,
                prefetch: pf,
            };

            let (e2, r2) = traverse_bfs_parallel(&v2, 0, &pool, &cfg, &mut par_scratch);
            assert_eq!(e2, v0_edges);
            assert_eq!(r2, v0_reached);

            let (e3, r3) = traverse_bfs_parallel(&v3, 0, &pool, &cfg, &mut par_scratch);
            assert_eq!(e3, v0_edges);
            assert_eq!(r3, v0_reached);
        }
    }
}

#[test]
fn test_stage2_reordering_on_v2_and_v3() {
    let n = 15_000usize;
    let rmat = generate_rmat_graph(n, 4, RmatConfig::default(), 0x9988_7766);
    let v0_base = V0CsrGraph::from_nodes_and_overflow(&rmat.nodes, &rmat.overflow);

    let mut orig_adj = vec![Vec::new(); n];
    for u in 0..n {
        let lo = v0_base.offsets[u] as usize;
        let hi = v0_base.offsets[u + 1] as usize;
        orig_adj[u] = v0_base.targets[lo..hi].to_vec();
    }

    let (_, base_reached, base_levels) = traverse_and_record_levels(&v0_base, 0);

    for &order in &[VertexOrder::DegreeSort, VertexOrder::BfsOrder] {
        let bfs_seed = (n - 1) as u32;
        let perm = compute_permutation(&orig_adj, order, bfs_seed);
        let (reord_adj, _) = apply_permutation(&orig_adj, &perm);

        let v2_reord = V2SlimArenaGraph::from_adjacency(&reord_adj);
        let v3_reord = V3InlineVisitedGraph::from_adjacency(&reord_adj);

        let reord_start = perm.orig_to_new[0];

        let (_, r2, lvl2) = traverse_and_record_levels(&v2_reord, reord_start);
        let (_, r3, lvl3) = traverse_and_record_levels(&v3_reord, reord_start);

        assert_eq!(r2, base_reached);
        assert_eq!(r3, base_reached);

        // Map levels back to original IDs
        let mut map2 = vec![u32::MAX; n];
        let mut map3 = vec![u32::MAX; n];
        for orig_u in 0..n {
            let new_u = perm.orig_to_new[orig_u] as usize;
            map2[orig_u] = lvl2[new_u];
            map3[orig_u] = lvl3[new_u];
        }

        assert_eq!(map2, base_levels, "V2 mapped levels mismatch for {:?}", order);
        assert_eq!(map3, base_levels, "V3 mapped levels mismatch for {:?}", order);
    }
}
