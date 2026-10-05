use state_search_engine::engine::SearchEngine;
use state_search_engine::hash::PathHasher;
use state_search_engine::node::StateNode;

/// Helper function to build a node record cleanly.
/// Empty link slots MUST use u32::MAX as sentinel — 0 is a valid node index.
fn make_node(id: u32, links: [u32; 4], count: u32) -> StateNode {
    StateNode {
        node_id: id,
        structural_mask: 0,
        outbound_count: count,
        metadata_flags: 0,
        outbound_links: links,
    }
}

#[test]
fn test_line_graph_reachability() {
    // Topology: 0 -> 1 -> 2 -> 3  (empty slots = u32::MAX)
    let nodes = vec![
        make_node(0, [1, u32::MAX, u32::MAX, u32::MAX], 1),
        make_node(1, [2, u32::MAX, u32::MAX, u32::MAX], 1),
        make_node(2, [3, u32::MAX, u32::MAX, u32::MAX], 1),
        make_node(3, [u32::MAX; 4], 0),
    ];

    let engine = SearchEngine::new(nodes);
    let edges_traversed = engine.traverse_bfs(0);

    // Should traverse exactly 3 edges across the line graph
    assert_eq!(edges_traversed, 3);
}

#[test]
fn test_cycle_graph_deduplication() {
    // Topology with a cycle: 0 -> 1 -> 2 -> 0 (loop back)
    // Node 2 links back to node index 0 — with u32::MAX sentinel, 0 is recognised
    // as a valid edge target, so the back-edge IS counted (but node 0 is not re-enqueued
    // because it is already in the visited set).
    let nodes = vec![
        make_node(0, [1, u32::MAX, u32::MAX, u32::MAX], 1),
        make_node(1, [2, u32::MAX, u32::MAX, u32::MAX], 1),
        make_node(2, [0, u32::MAX, u32::MAX, u32::MAX], 1), // back-edge to node 0
    ];

    let engine = SearchEngine::new(nodes);
    let edges_traversed = engine.traverse_bfs(0);

    // Edges inspected: 0->1, 1->2, 2->0 (already visited, so stops) = 3
    assert_eq!(edges_traversed, 3);
}

#[test]
fn test_disconnected_graph() {
    // Component 1: 0 -> 1 | Component 2: 2 -> 3
    let nodes = vec![
        make_node(0, [1, u32::MAX, u32::MAX, u32::MAX], 1),
        make_node(1, [u32::MAX; 4], 0),
        make_node(2, [3, u32::MAX, u32::MAX, u32::MAX], 1),
        make_node(3, [u32::MAX; 4], 0),
    ];

    let engine = SearchEngine::new(nodes);
    let edges_traversed = engine.traverse_bfs(0);

    // Traversal from 0 should never reach nodes 2 or 3
    assert_eq!(edges_traversed, 1);
}

#[test]
fn test_zobrist_hash_integration() {
    // Test PathHasher consistency across a path: 0 -> 1 -> 2
    let node_keys: [u64; 3] = [0x9e3779b97f4a7c15, 0xbf58476d1ce4e5b9, 0x94d049bb133111eb];

    let mut hasher1 = PathHasher::new(0);
    hasher1.update(node_keys[0], 1);
    hasher1.update(node_keys[1], 1);
    hasher1.update(node_keys[2], 1);

    let mut hasher2 = PathHasher::new(0);
    hasher2.update(node_keys[0], 1);
    hasher2.update(node_keys[1], 1);
    hasher2.update(node_keys[2], 1);

    // Identical paths must yield identical state hashes
    assert_eq!(hasher1.hash(), hasher2.hash());
}

#[test]
fn test_engine_matches_naive_bfs_on_100k_graph() {
    let node_count = 100_000usize;
    let mut nodes = Vec::with_capacity(node_count);

    for i in 0..node_count {
        let target_1 = ((i * 7) + 1) % node_count;
        let target_2 = ((i * 13) + 3) % node_count;

        nodes.push(StateNode {
            node_id: i as u32,
            structural_mask: 0,
            outbound_count: 2,
            metadata_flags: 0,
            outbound_links: [target_1 as u32, target_2 as u32, u32::MAX, u32::MAX],
        });
    }

    let engine = SearchEngine::new(nodes.clone());
    let engine_edges = engine.traverse_bfs(0);

    let mut visited = vec![false; nodes.len()];
    let mut frontier = vec![0u32];
    let mut next_frontier = Vec::new();
    let mut baseline_edges = 0u64;

    visited[0] = true;

    while !frontier.is_empty() {
        for &node_id in &frontier {
            let node = &nodes[node_id as usize];
            for neighbor_id in &node.outbound_links[..node.outbound_count as usize] {
                let neighbor_idx = *neighbor_id as usize;
                if neighbor_idx < nodes.len() {
                    baseline_edges += 1;
                    if !visited[neighbor_idx] {
                        visited[neighbor_idx] = true;
                        next_frontier.push(*neighbor_id);
                    }
                }
            }
        }

        frontier.clear();
        std::mem::swap(&mut frontier, &mut next_frontier);
    }

    assert_eq!(engine_edges, baseline_edges, "engine and naive BFS must match on the 100K starter graph");
}

#[test]
fn test_parallel_bfs_matches_single_threaded_on_100k() {
    let node_count = 100_000usize;
    let mut nodes = Vec::with_capacity(node_count);
    for i in 0..node_count {
        let t1 = ((i * 7) + 1) % node_count;
        let t2 = ((i * 13) + 3) % node_count;
        nodes.push(StateNode {
            node_id: i as u32,
            structural_mask: 0,
            outbound_count: 2,
            metadata_flags: 0,
            outbound_links: [t1 as u32, t2 as u32, u32::MAX, u32::MAX],
        });
    }

    let engine = SearchEngine::new(nodes);
    let seq_edges = engine.traverse_bfs(0);
    let par_edges_2 = engine.traverse_parallel_bfs(0, 2);
    let par_edges_4 = engine.traverse_parallel_bfs(0, 4);

    assert_eq!(seq_edges, par_edges_2, "2-thread parallel BFS must match single-threaded");
    assert_eq!(seq_edges, par_edges_4, "4-thread parallel BFS must match single-threaded");
}

#[test]
fn test_rmat_power_law_graph_generation() {
    use state_search_engine::graph::{generate_rmat_graph, RmatConfig};

    let node_count = 1024;
    let result = generate_rmat_graph(node_count, 4, RmatConfig::default(), 42);

    assert_eq!(result.nodes.len(), node_count);
    assert!(result.reached_nodes > 0, "R-MAT graph should reach nodes");
    assert!(result.max_depth > 0, "R-MAT graph should have positive BFS depth");
    assert_eq!(result.total_edges, node_count * 4);
}

#[test]
fn test_zobrist_deduplication_memory_and_false_positives() {
    let node_count = 10_000usize;
    let mut nodes = Vec::with_capacity(node_count);
    for i in 0..node_count {
        let t1 = ((i * 7) + 1) % node_count;
        let t2 = ((i * 13) + 3) % node_count;
        nodes.push(StateNode {
            node_id: i as u32,
            structural_mask: 0,
            outbound_count: 2,
            metadata_flags: 0,
            outbound_links: [t1 as u32, t2 as u32, u32::MAX, u32::MAX],
        });
    }

    let engine = SearchEngine::new(nodes);

    // Test with 14 bits (16K bits = 2KB memory)
    let (reached, exact_reachable, coverage_pct, _false_positives) =
        engine.evaluate_zobrist_bitstate_coverage(0, 14);

    assert!(reached > 0);
    assert!(exact_reachable >= reached);
    assert!(coverage_pct <= 100.0);
}

#[test]
fn test_learned_edge_priorities_reduces_expansions() {
    let node_count = 1_000usize;
    let mut nodes = Vec::with_capacity(node_count);
    for i in 0..node_count {
        let t1 = (i + 1) % node_count;
        let t2 = (i + 10) % node_count;
        nodes.push(StateNode {
            node_id: i as u32,
            structural_mask: 0,
            outbound_count: 2,
            metadata_flags: 0,
            outbound_links: [t1 as u32, t2 as u32, u32::MAX, u32::MAX],
        });
    }

    let engine = SearchEngine::new(nodes);
    let mut priorities = vec![Vec::new(); node_count];
    let target = 500u32;

    // Train on target
    let (exp_before, found1) =
        engine.search_with_learned_priorities(0, target, &mut priorities, true);
    assert!(found1);

    // Second search on same target (memoized path)
    let (exp_after, found2) =
        engine.search_with_learned_priorities(0, target, &mut priorities, false);
    assert!(found2);

    assert!(exp_after <= exp_before, "Learned edge priorities should not increase expansions");
}

#[test]
fn test_overflow_variable_degree_traversal() {
    // 1 node with degree 8
    let mut nodes = vec![StateNode {
        node_id: 0,
        structural_mask: 0,
        outbound_count: 8,
        metadata_flags: 0,
        outbound_links: [1, 2, 3, 0], // overflow starts at index 0
    }];

    for i in 1..=8 {
        nodes.push(StateNode {
            node_id: i,
            structural_mask: 0,
            outbound_count: 0,
            metadata_flags: 0,
            outbound_links: [u32::MAX; 4],
        });
    }

    let overflow = vec![4, 5, 6, 7, 8];
    let engine = SearchEngine::with_overflow(nodes, overflow);
    let edges = engine.traverse_bfs(0);
    assert_eq!(edges, 8, "Should traverse all 8 outbound edges");
}