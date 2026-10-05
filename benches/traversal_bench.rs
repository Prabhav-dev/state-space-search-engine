use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use state_search_engine::engine::SearchEngine;
use state_search_engine::frontier::{PartitionedFrontier, SharedCasFrontier};
use state_search_engine::graph::{generate_rmat_graph, generate_uniform_graph, RmatConfig};
use state_search_engine::node::{StateNode, MAX_INLINE_EDGES};

fn naive_bfs_edge_count(nodes: &[StateNode], start_node_id: u32) -> u64 {
    let mut visited = vec![false; nodes.len()];
    let mut frontier = vec![start_node_id];
    let mut next_frontier = Vec::new();
    let mut edges_traversed = 0u64;

    visited[start_node_id as usize] = true;

    while !frontier.is_empty() {
        for &node_id in &frontier {
            let node = &nodes[node_id as usize];
            let count = (node.outbound_count as usize).min(MAX_INLINE_EDGES);
            for neighbor_id in &node.outbound_links[..count] {
                let neighbor_idx = *neighbor_id as usize;
                if neighbor_idx < nodes.len() {
                    edges_traversed += 1;
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

    edges_traversed
}

fn run_partitioned_frontier_bfs(nodes: &[StateNode], start_node_id: u32) -> u64 {
    let mut frontier = PartitionedFrontier::new(nodes.len());
    let mut next_frontier = PartitionedFrontier::new(nodes.len());
    let mut visited = vec![false; nodes.len()];
    let mut edges_traversed = 0u64;

    if (start_node_id as usize) < nodes.len() {
        frontier.push(start_node_id);
        visited[start_node_id as usize] = true;
    }

    while !frontier.is_empty() {
        while let Some(node_id) = frontier.pop() {
            let node = &nodes[node_id as usize];
            let count = (node.outbound_count as usize).min(MAX_INLINE_EDGES);
            for neighbor_id in &node.outbound_links[..count] {
                let neighbor_idx = *neighbor_id as usize;
                if neighbor_idx < nodes.len() {
                    edges_traversed += 1;
                    if !visited[neighbor_idx] {
                        visited[neighbor_idx] = true;
                        next_frontier.push(*neighbor_id);
                    }
                }
            }
        }
        std::mem::swap(&mut frontier, &mut next_frontier);
        next_frontier.clear();
    }

    edges_traversed
}

fn run_shared_cas_frontier_bfs(nodes: &[StateNode], start_node_id: u32) -> u64 {
    let mut frontier = SharedCasFrontier::new(nodes.len());
    let mut next_frontier = SharedCasFrontier::new(nodes.len());
    let mut visited = vec![false; nodes.len()];
    let mut edges_traversed = 0u64;

    if (start_node_id as usize) < nodes.len() {
        frontier.push(start_node_id);
        visited[start_node_id as usize] = true;
    }

    loop {
        let mut work_done = false;
        while let Some(node_id) = frontier.pop() {
            work_done = true;
            let node = &nodes[node_id as usize];
            let count = (node.outbound_count as usize).min(MAX_INLINE_EDGES);
            for neighbor_id in &node.outbound_links[..count] {
                let neighbor_idx = *neighbor_id as usize;
                if neighbor_idx < nodes.len() {
                    edges_traversed += 1;
                    if !visited[neighbor_idx] {
                        visited[neighbor_idx] = true;
                        next_frontier.push(*neighbor_id);
                    }
                }
            }
        }

        if !work_done {
            break;
        }

        frontier.clear();
        std::mem::swap(&mut frontier, &mut next_frontier);
        next_frontier.clear();
    }

    edges_traversed
}

fn bench_engine_traversal(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine_bfs_throughput");

    let node_counts = [
        ("100K_nodes", 100_000),
        ("1M_nodes", 1_000_000),
        ("10M_nodes_exceeds_cache", 10_000_000),
    ];

    for (label, count) in node_counts.iter() {
        let nodes = generate_uniform_graph(*count, 2, 42);
        let engine = SearchEngine::new(nodes);

        group.throughput(Throughput::Elements(*count as u64));
        group.bench_with_input(BenchmarkId::new("uniform_graph", label), count, |b, _| {
            b.iter(|| {
                let traversed_edges = engine.traverse_bfs(black_box(0));
                black_box(traversed_edges);
            });
        });
    }

    group.finish();
}

fn bench_cache_cliff_uniform(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_cliff_uniform");
    for (label, count) in [("1M_uniform", 1_000_000), ("10M_uniform", 10_000_000)] {
        let nodes = generate_uniform_graph(count, 2, 42);
        let engine = SearchEngine::new(nodes);

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("throughput", label), &count, |b, _| {
            b.iter(|| {
                let edges = engine.traverse_bfs(black_box(0));
                black_box(edges);
            });
        });
    }
    group.finish();
}

fn bench_frontier_contention_power_law(c: &mut Criterion) {
    let mut group = c.benchmark_group("frontier_contention_power_law");
    let node_count = 100_000usize;
    let rmat = generate_rmat_graph(node_count, 4, RmatConfig::default(), 42);

    group.throughput(Throughput::Elements(node_count as u64));
    group.bench_function("partitioned_frontier", |b| {
        b.iter(|| {
            let edges = run_partitioned_frontier_bfs(&rmat.nodes, black_box(0));
            black_box(edges);
        });
    });

    group.bench_function("shared_cas_frontier", |b| {
        b.iter(|| {
            let edges = run_shared_cas_frontier_bfs(&rmat.nodes, black_box(0));
            black_box(edges);
        });
    });

    group.finish();
}

fn bench_engine_vs_baseline_100k(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine_vs_baseline_100k");
    let node_count = 100_000usize;
    let nodes = generate_uniform_graph(node_count, 2, 42);

    let engine = SearchEngine::new(nodes.clone());
    let baseline_edges = naive_bfs_edge_count(&nodes, 0);
    let engine_edges = engine.traverse_bfs(0);

    assert_eq!(engine_edges, baseline_edges, "engine should match the naive BFS baseline");

    group.throughput(Throughput::Elements(node_count as u64));
    group.bench_function("search_engine_vs_naive_bfs", |b| {
        b.iter(|| {
            let current_edges = engine.traverse_bfs(black_box(0));
            let baseline_edges = naive_bfs_edge_count(&nodes, black_box(0));
            black_box((current_edges, baseline_edges));
        });
    });
    group.finish();
}

fn bench_engine_vs_naive_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine_vs_naive_baseline");

    for (label, count) in [("100K", 100_000usize), ("1M", 1_000_000usize)] {
        let nodes = generate_uniform_graph(count, 2, 42);
        let engine = SearchEngine::new(nodes.clone());
        let baseline_edges = naive_bfs_edge_count(&nodes, 0);
        let engine_edges = engine.traverse_bfs(0);

        assert_eq!(engine_edges, baseline_edges, "engine must match naive BFS on the comparison graph");

        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(BenchmarkId::new("engine", label), &count, |b, _| {
            b.iter(|| {
                let edges = engine.traverse_bfs(black_box(0));
                black_box(edges);
            });
        });

        group.bench_with_input(BenchmarkId::new("naive_bfs", label), &count, |b, _| {
            b.iter(|| {
                let edges = naive_bfs_edge_count(&nodes, black_box(0));
                black_box(edges);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_engine_traversal,
    bench_cache_cliff_uniform,
    bench_frontier_contention_power_law,
    bench_engine_vs_baseline_100k,
    bench_engine_vs_naive_baseline
);
criterion_main!(benches);