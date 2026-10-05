//! Benchmark comparing the Rust Cache-Resident Engine against tuned GAPBS / Ligra CSR BFS baseline (Step 7).
//! Runs in release mode with:
//!   cargo run --release --example external_baseline_comparison

use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;
use rayon::prelude::*;

use state_search_engine::engine::SearchEngine;
use state_search_engine::graph::{generate_rmat_graph, RmatConfig};
use state_search_engine::node::StateNode;

const WARMUP: usize = 3;
const RUNS: usize = 10;

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Tuned GAPBS / Ligra style Compressed Sparse Row (CSR) Graph
pub struct GapbsCsrGraph {
    pub offsets: Vec<u32>,
    pub targets: Vec<u32>,
    pub num_nodes: usize,
    pub num_edges: usize,
}

impl GapbsCsrGraph {
    pub fn from_nodes(nodes: &[StateNode], overflow: &[u32]) -> Self {
        let mut offsets = Vec::with_capacity(nodes.len() + 1);
        let mut targets = Vec::new();

        for node in nodes {
            offsets.push(targets.len() as u32);
            let count = node.outbound_count as usize;
            if count <= 4 {
                targets.extend_from_slice(&node.outbound_links[..count]);
            } else {
                targets.extend_from_slice(&node.outbound_links[..3]);
                let off = node.outbound_links[3] as usize;
                let remaining = count - 3;
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

    /// Single-threaded level-synchronous CSR BFS (GAPBS standard top-down baseline)
    pub fn traverse_bfs_single(&self, start_node: u32) -> u64 {
        let mut visited = vec![false; self.num_nodes];
        let mut cur_frontier = vec![start_node];
        let mut next_frontier = Vec::new();
        let mut edges_traversed = 0u64;

        if (start_node as usize) < self.num_nodes {
            visited[start_node as usize] = true;
        }

        while !cur_frontier.is_empty() {
            for &u in &cur_frontier {
                let lo = self.offsets[u as usize] as usize;
                let hi = self.offsets[u as usize + 1] as usize;
                for &v in &self.targets[lo..hi] {
                    let v_idx = v as usize;
                    if v_idx < self.num_nodes {
                        edges_traversed += 1;
                        if !visited[v_idx] {
                            visited[v_idx] = true;
                            next_frontier.push(v);
                        }
                    }
                }
            }
            cur_frontier.clear();
            std::mem::swap(&mut cur_frontier, &mut next_frontier);
        }

        edges_traversed
    }

    /// Parallel level-synchronous CSR BFS (GAPBS / Ligra parallel baseline)
    pub fn traverse_bfs_parallel(&self, start_node: u32, num_threads: usize) -> u64 {
        if self.num_nodes == 0 || start_node as usize >= self.num_nodes {
            return 0;
        }

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        pool.install(|| {
            let visited: Vec<AtomicBool> = (0..self.num_nodes)
                .map(|_| AtomicBool::new(false))
                .collect();
            let total_edges = AtomicU64::new(0);

            visited[start_node as usize].store(true, Ordering::Relaxed);
            let mut current_frontier = vec![start_node];

            while !current_frontier.is_empty() {
                let chunk_size = (current_frontier.len() + num_threads - 1) / num_threads;

                let next_frontiers: Vec<Vec<u32>> = current_frontier
                    .par_chunks(chunk_size.max(1))
                    .map(|chunk| {
                        let mut local_next = Vec::new();
                        let mut local_edges = 0u64;

                        for &u in chunk {
                            let lo = self.offsets[u as usize] as usize;
                            let hi = self.offsets[u as usize + 1] as usize;
                            for &v in &self.targets[lo..hi] {
                                let v_idx = v as usize;
                                if v_idx < self.num_nodes {
                                    local_edges += 1;
                                    if !visited[v_idx].swap(true, Ordering::AcqRel) {
                                        local_next.push(v);
                                    }
                                }
                            }
                        }

                        total_edges.fetch_add(local_edges, Ordering::Relaxed);
                        local_next
                    })
                    .collect();

                current_frontier = next_frontiers.into_iter().flatten().collect();
            }

            total_edges.load(Ordering::Relaxed)
        })
    }
}

fn main() {
    println!("=========================================================================================");
    println!("        EXTERNAL BASELINE COMPARISON: CACHE-RESISTANT ENGINE VS GAPBS / LIGRA (STEP 7)  ");
    println!("=========================================================================================\n");

    for &n in &[500_000usize, 1_000_000usize, 4_000_000usize] {
        let rmat = generate_rmat_graph(n, 4, RmatConfig::default(), 0xDEAD_BEEF ^ n as u64);
        let gapbs = GapbsCsrGraph::from_nodes(&rmat.nodes, &rmat.overflow);
        let engine = SearchEngine::with_overflow(rmat.nodes, rmat.overflow);

        println!("--- GRAPH SIZE N = {n} (Edges = {}) ---", gapbs.num_edges);
        println!(
            "  Memory Footprint: Engine 32-byte Arena = {:.1} MB | GAPBS CSR = {:.1} MB",
            (n * 32) as f64 / 1e6,
            ((gapbs.offsets.len() + gapbs.targets.len()) * 4) as f64 / 1e6
        );

        for &threads in &[1, 4, 6] {
            let mut t_engine = Vec::new();
            let mut t_gapbs = Vec::new();
            let mut edges_traversed = 0u64;

            for r in 0..(WARMUP + RUNS) {
                let start = Instant::now();
                let e1 = if threads == 1 {
                    engine.traverse_bfs(black_box(0))
                } else {
                    engine.traverse_parallel_bfs(black_box(0), threads)
                };
                let d1 = start.elapsed().as_secs_f64() * 1e3;

                let start = Instant::now();
                let e2 = if threads == 1 {
                    gapbs.traverse_bfs_single(black_box(0))
                } else {
                    gapbs.traverse_bfs_parallel(black_box(0), threads)
                };
                let d2 = start.elapsed().as_secs_f64() * 1e3;

                assert_eq!(e1, e2);
                edges_traversed = e1;
                if r >= WARMUP {
                    t_engine.push(d1);
                    t_gapbs.push(d2);
                }
            }

            let m_eng = median(&mut t_engine);
            let m_gap = median(&mut t_gapbs);

            let mteps_eng = edges_traversed as f64 / m_eng / 1e3;
            let mteps_gap = edges_traversed as f64 / m_gap / 1e3;
            let ratio = m_gap / m_eng;

            println!(
                "    Threads: {:2} | Engine: {:8.3} ms ({:5.1} MTEPS) | GAPBS CSR: {:8.3} ms ({:5.1} MTEPS) | Win: {:.2}x ({})",
                threads,
                m_eng,
                mteps_eng,
                m_gap,
                mteps_gap,
                ratio,
                if ratio >= 1.0 { "Engine Faster" } else { "GAPBS CSR Faster" }
            );
        }
        println!();
    }

    println!("=========================================================================================");
}
