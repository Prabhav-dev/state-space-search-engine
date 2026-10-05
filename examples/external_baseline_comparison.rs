//! Benchmark comparing the Cache-Resident Inline Arena Engine against Native Rust CSR Baseline.
//! Runs in release mode with:
//!   cargo run --release --example external_baseline_comparison

use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;
use rayon::prelude::*;

use state_search_engine::engine::{EngineScratch, ParallelEngineScratch, SearchEngine};
use state_search_engine::graph::{generate_rmat_graph, RmatConfig};
use state_search_engine::node::{StateNode, MAX_INLINE_EDGES};

const WARMUP: usize = 3;
const RUNS: usize = 10;

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Native Rust Compressed Sparse Row (CSR) Graph Reference Implementation
pub struct NativeCsrGraph {
    pub offsets: Vec<u32>,
    pub targets: Vec<u32>,
    pub num_nodes: usize,
    pub num_edges: usize,
}

impl NativeCsrGraph {
    pub fn from_nodes(nodes: &[StateNode], overflow: &[u32]) -> Self {
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

    /// Single-threaded level-synchronous CSR BFS using pre-allocated scratch
    pub fn traverse_bfs_single(&self, start_node: u32, scratch: &mut EngineScratch) -> u64 {
        scratch.reset();
        let EngineScratch { visited, cur, next } = scratch;
        let mut edges_traversed = 0u64;

        if (start_node as usize) < self.num_nodes {
            let w = (start_node >> 6) as usize;
            let m = 1u64 << (start_node & 63);
            visited[w] |= m;
            cur.push(start_node);
        }

        while !cur.is_empty() {
            for &u in cur.iter() {
                let lo = self.offsets[u as usize] as usize;
                let hi = self.offsets[u as usize + 1] as usize;
                for &v in &self.targets[lo..hi] {
                    let v_idx = v as usize;
                    if v_idx < self.num_nodes {
                        edges_traversed += 1;
                        let w = (v >> 6) as usize;
                        let m = 1u64 << (v & 63);
                        if (visited[w] & m) == 0 {
                            visited[w] |= m;
                            next.push(v);
                        }
                    }
                }
            }
            cur.clear();
            std::mem::swap(cur, next);
        }

        edges_traversed
    }

    /// Multi-threaded level-synchronous CSR BFS using pre-allocated thread pool & visited buffer
    pub fn traverse_bfs_parallel_with_pool(
        &self,
        start_node: u32,
        pool: &rayon::ThreadPool,
        visited: &[AtomicBool],
    ) -> u64 {
        if self.num_nodes == 0 || start_node as usize >= self.num_nodes {
            return 0;
        }

        for v in visited {
            v.store(false, Ordering::Relaxed);
        }

        let num_threads = pool.current_num_threads();

        pool.install(|| {
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
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

    println!("=========================================================================================");
    println!("        BENCHMARK COMPARISON: INLINE ARENA ENGINE VS NATIVE RUST CSR BASELINE            ");
    println!("        Host System: {} {} | Threads Evaluated: 1, 4, 6                                ", os, arch);
    println!("=========================================================================================\n");

    for &n in &[500_000usize, 1_000_000usize, 4_000_000usize] {
        let rmat = generate_rmat_graph(n, 4, RmatConfig::default(), 0xDEAD_BEEF ^ n as u64);
        let csr = NativeCsrGraph::from_nodes(&rmat.nodes, &rmat.overflow);
        let engine = SearchEngine::with_overflow(rmat.nodes, rmat.overflow);

        println!("--- GRAPH SIZE N = {n} (Edges = {}) ---", csr.num_edges);
        println!(
            "  Memory Footprint: Engine 32-byte Arena = {:.1} MB | Native CSR = {:.1} MB",
            (n * 32) as f64 / 1e6,
            ((csr.offsets.len() + csr.targets.len()) * 4) as f64 / 1e6
        );

        for &threads in &[1, 4, 6] {
            let mut t_engine = Vec::new();
            let mut t_csr = Vec::new();
            let mut edges_traversed = 0u64;

            // Pre-allocate thread pools, scratch buffers outside timed loop
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            let mut engine_scratch = EngineScratch::new(n);
            let mut engine_par_scratch = ParallelEngineScratch::new(n);
            let mut csr_scratch = EngineScratch::new(n);
            let csr_par_visited: Vec<AtomicBool> = (0..n).map(|_| AtomicBool::new(false)).collect();

            for r in 0..(WARMUP + RUNS) {
                // Interleave execution order between runs to eliminate thermal/cache warm-up bias
                let run_engine_first = r % 2 == 0;

                let (d1, e1, d2, e2);

                if run_engine_first {
                    let start = Instant::now();
                    e1 = if threads == 1 {
                        engine.traverse_bfs_with_scratch(black_box(0), &mut engine_scratch)
                    } else {
                        engine.traverse_parallel_bfs_with_pool_and_scratch(black_box(0), &pool, &mut engine_par_scratch)
                    };
                    d1 = start.elapsed().as_secs_f64() * 1e3;

                    let start = Instant::now();
                    e2 = if threads == 1 {
                        csr.traverse_bfs_single(black_box(0), &mut csr_scratch)
                    } else {
                        csr.traverse_bfs_parallel_with_pool(black_box(0), &pool, &csr_par_visited)
                    };
                    d2 = start.elapsed().as_secs_f64() * 1e3;
                } else {
                    let start = Instant::now();
                    e2 = if threads == 1 {
                        csr.traverse_bfs_single(black_box(0), &mut csr_scratch)
                    } else {
                        csr.traverse_bfs_parallel_with_pool(black_box(0), &pool, &csr_par_visited)
                    };
                    d2 = start.elapsed().as_secs_f64() * 1e3;

                    let start = Instant::now();
                    e1 = if threads == 1 {
                        engine.traverse_bfs_with_scratch(black_box(0), &mut engine_scratch)
                    } else {
                        engine.traverse_parallel_bfs_with_pool_and_scratch(black_box(0), &pool, &mut engine_par_scratch)
                    };
                    d1 = start.elapsed().as_secs_f64() * 1e3;
                }

                assert_eq!(e1, e2);
                edges_traversed = e1;
                if r >= WARMUP {
                    t_engine.push(d1);
                    t_csr.push(d2);
                }
            }

            let m_eng = median(&mut t_engine);
            let m_csr = median(&mut t_csr);

            let mteps_eng = edges_traversed as f64 / m_eng / 1e3;
            let mteps_csr = edges_traversed as f64 / m_csr / 1e3;
            let ratio = m_csr / m_eng;

            println!(
                "    Threads: {:2} | Engine: {:8.3} ms ({:5.1} MTEPS) | Native CSR: {:8.3} ms ({:5.1} MTEPS) | Win: {:.2}x ({})",
                threads,
                m_eng,
                mteps_eng,
                m_csr,
                mteps_csr,
                ratio,
                if ratio >= 1.0 { "Engine Faster" } else { "CSR Faster" }
            );
        }
        println!();
    }

    println!("=========================================================================================");
}
