//! Stage 1 Gate Verification:
//! Evaluates whether V0 (using the unified BFS driver) matches the existing
//! NativeCsrGraph baseline within sample standard deviation (n=3 sessions).
//! Run with:
//!   cargo run --release --example stage1_gate

use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;
use rayon::prelude::*;

use state_search_engine::engine::EngineScratch;
use state_search_engine::graph::{generate_rmat_graph, RmatConfig};
use state_search_engine::node::{StateNode, MAX_INLINE_EDGES};
use state_search_engine::variants::bfs_driver::{traverse_bfs_parallel, traverse_bfs_single};
use state_search_engine::variants::traits::TraversalConfig;
use state_search_engine::variants::v0_csr::V0CsrGraph;
use state_search_engine::variants::visited::{SharedParallelScratch, SharedSingleScratch};

const SESSIONS: usize = 3;
const WARMUP: usize = 3;
const RUNS: usize = 10;

/// Existing Native Rust CSR Reference Implementation (identical to external_baseline_comparison.rs)
pub struct ExistingNativeCsrGraph {
    pub offsets: Vec<u32>,
    pub targets: Vec<u32>,
    pub num_nodes: usize,
    pub num_edges: usize,
}

impl ExistingNativeCsrGraph {
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

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn sample_sd(v: &[f64], mean_val: f64) -> f64 {
    if v.len() <= 1 {
        return 0.0;
    }
    let var = v.iter().map(|&x| (x - mean_val).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
    var.sqrt()
}

fn main() {
    println!("=========================================================================================");
    println!("                   STAGE 1 GATE: EXISTING CSR BASELINE VS V0 CSR                         ");
    println!("=========================================================================================\n");

    let sizes = [500_000usize, 1_000_000usize];
    let thread_counts = [1usize, 4, 6];

    for &n in &sizes {
        println!("-----------------------------------------------------------------------------------------");
        println!(" Graph Scale N = {n} (R-MAT, edge factor 4)");
        println!("-----------------------------------------------------------------------------------------");

        let rmat = generate_rmat_graph(n, 4, RmatConfig::default(), 0xDEAD_BEEF ^ n as u64);
        let existing_csr = ExistingNativeCsrGraph::from_nodes(&rmat.nodes, &rmat.overflow);
        let v0_csr = V0CsrGraph::from_nodes_and_overflow(&rmat.nodes, &rmat.overflow);

        for &threads in &thread_counts {
            // Arrays to store session medians: [session_idx]
            let mut existing_medians = Vec::new();
            let mut v0_medians = Vec::new();

            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();

            // Pre-allocated scratch buffers
            let mut existing_single_scratch = EngineScratch::new(n);
            let existing_par_visited: Vec<AtomicBool> = (0..n).map(|_| AtomicBool::new(false)).collect();

            let mut v0_single_scratch = SharedSingleScratch::new(n);
            let mut v0_par_scratch = SharedParallelScratch::new(n);

            let v0_cfg = TraversalConfig {
                num_threads: threads,
                test_before_atomic: false,
                prefetch: state_search_engine::variants::traits::PrefetchMode::None,
            };

            for session in 0..SESSIONS {
                let mut existing_runs = Vec::new();
                let mut v0_runs = Vec::new();

                for r in 0..(WARMUP + RUNS) {
                    let run_existing_first = (r + session) % 2 == 0;
                    let (d_exist, d_v0, e_exist, e_v0);

                    if run_existing_first {
                        let t0 = Instant::now();
                        e_exist = if threads == 1 {
                            existing_csr.traverse_bfs_single(black_box(0), &mut existing_single_scratch)
                        } else {
                            existing_csr.traverse_bfs_parallel_with_pool(black_box(0), &pool, &existing_par_visited)
                        };
                        d_exist = t0.elapsed().as_secs_f64() * 1e3;

                        let t1 = Instant::now();
                        e_v0 = if threads == 1 {
                            let (e, _) = traverse_bfs_single(&v0_csr, black_box(0), &v0_cfg, &mut v0_single_scratch);
                            e
                        } else {
                            let (e, _) = traverse_bfs_parallel(&v0_csr, black_box(0), &pool, &v0_cfg, &mut v0_par_scratch);
                            e
                        };
                        d_v0 = t1.elapsed().as_secs_f64() * 1e3;
                    } else {
                        let t1 = Instant::now();
                        e_v0 = if threads == 1 {
                            let (e, _) = traverse_bfs_single(&v0_csr, black_box(0), &v0_cfg, &mut v0_single_scratch);
                            e
                        } else {
                            let (e, _) = traverse_bfs_parallel(&v0_csr, black_box(0), &pool, &v0_cfg, &mut v0_par_scratch);
                            e
                        };
                        d_v0 = t1.elapsed().as_secs_f64() * 1e3;

                        let t0 = Instant::now();
                        e_exist = if threads == 1 {
                            existing_csr.traverse_bfs_single(black_box(0), &mut existing_single_scratch)
                        } else {
                            existing_csr.traverse_bfs_parallel_with_pool(black_box(0), &pool, &existing_par_visited)
                        };
                        d_exist = t0.elapsed().as_secs_f64() * 1e3;
                    }

                    assert_eq!(e_exist, e_v0, "Edge count mismatch between existing CSR and V0!");

                    if r >= WARMUP {
                        existing_runs.push(d_exist);
                        v0_runs.push(d_v0);
                    }
                }

                existing_medians.push(median(&mut existing_runs));
                v0_medians.push(median(&mut v0_runs));
            }

            let mean_exist = mean(&existing_medians);
            let sd_exist = sample_sd(&existing_medians, mean_exist);

            let mean_v0 = mean(&v0_medians);
            let sd_v0 = sample_sd(&v0_medians, mean_v0);

            let diff_pct = ((mean_v0 - mean_exist) / mean_exist) * 100.0;
            let within_sd = (mean_v0 - mean_exist).abs() <= (sd_exist.max(sd_v0));

            println!(
                "  Thread {:1}T | Exist: {:7.3} ms (SD: {:5.2}%) | V0: {:7.3} ms (SD: {:5.2}%) | Diff: {:+5.2}% | Within SD: {}",
                threads,
                mean_exist,
                (sd_exist / mean_exist) * 100.0,
                mean_v0,
                (sd_v0 / mean_v0) * 100.0,
                diff_pct,
                if within_sd { "PASS" } else { "BORDERLINE/OUTSIDE" }
            );
        }
        println!();
    }

    println!("=========================================================================================");
    println!(" Gate evaluation complete.");
    println!("=========================================================================================");
}
