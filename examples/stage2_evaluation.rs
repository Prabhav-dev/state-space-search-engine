//! Stage 2 Performance Evaluation:
//! Evaluates Existing CSR, V0 CSR, V2 Slim Arena, and V3 Inline-Visited
//! on R-MAT graphs across 1, 4, and 6 threads over 3 isolated sessions.
//! Run with:
//!   cargo run --release --example stage2_evaluation

use std::hint::black_box;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use state_search_engine::engine::EngineScratch;
use state_search_engine::graph::{generate_rmat_graph, RmatConfig};
use state_search_engine::variants::bfs_driver::{traverse_bfs_parallel, traverse_bfs_single};
use state_search_engine::variants::traits::{PrefetchMode, TraversalConfig};
use state_search_engine::variants::v0_csr::V0CsrGraph;
use state_search_engine::variants::v2_slim_arena::V2SlimArenaGraph;
use state_search_engine::variants::v3_inline_visited::V3InlineVisitedGraph;
use state_search_engine::variants::visited::{SharedParallelScratch, SharedSingleScratch};

// Reference implementation from external_baseline_comparison.rs
use crate::stage1_gate::ExistingNativeCsrGraph;

#[path = "stage1_gate.rs"]
mod stage1_gate;

const SESSIONS: usize = 3;
const WARMUP: usize = 3;
const RUNS: usize = 10;

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
    println!("        STAGE 2 EVALUATION: V0 (CSR) vs V2 (16B SLIM) vs V3 (16B INLINE VISITED)         ");
    println!("        Platform: Windows 11 AMD Ryzen 5 7430U | Thread Configurations: 1T, 4T, 6T | Sessions: 3");
    println!("=========================================================================================\n");

    let sizes = [500_000usize, 1_000_000usize];
    let thread_counts = [1usize, 4, 6];

    for &n in &sizes {
        let rmat = generate_rmat_graph(n, 4, RmatConfig::default(), 0xDEAD_BEEF ^ n as u64);
        let existing_csr = ExistingNativeCsrGraph::from_nodes(&rmat.nodes, &rmat.overflow);
        let v0 = V0CsrGraph::from_nodes_and_overflow(&rmat.nodes, &rmat.overflow);
        let v2 = V2SlimArenaGraph::from_nodes_and_overflow(&rmat.nodes, &rmat.overflow);
        let v3 = V3InlineVisitedGraph::from_nodes_and_overflow(&rmat.nodes, &rmat.overflow);

        let mem_csr = ((v0.offsets.len() + v0.targets.len()) * 4) as f64 / 1e6;
        let mem_v2 = ((v2.nodes.len() * 16) + (v2.overflow.len() * 4)) as f64 / 1e6;
        let mem_v3 = ((v3.nodes.len() * 16) + (v3.overflow.len() * 4)) as f64 / 1e6;

        println!("-----------------------------------------------------------------------------------------");
        println!(" Graph Scale N = {n} (Edges = {})", v0.num_edges);
        println!("   Memory Footprints: CSR = {:.2} MB | V2 = {:.2} MB | V3 = {:.2} MB", mem_csr, mem_v2, mem_v3);
        println!("-----------------------------------------------------------------------------------------");

        for &threads in &thread_counts {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();

            // Scratch buffers allocated outside timed regions
            let mut exist_scratch = EngineScratch::new(n);
            let exist_visited: Vec<AtomicBool> = (0..n).map(|_| AtomicBool::new(false)).collect();

            let mut single_scratch = SharedSingleScratch::new(n);
            let mut par_scratch = SharedParallelScratch::new(n);

            let cfg = TraversalConfig {
                num_threads: threads,
                test_before_atomic: false,
                prefetch: PrefetchMode::None,
            };

            let mut s_exist_med = Vec::new();
            let mut s_v0_med = Vec::new();
            let mut s_v2_med = Vec::new();
            let mut s_v3_med = Vec::new();
            let mut total_edges_traversed = 0u64;

            for session in 0..SESSIONS {
                let mut runs_exist = Vec::new();
                let mut runs_v0 = Vec::new();
                let mut runs_v2 = Vec::new();
                let mut runs_v3 = Vec::new();

                for r in 0..(WARMUP + RUNS) {
                    // Interleave 4 variants across runs to eliminate warm-up order bias
                    let order = (r + session) % 4;

                    let mut run_variant = |idx: usize| -> (f64, u64) {
                        let t = Instant::now();
                        let edges = match idx {
                            0 => {
                                if threads == 1 {
                                    existing_csr.traverse_bfs_single(black_box(0), &mut exist_scratch)
                                } else {
                                    existing_csr.traverse_bfs_parallel_with_pool(black_box(0), &pool, &exist_visited)
                                }
                            }
                            1 => {
                                if threads == 1 {
                                    traverse_bfs_single(&v0, black_box(0), &cfg, &mut single_scratch).0
                                } else {
                                    traverse_bfs_parallel(&v0, black_box(0), &pool, &cfg, &mut par_scratch).0
                                }
                            }
                            2 => {
                                if threads == 1 {
                                    traverse_bfs_single(&v2, black_box(0), &cfg, &mut single_scratch).0
                                } else {
                                    traverse_bfs_parallel(&v2, black_box(0), &pool, &cfg, &mut par_scratch).0
                                }
                            }
                            3 => {
                                if threads == 1 {
                                    traverse_bfs_single(&v3, black_box(0), &cfg, &mut single_scratch).0
                                } else {
                                    traverse_bfs_parallel(&v3, black_box(0), &pool, &cfg, &mut par_scratch).0
                                }
                            }
                            _ => unreachable!(),
                        };
                        let elapsed = t.elapsed().as_secs_f64() * 1e3;
                        (elapsed, edges)
                    };

                    let idxs = [order, (order + 1) % 4, (order + 2) % 4, (order + 3) % 4];
                    let mut times = [0.0; 4];
                    let mut edges = [0u64; 4];

                    for &idx in &idxs {
                        let (time_ms, edge_count) = run_variant(idx);
                        times[idx] = time_ms;
                        edges[idx] = edge_count;
                    }

                    assert_eq!(edges[0], edges[1]);
                    assert_eq!(edges[1], edges[2]);
                    assert_eq!(edges[2], edges[3]);
                    total_edges_traversed = edges[0];

                    if r >= WARMUP {
                        runs_exist.push(times[0]);
                        runs_v0.push(times[1]);
                        runs_v2.push(times[2]);
                        runs_v3.push(times[3]);
                    }
                }

                s_exist_med.push(median(&mut runs_exist));
                s_v0_med.push(median(&mut runs_v0));
                s_v2_med.push(median(&mut runs_v2));
                s_v3_med.push(median(&mut runs_v3));
            }

            let m_exist = mean(&s_exist_med);
            let sd_exist = sample_sd(&s_exist_med, m_exist);

            let m_v0 = mean(&s_v0_med);
            let sd_v0 = sample_sd(&s_v0_med, m_v0);

            let m_v2 = mean(&s_v2_med);
            let sd_v2 = sample_sd(&s_v2_med, m_v2);

            let m_v3 = mean(&s_v3_med);
            let sd_v3 = sample_sd(&s_v3_med, m_v3);

            let mteps = |ms: f64| total_edges_traversed as f64 / ms / 1e3;

            println!(
                " Thread {:1}T | Exist: {:7.3} ms ({:5.1} MTEPS, SD {:4.2}%) | V0: {:7.3} ms ({:5.1} MTEPS, SD {:4.2}%)",
                threads, m_exist, mteps(m_exist), (sd_exist / m_exist) * 100.0,
                m_v0, mteps(m_v0), (sd_v0 / m_v0) * 100.0
            );
            println!(
                "           | V2:    {:7.3} ms ({:5.1} MTEPS, SD {:4.2}%) | Speedup vs V0: {:.2}x ({})",
                m_v2, mteps(m_v2), (sd_v2 / m_v2) * 100.0,
                m_v0 / m_v2,
                if m_v0 / m_v2 >= 1.10 { "WIN >=10%" } else if m_v0 / m_v2 > 1.0 { "Faster <10%" } else { "Slower" }
            );
            println!(
                "           | V3:    {:7.3} ms ({:5.1} MTEPS, SD {:4.2}%) | Speedup vs V0: {:.2}x ({})",
                m_v3, mteps(m_v3), (sd_v3 / m_v3) * 100.0,
                m_v0 / m_v3,
                if m_v0 / m_v3 >= 1.10 { "WIN >=10%" } else if m_v0 / m_v3 > 1.0 { "Faster <10%" } else { "Slower" }
            );
            println!();
        }
    }
    println!("=========================================================================================");
}
