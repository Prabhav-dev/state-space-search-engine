//! Stage 2 Multi-Session Rigorous Empirical Study (10 Sessions, 95% CI)
//! - V0 vs V2 vs V3 with load-then-fetch_or and bit-clearing outside timed region
//! - Overflow fraction p_o and overflow array telemetry
//! - 4M and 8M scaling of V0 vs V2 with Original / DegreeSort orders and Prefetch None / k=8 / k=16
//! Run with:
//!   cargo run --release --example stage2_multisession_study

use std::hint::black_box;
use std::time::Instant;

use state_search_engine::graph::{generate_rmat_graph, RmatConfig};
use state_search_engine::variants::bfs_driver::{
    prepare_scratch_parallel_for_threads, prepare_scratch_single, traverse_bfs_parallel_prepared,
    traverse_bfs_single_prepared,
};
use state_search_engine::variants::reorder::{apply_permutation, compute_permutation};
use state_search_engine::variants::traits::{PrefetchMode, TraversalConfig, VertexOrder};
use state_search_engine::variants::v0_csr::V0CsrGraph;
use state_search_engine::variants::v2_slim_arena::V2SlimArenaGraph;
use state_search_engine::variants::v3_inline_visited::V3InlineVisitedGraph;
use state_search_engine::variants::visited::{SharedParallelScratch, SharedSingleScratch};

const SESSIONS: usize = 10;
const WARMUP: usize = 3;
const RUNS: usize = 10;
// Student's t critical value for df = 9 at 95% confidence level (two-tailed alpha = 0.05)
const T_CRIT_95_DF9: f64 = 2.262;

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

/// Computes (mean, sd, 95% CI half-width)
fn stats_95_ci(v: &[f64]) -> (f64, f64, f64) {
    let m = mean(v);
    let sd = sample_sd(v, m);
    let se = sd / (v.len() as f64).sqrt();
    let ci_hw = T_CRIT_95_DF9 * se;
    (m, sd, ci_hw)
}

fn print_overflow_telemetry(name: &str, num_nodes: usize, num_edges: usize, v2: &V2SlimArenaGraph) {
    let nodes_with_overflow = v2.nodes.iter().filter(|n| n.count > 3).count();
    let p_o = nodes_with_overflow as f64 / num_nodes as f64;
    let overflow_u32_len = v2.overflow.len();
    let overflow_mb = (overflow_u32_len * 4) as f64 / 1e6;
    let overflow_edge_pct = (overflow_u32_len as f64 / num_edges as f64) * 100.0;

    let base_node_mb = (num_nodes * 16) as f64 / 1e6;
    let total_v2_mb = base_node_mb + overflow_mb;
    let csr_mb = ((num_nodes + 1 + num_edges) * 4) as f64 / 1e6;
    let ratio_vs_csr = total_v2_mb / csr_mb;

    println!("  [Telemetry: {}]", name);
    println!("    Nodes: {} | Edges: {}", num_nodes, num_edges);
    println!("    Nodes with degree > 3: {} (p_o = {:.4} or {:.2}%)", nodes_with_overflow, p_o, p_o * 100.0);
    println!("    Overflow array size: {} elements ({:.2} MB, {:.2}% of all edges)", overflow_u32_len, overflow_mb, overflow_edge_pct);
    println!("    Footprint: CSR = {:.2} MB | V2 = {:.2} MB (base {:.2} MB + overflow {:.2} MB) -> Ratio: {:.2}x CSR", csr_mb, total_v2_mb, base_node_mb, overflow_mb, ratio_vs_csr);
}

fn run_part_a() {
    println!("\n=========================================================================================");
    println!(" PART A: V0 (CSR) vs V2 (16B SLIM) vs V3 (16B INLINE VISITED)");
    println!(" Strict load-then-fetch_or; bit clearing confirmed OUTSIDE timed region");
    println!(" 10 Isolated Sessions (3 warmup, median of 10 runs per session), 95% Confidence Intervals");
    println!("=========================================================================================\n");

    let sizes = [500_000usize, 1_000_000usize];
    let thread_counts = [1usize, 4, 6];

    for &n in &sizes {
        let rmat = generate_rmat_graph(n, 4, RmatConfig::default(), 0xDEAD_BEEF ^ n as u64);
        let v0 = V0CsrGraph::from_nodes_and_overflow(&rmat.nodes, &rmat.overflow);
        let v2 = V2SlimArenaGraph::from_nodes_and_overflow(&rmat.nodes, &rmat.overflow);
        let v3 = V3InlineVisitedGraph::from_nodes_and_overflow(&rmat.nodes, &rmat.overflow);

        println!("-----------------------------------------------------------------------------------------");
        println!(" Graph Scale N = {n} (Edges = {})", v0.num_edges);
        print_overflow_telemetry(&format!("R-MAT N={n}"), n, v0.num_edges, &v2);
        println!("-----------------------------------------------------------------------------------------");

        for &threads in &thread_counts {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();

            let mut single_scratch = SharedSingleScratch::new(n);
            let mut par_scratch = SharedParallelScratch::new(n);

            let cfg = TraversalConfig {
                num_threads: threads,
                test_before_atomic: false,
                prefetch: PrefetchMode::None,
            };

            let mut session_v0_meds = Vec::new();
            let mut session_v2_meds = Vec::new();
            let mut session_v3_meds = Vec::new();
            let mut session_speedup_v2 = Vec::new();
            let mut session_speedup_v3 = Vec::new();

            for session in 0..SESSIONS {
                let mut runs_v0 = Vec::new();
                let mut runs_v2 = Vec::new();
                let mut runs_v3 = Vec::new();

                for r in 0..(WARMUP + RUNS) {
                    let order = (r + session) % 3;

                    // Helper to execute single timed run with reset OUTSIDE timing window
                    let mut run_one = |var_idx: usize| -> f64 {
                        match var_idx {
                            0 => {
                                if threads == 1 {
                                    prepare_scratch_single(&v0, &mut single_scratch);
                                    let t = Instant::now();
                                    traverse_bfs_single_prepared(&v0, black_box(0), &cfg, &mut single_scratch);
                                    t.elapsed().as_secs_f64() * 1e3
                                } else {
                                    prepare_scratch_parallel_for_threads(&v0, &mut par_scratch, threads);
                                    let t = Instant::now();
                                    traverse_bfs_parallel_prepared(&v0, black_box(0), &pool, &cfg, &mut par_scratch);
                                    t.elapsed().as_secs_f64() * 1e3
                                }
                            }
                            1 => {
                                if threads == 1 {
                                    prepare_scratch_single(&v2, &mut single_scratch);
                                    let t = Instant::now();
                                    traverse_bfs_single_prepared(&v2, black_box(0), &cfg, &mut single_scratch);
                                    t.elapsed().as_secs_f64() * 1e3
                                } else {
                                    prepare_scratch_parallel_for_threads(&v2, &mut par_scratch, threads);
                                    let t = Instant::now();
                                    traverse_bfs_parallel_prepared(&v2, black_box(0), &pool, &cfg, &mut par_scratch);
                                    t.elapsed().as_secs_f64() * 1e3
                                }
                            }
                            2 => {
                                if threads == 1 {
                                    // Bit clearing confirmed strictly outside timed region
                                    prepare_scratch_single(&v3, &mut single_scratch);
                                    let t = Instant::now();
                                    traverse_bfs_single_prepared(&v3, black_box(0), &cfg, &mut single_scratch);
                                    t.elapsed().as_secs_f64() * 1e3
                                } else {
                                    // Bit clearing confirmed strictly outside timed region
                                    prepare_scratch_parallel_for_threads(&v3, &mut par_scratch, threads);
                                    let t = Instant::now();
                                    traverse_bfs_parallel_prepared(&v3, black_box(0), &pool, &cfg, &mut par_scratch);
                                    t.elapsed().as_secs_f64() * 1e3
                                }
                            }
                            _ => unreachable!(),
                        }
                    };

                    let idxs = [order, (order + 1) % 3, (order + 2) % 3];
                    let mut times = [0.0; 3];
                    for &idx in &idxs {
                        times[idx] = run_one(idx);
                    }

                    if r >= WARMUP {
                        runs_v0.push(times[0]);
                        runs_v2.push(times[1]);
                        runs_v3.push(times[2]);
                    }
                }

                let med_v0 = median(&mut runs_v0);
                let med_v2 = median(&mut runs_v2);
                let med_v3 = median(&mut runs_v3);

                session_v0_meds.push(med_v0);
                session_v2_meds.push(med_v2);
                session_v3_meds.push(med_v3);

                // Speedups computed strictly within the same interleaved session
                session_speedup_v2.push(med_v0 / med_v2);
                session_speedup_v3.push(med_v0 / med_v3);
            }

            let (m_v0, sd_v0, ci_v0) = stats_95_ci(&session_v0_meds);
            let (m_v2, sd_v2, ci_v2) = stats_95_ci(&session_v2_meds);
            let (m_v3, sd_v3, ci_v3) = stats_95_ci(&session_v3_meds);

            let (m_sp_v2, _, ci_sp_v2) = stats_95_ci(&session_speedup_v2);
            let (m_sp_v3, _, ci_sp_v3) = stats_95_ci(&session_speedup_v3);

            let mteps = |ms: f64| (v0.num_edges as f64) / ms / 1e3;

            println!(" Thread {:1}T:", threads);
            println!(
                "   V0 (CSR) : {:7.3} +/- {:5.3} ms (95% CI: [{:.3}, {:.3}], SD: {:4.2}%) | {:5.1} MTEPS",
                m_v0, ci_v0, m_v0 - ci_v0, m_v0 + ci_v0, (sd_v0 / m_v0) * 100.0, mteps(m_v0)
            );
            println!(
                "   V2 (Slim): {:7.3} +/- {:5.3} ms (95% CI: [{:.3}, {:.3}], SD: {:4.2}%) | {:5.1} MTEPS | Within-Session Speedup: {:.3}x +/- {:.3}x ({})",
                m_v2, ci_v2, m_v2 - ci_v2, m_v2 + ci_v2, (sd_v2 / m_v2) * 100.0, mteps(m_v2),
                m_sp_v2, ci_sp_v2,
                if m_sp_v2 >= 1.10 && (m_sp_v2 - ci_sp_v2) >= 1.10 { "WIN >=10%" } else { "Not Supported" }
            );
            println!(
                "   V3 (Inl) : {:7.3} +/- {:5.3} ms (95% CI: [{:.3}, {:.3}], SD: {:4.2}%) | {:5.1} MTEPS | Within-Session Speedup: {:.3}x +/- {:.3}x ({})",
                m_v3, ci_v3, m_v3 - ci_v3, m_v3 + ci_v3, (sd_v3 / m_v3) * 100.0, mteps(m_v3),
                m_sp_v3, ci_sp_v3,
                if m_sp_v3 >= 1.10 && (m_sp_v3 - ci_sp_v3) >= 1.10 { "WIN >=10%" } else { "Not Supported" }
            );
            println!();
        }
    }
}

fn run_part_b() {
    println!("\n=========================================================================================");
    println!(" PART B: V0 vs V2 SCALING AT 4M AND 8M NODES");
    println!(" Orders: Original, DegreeSort | Prefetch: None, k=8, k=16 | Thread Configurations: 1T, 4T, 6T");
    println!(" 10 Isolated Sessions (3 warmup, median of 10 runs per session), 95% Confidence Intervals");
    println!("=========================================================================================\n");

    let scales = [4_000_000usize, 8_000_000usize];
    let orders = [VertexOrder::Original, VertexOrder::DegreeSort];
    let prefetches = [
        ("None", PrefetchMode::None),
        ("k=8", PrefetchMode::KAhead(8)),
        ("k=16", PrefetchMode::KAhead(16)),
    ];
    let thread_counts = [1usize, 4, 6];

    for &n in &scales {
        println!("#########################################################################################");
        println!(" GRAPH SCALE N = {n} (Generating R-MAT edge factor 4, total edges = {})...", n * 4);
        let gen_start = Instant::now();
        let rmat = generate_rmat_graph(n, 4, RmatConfig::default(), 0x1337_C0DE ^ n as u64);
        println!("   Generated in {:.2} s.", gen_start.elapsed().as_secs_f64());

        // Extract base adjacency
        let num_edges = rmat.total_edges;
        let mut base_adj = vec![Vec::new(); n];
        for (i, node) in rmat.nodes.iter().enumerate() {
            let deg = node.outbound_count as usize;
            if deg <= 3 {
                base_adj[i].extend_from_slice(&node.outbound_links[..deg]);
            } else {
                base_adj[i].extend_from_slice(&node.outbound_links[..3]);
                let off = node.outbound_links[3] as usize;
                let rem = deg - 3;
                if off + rem <= rmat.overflow.len() {
                    base_adj[i].extend_from_slice(&rmat.overflow[off..off + rem]);
                }
            }
        }

        // Shared scratch allocations outside timed region
        let mut single_scratch = SharedSingleScratch::new(n);
        let mut par_scratch = SharedParallelScratch::new(n);

        for &order in &orders {
            let (adj, reorder_duration) = match order {
                VertexOrder::Original => (base_adj.clone(), std::time::Duration::ZERO),
                VertexOrder::DegreeSort => {
                    let perm = compute_permutation(&base_adj, VertexOrder::DegreeSort, 0);
                    apply_permutation(&base_adj, &perm)
                }
                _ => unreachable!(),
            };

            let v0 = V0CsrGraph::from_adjacency(&adj);
            let v2 = V2SlimArenaGraph::from_adjacency(&adj);

            println!("-----------------------------------------------------------------------------------------");
            println!(" Scale N = {n} | Order: {:?} (Reorder Build Time: {:.3} ms)", order, reorder_duration.as_secs_f64() * 1e3);
            print_overflow_telemetry(&format!("N={n} {:?}", order), n, num_edges, &v2);
            println!("-----------------------------------------------------------------------------------------");

            for &(pf_label, pf_mode) in &prefetches {
                println!(" >>> Prefetch Mode: {} <<<", pf_label);

                for &threads in &thread_counts {
                    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();

                    let cfg = TraversalConfig {
                        num_threads: threads,
                        test_before_atomic: false,
                        prefetch: pf_mode,
                    };

                    let mut session_v0_meds = Vec::new();
                    let mut session_v2_meds = Vec::new();
                    let mut session_speedups = Vec::new();

                    for session in 0..SESSIONS {
                        let mut runs_v0 = Vec::new();
                        let mut runs_v2 = Vec::new();

                        for r in 0..(WARMUP + RUNS) {
                            let v0_first = (r + session) % 2 == 0;
                            let (t_v0, t_v2);

                            if v0_first {
                                if threads == 1 {
                                    prepare_scratch_single(&v0, &mut single_scratch);
                                    let t = Instant::now();
                                    traverse_bfs_single_prepared(&v0, black_box(0), &cfg, &mut single_scratch);
                                    t_v0 = t.elapsed().as_secs_f64() * 1e3;

                                    prepare_scratch_single(&v2, &mut single_scratch);
                                    let t = Instant::now();
                                    traverse_bfs_single_prepared(&v2, black_box(0), &cfg, &mut single_scratch);
                                    t_v2 = t.elapsed().as_secs_f64() * 1e3;
                                } else {
                                    prepare_scratch_parallel_for_threads(&v0, &mut par_scratch, threads);
                                    let t = Instant::now();
                                    traverse_bfs_parallel_prepared(&v0, black_box(0), &pool, &cfg, &mut par_scratch);
                                    t_v0 = t.elapsed().as_secs_f64() * 1e3;

                                    prepare_scratch_parallel_for_threads(&v2, &mut par_scratch, threads);
                                    let t = Instant::now();
                                    traverse_bfs_parallel_prepared(&v2, black_box(0), &pool, &cfg, &mut par_scratch);
                                    t_v2 = t.elapsed().as_secs_f64() * 1e3;
                                }
                            } else {
                                if threads == 1 {
                                    prepare_scratch_single(&v2, &mut single_scratch);
                                    let t = Instant::now();
                                    traverse_bfs_single_prepared(&v2, black_box(0), &cfg, &mut single_scratch);
                                    t_v2 = t.elapsed().as_secs_f64() * 1e3;

                                    prepare_scratch_single(&v0, &mut single_scratch);
                                    let t = Instant::now();
                                    traverse_bfs_single_prepared(&v0, black_box(0), &cfg, &mut single_scratch);
                                    t_v0 = t.elapsed().as_secs_f64() * 1e3;
                                } else {
                                    prepare_scratch_parallel_for_threads(&v2, &mut par_scratch, threads);
                                    let t = Instant::now();
                                    traverse_bfs_parallel_prepared(&v2, black_box(0), &pool, &cfg, &mut par_scratch);
                                    t_v2 = t.elapsed().as_secs_f64() * 1e3;

                                    prepare_scratch_parallel_for_threads(&v0, &mut par_scratch, threads);
                                    let t = Instant::now();
                                    traverse_bfs_parallel_prepared(&v0, black_box(0), &pool, &cfg, &mut par_scratch);
                                    t_v0 = t.elapsed().as_secs_f64() * 1e3;
                                }
                            }

                            if r >= WARMUP {
                                runs_v0.push(t_v0);
                                runs_v2.push(t_v2);
                            }
                        }

                        let med_v0 = median(&mut runs_v0);
                        let med_v2 = median(&mut runs_v2);

                        session_v0_meds.push(med_v0);
                        session_v2_meds.push(med_v2);
                        session_speedups.push(med_v0 / med_v2);
                    }

                    let (m_v0, sd_v0, ci_v0) = stats_95_ci(&session_v0_meds);
                    let (m_v2, sd_v2, ci_v2) = stats_95_ci(&session_v2_meds);
                    let (m_sp, _, ci_sp) = stats_95_ci(&session_speedups);

                    let mteps = |ms: f64| (num_edges as f64) / ms / 1e3;

                    println!(
                        "   Thread {:1}T | V0: {:7.3} +/- {:5.3} ms (CI: [{:.2}, {:.2}], SD: {:4.2}%) {:5.1} MTEPS | V2: {:7.3} +/- {:5.3} ms (CI: [{:.2}, {:.2}], SD: {:4.2}%) {:5.1} MTEPS | Sp: {:.3}x +/- {:.3}x ({})",
                        threads,
                        m_v0, ci_v0, m_v0 - ci_v0, m_v0 + ci_v0, (sd_v0 / m_v0) * 100.0, mteps(m_v0),
                        m_v2, ci_v2, m_v2 - ci_v2, m_v2 + ci_v2, (sd_v2 / m_v2) * 100.0, mteps(m_v2),
                        m_sp, ci_sp,
                        if m_sp >= 1.10 && (m_sp - ci_sp) >= 1.10 { "WIN >=10%" } else { "Not Supported" }
                    );
                }
            }
            println!();
        }
    }
}

fn main() {
    run_part_a();
    run_part_b();
    println!("=========================================================================================");
    println!(" Full rigorous multi-session study complete.");
    println!("=========================================================================================");
}
