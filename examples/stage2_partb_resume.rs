//! Stage 2 Part B: V0 vs V2 SCALING – DegreeSort at 4M and both orders at 8M
//! Run with:
//!   cargo run --release --example stage2_partb_resume

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
use state_search_engine::variants::visited::{SharedParallelScratch, SharedSingleScratch};

const SESSIONS: usize = 10;
const WARMUP: usize = 3;
const RUNS: usize = 10;
const T_CRIT_95_DF9: f64 = 2.262;

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn sample_sd(v: &[f64], mean_val: f64) -> f64 {
    if v.len() <= 1 { return 0.0; }
    let var = v.iter().map(|&x| (x - mean_val).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
    var.sqrt()
}

fn stats_95_ci(v: &[f64]) -> (f64, f64, f64) {
    let m = mean(v);
    let sd = sample_sd(v, m);
    let se = sd / (v.len() as f64).sqrt();
    (m, sd, T_CRIT_95_DF9 * se)
}

fn print_overflow_telemetry(n: usize, num_edges: usize, v2: &V2SlimArenaGraph, order_label: &str) {
    let nodes_overflow = v2.nodes.iter().filter(|nd| nd.count > 3).count();
    let p_o = nodes_overflow as f64 / n as f64;
    let ov_len = v2.overflow.len();
    let ov_mb = (ov_len * 4) as f64 / 1e6;
    let ov_edge_pct = (ov_len as f64 / num_edges as f64) * 100.0;
    let base_mb = (n * 16) as f64 / 1e6;
    let total_v2_mb = base_mb + ov_mb;
    let csr_mb = ((n + 1 + num_edges) * 4) as f64 / 1e6;
    let ratio = total_v2_mb / csr_mb;
    println!("  [Telemetry: N={n} {order_label}]");
    println!("    p_o = {:.4} ({:.2}% nodes have degree > 3)", p_o, p_o * 100.0);
    println!("    Overflow: {ov_len} elements ({ov_mb:.2} MB, {ov_edge_pct:.2}% of all edges)");
    println!("    CSR = {csr_mb:.2} MB | V2 = {total_v2_mb:.2} MB (base {base_mb:.2} + overflow {ov_mb:.2}) -> {ratio:.2}x CSR");
}

fn bench_v0_v2(
    n: usize,
    num_edges: usize,
    v0: &V0CsrGraph,
    v2: &V2SlimArenaGraph,
    prefetch_label: &str,
    pf: PrefetchMode,
    threads: usize,
    pool: &rayon::ThreadPool,
    single_scratch: &mut SharedSingleScratch,
    par_scratch: &mut SharedParallelScratch,
) {
    let cfg = TraversalConfig { num_threads: threads, test_before_atomic: false, prefetch: pf };

    let mut v0_meds = Vec::new();
    let mut v2_meds = Vec::new();
    let mut speedups = Vec::new();

    for session in 0..SESSIONS {
        let mut runs_v0 = Vec::new();
        let mut runs_v2 = Vec::new();

        for r in 0..(WARMUP + RUNS) {
            let v0_first = (r + session) % 2 == 0;
            let (t_v0, t_v2);
            if v0_first {
                if threads == 1 {
                    prepare_scratch_single(v0, single_scratch);
                    let t = Instant::now();
                    traverse_bfs_single_prepared(v0, black_box(0), &cfg, single_scratch);
                    t_v0 = t.elapsed().as_secs_f64() * 1e3;
                    prepare_scratch_single(v2, single_scratch);
                    let t = Instant::now();
                    traverse_bfs_single_prepared(v2, black_box(0), &cfg, single_scratch);
                    t_v2 = t.elapsed().as_secs_f64() * 1e3;
                } else {
                    prepare_scratch_parallel_for_threads(v0, par_scratch, threads);
                    let t = Instant::now();
                    traverse_bfs_parallel_prepared(v0, black_box(0), pool, &cfg, par_scratch);
                    t_v0 = t.elapsed().as_secs_f64() * 1e3;
                    prepare_scratch_parallel_for_threads(v2, par_scratch, threads);
                    let t = Instant::now();
                    traverse_bfs_parallel_prepared(v2, black_box(0), pool, &cfg, par_scratch);
                    t_v2 = t.elapsed().as_secs_f64() * 1e3;
                }
            } else {
                if threads == 1 {
                    prepare_scratch_single(v2, single_scratch);
                    let t = Instant::now();
                    traverse_bfs_single_prepared(v2, black_box(0), &cfg, single_scratch);
                    t_v2 = t.elapsed().as_secs_f64() * 1e3;
                    prepare_scratch_single(v0, single_scratch);
                    let t = Instant::now();
                    traverse_bfs_single_prepared(v0, black_box(0), &cfg, single_scratch);
                    t_v0 = t.elapsed().as_secs_f64() * 1e3;
                } else {
                    prepare_scratch_parallel_for_threads(v2, par_scratch, threads);
                    let t = Instant::now();
                    traverse_bfs_parallel_prepared(v2, black_box(0), pool, &cfg, par_scratch);
                    t_v2 = t.elapsed().as_secs_f64() * 1e3;
                    prepare_scratch_parallel_for_threads(v0, par_scratch, threads);
                    let t = Instant::now();
                    traverse_bfs_parallel_prepared(v0, black_box(0), pool, &cfg, par_scratch);
                    t_v0 = t.elapsed().as_secs_f64() * 1e3;
                }
            }
            if r >= WARMUP {
                runs_v0.push(t_v0);
                runs_v2.push(t_v2);
            }
        }

        let m0 = median(&mut runs_v0);
        let m2 = median(&mut runs_v2);
        v0_meds.push(m0);
        v2_meds.push(m2);
        speedups.push(m0 / m2);
    }

    let (m0, sd0, ci0) = stats_95_ci(&v0_meds);
    let (m2, sd2, ci2) = stats_95_ci(&v2_meds);
    let (msp, _, cisp) = stats_95_ci(&speedups);
    let mteps = |ms: f64| num_edges as f64 / ms / 1e3;
    let win = if msp >= 1.10 && (msp - cisp) >= 1.10 { "WIN >=10%" } else { "Not Supported" };

    println!(
        "   {threads}T pf={prefetch_label:4} | V0: {:7.3} +/- {:5.3} ms (CI:[{:.2},{:.2}] SD:{:.2}%) {:5.1}MTEPS | V2: {:7.3} +/- {:5.3} ms (CI:[{:.2},{:.2}] SD:{:.2}%) {:5.1}MTEPS | Sp: {:.3}x +/- {:.3}x ({})",
        m0, ci0, m0-ci0, m0+ci0, (sd0/m0)*100.0, mteps(m0),
        m2, ci2, m2-ci2, m2+ci2, (sd2/m2)*100.0, mteps(m2),
        msp, cisp, win
    );
}

fn run_scale(n: usize) {
    println!("\n#########################################################################################");
    println!(" GRAPH SCALE N = {n} (R-MAT, edge factor 4, total edges = {})", n * 4);
    let gen_start = Instant::now();
    let rmat = generate_rmat_graph(n, 4, RmatConfig::default(), 0x1337_C0DE ^ n as u64);
    let num_edges = rmat.total_edges;
    println!("   Generated in {:.2} s.", gen_start.elapsed().as_secs_f64());

    // Extract base adjacency from nodes+overflow
    let mut base_adj: Vec<Vec<u32>> = vec![Vec::new(); n];
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

    let mut single_scratch = SharedSingleScratch::new(n);
    let mut par_scratch = SharedParallelScratch::new(n);

    let prefetches: &[(&str, PrefetchMode)] = &[
        ("None", PrefetchMode::None),
        ("k=8",  PrefetchMode::KAhead(8)),
        ("k=16", PrefetchMode::KAhead(16)),
    ];
    let thread_counts = [1usize, 4, 6];

    for &order in &[VertexOrder::Original, VertexOrder::DegreeSort] {
        let (adj, reorder_dur) = match order {
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
        println!(" Order: {order:?} | Reorder build time: {:.1} ms", reorder_dur.as_secs_f64() * 1e3);
        print_overflow_telemetry(n, num_edges, &v2, &format!("{order:?}"));
        println!("-----------------------------------------------------------------------------------------");

        for &threads in &thread_counts {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            for &(pf_label, pf_mode) in prefetches {
                bench_v0_v2(n, num_edges, &v0, &v2, pf_label, pf_mode, threads, &pool,
                            &mut single_scratch, &mut par_scratch);
            }
        }
        println!();
    }
}

fn main() {
    println!("=========================================================================================");
    println!(" PART B (RESUMED): V0 vs V2 DegreeSort @ 4M  +  V0 vs V2 Original/DegreeSort @ 8M");
    println!("=========================================================================================");

    // 4M DegreeSort only (Original was already printed; re-running both for consistency)
    run_scale(4_000_000);
    run_scale(8_000_000);

    println!("\n=========================================================================================");
    println!(" Part B complete.");
    println!("=========================================================================================");
}
