//! Fair two-stage prefetch comparison for V0 CSR and V2 slim arena.
//! Run with: cargo run --release --example stage2_two_stage_prefetch

#[path = "support/bench_support.rs"]
mod bench_support;

use std::hint::black_box;
use std::io;
use std::time::Instant;

use bench_support::{CsvLog, MachineState};
use state_search_engine::graph::{generate_rmat_graph, RmatConfig};
use state_search_engine::variants::bfs_driver::{
    prepare_scratch_parallel_for_threads, prepare_scratch_single, traverse_and_record_levels,
    traverse_bfs_parallel_prepared, traverse_bfs_single_prepared,
};
use state_search_engine::variants::traits::{GraphLayout, PrefetchMode, TraversalConfig};
use state_search_engine::variants::v0_csr::V0CsrGraph;
use state_search_engine::variants::v2_slim_arena::V2SlimArenaGraph;
use state_search_engine::variants::visited::{SharedParallelScratch, SharedSingleScratch};

const SESSIONS: usize = 10;
const WARMUPS: usize = 3;
const TIMED_RUNS: usize = 10;
const THREAD_COUNTS: [usize; 3] = [1, 4, 6];
const VARIANT_NAMES: [&str; 2] = ["V0", "V2"];
const PREFETCHES: [(&str, PrefetchMode); 4] = [
    ("None", PrefetchMode::None),
    ("TwoStage4", PrefetchMode::TwoStage(4)),
    ("TwoStage8", PrefetchMode::TwoStage(8)),
    ("TwoStage16", PrefetchMode::TwoStage(16)),
];

enum GraphRef<'a> {
    V0(&'a V0CsrGraph),
    V2(&'a V2SlimArenaGraph),
}

impl GraphLayout for GraphRef<'_> {
    fn num_nodes(&self) -> usize {
        match self {
            Self::V0(graph) => graph.num_nodes(),
            Self::V2(graph) => graph.num_nodes(),
        }
    }

    fn num_edges(&self) -> usize {
        match self {
            Self::V0(graph) => graph.num_edges(),
            Self::V2(graph) => graph.num_edges(),
        }
    }

    fn for_each_neighbor<F: FnMut(u32)>(&self, node_id: u32, f: F) {
        match self {
            Self::V0(graph) => graph.for_each_neighbor(node_id, f),
            Self::V2(graph) => graph.for_each_neighbor(node_id, f),
        }
    }

    fn prefetch_stage1(&self, node_id: u32) {
        match self {
            Self::V0(graph) => graph.prefetch_stage1(node_id),
            Self::V2(graph) => graph.prefetch_stage1(node_id),
        }
    }

    fn prefetch_stage2(&self, node_id: u32) {
        match self {
            Self::V0(graph) => graph.prefetch_stage2(node_id),
            Self::V2(graph) => graph.prefetch_stage2(node_id),
        }
    }
}

fn measure(
    graph: &GraphRef<'_>,
    start: u32,
    threads: usize,
    pool: &rayon::ThreadPool,
    config: &TraversalConfig,
    single_scratch: &mut SharedSingleScratch,
    parallel_scratch: &mut SharedParallelScratch,
) -> (f64, u64, usize) {
    if threads == 1 {
        prepare_scratch_single(graph, single_scratch);
        let started = Instant::now();
        let result = traverse_bfs_single_prepared(graph, black_box(start), config, single_scratch);
        let elapsed = started.elapsed().as_secs_f64() * 1e3;
        black_box(result);
        (elapsed, result.0, result.1)
    } else {
        prepare_scratch_parallel_for_threads(graph, parallel_scratch, threads);
        let started = Instant::now();
        let result = traverse_bfs_parallel_prepared(
            graph,
            black_box(start),
            pool,
            config,
            parallel_scratch,
        );
        let elapsed = started.elapsed().as_secs_f64() * 1e3;
        black_box(result);
        (elapsed, result.0, result.1)
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn run() -> io::Result<()> {
    let machine = MachineState::capture(&THREAD_COUNTS);
    machine.print_header();
    machine.require_ac_power()?;

    let mut log = CsvLog::create(
        "stage2_two_stage_prefetch",
        &machine,
        "session,run,warmup,n,threads,prefetch,k,variant,elapsed_ms,edges,reached",
    )?;

    for &num_nodes in &[4_000_000usize, 8_000_000] {
        let rmat = generate_rmat_graph(
            num_nodes,
            4,
            RmatConfig::default(),
            0x1337_C0DE ^ num_nodes as u64,
        );
        let v0 = V0CsrGraph::from_nodes_and_overflow(&rmat.nodes, &rmat.overflow);
        let v2 = V2SlimArenaGraph::from_nodes_and_overflow(&rmat.nodes, &rmat.overflow);
        let graphs = [GraphRef::V0(&v0), GraphRef::V2(&v2)];
        let baseline = traverse_and_record_levels(&v0, 0);
        let v2_levels = traverse_and_record_levels(&v2, 0);
        assert_eq!(baseline, v2_levels, "V2 differs from V0 before timing at N={num_nodes}");
        println!("Graph N={num_nodes} | edges={} | order=Original", v0.num_edges);

        for &threads in &THREAD_COUNTS {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool creation should succeed");
            let mut single_scratch = SharedSingleScratch::new(num_nodes);
            let mut parallel_scratch = SharedParallelScratch::new(num_nodes);

            for &(prefetch_name, prefetch) in &PREFETCHES {
                let k = match prefetch {
                    PrefetchMode::TwoStage(distance) => distance,
                    _ => 0,
                };
                let config = TraversalConfig {
                    num_threads: threads,
                    test_before_atomic: false,
                    prefetch,
                };
                let mut session_v0 = Vec::with_capacity(SESSIONS);
                let mut session_v2 = Vec::with_capacity(SESSIONS);

                for session in 0..SESSIONS {
                    let mut runs: [Vec<f64>; 2] =
                        std::array::from_fn(|_| Vec::with_capacity(TIMED_RUNS));
                    for run_index in 0..(WARMUPS + TIMED_RUNS) {
                        let first = (session + run_index) % 2;
                        let mut results = [(0.0, 0u64, 0usize); 2];
                        for position in 0..2 {
                            let variant = (first + position) % 2;
                            results[variant] = measure(
                                &graphs[variant],
                                0,
                                threads,
                                &pool,
                                &config,
                                &mut single_scratch,
                                &mut parallel_scratch,
                            );
                            let (elapsed_ms, edges, reached) = results[variant];
                            log.record(&format!(
                                "{session},{run_index},{},{num_nodes},{threads},{prefetch_name},{k},{},{elapsed_ms:.6},{edges},{reached}",
                                run_index < WARMUPS,
                                VARIANT_NAMES[variant]
                            ));
                        }
                        assert_eq!(results[0].1, results[1].1, "edge count mismatch for {prefetch_name}");
                        assert_eq!(results[0].2, results[1].2, "reached count mismatch for {prefetch_name}");
                        if run_index >= WARMUPS {
                            runs[0].push(results[0].0);
                            runs[1].push(results[1].0);
                        }
                    }
                    session_v0.push(median(&mut runs[0]));
                    session_v2.push(median(&mut runs[1]));
                }

                println!(
                    "  {threads}T {prefetch_name:10} | V0 mean session median {:.3} ms | V2 {:.3} ms",
                    session_v0.iter().sum::<f64>() / SESSIONS as f64,
                    session_v2.iter().sum::<f64>() / SESSIONS as f64
                );
            }
        }
    }

    let path = log.finish()?;
    println!("Raw CSV: {}", path.display());
    Ok(())
}

fn main() -> io::Result<()> {
    run()
}