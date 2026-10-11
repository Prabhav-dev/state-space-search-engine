//! Interleaved V0/V2/V3 comparison with the existing NativeCsrGraph control.
//! Run with: cargo run --release --example stage1_control_study

#[path = "support/bench_support.rs"]
mod bench_support;
#[allow(dead_code)]
#[path = "external_baseline_comparison.rs"]
mod existing_baseline;

use std::hint::black_box;
use std::io;
use std::time::Instant;

use bench_support::{CsvLog, MachineState};
use existing_baseline::NativeCsrGraph;
use state_search_engine::graph::{generate_rmat_graph, RmatConfig};
use state_search_engine::variants::bfs_driver::{
    prepare_scratch_parallel_for_threads, prepare_scratch_single, traverse_and_record_levels,
    traverse_bfs_parallel_prepared, traverse_bfs_single_prepared,
};
use state_search_engine::variants::traits::{prefetch_ptr, GraphLayout, PrefetchMode, TraversalConfig};
use state_search_engine::variants::v0_csr::V0CsrGraph;
use state_search_engine::variants::v2_slim_arena::V2SlimArenaGraph;
use state_search_engine::variants::v3_inline_visited::V3InlineVisitedGraph;
use state_search_engine::variants::visited::{SharedParallelScratch, SharedSingleScratch};

const SESSIONS: usize = 10;
const WARMUPS: usize = 3;
const TIMED_RUNS: usize = 10;
const THREAD_COUNTS: [usize; 3] = [1, 4, 6];
const T_CRIT_95_DF9: f64 = 2.262;
const VARIANTS: [&str; 4] = ["ExistingNativeCsr", "V0", "V2", "V3"];

enum GraphRef<'a> {
    Existing(&'a NativeCsrGraph),
    V0(&'a V0CsrGraph),
    V2(&'a V2SlimArenaGraph),
    V3(&'a V3InlineVisitedGraph),
}

impl GraphLayout for GraphRef<'_> {
    fn num_nodes(&self) -> usize {
        match self {
            Self::Existing(graph) => graph.num_nodes(),
            Self::V0(graph) => graph.num_nodes(),
            Self::V2(graph) => graph.num_nodes(),
            Self::V3(graph) => graph.num_nodes(),
        }
    }

    fn num_edges(&self) -> usize {
        match self {
            Self::Existing(graph) => graph.num_edges(),
            Self::V0(graph) => graph.num_edges(),
            Self::V2(graph) => graph.num_edges(),
            Self::V3(graph) => graph.num_edges(),
        }
    }

    fn for_each_neighbor<F: FnMut(u32)>(&self, node_id: u32, f: F) {
        match self {
            Self::Existing(graph) => graph.for_each_neighbor(node_id, f),
            Self::V0(graph) => graph.for_each_neighbor(node_id, f),
            Self::V2(graph) => graph.for_each_neighbor(node_id, f),
            Self::V3(graph) => graph.for_each_neighbor(node_id, f),
        }
    }

    fn prefetch_stage1(&self, node_id: u32) {
        match self {
            Self::Existing(graph) => graph.prefetch_stage1(node_id),
            Self::V0(graph) => graph.prefetch_stage1(node_id),
            Self::V2(graph) => graph.prefetch_stage1(node_id),
            Self::V3(graph) => graph.prefetch_stage1(node_id),
        }
    }

    fn prefetch_stage2(&self, node_id: u32) {
        match self {
            Self::Existing(graph) => graph.prefetch_stage2(node_id),
            Self::V0(graph) => graph.prefetch_stage2(node_id),
            Self::V2(graph) => graph.prefetch_stage2(node_id),
            Self::V3(graph) => graph.prefetch_stage2(node_id),
        }
    }

    fn has_internal_visited(&self) -> bool {
        matches!(self, Self::V3(graph) if graph.has_internal_visited())
    }

    fn internal_test_and_set(&self, node_id: u32) -> bool {
        match self {
            Self::V3(graph) => graph.internal_test_and_set(node_id),
            _ => false,
        }
    }

    fn internal_test_and_set_parallel(&self, node_id: u32, test_before_atomic: bool) -> bool {
        match self {
            Self::V3(graph) => graph.internal_test_and_set_parallel(node_id, test_before_atomic),
            _ => false,
        }
    }

    fn internal_reset_visited(&self) {
        if let Self::V3(graph) = self {
            graph.internal_reset_visited();
        }
    }

    fn internal_is_visited(&self, node_id: u32) -> bool {
        match self {
            Self::V3(graph) => graph.internal_is_visited(node_id),
            _ => false,
        }
    }
}

impl GraphLayout for NativeCsrGraph {
    fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    fn num_edges(&self) -> usize {
        self.num_edges
    }

    #[inline(always)]
    fn for_each_neighbor<F: FnMut(u32)>(&self, node_id: u32, mut f: F) {
        let node = node_id as usize;
        if node < self.num_nodes {
            let start = self.offsets[node] as usize;
            let end = self.offsets[node + 1] as usize;
            for &target in &self.targets[start..end] {
                f(target);
            }
        }
    }

    #[inline(always)]
    fn prefetch_stage1(&self, node_id: u32) {
        let node = node_id as usize;
        if node < self.num_nodes {
            prefetch_ptr(&self.offsets[node]);
        }
    }

    #[inline(always)]
    fn prefetch_stage2(&self, node_id: u32) {
        let node = node_id as usize;
        if node < self.num_nodes {
            let start = self.offsets[node] as usize;
            if start < self.targets.len() {
                prefetch_ptr(&self.targets[start]);
            }
        }
    }
}

fn measure<G: GraphLayout>(
    graph: &G,
    threads: usize,
    pool: &rayon::ThreadPool,
    config: &TraversalConfig,
    single_scratch: &mut SharedSingleScratch,
    parallel_scratch: &mut SharedParallelScratch,
) -> (f64, u64, usize) {
    if threads == 1 {
        prepare_scratch_single(graph, single_scratch);
        let start = Instant::now();
        let result = traverse_bfs_single_prepared(graph, black_box(0), config, single_scratch);
        let elapsed = start.elapsed().as_secs_f64() * 1e3;
        black_box(result);
        (elapsed, result.0, result.1)
    } else {
        prepare_scratch_parallel_for_threads(graph, parallel_scratch, threads);
        let start = Instant::now();
        let result = traverse_bfs_parallel_prepared(
            graph,
            black_box(0),
            pool,
            config,
            parallel_scratch,
        );
        let elapsed = start.elapsed().as_secs_f64() * 1e3;
        black_box(result);
        (elapsed, result.0, result.1)
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|left, right| left.total_cmp(right));
    values[values.len() / 2]
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn sample_sd(values: &[f64], average: f64) -> f64 {
    let sum_sq = values.iter().map(|value| (value - average).powi(2)).sum::<f64>();
    (sum_sq / (values.len() - 1) as f64).sqrt()
}

fn ci_half_width(values: &[f64]) -> f64 {
    T_CRIT_95_DF9 * sample_sd(values, mean(values)) / (values.len() as f64).sqrt()
}

fn verify_graphs(
    native: &NativeCsrGraph,
    v0: &V0CsrGraph,
    v2: &V2SlimArenaGraph,
    v3: &V3InlineVisitedGraph,
) {
    let baseline = traverse_and_record_levels(v0, 0);
    assert_eq!(traverse_and_record_levels(native, 0), baseline, "NativeCsrGraph BFS result differs from V0");
    assert_eq!(traverse_and_record_levels(v2, 0), baseline, "V2 BFS result differs from V0");
    assert_eq!(traverse_and_record_levels(v3, 0), baseline, "V3 BFS result differs from V0");
}

fn run() -> io::Result<()> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let (sizes, threads_to_run) = if arguments.len() == 3 && arguments[0] == "--only-cell" {
        let num_nodes = arguments[1].parse::<usize>().map_err(|error| {
            io::Error::new(io::ErrorKind::InvalidInput, format!("invalid node count: {error}"))
        })?;
        let threads = arguments[2].parse::<usize>().map_err(|error| {
            io::Error::new(io::ErrorKind::InvalidInput, format!("invalid thread count: {error}"))
        })?;
        (vec![num_nodes], vec![threads])
    } else if arguments.is_empty() {
        (vec![500_000usize, 1_000_000], THREAD_COUNTS.to_vec())
    } else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: stage1_control_study [--only-cell <nodes> <threads>]",
        ));
    };

    let machine = MachineState::capture(&threads_to_run);
    machine.print_header();
    let mut log = CsvLog::create(
        "stage1_control",
        &machine,
        "session,run,warmup,n,threads,test_before_atomic,prefetch,variant,elapsed_ms,edges,reached,ac_battery,power_plan",
    )?;
    log.record_session_state("startup", &machine);
    log.flush()?;
    machine.require_ac_power()?;

    for &num_nodes in &sizes {
        let rmat = generate_rmat_graph(
            num_nodes,
            4,
            RmatConfig::default(),
            0xDEAD_BEEF ^ num_nodes as u64,
        );
        let native = NativeCsrGraph::from_nodes(&rmat.nodes, &rmat.overflow);
        let v0 = V0CsrGraph::from_nodes_and_overflow(&rmat.nodes, &rmat.overflow);
        let v2 = V2SlimArenaGraph::from_nodes_and_overflow(&rmat.nodes, &rmat.overflow);
        let v3 = V3InlineVisitedGraph::from_nodes_and_overflow(&rmat.nodes, &rmat.overflow);
        verify_graphs(&native, &v0, &v2, &v3);

        println!("Graph N={num_nodes} | edges={} | order=Original", v0.num_edges);
        for &threads in &threads_to_run {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool creation should succeed");
            let mut single_scratch = SharedSingleScratch::new(num_nodes);
            let mut parallel_scratch = SharedParallelScratch::new(num_nodes);
            let graphs = [
                GraphRef::Existing(&native),
                GraphRef::V0(&v0),
                GraphRef::V2(&v2),
                GraphRef::V3(&v3),
            ];
            for &test_before_atomic in &[false, true] {
                let config = TraversalConfig {
                    num_threads: threads,
                    test_before_atomic,
                    prefetch: PrefetchMode::None,
                };
                let mut session_medians: [Vec<f64>; 4] =
                    std::array::from_fn(|_| Vec::with_capacity(SESSIONS));
                let mut paired_v0_over_existing = Vec::with_capacity(SESSIONS);

                for session in 0..SESSIONS {
                    let session_machine = MachineState::capture_session(&[threads], session);
                    log.record_session_state(&session.to_string(), &session_machine);
                    log.flush()?;
                    session_machine.require_ac_power()?;
                    let mut session_runs: [Vec<f64>; 4] =
                        std::array::from_fn(|_| Vec::with_capacity(TIMED_RUNS));
                    for run_index in 0..(WARMUPS + TIMED_RUNS) {
                        let first_variant = (session + run_index) % VARIANTS.len();
                        let mut results = [(0.0, 0u64, 0usize); 4];
                        for position in 0..VARIANTS.len() {
                            let variant = (first_variant + position) % VARIANTS.len();
                            let (elapsed_ms, edges, reached) = measure(
                                &graphs[variant],
                                threads,
                                &pool,
                                &config,
                                &mut single_scratch,
                                &mut parallel_scratch,
                            );
                            results[variant] = (elapsed_ms, edges, reached);
                            log.record(&format!(
                                "{session},{run_index},{},{num_nodes},{threads},{},{},{},{elapsed_ms:.6},{edges},{reached},{}",
                                run_index < WARMUPS,
                                config.test_before_atomic,
                                "None",
                                VARIANTS[variant],
                                session_machine.csv_session_fields()
                            ));
                        }

                        for (variant, result) in results.iter().enumerate() {
                            assert_eq!(result.1, results[1].1, "{} edge count differs from V0", VARIANTS[variant]);
                            assert_eq!(result.2, results[1].2, "{} reached count differs from V0", VARIANTS[variant]);
                            if run_index >= WARMUPS {
                                session_runs[variant].push(result.0);
                            }
                        }
                    }

                    for variant in 0..VARIANTS.len() {
                        session_medians[variant].push(median(&mut session_runs[variant]));
                    }
                    paired_v0_over_existing.push(
                        session_medians[0][session] / session_medians[1][session],
                    );
                    log.flush()?;
                }

                let overhead_percent: Vec<f64> = paired_v0_over_existing
                    .iter()
                    .map(|ratio| (ratio - 1.0) * 100.0)
                    .collect();
                println!(
                    "  {threads}T | V0 vs existing overhead: {:+.2}% +/- {:.2}% (95% CI) | test_before_atomic={test_before_atomic}",
                    mean(&overhead_percent),
                    ci_half_width(&overhead_percent),
                );
                for variant in 0..VARIANTS.len() {
                    let values = &session_medians[variant];
                    let average = mean(values);
                    println!(
                        "    {:18} {:8.3} ms +/- {:6.3} ms (95% CI), sample SD {:5.2}%",
                        VARIANTS[variant],
                        average,
                        ci_half_width(values),
                        sample_sd(values, average) / average * 100.0
                    );
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn existing_native_csr_matches_variant_levels() {
        let rmat = generate_rmat_graph(2_000, 4, RmatConfig::default(), 0xA11CE);
        let native = NativeCsrGraph::from_nodes(&rmat.nodes, &rmat.overflow);
        let v0 = V0CsrGraph::from_nodes_and_overflow(&rmat.nodes, &rmat.overflow);
        let v2 = V2SlimArenaGraph::from_nodes_and_overflow(&rmat.nodes, &rmat.overflow);
        let v3 = V3InlineVisitedGraph::from_nodes_and_overflow(&rmat.nodes, &rmat.overflow);
        verify_graphs(&native, &v0, &v2, &v3);
    }
}