//! Latency-bound random-walk and DFS comparison for V0, V2, and V3.
//! Run with: cargo run --release --example stage3_latency_workloads

#[path = "support/bench_support.rs"]
mod bench_support;

use std::hint::black_box;
use std::io;
use std::time::Instant;

use bench_support::{CsvLog, MachineState};
use state_search_engine::graph::{generate_rmat_graph, RmatConfig};
use state_search_engine::variants::reorder::{apply_permutation, compute_permutation};
use state_search_engine::variants::traits::{GraphLayout, VertexOrder};
use state_search_engine::variants::v0_csr::V0CsrGraph;
use state_search_engine::variants::v2_slim_arena::V2SlimArenaGraph;
use state_search_engine::variants::v3_inline_visited::V3InlineVisitedGraph;

const SESSIONS: usize = 3;
const WARMUPS: usize = 1;
const TIMED_RUNS: usize = 3;
const WALKS: usize = 20_000;
const STEPS_PER_WALK: usize = 1_000;
const INTERLEAVED_WALKERS: [usize; 2] = [8, 16];
const SIZES: [usize; 3] = [1_000_000, 4_000_000, 8_000_000];
const PREDICTION: &str = "PRE-REGISTERED: V2 beats V0 CSR by >=1.3x on the single-chain random walk at >=4M nodes; falsified if the paired ratio is <=1.1x.";
const VARIANT_NAMES: [&str; 3] = ["V0", "V2", "V3"];

enum GraphRef<'a> {
    V0(&'a V0CsrGraph),
    V2(&'a V2SlimArenaGraph),
    V3(&'a V3InlineVisitedGraph),
}

impl GraphLayout for GraphRef<'_> {
    fn num_nodes(&self) -> usize {
        match self {
            Self::V0(graph) => graph.num_nodes(),
            Self::V2(graph) => graph.num_nodes(),
            Self::V3(graph) => graph.num_nodes(),
        }
    }

    fn num_edges(&self) -> usize {
        match self {
            Self::V0(graph) => graph.num_edges(),
            Self::V2(graph) => graph.num_edges(),
            Self::V3(graph) => graph.num_edges(),
        }
    }

    fn for_each_neighbor<F: FnMut(u32)>(&self, node_id: u32, f: F) {
        match self {
            Self::V0(graph) => graph.for_each_neighbor(node_id, f),
            Self::V2(graph) => graph.for_each_neighbor(node_id, f),
            Self::V3(graph) => graph.for_each_neighbor(node_id, f),
        }
    }

    fn has_internal_visited(&self) -> bool {
        matches!(self, Self::V3(_))
    }
}

trait IndexedGraph {
    fn node_count(&self) -> usize;
    fn degree(&self, node_id: u32) -> usize;
    fn neighbor_at(&self, node_id: u32, index: usize) -> Option<u32>;
}

impl IndexedGraph for GraphRef<'_> {
    #[inline(always)]
    fn node_count(&self) -> usize {
        match self {
            Self::V0(graph) => graph.num_nodes,
            Self::V2(graph) => graph.num_nodes,
            Self::V3(graph) => graph.num_nodes,
        }
    }

    #[inline(always)]
    fn degree(&self, node_id: u32) -> usize {
        let node = node_id as usize;
        match self {
            Self::V0(graph) if node < graph.num_nodes => {
                graph.offsets[node + 1] as usize - graph.offsets[node] as usize
            }
            Self::V2(graph) if node < graph.num_nodes => graph.nodes[node].count as usize,
            Self::V3(graph) if node < graph.num_nodes => {
                (graph.nodes[node].count_and_visited.load(std::sync::atomic::Ordering::Relaxed)
                    & state_search_engine::variants::v3_inline_visited::COUNT_MASK) as usize
            }
            _ => 0,
        }
    }

    #[inline(always)]
    fn neighbor_at(&self, node_id: u32, index: usize) -> Option<u32> {
        let node_id = node_id as usize;
        match self {
            Self::V0(graph) if node_id < graph.num_nodes => {
                let start = graph.offsets[node_id] as usize;
                let end = graph.offsets[node_id + 1] as usize;
                (index < end - start).then(|| graph.targets[start + index])
            }
            Self::V2(graph) if node_id < graph.num_nodes => {
                let node = &graph.nodes[node_id];
                let degree = node.count as usize;
                if index >= degree {
                    None
                } else if degree <= 3 || index < 2 {
                    Some(node.edges[index])
                } else {
                    let overflow_index = node.edges[2] as usize + index - 2;
                    graph.overflow.get(overflow_index).copied()
                }
            }
            Self::V3(graph) if node_id < graph.num_nodes => {
                let node = &graph.nodes[node_id];
                let degree = (node.count_and_visited.load(std::sync::atomic::Ordering::Relaxed)
                    & state_search_engine::variants::v3_inline_visited::COUNT_MASK) as usize;
                if index >= degree {
                    None
                } else if degree <= 3 || index < 2 {
                    Some(node.edges[index])
                } else {
                    let overflow_index = node.edges[2] as usize + index - 2;
                    graph.overflow.get(overflow_index).copied()
                }
            }
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
struct SplitMix64(u64);

impl SplitMix64 {
    #[inline(always)]
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }
}

#[inline(always)]
fn checksum_step(checksum: u64, node_id: u32) -> u64 {
    checksum.rotate_left(7) ^ (node_id as u64).wrapping_add(0x9E37_79B9)
}

fn single_chain_walk<G: IndexedGraph>(graph: &G, walks: usize, steps: usize, seed: u64) -> u64 {
    let node_count = graph.node_count();
    if node_count == 0 {
        return 0;
    }
    let mut rng = SplitMix64(seed);
    let mut checksum = 0xCBF2_9CE4_8422_2325;
    for _ in 0..walks {
        let mut current = (rng.next() % node_count as u64) as u32;
        for _ in 0..steps {
            let degree = graph.degree(current);
            current = if degree == 0 {
                (rng.next() % node_count as u64) as u32
            } else {
                let index = (rng.next() % degree as u64) as usize;
                graph.neighbor_at(current, index).expect("degree and indexed neighbors must agree")
            };
            checksum = checksum_step(checksum, current);
        }
    }
    checksum
}

struct WalkerScratch {
    states: Vec<SplitMix64>,
    current: Vec<u32>,
    checksums: Vec<u64>,
}

impl WalkerScratch {
    fn new(walkers: usize) -> Self {
        Self {
            states: vec![SplitMix64(0); walkers],
            current: vec![0; walkers],
            checksums: vec![0xCBF2_9CE4_8422_2325; walkers],
        }
    }

    fn reset(&mut self, graph: &GraphRef<'_>, seed: u64) {
        let node_count = graph.num_nodes();
        for walker in 0..self.states.len() {
            self.states[walker] = SplitMix64(seed ^ (walker as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
            self.current[walker] = (self.states[walker].next() % node_count as u64) as u32;
            self.checksums[walker] = 0xCBF2_9CE4_8422_2325;
        }
    }
}

fn interleaved_walk<G: IndexedGraph>(
    graph: &G,
    scratch: &mut WalkerScratch,
    total_steps: usize,
) -> u64 {
    let walkers = scratch.states.len();
    assert_eq!(total_steps % walkers, 0);
    let rounds = total_steps / walkers;
    for _ in 0..rounds {
        for walker in 0..walkers {
            let current = scratch.current[walker];
            let degree = graph.degree(current);
            let next = if degree == 0 {
                (scratch.states[walker].next() % graph.node_count() as u64) as u32
            } else {
                let index = (scratch.states[walker].next() % degree as u64) as usize;
                graph.neighbor_at(current, index).expect("degree and indexed neighbors must agree")
            };
            scratch.current[walker] = next;
            scratch.checksums[walker] = checksum_step(scratch.checksums[walker], next);
        }
    }
    scratch
        .checksums
        .iter()
        .zip(&scratch.current)
        .fold(0u64, |sum, (&checksum, &node)| checksum_step(sum ^ checksum, node))
}

struct DfsScratch {
    visited: Vec<bool>,
    stack: Vec<u32>,
    order: Vec<u32>,
}

impl DfsScratch {
    fn new(node_count: usize) -> Self {
        Self {
            visited: vec![false; node_count],
            stack: Vec::with_capacity(node_count),
            order: Vec::with_capacity(node_count),
        }
    }

    fn reset(&mut self) {
        self.visited.fill(false);
        self.stack.clear();
        self.order.clear();
    }
}

fn iterative_dfs<G: IndexedGraph>(graph: &G, start: u32, scratch: &mut DfsScratch) -> u64 {
    if (start as usize) >= scratch.visited.len() {
        return 0;
    }
    scratch.visited[start as usize] = true;
    scratch.stack.push(start);
    let mut checksum = 0xCBF2_9CE4_8422_2325;
    while let Some(node) = scratch.stack.pop() {
        scratch.order.push(node);
        checksum = checksum_step(checksum, node);
        for index in (0..graph.degree(node)).rev() {
            let neighbor = graph.neighbor_at(node, index).expect("degree and indexed neighbors must agree");
            let neighbor_index = neighbor as usize;
            if neighbor_index < scratch.visited.len() && !scratch.visited[neighbor_index] {
                scratch.visited[neighbor_index] = true;
                scratch.stack.push(neighbor);
            }
        }
    }
    checksum
}

fn adjacency_from_rmat(rmat: &state_search_engine::graph::RmatGraphResult) -> Vec<Vec<u32>> {
    rmat
        .nodes
        .iter()
        .map(|node| {
            let degree = node.outbound_count as usize;
            if degree <= 3 {
                node.outbound_links[..degree].to_vec()
            } else {
                let mut neighbors = node.outbound_links[..3].to_vec();
                let overflow_start = node.outbound_links[3] as usize;
                let overflow_count = degree - 3;
                neighbors.extend_from_slice(&rmat.overflow[overflow_start..overflow_start + overflow_count]);
                neighbors
            }
        })
        .collect()
}

fn build_graphs(adjacency: &[Vec<u32>]) -> (V0CsrGraph, V2SlimArenaGraph, V3InlineVisitedGraph) {
    (
        V0CsrGraph::from_adjacency(adjacency),
        V2SlimArenaGraph::from_adjacency(adjacency),
        V3InlineVisitedGraph::from_adjacency(adjacency),
    )
}

fn verify_workloads(graphs: &[GraphRef<'_>; 3]) {
    let reference_walk = single_chain_walk(&graphs[0], 200, 300, 0xA11CE);
    let reference_interleaved = {
        let mut scratch = WalkerScratch::new(8);
        scratch.reset(&graphs[0], 0xA11CE);
        interleaved_walk(&graphs[0], &mut scratch, 2_400)
    };
    let mut reference_dfs = DfsScratch::new(graphs[0].num_nodes());
    reference_dfs.reset();
    iterative_dfs(&graphs[0], 0, &mut reference_dfs);
    let reference_order = reference_dfs.order.clone();
    let reference_visited = reference_dfs.visited.clone();

    for (index, graph) in graphs.iter().enumerate().skip(1) {
        assert_eq!(
            single_chain_walk(graph, 200, 300, 0xA11CE),
            reference_walk,
            "{} single-chain walk checksum differs from V0",
            VARIANT_NAMES[index]
        );
        let mut scratch = WalkerScratch::new(8);
        scratch.reset(graph, 0xA11CE);
        assert_eq!(
            interleaved_walk(graph, &mut scratch, 2_400),
            reference_interleaved,
            "{} interleaved walk checksum differs from V0",
            VARIANT_NAMES[index]
        );
        let mut dfs = DfsScratch::new(graph.num_nodes());
        dfs.reset();
        iterative_dfs(graph, 0, &mut dfs);
        assert_eq!(dfs.order, reference_order, "{} DFS order differs from V0", VARIANT_NAMES[index]);
        assert_eq!(dfs.visited, reference_visited, "{} DFS reached set differs from V0", VARIANT_NAMES[index]);
    }
}

fn record_workload(
    log: &mut CsvLog,
    graph: &GraphRef<'_>,
    variant: &str,
    num_nodes: usize,
    order: &str,
    workload: &str,
    walkers: usize,
    session: usize,
    run: usize,
    warmup: bool,
    elapsed_ms: f64,
    checksum: u64,
    reached: usize,
) {
    log.record(&format!(
        "{session},{run},{warmup},{num_nodes},{order},{workload},{walkers},{variant},{elapsed_ms:.6},{checksum},{reached}"
    ));
    black_box((graph.num_nodes(), graph.num_edges()));
}

fn run_graph_order(
    num_nodes: usize,
    order_label: &str,
    adjacency: &[Vec<u32>],
    log: &mut CsvLog,
) {
    let (v0, v2, v3) = build_graphs(adjacency);
    let graphs = [GraphRef::V0(&v0), GraphRef::V2(&v2), GraphRef::V3(&v3)];
    verify_workloads(&graphs);
    println!("Graph N={num_nodes} | order={order_label} | edges={}", v0.num_edges);

    for session in 0..SESSIONS {
        for run in 0..(WARMUPS + TIMED_RUNS) {
            let first_variant = (session + run) % graphs.len();
            let mut checksums = [0u64; 3];
            for position in 0..graphs.len() {
                let variant_index = (first_variant + position) % graphs.len();
                let graph = &graphs[variant_index];
                let started = Instant::now();
                checksums[variant_index] = single_chain_walk(
                    graph,
                    WALKS,
                    STEPS_PER_WALK,
                    0x51A6_1E00 ^ num_nodes as u64 ^ session as u64,
                );
                let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
                black_box(checksums[variant_index]);
                record_workload(
                    log,
                    graph,
                    VARIANT_NAMES[variant_index],
                    num_nodes,
                    order_label,
                    "single_chain",
                    1,
                    session,
                    run,
                    run < WARMUPS,
                    elapsed_ms,
                    checksums[variant_index],
                    WALKS * STEPS_PER_WALK,
                );
            }
            assert_eq!(checksums[0], checksums[1], "single-chain V0/V2 checksum mismatch");
            assert_eq!(checksums[0], checksums[2], "single-chain V0/V3 checksum mismatch");
        }
    }

    for &walker_count in &INTERLEAVED_WALKERS {
        let mut walker_scratch: [WalkerScratch; 3] =
            std::array::from_fn(|_| WalkerScratch::new(walker_count));
        for session in 0..SESSIONS {
            for run in 0..(WARMUPS + TIMED_RUNS) {
                let first_variant = (session + run) % graphs.len();
                let mut checksums = [0u64; 3];
                for position in 0..graphs.len() {
                    let variant_index = (first_variant + position) % graphs.len();
                    let graph = &graphs[variant_index];
                    walker_scratch[variant_index].reset(
                        graph,
                        0x1A7E_0000 ^ num_nodes as u64 ^ session as u64,
                    );
                    let started = Instant::now();
                    checksums[variant_index] = interleaved_walk(
                        graph,
                        &mut walker_scratch[variant_index],
                        WALKS * STEPS_PER_WALK,
                    );
                    let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
                    black_box(checksums[variant_index]);
                    record_workload(
                        log,
                        graph,
                        VARIANT_NAMES[variant_index],
                        num_nodes,
                        order_label,
                        "interleaved",
                        walker_count,
                        session,
                        run,
                        run < WARMUPS,
                        elapsed_ms,
                        checksums[variant_index],
                        WALKS * STEPS_PER_WALK,
                    );
                }
                assert_eq!(checksums[0], checksums[1], "interleaved V0/V2 checksum mismatch");
                assert_eq!(checksums[0], checksums[2], "interleaved V0/V3 checksum mismatch");
            }
        }
    }

    let mut dfs_scratch: [DfsScratch; 3] = std::array::from_fn(|_| DfsScratch::new(num_nodes));
    for session in 0..SESSIONS {
        for run in 0..(WARMUPS + TIMED_RUNS) {
            let first_variant = (session + run) % graphs.len();
            let mut checksums = [0u64; 3];
            let mut reached = [0usize; 3];
            for position in 0..graphs.len() {
                let variant_index = (first_variant + position) % graphs.len();
                let graph = &graphs[variant_index];
                dfs_scratch[variant_index].reset();
                let started = Instant::now();
                checksums[variant_index] =
                    iterative_dfs(graph, 0, &mut dfs_scratch[variant_index]);
                let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
                reached[variant_index] = dfs_scratch[variant_index].order.len();
                black_box((checksums[variant_index], reached[variant_index]));
                record_workload(
                    log,
                    graph,
                    VARIANT_NAMES[variant_index],
                    num_nodes,
                    order_label,
                    "iterative_dfs",
                    1,
                    session,
                    run,
                    run < WARMUPS,
                    elapsed_ms,
                    checksums[variant_index],
                    reached[variant_index],
                );
            }
            assert_eq!(checksums[0], checksums[1], "DFS V0/V2 visit order checksum mismatch");
            assert_eq!(checksums[0], checksums[2], "DFS V0/V3 visit order checksum mismatch");
            assert_eq!(reached[0], reached[1], "DFS V0/V2 reached count mismatch");
            assert_eq!(reached[0], reached[2], "DFS V0/V3 reached count mismatch");
        }
    }
}

fn run() -> io::Result<()> {
    let machine = MachineState::capture(&[1]);
    machine.print_header();
    machine.require_ac_power()?;
    let mut log = CsvLog::create(
        "stage3_latency_workloads",
        &machine,
        "session,run,warmup,n,order,workload,walkers,variant,elapsed_ms,checksum,reached",
    )?;
    log.add_header_comment(PREDICTION);
    println!("{PREDICTION}");
    println!("Pilot protocol: {SESSIONS} sessions, {WARMUPS} warmup, median of {TIMED_RUNS} timed runs; walks={WALKS}, steps_per_walk={STEPS_PER_WALK}");

    for &num_nodes in &SIZES {
        println!("Generating R-MAT N={num_nodes}...");
        let rmat = generate_rmat_graph(
            num_nodes,
            4,
            RmatConfig::default(),
            0x51A6_1E00 ^ num_nodes as u64,
        );
        let adjacency = adjacency_from_rmat(&rmat);
        drop(rmat);
        run_graph_order(num_nodes, "Original", &adjacency, &mut log);

        let permutation = compute_permutation(&adjacency, VertexOrder::DegreeSort, 0);
        let (degree_sorted, _) = apply_permutation(&adjacency, &permutation);
        run_graph_order(num_nodes, "DegreeSort", &degree_sorted, &mut log);
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
    fn all_latency_workloads_match_v0_on_each_layout() {
        let rmat = generate_rmat_graph(3_000, 4, RmatConfig::default(), 0xA11CE);
        let adjacency = adjacency_from_rmat(&rmat);
        let (v0, v2, v3) = build_graphs(&adjacency);
        verify_workloads(&[GraphRef::V0(&v0), GraphRef::V2(&v2), GraphRef::V3(&v3)]);
    }
}