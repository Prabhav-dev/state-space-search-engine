//! Latency-bound random-walk and DFS comparison for V0, V2, and V3.
//! Run with: cargo run --release --example stage3_latency_workloads

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
const VARIANT_NAMES: [&str; 4] = ["ExistingNativeCsr", "V0", "V2", "V3"];

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

    fn has_internal_visited(&self) -> bool {
        matches!(self, Self::V3(_))
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
            Self::Existing(graph) => graph.num_nodes,
            Self::V0(graph) => graph.num_nodes,
            Self::V2(graph) => graph.num_nodes,
            Self::V3(graph) => graph.num_nodes,
        }
    }

    #[inline(always)]
    fn degree(&self, node_id: u32) -> usize {
        let node = node_id as usize;
        match self {
            Self::Existing(graph) if node < graph.num_nodes => {
                graph.offsets[node + 1] as usize - graph.offsets[node] as usize
            }
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
            Self::Existing(graph) if node_id < graph.num_nodes => {
                let start = graph.offsets[node_id] as usize;
                let end = graph.offsets[node_id + 1] as usize;
                (index < end - start).then(|| graph.targets[start + index])
            }
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

fn generate_low_degree_adjacency(num_nodes: usize, seed: u64) -> Vec<Vec<u32>> {
    let mut rng = SplitMix64(seed);
    (0..num_nodes)
        .map(|_| {
            let degree = 2 + (rng.next() % 2) as usize;
            (0..degree)
                .map(|_| (rng.next() % num_nodes as u64) as u32)
                .collect()
        })
        .collect()
}

fn run_family(
    family: &str,
    num_nodes: usize,
    adjacency: &[Vec<u32>],
    log: &mut CsvLog,
) -> io::Result<()> {
    run_graph_order(family, num_nodes, "Original", adjacency, log)?;
    let permutation = compute_permutation(adjacency, VertexOrder::DegreeSort, 0);
    let (degree_sorted, _) = apply_permutation(adjacency, &permutation);
    run_graph_order(family, num_nodes, "DegreeSort", &degree_sorted, log)
}

fn build_native_csr(adjacency: &[Vec<u32>]) -> NativeCsrGraph {
    let num_nodes = adjacency.len();
    let mut offsets = Vec::with_capacity(num_nodes + 1);
    let mut targets = Vec::new();
    for neighbors in adjacency {
        offsets.push(targets.len() as u32);
        targets.extend_from_slice(neighbors);
    }
    offsets.push(targets.len() as u32);
    let num_edges = targets.len();
    NativeCsrGraph {
        offsets,
        targets,
        num_nodes,
        num_edges,
    }
}

fn build_graphs(
    adjacency: &[Vec<u32>],
) -> (NativeCsrGraph, V0CsrGraph, V2SlimArenaGraph, V3InlineVisitedGraph) {
    (
        build_native_csr(adjacency),
        V0CsrGraph::from_adjacency(adjacency),
        V2SlimArenaGraph::from_adjacency(adjacency),
        V3InlineVisitedGraph::from_adjacency(adjacency),
    )
}

fn verify_workloads(graphs: &[GraphRef<'_>; 4]) {
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
            "{} single-chain walk checksum differs from ExistingNativeCsr",
            VARIANT_NAMES[index]
        );
        let mut scratch = WalkerScratch::new(8);
        scratch.reset(graph, 0xA11CE);
        assert_eq!(
            interleaved_walk(graph, &mut scratch, 2_400),
            reference_interleaved,
            "{} interleaved walk checksum differs from ExistingNativeCsr",
            VARIANT_NAMES[index]
        );
        let mut dfs = DfsScratch::new(graph.num_nodes());
        dfs.reset();
        iterative_dfs(graph, 0, &mut dfs);
        assert_eq!(dfs.order, reference_order, "{} DFS order differs from ExistingNativeCsr", VARIANT_NAMES[index]);
        assert_eq!(dfs.visited, reference_visited, "{} DFS reached set differs from ExistingNativeCsr", VARIANT_NAMES[index]);
    }
}

fn record_workload(
    log: &mut CsvLog,
    graph: &GraphRef<'_>,
    variant: &str,
    family: &str,
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
    session_power: &str,
) {
    log.record(&format!(
        "{session},{run},{warmup},{family},{num_nodes},{order},{workload},{walkers},{variant},{elapsed_ms:.6},{checksum},{reached},{session_power}"
    ));
    black_box((graph.num_nodes(), graph.num_edges()));
}

fn run_graph_order(
    family: &str,
    num_nodes: usize,
    order_label: &str,
    adjacency: &[Vec<u32>],
    log: &mut CsvLog,
) -> io::Result<()> {
    let (native, v0, v2, v3) = build_graphs(adjacency);
    let graphs = [
        GraphRef::Existing(&native),
        GraphRef::V0(&v0),
        GraphRef::V2(&v2),
        GraphRef::V3(&v3),
    ];
    verify_workloads(&graphs);
    let overflow_nodes = adjacency.iter().filter(|neighbors| neighbors.len() > 3).count();
    let p_o = overflow_nodes as f64 / num_nodes as f64;
    println!("Graph={family} | N={num_nodes} | order={order_label} | edges={} | p_o={p_o:.6} ({:.3}%)", v0.num_edges, p_o * 100.0);

    for session in 0..SESSIONS {
        let session_machine = MachineState::capture_session(&[1], session);
        log.record_session_state(&session.to_string(), &session_machine);
        log.flush()?;
        session_machine.require_ac_power()?;
        let session_power = session_machine.csv_session_fields();
        for run in 0..(WARMUPS + TIMED_RUNS) {
            let first_variant = (session + run) % graphs.len();
            let mut checksums = [0u64; 4];
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
                    family,
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
                    &session_power,
                );
            }
            for variant in 1..graphs.len() {
                assert_eq!(checksums[0], checksums[variant], "single-chain {} checksum mismatch", VARIANT_NAMES[variant]);
            }
        }
        log.flush()?;
    }

    for &walker_count in &INTERLEAVED_WALKERS {
        let mut walker_scratch: [WalkerScratch; 4] =
            std::array::from_fn(|_| WalkerScratch::new(walker_count));
        for session in 0..SESSIONS {
            let session_machine = MachineState::capture_session(&[1], session);
            log.record_session_state(&session.to_string(), &session_machine);
            log.flush()?;
            session_machine.require_ac_power()?;
            let session_power = session_machine.csv_session_fields();
            for run in 0..(WARMUPS + TIMED_RUNS) {
                let first_variant = (session + run) % graphs.len();
                let mut checksums = [0u64; 4];
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
                        family,
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
                        &session_power,
                    );
                }
                for variant in 1..graphs.len() {
                    assert_eq!(checksums[0], checksums[variant], "interleaved {} checksum mismatch", VARIANT_NAMES[variant]);
                }
            }
            log.flush()?;
        }
    }

    let mut dfs_scratch: [DfsScratch; 4] = std::array::from_fn(|_| DfsScratch::new(num_nodes));
    for session in 0..SESSIONS {
        let session_machine = MachineState::capture_session(&[1], session);
        log.record_session_state(&session.to_string(), &session_machine);
        log.flush()?;
        session_machine.require_ac_power()?;
        let session_power = session_machine.csv_session_fields();
        for run in 0..(WARMUPS + TIMED_RUNS) {
            let first_variant = (session + run) % graphs.len();
            let mut checksums = [0u64; 4];
            let mut reached = [0usize; 4];
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
                    family,
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
                    &session_power,
                );
            }
            for variant in 1..graphs.len() {
                assert_eq!(checksums[0], checksums[variant], "DFS {} visit order checksum mismatch", VARIANT_NAMES[variant]);
                assert_eq!(reached[0], reached[variant], "DFS {} reached count mismatch", VARIANT_NAMES[variant]);
            }
        }
        log.flush()?;
    }
    Ok(())
}

fn run() -> io::Result<()> {
    let machine = MachineState::capture(&[1]);
    machine.print_header();
    let mut log = CsvLog::create(
        "stage3_latency_workloads",
        &machine,
        "session,run,warmup,family,n,order,workload,walkers,variant,elapsed_ms,checksum,reached,ac_battery,power_plan",
    )?;
    log.add_header_comment(PREDICTION)?;
    log.record_session_state("startup", &machine);
    log.flush()?;
    machine.require_ac_power()?;
    println!("{PREDICTION}");
    println!("Pilot protocol: {SESSIONS} sessions, {WARMUPS} warmup, median of {TIMED_RUNS} timed runs; walks={WALKS}, steps_per_walk={STEPS_PER_WALK}");

    for &num_nodes in &SIZES {
        println!("Generating R-MAT N={num_nodes}...");
        {
            let rmat = generate_rmat_graph(
                num_nodes,
                4,
                RmatConfig::default(),
                0x51A6_1E00 ^ num_nodes as u64,
            );
            let adjacency = adjacency_from_rmat(&rmat);
            drop(rmat);
            run_family("RMAT", num_nodes, &adjacency, &mut log)?;
        }
        let low_degree = generate_low_degree_adjacency(
            num_nodes,
            0x10_D3_6AEE ^ num_nodes as u64,
        );
        run_family("RandomDegree2to3", num_nodes, &low_degree, &mut log)?;
    }

    let path = log.finish()?;
    println!("Raw CSV: {}", path.display());
    Ok(())
}

fn run_graph_telemetry() -> io::Result<()> {
    let machine = MachineState::capture(&[1]);
    machine.print_header();
    machine.require_ac_power()?;
    let mut log = CsvLog::create(
        "stage3_graph_telemetry",
        &machine,
        "family,n,edges,overflow_nodes,p_o",
    )?;
    for &num_nodes in &SIZES {
        let rmat = generate_rmat_graph(
            num_nodes,
            4,
            RmatConfig::default(),
            0x51A6_1E00 ^ num_nodes as u64,
        );
        let overflow_nodes = rmat.nodes.iter().filter(|node| node.outbound_count > 3).count();
        let p_o = overflow_nodes as f64 / num_nodes as f64;
        println!("Graph=RMAT | N={num_nodes} | edges={} | p_o={p_o:.6} ({:.3}%)", rmat.total_edges, p_o * 100.0);
        log.record(&format!("RMAT,{num_nodes},{},{overflow_nodes},{p_o:.9}", rmat.total_edges));

        let adjacency = generate_low_degree_adjacency(
            num_nodes,
            0x10_D3_6AEE ^ num_nodes as u64,
        );
        let edges: usize = adjacency.iter().map(Vec::len).sum();
        let overflow_nodes = adjacency.iter().filter(|neighbors| neighbors.len() > 3).count();
        let p_o = overflow_nodes as f64 / num_nodes as f64;
        println!("Graph=RandomDegree2to3 | N={num_nodes} | edges={edges} | p_o={p_o:.6} ({:.3}%)", p_o * 100.0);
        log.record(&format!("RandomDegree2to3,{num_nodes},{edges},{overflow_nodes},{p_o:.9}"));
        log.flush()?;
    }
    println!("Telemetry CSV: {}", log.finish()?.display());
    Ok(())
}

fn main() -> io::Result<()> {
    if std::env::args().nth(1).as_deref() == Some("--telemetry-only") {
        run_graph_telemetry()
    } else {
        run()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_latency_workloads_match_v0_on_each_layout() {
        let rmat = generate_rmat_graph(3_000, 4, RmatConfig::default(), 0xA11CE);
        let adjacency = adjacency_from_rmat(&rmat);
        let (native, v0, v2, v3) = build_graphs(&adjacency);
        verify_workloads(&[
            GraphRef::Existing(&native),
            GraphRef::V0(&v0),
            GraphRef::V2(&v2),
            GraphRef::V3(&v3),
        ]);
    }

    #[test]
    fn low_degree_graph_matches_control_and_has_no_overflow_nodes() {
        let adjacency = generate_low_degree_adjacency(3_000, 0x1D_2D_3D);
        assert!(adjacency.iter().all(|neighbors| (2..=3).contains(&neighbors.len())));
        assert_eq!(adjacency.iter().filter(|neighbors| neighbors.len() > 3).count(), 0);
        let (native, v0, v2, v3) = build_graphs(&adjacency);
        verify_workloads(&[
            GraphRef::Existing(&native),
            GraphRef::V0(&v0),
            GraphRef::V2(&v2),
            GraphRef::V3(&v3),
        ]);
    }
}