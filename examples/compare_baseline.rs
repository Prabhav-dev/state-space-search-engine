//! Layout-isolating BFS comparison. Run with:
//!   cargo run --release --example compare_baseline

use std::hint::black_box;
use std::time::Instant;

use state_search_engine::engine::{EngineScratch, SearchEngine};
use state_search_engine::graph::SimpleRng;
use state_search_engine::node::{StateNode, MAX_INLINE_EDGES};

const WARMUP: usize = 3;
const RUNS: usize = 15;

/// Uniform random graph, out-degree 2 using deterministic PRNG.
fn random_edges(n: usize, seed: u64) -> Vec<[u32; 2]> {
    let mut rng = SimpleRng::new(seed);
    (0..n)
        .map(|_| [rng.gen_range(n as u64) as u32, rng.gen_range(n as u64) as u32])
        .collect()
}

fn build_aos(edges: &[[u32; 2]]) -> Vec<StateNode> {
    edges
        .iter()
        .enumerate()
        .map(|(i, e)| StateNode {
            node_id: i as u32,
            structural_mask: 0,
            outbound_count: 2,
            metadata_flags: 0,
            outbound_links: [e[0], e[1], u32::MAX, u32::MAX],
        })
        .collect()
}

fn build_csr(edges: &[[u32; 2]]) -> (Vec<u32>, Vec<u32>) {
    let mut offsets = Vec::with_capacity(edges.len() + 1);
    let mut targets = Vec::with_capacity(edges.len() * 2);
    for e in edges {
        offsets.push(targets.len() as u32);
        targets.extend_from_slice(e);
    }
    offsets.push(targets.len() as u32);
    (offsets, targets)
}

#[inline(always)]
fn test_and_set(visited: &mut [u64], v: u32) -> bool {
    let w = (v >> 6) as usize;
    let m = 1u64 << (v & 63);
    let old = visited[w];
    visited[w] = old | m;
    (old & m) != 0
}

fn aos_bitmap_bfs(nodes: &[StateNode], start: u32, s: &mut EngineScratch) -> u64 {
    s.reset();
    let EngineScratch { visited, cur, next } = s;
    let mut edges = 0u64;
    test_and_set(visited, start);
    cur.push(start);
    while !cur.is_empty() {
        for &u in cur.iter() {
            let node = &nodes[u as usize];
            let count = (node.outbound_count as usize).min(MAX_INLINE_EDGES);
            for &v in &node.outbound_links[..count] {
                edges += 1;
                if !test_and_set(visited, v) {
                    next.push(v);
                }
            }
        }
        cur.clear();
        std::mem::swap(cur, next);
    }
    edges
}

fn csr_bitmap_bfs(offsets: &[u32], targets: &[u32], start: u32, s: &mut EngineScratch) -> u64 {
    s.reset();
    let EngineScratch { visited, cur, next } = s;
    let mut edges = 0u64;
    test_and_set(visited, start);
    cur.push(start);
    while !cur.is_empty() {
        for &u in cur.iter() {
            let lo = offsets[u as usize] as usize;
            let hi = offsets[u as usize + 1] as usize;
            for &v in &targets[lo..hi] {
                edges += 1;
                if !test_and_set(visited, v) {
                    next.push(v);
                }
            }
        }
        cur.clear();
        std::mem::swap(cur, next);
    }
    edges
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    println!("release-mode BFS, median of {RUNS} interleaved runs ({WARMUP} warmup discarded)");
    println!("MTEPS = million edges traversed per second\n");

    for &n in &[100_000usize, 250_000, 500_000, 1_000_000, 4_000_000] {
        let edges_list = random_edges(n, 0xC0FF_EE00 ^ n as u64);
        let aos = build_aos(&edges_list);
        let (offsets, targets) = build_csr(&edges_list);
        let engine = SearchEngine::new(aos.clone());
        let mut s_engine = EngineScratch::new(n);
        let mut s_aos = EngineScratch::new(n);
        let mut s_csr = EngineScratch::new(n);

        let (mut ta, mut tb, mut tc) = (Vec::new(), Vec::new(), Vec::new());
        let mut edges_seen = 0u64;

        for r in 0..(WARMUP + RUNS) {
            let t = Instant::now();
            let ea = engine.traverse_bfs_with_scratch(black_box(0), &mut s_engine);
            let da = t.elapsed().as_secs_f64() * 1e3;

            let t = Instant::now();
            let eb = aos_bitmap_bfs(&aos, black_box(0), &mut s_aos);
            let db = t.elapsed().as_secs_f64() * 1e3;

            let t = Instant::now();
            let ec = csr_bitmap_bfs(&offsets, &targets, black_box(0), &mut s_csr);
            let dc = t.elapsed().as_secs_f64() * 1e3;

            assert_eq!(ea, eb, "engine vs aos_bitmap edge count mismatch");
            assert_eq!(eb, ec, "aos_bitmap vs csr edge count mismatch");
            edges_seen = ea;
            if r >= WARMUP {
                ta.push(da);
                tb.push(db);
                tc.push(dc);
            }
        }

        let reached = edges_seen / 2; // every node has out-degree 2
        let (ma, mb, mc) = (median(&mut ta), median(&mut tb), median(&mut tc));
        let mteps = |ms: f64| edges_seen as f64 / ms / 1e3;
        let mb_of = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);

        println!(
            "n={n}  reached={reached} ({:.1}%)  edges={edges_seen}",
            100.0 * reached as f64 / n as f64
        );
        println!(
            "  footprint: AoS {:.1} MB | CSR {:.1} MB | bitmap {:.2} MB",
            mb_of(n * 32),
            mb_of((offsets.len() + targets.len()) * 4),
            mb_of(s_csr.visited.len() * 8)
        );
        println!("  A engine     : {ma:8.3} ms  ({:.1} MTEPS)", mteps(ma));
        println!("  B aos_bitmap : {mb:8.3} ms  ({:.1} MTEPS)", mteps(mb));
        println!("  C csr_bitmap : {mc:8.3} ms  ({:.1} MTEPS)", mteps(mc));
        println!("  A/B = {:.2}x   B/C = {:.2}x\n", ma / mb, mb / mc);
    }
}