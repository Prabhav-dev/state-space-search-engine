# Discrete State-Space Search Engine (`state_search_engine`)

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
[![Rust](https://img.shields.io/badge/rust-2021%20edition-orange.svg)](https://www.rust-lang.org/)
[![Build Status](https://img.shields.io/badge/tests-14%20passed-brightgreen.svg)]()

A high-performance, cache-conscious research prototype and benchmark suite written in Rust for discrete state-space graph search algorithms. Designed to explore cache-resident data structures, lock-free parallel state traversal, probabilistic path deduplication, and hardware-level memory hierarchy dynamics.

---

## Technical Overview

Modern state-space exploration algorithms (such as model checking, automated planning, and graph traversal) spend the vast majority of their execution time bound by CPU memory bandwidth and cache line latency. Traditional graph representations like **Compressed Sparse Row (CSR)** cause nested pointer-chasing and multiple dependent memory lookups per node expansion.

`state_search_engine` introduces a **32-byte cache-aligned node structure** (`StateNode`) with inline out-edges and contiguous overflow storage, paired with **thread-partitioned frontiers** and **zero-allocation scratch space** to minimize cache line misses and lock contention.

---

## Architectural Highlights

### 1. 32-Byte Cache-Aligned Node Layout (`StateNode`)
- Enforced with `#[repr(C, align(32))]`. Fits two node records perfectly into a standard 64-byte L1/L2 cache line.
- Embeds up to **3 inline out-edges** directly within the node record (`outbound_links: [u32; 4]`, with `MAX_INLINE_EDGES = 3`).
- For nodes with out-degree $>3$, the 4th link slot stores a 32-bit offset into a contiguous secondary `overflow` vector, preserving uniform node record size.

```rust
pub const MAX_INLINE_EDGES: usize = 3;

#[repr(C)]
#[repr(align(32))]
pub struct StateNode {
    pub node_id: u32,
    pub structural_mask: u32,
    pub outbound_count: u32,
    pub metadata_flags: u32,
    pub outbound_links: [u32; 4], // 3 inline edges + 1 overflow index (when outbound_count > 3)
}
```

### 2. Zero-Allocation Traversal (`EngineScratch` & `ParallelEngineScratch`)
- Pre-allocates visited bit-vectors (`Vec<u64>` or `Vec<AtomicBool>`) and frontier queues (`cur` and `next` or `SharedCasFrontier`).
- Eliminates dynamic memory allocations and thread pool construction inside hot traversal timing blocks, ensuring measurements reflect pure graph traversal runtime.

### 3. Thread-Partitioned Frontiers (`H1`)
- Multi-threaded level-synchronous BFS uses local partitioned queues per worker thread (`Rayon` chunks).
- Avoids atomic queue pop contention and cache line bouncing suffered by shared CAS queue architectures on high-degree power-law graphs.

### 4. Zobrist Path Hashing & Bitstate Table (`H3`)
- Incremental 64-bit Zobrist path hashing (`PathHasher`) paired with a configurable power-of-two bitstate table (`ZobristVisitedSet`) using true 64-bit Zobrist XOR hashing for probabilistic path deduplication in deep state graphs.

### 5. Learned Context-Keyed Edge Priorities (`H4`)
- Online target-directed search with edge priority training to record successful trajectory paths and evaluate generalization on held-out test targets.

---

## Verified Hypothesis Audit Matrix (H1–H5)

All hypotheses were empirically evaluated in release mode (`opt-level=3`, `lto=true`, `codegen-units=1`):

| ID | Hypothesis Statement | Falsification Criterion | Empirical Status | Key Finding |
| :--- | :--- | :--- | :--- | :--- |
| **H1** | Parallel level-synchronous BFS with partitioned frontiers beats single-thread BFS. | Single-thread BFS matches/beats parallel BFS at all thread counts. | **SUPPORTED** | **2.32x speedup** at 4M nodes (67.8 MTEPS) using 6 threads. |
| **H2** | Shared CAS frontiers suffer higher degradation on power-law graphs vs. partitioned frontiers. | No throughput gap between partitioned frontiers and shared CAS frontiers. | **SUPPORTED** | Partitioned frontiers outperform Shared CAS queues by **3.57x–4.52x** on R-MAT power-law graphs due to CAS queue contention on hub nodes. |
| **H3** | Zobrist-style path dedup uses less memory than exact bitmap at acceptable error. | Memory footprint or error rate is worse than exact bitmap. | **FALSIFIED** | $2^{16}$ (8 KB) table suffers $>93\%$ coverage loss on 1M nodes; exact 1-bit bitmap (122 KB for 1M nodes) is far superior. |
| **H4** | Context-keyed learned edge priorities reduce nodes expanded on repeated queries. | Node expansion counts remain unchanged vs unguided search on held-out test queries. | **UNSUPPORTED** | Shows 0.0% to -12.5% savings on held-out test targets (acts as path memoization on training targets only). |
| **H5** | Inline fixed-degree adjacency is faster than CSR once working set exceeds L3 cache. | CSR matches or beats inline layout out of cache or on variable-degree graphs. | **PARTIALLY SUPPORTED** | Inline layout excels on uniform low-degree ($\le 3$) graphs; Native Rust CSR baseline is faster on variable-degree R-MAT graphs (0.76x–0.88x ratio) due to overflow vector indirect lookups. |

---

## Multi-Session Empirical Reproduction Table

Statistical variance across 3 isolated benchmark runs on uniform random graphs (generated via `scripts/generate_tables.py` from raw session logs):

| Node Count | Session 1 (ms) | Session 2 (ms) | Session 3 (ms) | Mean (ms) | Sample SD (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **100K** (100,000) | 1.155 | 1.143 | 1.129 | 1.142 | 1.14% |
| **500K** (500,000) | 15.794 | 16.479 | 14.181 | 15.485 | 7.62% |
| **1M** (1,000,000) | 36.726 | 36.717 | 33.694 | 35.712 | 4.89% |
| **4M** (4,000,000) | 165.274 | 162.522 | 157.144 | 161.647 | 2.56% |

---

## Directory Structure

```
.
├── Cargo.toml                  # Crate configuration & release profile optimizations
├── Cargo.lock                  # Locked dependency graph
├── LICENSE-MIT                 # MIT License file
├── LICENSE-APACHE              # Apache-2.0 License file
├── README.md                   # Project documentation & paper synthesis
├── raw_session_1.log           # Raw benchmark session 1 execution log
├── raw_session_2.log           # Raw benchmark session 2 execution log
├── raw_session_3.log           # Raw benchmark session 3 execution log
├── scripts/                    # Automated reproducible table generators
│   └── generate_tables.py      # Parses raw .log files directly into Markdown/LaTeX tables
├── src/                        # Core Engine Library
│   ├── lib.rs                  # Module exports and primary types
│   ├── engine.rs               # SearchEngine, EngineScratch, ParallelEngineScratch, and parallel BFS
│   ├── frontier.rs             # PartitionedFrontier & SharedCasFrontier
│   ├── graph.rs                # R-MAT generator, PRNG uniform generator, and reachability analysis
│   ├── hash.rs                 # PathHasher & ZobristVisitedSet
│   └── node.rs                 # 32-byte StateNode layout definition & MAX_INLINE_EDGES constant
├── benches/                    # Criterion Micro-benchmarks
│   ├── pointer_chase.rs        # Dependent load pointer chasing bench
│   └── traversal_bench.rs      # Engine vs CSR traversal throughput bench
├── examples/                   # Executable Analysis Suites
│   ├── compare_baseline.rs     # Layout-isolating BFS comparison
│   ├── external_baseline_comparison.rs # Engine vs. Native Rust CSR Baseline comparison
│   ├── multisession_archive.rs # Multi-session statistical variance analyzer
│   └── v02_full_analysis.rs    # Comprehensive H1-H5 empirical test runner
└── tests/                      # Integration Test Suite
    └── integration_tests.rs    # 10 integration tests validating engine correctness
```

---

## Quick Start & Reproduction Guide

### Prerequisites
- **Rust Toolchain**: 1.70 or newer (`cargo` and `rustc`)
- **Python**: 3.8+ (for automated table generation)

### Building in Release Mode
```bash
cargo build --release
```

### Running Integration Test Suite
```bash
cargo test
```
*Expected Output: 14 passed (4 unit tests, 10 integration tests).*

### Executing Full Analysis Suite (H1–H5 Evaluation)
```bash
cargo run --release --example v02_full_analysis
```

### Executing Native Rust CSR Baseline Comparison
```bash
cargo run --release --example external_baseline_comparison
```

### Executing Multi-Session Statistical Variance Test
```bash
cargo run --release --example multisession_archive
```

### Reproducing Paper Tables from Raw Logs
```bash
python scripts/generate_tables.py
```

---

## Code Example

```rust
use state_search_engine::{
    engine::{EngineScratch, ParallelEngineScratch, SearchEngine},
    graph::{generate_rmat_graph, RmatConfig},
};

fn main() {
    // 1. Generate a 1,000,000 node R-MAT power-law graph
    let graph_result = generate_rmat_graph(1_000_000, 4, RmatConfig::default(), 42);
    let engine = SearchEngine::with_overflow(graph_result.nodes, graph_result.overflow);

    // 2. Pre-allocate zero-allocation scratch space for single-threaded BFS
    let mut scratch = EngineScratch::new(engine.num_nodes());

    // 3. Execute high-throughput single-threaded BFS
    let edges_traversed = engine.traverse_bfs_with_scratch(0, &mut scratch);
    println!("Traversed {} edges cleanly with zero allocations!", edges_traversed);

    // 4. Execute parallel level-synchronous BFS (4 threads) with pre-allocated thread pool
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    let mut par_scratch = ParallelEngineScratch::new(engine.num_nodes());
    let parallel_edges = engine.traverse_parallel_bfs_with_pool_and_scratch(0, &pool, &mut par_scratch);
    println!("Parallel BFS traversed {} edges!", parallel_edges);
}
```

---

## License

Dual-licensed under either of:

- **MIT License** ([`LICENSE-MIT`](LICENSE-MIT) or http://opensource.org/licenses/MIT)
- **Apache License, Version 2.0** ([`LICENSE-APACHE`](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.
