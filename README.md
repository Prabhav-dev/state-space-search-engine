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
- Embeds up to **3 inline out-edges** directly within the node record (`outbound_links: [u32; 4]`).
- For nodes with out-degree $>4$, the 4th link slot stores a 32-bit offset into a contiguous secondary `overflow` vector, preserving uniform node record size.

```rust
#[repr(C)]
#[repr(align(32))]
pub struct StateNode {
    pub node_id: u32,
    pub structural_mask: u32,
    pub outbound_count: u32,
    pub metadata_flags: u32,
    pub outbound_links: [u32; 4], // 3 inline edges + 1 overflow index
}
```

### 2. Zero-Allocation Traversal (`EngineScratch`)
- Pre-allocates visited bit-vectors (`Vec<u64>`) and double-buffered frontier queues (`cur` and `next`).
- Eliminates dynamic memory allocations inside the hot traversal loop, ensuring measurements reflect raw memory bandwidth and CPU execution throughput.

### 3. Thread-Partitioned Frontiers (`H1`)
- Multi-threaded level-synchronous BFS uses local partitioned queues per worker thread (`Rayon` chunks).
- Avoids atomic queue pop contention and cache line bouncing suffered by shared CAS queue architectures on high-degree power-law graphs.

### 4. Zobrist Path Hashing & Bitstate Table (`H3`)
- Incremental 64-bit Zobrist path hashing (`PathHasher`) paired with a configurable power-of-two bitstate table (`ZobristVisitedSet`) for probabilistic path deduplication in deep state graphs.

### 5. Learned Context-Keyed Edge Priorities (`H4`)
- Online target-directed search with edge priority training to record successful trajectory paths and accelerate repeat state expansions.

---

## Micro-Architecture & Memory Hierarchy Analysis

Tested on AMD Ryzen 5 7430U architecture (Zen 3 core, 6 Cores / 12 SMT Threads, 16 MB unified L3 LLC, Linux x86_64).

### The Out-of-Cache Crossover ($N \ge 4\text{M}$ Nodes)
When working sets fit within the 16 MB L3 cache ($N \le 500\text{k}$ nodes), memory latency is masked by L1/L2/L3 bandwidth. However, when working sets exceed L3 cache size ($N = 4\text{M}$ nodes = **128 MB** total footprint):

- **CSR Representation**: Expanding node $u$ requires reading `offsets[u]` to locate adjacency boundaries, followed by reading `targets[offsets[u]]`. This forces **2 dependent cache line misses per node expansion**.
- **Inline Arena (`StateNode`)**: Expanding node $u$ reads the contiguous `StateNode` struct directly. For degree $\le 4$, all neighbor IDs are fetched in the **same 32-byte cache line hit**, resulting in only **1 cache line miss per node expansion**.
- **Empirical Gain**: The 32-byte Inline Arena achieves **1.17x–1.36x throughput speedup** over traditional CSR out of cache on variable-degree R-MAT graphs.

---

## Verified Hypothesis Audit Matrix (H1–H5)

All hypotheses were empirically evaluated in release mode (`opt-level=3`, `lto=true`, `codegen-units=1`):

| ID | Hypothesis Statement | Falsification Criterion | Empirical Status | Key Finding |
| :--- | :--- | :--- | :--- | :--- |
| **H1** | Pinned workers with partitioned frontiers beat tuned single-thread BFS. | Single-thread matches/beats parallel BFS at all thread counts. | **SUPPORTED** | **2.17x speedup** at 4M nodes (109.5 MTEPS) using 6 threads. |
| **H2** | Shared CAS frontiers suffer higher degradation on power-law graphs vs. uniform graphs. | No throughput gap between graph types with multi-threaded CAS queues. | **SUPPORTED** | Partitioned frontiers outperform Shared CAS queues by **2.94x–3.58x** on R-MAT graphs. |
| **H3** | Zobrist-style path dedup uses less memory than exact bitmap at acceptable error. | Memory footprint or error rate is worse than exact bitmap. | **FALSIFIED** | $2^{16}$ (8 KB) table suffers 94.8% coverage loss; exact 1-bit bitmap (122 KB for 1M nodes) is far superior. |
| **H4** | Context-keyed learned edge priorities reduce nodes expanded on repeated queries. | Node expansions remain unchanged vs unguided search. | **UNSUPPORTED** | Shows 0.0% to -12.5% generalization savings on held-out test targets (memoization only). |
| **H5** | Inline fixed-degree adjacency is faster than CSR once working set exceeds L3 cache. | CSR matches or beats inline layout out of cache or on power-law graphs. | **SUPPORTED** | **1.17x–1.36x faster** than CSR on variable-degree R-MAT graphs at $N=1\text{M}, 4\text{M}$ out-of-cache. |

---

## Directory Structure

```
.
├── Cargo.toml                  # Crate configuration & release profile optimizations
├── Cargo.lock                  # Locked dependency graph
├── LICENSE-MIT                 # MIT License file
├── LICENSE-APACHE              # Apache-2.0 License file
├── README.md                   # Project documentation
├── src/                        # Core Engine Library
│   ├── lib.rs                  # Module exports and primary types
│   ├── engine.rs               # SearchEngine, EngineScratch, and parallel BFS
│   ├── frontier.rs             # PartitionedFrontier & SharedCasFrontier
│   ├── graph.rs                # R-MAT Kronecker power-law generator & analysis
│   ├── hash.rs                 # PathHasher & ZobristVisitedSet
│   └── node.rs                 # 32-byte StateNode layout definition
├── benches/                    # Criterion Micro-benchmarks
│   ├── pointer_chase.rs        # Pointer chasing & memory lookup latency bench
│   └── traversal_bench.rs      # Engine vs CSR traversal throughput bench
├── examples/                   # Executable Analysis Suites
│   ├── compare_baseline.rs     # Baseline performance comparator
│   ├── external_baseline_comparison.rs # Engine vs. GAPBS/Ligra CSR comparison
│   ├── multisession_archive.rs # Multi-session statistical variance analyzer
│   └── v02_full_analysis.rs    # Comprehensive H1-H5 empirical test runner
└── tests/                      # Integration Test Suite
    └── integration_tests.rs    # 10 integration tests validating engine correctness
```

---

## Quick Start & Reproduction Guide

### Prerequisites
- **Rust Toolchain**: 1.70 or newer (`cargo` and `rustc`)

### Building in Release Mode
```bash
cargo build --release
```

### Running Test Suite
```bash
cargo test
```
*Expected Output: 14 passed (4 unit tests, 10 integration tests).*

### Executing Full Analysis Suite (H1–H5 Evaluation)
```bash
cargo run --release --example v02_full_analysis
```

### Executing External CSR Baseline Comparison
```bash
cargo run --release --example external_baseline_comparison
```

### Executing Multi-Session Statistical Variance Test
```bash
cargo run --release --example multisession_archive
```

### Running Criterion Benchmarks
```bash
cargo bench
```

---

## Code Example

```rust
use state_search_engine::{
    engine::{EngineScratch, SearchEngine},
    graph::{generate_rmat_graph, RmatConfig},
};

fn main() {
    // 1. Generate a 1,000,000 node R-MAT power-law graph
    let graph_result = generate_rmat_graph(1_000_000, 4, RmatConfig::default(), 42);
    let engine = SearchEngine::with_overflow(graph_result.nodes, graph_result.overflow);

    // 2. Pre-allocate zero-allocation scratch space
    let mut scratch = EngineScratch::new(engine.num_nodes());

    // 3. Execute high-throughput single-threaded BFS
    let edges_traversed = engine.traverse_bfs_with_scratch(0, &mut scratch);
    println!("Traversed {} edges cleanly with zero allocations!", edges_traversed);

    // 4. Execute parallel level-synchronous BFS (4 threads)
    let parallel_edges = engine.traverse_parallel_bfs(0, 4);
    println!("Parallel BFS traversed {} edges!", parallel_edges);
}
```

---

## License

Dual-licensed under either of:

- **MIT License** ([`LICENSE-MIT`](LICENSE-MIT) or http://opensource.org/licenses/MIT)
- **Apache License, Version 2.0** ([`LICENSE-APACHE`](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.
