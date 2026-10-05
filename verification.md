# Verification & Empirical Performance Telemetry (`verification.md`)

This document records the exact, un-truncated empirical performance numbers, throughput metrics (MTEPS), memory footprints, statistical variance, and audit findings generated directly by running the executable benchmark suite in release mode (`opt-level=3`, `lto=true`, `codegen-units=1`).

---

## 1. Multi-Session Statistical Variance Metrics

Measured across 3 isolated, cold-start benchmark sessions on uniform random graphs (10 timed runs per session, excluding 3 warmup runs). Processed via `scripts/generate_tables.py`:

| Node Count | Memory (MB) | Session 1 Median (ms) | Session 2 Median (ms) | Session 3 Median (ms) | Mean Runtime (ms) | Sample SD (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **100,000** | 3.2 MB | 1.155 ms | 1.143 ms | 1.129 ms | 1.142 ms | **1.14%** |
| **500,000** | 16.0 MB | 15.794 ms | 16.479 ms | 14.181 ms | 15.485 ms | **7.62%** |
| **1,000,000** | 32.0 MB | 36.726 ms | 36.717 ms | 33.694 ms | 35.712 ms | **4.89%** |
| **4,000,000** | 128.0 MB | 165.274 ms | 162.522 ms | 157.144 ms | 161.647 ms | **2.56%** |

---

## 2. Hypothesis H1: Parallel Level-Synchronous BFS Scaling

Evaluates level-synchronous parallel BFS throughput using per-thread partitioned frontiers across 1, 2, 4, and 6 Rayon worker threads.

### 1,000,000 Nodes (32.0 MB Footprint, Fitting inside/near L3)
- **1 Thread**: 32.626 ms | **48.9 MTEPS** | Speedup: **1.00x** (100.0% Efficiency)
- **2 Threads**: 28.947 ms | **55.1 MTEPS** | Speedup: **1.13x** (56.4% Efficiency)
- **4 Threads**: 25.451 ms | **62.7 MTEPS** | Speedup: **1.28x** (32.0% Efficiency)
- **6 Threads**: 25.691 ms | **62.1 MTEPS** | Speedup: **1.27x** (21.2% Efficiency)

### 4,000,000 Nodes (128.0 MB Footprint, Out-of-Cache L3 Spill)
- **1 Thread**: 218.407 ms | **29.2 MTEPS** | Speedup: **1.00x** (100.0% Efficiency)
- **2 Threads**: 173.423 ms | **36.7 MTEPS** | Speedup: **1.26x** (63.0% Efficiency)
- **4 Threads**: 115.253 ms | **55.3 MTEPS** | Speedup: **1.90x** (47.4% Efficiency)
- **6 Threads**: 93.974 ms | **67.8 MTEPS** | Speedup: **2.32x** (38.7% Efficiency)

---

## 3. Hypothesis H2: Shared CAS vs. Partitioned Frontiers on R-MAT Power-Law Graphs

Direct comparison of 4-thread Partitioned Frontiers versus 4-thread Shared CAS Atomic Queue Frontiers on R-MAT scale-free power-law graphs with hub nodes.

| Graph Scale $N$ | Total Edges | Reached Nodes | Partitioned Frontier (ms) | Partitioned MTEPS | Shared CAS Frontier (ms) | Shared CAS MTEPS | Throughput Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **100,000** | 400,000 | 41,750 (41.8%) | 2.467 ms | 156.0 MTEPS | 11.144 ms | 34.5 MTEPS | **4.52x Faster** |
| **500,000** | 2,000,000 | 178,291 (35.7%) | 12.850 ms | 150.3 MTEPS | 47.727 ms | 40.5 MTEPS | **3.71x Faster** |
| **1,000,000** | 4,000,000 | 339,533 (34.0%) | 25.601 ms | 151.1 MTEPS | 91.454 ms | 42.3 MTEPS | **3.57x Faster** |

---

## 4. Hypothesis H3: Zobrist Bitstate Deduplication Telemetry

Evaluates probabilistic Zobrist bitstate table coverage loss and false-positive collision rates using true 64-bit Zobrist XOR hashing.

| Node Count | Exact Bitmap Size | Zobrist Table Power ($2^k$) | Zobrist Memory | Memory Ratio vs. Bitmap | Reached States | Exact Reachable States | Coverage (%) | False Positive Collisions |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **100,000** | 12.21 KB | $2^{16}$ (64K bits) | 8.0 KB | 0.66x | 33,764 | 79,696 | **42.4%** | 26,672 |
| **100,000** | 12.21 KB | $2^{18}$ (256K bits) | 32.0 KB | 2.62x | 62,484 | 79,696 | **78.4%** | 29,491 |
| **100,000** | 12.21 KB | $2^{20}$ (1M bits) | 128.0 KB | 10.49x | 75,982 | 79,696 | **95.3%** | 14,421 |
| **500,000** | 61.04 KB | $2^{16}$ (64K bits) | 8.0 KB | 0.13x | 47,383 | 398,874 | **11.9%** | 44,875 |
| **500,000** | 61.04 KB | $2^{18}$ (256K bits) | 32.0 KB | 0.52x | 145,863 | 398,874 | **36.6%** | 119,866 |
| **500,000** | 61.04 KB | $2^{20}$ (1M bits) | 128.0 KB | 2.10x | 294,283 | 398,874 | **73.8%** | 155,796 |
| **1,000,000** | 122.07 KB | $2^{16}$ (64K bits) | 8.0 KB | 0.07x | 49,583 | 797,864 | **6.2%** | 48,158 |
| **1,000,000** | 122.07 KB | $2^{18}$ (256K bits) | 32.0 KB | 0.26x | 172,422 | 797,864 | **21.6%** | 155,496 |
| **1,000,000** | 122.07 KB | $2^{20}$ (1M bits) | 128.0 KB | 1.05x | 441,649 | 797,864 | **55.4%** | 308,097 |

---

## 5. Hypothesis H4: Learned Context-Keyed Edge Priorities

Evaluation of expansion savings on held-out test targets following training on target set $T_{train} = \{1000, 2000, 5000, 10000\}$ ($N = 100,000$ R-MAT graph):

- **Held-Out Test Target `5005`**:
  - Unlearned Expansions: **29,741**
  - Learned Expansions: **33,472**
  - Expansion Savings: **-12.5%** (overhead due to non-generalizing edge reordering)
- **Held-Out Test Target `10005`**:
  - Unlearned Expansions: **27,500**
  - Learned Expansions: **27,500**
  - Expansion Savings: **0.0%** (no effect on unvisited subgraphs)

---

## 6. Hypothesis H5 & Baseline Comparison: 32-Byte Inline Arena vs. Native Rust CSR

Interleaved execution benchmark comparing the 32-byte Inline Arena (`StateNode` with overflow) against the Native Rust CSR Baseline (`NativeCsrGraph`) on R-MAT power-law graphs.

| Graph Scale $N$ | Total Edges | Arena Memory | CSR Memory | Threads | Engine Runtime (ms) | Engine Throughput | Native CSR Runtime (ms) | Native CSR Throughput | Throughput Ratio (CSR / Engine) | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **500,000** | 2,000,000 | 16.0 MB | 10.0 MB | 1 | 20.969 ms | 92.1 MTEPS | 16.200 ms | 119.2 MTEPS | 0.77x | CSR Faster |
| **500,000** | 2,000,000 | 16.0 MB | 10.0 MB | 4 | 13.805 ms | 139.9 MTEPS | 13.180 ms | 146.6 MTEPS | 0.95x | CSR Faster |
| **500,000** | 2,000,000 | 16.0 MB | 10.0 MB | 6 | 11.086 ms | 174.2 MTEPS | 10.841 ms | 178.2 MTEPS | 0.98x | CSR Faster |
| **1,000,000** | 4,000,000 | 32.0 MB | 20.0 MB | 1 | 47.483 ms | 81.5 MTEPS | 37.431 ms | 103.4 MTEPS | 0.79x | CSR Faster |
| **1,000,000** | 4,000,000 | 32.0 MB | 20.0 MB | 4 | 28.106 ms | 137.6 MTEPS | 26.325 ms | 147.0 MTEPS | 0.94x | CSR Faster |
| **1,000,000** | 4,000,000 | 32.0 MB | 20.0 MB | 6 | 23.116 ms | 167.4 MTEPS | 21.049 ms | 183.8 MTEPS | 0.91x | CSR Faster |
| **4,000,000** | 16,000,000 | 128.0 MB | 80.0 MB | 1 | 224.179 ms | 69.2 MTEPS | 206.315 ms | 75.2 MTEPS | 0.92x | CSR Faster |
| **4,000,000** | 16,000,000 | 128.0 MB | 80.0 MB | 4 | 114.044 ms | 136.1 MTEPS | 117.136 ms | 132.5 MTEPS | **1.03x** | **Engine Faster** |
| **4,000,000** | 16,000,000 | 128.0 MB | 80.0 MB | 6 | 89.745 ms | 172.9 MTEPS | 94.095 ms | 164.9 MTEPS | **1.05x** | **Engine Faster** |

---

## 7. Command Reproducibility Summary

```bash
# 1. Run multi-session archive to generate raw logs
cargo run --release --example multisession_archive

# 2. Automatically generate reproducible summary tables from raw logs
python scripts/generate_tables.py

# 3. Run full empirical analysis matrix (H1-H5)
cargo run --release --example v02_full_analysis

# 4. Run baseline comparative benchmark
cargo run --release --example external_baseline_comparison
```
