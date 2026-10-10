# Findings Snapshot

Run date: 2026-10-10. Benchmark source commit: `f7d86f9ca2d0246760f94f5be6c3dc1ab575b07b`. Release profile only. Machine headers record Windows x86_64, Balanced power plan, 12 logical threads, and AC power for Tasks 1-2. The worktree was dirty. No hardware-counter causes are claimed.

## Task 1

Command: `cargo run --release --example stage1_control_study`. Ten interleaved sessions; each session used 3 warmups and the median of 10 timed runs. The raw log and CSV-generated tables are `logs/stage1_control_2026-10-10_20-49-55-356.csv` and `results/stage1_control_2026-10-10_20-49-55-356_tables.md`.

All layouts produced the same edge counts, reached counts, and BFS levels. The `ExistingNativeCsr` row uses the existing graph representation with the shared BFS driver, not its original traversal methods; it isolates layout while keeping reset/allocation outside timing.

V0/Existing paired overhead was supported at 500K for 1T (+9.07%, 95% CI +/- 4.22 percentage points), 4T (+3.12%, +/- 2.72), and 6T (+2.44%, +/- 1.32), and at 1M for 1T (+1.47%, +/- 0.80) and 6T (+1.68%, +/- 1.40). At 1M/4T (+0.02%, +/- 1.19), there was no detectable difference.

With V2 and V3 both using `test_before_atomic=false`, V0/V2 ratios were below 1 with CIs excluding 1 in all six size/thread configurations; V2 was slower in this controlled BFS comparison. V2/V3 had no detectable difference at 500K/1T and 6T; V3 was slower at 500K/4T and all 1M thread counts. These results do not establish a cause.

## Task 2

Command: `cargo run --release --example stage2_two_stage_prefetch`. Ten interleaved sessions; 3 warmups and median of 10 timed runs. The study covered original-order 4M/8M graphs, 1T/4T/6T, no-prefetch, and matching two-stage `k=4,8,16` for V0/V2. Raw log and generated tables: `logs/stage2_two_stage_prefetch_2026-10-10_20-54-10-159.csv` and `results/stage2_two_stage_prefetch_2026-10-10_20-54-10-159_tables.md`.

Among rows passing the 8% SD gate, V2 was supported as faster than V0 for no-prefetch at 4M/1T, 4M/6T, 8M/4T, and 8M/6T. Two-stage comparisons showed no detectable difference in most stable rows; V2 was supported as faster at 8M/4T for k=4 and k=16, and 8M/6T for k=4. Rows exceeding 8% SD are marked rerun-required in the generated table and are not used for these claims.

## Task 3 And Later

The preregistered prediction is recorded in the Task 3 harness: V2 beats V0 by at least 1.3x on single-chain random walk at 4M+; falsified if the paired ratio is at most 1.1x. Outcome: **not run**. On launch, the machine-state probe reported battery power and the harness refused before graph generation or measurement. The prediction is neither supported nor falsified.

Task 4 reruns of rows with SD above 8%, Task 5 parallel-scaling investigation, and final ten-session latency results are not run. No V4/V5 work was started.