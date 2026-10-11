#!/usr/bin/env python3
"""Generate findings.md from Task 1-3 raw CSV logs."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

T_CRIT_DF9 = 2.262
T_CRIT_DF2 = 4.303
VARIANTS = ("ExistingNativeCsr", "V0", "V2", "V3")


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def mean(values):
    return sum(values) / len(values)


def sample_sd(values):
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1))


def ci_half_width(values, t_critical):
    return t_critical * sample_sd(values) / math.sqrt(len(values))


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(line for line in source if not line.startswith("#")))


def session_medians(rows, key_fields, variant_field="variant"):
    grouped = defaultdict(list)
    for row in rows:
        if row["warmup"].lower() == "true":
            continue
        key = tuple(row[field] for field in key_fields) + (row[variant_field],)
        grouped[key].append(float(row["elapsed_ms"]))
    return {key: median(values) for key, values in grouped.items()}


def ratio_summary(medians, config_key, numerator, denominator, sessions, t_critical):
    ratios = [
        medians[config_key + (str(session), numerator)]
        / medians[config_key + (str(session), denominator)]
        for session in range(sessions)
    ]
    average = mean(ratios)
    ci = ci_half_width(ratios, t_critical)
    return average, ci, ratios


def bar_status(ratio, ci):
    if ratio - ci >= 1.10:
        return "clears 1.10x"
    return "does not clear 1.10x"


def effect_status(ratio, ci):
    if ratio - ci > 1.0:
        return "supported: compared layout faster"
    if ratio + ci < 1.0:
        return "supported: Existing faster"
    return "no detectable difference"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-original", type=Path, default=Path("logs/stage1_control_2026-10-10_20-49-55-356.csv"))
    parser.add_argument("--stage1-rerun", type=Path, default=Path("logs/stage1_control_2026-10-10_21-43-05-559.csv"))
    parser.add_argument("--stage2", type=Path, default=Path("logs/stage2_two_stage_prefetch_2026-10-10_20-54-10-159.csv"))
    parser.add_argument("--stage3", type=Path)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--output", type=Path, default=Path("findings.md"))
    args = parser.parse_args()

    output = [
        "# Findings Snapshot",
        "",
        "Run date: 2026-10-10. Benchmark source commit: `a2fd78f99b148b1d29746ece6d3441ff9bce2342`; experiment harness changes were uncommitted during these reruns, as recorded by the dirty-worktree metadata. Release profile only. Commands: `cargo run --release --example stage1_control_study`; `cargo run --release --example stage1_control_study -- --only-cell 1000000 6`; `cargo run --release --example stage2_two_stage_prefetch`; `cargo run --release --example stage3_latency_workloads`.",
        "",
        "All ratios are paired within interleaved sessions and are numerator time / denominator time: above 1 means the denominator was faster, below 1 means the numerator was faster. A comparison clears the 10% bar only when the lower 95% CI is at least 1.10x. No hardware-counter causes are claimed.",
        "",
        "Nominal 95% CIs have about a 1-in-20 chance to exclude 1 under a true no-effect model. The comparisons below are not multiplicity-adjusted; expected chance exclusions are listed per study, not treated as findings.",
        "Across the planned 192 nominal ratio intervals (24 Task 1, 24 Task 2, 144 Task 3), this corresponds to about 9.6 chance exclusions under a global null if intervals were independent; the paired layout comparisons are correlated, so this is only a reference, not an adjusted expectation.",
        "",
        "## Task 1: Existing CSR Control",
        "",
        "The historical run tested 6 size/thread configurations (18 layout/control ratios). The single-cell rerun tested two additional conditions at 1M/6T (`test_before_atomic=false` and `true`; 6 ratios). Total: 8 run-configurations and 24 paired ratios. The 1M/6T rerun is a separate interleaved session set; its estimate change is descriptive only and is not a valid cross-run performance comparison.",
        "",
        f"Historical raw CSV: `{args.stage1_original.as_posix()}`. Rerun raw CSV: `{args.stage1_rerun.as_posix()}`. Full generated timing tables: [historical](results/stage1_control_2026-10-10_20-49-55-356_tables.md) and [rerun](results/stage1_control_2026-10-10_21-43-05-559_tables.md).",
        "",
        "| Run | N | Threads | Test-before-atomic | Existing/V0 (95% CI) | Existing/V2 (95% CI) | Existing/V3 (95% CI) | 10% bar cleared |",
        "|---|---:|---:|---|---:|---:|---:|---|",
    ]

    stage1_cases = []
    for run_name, path in (("logged", args.stage1_original), ("rerun", args.stage1_rerun)):
        rows = load_rows(path)
        medians = session_medians(rows, ("n", "threads", "test_before_atomic", "session"))
        configs = sorted({key[:3] for key in medians})
        sessions = max(int(key[3]) for key in medians) + 1
        for n, threads, test_before in configs:
            config = (n, threads, test_before)
            summaries = []
            for variant in ("V0", "V2", "V3"):
                ratio, ci, _ = ratio_summary(medians, config, "ExistingNativeCsr", variant, sessions, T_CRIT_DF9)
                summaries.append((variant, ratio, ci))
            clears = [variant for variant, ratio, ci in summaries if ratio - ci >= 1.10]
            test_label = str(test_before).lower()
            values = [f"{ratio:.4f} +/- {ci:.4f}x" for _, ratio, ci in summaries]
            output.append(
                f"| {run_name} | {int(n):,} | {threads}T | {test_label} | {values[0]} | {values[1]} | {values[2]} | {', '.join(clears) if clears else 'none'} |"
            )
            stage1_cases.append((run_name, int(n), threads, test_label, summaries))

    output.extend([
        "",
        "For 1M/6T with `test_before_atomic=false`, logged then rerun Existing/variant ratios were: V0 0.9838 +/- 0.0134x then 0.9886 +/- 0.0202x (shift +0.0048x); V2 0.9057 +/- 0.0165x then 1.0008 +/- 0.0205x (shift +0.0951x); V3 0.7789 +/- 0.0150x then 0.7090 +/- 0.0141x (shift -0.0699x). These cross-run shifts are descriptive only, not valid performance comparisons under the interleaved-session rule. No rerun ratio clears 1.10x. The `true` rerun gave Existing/V0=1.0194 +/- 0.0493x, Existing/V2=0.9241 +/- 0.0378x, and Existing/V3=0.6646 +/- 0.0246x; none clears 1.10x.",
        "",
        "All 24 Task 1 ratios fail to clear the 1.10x bar. The historical 500K/4T row has V3 session SD 9.63% and remains rerun-required under Task 4. The prior Task 1 session headers recorded AC power, but did not record a fresh power check per session; the rerun did.",
        "",
        "## Task 2: Historical V0/V2 Prefetch",
        "",
        "The study tested 24 configurations (2 sizes x 3 thread counts x 4 prefetch modes) and produced 24 paired V0/V2 ratios. Existing Native CSR was not included in this raw CSV, so ratios against Existing cannot be reconstructed; no control ratio is inferred. About 1.2 nominal CIs would be expected to exclude 1 by chance if all 24 null comparisons had no effect.",
        "",
        f"Raw CSV: `{args.stage2.as_posix()}`. Full table: [Task 2 ratios and CIs](results/stage2_two_stage_prefetch_2026-10-10_20-54-10-159_tables.md).",
        "",
        "| N | Threads | Prefetch | V0/V2 paired ratio (95% CI) | 10% bar | Status |",
        "|---:|---:|---|---:|---|---|",
    ])

    rows = load_rows(args.stage2)
    medians = session_medians(rows, ("n", "threads", "prefetch", "session"))
    configs = sorted({key[:3] for key in medians})
    for n, threads, prefetch in configs:
        ratio, ci, _ = ratio_summary(medians, (n, threads, prefetch), "V0", "V2", 10, T_CRIT_DF9)
        v0_times = [medians[(n, threads, prefetch, str(s), "V0")] for s in range(10)]
        v2_times = [medians[(n, threads, prefetch, str(s), "V2")] for s in range(10)]
        high_sd = max(sample_sd(v0_times) / mean(v0_times), sample_sd(v2_times) / mean(v2_times)) > 0.08
        if high_sd:
            status = "unresolved: SD > 8%"
        elif ratio - ci > 1.0:
            status = "supported: V2 faster"
        elif ratio + ci < 1.0:
            status = "supported: V0 faster"
        else:
            status = "no detectable difference"
        bar = "rerun required" if high_sd else bar_status(ratio, ci)
        output.append(f"| {int(n):,} | {threads}T | {prefetch} | {ratio:.4f} +/- {ci:.4f}x | {bar} | {status} |")

    output.extend([
        "",
        "The Task 2 source log recorded an AC-power header but did not check or log power at each session start. Rows with variant SD above 8% remain rerun-required and are not treated as support.",
        "",
        "## Task 3: Latency Workloads",
        "",
        "The unchanged prediction is: V2 beats CSR by >=1.3x on a single-chain random walk at >=4M nodes; falsified if the paired ratio is <=1.1x. Existing Native CSR is the control for these comparisons. The study contains 48 workload configurations (12 graph-size/family/order combinations x 4 workload/walker settings) and 144 paired layout/control ratios. About 7.2 nominal CIs could exclude 1 by chance under a global no-effect model; the intervals are not multiplicity-adjusted.",
        "",
        f"Raw CSV: `{args.stage3.as_posix() if args.stage3 else 'not run'}`. Graph telemetry CSV: `{args.telemetry.as_posix() if args.telemetry else 'not run'}`. Full ratios, SDs, CIs, power fields, checksums, and p_o values: [Task 3 generated report](results/{args.stage3.stem}_tables.md) if present.",
        "",
        "| N | Graph | Order | Workload | Walkers | Existing/V0 (95% CI) | Existing/V2 (95% CI) | Existing/V3 (95% CI) | 10% bar cleared |",
        "|---:|---|---|---|---:|---:|---:|---:|---|",
    ])

    if args.stage3 and args.stage3.exists():
        rows = load_rows(args.stage3)
        medians = session_medians(rows, ("n", "family", "order", "workload", "walkers", "session"))
        configs = sorted({key[:5] for key in medians})
        assert len(configs) == 48, f"expected 48 Task 3 workload configurations, found {len(configs)}"
        for n, family, order, workload, walkers in configs:
            summaries = []
            for variant in ("V0", "V2", "V3"):
                ratio, ci, _ = ratio_summary(medians, (n, family, order, workload, walkers), "ExistingNativeCsr", variant, 3, T_CRIT_DF2)
                summaries.append((variant, ratio, ci))
            clears = [variant for variant, ratio, ci in summaries if ratio - ci >= 1.10]
            values = [f"{ratio:.4f} +/- {ci:.4f}x" for _, ratio, ci in summaries]
            output.append(
                f"| {int(n):,} | {family} | {order} | {workload} | {walkers} | {values[0]} | {values[1]} | {values[2]} | {', '.join(clears) if clears else 'none'} |"
            )

        output.extend([
            "",
            "### Registered Prediction Outcomes",
            "",
            "| N | Graph | Order | Existing/V2 ratio (95% CI) | Outcome |",
            "|---:|---|---|---:|---|",
        ])

        predictions = []
        for n, family, order, workload, walkers in configs:
            if int(n) < 4_000_000 or workload != "single_chain":
                continue
            ratio, ci, _ = ratio_summary(medians, (n, family, order, workload, walkers), "ExistingNativeCsr", "V2", 3, T_CRIT_DF2)
            existing_times = [medians[(n, family, order, workload, walkers, str(s), "ExistingNativeCsr")] for s in range(3)]
            v2_times = [medians[(n, family, order, workload, walkers, str(s), "V2")] for s in range(3)]
            high_sd = max(sample_sd(existing_times) / mean(existing_times), sample_sd(v2_times) / mean(v2_times)) > 0.08
            if high_sd:
                outcome = "unresolved: SD > 8%"
            elif ratio - ci >= 1.3:
                outcome = "supported"
            elif ratio + ci <= 1.1:
                outcome = "falsified"
            else:
                outcome = "unresolved"
            predictions.append(outcome)
            output.append(f"| {int(n):,} | {family} | {order} | {ratio:.4f} +/- {ci:.4f}x | {outcome} |")
        output.extend([
            "",
            "The prediction is reported for each qualifying graph-size/order cell, not pooled across graph families. Generated low-degree graphs have out-degree 2 or 3 by construction (`p_o=0`); R-MAT `p_o` values are printed per graph by the harness.",
            "",
            "## Threats And Scope",
            "",
            "The studies use one Windows laptop with no hardware counters. Thermal/boost variation remains possible despite interleaving; the Task 3 pilot has only 3 sessions and t intervals with 2 degrees of freedom. Task 2 lacks an Existing Native CSR row and per-session power records. The Task 1 original run also predates per-session power checks. No roadNet-CA input was supplied, so the added comparison uses a deterministic random degree-2/3 graph. No cause is asserted from timing alone. Task 4/5 are not run; V4/V5 were not started.",
        ])
        if args.telemetry and args.telemetry.exists():
            telemetry_rows = load_rows(args.telemetry)
            output.extend([
                "",
                "## Graph Telemetry",
                "",
                "| N | Graph | Edges | Overflow nodes | p_o |",
                "|---:|---|---:|---:|---:|",
            ])
            for row in telemetry_rows:
                output.append(
                    f"| {int(row['n']):,} | {row['family']} | {int(row['edges'])} | {int(row['overflow_nodes'])} | {float(row['p_o']):.6f} ({float(row['p_o']) * 100:.3f}%) |"
                )

    args.output.write_text("\n".join(output) + "\n", encoding="utf-8", newline="\n")
    print(f"Generated findings: {args.output}")


if __name__ == "__main__":
    main()
