#!/usr/bin/env python3
"""Generate Task 3 latency-workload tables from a raw CSV log."""

import argparse
import contextlib
import csv
import math
import shutil
from collections import defaultdict
from pathlib import Path

T_CRIT_95_DF2 = 4.303
VARIANTS = ("ExistingNativeCsr", "V0", "V2", "V3")


def mean(values):
    return sum(values) / len(values)


def sample_sd(values):
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1))


def median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def ci_half_width(values):
    return T_CRIT_95_DF2 * sample_sd(values) / math.sqrt(len(values))


def status_for_variant(ratio, ci):
    if ratio - ci > 1.0:
        return "supported: variant faster"
    if ratio + ci < 1.0:
        return "supported: Existing faster"
    return "no detectable difference"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    observations = defaultdict(list)
    checks = {}
    with args.csv_path.open(newline="", encoding="utf-8") as source:
        rows = csv.DictReader(line for line in source if not line.startswith("#"))
        for row in rows:
            if row["warmup"] == "true":
                continue
            key = (
                int(row["n"]),
                row["family"],
                row["order"],
                row["workload"],
                int(row["walkers"]),
                int(row["session"]),
                row["variant"],
            )
            observations[key].append(float(row["elapsed_ms"]))
            checks[key] = (row["checksum"], int(row["reached"]))
            assert row["ac_battery"] == "AC power", f"non-AC sample found: {key}"

    session_medians = {key: median(values) for key, values in observations.items()}
    configurations = sorted({key[:5] for key in session_medians})
    assert len(configurations) == 48, f"expected 48 workload configurations, found {len(configurations)}"
    for n, family, order, workload, walkers in configurations:
        for variant in VARIANTS:
            for session in range(3):
                key = (n, family, order, workload, walkers, session, variant)
                assert key in session_medians, f"missing session: {key}"
                assert len(observations[key]) == 3, f"expected 3 timed runs for {key}"
    for n, family, order, workload, walkers in configurations:
        for session in range(3):
            for variant in VARIANTS[1:]:
                base = checks[(n, family, order, workload, walkers, session, "ExistingNativeCsr")]
                other = checks[(n, family, order, workload, walkers, session, variant)]
                assert other == base, f"{variant} correctness output differs for {(n, order, workload, walkers, session)}"

    graph_telemetry = {}
    if args.telemetry:
        with args.telemetry.open(newline="", encoding="utf-8") as source:
            telemetry_rows = csv.DictReader(line for line in source if not line.startswith("#"))
            for row in telemetry_rows:
                graph_telemetry[(int(row["n"]), row["family"])] = (
                    int(row["edges"]), int(row["overflow_nodes"]), float(row["p_o"])
                )

    output = args.output or Path("results") / f"{args.csv_path.stem}_tables.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_copy = output.parent / args.csv_path.name
    if args.csv_path.resolve() != raw_copy.resolve():
        shutil.copyfile(args.csv_path, raw_copy)

    with output.open("w", encoding="utf-8", newline="\n") as report:
        with contextlib.redirect_stdout(report):
            print(f"Source CSV: `{args.csv_path.as_posix()}`\n")
            if args.telemetry:
                print(f"Graph telemetry CSV: `{args.telemetry.as_posix()}`\n")
            print("Pilot: 3 sessions, one warmup, median of 3 timed runs; t critical=4.303 (df=2).")
            print(f"Workload configurations: {len(configurations)}; paired layout ratios vs Existing Native CSR: {len(configurations) * 3}.")
            print("About 1 in 20 nominal 95% CIs can exclude 1 by chance under a no-effect model; these many comparisons are not multiplicity-adjusted.")
            print("\n| N | Graph | Edges | Nodes with degree >3 | p_o |")
            print("|---:|---|---:|---:|---:|")
            if graph_telemetry:
                for n, family in sorted(graph_telemetry):
                    edges, overflow_nodes, p_o = graph_telemetry[(n, family)]
                    print(f"| {n} | {family} | {edges} | {overflow_nodes} | {p_o:.6f} ({p_o * 100:.3f}%) |")
            else:
                for n, family in sorted({(n, family) for n, family, _, _, _ in configurations}):
                    print(f"| {n} | {family} | not in timing CSV | not in timing CSV | not recorded |")
            print("\n| N | Graph | Order | Workload | Walkers | Variant | Mean session median (ms) | Sample SD (%) | 95% CI (ms) |")
            print("|---:|---|---|---|---:|---|---:|---:|---:|")
            for n, family, order, workload, walkers in configurations:
                for variant in VARIANTS:
                    values = [session_medians[(n, family, order, workload, walkers, session, variant)] for session in range(3)]
                    average = mean(values)
                    print(
                        f"| {n} | {family} | {order} | {workload} | {walkers} | {variant} | {average:.6f} | "
                        f"{sample_sd(values) / average * 100:.2f}% | +/- {ci_half_width(values):.6f} |"
                    )

            print("\n| N | Graph | Order | Workload | Walkers | Comparison (Existing / variant) | Mean paired ratio | 95% CI | 10% bar | Status vs 1.0 |")
            print("|---:|---|---|---|---:|---|---:|---:|---|---|")
            for n, family, order, workload, walkers in configurations:
                for variant in ("V0", "V2", "V3"):
                    ratios = [
                        session_medians[(n, family, order, workload, walkers, session, "ExistingNativeCsr")]
                        / session_medians[(n, family, order, workload, walkers, session, variant)]
                        for session in range(3)
                    ]
                    existing_times = [session_medians[(n, family, order, workload, walkers, session, "ExistingNativeCsr")] for session in range(3)]
                    variant_times = [session_medians[(n, family, order, workload, walkers, session, variant)] for session in range(3)]
                    average = mean(ratios)
                    ci = ci_half_width(ratios)
                    sd_flag = max(sample_sd(existing_times) / mean(existing_times), sample_sd(variant_times) / mean(variant_times)) > 0.08
                    if sd_flag:
                        status = "unresolved: SD > 8%"
                    else:
                        status = status_for_variant(average, ci)
                    bar = "clears >=1.10x" if average - ci >= 1.10 and not sd_flag else "does not clear"
                    print(f"| {n} | {family} | {order} | {workload} | {walkers} | Existing / {variant} | {average:.4f}x | +/- {ci:.4f}x | {bar} | {status} |")

            print("\n## Preregistered Prediction")
            print("V2 beats CSR by >=1.3x on the single-chain random walk at >=4M nodes; falsified if the paired ratio is <=1.1x. The registered threshold is unchanged; Existing Native CSR is the control used here.")
            prediction_rows = []
            print("\n| N | Graph | Order | Existing / V2 ratio | 95% CI | Outcome |")
            print("|---:|---|---|---:|---:|---|")
            for n, family, order, workload, walkers in configurations:
                if n < 4_000_000 or workload != "single_chain":
                    continue
                ratios = [
                    session_medians[(n, family, order, workload, walkers, session, "ExistingNativeCsr")]
                    / session_medians[(n, family, order, workload, walkers, session, "V2")]
                    for session in range(3)
                ]
                average = mean(ratios)
                ci = ci_half_width(ratios)
                existing_times = [session_medians[(n, family, order, workload, walkers, session, "ExistingNativeCsr")] for session in range(3)]
                v2_times = [session_medians[(n, family, order, workload, walkers, session, "V2")] for session in range(3)]
                sd_flag = max(sample_sd(existing_times) / mean(existing_times), sample_sd(v2_times) / mean(v2_times)) > 0.08
                if sd_flag:
                    outcome = "unresolved: SD > 8%"
                elif average - ci >= 1.3:
                    outcome = "supported"
                elif average + ci <= 1.1:
                    outcome = "falsified"
                else:
                    outcome = "unresolved"
                prediction_rows.append(outcome)
                print(f"| {n} | {family} | {order} | {average:.4f}x | +/- {ci:.4f}x | {outcome} |")
            print("\nPrediction outcomes are reported per graph-size/order cell; no pooled overall outcome is substituted for those cells.")

            print("\n| N | Graph | Order | Variant | DFS reached nodes | Visit-order checksum |")
            print("|---:|---|---|---|---:|---:|")
            for n, family, order, workload, walkers in configurations:
                if workload != "iterative_dfs":
                    continue
                for variant in VARIANTS:
                    checksum, reached = checks[(n, family, order, workload, walkers, 0, variant)]
                    print(f"| {n} | {family} | {order} | {variant} | {reached} | {checksum} |")

    print(f"Generated table: {output}")


if __name__ == "__main__":
    main()