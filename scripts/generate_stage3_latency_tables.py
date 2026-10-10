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
VARIANTS = ("V0", "V2", "V3")


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


def status_for_v2(ratio, ci):
    if ratio - ci > 1.0:
        return "supported"
    if ratio + ci < 1.0:
        return "not supported"
    return "no detectable difference"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
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
                row["order"],
                row["workload"],
                int(row["walkers"]),
                int(row["session"]),
                row["variant"],
            )
            observations[key].append(float(row["elapsed_ms"]))
            checks[key] = (row["checksum"], int(row["reached"]))

    session_medians = {key: median(values) for key, values in observations.items()}
    configurations = sorted({key[:4] for key in session_medians})
    for n, order, workload, walkers in configurations:
        for variant in VARIANTS:
            for session in range(3):
                key = (n, order, workload, walkers, session, variant)
                assert key in session_medians, f"missing session: {key}"
                assert len(observations[key]) == 3, f"expected 3 timed runs for {key}"
    for n, order, workload, walkers in configurations:
        for session in range(3):
            for variant in VARIANTS[1:]:
                base = checks[(n, order, workload, walkers, session, "V0")]
                other = checks[(n, order, workload, walkers, session, variant)]
                assert other == base, f"{variant} correctness output differs for {(n, order, workload, walkers, session)}"

    output = args.output or Path("results") / f"{args.csv_path.stem}_tables.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_copy = output.parent / args.csv_path.name
    if args.csv_path.resolve() != raw_copy.resolve():
        shutil.copyfile(args.csv_path, raw_copy)

    with output.open("w", encoding="utf-8", newline="\n") as report:
        with contextlib.redirect_stdout(report):
            print(f"Source CSV: `{args.csv_path.as_posix()}`\n")
            print("Pilot: 3 sessions, one warmup, median of 3 timed runs; t critical=4.303 (df=2).")
            print("\n| N | Order | Workload | Walkers | Variant | Mean session median (ms) | Sample SD (%) | 95% CI (ms) |")
            print("|---:|---|---|---:|---|---:|---:|---:|")
            for n, order, workload, walkers in configurations:
                for variant in VARIANTS:
                    values = [session_medians[(n, order, workload, walkers, session, variant)] for session in range(3)]
                    average = mean(values)
                    print(
                        f"| {n} | {order} | {workload} | {walkers} | {variant} | {average:.6f} | "
                        f"{sample_sd(values) / average * 100:.2f}% | +/- {ci_half_width(values):.6f} |"
                    )

            print("\n| N | Order | Workload | Walkers | Paired V0 / V2 ratio | 95% CI | V2 faster status |")
            print("|---:|---|---|---:|---:|---:|---|")
            for n, order, workload, walkers in configurations:
                ratios = [
                    session_medians[(n, order, workload, walkers, session, "V0")]
                    / session_medians[(n, order, workload, walkers, session, "V2")]
                    for session in range(3)
                ]
                average = mean(ratios)
                ci = ci_half_width(ratios)
                print(f"| {n} | {order} | {workload} | {walkers} | {average:.4f}x | +/- {ci:.4f}x | {status_for_v2(average, ci)} |")

            print("\n## Preregistered Prediction")
            print("V2 beats V0 by >=1.3x on single-chain random walk at >=4M nodes; falsified if the paired ratio is <=1.1x.")
            prediction_rows = []
            print("\n| N | Order | V0 / V2 ratio | 95% CI | Outcome |")
            print("|---:|---|---:|---:|---|")
            for n, order, workload, walkers in configurations:
                if n < 4_000_000 or workload != "single_chain":
                    continue
                ratios = [
                    session_medians[(n, order, workload, walkers, session, "V0")]
                    / session_medians[(n, order, workload, walkers, session, "V2")]
                    for session in range(3)
                ]
                average = mean(ratios)
                ci = ci_half_width(ratios)
                if average - ci >= 1.3:
                    outcome = "supported"
                elif average + ci <= 1.1:
                    outcome = "falsified"
                else:
                    outcome = "unresolved"
                prediction_rows.append(outcome)
                print(f"| {n} | {order} | {average:.4f}x | +/- {ci:.4f}x | {outcome} |")
            if prediction_rows and all(outcome == "falsified" for outcome in prediction_rows):
                overall = "falsified"
            elif "supported" in prediction_rows:
                overall = "supported"
            else:
                overall = "unresolved"
            print(f"\nOverall pilot outcome: **{overall}** (n=3 sessions; Task 4 final protocol not yet run).")

            print("\n| N | Order | Variant | DFS reached nodes | Visit-order checksum |")
            print("|---:|---|---|---:|---:|")
            for n, order, workload, walkers in configurations:
                if workload != "iterative_dfs":
                    continue
                for variant in VARIANTS:
                    checksum, reached = checks[(n, order, workload, walkers, 0, variant)]
                    print(f"| {n} | {order} | {variant} | {reached} | {checksum} |")

    print(f"Generated table: {output}")


if __name__ == "__main__":
    main()