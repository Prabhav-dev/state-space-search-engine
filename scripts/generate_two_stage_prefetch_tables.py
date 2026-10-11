#!/usr/bin/env python3
"""Generate Task 2 summary tables from a stage2_two_stage_prefetch CSV."""

import argparse
import contextlib
import csv
import math
import shutil
from collections import defaultdict
from pathlib import Path

T_CRIT_95_DF9 = 2.262


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
    return T_CRIT_95_DF9 * sample_sd(values) / math.sqrt(len(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    observations = defaultdict(list)
    with args.csv_path.open(newline="", encoding="utf-8") as source:
        rows = csv.DictReader(line for line in source if not line.startswith("#"))
        for row in rows:
            if row["warmup"] == "true":
                continue
            key = (
                int(row["n"]),
                int(row["threads"]),
                row["prefetch"],
                int(row["session"]),
                row["variant"],
            )
            observations[key].append(float(row["elapsed_ms"]))

    session_medians = {key: median(values) for key, values in observations.items()}
    configurations = sorted({key[:3] for key in session_medians})
    assert len(configurations) == 24, f"expected 24 Task 2 configurations, found {len(configurations)}"
    for n, threads, prefetch in configurations:
        for variant in ("V0", "V2"):
            for session in range(10):
                key = (n, threads, prefetch, session, variant)
                assert key in session_medians, f"missing session: {key}"
                assert len(observations[key]) == 10, f"expected 10 timed runs for {key}"
    output = args.output or Path("results") / f"{args.csv_path.stem}_tables.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_copy = output.parent / args.csv_path.name
    if args.csv_path.resolve() != raw_copy.resolve():
        shutil.copyfile(args.csv_path, raw_copy)

    with output.open("w", encoding="utf-8", newline="\n") as report:
        with contextlib.redirect_stdout(report):
            print(f"Source CSV: `{args.csv_path.as_posix()}`\n")
            print(f"Configurations tested: {len(configurations)}. Existing Native CSR was not recorded in this CSV, so no V2/V3-to-Existing ratios can be reconstructed.")
            print("About 1 in 20 nominal 95% CIs can exclude 1 by chance under a no-effect model; no multiplicity correction was applied.")
            print("| N | Threads | Prefetch | Variant | Mean session median (ms) | Sample SD (%) | 95% CI (ms) |")
            print("|---:|---:|---|---|---:|---:|---:|")
            for n, threads, prefetch in configurations:
                for variant in ("V0", "V2"):
                    values = [session_medians[(n, threads, prefetch, session, variant)] for session in range(10)]
                    average = mean(values)
                    print(
                        f"| {n} | {threads}T | {prefetch} | {variant} | {average:.6f} | "
                        f"{sample_sd(values) / average * 100:.2f}% | +/- {ci_half_width(values):.6f} |"
                    )

            print("\n| N | Threads | Prefetch | Paired V0 / V2 ratio | 95% CI | 10% bar | Status vs 1.0 |")
            print("|---:|---:|---|---:|---:|---|---|")
            for n, threads, prefetch in configurations:
                ratios = [
                    session_medians[(n, threads, prefetch, session, "V0")]
                    / session_medians[(n, threads, prefetch, session, "V2")]
                    for session in range(10)
                ]
                average = mean(ratios)
                ci = ci_half_width(ratios)
                sd_v0_pct = sample_sd([
                    session_medians[(n, threads, prefetch, session, "V0")]
                    for session in range(10)
                ]) / mean([
                    session_medians[(n, threads, prefetch, session, "V0")]
                    for session in range(10)
                ]) * 100.0
                sd_v2_pct = sample_sd([
                    session_medians[(n, threads, prefetch, session, "V2")]
                    for session in range(10)
                ]) / mean([
                    session_medians[(n, threads, prefetch, session, "V2")]
                    for session in range(10)
                ]) * 100.0
                if max(sd_v0_pct, sd_v2_pct) > 8.0:
                    status = "rerun required: variant SD > 8%"
                elif average - ci <= 1.0 <= average + ci:
                    status = "no detectable difference"
                elif average - ci > 1.0:
                    status = "supported: V2 faster"
                else:
                    status = "supported: V0 faster"
                bar = "clears >=1.10x" if average - ci >= 1.10 and max(sd_v0_pct, sd_v2_pct) <= 8.0 else "does not clear"
                print(f"| {n} | {threads}T | {prefetch} | {average:.4f}x | +/- {ci:.4f}x | {bar} | {status} |")

    print(f"Generated table: {output}")


if __name__ == "__main__":
    main()