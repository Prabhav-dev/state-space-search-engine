#!/usr/bin/env python3
"""Generate control-study Markdown tables directly from a stage1_control CSV."""

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

    with args.csv_path.open(newline="", encoding="utf-8") as source:
        rows = csv.DictReader(line for line in source if not line.startswith("#"))
        sessions = defaultdict(list)
        reached = {}
        for row in rows:
            if row["warmup"] == "true":
                continue
            key = (int(row["n"]), int(row["threads"]), int(row["session"]), row["variant"])
            sessions[key].append(float(row["elapsed_ms"]))
            reached[key[:3] + (row["variant"],)] = (int(row["edges"]), int(row["reached"]))

    session_medians = {key: median(values) for key, values in sessions.items()}
    configurations = sorted({key[:2] for key in session_medians})
    destination = args.output or Path("results") / f"{args.csv_path.stem}_tables.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw_copy = destination.parent / args.csv_path.name
    if args.csv_path.resolve() != raw_copy.resolve():
        shutil.copyfile(args.csv_path, raw_copy)
    with destination.open("w", encoding="utf-8", newline="\n") as report:
        with contextlib.redirect_stdout(report):
            print(f"Source CSV: `{args.csv_path.as_posix()}`\n")
            print("| N | Threads | Variant | Mean session median (ms) | Sample SD (%) | 95% CI (ms) |")
            print("|---:|---:|---|---:|---:|---:|")
            for n, threads in configurations:
                for variant in ("ExistingNativeCsr", "V0", "V2", "V3"):
                    values = [session_medians[(n, threads, session, variant)] for session in range(10)]
                    average = mean(values)
                    sd = sample_sd(values)
                    ci = ci_half_width(values)
                    print(f"| {n} | {threads}T | {variant} | {average:.6f} | {sd / average * 100:.2f}% | +/- {ci:.6f} |")

            print("\n| N | Threads | Comparison | Mean paired overhead (%) | 95% CI (percentage points) | Status |")
            print("|---:|---:|---|---:|---:|---|")
            for n, threads in configurations:
                overhead = []
                for session in range(10):
                    existing = session_medians[(n, threads, session, "ExistingNativeCsr")]
                    v0 = session_medians[(n, threads, session, "V0")]
                    overhead.append((v0 / existing - 1.0) * 100.0)
                average = mean(overhead)
                ci = ci_half_width(overhead)
                if average - ci <= 0.0 <= average + ci:
                    status = "no detectable difference"
                elif average - ci > 0.0:
                    status = "supported: V0 overhead"
                else:
                    status = "not supported: V0 overhead"
                print(f"| {n} | {threads}T | V0 / Existing | {average:+.2f}% | +/- {ci:.2f} | {status} |")

            print("\n| N | Threads | Comparison | Mean paired speedup | 95% CI | Status |")
            print("|---:|---:|---|---:|---:|---|")
            for n, threads in configurations:
                for comparison, numerator, denominator in (
                    ("V0 / V2", "V0", "V2"),
                    ("V0 / V3", "V0", "V3"),
                    ("V2 / V3", "V2", "V3"),
                ):
                    ratios = [
                        session_medians[(n, threads, session, numerator)]
                        / session_medians[(n, threads, session, denominator)]
                        for session in range(10)
                    ]
                    average = mean(ratios)
                    ci = ci_half_width(ratios)
                    if average - ci <= 1.0 <= average + ci:
                        status = "no detectable difference"
                    elif average - ci > 1.0:
                        status = f"supported: {denominator} faster"
                    else:
                        status = f"supported: {numerator} faster"
                    print(f"| {n} | {threads}T | {comparison} | {average:.4f}x | +/- {ci:.4f}x | {status} |")

            print("\n| N | Threads | Variant | Edges | Reached nodes |")
            print("|---:|---:|---|---:|---:|")
            for n, threads in configurations:
                for variant in ("ExistingNativeCsr", "V0", "V2", "V3"):
                    edges, reached_count = reached[(n, threads, 0, variant)]
                    print(f"| {n} | {threads}T | {variant} | {edges} | {reached_count} |")
    print(f"Generated table: {destination}")


if __name__ == "__main__":
    main()