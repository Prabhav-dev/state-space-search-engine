#!/usr/bin/env python3
"""
Automated Table Generator script (`scripts/generate_tables.py`).
Parses raw `.log` output files (raw_session_1.log, raw_session_2.log, raw_session_3.log)
directly into Markdown and LaTeX tables to guarantee 100% reproducibility.
"""

import os
import re
import math
import sys

def parse_raw_session_log(filepath):
    """Parses a raw session log file and returns medians per graph size."""
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.", file=sys.stderr)
        sys.exit(1)

    size_runs = {}
    size_medians = {}

    with open(filepath, "r") as f:
        for line in f:
            # Match run lines: Size=100000 | Run=3 | Edges=159398 | Time=1.1958 ms
            m_run = re.search(r"Size=(\d+)\s*\|\s*Run=(\d+)\s*\|\s*Edges=(\d+)\s*\|\s*Time=([\d\.]+)\s*ms", line)
            if m_run:
                size = int(m_run.group(1))
                run_idx = int(m_run.group(2))
                time_ms = float(m_run.group(4))
                if run_idx >= 3:  # Exclude warmup runs 0..2
                    size_runs.setdefault(size, []).append(time_ms)
                continue

            # Match median summary line: Size=100000 | SESSION 1 MEDIAN = 1.1548 ms
            m_med = re.search(r"Size=(\d+)\s*\|\s*SESSION\s+\d+\s+MEDIAN\s*=\s*([\d\.]+)\s*ms", line)
            if m_med:
                size = int(m_med.group(1))
                med_ms = float(m_med.group(2))
                size_medians[size] = med_ms

    # If median line wasn't matched directly, compute median from runs
    for size, runs in size_runs.items():
        if size not in size_medians and runs:
            sorted_runs = sorted(runs)
            size_medians[size] = sorted_runs[len(sorted_runs) // 2]

    return size_medians

fn_sample_sd = lambda vals, m: math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs = [
        os.path.join(root_dir, "raw_session_1.log"),
        os.path.join(root_dir, "raw_session_2.log"),
        os.path.join(root_dir, "raw_session_3.log"),
    ]

    session_data = [parse_raw_session_log(log) for log in logs]
    sizes = sorted(list(session_data[0].keys()))

    print("=== GENERATED MARKDOWN TABLE (Table 4.1a: Multi-Session Benchmark Timing) ===")
    print("| Node Count | Session 1 (ms) | Session 2 (ms) | Session 3 (ms) | Mean (ms) | Sample SD (%) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")

    rows = []
    for sz in sizes:
        s1 = session_data[0].get(sz, 0.0)
        s2 = session_data[1].get(sz, 0.0)
        s3 = session_data[2].get(sz, 0.0)
        vals = [s1, s2, s3]
        mean_val = sum(vals) / len(vals)
        sd_val = fn_sample_sd(vals, mean_val)
        sd_pct = (sd_val / mean_val) * 100.0 if mean_val > 0 else 0.0

        label = f"{sz//1000}K" if sz < 1000000 else f"{sz//1000000}M"
        print(f"| **{label}** ({sz:,}) | {s1:.3f} | {s2:.3f} | {s3:.3f} | {mean_val:.3f} | {sd_pct:.2f}% |")
        rows.append((label, sz, s1, s2, s3, mean_val, sd_pct))

    print("\n=== GENERATED LATEX TABLE ===")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{|r|r|r|r|r|r|}")
    print("\\hline")
    print("\\textbf{Nodes} & \\textbf{Session 1 (ms)} & \\textbf{Session 2 (ms)} & \\textbf{Session 3 (ms)} & \\textbf{Mean (ms)} & \\textbf{Sample SD (\\%)} \\\\")
    print("\\hline")
    for label, sz, s1, s2, s3, mean_val, sd_pct in rows:
        print(f"{label} & {s1:.3f} & {s2:.3f} & {s3:.3f} & {mean_val:.3f} & {sd_pct:.2f}\\% \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Multi-Session Timing Variance Across 3 Isolated Benchmark Runs}")
    print("\\end{table}")

if __name__ == "__main__":
    main()
