"""Parse an nsight API trace TSV and plot per-call-type stats.

Usage:
    python3 scripts/plot_nsight.py --input results/nsight_summary.txt
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def parse_duration_us(duration_str: str) -> float:
    """Convert a duration string like '2.305 ms' to microseconds."""
    value, unit = duration_str.strip().split()
    value = float(value)
    if unit == "ns":
        return value / 1000
    if unit == "μs":
        return value
    if unit == "ms":
        return value * 1000
    if unit == "s":
        return value * 1_000_000
    return value 


def parse_nsight(path: str) -> dict[str, dict]:
    """Group rows by CUDA call name and accumulate total time and call count."""
    totals = defaultdict(lambda: {"total_us": 0.0, "count": 0})

    lines = Path(path).read_text().splitlines()
    for line in lines[1:]:  # skip header
        parts = line.split("\t")
        duration_str = parts[1].strip()
        name = parts[2].strip()
        us = parse_duration_us(duration_str)
        totals[name]["total_us"] += us
        totals[name]["count"] += 1

    return totals


def plot(totals: dict, out_dir: Path) -> None:
    TOP = 15
    sorted_names = sorted(totals, key=lambda n: totals[n]["total_us"], reverse=True)[:TOP]
    sorted_names = sorted_names[::-1]  # flip so largest is at top of horizontal chart

    names     = sorted_names
    total_ms  = [totals[n]["total_us"] / 1000 for n in names]
    counts    = [totals[n]["count"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Nsight CUDA API call summary", fontsize=14)

    axes[0].barh(names, total_ms, color="steelblue")
    axes[0].set_xlabel("Total time (ms)")
    axes[0].set_title("Total time per call type")

    axes[1].barh(names, counts, color="darkorange")
    axes[1].set_xlabel("Number of calls")
    axes[1].set_title("Call count per call type")

    # axes[2].barh(names, avg_us, color="seagreen")
    # axes[2].set_xlabel("Avg duration per call (μs)")
    # axes[2].set_title("Avg latency per call type")

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "nsight_calls.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Nsight CUDA API trace summary.")
    parser.add_argument("--input", required=True, help="Path to nsight TSV file.")
    parser.add_argument("--out-dir", default="results/figures", help="Output directory for the plot.")
    args = parser.parse_args()

    totals = parse_nsight(args.input)
    plot(totals, Path(args.out_dir))


if __name__ == "__main__":
    main()