"""Parse a strace -cw summary file and plot syscall stats.

Usage:
    # 1. Capture strace output
    strace -cw python3 benchmarks/cpu_to_gpu.py 2> results/strace_summary.txt

    # 2. Plot it
    python3 scripts/plot_strace.py --input results/strace_summary.txt
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def parse_strace(path: str) -> list[dict]:
    """Parse the strace -c summary table into a list of dicts."""
    rows = []
    in_table = False
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("% time"):
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("---") or line.startswith("% time"):
            continue
        # columns: pct_time seconds usecs/call calls [errors] syscall
        parts = line.split()
        if len(parts) == 6:
            pct, seconds, usecs, calls, errors, syscall = parts
        elif len(parts) == 5:
            pct, seconds, usecs, calls, syscall = parts
            errors = "0"
        else:
            continue
        if syscall != "total":
            rows.append({
                "pct_time":   float(pct),
                "seconds":    float(seconds),
                "usecs_call": int(usecs),
                "calls":      int(calls),
                "errors":     int(errors),
                "syscall":    syscall,
            })

    return rows


def plot(rows: list[dict], out_dir: Path) -> None:
    TOP = 15
    rows = sorted(rows, key=lambda r: r["pct_time"], reverse=True)[:TOP]
    labels = [r["syscall"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("strace syscall summary", fontsize=14)

    # --- total time in ms ---
    ms = [r["seconds"] * 1000 for r in rows]
    axes[0].barh(labels[::-1], ms[::-1], color="steelblue")
    axes[0].set_xlabel("Total time (ms)")
    axes[0].set_title("Total time per syscall")

    # --- call count ---
    calls = [r["calls"] for r in rows]
    axes[1].barh(labels[::-1], calls[::-1], color="darkorange")
    axes[1].set_xlabel("Number of calls")
    axes[1].set_title("Call count per syscall")

    # # --- avg usecs/call ---
    # usecs = [r["usecs_call"] for r in rows]
    # axes[2].barh(labels[::-1], usecs[::-1], color="seagreen")
    # axes[2].set_xlabel("Average time per call (µs)")
    # axes[2].set_title("Avg latency per syscall")

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "strace_syscalls.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot strace -cw summary.")
    parser.add_argument("--input", required=True, help="Path to strace output file.")
    parser.add_argument("--out-dir", default="results/figures", help="Output directory for the plot.")
    args = parser.parse_args()

    rows = parse_strace(args.input)
    
    plot(rows, Path(args.out_dir))


if __name__ == "__main__":
    main()