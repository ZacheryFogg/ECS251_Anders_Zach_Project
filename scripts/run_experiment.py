#!/usr/bin/env python3
"""CLI entry point for running benchmark experiments."""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.config import load_configs
from infra.runner import run_and_save
from infra.system_info import collect_system_info


def main():
    parser = argparse.ArgumentParser(description="Run ECS 251 benchmark experiments.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--output", default=None, help="Override CSV output path.")
    parser.add_argument("--dry-run", action="store_true", help="Show grid without running.")
    parser.add_argument("--nsight", action="store_true", help="Enable Nsight for all configs.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    configs = load_configs(args.config)
    if args.nsight:
        for cfg in configs:
            cfg.nsight = True

    total_runs = sum(c.repetitions for c in configs)
    logger.info(f"Loaded {len(configs)} configs ({total_runs} total runs)")

    if args.dry_run:
        print(f"\nDRY RUN: {len(configs)} configs, {total_runs} total runs\n")
        for i, cfg in enumerate(configs, 1):
            print(f"  [{i:3d}] {cfg.label()}")
        print(f"\nSystem: {collect_system_info().summary()}")
        return

    df = run_and_save(configs, csv_path=args.output, progress=not args.no_progress)

    print(f"\nCompleted: {len(df)} result rows")
    output_path = args.output or (configs[0].csv_path if configs else "")
    if output_path:
        print(f"Saved to:  {output_path}")

    if not df.empty and "wall_ms_mean" in df.columns:
        print("\nSummary (mean wall_ms per config):")
        cols = [c for c in ["model", "batch_size", "num_workers", "pin_memory"] if c in df.columns]
        if cols:
            print(df.groupby(cols)["wall_ms_mean"].mean().reset_index().sort_values("wall_ms_mean").to_string(index=False))


if __name__ == "__main__":
    main()
