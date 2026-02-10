#!/usr/bin/env python3
"""Run a single benchmark config from a JSON file. Used as the subprocess target for Nsight profiling."""

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.config import ExperimentConfig, _make_config
from infra.results import save_results
from infra.runner import _get_benchmark_fn
from infra.system_info import collect_system_info


def main():
    if len(sys.argv) < 2:
        print("Usage: run_single.py <config.json> [--output <csv_path>]")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = sys.argv[1]
    with open(config_path) as f:
        config = _make_config(json.load(f))

    output = None
    if "--output" in sys.argv:
        output = sys.argv[sys.argv.index("--output") + 1]

    system_info = collect_system_info()
    benchmark_fn = _get_benchmark_fn(config.experiment)
    rows = benchmark_fn(config, system_info)

    if output:
        save_results(rows, output)

    # Also print JSON to stdout so the parent process can capture it
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
