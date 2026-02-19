#!/usr/bin/env python3
""" Entry point for running benchmark experiments."""

import argparse
import logging
import sys
from pathlib import Path

# module importing stuff 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.config import load_configs
from infra.nsight import profile_config
from benchmarks.training import run as run_training


def main():
    parser = argparse.ArgumentParser(description="Run ECS 251 benchmark experiments.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--nsight", action="store_true", help="Enable Nsight for all configs.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    # Make logger 
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Expand config that defines mutliple experiments into multiple configs
    configs = load_configs(args.config)
    
    if args.nsight:
        for cfg in configs:
            cfg.nsight = True

    logger.info(f"Loaded {len(configs)} configs")

    # Loop through experiments 
    for cfg in configs:
        logger.info(f"Running: model={cfg.model} | bs={cfg.batch_size} | workers={cfg.num_workers} | pin={cfg.pin_memory}")

        # If using nsight, we call some special wrapper logic in nsight module that will do some stuff then call training.run
        if cfg.nsight:
            trace, _ = profile_config(cfg)
            if trace:
                logger.info(f"Nsight trace: {trace}")
        # If no nsight, then just call training.run directly
        else:
            run_training(cfg)

    print(f"\nCompleted {len(configs)} experiments.")


if __name__ == "__main__":
    main()
