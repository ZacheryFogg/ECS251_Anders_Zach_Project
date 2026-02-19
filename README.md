ECS 251 Project
Anders Museth & Zachery Fogg

## Files

    scripts/run_experiment.py    - CLI entry point. Loads a YAML config, runs training benchmarks.
    configs/training_nsight.yaml - Example config. Defines a grid of experiments (model, batch_size, num_workers, pin_memory).
    benchmarks/training.py       - Training loop on MNIST with NVTX annotations for Nsight profiling.
    infra/config.py              - ExperimentConfig dataclass + YAML loader that expands matrix configs.
    infra/models.py              - pytorch model definitions
    infra/nsight.py              - runs a benchmark under `nsys profile` as a subprocess, one trace per config.

## Usage

Run all configs from a YAML file:

    python scripts/run_experiment.py --config configs/training_nsight.yaml

Run with per-config Nsight traces (so not one big trace for all experiments):

    python scripts/run_experiment.py --config configs/training_nsight.yaml --nsight

Profile the whole process as one Nsight trace:

    nsys profile -o results/nsight/my_trace --trace=cuda,osrt,nvtx -- python scripts/run_experiment.py --config configs/training_nsight.yaml
