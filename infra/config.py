import itertools
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    """
    One fully-specified experiment run.
    Defines defaults
    """

    experiment: str = "unnamed"

    model: str = "small_cnn"
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False

    device: str = "cuda:0"
    use_compile: bool = False
    warmup_batches: int = 5
    timed_batches: int = 50
    repetitions: int = 3

    nsight: bool = False
    nsight_args: list[str] = field(default_factory=list)
    nsight_dir: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

def load_configs(yaml_path: str | Path) -> list[ExperimentConfig]:
    """Load a YAML config file and expand any matrix combinations."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    matrix = raw.get("matrix", {})
    fixed = raw.get("fixed", {})
    output = raw.get("output", {})

    base: dict[str, Any] = {}
    for key in ("experiment", "description"):
        if key in raw:
            base[key] = raw[key]

    base.update(fixed)
    base.update(raw.get("profiling", {}))
    if "nsight_dir" in output:
        base["nsight_dir"] = output["nsight_dir"]

    valid = {f.name for f in fields(ExperimentConfig)}

    # if not matrix, then the config is just one experiment
    if not matrix:
        return [ExperimentConfig(**{k: v for k, v in base.items() if k in valid})]

    # if yes matrix, then config is defining multiple expeiments in a grid, so expand into multiple configs
    keys = list(matrix.keys())
    value_lists = [v if isinstance(v, list) else [v] for v in matrix.values()]

    configs = []
    for combo in itertools.product(*value_lists):
        merged = dict(base)
        for key, val in zip(keys, combo):
            merged[key] = val
        configs.append(ExperimentConfig(**{k: v for k, v in merged.items() if k in valid}))

    return configs
