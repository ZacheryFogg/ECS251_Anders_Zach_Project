import itertools
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    """One fully-specified experiment run (before repetitions)."""

    experiment: str = "unnamed"
    description: str = ""

    model: str = "small_cnn"
    dataset: str = "mnist"
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False

    device: str = "cpu"
    use_compile: bool = False
    epochs: int = 1
    warmup_batches: int = 5
    timed_batches: int = 50
    repetitions: int = 3

    nsight: bool = False
    nsight_args: list[str] = field(default_factory=list)

    csv_path: str = ""
    nsight_dir: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def label(self) -> str:
        return (
            f"model={self.model} | bs={self.batch_size} | "
            f"workers={self.num_workers} | pin={self.pin_memory} | "
            f"compile={self.use_compile}"
        )


def load_configs(yaml_path: str | Path) -> list[ExperimentConfig]:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    return _expand_config(raw)


def _expand_config(raw: dict) -> list[ExperimentConfig]:
    matrix = raw.get("matrix", {})
    fixed = raw.get("fixed", {})
    output = raw.get("output", {})

    base: dict[str, Any] = {}
    for key in ("experiment", "description"):
        if key in raw:
            base[key] = raw[key]

    base.update(fixed)
    base.update(raw.get("profiling", {}))
    if "csv" in output:
        base["csv_path"] = output["csv"]
    if "nsight_dir" in output:
        base["nsight_dir"] = output["nsight_dir"]

    if not matrix:
        return [_make_config(base)]

    keys = list(matrix.keys())
    value_lists = [v if isinstance(v, list) else [v] for v in matrix.values()]

    configs = []
    for combo in itertools.product(*value_lists):
        merged = dict(base)
        for key, val in zip(keys, combo):
            merged[key] = val
        configs.append(_make_config(merged))

    return configs


def _make_config(params: dict) -> ExperimentConfig:
    valid = {f.name for f in fields(ExperimentConfig)}
    return ExperimentConfig(**{k: v for k, v in params.items() if k in valid})
