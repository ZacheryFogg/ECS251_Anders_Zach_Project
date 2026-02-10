import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def save_results(rows, path, append=False):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if append and out.exists():
        df.to_csv(out, mode="a", header=False, index=False)
    else:
        df.to_csv(out, index=False)
    logger.info(f"Wrote {len(df)} rows to {out}")
    return out


def make_result_row(config, rep, timer_stats, extra=None):
    wall_mean = timer_stats.get("wall_ms_mean", 0.0) or 0.0
    bps = (1000.0 / wall_mean) if wall_mean > 0 else 0.0

    row = {
        "experiment": config.experiment,
        "model": config.model,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "use_compile": config.use_compile,
        "device": config.device,
        "repetition": rep,
        "wall_ms_mean": wall_mean,
        "wall_ms_std": timer_stats.get("wall_ms_std", 0.0),
        "wall_ms_min": timer_stats.get("wall_ms_min", 0.0),
        "wall_ms_max": timer_stats.get("wall_ms_max", 0.0),
        "cuda_ms_mean": timer_stats.get("cuda_ms_mean"),
        "cuda_ms_std": timer_stats.get("cuda_ms_std"),
        "batches_per_sec": bps,
        "samples_per_sec": bps * config.batch_size,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        row.update(extra)
    return row
