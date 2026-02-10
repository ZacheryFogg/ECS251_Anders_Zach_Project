import importlib
import logging

import pandas as pd
from tqdm import tqdm

from infra.config import ExperimentConfig
from infra.nsight import profile_config
from infra.results import save_results
from infra.system_info import collect_system_info

logger = logging.getLogger(__name__)

benchmark_types = {
    "data_loading": "benchmarks.data_loading",
    "cpu_gpu_transfer": "benchmarks.cpu_gpu_transfer",
    "kernel_dispatch": "benchmarks.kernel_dispatch",
}

def run_all(configs: list[ExperimentConfig], progress=True) -> list[dict]:
    system_info = collect_system_info()
    logger.info(f"Starting {len(configs)} configs ({system_info.summary()})")

    all_rows = []
    iterator = tqdm(configs, desc="Experiments", disable=not progress)
    for config in iterator:
        iterator.set_postfix_str(config.label(), refresh=True)
        try:
            if config.nsight:
                trace_path, rows = profile_config(config)
                if trace_path:
                    logger.info(f"Nsight trace: {trace_path}")
            else:
                benchmark_fn = importlib.import_module(benchmark_types[config.experiment]).run
                rows = benchmark_fn(config, system_info)
            all_rows.extend(rows)
        except Exception:
            logger.exception(f"Failed: {config.label()}")

    logger.info(f"Completed. Total results: {len(all_rows)}")
    return all_rows


def run_and_save(configs, csv_path=None, progress=True) -> pd.DataFrame:
    rows = run_all(configs, progress=progress)

    output = csv_path or (configs[0].csv_path if configs else "")
    if output:
        save_results(rows, output)
    else:
        logger.warning("No CSV path specified; results not saved.")

    return pd.DataFrame(rows)
