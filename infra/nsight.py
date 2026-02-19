"""Launch Nsight Systems profiling for a benchmark config."""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_NSIGHT_ARGS = [
    "--trace=cuda,osrt,nvtx",
    "--sample=none",
    "--cudabacktrace=true",
    "--cuda-memory-usage=true",
]


def profile_config(config, nsight_dir=None, timeout=600) -> tuple[Path | None, None]:
    """Run a benchmark under nsys profile via subprocess."""
    if not shutil.which("nsys"):
        logger.error("nsys not found on PATH. Skipping Nsight profiling.")
        return None, None

    nsight_dir = Path(nsight_dir or config.nsight_dir or "results/nsight")
    nsight_dir.mkdir(parents=True, exist_ok=True)

    trace_name = f"{config.model}__bs{config.batch_size}__w{config.num_workers}__pin{config.pin_memory}"
    trace_path = nsight_dir / trace_name

    # Write config as JSON so the subprocess can reconstruct it
    config_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(config.as_dict(), config_file)
    config_file.close()

    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")

    try:
        cmd = [
            "nsys", "profile",
            "-o", str(trace_path),
            "--force-overwrite=true",
            *(config.nsight_args or DEFAULT_NSIGHT_ARGS),
            "--",
            "python", "-c",
            f"import json, sys; sys.path.insert(0, '.'); "
            f"from infra.config import ExperimentConfig; "
            f"from benchmarks.training import run; "
            f"cfg = json.load(open('{config_file.name}')); "
            f"run(ExperimentConfig(**{{k: v for k, v in cfg.items() if hasattr(ExperimentConfig, k)}}))",
        ]

        logger.info(f"Nsight: {' '.join(cmd)}")
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True, env=env)

        rep_file = Path(f"{trace_path}.nsys-rep")
        actual_trace = rep_file if rep_file.exists() else trace_path
        logger.info(f"Nsight trace saved: {actual_trace}")
        return actual_trace, None

    except subprocess.CalledProcessError as e:
        logger.error(f"Nsight failed: {e.stderr[:500] if e.stderr else e}")
        return None, None
    except subprocess.TimeoutExpired:
        logger.error(f"Nsight timed out after {timeout}s")
        return None, None
    finally:
        Path(config_file.name).unlink(missing_ok=True)
