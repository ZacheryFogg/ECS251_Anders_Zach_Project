import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_NSIGHT_ARGS = [
    "--trace=cuda,osrt,nvtx",
    "--sample=none",
    "--cudabacktrace=true",
]

SINGLE_RUNNER_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_single.py"


def nsight_available() -> bool:
    return shutil.which("nsys") is not None


def nsight_output_name(config) -> str:
    return (
        f"{config.experiment}"
        f"__{config.model}"
        f"__bs{config.batch_size}"
        f"__w{config.num_workers}"
        f"__pin{config.pin_memory}"
        f"__compile{config.use_compile}"
    )


def profile_config(config, nsight_dir=None, timeout=600) -> tuple[Path | None, list[dict]]:
    if not nsight_available():
        logger.error("nsys not found on PATH. Skipping Nsight profiling.")
        return None, []

    nsight_dir = Path(nsight_dir or config.nsight_dir or "results/nsight")
    nsight_dir.mkdir(parents=True, exist_ok=True)

    trace_path = nsight_dir / nsight_output_name(config)

    # Write config to a temp JSON file for the subprocess
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config.as_dict(), f)
        config_json = f.name

    try:
        cmd = [
            "nsys", "profile",
            f"-o={trace_path}",
            "--force-overwrite=true",
            *(config.nsight_args or DEFAULT_NSIGHT_ARGS),
            "--",
            "python", str(SINGLE_RUNNER_SCRIPT), config_json,
        ]

        logger.info(f"Nsight: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=True,
        )

        # run_single.py prints JSON results to stdout
        rows = []
        if result.stdout.strip():
            try:
                rows = json.loads(result.stdout.strip().split("\n")[-1])
            except json.JSONDecodeError:
                logger.warning("Could not parse results from Nsight subprocess")

        rep_file = Path(f"{trace_path}.nsys-rep")
        actual_trace = rep_file if rep_file.exists() else trace_path
        logger.info(f"Nsight trace saved: {actual_trace}")
        return actual_trace, rows

    except subprocess.CalledProcessError as e:
        logger.error(f"Nsight failed: {e.stderr[:500] if e.stderr else e}")
        return None, []
    except subprocess.TimeoutExpired:
        logger.error(f"Nsight timed out after {timeout}s")
        return None, []
    finally:
        Path(config_json).unlink(missing_ok=True)
