import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from infra.config import ExperimentConfig
from infra.models import get_model
from infra.results import make_result_row
from infra.system_info import SystemInfo
from infra.timing import MultiTimer

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def run(config: ExperimentConfig, system_info: SystemInfo) -> list[dict]:
    device = torch.device(config.device)
    model = get_model(config.model, device=device, compile=config.use_compile)

    dataset = datasets.MNIST(str(DATA_DIR), train=True, download=True, transform=MNIST_TRANSFORM)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=True,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
    )

    results = []

    for rep in range(config.repetitions):
        # Warmup
        _run_batches(model, loader, device, config.warmup_batches)

        # Timed: measure load and forward separately
        load_timer = MultiTimer(device=str(device))
        forward_timer = MultiTimer(device=str(device))
        total_timer = MultiTimer(device=str(device))

        loader_iter = iter(loader)
        for _ in range(config.timed_batches):
            try:
                with total_timer.measure():
                    with load_timer.measure():
                        images, labels = next(loader_iter)
                    images = images.to(device, non_blocking=config.pin_memory)
                    with torch.no_grad():
                        with forward_timer.measure():
                            model(images)
            except StopIteration:
                loader_iter = iter(loader)

        row = make_result_row(config, rep, total_timer.stats(), extra={
            "load_wall_ms_mean": (load_timer.stats().get("wall_ms_mean") or 0.0),
            "load_wall_ms_std": (load_timer.stats().get("wall_ms_std") or 0.0),
            "forward_wall_ms_mean": (forward_timer.stats().get("wall_ms_mean") or 0.0),
            "forward_wall_ms_std": (forward_timer.stats().get("wall_ms_std") or 0.0),
        })
        results.append(row)

        logger.info(
            f"  Rep {rep + 1}: total={row['wall_ms_mean']:.2f}ms/batch "
            f"(load={row['load_wall_ms_mean']:.2f}ms, fwd={row['forward_wall_ms_mean']:.2f}ms) "
            f"| {row['batches_per_sec']:.1f} batches/s"
        )

    return results


def _run_batches(model, loader, device, num_batches):
    loader_iter = iter(loader)
    for _ in range(num_batches):
        try:
            images, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            images, _ = next(loader_iter)
        images = images.to(device)
        with torch.no_grad():
            model(images)
