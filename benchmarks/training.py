"""Training benchmark with NVTX annotations for Nsight Systems profiling.

Runs a training loop (forward + backward + optimizer) on MNIST with NVTX
annotations so Nsight can decompose the pipeline into phases:
  data_loading, host_to_device, forward, loss, backward, optimizer_step
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from infra.config import ExperimentConfig
from infra.models import get_model

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

# these helpers make it so this code won't crash if no cuda device
def _nvtx_push(name: str, device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.nvtx.range_push(name)


def _nvtx_pop(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.nvtx.range_pop()


def run(config: ExperimentConfig) -> None:
    device = torch.device(config.device)

    # Get a pytorch model 
    model = get_model(config.model, device=device, compile=config.use_compile, training=True)

    # Loss function and optimz  
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # MNIST dataloader
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


    for rep in range(config.repetitions):

        # Warmup everything 
        _warmup(model, criterion, optimizer, loader, device, config)

        
        loader_iter = iter(loader)
        for batch_idx in range(config.timed_batches):
            try:

                _nvtx_push("data_loading", device) # this tells Nsight what to lable the operations in this "block" in its timeline view
                images, labels = next(loader_iter)
                _nvtx_pop(device)

                _nvtx_push("host_to_device", device)
                images = images.to(device, non_blocking=config.pin_memory)
                labels = labels.to(device, non_blocking=config.pin_memory)
                _nvtx_pop(device)

                _nvtx_push("forward", device)
                output = model(images)
                _nvtx_pop(device)

                _nvtx_push("loss", device)
                loss = criterion(output, labels)
                _nvtx_pop(device)

                _nvtx_push("backward", device)
                loss.backward()
                _nvtx_pop(device)

                _nvtx_push("optimizer_step", device)
                optimizer.step()
                optimizer.zero_grad()
                _nvtx_pop(device)

            except StopIteration:
                loader_iter = iter(loader)

        logger.info(f"Rep {rep + 1}/{config.repetitions} complete")


def _warmup(model, criterion, optimizer, loader, device, config):
    """Run a few un-timed training steps to warm up CUDA and the data pipeline."""
    loader_iter = iter(loader)
    for _ in range(config.warmup_batches):
        try:
            images, labels = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            images, labels = next(loader_iter)
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()