import time
from dataclasses import dataclass, field

import torch


@dataclass
class TimingResult:
    wall_ms: float = 0.0
    cuda_ms: float | None = None


class Timer:
    """Context manager measuring wall-clock and optional CUDA event time."""

    def __init__(self, device="cpu"):
        self._device = torch.device(device)
        self._use_cuda = self._device.type == "cuda" and torch.cuda.is_available()
        self._start_time = 0.0
        self._end_time = 0.0
        self._start_event = None
        self._end_event = None
        self.result = TimingResult()

    def reset(self):
        self._start_time = 0.0
        self._end_time = 0.0
        self._start_event = None
        self._end_event = None
        self.result = TimingResult()

    def __enter__(self):
        if self._use_cuda:
            torch.cuda.synchronize(self._device)
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self._use_cuda and self._end_event is not None:
            self._end_event.record()
            torch.cuda.synchronize(self._device)
        self._end_time = time.perf_counter()

        self.result.wall_ms = (self._end_time - self._start_time) * 1000.0
        if self._use_cuda and self._start_event and self._end_event:
            self.result.cuda_ms = self._start_event.elapsed_time(self._end_event)


@dataclass
class MultiTimer:
    """Accumulates multiple Timer measurements and computes stats."""

    device: str = "cpu"
    results: list[TimingResult] = field(default_factory=list)

    def measure(self):
        return _AppendingTimer(self, self.device)

    def stats(self):
        if not self.results:
            return {}

        wall = [r.wall_ms for r in self.results]
        cuda = [r.cuda_ms for r in self.results if r.cuda_ms is not None]

        out = {
            "wall_ms_mean": _mean(wall),
            "wall_ms_std": _std(wall),
            "wall_ms_min": min(wall),
            "wall_ms_max": max(wall),
            "n_samples": float(len(wall)),
        }
        if cuda:
            out["cuda_ms_mean"] = _mean(cuda)
            out["cuda_ms_std"] = _std(cuda)
            out["cuda_ms_min"] = min(cuda)
            out["cuda_ms_max"] = max(cuda)
        return out


class _AppendingTimer(Timer):
    def __init__(self, parent, device):
        super().__init__(device)
        self._parent = parent

    def __exit__(self, *exc):
        super().__exit__(*exc)
        self._parent.results.append(self.result)


def _mean(vals):
    return sum(vals) / len(vals)


def _std(vals):
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
