import multiprocessing
import platform
from dataclasses import asdict, dataclass


@dataclass
class SystemInfo:
    python_version: str = ""
    torch_version: str = ""
    cuda_version: str = ""
    gpu_name: str = ""
    gpu_count: int = 0
    cpu_model: str = ""
    cpu_count: int = 0
    os_name: str = ""
    os_version: str = ""

    def as_dict(self):
        return asdict(self)

    def summary(self) -> str:
        gpu = f" | GPU: {self.gpu_name} x{self.gpu_count}" if self.gpu_name else ""
        return (
            f"Python {self.python_version} | PyTorch {self.torch_version} | "
            f"CUDA {self.cuda_version}{gpu} | "
            f"CPU: {self.cpu_model} ({self.cpu_count} cores) | "
            f"{self.os_name} {self.os_version}"
        )


def collect_system_info() -> SystemInfo:
    info = SystemInfo(
        python_version=platform.python_version(),
        cpu_model=platform.processor() or "unknown",
        cpu_count=multiprocessing.cpu_count() or 0,
        os_name=platform.system(),
        os_version=platform.release(),
    )

    try:
        import torch
        info.torch_version = torch.__version__
        if torch.cuda.is_available():
            info.cuda_version = torch.version.cuda or ""
            info.gpu_count = torch.cuda.device_count()
            if info.gpu_count > 0:
                info.gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        info.torch_version = "not installed"

    return info
