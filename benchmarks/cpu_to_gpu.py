"""Simple benchmark: time moving tensors of different sizes from CPU to GPU.

Run under Nsight:
    nsys profile -o results/nsight/cpu_to_gpu --trace=cuda,osrt,nvtx python3 benchmarks/cpu_to_gpu.py

Run under strace (syscall summary):
    strace -c python3 benchmarks/cpu_to_gpu.py
    time strace -cw python3 benchmarks/cpu_to_gpu.py

Run under strace (ioctl/mmap only, one line per call):
    strace -e trace=ioctl,mmap,munmap,mprotect python3 benchmarks/cpu_to_gpu.py
"""

import torch

SIZES = [
    ("1 KB",   256),
    ("1 MB",   16384),
    ("10 MB",  128 * 1024),
    ("100 MB", 1280 * 1024),
    ("1 GB",   1280 * 10240),
]

WARMUP = 5
REPS   = 20
DEVICE = "cuda:0"


def benchmark(label, n_floats, warmup=WARMUP, reps=REPS):
    tensor = torch.randn(n_floats)
    # warmup
    for _ in range(warmup):
        t = tensor.to(DEVICE)
        print(t.device)
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    torch.cuda.nvtx.range_push(label)
    start.record()
    for _ in range(reps):
        t = tensor.to(DEVICE)
    end.record()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    return start.elapsed_time(end) / reps  


if __name__ == "__main__":
    print(f"{'Size':<10} {'Avg (ms)':>10} {'GB/s':>10}")
    print("-" * 32)
    for label, n in SIZES:
        ms = benchmark(label, n)
        bytes_ = n * 4  
        gb_s = (bytes_ / 1e9) / (ms / 1e3)
        print(f"{label:<10} {ms:>10.3f} {gb_s:>10.2f}")