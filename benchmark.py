import timeit
from functools import partial

import numpy as np
import scipy.spatial.distance
import torch
import torch.nn
import torch.nn.functional


def benchmark_pdist(n, m, backend="scipy", device="cpu", dtype="float32", seed=0, **kwargs):
    """Benchmark pairwise distance computation.

    Args:
        n (int): Number of points.
        m (int): Dimension of points.
        backend (str, optional): Framework to use, either "scipy" or "torch". Defaults to "scipy".
        device (str, optional): Either "cpu" or "cuda". Defaults to "cpu".
        dtype (str, optional): Datatype. Defaults to "float32".
        seed (int, optional): Random seed. Defaults to 0.
    """

    rng = np.random.default_rng(seed)
    x = rng.random((n, m)).astype(dtype)

    if backend == "scipy":
        if device == "cuda":
            raise ValueError("CUDA not supported for scipy backend.")
        t = timeit.timeit(lambda: scipy.spatial.distance.pdist(x), **kwargs)
    elif backend == "torch":
        x = torch.from_numpy(x).to(device)
        t = timeit.timeit(lambda: torch.nn.functional.pdist(x), **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return t


def main():
    n = 15
    m = 3
    dtype = "float32"
    seed = 0
    number = 1000000
    device = "cpu"

    benchmark_pdist_fn = partial(
        benchmark_pdist, n, m, dtype=dtype, seed=seed, number=number, device=device
    )

    torch_time = benchmark_pdist_fn(backend="torch")
    scipy_time = benchmark_pdist_fn(backend="scipy")
    torch_gpu_time = benchmark_pdist_fn(backend="torch", device="cuda")

    print("pdist benchmark:")
    print(f"  torch: {torch_time:.3f} s")
    print(f"  scipy: {scipy_time:.3f} s")
    print(f"  torch (gpu): {torch_gpu_time:.3f} s")


if __name__ == "__main__":
    main()
