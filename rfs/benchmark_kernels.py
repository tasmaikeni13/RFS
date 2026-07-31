from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch

from .kernels import load_extension
from .roots import inverse_fourth_root_eigh, inverse_fourth_root_rfs


def measure(function: Callable[[], object], warmup: int = 5, repeats: int = 25) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000)
    return statistics.median(values)


def main() -> None:
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("Kernel benchmark requires ROCm")
    extension = load_extension(required=True)
    results: dict[str, object] = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(0),
    }

    size = 124_439_808
    parameter = torch.randn(size, device="cuda", dtype=torch.bfloat16)
    gradient = torch.randn_like(parameter)
    exp_avg = torch.zeros(size, device="cuda")
    exp_avg_sq = torch.zeros(size, device="cuda")
    reference_parameter = parameter.clone()
    reference_m = exp_avg.clone()
    reference_v = exp_avg_sq.clone()
    native_ms = measure(
        lambda: extension.adamw_step(
            parameter, gradient, exp_avg, exp_avg_sq, 6e-4, 0.9, 0.95, 1e-8, 0.1, 0.1, 0.05
        ),
        warmup=3,
        repeats=10,
    )

    def torch_adam() -> None:
        reference_m.mul_(0.9).add_(gradient.float(), alpha=0.1)
        reference_v.mul_(0.95).addcmul_(gradient.float(), gradient.float(), value=0.05)
        update = (reference_m / 0.1) / ((reference_v / 0.05).sqrt() + 1e-8)
        reference_parameter.mul_(1 - 6e-5).add_(update.to(torch.bfloat16), alpha=-6e-4)

    torch_ms = measure(torch_adam, warmup=3, repeats=10)
    results["adamw"] = {
        "elements": size,
        "native_ms": native_ms,
        "torch_ms": torch_ms,
        "speedup": torch_ms / native_ms,
    }

    generator = torch.Generator(device="cuda").manual_seed(17)
    raw = torch.randn(61, 768, 768, generator=generator, device="cuda") / 768**0.5
    factors = raw @ raw.mT
    root_iterations = 60
    matrix_epsilon = 1e-3
    rfs_ms = measure(
        lambda: inverse_fourth_root_rfs(factors, matrix_epsilon, root_iterations, True),
        warmup=1,
        repeats=3,
    )
    eigh_ms = measure(
        lambda: inverse_fourth_root_eigh(factors, matrix_epsilon), warmup=1, repeats=3
    )
    root, residual = inverse_fourth_root_rfs(factors, matrix_epsilon, root_iterations, True)
    reference = inverse_fourth_root_eigh(factors, matrix_epsilon)
    relative = torch.linalg.matrix_norm(root - reference, ord="fro", dim=(-2, -1)) / (
        torch.linalg.matrix_norm(reference, ord="fro", dim=(-2, -1))
    )
    results["inverse_fourth_root"] = {
        "batch": 61,
        "dimension": 768,
        "iterations": root_iterations,
        "matrix_epsilon": matrix_epsilon,
        "rfs_ms": rfs_ms,
        "eigh_ms": eigh_ms,
        "speedup": eigh_ms / rfs_ms,
        "max_certificate": float(residual.max()),
        "max_relative_to_eigh": float(relative.max()),
    }
    results["created_unix"] = time.time()
    output = Path("artifacts/kernel_benchmark.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
