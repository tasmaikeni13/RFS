from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import torch

_EXTENSION: Any | None = None
_FAILED = False


def load_extension(required: bool = False) -> Any | None:
    global _EXTENSION, _FAILED
    if _EXTENSION is not None:
        return _EXTENSION
    if _FAILED:
        if required:
            raise RuntimeError("The RFS HIP extension failed to load")
        return None
    if not torch.cuda.is_available() or torch.version.hip is None:
        _FAILED = True
        if required:
            raise RuntimeError("The RFS HIP extension requires ROCm")
        return None
    try:
        import importlib.util

        from torch.utils.cpp_extension import load

        source = Path(__file__).resolve().parent
        sdk_spec = importlib.util.find_spec("_rocm_sdk_core")
        sdk_lib = Path(sdk_spec.origin).resolve().parent / "lib" if sdk_spec else None
        link_flags = []
        if sdk_lib is not None and sdk_lib.exists():
            link_flags = [f"-L{sdk_lib}", f"-Wl,-rpath,{sdk_lib}"]
        os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx942")
        _EXTENSION = load(
            name="rfs_mi300x_hip",
            sources=[str(source / "bindings.cpp"), str(source / "mi300x_kernels.hip")],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx942"],
            extra_ldflags=link_flags,
            verbose=os.getenv("RFS_KERNEL_VERBOSE", "0") == "1",
        )
    except Exception as error:
        _FAILED = True
        if required:
            raise
        warnings.warn(f"RFS HIP extension unavailable; using PyTorch fallback: {error}")
    return _EXTENSION


def extension_available() -> bool:
    return load_extension(required=False) is not None


def ema_(state: torch.Tensor, sample: torch.Tensor, beta: float, native: bool = True) -> None:
    extension = load_extension() if native else None
    if extension is not None and state.is_cuda and state.dtype == sample.dtype == torch.float32:
        extension.ema(state.contiguous(), sample.contiguous(), beta)
    else:
        state.mul_(beta).add_(sample, alpha=1.0 - beta)


def affine_identity(
    matrix: torch.Tensor, diagonal: float, scale: float, native: bool = True
) -> torch.Tensor:
    extension = load_extension() if native else None
    if extension is not None and matrix.is_cuda and matrix.dtype in (torch.float32, torch.float64):
        return extension.affine_identity(matrix.contiguous(), diagonal, scale)
    result = matrix.mul(scale)
    result.diagonal(dim1=-2, dim2=-1).add_(diagonal)
    return result


def symmetrize(matrix: torch.Tensor, native: bool = True) -> torch.Tensor:
    extension = load_extension() if native else None
    if extension is not None and matrix.is_cuda and matrix.dtype in (torch.float32, torch.float64):
        return extension.symmetrize(matrix.contiguous())
    return 0.5 * (matrix + matrix.transpose(-2, -1))


def adamw_step_(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    bias_correction1: float,
    bias_correction2: float,
    native: bool = True,
) -> None:
    extension = load_extension() if native else None
    if extension is not None and parameter.is_cuda and gradient.dtype == parameter.dtype:
        extension.adamw_step(
            parameter.contiguous(),
            gradient.contiguous(),
            exp_avg,
            exp_avg_sq,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction1,
            bias_correction2,
        )
        return
    exp_avg.mul_(beta1).add_(gradient.float(), alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(gradient.float(), gradient.float(), value=1.0 - beta2)
    update = (exp_avg / bias_correction1) / ((exp_avg_sq / bias_correction2).sqrt() + eps)
    parameter.mul_(1.0 - lr * weight_decay).add_(update.to(parameter.dtype), alpha=-lr)


def parameter_step_(
    parameter: torch.Tensor,
    update: torch.Tensor,
    lr: float,
    weight_decay: float,
    native: bool = True,
) -> None:
    extension = load_extension() if native else None
    if extension is not None and parameter.is_cuda:
        extension.parameter_step(parameter.contiguous(), update.contiguous(), lr, weight_decay)
    else:
        parameter.mul_(1.0 - lr * weight_decay).add_(update.to(parameter.dtype), alpha=-lr)
