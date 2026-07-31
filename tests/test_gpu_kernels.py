from __future__ import annotations

import pytest
import torch

from rfs.kernels import load_extension
from rfs.roots import inverse_fourth_root_eigh, inverse_fourth_root_rfs

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None, reason="requires ROCm"
)


def test_primitive_kernel_parity() -> None:
    extension = load_extension(required=True)
    state = torch.randn(100_003, device="cuda")
    sample = torch.randn_like(state)
    expected = 0.91 * state + 0.09 * sample
    extension.ema(state, sample, 0.91)
    torch.cuda.synchronize()
    torch.testing.assert_close(state, expected, rtol=2e-6, atol=2e-6)

    matrix = torch.randn(3, 65, 65, device="cuda")
    actual = extension.symmetrize(matrix)
    torch.testing.assert_close(actual, (matrix + matrix.mT) / 2, rtol=0, atol=0)
    actual = extension.affine_identity(matrix, 1.5, -0.5)
    expected = -0.5 * matrix
    expected.diagonal(dim1=-2, dim2=-1).add_(1.5)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    double_matrix = torch.randn(2, 9, 9, device="cuda", dtype=torch.float64)
    double_actual = extension.symmetrize(double_matrix.contiguous())
    torch.testing.assert_close(
        double_actual, (double_matrix + double_matrix.mT) / 2, rtol=0, atol=0
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_adamw_kernel_parity(dtype: torch.dtype) -> None:
    extension = load_extension(required=True)
    parameter = torch.randn(100_003, device="cuda", dtype=dtype)
    gradient = torch.randn_like(parameter)
    exp_avg = torch.randn(parameter.shape, device="cuda", dtype=torch.float32)
    exp_avg_sq = torch.rand(parameter.shape, device="cuda", dtype=torch.float32)
    expected_p = parameter.clone()
    expected_m = exp_avg.clone()
    expected_v = exp_avg_sq.clone()
    lr, beta1, beta2, eps, decay = 3e-4, 0.9, 0.95, 1e-8, 0.1
    bc1, bc2 = 1 - beta1**7, 1 - beta2**7
    expected_m.mul_(beta1).add_(gradient.float(), alpha=1 - beta1)
    expected_v.mul_(beta2).addcmul_(gradient.float(), gradient.float(), value=1 - beta2)
    update = (expected_m / bc1) / ((expected_v / bc2).sqrt() + eps)
    expected_p = (expected_p.float() * (1 - lr * decay) - lr * update).to(dtype)
    extension.adamw_step(
        parameter, gradient, exp_avg, exp_avg_sq, lr, beta1, beta2, eps, decay, bc1, bc2
    )
    torch.cuda.synchronize()
    parameter_atol = 3e-7 if dtype == torch.float32 else 0
    torch.testing.assert_close(parameter, expected_p, rtol=0, atol=parameter_atol)
    torch.testing.assert_close(exp_avg, expected_m, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(exp_avg_sq, expected_v, rtol=2e-6, atol=2e-6)


def test_rfs_root_certificate_and_eigh_parity() -> None:
    generator = torch.Generator(device="cuda").manual_seed(123)
    value = torch.randn(4, 64, 64, generator=generator, device="cuda")
    factors = value @ value.mT + 0.1 * torch.eye(64, device="cuda")
    root, residual = inverse_fourth_root_rfs(factors, epsilon=1e-6, iterations=20, native=True)
    reference = inverse_fourth_root_eigh(factors, epsilon=1e-6)
    relative = torch.linalg.matrix_norm(root - reference, ord="fro", dim=(-2, -1)) / (
        torch.linalg.matrix_norm(reference, ord="fro", dim=(-2, -1))
    )
    assert float(residual.max()) < 1e-4
    assert float(relative.max()) < 3e-5
