from __future__ import annotations

import pytest
import torch

from rfs.config import OptimizerConfig
from rfs.optimizers import build_optimizer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


@pytest.mark.parametrize("name", ["adamw", "shampoo", "soap", "rfs"])
def test_optimizer_smoke(name: str) -> None:
    torch.manual_seed(7)
    layer = torch.nn.Sequential(
        torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 16)
    ).to(device="cuda", dtype=torch.bfloat16)
    config = OptimizerConfig(
        name=name,
        lr=1e-3,
        max_preconditioner_dim=64,
        precondition_frequency=1,
        start_preconditioning_step=1,
        root_iterations=60,
        use_hip_kernels=True,
    )
    optimizer = build_optimizer(layer.parameters(), config)
    before = [parameter.detach().clone() for parameter in layer.parameters()]
    for _ in range(2):
        inputs = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        loss = layer(inputs).float().square().mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    for previous, parameter in zip(before, layer.parameters(), strict=True):
        assert torch.isfinite(parameter).all()
        assert not torch.equal(previous, parameter)
