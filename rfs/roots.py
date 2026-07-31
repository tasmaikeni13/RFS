from __future__ import annotations

import torch

from .kernels import affine_identity, symmetrize


@torch.no_grad()
def regularize(factors: torch.Tensor, epsilon: float) -> torch.Tensor:
    width = factors.shape[-1]
    scale = factors.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-12)
    eye = torch.eye(width, device=factors.device, dtype=factors.dtype)
    return symmetrize(factors) + (epsilon * scale)[..., None, None] * eye


@torch.no_grad()
def inverse_fourth_root_eigh(factors: torch.Tensor, epsilon: float) -> torch.Tensor:
    # Use the same fp64 refresh precision as RFS so this comparison isolates the
    # root engine rather than a precision choice.
    matrix = regularize(factors.double(), epsilon)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    floor = torch.finfo(matrix.dtype).eps * eigenvalues.amax(dim=-1, keepdim=True)
    eigenvalues = eigenvalues.clamp_min(floor)
    return symmetrize(
        (eigenvectors * eigenvalues.rsqrt().sqrt().unsqueeze(-2)) @ eigenvectors.transpose(-2, -1)
    ).float()


@torch.no_grad()
def _coupled_invsqrt(
    matrix: torch.Tensor, iterations: int, native: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Safeguarded coupled Newton--Schulz returning sqrt and inverse sqrt.

    Early language-model gradient factors are nearly rank deficient. In fp32 the
    coupled iteration can improve and then lose accuracy in those tiny eigenspaces.
    A per-matrix floor detector stops before a late iterate can contaminate an
    otherwise valid root.
    """
    width = matrix.shape[-1]
    identity = torch.eye(width, device=matrix.device, dtype=matrix.dtype).expand_as(matrix)
    y = matrix.clone()
    z = identity.clone()
    best_error = torch.full(
        matrix.shape[:-2], float("inf"), device=matrix.device, dtype=matrix.dtype
    )
    stalled = torch.zeros(matrix.shape[:-2], device=matrix.device, dtype=torch.int32)
    active = torch.ones(matrix.shape[:-2], device=matrix.device, dtype=torch.bool)
    tolerance = 50.0 * torch.finfo(matrix.dtype).eps
    for _ in range(iterations):
        product = torch.bmm(z, y)
        error = torch.linalg.matrix_norm(product - identity, ord="fro", dim=(-2, -1)) / width**0.5
        on_floor = error < 0.1
        meaningful = error < 0.95 * best_error
        best_error = torch.where(on_floor & meaningful, error, best_error)
        stalled = torch.where(
            on_floor & ~meaningful,
            stalled + 1,
            torch.zeros_like(stalled),
        )
        active &= (error >= tolerance) & (stalled < 4) & torch.isfinite(error)
        if not bool(active.any()):
            break
        t = affine_identity(product, diagonal=1.5, scale=-0.5, native=native)
        next_y = symmetrize(torch.bmm(y, t), native=native)
        next_z = symmetrize(torch.bmm(t, z), native=native)
        mask = active[..., None, None]
        y = torch.where(mask, next_y, y)
        z = torch.where(mask, next_z, z)
    return y, z


@torch.no_grad()
def inverse_fourth_root_rfs(
    factors: torch.Tensor,
    epsilon: float,
    iterations: int = 60,
    native: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cold, eigendecomposition-free inverse fourth root plus residual certificate."""
    # MI300X provides native fp64 matrix throughput. Performing only the root
    # refresh in fp64 is substantially more stable on early low-rank factors.
    matrix = regularize(factors.double(), epsilon)
    frobenius = torch.linalg.matrix_norm(matrix, ord="fro", dim=(-2, -1)).clamp_min(1e-30)
    normalized = matrix / frobenius[:, None, None]
    square_root, _ = _coupled_invsqrt(normalized, iterations, native)
    second_scale = torch.linalg.matrix_norm(square_root, ord="fro", dim=(-2, -1)).clamp_min(1e-30)
    _, inverse_half_root = _coupled_invsqrt(
        square_root / second_scale[:, None, None], iterations, native
    )
    scale = frobenius.pow(-0.25) * second_scale.pow(-0.5)
    root = symmetrize(scale[:, None, None] * inverse_half_root, native=native)
    root2 = torch.bmm(root, root)
    residual_matrix = torch.bmm(torch.bmm(root2, root2), matrix)
    identity = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    residual = (
        torch.linalg.matrix_norm(residual_matrix - identity, ord="fro", dim=(-2, -1))
        / matrix.shape[-1] ** 0.5
    )
    return root.float(), residual.float()
