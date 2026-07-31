from __future__ import annotations

import collections
from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import Optimizer

from .config import OptimizerConfig
from .kernels import adamw_step_, ema_, parameter_step_
from .roots import inverse_fourth_root_eigh, inverse_fourth_root_rfs, regularize


class NativeAdamW(Optimizer):
    """AdamW with a single MI300X HIP kernel per parameter."""

    def __init__(self, params: Iterable[Any], config: OptimizerConfig) -> None:
        defaults = {
            "lr": config.lr,
            "betas": (config.beta1, config.beta2),
            "eps": config.eps,
            "weight_decay": config.weight_decay,
            "decay": True,
        }
        super().__init__(params, defaults)
        self.native = config.use_hip_kernels

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.float32)
                state["step"] += 1
                step = state["step"]
                adamw_step_(
                    parameter,
                    parameter.grad,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    lr=group["lr"],
                    beta1=beta1,
                    beta2=beta2,
                    eps=group["eps"],
                    weight_decay=group["weight_decay"] if group["decay"] else 0.0,
                    bias_correction1=1.0 - beta1**step,
                    bias_correction2=1.0 - beta2**step,
                    native=self.native,
                )
        return loss


class ShampooFamily(Optimizer):
    """Shared Shampoo state; `root_engine` isolates eig versus RFS roots."""

    def __init__(self, params: Iterable[Any], config: OptimizerConfig, root_engine: str) -> None:
        if root_engine not in {"eigh", "rfs"}:
            raise ValueError(root_engine)
        defaults = {"lr": config.lr, "weight_decay": config.weight_decay, "decay": True}
        super().__init__(params, defaults)
        self.config = config
        self.root_engine = root_engine
        self.native = config.use_hip_kernels
        self.last_diagnostics: dict[str, float] = {}

    def _init_state(self, parameter: torch.Tensor) -> dict[str, Any]:
        state = self.state[parameter]
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
        state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.float32)
        state["left"] = None
        state["right"] = None
        state["left_root"] = None
        state["right_root"] = None
        if parameter.ndim == 2:
            rows, columns = parameter.shape
            if rows <= self.config.max_preconditioner_dim:
                state["left"] = torch.zeros((rows, rows), device=parameter.device)
            if columns <= self.config.max_preconditioner_dim:
                state["right"] = torch.zeros((columns, columns), device=parameter.device)
        return state

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        pending: dict[int, list[tuple[dict[str, Any], str]]] = collections.defaultdict(list)
        records: list[tuple[torch.Tensor, dict[str, Any], dict[str, Any]]] = []
        beta1, beta2 = self.config.beta1, self.config.beta2

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter] or self._init_state(parameter)
                state["step"] += 1
                gradient = parameter.grad.float()
                state["exp_avg"].mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                if parameter.ndim == 2:
                    if state["left"] is not None:
                        gram = gradient @ gradient.T / gradient.shape[1]
                        ema_(state["left"], gram, self.config.shampoo_beta, self.native)
                    if state["right"] is not None:
                        gram = gradient.T @ gradient / gradient.shape[0]
                        ema_(state["right"], gram, self.config.shampoo_beta, self.native)
                refresh = state["step"] >= self.config.start_preconditioning_step and (
                    state["step"] == self.config.start_preconditioning_step
                    or state["step"] % self.config.precondition_frequency == 0
                )
                if refresh:
                    for side in ("left", "right"):
                        factor = state[side]
                        if factor is not None:
                            pending[factor.shape[0]].append((state, side))
                records.append((parameter, state, group))

        residuals = []
        for entries in pending.values():
            factors = torch.stack([state[side] for state, side in entries])
            if self.root_engine == "eigh":
                roots = inverse_fourth_root_eigh(factors, self.config.matrix_eps)
            else:
                roots, residual = inverse_fourth_root_rfs(
                    factors,
                    self.config.matrix_eps,
                    self.config.root_iterations,
                    self.native,
                )
                residuals.append(residual)
            for root, (state, side) in zip(roots.unbind(), entries, strict=True):
                state[f"{side}_root"] = root

        for parameter, state, group in records:
            step = state["step"]
            bc1 = 1.0 - beta1**step
            bc2 = 1.0 - beta2**step
            graft = (state["exp_avg"] / bc1) / (
                (state["exp_avg_sq"] / bc2).sqrt() + self.config.eps
            )
            if state["left_root"] is None and state["right_root"] is None:
                update = graft
            else:
                update = state["exp_avg"] / bc1
                if state["left_root"] is not None:
                    update = state["left_root"] @ update
                if state["right_root"] is not None:
                    update = update @ state["right_root"]
                if self.config.graft:
                    target_norm = torch.linalg.vector_norm(graft)
                    update_norm = torch.linalg.vector_norm(update).clamp_min(1e-16)
                    update = update * (target_norm / update_norm)
            parameter_step_(
                parameter,
                update,
                group["lr"],
                group["weight_decay"] if group["decay"] else 0.0,
                self.native,
            )
        if residuals:
            values = torch.cat(residuals)
            self.last_diagnostics = {
                "root_residual_mean": float(values.mean()),
                "root_residual_max": float(values.max()),
            }
        return loss


class SOAP(Optimizer):
    """SOAP with the reference implementation's initialization and QR refresh."""

    def __init__(self, params: Iterable[Any], config: OptimizerConfig) -> None:
        defaults = {"lr": config.lr, "weight_decay": config.weight_decay, "decay": True}
        super().__init__(params, defaults)
        self.config = config
        self.native = config.use_hip_kernels
        self.last_diagnostics: dict[str, float] = {}

    def _init_state(self, parameter: torch.Tensor) -> dict[str, Any]:
        state = self.state[parameter]
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
        state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.float32)
        state["left"] = state["right"] = None
        state["left_q"] = state["right_q"] = None
        if parameter.ndim == 2:
            rows, columns = parameter.shape
            if rows <= self.config.max_preconditioner_dim:
                state["left"] = torch.zeros((rows, rows), device=parameter.device)
            if columns <= self.config.max_preconditioner_dim:
                state["right"] = torch.zeros((columns, columns), device=parameter.device)
        return state

    @staticmethod
    def _project(value: torch.Tensor, state: dict[str, Any]) -> torch.Tensor:
        if state["left_q"] is not None:
            value = state["left_q"].T @ value
        if state["right_q"] is not None:
            value = value @ state["right_q"]
        return value

    @staticmethod
    def _project_back(value: torch.Tensor, state: dict[str, Any]) -> torch.Tensor:
        if state["left_q"] is not None:
            value = state["left_q"] @ value
        if state["right_q"] is not None:
            value = value @ state["right_q"].T
        return value

    def _update_factors(self, gradient: torch.Tensor, state: dict[str, Any]) -> None:
        if gradient.ndim != 2:
            return
        if state["left"] is not None:
            gram = gradient @ gradient.T / gradient.shape[1]
            ema_(state["left"], gram, self.config.shampoo_beta, self.native)
        if state["right"] is not None:
            gram = gradient.T @ gradient / gradient.shape[0]
            ema_(state["right"], gram, self.config.shampoo_beta, self.native)

    def _initialize_basis(self, state: dict[str, Any]) -> None:
        for side in ("left", "right"):
            factor = state[side]
            if factor is None:
                continue
            matrix = regularize(factor.unsqueeze(0), self.config.matrix_eps)[0]
            _, q = torch.linalg.eigh(matrix)
            state[f"{side}_q"] = q.flip(-1)

    def _refresh_basis(self, state: dict[str, Any]) -> None:
        # The reference SOAP implementation transports the first moment through
        # the original coordinates and permutes (rather than densely rotates)
        # the diagonal second moment before one power iteration plus QR.
        original_m = self._project_back(state["exp_avg"], state)
        for side, axis in (("left", 0), ("right", 1)):
            factor = state[side]
            q = state[f"{side}_q"]
            if factor is None or q is None:
                continue
            estimated = torch.diagonal(q.T @ factor @ q)
            order = torch.argsort(estimated, descending=True)
            state["exp_avg_sq"] = state["exp_avg_sq"].index_select(axis, order)
            q, _ = torch.linalg.qr(factor @ q.index_select(1, order))
            state[f"{side}_q"] = q
        state["exp_avg"] = self._project(original_m, state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        beta1, beta2 = self.config.beta1, self.config.beta2
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                first_step = not self.state[parameter]
                state = self._init_state(parameter) if first_step else self.state[parameter]
                gradient = parameter.grad.float()
                if first_step:
                    self._update_factors(gradient, state)
                    if parameter.ndim == 2:
                        self._initialize_basis(state)
                    # SOAP deliberately uses the first gradient only to seed its
                    # preconditioner, preventing same-gradient basis leakage.
                    continue
                state["step"] += 1
                step = state["step"]
                projected = self._project(gradient, state) if parameter.ndim == 2 else gradient
                state["exp_avg"].mul_(beta1).add_(projected, alpha=1.0 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(projected, projected, value=1.0 - beta2)
                bc1, bc2 = 1.0 - beta1**step, 1.0 - beta2**step
                update = (state["exp_avg"] / bc1) / (
                    (state["exp_avg_sq"] / bc2).sqrt() + self.config.eps
                )
                if parameter.ndim == 2:
                    update = self._project_back(update, state)
                parameter_step_(
                    parameter,
                    update,
                    group["lr"],
                    group["weight_decay"] if group["decay"] else 0.0,
                    self.native,
                )
                self._update_factors(gradient, state)
                if parameter.ndim == 2 and step % self.config.precondition_frequency == 0:
                    self._refresh_basis(state)
        return loss


def build_optimizer(params: Iterable[Any], config: OptimizerConfig) -> Optimizer:
    name = config.name.lower()
    if name == "adamw":
        return NativeAdamW(params, config)
    if name == "shampoo":
        return ShampooFamily(params, config, root_engine="eigh")
    if name == "rfs":
        return ShampooFamily(params, config, root_engine="rfs")
    if name == "soap":
        return SOAP(params, config)
    raise ValueError(f"Unknown optimizer {config.name!r}; choose adamw, shampoo, soap, or rfs")
