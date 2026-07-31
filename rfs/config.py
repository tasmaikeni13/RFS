from __future__ import annotations

import argparse
import dataclasses
import json
import tomllib
from pathlib import Path
from typing import Any


@dataclasses.dataclass
class ModelConfig:
    vocab_size: int = 50257
    context_length: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


@dataclasses.dataclass
class DataConfig:
    root: str = "data/fineweb_edu_1b_gpt2"
    train_tokens: int = 998_000_000
    val_tokens: int = 1_000_000
    test_tokens: int = 1_000_000


@dataclasses.dataclass
class TrainConfig:
    seed: int = 1
    micro_batch_size: int = 64
    gradient_accumulation_steps: int = 8
    max_tokens: int = 998_000_000
    warmup_tokens: int = 10_000_000
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_tokens: int = 1_000_000
    log_interval: int = 20
    checkpoint_interval: int = 500
    compile: bool = True
    precision: str = "bfloat16"


@dataclasses.dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 6e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    eps: float = 1e-8
    shampoo_beta: float = 0.95
    matrix_eps: float = 1e-3
    max_preconditioner_dim: int = 1024
    precondition_frequency: int = 10
    start_preconditioning_step: int = 10
    root_iterations: int = 60
    graft: bool = True
    use_hip_kernels: bool = True


@dataclasses.dataclass
class RuntimeConfig:
    output_dir: str = "artifacts/runs"
    hourly_cost_usd: float = 2.0
    price_increase_unix: int = 1_785_522_600
    future_hourly_cost_usd: float = 2.5
    total_budget_usd: float = 75.0
    budget_reserve_fraction: float = 0.1
    save_final_checkpoint: bool = False


@dataclasses.dataclass
class ExperimentConfig:
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
    runtime: RuntimeConfig = dataclasses.field(default_factory=RuntimeConfig)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        import hashlib

        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


_SECTIONS = {
    "model": ModelConfig,
    "data": DataConfig,
    "train": TrainConfig,
    "optimizer": OptimizerConfig,
    "runtime": RuntimeConfig,
}


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    for override in overrides or []:
        key, raw_value = override.split("=", 1)
        section, field = key.split(".", 1)
        if section not in _SECTIONS:
            raise KeyError(f"Unknown config section: {section}")
        raw.setdefault(section, {})[field] = _parse_value(raw_value)
    sections: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        allowed = {item.name for item in dataclasses.fields(cls)}
        unknown = set(raw.get(name, {})) - allowed
        if unknown:
            raise KeyError(f"Unknown {name} keys: {sorted(unknown)}")
        sections[name] = cls(**raw.get(name, {}))
    return ExperimentConfig(**sections)


def _parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def config_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/base.toml")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override a TOML value; may be repeated.",
    )
    return parser
