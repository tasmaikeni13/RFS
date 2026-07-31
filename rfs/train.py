from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import config_parser, load_config
from .data import PackedTokenDataset
from .logging import TelemetrySampler, append_jsonl, system_metadata
from .model import DecoderTransformer, parameter_groups
from .optimizers import build_optimizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def learning_rate(
    tokens: int, max_tokens: int, warmup_tokens: int, peak: float, floor: float
) -> float:
    if tokens < warmup_tokens:
        return peak * (tokens + 1) / max(1, warmup_tokens)
    progress = min(1.0, (tokens - warmup_tokens) / max(1, max_tokens - warmup_tokens))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (floor + (1.0 - floor) * cosine)


def compute_cost(start_unix: float, end_unix: float, runtime: Any) -> float:
    """Price elapsed GPU time across the announced August 1 rate boundary."""
    boundary = float(runtime.price_increase_unix)
    old_seconds = max(0.0, min(end_unix, boundary) - start_unix)
    new_seconds = max(0.0, end_unix - max(start_unix, boundary))
    return (
        old_seconds * runtime.hourly_cost_usd + new_seconds * runtime.future_hourly_cost_usd
    ) / 3600.0


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: PackedTokenDataset,
    batch_size: int,
    eval_tokens: int,
    device: torch.device,
) -> float:
    model.eval()
    tokens_per_batch = batch_size * dataset.context_length
    steps = max(1, min(len(dataset) // tokens_per_batch, eval_tokens // tokens_per_batch))
    losses = []
    for step in range(steps):
        x, y = dataset.batch(batch_size, step, device, seed=0)
        _, loss = model(x, y)
        losses.append(loss.detach().float())
    value = float(torch.stack(losses).mean())
    model.train()
    return value


def save_checkpoint(
    path: Path,
    model: DecoderTransformer,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokens: int,
    config: dict[str, Any],
    unix_start: float,
) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "tokens": tokens,
            "config": config,
            "unix_start": unix_start,
            "rng_cpu": torch.get_rng_state(),
            "rng_gpu": torch.cuda.get_rng_state(),
        },
        temporary,
    )
    os.replace(temporary, path)


def truncate_metrics(path: Path, maximum_step: int) -> None:
    """Discard rows newer than a checkpoint before an interrupted run resumes."""
    if not path.exists():
        return
    retained = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record.get("step", 0)) <= maximum_step:
                retained.append(json.dumps(record, sort_keys=True))
    temporary = path.with_suffix(".tmp")
    temporary.write_text("\n".join(retained) + ("\n" if retained else ""))
    os.replace(temporary, path)


def main() -> None:
    parser = config_parser("Train the 125M decoder-only benchmark")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    if config.optimizer.name not in {"adamw", "shampoo", "soap", "rfs"}:
        raise ValueError("optimizer.name must be adamw, shampoo, soap, or rfs")
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("This benchmark requires a ROCm GPU")
    device = torch.device("cuda", 0)
    set_seed(config.train.seed)

    train_data = PackedTokenDataset(config.data.root, "train", config.model.context_length)
    val_data = PackedTokenDataset(config.data.root, "val", config.model.context_length)
    test_data = PackedTokenDataset(config.data.root, "test", config.model.context_length)
    run_name = args.run_name or (
        f"{config.optimizer.name}-seed{config.train.seed}-{config.fingerprint()}"
    )
    run_dir = Path(config.runtime.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config.as_dict(), indent=2) + "\n")
    metadata = system_metadata()
    metadata["run_name"] = run_name
    metadata["dataset"] = train_data.metadata
    (run_dir / "system.json").write_text(json.dumps(metadata, indent=2) + "\n")

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[config.train.precision]
    raw_model = DecoderTransformer(config.model).to(device=device, dtype=dtype)
    optimizer = build_optimizer(parameter_groups(raw_model), config.optimizer)
    start_step = 0
    trained_tokens = 0
    unix_start = time.time()
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        trained_tokens = int(checkpoint["tokens"])
        unix_start = float(checkpoint.get("unix_start", unix_start))
        torch.set_rng_state(checkpoint["rng_cpu"].cpu())
        torch.cuda.set_rng_state(checkpoint["rng_gpu"].to(device))
        truncate_metrics(run_dir / "metrics.jsonl", start_step)

    model: torch.nn.Module = raw_model
    if config.train.compile:
        model = torch.compile(raw_model, mode="max-autotune-no-cudagraphs", dynamic=False)
    model.train()
    tokens_per_micro = config.train.micro_batch_size * config.model.context_length
    tokens_per_step = tokens_per_micro * config.train.gradient_accumulation_steps
    total_steps = config.train.max_tokens // tokens_per_step
    if total_steps <= 0:
        raise ValueError("max_tokens must cover at least one optimizer step")
    if start_step >= total_steps:
        raise ValueError("Checkpoint already reached the configured token budget")

    failed_marker = run_dir / "FAILED"
    failed_marker.unlink(missing_ok=True)
    telemetry = TelemetrySampler(
        run_dir / "telemetry.jsonl", session_id=f"{time.time_ns()}"
    )
    telemetry.start()
    torch.cuda.reset_peak_memory_stats()
    wall_start = time.monotonic()
    last_end = wall_start
    status = "failed"
    try:
        for step in range(start_step, total_steps):
            lr = learning_rate(
                trained_tokens,
                config.train.max_tokens,
                config.train.warmup_tokens,
                config.optimizer.lr,
                config.train.min_lr_ratio,
            )
            set_lr(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)
            train_start = time.monotonic()
            losses = []
            for accumulation in range(config.train.gradient_accumulation_steps):
                micro_index = step * config.train.gradient_accumulation_steps + accumulation
                x, y = train_data.batch(
                    config.train.micro_batch_size,
                    micro_index,
                    device,
                    seed=config.train.seed,
                )
                _, loss = model(x, y)
                (loss / config.train.gradient_accumulation_steps).backward()
                losses.append(loss.detach().float())
            torch.cuda.synchronize()
            train_ms = (time.monotonic() - train_start) * 1000
            grad_norm = torch.nn.utils.clip_grad_norm_(
                raw_model.parameters(), config.train.grad_clip
            )
            optimizer_start = time.monotonic()
            optimizer.step()
            torch.cuda.synchronize()
            optimizer_ms = (time.monotonic() - optimizer_start) * 1000
            trained_tokens += tokens_per_step
            now = time.monotonic()
            step_ms = (now - last_end) * 1000
            last_end = now
            train_loss = float(torch.stack(losses).mean())
            metrics: dict[str, Any] = {
                "kind": "train",
                "step": step + 1,
                "tokens": trained_tokens,
                "lr": lr,
                "train_loss": train_loss,
                "perplexity": math.exp(min(20.0, train_loss)),
                "grad_norm": float(grad_norm),
                "train_ms": train_ms,
                "optimizer_ms": optimizer_ms,
                "step_ms": step_ms,
                "tokens_per_second": tokens_per_step / (step_ms / 1000),
                "wall_seconds": time.time() - unix_start,
                "estimated_cost_usd": compute_cost(unix_start, time.time(), config.runtime),
                "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            }
            diagnostics = getattr(optimizer, "last_diagnostics", {})
            metrics.update(diagnostics)
            if (step + 1) % config.train.eval_interval == 0 or step + 1 == total_steps:
                eval_start = time.monotonic()
                metrics["val_loss"] = evaluate(
                    model,
                    val_data,
                    config.train.micro_batch_size,
                    config.train.eval_tokens,
                    device,
                )
                metrics["val_perplexity"] = math.exp(min(20.0, metrics["val_loss"]))
                metrics["eval_ms"] = (time.monotonic() - eval_start) * 1000
            append_jsonl(run_dir / "metrics.jsonl", metrics)
            if "eval_ms" in metrics:
                # Keep validation overhead explicit instead of charging it to the
                # following training step's throughput.
                last_end = time.monotonic()
            if (step + 1) % config.train.log_interval == 0:
                val_text = f" val={metrics['val_loss']:.4f}" if "val_loss" in metrics else ""
                print(
                    f"{run_name} step={step + 1}/{total_steps} tokens={trained_tokens:,} "
                    f"loss={train_loss:.4f}{val_text} tok/s={metrics['tokens_per_second']:,.0f} "
                    f"opt={optimizer_ms:.1f}ms cost=${metrics['estimated_cost_usd']:.3f}",
                    flush=True,
                )
            if config.train.checkpoint_interval and (
                (step + 1) % config.train.checkpoint_interval == 0
            ):
                save_checkpoint(
                    run_dir / "checkpoint.pt",
                    raw_model,
                    optimizer,
                    step + 1,
                    trained_tokens,
                    config.as_dict(),
                    unix_start,
                )
        test_loss = evaluate(
            model,
            test_data,
            config.train.micro_batch_size,
            config.data.test_tokens,
            device,
        )
        wall_seconds = time.time() - unix_start
        summary = {
            "status": "complete",
            "run_name": run_name,
            "optimizer": config.optimizer.name,
            "seed": config.train.seed,
            "steps": total_steps,
            "trained_tokens": trained_tokens,
            "parameter_count": raw_model.parameter_count,
            "test_loss": test_loss,
            "test_perplexity": math.exp(min(20.0, test_loss)),
            "wall_seconds": wall_seconds,
            "cost_usd": compute_cost(unix_start, time.time(), config.runtime),
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if config.runtime.save_final_checkpoint:
            save_checkpoint(
                run_dir / "final.pt",
                raw_model,
                optimizer,
                total_steps,
                trained_tokens,
                config.as_dict(),
                unix_start,
            )
        status = "complete"
        failed_marker.unlink(missing_ok=True)
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        telemetry.stop()
        if status != "complete":
            failed_marker.write_text(f"aborted_unix={time.time()}\n")


if __name__ == "__main__":
    main()
