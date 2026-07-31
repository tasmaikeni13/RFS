from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def override_args(values: dict[str, Any]) -> list[str]:
    result = []
    for key, value in values.items():
        if isinstance(value, bool):
            value = str(value).lower()
        result.extend(("--set", f"{key}={value}"))
    return result


def read_validation(run_dir: Path) -> float:
    validation = None
    with (run_dir / "metrics.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            if "val_loss" in record:
                validation = float(record["val_loss"])
    if validation is None:
        raise RuntimeError(f"No validation metric in {run_dir}")
    return validation


def run_trial(
    base_config: str,
    root: Path,
    stage: int,
    optimizer: str,
    index: int,
    tokens: int,
    values: dict[str, Any],
) -> dict[str, Any]:
    values = dict(values)
    if "optimizer.precondition_frequency" in values:
        values.setdefault("optimizer.start_preconditioning_step", 10)
    name = f"stage{stage}-{optimizer}-{index:02d}"
    run_dir = root / name
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        return {
            "name": name,
            "optimizer": optimizer,
            "overrides": values,
            "val_loss": read_validation(run_dir),
            **summary,
        }
    overrides = {
        "optimizer.name": optimizer,
        "train.seed": 1234,
        "train.max_tokens": tokens,
        "train.warmup_tokens": max(1_000_000, tokens // 20),
        "train.eval_interval": 1000000,
        "train.checkpoint_interval": 0,
        "runtime.output_dir": str(root),
        "runtime.save_final_checkpoint": False,
        **values,
    }
    command = [
        sys.executable,
        "-m",
        "rfs.train",
        "--config",
        base_config,
        "--run-name",
        name,
        *override_args(overrides),
    ]
    environment = os.environ.copy()
    environment.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(".torch_cache").resolve()))
    started = time.time()
    result = subprocess.run(command, env=environment, check=False)
    if result.returncode:
        raise RuntimeError(f"Trial {name} failed with exit code {result.returncode}")
    summary = json.loads(summary_path.read_text())
    orchestration_seconds = time.time() - started
    return {
        "name": name,
        "optimizer": optimizer,
        "overrides": values,
        "val_loss": read_validation(run_dir),
        "orchestration_seconds": orchestration_seconds,
        "orchestration_cost_usd": orchestration_seconds / 1800.0,
        **summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Budgeted two-stage optimizer sweep")
    parser.add_argument("--base-config", default="configs/base.toml")
    parser.add_argument("--grid", type=Path, default=Path("configs/sweep.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/sweeps/125m_1b"))
    args = parser.parse_args()
    grid = json.loads(args.grid.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []
    spent = 0.0

    stage1: dict[str, list[dict[str, Any]]] = {}
    for optimizer, trials in grid["optimizers"].items():
        stage1[optimizer] = []
        for index, values in enumerate(trials):
            result = run_trial(
                args.base_config,
                args.output,
                1,
                optimizer,
                index,
                grid["stage1_tokens"],
                values,
            )
            stage1[optimizer].append(result)
            all_results.append(result)
            spent += float(result.get("orchestration_cost_usd", result["cost_usd"]))
            (args.output / "sweep_results.json").write_text(
                json.dumps(all_results, indent=2) + "\n"
            )
            if spent >= grid["sweep_budget_usd"]:
                raise RuntimeError(f"Sweep hard cap reached at ${spent:.2f}")

    best: dict[str, dict[str, Any]] = {}
    for optimizer, results in stage1.items():
        finalists = sorted(results, key=lambda item: item["val_loss"])[: grid["keep_per_optimizer"]]
        stage2 = []
        for index, finalist in enumerate(finalists):
            result = run_trial(
                args.base_config,
                args.output,
                2,
                optimizer,
                index,
                grid["stage2_tokens"],
                finalist["overrides"],
            )
            stage2.append(result)
            all_results.append(result)
            spent += float(result.get("orchestration_cost_usd", result["cost_usd"]))
            (args.output / "sweep_results.json").write_text(
                json.dumps(all_results, indent=2) + "\n"
            )
            if spent >= grid["sweep_budget_usd"]:
                raise RuntimeError(f"Sweep hard cap reached at ${spent:.2f}")
        best[optimizer] = min(stage2, key=lambda item: item["val_loss"])

    output = {
        "spent_usd": spent,
        "best": best,
        "selection_metric": "lowest validation cross-entropy after stage2_tokens",
    }
    (args.output / "best.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
