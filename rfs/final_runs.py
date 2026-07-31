from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .config import load_config
from .sweep import override_args
from .train import compute_cost

RESEARCH_START_UNIX = 1_785_494_879  # 2026-07-31 10:47:59 UTC
OPTIMIZERS = ("adamw", "shampoo", "soap", "rfs")
SEEDS = (1, 2, 3)
RUN_ORDER = tuple(
    (OPTIMIZERS[(position + seed_index) % len(OPTIMIZERS)], seed)
    for seed_index, seed in enumerate(SEEDS)
    for position in range(len(OPTIMIZERS))
)


def run_one(
    base_config: str,
    output: Path,
    optimizer: str,
    seed: int,
    selected: dict[str, Any],
) -> dict[str, Any]:
    name = f"{optimizer}-seed{seed}"
    run_dir = output / name
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    overrides = {
        "optimizer.name": optimizer,
        "train.seed": seed,
        "runtime.output_dir": str(output),
        **selected,
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
    checkpoint = run_dir / "checkpoint.pt"
    if checkpoint.exists():
        command.extend(("--resume", str(checkpoint)))
    environment = os.environ.copy()
    environment.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(".torch_cache").resolve()))
    result = subprocess.run(command, env=environment, check=False)
    if result.returncode:
        raise RuntimeError(f"Final run {name} failed with exit code {result.returncode}")
    return json.loads(summary_path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the budget-gated 4 x 3 final matrix")
    parser.add_argument("--base-config", default="configs/base.toml")
    parser.add_argument(
        "--best",
        type=Path,
        default=Path("artifacts/sweeps/matrix_refinement/best.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/final"))
    args = parser.parse_args()
    config = load_config(args.base_config)
    selection = json.loads(args.best.read_text())["best"]
    if set(selection) != set(OPTIMIZERS):
        raise RuntimeError("Selection file must contain all four optimizers")
    args.output.mkdir(parents=True, exist_ok=True)
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    for summary_path in sorted(args.output.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        key = (str(summary["optimizer"]), int(summary["seed"]))
        if key in RUN_ORDER:
            completed[key] = summary

    # Rotate optimizer order by seed to reduce correlation with server age,
    # thermals, and the announced price boundary.
    for optimizer, seed in RUN_ORDER:
        key = (optimizer, seed)
        if key in completed:
            result = completed[key]
        else:
            now = time.time()
            spent = compute_cost(RESEARCH_START_UNIX, now, config.runtime)
            # A deliberately conservative per-run allowance protects the hard
            # ceiling even if a matrix optimizer is slower than its pilot.
            next_run_allowance = 2.5 * config.runtime.future_hourly_cost_usd
            if spent + next_run_allowance > config.runtime.total_budget_usd:
                raise RuntimeError(
                    f"Budget gate: ${spent:.2f} spent and ${next_run_allowance:.2f} "
                    f"reserved for the next run exceeds "
                    f"${config.runtime.total_budget_usd:.2f}"
                )
            result = run_one(
                args.base_config,
                args.output,
                optimizer,
                seed,
                selection[optimizer]["overrides"],
            )
        completed[(optimizer, seed)] = result
        results = [completed[key] for key in RUN_ORDER if key in completed]
        ledger = {
            "research_start_unix": RESEARCH_START_UNIX,
            "updated_unix": time.time(),
            "continuous_server_cost_usd": compute_cost(
                RESEARCH_START_UNIX, time.time(), config.runtime
            ),
            "absolute_budget_usd": config.runtime.total_budget_usd,
            "completed_runs": results,
        }
        (args.output / "budget_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps(ledger, indent=2))


if __name__ == "__main__":
    main()
