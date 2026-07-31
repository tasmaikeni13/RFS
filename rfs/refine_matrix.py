from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .sweep import run_trial


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refine damping and activation for Shampoo and RFS"
    )
    parser.add_argument("--base-config", default="configs/base.toml")
    parser.add_argument(
        "--grid", type=Path, default=Path("configs/matrix_refinement.json")
    )
    parser.add_argument(
        "--initial-best", type=Path, default=Path("artifacts/sweeps/125m_1b/best.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/sweeps/matrix_refinement")
    )
    args = parser.parse_args()
    grid = json.loads(args.grid.read_text())
    initial = json.loads(args.initial_best.read_text())
    args.output.mkdir(parents=True, exist_ok=True)

    trials: list[dict[str, Any]] = []
    spent = 0.0
    for optimizer in grid["optimizers"]:
        for index, settings in enumerate(grid["settings"]):
            result = run_trial(
                args.base_config,
                args.output,
                3,
                optimizer,
                index,
                int(grid["tokens"]),
                settings,
            )
            trials.append(result)
            spent += float(result.get("orchestration_cost_usd", result["cost_usd"]))
            (args.output / "refinement_results.json").write_text(
                json.dumps(trials, indent=2) + "\n"
            )
            if spent >= float(grid["budget_usd"]):
                raise RuntimeError(f"Matrix refinement hard cap reached at ${spent:.2f}")

    chosen = dict(initial["best"])
    selection_pool: dict[str, list[dict[str, Any]]] = {}
    for optimizer in grid["optimizers"]:
        candidates = [initial["best"][optimizer]] + [
            trial for trial in trials if trial["optimizer"] == optimizer
        ]
        selection_pool[optimizer] = candidates
        chosen[optimizer] = min(candidates, key=lambda item: item["val_loss"])

    result = {
        "spent_usd": spent,
        "initial_sweep": str(args.initial_best),
        "best": chosen,
        "selection_pool": selection_pool,
        "selection_metric": "lowest validation cross-entropy after 100M tokens",
        "reason": (
            "Matched refinement of activation timing and scale-relative damping "
            "after an early-preconditioning loss transient in Shampoo and RFS."
        ),
    }
    (args.output / "best.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
