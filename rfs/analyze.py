from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

COLORS = {"adamw": "#4C78A8", "shampoo": "#F58518", "soap": "#54A24B", "rfs": "#E45756"}
LABELS = {"adamw": "AdamW", "shampoo": "Shampoo", "soap": "SOAP", "rfs": "RFS"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def smooth(values: np.ndarray, width: int = 21) -> np.ndarray:
    if values.size < width:
        return values
    kernel = np.ones(width) / width
    padded = np.pad(values, (width // 2, width // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def save_figure(figure: plt.Figure, output: Path, name: str) -> None:
    figure.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight")
    figure.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate final runs and render paper figures")
    parser.add_argument("--runs", type=Path, default=Path("artifacts/final"))
    parser.add_argument("--output", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary_path in sorted(args.runs.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        directory = summary_path.parent
        summary["metrics"] = read_jsonl(directory / "metrics.jsonl")
        summary["telemetry"] = read_jsonl(directory / "telemetry.jsonl")
        runs[summary["optimizer"]].append(summary)
    if set(runs) != set(COLORS) or any(len(items) != 3 for items in runs.values()):
        raise RuntimeError("Analysis requires three complete seeds for all four optimizers")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    for optimizer, color in COLORS.items():
        items = runs[optimizer]
        token_axis = np.array([record["tokens"] for record in items[0]["metrics"]]) / 1e9
        train = np.stack(
            [
                smooth(np.array([record["train_loss"] for record in item["metrics"]]))
                for item in items
            ]
        )
        axes[0].plot(token_axis, train.mean(0), label=LABELS[optimizer], color=color)
        axes[0].fill_between(
            token_axis,
            train.mean(0) - train.std(0, ddof=1),
            train.mean(0) + train.std(0, ddof=1),
            color=color,
            alpha=0.16,
        )
        validation = [[r for r in item["metrics"] if "val_loss" in r] for item in items]
        val_tokens = np.array([record["tokens"] for record in validation[0]]) / 1e9
        val_loss = np.stack([[record["val_loss"] for record in item] for item in validation])
        axes[1].errorbar(
            val_tokens,
            val_loss.mean(0),
            yerr=val_loss.std(0, ddof=1),
            label=LABELS[optimizer],
            color=color,
            marker="o",
            markersize=3,
            capsize=2,
        )
    for axis, title, ylabel in zip(
        axes,
        ("Training loss (21-step moving mean)", "Held-out validation loss"),
        ("Cross-entropy", "Cross-entropy"),
        strict=True,
    ):
        axis.set(title=title, xlabel="Training tokens (billions)", ylabel=ylabel)
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=2)
    save_figure(figure, args.output, "learning_curves")

    rows = []
    for optimizer in COLORS:
        items = runs[optimizer]
        summaries = {
            key: np.array([float(item[key]) for item in items])
            for key in (
                "test_loss",
                "wall_seconds",
                "cost_usd",
                "peak_vram_bytes",
                "trained_tokens",
            )
        }
        step_rates = np.concatenate(
            [[record["tokens_per_second"] for record in item["metrics"][1:]] for item in items]
        )
        optimizer_ms = np.concatenate(
            [[record["optimizer_ms"] for record in item["metrics"][1:]] for item in items]
        )
        telemetry = [record for item in items for record in item["telemetry"]]
        powers = np.array([record["power_w"] for record in telemetry if record.get("power_w")])
        busy = np.array(
            [record["gpu_busy_percent"] for record in telemetry if record.get("gpu_busy_percent") is not None]
        )
        temperatures = np.array(
            [record["junction_temp_c"] for record in telemetry if record.get("junction_temp_c")]
        )
        energies = []
        for item in items:
            sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for record in item["telemetry"]:
                if record.get("power_w"):
                    sessions[str(record.get("session_id", "legacy"))].append(record)
            energy = 0.0
            measured = False
            for samples in sessions.values():
                if len(samples) >= 2:
                    measured = True
                    energy += float(
                        np.trapezoid(
                            [record["power_w"] for record in samples],
                            [record["unix"] for record in samples],
                        )
                        / 3.6e6
                    )
            if measured:
                energies.append(energy)
        rows.append(
            {
                "optimizer": optimizer,
                "test_loss_mean": summaries["test_loss"].mean(),
                "test_loss_std": summaries["test_loss"].std(ddof=1),
                "wall_hours_mean": summaries["wall_seconds"].mean() / 3600,
                "wall_hours_std": summaries["wall_seconds"].std(ddof=1) / 3600,
                "cost_usd_mean": summaries["cost_usd"].mean(),
                "throughput_median_tokens_s": np.median(step_rates),
                "optimizer_ms_median": np.median(optimizer_ms),
                "optimizer_ms_p95": np.quantile(optimizer_ms, 0.95),
                "peak_vram_gib_mean": summaries["peak_vram_bytes"].mean() / 2**30,
                "power_w_mean": powers.mean() if powers.size else np.nan,
                "energy_kwh_mean": np.mean(energies) if energies else np.nan,
                "gpu_busy_percent_mean": busy.mean() if busy.size else np.nan,
                "junction_temp_c_max": temperatures.max() if temperatures.size else np.nan,
            }
        )

    with (args.output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    table_lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Optimizer & Test loss & Wall (h) & kTok/s & Peak GiB \\",
        r"\midrule",
    ]
    for row in rows:
        table_lines.append(
            f"{LABELS[row['optimizer']]} & "
            f"{row['test_loss_mean']:.4f} $\\pm$ {row['test_loss_std']:.4f} & "
            f"{row['wall_hours_mean']:.3f} & "
            f"{row['throughput_median_tokens_s'] / 1e3:.1f} & "
            f"{row['peak_vram_gib_mean']:.1f} \\\\"
        )
    table_lines.extend((r"\bottomrule", r"\end{tabular}"))
    (args.output / "results_table.tex").write_text("\n".join(table_lines) + "\n")

    labels = [LABELS[row["optimizer"]] for row in rows]
    colors = [COLORS[row["optimizer"]] for row in rows]
    x = np.arange(len(rows))
    figure, axes_grid = plt.subplots(2, 2, figsize=(9.5, 7.0))
    axes = axes_grid.flatten()
    test_means = np.array([row["test_loss_mean"] for row in rows])
    test_stds = np.array([row["test_loss_std"] for row in rows])
    axes[0].errorbar(
        x,
        test_means,
        yerr=test_stds,
        ecolor="#444444",
        fmt="none",
        capsize=4,
    )
    axes[0].scatter(x, test_means, c=colors, s=46, zorder=3)
    margin = max(0.01, 0.15 * (test_means.max() - test_means.min() + test_stds.max()))
    axes[0].set_ylim((test_means - test_stds).min() - margin, (test_means + test_stds).max() + margin)
    axes[1].bar(
        x,
        [row["wall_hours_mean"] for row in rows],
        color=colors,
        yerr=[row["wall_hours_std"] for row in rows],
        capsize=3,
    )
    axes[2].bar(x, [row["throughput_median_tokens_s"] / 1e3 for row in rows], color=colors)
    axes[3].bar(x, [row["energy_kwh_mean"] for row in rows], color=colors)
    for axis, title, ylabel in zip(
        axes,
        ("Final test loss", "End-to-end wall time", "Steady-state throughput", "GPU energy"),
        ("Cross-entropy", "Hours", "Thousands of tokens/s", "kWh per run"),
        strict=True,
    ):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, labels, rotation=20)
        axis.grid(axis="y", alpha=0.25)
    save_figure(figure, args.output, "outcomes_and_systems")


if __name__ == "__main__":
    main()
