from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from rfs.train import compute_cost, truncate_metrics


def test_cost_spans_price_boundary() -> None:
    runtime = SimpleNamespace(
        price_increase_unix=10_000,
        hourly_cost_usd=2.0,
        future_hourly_cost_usd=2.5,
    )
    assert compute_cost(6_400, 13_600, runtime) == pytest.approx(4.5)


def test_metrics_are_truncated_to_checkpoint(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    rows = [{"step": step, "loss": 10.0 - step} for step in range(1, 5)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    truncate_metrics(path, 2)

    retained = [json.loads(line) for line in path.read_text().splitlines()]
    assert retained == rows[:2]
