from __future__ import annotations

from rfs.config import load_config
from rfs.model import DecoderTransformer


def test_base_model_parameter_count() -> None:
    config = load_config("configs/base.toml")
    model = DecoderTransformer(config.model)
    assert model.parameter_count == 124_439_808


def test_overrides_are_typed() -> None:
    config = load_config(
        "configs/base.toml",
        ["train.seed=9", "train.compile=false", "optimizer.lr=0.001"],
    )
    assert config.train.seed == 9
    assert config.train.compile is False
    assert config.optimizer.lr == 0.001
