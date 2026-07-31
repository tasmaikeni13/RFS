from __future__ import annotations

from rfs.prepare_fineweb import document_split


def test_document_split_is_stable_and_disjoint() -> None:
    values = [document_split(f"document-{index}") for index in range(100_000)]
    assert values == [document_split(f"document-{index}") for index in range(100_000)]
    counts = {split: values.count(split) for split in ("train", "val", "test")}
    assert 99_500 < counts["train"] < 99_950
    assert 50 < counts["val"] < 150
    assert 50 < counts["test"] < 150
