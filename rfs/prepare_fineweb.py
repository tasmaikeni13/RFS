from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import tiktoken
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
SOURCE_PREFIX = "sample/10BT"
DEFAULT_COUNTS = {"train": 998_000_000, "val": 1_000_000, "test": 1_000_000}


def document_split(document_id: str) -> str:
    """Stable 99.8/0.1/0.1 document-level split independent of row order."""
    value = int.from_bytes(hashlib.blake2b(document_id.encode(), digest_size=8).digest(), "big")
    bucket = value % 1000
    if bucket == 0:
        return "val"
    if bucket == 1:
        return "test"
    return "train"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def digest_file(path: Path, chunk_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(
    output: Path,
    counts: dict[str, int],
    cache: Path,
    batch_rows: int,
    workers: int,
    keep_raw: bool,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    progress: dict[str, Any]
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress["targets"] != counts:
            raise RuntimeError("Existing progress targets differ; use a new output directory")
    else:
        progress = {
            "targets": counts,
            "written": {name: 0 for name in counts},
            "documents": {name: 0 for name in counts},
            "shard_index": 0,
            "row_index": 0,
            "started_unix": time.time(),
        }
        atomic_json(progress_path, progress)

    handles = {}
    for split in counts:
        path = output / f"{split}.bin"
        path.touch(exist_ok=True)
        expected_bytes = int(progress["written"][split]) * 2
        with path.open("r+b") as handle:
            handle.truncate(expected_bytes)
        handles[split] = path.open("ab", buffering=8 << 20)

    tokenizer = tiktoken.get_encoding("gpt2")
    eos = tokenizer.eot_token
    api = HfApi()
    info = api.dataset_info(REPO_ID)
    revision = info.sha
    files = sorted(
        item.rfilename
        for item in info.siblings
        if item.rfilename.startswith(SOURCE_PREFIX) and item.rfilename.endswith(".parquet")
    )
    if not files:
        raise RuntimeError(f"No Parquet shards found under {SOURCE_PREFIX}")

    try:
        for shard_index in range(int(progress["shard_index"]), len(files)):
            if all(int(progress["written"][key]) >= counts[key] for key in counts):
                break
            filename = files[shard_index]
            local = Path(
                hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=filename,
                    revision=revision,
                    cache_dir=cache,
                )
            )
            parquet = pq.ParquetFile(local)
            absolute_row = 0
            resume_row = int(progress["row_index"]) if shard_index == progress["shard_index"] else 0
            for record_batch in parquet.iter_batches(batch_size=batch_rows, columns=["id", "text"]):
                next_row = absolute_row + record_batch.num_rows
                if next_row <= resume_row:
                    absolute_row = next_row
                    continue
                ids = record_batch.column(0).to_pylist()
                texts = record_batch.column(1).to_pylist()
                skip = max(0, resume_row - absolute_row)
                ids, texts = ids[skip:], texts[skip:]
                token_batches = tokenizer.encode_batch(
                    texts,
                    num_threads=workers,
                    allowed_special=set(),
                    disallowed_special=(),
                )
                for document_id, tokens in zip(ids, token_batches, strict=True):
                    split = document_split(document_id)
                    remaining = counts[split] - int(progress["written"][split])
                    if remaining <= 0:
                        continue
                    tokens.append(eos)
                    array = np.asarray(tokens[:remaining], dtype=np.uint16)
                    handles[split].write(array.tobytes())
                    progress["written"][split] += int(array.size)
                    progress["documents"][split] += 1
                absolute_row = next_row
                progress["shard_index"] = shard_index
                progress["row_index"] = absolute_row
                for handle in handles.values():
                    handle.flush()
                atomic_json(progress_path, progress)
                written = sum(progress["written"].values())
                print(
                    f"shard={shard_index:02d} rows={absolute_row:,} "
                    f"tokens={written:,}/{sum(counts.values()):,}",
                    flush=True,
                )
            progress["shard_index"] = shard_index + 1
            progress["row_index"] = 0
            atomic_json(progress_path, progress)
            if not keep_raw:
                # Hugging Face's cache uses links and shared blobs; removing it here
                # is unsafe. `keep_raw` is retained as manifest intent and cleanup is
                # performed explicitly after verification.
                pass
    finally:
        for handle in handles.values():
            handle.close()

    incomplete = {key: counts[key] - int(progress["written"][key]) for key in counts}
    if any(value for value in incomplete.values()):
        raise RuntimeError(f"Source exhausted before quotas were met: {incomplete}")

    split_metadata = {}
    for split, count in counts.items():
        path = output / f"{split}.bin"
        split_metadata[split] = {
            "tokens": count,
            "documents": int(progress["documents"][split]),
            "bytes": path.stat().st_size,
            "sha256": digest_file(path),
        }
    metadata = {
        "name": "FineWeb-Edu 1B GPT-2-token corpus",
        "source": {"repo_id": REPO_ID, "config": DATASET_CONFIG, "revision": revision},
        "tokenizer": {"name": "gpt2", "vocab_size": tokenizer.n_vocab, "eos_token": eos},
        "split_method": "blake2b(document id) modulo 1000: val=0, test=1, train=2..999",
        "total_tokens": sum(counts.values()),
        "splits": split_metadata,
        "created_unix": time.time(),
    }
    atomic_json(output / "metadata.json", metadata)
    progress_path.unlink(missing_ok=True)
    print(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an exact 1B-token FineWeb-Edu corpus")
    parser.add_argument("--output", type=Path, default=Path("data/fineweb_edu_1b_gpt2"))
    parser.add_argument("--cache", type=Path, default=Path("artifacts/raw/huggingface"))
    parser.add_argument("--train-tokens", type=int, default=DEFAULT_COUNTS["train"])
    parser.add_argument("--val-tokens", type=int, default=DEFAULT_COUNTS["val"])
    parser.add_argument("--test-tokens", type=int, default=DEFAULT_COUNTS["test"])
    parser.add_argument("--batch-rows", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=min(20, os.cpu_count() or 1))
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()
    counts = {
        "train": args.train_tokens,
        "val": args.val_tokens,
        "test": args.test_tokens,
    }
    if sum(counts.values()) != 1_000_000_000 and counts == DEFAULT_COUNTS:
        raise AssertionError("Default corpus must contain exactly one billion tokens")
    prepare(args.output, counts, args.cache, args.batch_rows, args.workers, args.keep_raw)


if __name__ == "__main__":
    main()
