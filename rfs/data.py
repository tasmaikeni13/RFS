from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class PackedTokenDataset:
    """Deterministic contiguous batches from a uint16 GPT-2 token stream."""

    def __init__(self, root: str | Path, split: str, context_length: int) -> None:
        self.root = Path(root)
        self.split = split
        self.context_length = context_length
        metadata_path = self.root / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing {metadata_path}. Run `python -m rfs.prepare_fineweb` first."
            )
        self.metadata = json.loads(metadata_path.read_text())
        expected = int(self.metadata["splits"][split]["tokens"])
        path = self.root / f"{split}.bin"
        actual = path.stat().st_size // np.dtype(np.uint16).itemsize
        if actual != expected:
            raise RuntimeError(f"{path} contains {actual:,} tokens, expected {expected:,}")
        self.tokens = np.memmap(path, mode="r", dtype=np.uint16)

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    def batch(
        self,
        batch_size: int,
        batch_index: int,
        device: torch.device,
        seed: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = batch_size * self.context_length
        usable = len(self) - 1
        if count > usable:
            raise ValueError(f"Batch needs {count + 1:,} tokens but {self.split} has {len(self):,}")
        # Each seed is a cyclic rotation of the same stream, preserving the token
        # multiset while changing sequence boundaries and minibatch order.
        seed_offset = (seed * 104_729 * self.context_length) % usable
        start = (seed_offset + batch_index * count) % usable
        end = start + count + 1
        if end <= len(self):
            chunk = np.asarray(self.tokens[start:end])
        else:
            first = np.asarray(self.tokens[start:])
            second = np.asarray(self.tokens[: end - len(self)])
            chunk = np.concatenate((first, second))
        # Copying avoids exposing a read-only memmap to PyTorch; host-to-device
        # transfer is tiny relative to the transformer step.
        values = torch.from_numpy(np.array(chunk, dtype=np.int64, copy=True))
        x = values[:-1].view(batch_size, self.context_length)
        y = values[1:].view(batch_size, self.context_length)
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
