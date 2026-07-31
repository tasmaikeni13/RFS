from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify token counts and SHA-256 corpus manifest")
    parser.add_argument("root", type=Path, nargs="?", default=Path("data/fineweb_edu_1b_gpt2"))
    args = parser.parse_args()
    metadata = json.loads((args.root / "metadata.json").read_text())
    verified = {}
    for split, expected in metadata["splits"].items():
        path = args.root / f"{split}.bin"
        actual = {"bytes": path.stat().st_size, "sha256": digest(path)}
        if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
            raise RuntimeError(f"{split} failed verification: {actual} != {expected}")
        verified[split] = actual
    if sum(item["tokens"] for item in metadata["splits"].values()) != 1_000_000_000:
        raise RuntimeError("Corpus manifest is not exactly one billion tokens")
    print(json.dumps({"status": "verified", "splits": verified}, indent=2))


if __name__ == "__main__":
    main()
