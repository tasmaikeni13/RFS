from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import torch


def command(*args: str) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=False, timeout=30)
    return result.stdout.strip()


def read_meminfo() -> dict[str, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0]) * 1024
    return values


def main() -> None:
    properties = torch.cuda.get_device_properties(0)
    disk = shutil.disk_usage("/workspace")
    payload: dict[str, Any] = {
        "created_unix": time.time(),
        "host": {
            "hostname": platform.node(),
            "kernel": platform.release(),
            "cpu_count": os.cpu_count(),
            "lscpu": command("lscpu"),
            "memory_bytes": read_meminfo(),
            "workspace_disk_bytes": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
            },
        },
        "accelerator": {
            "name": properties.name,
            "architecture": getattr(properties, "gcnArchName", None),
            "compute_units": properties.multi_processor_count,
            "hbm_bytes": properties.total_memory,
            "rocm_smi": command("rocm-smi", "--showallinfo"),
            "rocminfo": command("rocminfo"),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_hip": torch.version.hip,
            "git_commit": command("git", "rev-parse", "HEAD"),
            "container_image": "rfs-mi300x:rocm7.14-pytorch2.12",
            "base_image": (
                "rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.12.0"
                "@sha256:c38eeda81d85f00fbe35d3d50ce42ce59c524e87d810624f4eb5c52fddb3b9ad"
            ),
        },
    }
    output = Path("artifacts/system_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["accelerator"], indent=2))


if __name__ == "__main__":
    main()
