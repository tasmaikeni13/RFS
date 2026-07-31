from __future__ import annotations

import json
import os
import platform
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import torch


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command, capture_output=True, text=True, timeout=15, check=False
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def system_metadata() -> dict[str, Any]:
    gpu = None
    properties = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        item = torch.cuda.get_device_properties(0)
        properties = {
            "name": item.name,
            "total_memory_bytes": item.total_memory,
            "multi_processor_count": item.multi_processor_count,
            "gcn_arch_name": getattr(item, "gcnArchName", None),
        }
    return {
        "created_unix": time.time(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "gpu": gpu,
        "gpu_properties": properties,
        "rocm_smi": command_output(["rocm-smi", "--showproductname", "--showvbios"]),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(
            command_output(["git", "status", "--porcelain", "--untracked-files=no"])
        ),
        "environment": {
            key: os.getenv(key)
            for key in (
                "PYTORCH_ROCM_ARCH",
                "HIP_VISIBLE_DEVICES",
                "TORCHINDUCTOR_CACHE_DIR",
                "TORCH_EXTENSIONS_DIR",
            )
        },
    }


class TelemetrySampler:
    """Low-overhead sysfs telemetry, sampled independently from GPU synchronization."""

    def __init__(
        self, output: Path, interval_seconds: float = 5.0, session_id: str | None = None
    ) -> None:
        self.output = output
        self.interval = interval_seconds
        self.session_id = session_id or f"{time.time_ns()}"
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="gpu-telemetry", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=self.interval + 2)

    @staticmethod
    def _read_number(pattern: str, scale: float = 1.0) -> float | None:
        import glob

        for name in glob.glob(pattern):
            try:
                return float(Path(name).read_text().strip()) / scale
            except (OSError, ValueError):
                continue
        return None

    @classmethod
    def _read_first(cls, patterns: tuple[str, ...], scale: float) -> float | None:
        for pattern in patterns:
            value = cls._read_number(pattern, scale)
            if value is not None:
                return value
        return None

    def _run(self) -> None:
        while not self.stop_event.is_set():
            sample = {
                "session_id": self.session_id,
                "unix": time.time(),
                "monotonic_s": time.monotonic(),
                "gpu_busy_percent": self._read_number(
                    "/sys/class/drm/card*/device/gpu_busy_percent"
                ),
                "power_w": self._read_first(
                    (
                        "/sys/class/drm/card*/device/hwmon/hwmon*/power1_average",
                        "/sys/class/drm/card*/device/hwmon/hwmon*/power1_input",
                    ),
                    1_000_000,
                ),
                "junction_temp_c": self._read_first(
                    (
                        "/sys/class/drm/card*/device/hwmon/hwmon*/temp2_input",
                        "/sys/class/drm/card*/device/hwmon/hwmon*/temp1_input",
                    ),
                    1_000,
                ),
                "vram_used_bytes": self._read_number(
                    "/sys/class/drm/card*/device/mem_info_vram_used"
                ),
            }
            append_jsonl(self.output, sample)
            self.stop_event.wait(self.interval)
