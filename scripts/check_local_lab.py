from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=20).strip()
    except Exception:
        return None


def _ram_gib() -> float | None:
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                kib = int(line.split()[1])
                return round(kib / 1024**2, 2)
    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return round(stat.ullTotalPhys / 1024**3, 2)
        except Exception:
            return None
    return None


def collect() -> dict:
    payload: dict = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "cpu": platform.processor() or platform.machine(),
        "ram_gib": _ram_gib(),
        "cwd": str(Path.cwd()),
    }

    if shutil.which("git"):
        payload["git_head"] = _run(["git", "rev-parse", "HEAD"])
        payload["git_branch"] = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])

    if shutil.which("nvidia-smi"):
        query = _run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
        payload["nvidia_smi"] = query
    else:
        payload["nvidia_smi"] = None

    try:
        import torch

        payload["torch"] = torch.__version__
        payload["torch_cuda_version"] = torch.version.cuda
        payload["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            payload["gpu_name"] = props.name
            payload["gpu_dedicated_vram_gib"] = round(props.total_memory / 1024**3, 2)
            payload["gpu_compute_capability"] = list(torch.cuda.get_device_capability(0))
        else:
            payload["gpu_name"] = None
            payload["gpu_dedicated_vram_gib"] = None
    except Exception as exc:
        payload["torch_error"] = f"{type(exc).__name__}: {exc}"
        payload["cuda_available"] = False

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture the canonical U-MIRAGE local lab state")
    parser.add_argument("--output", default="artifacts/local_lab/hardware.json")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    payload = collect()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nWrote hardware manifest: {output}")

    if args.require_cuda and not payload.get("cuda_available"):
        raise SystemExit("CUDA is required for the real local gate but is not available.")


if __name__ == "__main__":
    main()
