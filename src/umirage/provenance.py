from __future__ import annotations
import json, platform, subprocess, sys, hashlib
from pathlib import Path

def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None

def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def environment_manifest(extra: dict | None = None) -> dict:
    manifest={
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_sha": _run(["git","rev-parse","HEAD"]),
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": _run(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv,noheader"]),
    }
    try:
        import torch
        manifest.update({"torch":torch.__version__,"cuda_runtime":torch.version.cuda,"cuda_available":torch.cuda.is_available()})
    except Exception:
        manifest["torch"]=None
    if extra: manifest.update(extra)
    return manifest

def write_manifest(path: str | Path, extra: dict | None = None) -> None:
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(environment_manifest(extra),indent=2,sort_keys=True),encoding="utf-8")
