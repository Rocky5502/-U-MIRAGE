#!/usr/bin/env python3
"""Fail if tracked repository files contain common credential patterns."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

PATTERNS = {
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "Anthropic-style key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    "Private key block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}

BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".pt", ".pth", ".bin",
    ".safetensors",
}


def tracked_files() -> list[Path]:
    raw = subprocess.check_output(["git", "ls-files", "-z"])
    return [Path(x.decode("utf-8")) for x in raw.split(b"\0") if x]


def main() -> int:
    findings: list[tuple[str, str]] = []

    for path in tracked_files():
        if path.suffix.lower() in BINARY_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append((str(path), label))

    if findings:
        print("SECRET_SCAN=FAIL")
        for path, label in findings:
            print(f"{path}: {label}")
        print("Rotate exposed credentials before rewriting/removing history.")
        return 1

    print("SECRET_SCAN=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
