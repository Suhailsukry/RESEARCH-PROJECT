from __future__ import annotations

import json
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str = "auto"):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run_info(seed: int, config_dict: dict) -> dict:
    """Everything needed to trace a reported number back to a run."""
    import torch

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        commit = ""
    return {"seed": seed, "git_commit": commit or None, "python": sys.version.split()[0],
            "torch": torch.__version__, "platform": platform.platform(), "config": config_dict}


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
