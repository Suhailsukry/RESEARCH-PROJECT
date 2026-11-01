"""Demo: Feature ablation showcase (face none/reduced/full) for R4.
Runs short training per config to show feature_dim changes.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from signbridge_brain.features.landmarks import feature_dim
from signbridge_brain.config import load_config


def main():
    base = load_config("configs/default.yaml")
    for face in ["none", "reduced", "full"]:
        cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
        cfg["features"]["face"] = face
        out = Path(f"configs/ablate_face_{face}.yaml")
        out.write_text(yaml.safe_dump(cfg))
        c = load_config(out)
        print(f"face={face}: feature_dim={feature_dim(c.features)} (pose+hands + face {face})")

    print("\nTo run short ablations:")
    for face in ["none", "reduced", "full"]:
        print(f"  py -m signbridge_brain.train --config configs/ablate_face_{face}.yaml --seed 1 --epochs 3 --run-name ablate_face_{face}")


if __name__ == "__main__":
    main()
