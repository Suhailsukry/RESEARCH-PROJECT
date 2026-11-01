"""Demo: Component C inference contract using trained checkpoint on mock data.
Shows infer.load_predictor().predict() returning the expected Prediction object.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from signbridge_brain import infer
from signbridge_brain.manifest import load_manifest, select


def main():
    root = Path("data/mock")
    items = load_manifest(root)
    # Pick any sentence sequence from val synthetic
    val_items = select(items, split="val", source="synthetic")
    if not val_items:
        raise SystemExit("No val synthetic items found. Run make_mock_data.py first.")

    item = val_items[0]
    raw = np.load(root / item.path)
    print(f"Loading sequence: {item.path} (shape {raw.shape}), glosses={item.glosses}")

    # Find best checkpoint from a recent run if exists
    runs_dir = Path("runs")
    ckpt = None
    for cand in [
        runs_dir / "pp1_demo_seed1" / "best.pt",
        runs_dir / "bilstm_ctc_seed1" / "best.pt",
    ]:
        if cand.exists():
            ckpt = cand
            break
    if ckpt is None:
        # Try to find any bilstm_ctc run
        for p in runs_dir.glob("bilstm_ctc_seed*/best.pt"):
            ckpt = p
            break
    if ckpt is None:
        raise SystemExit("No trained checkpoint found. Run: py -m signbridge_brain.train --config configs/default.yaml --seed 1 --epochs 8 --run-name pp1_demo")

    print(f"Loading checkpoint: {ckpt}")
    pred = infer.load_predictor(str(ckpt)).predict(raw)
    print("\n=== PREDICTION (Component C contract) ===")
    print(f"glosses: {pred.glosses}")
    print(f"sinhala: {pred.sinhala}")
    print(f"confidence: {pred.confidence:.4f}")
    print("\nThis matches infer.Prediction(glosses, sinhala, confidence). Normalisation/feature selection happens inside predict().")


if __name__ == "__main__":
    main()
