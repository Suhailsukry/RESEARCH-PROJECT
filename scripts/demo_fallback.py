"""Demo: Appendix-B fallback (sliding window + duplicate suppression) on mock data.
Shows the safety net for CTC blank-collapse (R1).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from signbridge_brain.features.landmarks import preprocess
from signbridge_brain.manifest import load_manifest, select
from signbridge_brain.models.fallback import WindowClassifier, sliding_window_decode
from signbridge_brain.vocab import load_vocab_bundle


def main():
    cfg_path = "configs/default.yaml"
    import yaml
    from signbridge_brain.config import load_config

    cfg = load_config(cfg_path)
    bundle = load_vocab_bundle(cfg.data.vocab_file)
    vocab = bundle.vocab

    root = Path(cfg.data.root)
    items = load_manifest(root)
    # Use an isolated clip to demo window classification
    clips = select(items, split="train", source="synthetic")  # fallback uses synthetic in mock
    # Try to find anything short
    if not clips:
        clips = items
    item = clips[0]
    raw = np.load(root / item.path)
    x = preprocess(raw, cfg.features)  # (T, D)
    print(f"Example: {item.path}, glosses={item.glosses}, shape={x.shape}")

    # Load fallback checkpoint
    ckpt = Path("runs") / "fallback_seed1" / "best.pt"
    if not ckpt.exists():
        raise SystemExit("Train fallback first: py -m signbridge_brain.train_fallback --config configs/default.yaml --seed 1")
    ck = torch.load(ckpt, map_location="cpu")
    num_classes = len(vocab.itos)
    model = WindowClassifier(input_dim=x.shape[1], num_classes=num_classes)
    model.load_state_dict(ck.get("model_state") or ck["model"])
    model.eval()
    print(f"Loaded fallback checkpoint: {ckpt} (val clip acc from training ~0.984)")

    # Demo sliding window decode on a longer mock sequence (simulating continuous)
    sents = select(items, split="val", source="synthetic")
    sitem = sents[0]
    sraw = np.load(root / sitem.path)
    sx = preprocess(sraw, cfg.features)
    print(f"\nContinuous sequence: {sitem.path} shape={sx.shape}, true glosses={sitem.glosses}")
    ids, conf = sliding_window_decode(
        model,
        sx,
        window=cfg.fallback.window,
        stride=cfg.fallback.stride,
        threshold=cfg.fallback.threshold,
    )
    glosses = vocab.decode(ids)
    print(f"Sliding window decode (window={cfg.fallback.window}, stride={cfg.fallback.stride}, thresh={cfg.fallback.threshold}):")
    print(f"  predicted gloss IDs: {ids}")
    print(f"  predicted glosses:   {glosses}")
    print(f"  avg confidence:      {conf:.4f}")
    print("\nConsecutive duplicates merged by sliding_window_decode (Appendix B).")


if __name__ == "__main__":
    main()
