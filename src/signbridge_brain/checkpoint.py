"""Self-contained checkpoints: weights + everything needed to rebuild the predictor.

Only plain Python types + tensors are stored so torch.load(weights_only=True) works.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .config import FeatureConfig
from .models import build_model
from .vocab import SinhalaMapper, Vocabulary


def save_checkpoint(path, *, model_type: str, model, model_kwargs: dict, feature_cfg: FeatureConfig,
                    vocab: Vocabulary, sinhala: SinhalaMapper, seed: int, extra: dict | None = None) -> None:
    from dataclasses import asdict

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_type": model_type,
        "model_kwargs": model_kwargs,
        "model_state": model.state_dict(),
        "feature_cfg": asdict(feature_cfg),
        "vocab": vocab.to_list(),
        "sinhala": sinhala.to_dict(),
        "seed": int(seed),
        "extra": extra or {},
    }, path)


def load_checkpoint(path, device="cpu") -> dict:
    return torch.load(path, map_location=device, weights_only=True)


def rebuild(ckpt: dict, device="cpu"):
    """Return (model in eval mode, FeatureConfig, Vocabulary, SinhalaMapper)."""
    model = build_model(ckpt["model_type"], ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return (model, FeatureConfig(**ckpt["feature_cfg"]), Vocabulary.from_list(ckpt["vocab"]),
            SinhalaMapper.from_dict(ckpt["sinhala"]))
