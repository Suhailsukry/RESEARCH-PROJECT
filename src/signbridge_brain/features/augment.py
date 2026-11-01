"""Training-time augmentation on raw (T, 543, 3) landmarks. numpy only."""
from __future__ import annotations

import numpy as np

from ..config import AugmentConfig
from .landmarks import LEFT_HAND, RIGHT_HAND


def time_stretch(lm: np.ndarray, factor: float) -> np.ndarray:
    T = lm.shape[0]
    new_T = max(2, int(round(T * factor)))
    pos = np.linspace(0, T - 1, new_T)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    f = (pos - i0).astype(np.float32)[:, None, None]
    out = lm[i0] * (1 - f) + lm[i1] * f
    missing = ~np.any(lm != 0, axis=-1)
    out[missing[i0] | missing[i1]] = 0.0  # never interpolate across a missing landmark
    return out.astype(np.float32)


def jitter(lm: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    present = np.any(lm != 0, axis=-1, keepdims=True)
    return (lm + rng.normal(0.0, std, lm.shape).astype(np.float32) * present).astype(np.float32)


def landmark_dropout(lm: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Randomly zero a whole hand in a frame (simulates detection failures)."""
    out = lm.copy()
    T = lm.shape[0]
    for hand in (LEFT_HAND, RIGHT_HAND):
        drop = rng.random(T) < p
        out[drop, hand] = 0.0
    return out


def augment(lm: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator) -> np.ndarray:
    if not cfg.enabled:
        return lm
    lo, hi = cfg.time_stretch
    lm = time_stretch(lm, float(rng.uniform(lo, hi)))
    if cfg.noise_std > 0:
        lm = jitter(lm, cfg.noise_std, rng)
    if cfg.landmark_dropout_p > 0:
        lm = landmark_dropout(lm, cfg.landmark_dropout_p, rng)
    return lm
