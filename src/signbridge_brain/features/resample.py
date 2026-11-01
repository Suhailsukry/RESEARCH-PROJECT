"""numpy-only clip resampling (no torch import)."""
from __future__ import annotations

import numpy as np


def resample_clip(x: np.ndarray, n: int) -> np.ndarray:
    """Linear-interpolate a (T, D) clip to exactly n frames."""
    T = x.shape[0]
    if T == n:
        return x
    pos = np.linspace(0, T - 1, n)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    f = (pos - i0).astype(np.float32)[:, None]
    return (x[i0] * (1 - f) + x[i1] * f).astype(np.float32)
