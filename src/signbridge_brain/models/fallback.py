"""Appendix B fallback: isolated-window classifier + sliding-window decoder with duplicate suppression."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..features.resample import resample_clip  # noqa: F401  (re-exported)


class WindowClassifier(nn.Module):
    """(B, W, D) fixed-length window -> (B, C) logits. Class 0 = background / no sign."""

    def __init__(self, input_dim: int, num_classes: int, hidden_size: int = 128, num_layers: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.gru = nn.GRU(input_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(2 * hidden_size), nn.Dropout(dropout),
                                  nn.Linear(2 * hidden_size, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(self.norm(x))
        return self.head(out.mean(dim=1))


@torch.no_grad()
def sliding_window_decode(model: WindowClassifier, feats: np.ndarray, window: int = 30, stride: int = 5,
                          threshold: float = 0.75, device="cpu") -> tuple[list[int], float]:
    """feats: preprocessed (T, D). Returns (gloss ids, mean confidence of kept windows).

    Steps (Appendix B): slide window -> classify -> drop background / low confidence ->
    merge consecutive identical tokens. Limitation: a genuinely repeated sign
    (e.g. PAIN PAIN) is merged into one token.
    """
    model.eval()
    T = feats.shape[0]
    if T < window:
        feats = np.concatenate([feats, np.zeros((window - T, feats.shape[1]), dtype=np.float32)])
        T = window
    starts = list(range(0, T - window + 1, stride))
    batch = torch.from_numpy(np.stack([feats[s:s + window] for s in starts])).to(device)
    probs = torch.softmax(model(batch), dim=-1).cpu().numpy()
    cls, conf = probs.argmax(axis=1), probs.max(axis=1)
    ids: list[int] = []
    kept: list[float] = []
    for c, p in zip(cls, conf):
        if c == 0 or p < threshold:
            continue
        kept.append(float(p))
        if not ids or ids[-1] != int(c):
            ids.append(int(c))
    return ids, (float(np.mean(kept)) if kept else 0.0)
