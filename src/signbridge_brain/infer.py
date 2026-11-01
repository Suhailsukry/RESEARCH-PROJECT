"""Inference interface for Component C (real-time). The contract:

    predictor = load_predictor("runs/bilstm_ctc_seed1/best.pt")
    pred = predictor.predict(window)   # window: raw landmarks (T, 1629) or (T, 543, 3)
    pred.glosses, pred.sinhala, pred.confidence

Normalisation + feature selection happen INSIDE predict(); callers pass raw MediaPipe output.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from .checkpoint import load_checkpoint, rebuild
from .decoding.ctc import decode
from .features.landmarks import N_COORDS, N_LANDMARKS, preprocess
from .models.fallback import sliding_window_decode


@dataclass
class Prediction:
    glosses: list[str]
    sinhala: str
    confidence: float  # mean max-prob over non-blank frames/windows; 0.0 if nothing detected
    blank_ratio: float = float("nan")  # CTC only; ~1.0 on many windows suggests collapse or idle signer
    ids: list[int] = field(default_factory=list)


class CTCPredictor:
    def __init__(self, ckpt: dict, device="cpu", method: str = "greedy", beam_width: int = 10):
        self.device = device
        self.model, self.feature_cfg, self.vocab, self.sinhala = rebuild(ckpt, device)
        self.method, self.beam_width = method, beam_width

    @torch.no_grad()
    def predict(self, raw) -> Prediction:
        x = torch.from_numpy(preprocess(raw, self.feature_cfg))[None].to(self.device)
        logp, _ = self.model(x, torch.tensor([x.shape[1]]))
        lp = logp[0].cpu().numpy()
        ids = decode(lp, self.method, self.beam_width, self.vocab.blank_id)
        top = lp.argmax(axis=-1)
        nonblank = top != self.vocab.blank_id
        conf = float(np.exp(lp.max(axis=-1))[nonblank].mean()) if nonblank.any() else 0.0
        glosses = self.vocab.decode(ids)
        return Prediction(glosses, self.sinhala.render(glosses), conf, float(1.0 - nonblank.mean()), ids)


class SlidingWindowPredictor:
    def __init__(self, ckpt: dict, device="cpu"):
        self.device = device
        self.model, self.feature_cfg, self.vocab, self.sinhala = rebuild(ckpt, device)
        self.params = ckpt["extra"]["decode"]

    def predict(self, raw) -> Prediction:
        feats = preprocess(raw, self.feature_cfg)
        ids, conf = sliding_window_decode(self.model, feats, self.params["window"], self.params["stride"],
                                          self.params["threshold"], self.device)
        glosses = self.vocab.decode(ids)
        return Prediction(glosses, self.sinhala.render(glosses), conf, ids=ids)


def load_predictor(path, device="cpu", method: str = "greedy", beam_width: int = 10):
    ckpt = load_checkpoint(path, device)
    if ckpt["model_type"] == "bilstm_ctc":
        return CTCPredictor(ckpt, device, method, beam_width)
    if ckpt["model_type"] == "fallback_window":
        return SlidingWindowPredictor(ckpt, device)
    raise ValueError(f"Unknown model_type {ckpt['model_type']!r}")


class StreamingBrain:
    """Frame-by-frame wrapper: push() one (543,3)/(1629,) frame; returns a Prediction every
    `stride_frames` once `min_frames` are buffered, else None.

    Re-decodes the whole rolling window each time, so output can flicker. Stabilising it
    (agreement across consecutive predictions, etc.) is Component C's job.
    """

    def __init__(self, predictor, window_frames: int = 90, stride_frames: int = 15, min_frames: int = 45):
        self.predictor = predictor
        self.buf: deque = deque(maxlen=window_frames)
        self.stride, self.min_frames, self._since = stride_frames, min_frames, 0

    def push(self, frame) -> Prediction | None:
        self.buf.append(np.asarray(frame, dtype=np.float32).reshape(N_LANDMARKS, N_COORDS))
        self._since += 1
        if len(self.buf) >= self.min_frames and self._since >= self.stride:
            self._since = 0
            return self.predictor.predict(np.stack(self.buf))
        return None

    def reset(self) -> None:
        self.buf.clear()
        self._since = 0
