"""Landmark layout, subsets, normalisation, and the raw -> model-input transform.

Storage format (shared interface with Components A and C):
    one sequence = .npy float32 of shape (T, 1629) = (T, 543*3), raw MediaPipe coords.
Landmark order inside the 543 (see FEATURE_SPEC.md):
    [0:33] pose | [33:54] left hand | [54:75] right hand | [75:543] face mesh
Missing landmarks (e.g. hand not detected) are stored as exact zeros.
"""
from __future__ import annotations

import numpy as np

from ..config import FeatureConfig

N_POSE, N_HAND, N_FACE = 33, 21, 468
N_LANDMARKS = N_POSE + 2 * N_HAND + N_FACE  # 543
N_COORDS = 3
FEATURE_DIM = N_LANDMARKS * N_COORDS  # 1629

POSE = slice(0, 33)
LEFT_HAND = slice(33, 54)
RIGHT_HAND = slice(54, 75)
FACE = slice(75, 543)

SHOULDER_A, SHOULDER_B = 11, 12  # pose indices used for centring / scaling

# Reduced face subset (indices into the 468 face mesh): lips, eyes, brows, nose.
# Standard MediaPipe FaceMesh contour indices; re-verify against the pinned version.
_LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409]
_LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415]
_LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
_RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
_LEFT_BROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
_RIGHT_BROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
_NOSE = [1, 4, 6, 168]
REDUCED_FACE_IDX = sorted(set(_LIPS_OUTER + _LIPS_INNER + _LEFT_EYE + _RIGHT_EYE + _LEFT_BROW + _RIGHT_BROW + _NOSE))


def selected_indices(cfg: FeatureConfig) -> np.ndarray:
    """Landmark indices (into the 543) kept by this feature config."""
    idx: list[int] = []
    if cfg.use_pose:
        idx += list(range(*POSE.indices(N_LANDMARKS)))
    if cfg.use_hands:
        idx += list(range(*LEFT_HAND.indices(N_LANDMARKS)))
        idx += list(range(*RIGHT_HAND.indices(N_LANDMARKS)))
    if cfg.face == "full":
        idx += list(range(*FACE.indices(N_LANDMARKS)))
    elif cfg.face == "reduced":
        idx += [FACE.start + i for i in REDUCED_FACE_IDX]
    if not idx:
        raise ValueError("Feature config selects zero landmarks")
    return np.array(sorted(idx), dtype=np.int64)


def feature_dim(cfg: FeatureConfig) -> int:
    d = len(selected_indices(cfg)) * N_COORDS
    return d * 2 if cfg.add_velocity else d


def to_landmarks(raw) -> np.ndarray:
    """Accept (T, 1629) or (T, 543, 3); return float32 (T, 543, 3)."""
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 2:
        if arr.shape[1] != FEATURE_DIM:
            raise ValueError(f"Expected (T, {FEATURE_DIM}), got {arr.shape}")
        arr = arr.reshape(arr.shape[0], N_LANDMARKS, N_COORDS)
    elif arr.ndim != 3 or arr.shape[1:] != (N_LANDMARKS, N_COORDS):
        raise ValueError(f"Expected (T, {N_LANDMARKS}, {N_COORDS}), got {arr.shape}")
    return arr


def normalise_sequence(lm: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Per-frame: centre on the mid-shoulder point, divide by shoulder width (xy).

    Makes features invariant to signer position/distance from the camera.
    If shoulders are missing in a frame, reuse the last valid centre/scale.
    Missing (all-zero) landmarks stay exactly zero.
    """
    T = lm.shape[0]
    missing = ~np.any(lm != 0, axis=-1)  # (T, 543)
    out = np.zeros_like(lm)

    def shoulders_ok(t: int) -> bool:
        if missing[t, SHOULDER_A] or missing[t, SHOULDER_B]:
            return False
        return float(np.linalg.norm(lm[t, SHOULDER_A, :2] - lm[t, SHOULDER_B, :2])) > eps

    center, scale = np.zeros(3, dtype=np.float32), 1.0
    first_valid = next((t for t in range(T) if shoulders_ok(t)), None)
    if first_valid is not None:  # back-fill frames before the first valid shoulders
        a, b = lm[first_valid, SHOULDER_A], lm[first_valid, SHOULDER_B]
        center, scale = (a + b) / 2, float(np.linalg.norm(a[:2] - b[:2]))

    for t in range(T):
        if shoulders_ok(t):
            a, b = lm[t, SHOULDER_A], lm[t, SHOULDER_B]
            center, scale = (a + b) / 2, float(np.linalg.norm(a[:2] - b[:2]))
        out[t] = (lm[t] - center) / scale
    out[missing] = 0.0
    return out.astype(np.float32)


def preprocess(raw, cfg: FeatureConfig) -> np.ndarray:
    """raw landmarks -> model input of shape (T, feature_dim(cfg)), float32."""
    lm = normalise_sequence(to_landmarks(raw))
    idx = selected_indices(cfg)
    x = lm[:, idx, :].reshape(lm.shape[0], -1)
    if cfg.add_velocity:
        vel = np.diff(x, axis=0, prepend=x[:1])
        x = np.concatenate([x, vel], axis=1)
    return np.ascontiguousarray(x, dtype=np.float32)
