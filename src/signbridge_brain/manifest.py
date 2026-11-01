"""Dataset manifests + signer-independence guard (quality gate: no signer leakage)."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

SPLITS = ("train", "val", "test", "real")
FINAL_ONLY_SPLITS = ("test", "real")  # never used for tuning


@dataclass
class SeqItem:
    id: str
    path: str  # relative to dataset root; .npy of shape (T, 1629) float32
    glosses: list
    signer_id: int
    split: str  # train | val | test | real
    source: str  # synthetic | real
    n_frames: int
    spans: Optional[list] = None  # [[start, end], ...] per gloss; synthetic mock only


@dataclass
class ClipItem:
    id: str
    path: str  # .npy (T, 1629)
    label: str  # gloss, or "<blank>" for background windows
    signer_id: int
    split: str


def _save(root: Path, name: str, items) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with open(root / name, "w", encoding="utf-8") as f:
        json.dump([asdict(i) for i in items], f, ensure_ascii=False, indent=1)


def _load(root: Path, name: str, cls):
    with open(Path(root) / name, "r", encoding="utf-8") as f:
        return [cls(**d) for d in json.load(f)]


def save_manifest(root, items) -> None:
    _save(Path(root), "manifest.json", items)


def load_manifest(root) -> list[SeqItem]:
    return _load(root, "manifest.json", SeqItem)


def save_clip_manifest(root, items) -> None:
    _save(Path(root), "isolated_manifest.json", items)


def load_clip_manifest(root) -> list[ClipItem]:
    return _load(root, "isolated_manifest.json", ClipItem)


def select(items, split: Optional[str] = None, source: Optional[str] = None):
    out = list(items)
    if split is not None:
        out = [i for i in out if i.split == split]
    if source is not None:
        out = [i for i in out if getattr(i, "source", source) == source]
    return out


def assert_signer_independent(items) -> None:
    """Raise if any signer appears in more than one split."""
    by_split: dict[str, set] = {}
    for it in items:
        by_split.setdefault(it.split, set()).add(it.signer_id)
    splits = sorted(by_split)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            overlap = by_split[a] & by_split[b]
            if overlap:
                raise ValueError(f"Signer leakage between {a!r} and {b!r}: signers {sorted(overlap)}")
