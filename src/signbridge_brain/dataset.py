"""Torch datasets + collate functions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .config import AugmentConfig, FeatureConfig
from .features.augment import augment
from .features.landmarks import preprocess, to_landmarks
from .manifest import ClipItem, SeqItem
from .features.resample import resample_clip
from .vocab import BLANK, Vocabulary


class SequenceDataset(Dataset):
    """Sentence-level sequences for CTC training/eval."""

    def __init__(self, root, items: list[SeqItem], vocab: Vocabulary, feature_cfg: FeatureConfig,
                 augment_cfg: AugmentConfig | None = None, train: bool = False):
        self.root, self.items, self.vocab = Path(root), items, vocab
        self.feature_cfg, self.augment_cfg, self.train = feature_cfg, augment_cfg, train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        it = self.items[i]
        raw = to_landmarks(np.load(self.root / it.path))
        if self.train and self.augment_cfg is not None and self.augment_cfg.enabled:
            # seed from torch's RNG: reproducible under set_seed, different every epoch
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
            raw = augment(raw, self.augment_cfg, rng)
        feats = preprocess(raw, self.feature_cfg)
        target = torch.tensor(self.vocab.encode(it.glosses), dtype=torch.long)
        return torch.from_numpy(feats), target


def collate_ctc(batch):
    xs, ys = zip(*batch)
    x_lens = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    y_lens = torch.tensor([len(y) for y in ys], dtype=torch.long)
    return pad_sequence(xs, batch_first=True), x_lens, pad_sequence(ys, batch_first=True, padding_value=0), y_lens


class ClipDataset(Dataset):
    """Isolated clips resampled to a fixed window, for the fallback classifier."""

    def __init__(self, root, items: list[ClipItem], vocab: Vocabulary, feature_cfg: FeatureConfig,
                 window: int = 30):
        self.root, self.items, self.vocab = Path(root), items, vocab
        self.feature_cfg, self.window = feature_cfg, window

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        it = self.items[i]
        feats = preprocess(np.load(self.root / it.path), self.feature_cfg)
        feats = resample_clip(feats, self.window)
        label = 0 if it.label == BLANK else self.vocab.stoi[it.label]
        return torch.from_numpy(feats), torch.tensor(label, dtype=torch.long)
