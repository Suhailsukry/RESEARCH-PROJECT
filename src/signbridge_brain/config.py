"""Typed config loaded from YAML. Unknown keys raise, so typos never pass silently."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    root: str = "data/mock"
    vocab_file: str = "configs/mock_vocab.json"
    train_split: str = "train"
    val_split: str = "val"


@dataclass
class FeatureConfig:
    use_pose: bool = True
    use_hands: bool = True
    face: str = "reduced"  # none | reduced | full
    add_velocity: bool = False

    def __post_init__(self) -> None:
        if self.face not in ("none", "reduced", "full"):
            raise ValueError(f"features.face must be none|reduced|full, got {self.face!r}")


@dataclass
class ModelConfig:
    type: str = "bilstm_ctc"
    proj_dim: int = 256
    hidden_size: int = 192
    num_layers: int = 2
    dropout: float = 0.3
    input_dropout: float = 0.1


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-2
    grad_clip: float = 5.0
    num_workers: int = 0
    device: str = "auto"
    runs_dir: str = "runs"
    blank_check_every: int = 5


@dataclass
class AugmentConfig:
    enabled: bool = True
    time_stretch: tuple = (0.85, 1.15)
    noise_std: float = 0.01
    landmark_dropout_p: float = 0.02

    def __post_init__(self) -> None:
        self.time_stretch = tuple(self.time_stretch)


@dataclass
class FallbackConfig:
    window: int = 30
    stride: int = 5
    threshold: float = 0.75
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.2


@dataclass
class DecodeConfig:
    method: str = "greedy"  # greedy | beam
    beam_width: int = 10


_SECTIONS = {
    "data": DataConfig,
    "features": FeatureConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "augment": AugmentConfig,
    "fallback": FallbackConfig,
    "decode": DecodeConfig,
}


@dataclass
class Config:
    seed: int = 1
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    fallback: FallbackConfig = field(default_factory=FallbackConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Config":
        d = dict(d or {})
        kwargs: dict[str, Any] = {}
        for key, value in d.items():
            if key == "seed":
                kwargs["seed"] = int(value)
            elif key in _SECTIONS:
                kwargs[key] = _build(_SECTIONS[key], value)
            else:
                raise ValueError(f"Unknown top-level config key: {key!r}")
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["augment"]["time_stretch"] = list(out["augment"]["time_stretch"])
        return out


def _build(section_cls, values: dict[str, Any] | None):
    values = dict(values or {})
    known = {f.name for f in dataclasses.fields(section_cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown keys for {section_cls.__name__}: {sorted(unknown)}")
    return section_cls(**values)


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        return Config.from_dict(yaml.safe_load(f))
