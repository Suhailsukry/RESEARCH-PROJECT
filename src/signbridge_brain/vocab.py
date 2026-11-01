"""Gloss vocabulary (CTC blank = id 0) and gloss -> Sinhala rendering."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

BLANK = "<blank>"


class Vocabulary:
    """id 0 is always the CTC blank. Real glosses start at 1."""

    def __init__(self, glosses: list[str]):
        if len(set(glosses)) != len(glosses):
            raise ValueError("Duplicate glosses in vocabulary")
        if BLANK in glosses:
            raise ValueError(f"{BLANK!r} is reserved")
        self.itos: list[str] = [BLANK] + list(glosses)
        self.stoi: dict[str, int] = {g: i for i, g in enumerate(self.itos)}

    @property
    def blank_id(self) -> int:
        return 0

    @property
    def size(self) -> int:
        return len(self.itos)

    @property
    def glosses(self) -> list[str]:
        return self.itos[1:]

    def encode(self, glosses: list[str]) -> list[int]:
        return [self.stoi[g] for g in glosses]

    def decode(self, ids) -> list[str]:
        return [self.itos[int(i)] for i in ids if int(i) != 0]

    def to_list(self) -> list[str]:
        return list(self.itos)

    @classmethod
    def from_list(cls, itos: list[str]) -> "Vocabulary":
        if not itos or itos[0] != BLANK:
            raise ValueError("Serialized vocabulary must start with the blank token")
        return cls(itos[1:])


class SinhalaMapper:
    """Gloss sequence -> Sinhala text.

    MVP: word-by-word lookup joined with spaces, plus optional whole-sentence overrides
    (exact gloss-sequence match). Real SLSL gloss order != Sinhala grammar, so the proper
    gloss->sentence grammar belongs with Component C / SLAD guidance. Strings in the mock
    vocabulary are placeholders.
    """

    def __init__(self, word_map: dict[str, str], overrides: dict[str, str] | None = None):
        self.word_map = dict(word_map)
        self.overrides = dict(overrides or {})

    def render(self, glosses: list[str]) -> str:
        key = " ".join(glosses)
        if key in self.overrides:
            return self.overrides[key]
        return " ".join(self.word_map.get(g, g) for g in glosses)

    def to_dict(self) -> dict:
        return {"word_map": self.word_map, "overrides": self.overrides}

    @classmethod
    def from_dict(cls, d: dict) -> "SinhalaMapper":
        return cls(d.get("word_map", {}), d.get("overrides", {}))


@dataclass
class VocabBundle:
    vocab: Vocabulary
    sinhala: SinhalaMapper
    sentences: list[list[str]]


def load_vocab_bundle(path: str | Path) -> VocabBundle:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    glosses = [g["gloss"] for g in raw["glosses"]]
    word_map = {g["gloss"]: g["sinhala"] for g in raw["glosses"]}
    sentences = raw.get("sentences", [])
    for s in sentences:
        for g in s:
            if g not in glosses:
                raise ValueError(f"Sentence uses unknown gloss {g!r}")
    return VocabBundle(
        vocab=Vocabulary(glosses),
        sinhala=SinhalaMapper(word_map, raw.get("sentence_overrides", {})),
        sentences=sentences,
    )
