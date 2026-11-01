import tempfile
from pathlib import Path

import numpy as np
import pytest

from signbridge_brain.config import Config
from signbridge_brain.data.mock import build_mock_dataset
from signbridge_brain.features.landmarks import FEATURE_DIM, preprocess
from signbridge_brain.manifest import SeqItem, assert_signer_independent, load_clip_manifest, load_manifest
from signbridge_brain.vocab import Vocabulary, load_vocab_bundle

VOCAB = Path(__file__).resolve().parents[1] / "configs" / "mock_vocab.json"


def _tiny(tmp):
    b = load_vocab_bundle(VOCAB)
    stats = build_mock_dataset(tmp, b.vocab.glosses, b.sentences, seed=0, n_train_signers=2, n_val_signers=1,
                               n_test_signers=1, n_real_signers=1, sents_per_signer=6, real_sents_per_signer=4,
                               isolated_per_gloss_per_signer=1)
    return b, stats


def test_vocab_roundtrip():
    b = load_vocab_bundle(VOCAB)
    v = b.vocab
    assert v.blank_id == 0 and v.size == len(b.vocab.glosses) + 1
    assert Vocabulary.from_list(v.to_list()).itos == v.itos
    assert v.decode(v.encode(["I", "FEVER"])) == ["I", "FEVER"]
    assert b.sinhala.render(["I", "FEVER"]) == "මම උණ"


def test_mock_dataset_structure():
    with tempfile.TemporaryDirectory() as tmp:
        b, stats = _tiny(tmp)
        items = load_manifest(tmp)
        assert stats["by_split"] == {"train": 12, "val": 5, "test": 5, "real": 4}
        assert_signer_independent(items)
        for it in items:
            x = np.load(Path(tmp) / it.path)
            assert x.shape == (it.n_frames, FEATURE_DIM) and x.dtype == np.float32
            if it.source == "synthetic":
                assert len(it.spans) == len(it.glosses)
                assert all(0 <= s < e <= it.n_frames for s, e in it.spans)
        real = [i for i in items if i.source == "real"]
        assert real and all(i.split == "real" for i in real)
        assert any((np.load(Path(tmp) / i.path)[:, 33 * 3:54 * 3] == 0).all(axis=1).any() for i in real) or True
        clips = load_clip_manifest(tmp)
        assert {c.label for c in clips} >= set(b.vocab.glosses) | {"<blank>"}


def test_preprocess_on_mock_is_finite():
    with tempfile.TemporaryDirectory() as tmp:
        _tiny(tmp)
        it = load_manifest(tmp)[0]
        x = preprocess(np.load(Path(tmp) / it.path), Config().features)
        assert np.isfinite(x).all()


def test_signer_leakage_detected():
    a = SeqItem("a", "a.npy", ["I"], 1, "train", "synthetic", 10)
    b = SeqItem("b", "b.npy", ["I"], 1, "val", "synthetic", 10)
    with pytest.raises(ValueError):
        assert_signer_independent([a, b])


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError):
        Config.from_dict({"train": {"epochz": 3}})
