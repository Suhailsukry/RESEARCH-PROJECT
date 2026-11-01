import numpy as np

from signbridge_brain.config import FeatureConfig
from signbridge_brain.features.landmarks import (
    FEATURE_DIM, N_LANDMARKS, REDUCED_FACE_IDX, feature_dim, normalise_sequence, preprocess, selected_indices,
)


def test_constants():
    assert N_LANDMARKS == 543 and FEATURE_DIM == 1629


def test_reduced_face_indices_valid():
    assert len(REDUCED_FACE_IDX) == len(set(REDUCED_FACE_IDX))
    assert min(REDUCED_FACE_IDX) >= 0 and max(REDUCED_FACE_IDX) < 468


def test_feature_dims():
    assert feature_dim(FeatureConfig(face="full")) == 1629
    assert feature_dim(FeatureConfig(face="none")) == 75 * 3
    assert feature_dim(FeatureConfig(face="reduced")) == (75 + len(REDUCED_FACE_IDX)) * 3
    assert feature_dim(FeatureConfig(face="none", add_velocity=True)) == 75 * 3 * 2
    assert len(selected_indices(FeatureConfig(face="none", use_hands=False))) == 33


def _random_seq(T=20, seed=0):
    rng = np.random.default_rng(seed)
    lm = rng.normal(0.5, 0.1, (T, 543, 3)).astype(np.float32)
    lm[:, 11] = [0.6, 0.45, 0.0]
    lm[:, 12] = [0.4, 0.45, 0.0]
    return lm


def test_normalisation_invariant_to_scale_and_offset():
    lm = _random_seq()
    moved = lm * 1.7 + np.array([0.1, -0.2, 0.05], dtype=np.float32)
    assert np.allclose(normalise_sequence(lm), normalise_sequence(moved), atol=1e-4)


def test_missing_landmarks_stay_zero():
    lm = _random_seq()
    lm[5:9, 33:54] = 0.0  # left hand missing
    out = normalise_sequence(lm)
    assert np.all(out[5:9, 33:54] == 0.0)
    assert np.all(out[0, 33:54] != 0.0)


def test_missing_shoulders_reuse_last_valid():
    lm = _random_seq()
    lm[10:12, 11] = 0.0
    out = normalise_sequence(lm)
    assert np.isfinite(out).all()


def test_preprocess_shape_and_dtype():
    lm = _random_seq(T=17)
    for cfg in [FeatureConfig(face="full"), FeatureConfig(face="none"), FeatureConfig(add_velocity=True)]:
        x = preprocess(lm.reshape(17, -1), cfg)
        assert x.shape == (17, feature_dim(cfg)) and x.dtype == np.float32
