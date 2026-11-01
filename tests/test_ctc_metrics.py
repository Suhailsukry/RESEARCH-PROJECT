import numpy as np

from signbridge_brain.decoding.ctc import beam_search_decode, collapse, greedy_decode
from signbridge_brain.metrics import corpus_scores, edit_counts, mean_ci


def _peaky(path, C=5, noise=0.01):
    peak = 1.0 - (C - 1) * noise
    lp = np.full((len(path), C), np.log(noise))
    for t, c in enumerate(path):
        lp[t, c] = np.log(peak)
    return lp


def test_collapse_repeats_and_blanks():
    assert collapse([0, 1, 1, 0, 2, 2, 0]) == [1, 2]
    assert collapse([1, 0, 1]) == [1, 1]  # blank separates a genuine repeat
    assert collapse([0, 0, 0]) == []


def test_greedy_and_beam_agree_on_clear_signal():
    lp = _peaky([0, 1, 1, 0, 0, 3, 3, 3, 0, 2, 0, 2])
    assert greedy_decode(lp) == [1, 3, 2, 2]
    assert beam_search_decode(lp, beam_width=5) == [1, 3, 2, 2]


def test_beam_long_sequence_no_underflow():
    # 600 frames; clean input so the best total-probability hypothesis equals the best path.
    # (With ~1% noise per frame, beam search can legitimately prefer an extra split token.)
    path = ([0] * 5 + [1] * 5) * 60
    out = beam_search_decode(_peaky(path, noise=0.0005), beam_width=4)
    assert out == [1] * 60


def test_edit_counts():
    assert edit_counts(["a", "b", "c"], ["a", "c"]) == (0, 1, 0)
    assert edit_counts(["a", "b"], ["a", "x"]) == (1, 0, 0)
    assert edit_counts(["a"], ["a", "b"]) == (0, 0, 1)


def test_corpus_scores():
    s = corpus_scores([["a", "b", "c"], ["d"]], [["a", "c"], ["d"]])
    assert abs(s["wer"] - 0.25) < 1e-9 and s["deletions"] == 1
    assert s["sentence_acc"] == 0.5
    assert corpus_scores([["a"]], [[]])["empty_hyp_ratio"] == 1.0


def test_mean_ci():
    m, h = mean_ci([10.0, 12.0, 14.0])
    assert abs(m - 12.0) < 1e-9 and h > 0
    assert np.isnan(mean_ci([5.0])[1])
