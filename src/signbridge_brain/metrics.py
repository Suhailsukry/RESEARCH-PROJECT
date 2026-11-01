"""WER (with substitution/deletion/insertion breakdown), sentence accuracy, seed aggregation."""
from __future__ import annotations

import math

import numpy as np


def edit_counts(ref: list, hyp: list) -> tuple[int, int, int]:
    """Return (substitutions, deletions, insertions) from a minimal alignment."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int64)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    i, j, s, dele, ins = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            s += int(ref[i - 1] != hyp[j - 1])
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, dele, ins


def corpus_scores(refs: list[list], hyps: list[list]) -> dict:
    """WER is a fraction (0.25 = 25%). High deletions + empty hyps = CTC blank collapse."""
    assert len(refs) == len(hyps)
    S = D = I = N = exact = 0
    for r, h in zip(refs, hyps):
        s, dl, ins = edit_counts(list(r), list(h))
        S, D, I, N = S + s, D + dl, I + ins, N + len(r)
        exact += int(list(r) == list(h))
    n = len(refs)
    return {
        "n_sentences": n,
        "n_ref_words": N,
        "wer": (S + D + I) / N if N else float("nan"),
        "substitutions": S,
        "deletions": D,
        "insertions": I,
        "sentence_acc": exact / n if n else float("nan"),
        "empty_hyp_ratio": sum(1 for h in hyps if len(h) == 0) / n if n else float("nan"),
    }


def mean_ci(values, confidence: float = 0.95) -> tuple[float, float]:
    """Mean and half-width of the t-based confidence interval across seeds."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return float("nan"), float("nan")
    if len(v) == 1:
        return float(v[0]), float("nan")
    from scipy import stats

    sem = v.std(ddof=1) / math.sqrt(len(v))
    return float(v.mean()), float(sem * stats.t.ppf((1 + confidence) / 2, len(v) - 1))
