"""CTC decoding (numpy only): greedy and prefix beam search. blank id defaults to 0."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def collapse(ids, blank: int = 0) -> list[int]:
    """Merge repeats, then drop blanks (standard CTC collapse)."""
    out: list[int] = []
    prev = None
    for i in ids:
        i = int(i)
        if i != prev and i != blank:
            out.append(i)
        prev = i
    return out


def greedy_decode(log_probs: np.ndarray, blank: int = 0) -> list[int]:
    """log_probs: (T, C)."""
    return collapse(np.argmax(log_probs, axis=-1), blank)


def beam_search_decode(log_probs: np.ndarray, beam_width: int = 10, blank: int = 0, prune: float = 1e-3) -> list[int]:
    """CTC prefix beam search, no language model. log_probs: (T, C)."""
    probs = np.exp(np.asarray(log_probs, dtype=np.float64))
    T, _ = probs.shape
    beams: dict[tuple, tuple[float, float]] = {(): (1.0, 0.0)}  # prefix -> (p_blank, p_nonblank)
    for t in range(T):
        p_t = probs[t]
        cand = [int(c) for c in np.nonzero(p_t >= prune)[0] if c != blank]
        nxt: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for prefix, (pb, pnb) in beams.items():
            total = pb + pnb
            nxt[prefix][0] += total * p_t[blank]
            last = prefix[-1] if prefix else -1
            if prefix:
                nxt[prefix][1] += pnb * p_t[last]  # same symbol continues (no blank in between)
            for c in cand:
                if c == last:
                    nxt[prefix + (c,)][1] += pb * p_t[c]  # repeat only valid after a blank
                else:
                    nxt[prefix + (c,)][1] += total * p_t[c]
        ranked = sorted(nxt.items(), key=lambda kv: kv[1][0] + kv[1][1], reverse=True)[:beam_width]
        z = sum(v[0] + v[1] for _, v in ranked) or 1.0  # rescale: avoids underflow on long inputs
        beams = {k: (v[0] / z, v[1] / z) for k, v in ranked}
    best = max(beams.items(), key=lambda kv: kv[1][0] + kv[1][1])
    return list(best[0])


def decode(log_probs: np.ndarray, method: str = "greedy", beam_width: int = 10, blank: int = 0) -> list[int]:
    if method == "greedy":
        return greedy_decode(log_probs, blank)
    if method == "beam":
        return beam_search_decode(log_probs, beam_width, blank)
    raise ValueError(f"Unknown decode method {method!r}")
