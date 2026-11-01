"""Evaluate a checkpoint on a split, reporting synthetic and real results separately.

    python -m signbridge_brain.evaluate --checkpoint runs/bilstm_ctc_seed1/best.pt --split val
    python -m signbridge_brain.evaluate --checkpoint ... --split test --final   # report time only

test and real splits REQUIRE --final so they are never touched while tuning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .infer import load_predictor
from .manifest import FINAL_ONLY_SPLITS, load_manifest, select
from .metrics import corpus_scores
from .utils import pick_device

TARGETS = {"synthetic": 0.30, "real": 0.50}  # WER targets from the plan (README 5.4)


def evaluate(predictor, root: Path, items) -> dict:
    refs, hyps, by_source = [], [], {}
    for it in items:
        pred = predictor.predict(np.load(root / it.path))
        refs.append(list(it.glosses))
        hyps.append(list(pred.glosses))
        by_source.setdefault(it.source, ([], []))
        by_source[it.source][0].append(refs[-1])
        by_source[it.source][1].append(hyps[-1])
    out = {"overall": corpus_scores(refs, hyps), "by_source": {}}
    for src, (r, h) in by_source.items():
        s = corpus_scores(r, h)
        s["wer_target"] = TARGETS.get(src)
        s["meets_target"] = bool(TARGETS.get(src) is not None and s["wer"] < TARGETS[src])
        out["by_source"][src] = s
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default="data/mock")
    ap.add_argument("--split", default="val", choices=["train", "val", "test", "real"])
    ap.add_argument("--final", action="store_true", help="required for test/real splits")
    ap.add_argument("--decode", default="greedy", choices=["greedy", "beam"])
    ap.add_argument("--beam-width", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="write JSON results here")
    args = ap.parse_args(argv)

    if args.split in FINAL_ONLY_SPLITS and not args.final:
        raise SystemExit(f"Split {args.split!r} is final-reporting only (never tune on it). Re-run with --final.")

    root = Path(args.data_root)
    items = select(load_manifest(root), split=args.split)
    if not items:
        raise SystemExit(f"No items in split {args.split!r}")
    predictor = load_predictor(args.checkpoint, pick_device(args.device), args.decode, args.beam_width)
    result = evaluate(predictor, root, items)
    result.update(checkpoint=str(args.checkpoint), split=args.split, decode=args.decode)

    for src, s in result["by_source"].items():
        print(f"[{src:9s}] n={s['n_sentences']:3d}  WER={s['wer']*100:5.1f}%  "
              f"(S={s['substitutions']} D={s['deletions']} I={s['insertions']})  "
              f"sent_acc={s['sentence_acc']*100:5.1f}%  target<{(s['wer_target'] or 0)*100:.0f}%")
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / f"eval_{args.split}_{args.decode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
