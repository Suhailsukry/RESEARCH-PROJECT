"""Mean +/- 95% CI across seeds.  python scripts/aggregate_seeds.py "runs/bilstm_ctc_seed*/eval_val_greedy.json" """
import glob
import json
import sys

from signbridge_brain.metrics import mean_ci


def main():
    files = sorted(glob.glob(sys.argv[1]))
    if not files:
        raise SystemExit("No files matched")
    runs = [json.load(open(f, encoding="utf-8")) for f in files]
    print(f"{len(runs)} runs: {files}")
    for src in sorted({s for r in runs for s in r["by_source"]}):
        for key in ("wer", "sentence_acc"):
            vals = [r["by_source"][src][key] * 100 for r in runs if src in r["by_source"]]
            m, h = mean_ci(vals)
            print(f"[{src}] {key}: {m:.1f}% +/- {h:.1f}  (n_seeds={len(vals)})")
    if len(runs) < 3:
        print("WARNING: fewer than 3 seeds; CI is unreliable (quality gate: no single-seed results).")


if __name__ == "__main__":
    main()
