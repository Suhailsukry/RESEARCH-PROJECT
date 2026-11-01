# Starter prompts for opencode (run in order; one task per session, commit after each)

Tip: opencode reads `AGENTS.md` automatically. Each prompt below is scoped so the agent can verify its own work.

## 1. Bring-up (first thing; the torch code has never been executed)
> Install deps (`pip install -e ".[dev]"`), run `python scripts/make_mock_data.py --out data/mock`, then `pytest -q`.
> Fix any failures in `src/` (not by weakening tests). Then run a 3-epoch smoke train
> (`python -m signbridge_brain.train --epochs 3`) and confirm checkpoints + metrics.jsonl are written and
> `evaluate --split val` runs. Report exactly what you changed and why.

## 2. First real training run + CTC health
> Train 60 epochs on mock data with seeds 1, 2, 3. Plot train/val loss, val WER and blank_ratio per epoch to
> `results/curves_seed*.png` (add `scripts/plot_curves.py`). If blank collapse persists past epoch 15, diagnose
> (LR, sequence length, input scaling) and propose a fix before changing defaults.

## 3. Face-landmark ablation (risk R4)
> Add `scripts/run_ablation.py` that trains `features.face = none | reduced | full` x 3 seeds on mock data,
> evaluates on VAL only, and prints a table of mean WER +/- 95% CI via `metrics.mean_ci`. Do not touch test/real.

## 4. Fallback comparison (Appendix B)
> Train the fallback with `train_fallback`, evaluate it with `evaluate` on val, and write a short comparison
> table vs BiLSTM-CTC (WER, sentence acc, deletions/insertions). Tune window/stride/threshold on VAL only.

## 5. Real data adapter (when Component A delivers)
> Write `src/signbridge_brain/data/adapters.py` that converts Component A's dataset layout into our
> `manifest.json` format with signer-independent splits. Do not change the (T, 1629) storage contract.
> Add tests with a tiny fixture. Ask me for the real layout before guessing.

## 6. Hand-off to Component C
> Profile `CTCPredictor.predict` on CPU for a 90-frame window; report latency. Then try dynamic quantisation
> (`torch.quantization.quantize_dynamic` on LSTM/Linear) and report the WER/latency trade-off on VAL.

## 7. Final evaluation (only at results freeze, checkpoint S8)
> Run `evaluate --split test --final` and `--split real --final` for the 3 chosen-seed checkpoints, then
> `scripts/aggregate_seeds.py`. Save outputs under `results/final/`. No further tuning after this.
