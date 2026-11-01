# AGENTS.md: SignBridge Component B ("the brain")

You are helping build **Component B** of SignBridge (SLIIT IT4010, KND_CDAP_DS_06): MediaPipe feature
engineering + a stacked **BiLSTM + CTC** model for continuous Sri Lankan Sign Language (SLSL) -> gloss
sequences -> Sinhala text. This repo is ONLY Component B. Component A (data/sentence construction) and
Component C (real-time inference, robustness) live elsewhere; talk to them only through the interfaces below.

## Interfaces (do not change shapes/formats without updating FEATURE_SPEC.md and telling the team)
- **Stored sequence:** `.npy` float32 `(T, 1629)` = `(T, 543*3)`, raw MediaPipe Holistic coords.
  Order: pose 33 | left hand 21 | right hand 21 | face 468. Missing landmark = exact zeros.
- **Dataset root:** `manifest.json` (sentences) + `isolated_manifest.json` (single-sign clips). See `manifest.py`.
- **Inference contract for Component C:** `infer.load_predictor(ckpt).predict(raw_window) -> Prediction`
  (glosses, sinhala, confidence). Normalisation/feature selection happen inside `predict()`.
- **CTC:** class 0 = blank. Vocabulary in `vocab.py`; checkpoints are self-contained (`checkpoint.py`).

## Hard rules (from the project plan)
1. Python 3.10/3.11. One DL framework: **PyTorch** (confirm with team in Week 2). Pin MediaPipe exactly.
2. **Test set discipline:** never read `test` or `real` splits while tuning. `evaluate.py` enforces `--final`.
3. **Signer-independent splits** only; `assert_signer_independent` must pass.
4. **Reproducibility:** every run logs seed + config + git commit (`runs/<name>_seed<k>/`). Report >= 3 seeds with CIs
   (`scripts/aggregate_seeds.py`). Never report a single-seed number without a caveat.
5. **Report synthetic and real results separately** (already done in `evaluate.py`).
6. **CTC blank collapse (risk R1):** monitored every 5 epochs in `train.py`; checkpoint every epoch; keep the
   sliding-window fallback (`models/fallback.py`, `train_fallback.py`) working in parallel.
7. **Face ablation (risk R4):** `features.face: none|reduced|full` in config. Any feature change -> log in FEATURE_SPEC.md.
8. **Scope is frozen after Phase 2.** Ideas that aren't in the plan go in `docs/FUTURE_WORK.md`, not in code.
9. Targets: WER < 30% (synthetic), < 50% (real continuous); sentence acc > 60%.

## Current state
- Everything runs on **MOCK data** (`scripts/make_mock_data.py`). Mock glosses/Sinhala are placeholders.
- numpy-side tests pass. The PyTorch parts (model, train, evaluate, infer) were written but NOT executed in the
  authoring environment (no torch there). First job: run `pytest`, fix whatever breaks, then train.

## Commands
```
pip install -e ".[dev]"
python scripts/make_mock_data.py --out data/mock
pytest -q
python -m signbridge_brain.train --config configs/default.yaml --seed 1
python -m signbridge_brain.evaluate --checkpoint runs/bilstm_ctc_seed1/best.pt --split val
python -m signbridge_brain.train_fallback --config configs/default.yaml --seed 1
```

## Style
Small modules, type hints, no notebooks in `src/`. Add or update a test for every behaviour change.
Don't add dependencies without asking. Keep numpy-only modules free of `import torch`.
