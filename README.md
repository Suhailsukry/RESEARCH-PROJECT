# SignBridge Brain (Component B)

SignBridge Brain is the model and feature-engineering component of the SignBridge
research project (SLIIT IT4010, KND_CDAP_DS_06). It processes MediaPipe Holistic
landmark sequences and predicts continuous Sri Lankan Sign Language (SLSL) glosses,
which are mapped to Sinhala text.

The primary model is a stacked bidirectional LSTM trained with Connectionist
Temporal Classification (CTC). An isolated-sign sliding-window classifier is
included as a fallback. This repository covers Component B only; dataset
construction (Component A) and the real-time application (Component C) are
maintained separately.

> **Prototype status:** The included data generator creates synthetic mock
> landmarks, placeholder glosses, and placeholder Sinhala translations. These
> are useful for exercising the pipeline, but they are not real SLSL data and
> mock results are not research results.

## Requirements

- Python 3.10 or 3.11
- PyTorch 2.2 or newer
- MediaPipe and OpenCV are optional and only needed to extract landmarks from
  real videos. MediaPipe is pinned to `0.10.14`.

The project is configured for Python `<3.12` in `pyproject.toml`.

## Installation

Create and activate a virtual environment, then install the project and its
development dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On macOS or Linux, use `python3.11 -m venv .venv` and
`source .venv/bin/activate` instead.

To install the optional real-video extraction dependencies:

```powershell
python -m pip install -e ".[extract]"
```

## Quick start: mock data

Generate a mock dataset, run the tests, train the CTC model, and evaluate on the
validation split:

```powershell
python scripts/make_mock_data.py --out data/mock
pytest -q
python -m signbridge_brain.train --config configs/default.yaml --seed 1
python -m signbridge_brain.evaluate --checkpoint runs/bilstm_ctc_seed1/best.pt --split val
```

The default mock generator creates a relatively large dataset. For a smaller
iteration, reduce the number of generated sentences per signer:

```powershell
python scripts/make_mock_data.py --out data/mock --sents-per-signer 10
```

Training reads only the training and validation splits. The evaluation command
can also evaluate `test` and `real`, but those splits require the explicit
`--final` option and should only be used for final reporting:

```powershell
python -m signbridge_brain.evaluate --checkpoint runs/bilstm_ctc_seed1/best.pt --split test --final
```

To train and evaluate the fallback classifier:

```powershell
python -m signbridge_brain.train_fallback --config configs/default.yaml --seed 1
```

Each run writes its configuration, run metadata, metrics, and checkpoints under
`runs/`. Use at least three seeds and aggregate results with confidence
intervals:

```powershell
python -m signbridge_brain.train --config configs/default.yaml --seed 5
python -m signbridge_brain.train --config configs/default.yaml --seed 9
python -m signbridge_brain.evaluate --checkpoint runs/bilstm_ctc_seed1/best.pt --split val
python -m signbridge_brain.evaluate --checkpoint runs/bilstm_ctc_seed5/best.pt --split val
python -m signbridge_brain.evaluate --checkpoint runs/bilstm_ctc_seed9/best.pt --split val
python scripts/aggregate_seeds.py "runs/bilstm_ctc_seed*/eval_val_greedy.json"
```

## Pipeline and data format

The pipeline is:

1. Extract or load frame-level MediaPipe Holistic landmarks.
2. Normalize landmark sequences and select the configured pose, hand, and face
   features.
3. Train a BiLSTM-CTC model to predict gloss sequences without frame-level
   alignment.
4. Decode glosses and map them to Sinhala text using the configured vocabulary.

Stored sequences use NumPy `.npy` arrays with `float32` values and shape
`(T, 1629)`, where `T` is the number of frames. The 1629 values per frame are
the `(x, y, z)` coordinates for 543 landmarks, ordered as pose (33), left hand
(21), right hand (21), and face mesh (468). Undetected landmarks are zeros.
Dataset roots contain `manifest.json` for sentence sequences and
`isolated_manifest.json` for isolated-sign clips. See
[`FEATURE_SPEC.md`](FEATURE_SPEC.md) and
[`src/signbridge_brain/manifest.py`](src/signbridge_brain/manifest.py) for the
interfaces and manifest fields.

The default feature configuration uses pose, both hands, and a reduced face
subset (513 input dimensions). The face ablations are configured in
[`configs/ablate_face_none.yaml`](configs/ablate_face_none.yaml),
[`configs/ablate_face_reduced.yaml`](configs/ablate_face_reduced.yaml), and
[`configs/ablate_face_full.yaml`](configs/ablate_face_full.yaml). Any change to
the feature layout or normalization must also be recorded in
[`FEATURE_SPEC.md`](FEATURE_SPEC.md).

## Using real video

The optional extractor writes one landmark array per video:

```powershell
python -m signbridge_brain.features.extract --src videos --dst data/landmarks
```

It accepts a video file or a directory of videos. The output arrays use the
storage format above. Dataset manifests are a separate input: provide
`manifest.json` and `isolated_manifest.json` in the dataset root according to
the schemas in [`src/signbridge_brain/manifest.py`](src/signbridge_brain/manifest.py).
The mock vocabulary and translations in `configs/mock_vocab.json` must be
replaced with the appropriate project data before training or interpreting
results as SLSL recognition.

## Inference interface

Component C can load a self-contained checkpoint and pass raw landmarks to the
predictor. Normalization and feature selection are applied inside `predict()`:

```python
from signbridge_brain.infer import load_predictor

predictor = load_predictor("runs/bilstm_ctc_seed1/best.pt")
prediction = predictor.predict(raw_window)  # (T, 1629) or (T, 543, 3)

print(prediction.glosses)
print(prediction.sinhala)
print(prediction.confidence)
```

`StreamingBrain` provides a frame-by-frame wrapper. It re-decodes a rolling
window and may produce flickering predictions; temporal stabilization belongs
to Component C.

## Evaluation and research safeguards

- Splits must be signer-independent; training checks this before fitting.
- `test` and `real` evaluation are final-only (`--final`) to prevent tuning on
  held-out data.
- Synthetic and real-source results are reported separately.
- Record the seed, configuration, and commit for each run. Report results from
  at least three seeds with confidence intervals; do not present a single-seed
  result without a caveat.
- The project targets are WER below 30% on synthetic data and below 50% on real
  continuous data, with sentence accuracy above 60%. These are targets, not
  claims of achieved performance.

## Repository layout

```text
configs/                 Training defaults, mock vocabulary, and face ablations
scripts/                 Mock-data generation, seed aggregation, and demos
src/signbridge_brain/
  data/                  Mock dataset generator
  decoding/              CTC decoding
  features/              Landmark extraction, normalization, and augmentation
  models/                BiLSTM-CTC and sliding-window fallback
  train.py               CTC training
  train_fallback.py      Fallback classifier training
  evaluate.py            Split-aware evaluation
  infer.py               Component C prediction interface
tests/                   Unit tests
FEATURE_SPEC.md          Landmark and feature interface specification
```

Generated datasets, training runs, checkpoints, and NumPy arrays are excluded
from version control by `.gitignore`.
