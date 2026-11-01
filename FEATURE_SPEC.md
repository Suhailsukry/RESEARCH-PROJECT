# FEATURE_SPEC.md

Record every change to landmark subsets, normalisation, or storage here (plan: README "Notes for Coding Agents" #2).

## Versions (fill in after Week-1 verification on all three machines)
| Item | Value |
|---|---|
| Python | TODO (3.10 or 3.11) |
| MediaPipe | TODO (pin exact; `requirements`/`pyproject` suggest 0.10.14, verify) |
| OpenCV | TODO |
| PyTorch | TODO |

## Storage format (interface with Components A and C)
- One sequence per `.npy`, float32, shape `(T, 1629)` = `(T, 543 x 3)`, raw (x, y, z) from MediaPipe Holistic.
- Landmark order within the 543: `[0:33]` pose, `[33:54]` left hand, `[54:75]` right hand, `[75:543]` face mesh.
- Not detected => exact zeros. `refine_face_landmarks=False` (exactly 468 face points).
- MediaPipe's own left/right hand labels are used unmirrored. All three components must do the same.

## Normalisation (v1, `features/landmarks.py::normalise_sequence`)
- Per frame: subtract the mid-shoulder point (pose 11, 12), divide by shoulder width (xy distance).
- If shoulders are missing in a frame, reuse the last valid centre/scale (back-filled from the first valid frame).
- Missing landmarks stay exactly 0 after normalisation.
- Not done yet: hand interpolation over short dropouts, left/right mirroring augmentation, z-coordinate rescaling.

## Feature subsets (`FeatureConfig`)
| Option | Values | Landmarks | Dim (xyz) |
|---|---|---|---|
| pose + hands + face=none | | 33 + 42 = 75 | 225 |
| pose + hands + face=reduced (default) | lips, eyes, brows, nose = 96 face pts | 171 | 513 |
| pose + hands + face=full | all 468 | 543 | 1629 |
| add_velocity | frame-to-frame delta appended | x2 | x2 |

## Change log
| Date | Change | Reason | Author |
|---|---|---|---|
| v0.1 | Initial spec on mock data | scaffold | |
