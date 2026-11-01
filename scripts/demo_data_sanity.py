"""Data sanity report for presentation - shows discipline without real data."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from signbridge_brain.manifest import assert_signer_independent, load_manifest
from signbridge_brain.features.landmarks import N_POSE, N_HAND, N_FACE, FEATURE_DIM


def main():
    root = Path("data/mock")
    items = load_manifest(root)
    print("=== DATA SANITY (mock) ===")
    print(f"Total items: {len(items)}")
    print(f"Counts by (source,split): {Counter((i.source,i.split) for i in items)}")
    print(f"Unique signers: {sorted(set(getattr(i,'signer_id', getattr(i,'signer','?')) for i in items))}")
    try:
        assert_signer_independent(items)
        print("Signer-independent split: OK (no leakage)")
    except Exception as e:
        print(f"Signer-independent split: FAILED - {e}")

    # Check a few sequences
    for i in items[:3]:
        raw = np.load(root / i.path)
        miss = (raw.reshape(raw.shape[0], -1) == 0).mean()
        print(f"\n{i.path}: shape={raw.shape} (expect (T,{FEATURE_DIM})), missing_frac={miss:.3f}")
        # breakdown
        r = raw.reshape(raw.shape[0], 543, 3)
        miss_pose = (~np.any(r[:, :N_POSE] != 0, axis=-1)).mean()
        miss_lh = (~np.any(r[:, N_POSE:N_POSE+N_HAND] != 0, axis=-1)).mean()
        miss_rh = (~np.any(r[:, N_POSE+N_HAND:N_POSE+2*N_HAND] != 0, axis=-1)).mean()
        miss_face = (~np.any(r[:, N_POSE+2*N_HAND:N_POSE+2*N_HAND+N_FACE] != 0, axis=-1)).mean()
        print(f"  missing per part: pose={miss_pose:.2f} Lh={miss_lh:.2f} Rh={miss_rh:.2f} face={miss_face:.2f}")
    print("\nInterface: (T,1629)= (T,543*3), order pose|Lh|Rh|face, missing=zeros. Enforced by code.")


if __name__ == "__main__":
    main()
