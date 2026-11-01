"""Video -> raw landmark .npy using MediaPipe Holistic (real data path; not used by mock data).

    python -m signbridge_brain.features.extract --src videos/ --dst data/landmarks/

Output per video: float32 (T, 1629), order pose|left hand|right hand|face, zeros when not detected.
Requires: pip install -e ".[extract]"  (mediapipe + opencv-python). Pin versions in FEATURE_SPEC.md.
NOTE: MediaPipe's "left"/"right" hand follows its own convention; keep it unchanged and
unmirrored across all components, or the feature layout silently differs between A, B and C.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .landmarks import N_FACE, N_HAND, N_LANDMARKS, N_POSE, FEATURE_DIM


def _fill(arr: np.ndarray, start: int, landmark_list, n: int) -> None:
    if landmark_list is None:
        return
    pts = landmark_list.landmark
    if len(pts) != n:
        return
    arr[start:start + n] = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32)


def results_to_frame(results) -> np.ndarray:
    """One MediaPipe Holistic result -> (543, 3), zeros for anything not detected."""
    frame = np.zeros((N_LANDMARKS, 3), dtype=np.float32)
    _fill(frame, 0, results.pose_landmarks, N_POSE)
    _fill(frame, N_POSE, results.left_hand_landmarks, N_HAND)
    _fill(frame, N_POSE + N_HAND, results.right_hand_landmarks, N_HAND)
    _fill(frame, N_POSE + 2 * N_HAND, results.face_landmarks, N_FACE)
    return frame


class HolisticExtractor:
    """Reusable extractor (Component C can reuse this for the live webcam loop)."""

    def __init__(self, model_complexity: int = 1, static_image_mode: bool = False):
        import mediapipe as mp  # lazy: mock-data workflow does not need it

        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            refine_face_landmarks=False,  # keeps exactly 468 face points
        )

    def frame(self, bgr_image) -> np.ndarray:
        import cv2

        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        return results_to_frame(self._holistic.process(rgb))

    def close(self) -> None:
        self._holistic.close()


def extract_video(path: str | Path, extractor: HolisticExtractor | None = None) -> np.ndarray:
    import cv2

    own = extractor is None
    extractor = extractor or HolisticExtractor()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    frames = []
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(extractor.frame(img).reshape(-1))
    finally:
        cap.release()
        if own:
            extractor.close()
    if not frames:
        raise IOError(f"No frames decoded from {path}")
    out = np.stack(frames).astype(np.float32)
    assert out.shape[1] == FEATURE_DIM
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Extract MediaPipe Holistic landmarks from videos")
    ap.add_argument("--src", required=True, help="video file or directory")
    ap.add_argument("--dst", required=True, help="output directory for .npy files")
    ap.add_argument("--ext", default=".mp4,.avi,.mov,.mkv")
    ap.add_argument("--complexity", type=int, default=1)
    args = ap.parse_args(argv)

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    exts = tuple(args.ext.split(","))
    videos = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() in exts)
    extractor = HolisticExtractor(model_complexity=args.complexity)
    try:
        for i, v in enumerate(videos, 1):
            out = dst / (v.stem + ".npy")
            if out.exists():
                continue
            np.save(out, extract_video(v, extractor))
            print(f"[{i}/{len(videos)}] {v.name} -> {out.name}")
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
