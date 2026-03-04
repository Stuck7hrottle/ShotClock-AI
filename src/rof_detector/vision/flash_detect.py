from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

from rof_detector.vision.roi import ROI, parse_roi, select_roi_interactive

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def _roi_patch(frame: np.ndarray, roi: ROI) -> np.ndarray:
    h, w = frame.shape[:2]
    x0 = max(0, min(w - 1, roi.x))
    y0 = max(0, min(h - 1, roi.y))
    x1 = max(0, min(w, roi.x + roi.w))
    y1 = max(0, min(h, roi.y + roi.h))
    return frame[y0:y1, x0:x1]


def confirm_shots_with_flash(
    video_path: Path,
    audio_events: List[Dict],
    roi: Optional[str],
    roi_interactive: bool = False,
) -> List[Dict]:
    if cv2 is None:
        raise RuntimeError("opencv-python not installed. Install with: pip install -e '.[video]'")

    if roi_interactive:
        roi_obj = select_roi_interactive(video_path)
    elif roi:
        roi_obj = parse_roi(roi)
    else:
        raise ValueError("ROI required unless --no-vision is set.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    video_events = []
    for e in audio_events:
        t = float(e["t"])
        idx = int(round(t * fps))
        window = range(max(0, idx - 2), idx + 3)

        patches = []
        for fi in window:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            patch = _roi_patch(gray, roi_obj).astype(np.float32)
            patches.append(patch)

        if len(patches) < 3:
            video_score = 0.0
        else:
            base = np.median(np.stack(patches[:2], axis=0), axis=0)
            deltas = [float(np.max(p - base)) for p in patches[2:]]
            peak = max(deltas) if deltas else 0.0
            video_score = float(max(0.0, min(1.0, peak / 255.0)))

        video_events.append({"t": t, "video_score": video_score, "roi": roi_obj.as_tuple()})

    cap.release()
    return video_events
