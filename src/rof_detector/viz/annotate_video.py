from __future__ import annotations

from pathlib import Path
from typing import List, Dict

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def annotate_video_with_events(video_path: Path, events: List[Dict], out_path: Path) -> None:
    if cv2 is None:
        raise RuntimeError("opencv-python not installed. Install with: pip install -e '.[video]'")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    event_frames = set(int(round(float(e["t"]) * fps)) for e in events)

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in event_frames:
            cv2.putText(
                frame, "SHOT", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA
            )
        writer.write(frame)
        i += 1

    cap.release()
    writer.release()
