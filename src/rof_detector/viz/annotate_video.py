from __future__ import annotations

from pathlib import Path

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

# BGR colors (OpenCV convention), matched to the tiers used in viz/plots.py.
_STRONG_COLOR_BGR = (80, 152, 26)  # green
_MODERATE_COLOR_BGR = (49, 174, 253)  # orange
_WEAK_COLOR_BGR = (39, 48, 215)  # red


def _event_score(e: dict) -> float:
    # `confidence` is only meaningful once video confirmation has blended in;
    # without it, fuse_scores caps confidence at 0.7 * audio_score, which
    # would make every audio-only detection look weak regardless of how
    # strong the impulse actually was. Prefer audio_score in that case.
    if e.get("video_score") is not None and e.get("confidence") is not None:
        return float(e["confidence"])
    for key in ("audio_score", "confidence"):
        v = e.get(key)
        if v is not None:
            return float(v)
    return 0.0


def _event_color_bgr(score: float) -> tuple[int, int, int]:
    if score >= 0.75:
        return _STRONG_COLOR_BGR
    if score >= 0.5:
        return _MODERATE_COLOR_BGR
    return _WEAK_COLOR_BGR


def annotate_video_with_events(video_path: Path, events: list[dict], out_path: Path) -> None:
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

    # Keep a short window per event so the label is readable rather than a
    # single-frame flash, and carry the event's confidence for coloring.
    hold_frames = max(1, round(0.25 * fps))
    event_by_frame: dict[int, dict] = {}
    for e in events:
        center = round(float(e["t"]) * fps)
        for fi in range(center, center + hold_frames):
            event_by_frame[fi] = e

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        e = event_by_frame.get(i)
        if e is not None:
            score = _event_score(e)
            color = _event_color_bgr(score)
            label = f"SHOT {score:.2f}"
            cv2.putText(
                frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA
            )
        writer.write(frame)
        i += 1

    cap.release()
    writer.release()
