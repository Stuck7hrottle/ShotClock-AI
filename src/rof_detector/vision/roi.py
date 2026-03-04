from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass(frozen=True)
class ROI:
    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


def parse_roi(s: str) -> ROI:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError('ROI must be "x,y,w,h"')
    x, y, w, h = (int(p) for p in parts)
    if w <= 0 or h <= 0:
        raise ValueError("ROI width/height must be > 0")
    return ROI(x=x, y=y, w=w, h=h)


def select_roi_interactive(video_path) -> ROI:
    if cv2 is None:
        raise RuntimeError("opencv-python not installed. Install with: pip install -e '.[video]'")
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read first frame for ROI selection.")
    r = cv2.selectROI("Select muzzle ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    x, y, w, h = map(int, r)
    return ROI(x=x, y=y, w=w, h=h)
