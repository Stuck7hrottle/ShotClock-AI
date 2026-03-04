from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict


def export_events_csv(path: Path, events: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["t", "confidence", "audio_score", "video_score"])
        w.writeheader()
        for e in events:
            w.writerow(
                {
                    "t": e.get("t"),
                    "confidence": e.get("confidence"),
                    "audio_score": e.get("audio_score"),
                    "video_score": e.get("video_score"),
                }
            )
