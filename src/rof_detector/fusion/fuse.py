from __future__ import annotations

from typing import List, Dict, Optional


def fuse_scores(audio_events: List[Dict], video_events: Optional[List[Dict]]) -> List[Dict]:
    out: List[Dict] = []
    video_by_t = {}
    if video_events:
        for ve in video_events:
            video_by_t[round(float(ve["t"]), 3)] = ve

    for ae in audio_events:
        t = float(ae["t"])
        a = float(ae.get("audio_score", 0.0))
        ve = video_by_t.get(round(t, 3))
        v = float(ve.get("video_score", 0.0)) if ve else 0.0

        final = 0.7 * a + 0.3 * v
        if a >= 0.75 or v >= 0.80:
            final = max(final, 0.75)

        out.append(
            {
                "t": t,
                "confidence": float(max(0.0, min(1.0, final))),
                "audio_score": a,
                "video_score": v if ve else None,
            }
        )
    return out
