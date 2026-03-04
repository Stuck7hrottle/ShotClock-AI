from __future__ import annotations

from typing import List, Dict
import numpy as np


def segment_bursts(times_s: List[float], burst_gap_s: float = 0.4) -> List[Dict]:
    times = np.array(times_s, dtype=float)
    if times.size == 0:
        return []
    bursts = []
    start = 0
    for i in range(1, len(times)):
        if (times[i] - times[i - 1]) > burst_gap_s:
            bursts.append({"start_index": int(start), "end_index": int(i - 1)})
            start = i
    bursts.append({"start_index": int(start), "end_index": int(len(times) - 1)})
    return bursts


def summarize_bursts(times_s: List[float], bursts: List[Dict]) -> List[Dict]:
    times = np.array(times_s, dtype=float)
    out = []
    for b in bursts:
        s = int(b["start_index"])
        e = int(b["end_index"])
        seg = times[s : e + 1]
        if seg.size < 2:
            out.append(
                {
                    "start_t": float(seg[0]) if seg.size else None,
                    "end_t": float(seg[-1]) if seg.size else None,
                    "n_shots": int(seg.size),
                    "mean_rpm": None,
                }
            )
            continue
        dt = np.diff(seg)
        rpm = 60.0 / dt
        out.append(
            {
                "start_t": float(seg[0]),
                "end_t": float(seg[-1]),
                "n_shots": int(seg.size),
                "duration_s": float(seg[-1] - seg[0]),
                "mean_rpm": float(np.mean(rpm)),
                "median_rpm": float(np.median(rpm)),
                "max_rpm": float(np.max(rpm)),
            }
        )
    return out
