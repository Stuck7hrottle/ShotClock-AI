from __future__ import annotations

from typing import List, Dict
import numpy as np


def compute_rof(times_s: List[float]) -> Dict:
    times = np.array(times_s, dtype=float)
    if times.size < 2:
        return {"n_shots": int(times.size), "duration_s": 0.0, "mean_rpm": None, "series": []}

    dt = np.diff(times)
    rpm = 60.0 / dt
    series = [
        {"t": float(times[i]), "dt_s": float(dt[i - 1]), "rpm": float(rpm[i - 1])}
        for i in range(1, len(times))
    ]

    duration = float(times[-1] - times[0])
    return {
        "n_shots": int(times.size),
        "duration_s": duration,
        "mean_rpm": float(np.mean(rpm)),
        "median_rpm": float(np.median(rpm)),
        "p10_rpm": float(np.percentile(rpm, 10)),
        "p90_rpm": float(np.percentile(rpm, 90)),
        "max_rpm": float(np.max(rpm)),
        "series": series,
    }
