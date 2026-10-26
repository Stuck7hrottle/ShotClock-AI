from __future__ import annotations

import numpy as np


def compute_rof(times_s: list[float], burst_gap_s: float = 0.25) -> dict:
    """Rate-of-fire statistics for a list of shot times.

    `mean_rpm` is 60 / mean split, i.e. shots-per-minute over the whole span.
    Averaging the instantaneous 60/dt values instead is biased upwards (splits
    of 0.08 s and 0.12 s average to 625 RPM, though the true rate is 600).

    `cyclic_rpm` uses only splits inside bursts (<= `burst_gap_s`), so pauses
    between bursts or trigger pulls don't drag it down. It is the number
    people usually mean by a gun's rate of fire.
    """
    times = np.array(times_s, dtype=float)
    if times.size < 2:
        return {
            "n_shots": int(times.size),
            "duration_s": 0.0,
            "mean_rpm": None,
            "cyclic_rpm": None,
            "series": [],
        }

    dt = np.diff(times)
    rpm = 60.0 / dt
    series = [
        {"t": float(times[i]), "dt_s": float(dt[i - 1]), "rpm": float(rpm[i - 1])}
        for i in range(1, len(times))
    ]

    in_burst = dt[dt <= burst_gap_s]
    duration = float(times[-1] - times[0])
    return {
        "n_shots": int(times.size),
        "duration_s": duration,
        "mean_rpm": float(60.0 / np.mean(dt)),
        "cyclic_rpm": float(60.0 / np.mean(in_burst)) if in_burst.size else None,
        "median_rpm": float(np.median(rpm)),
        "p10_rpm": float(np.percentile(rpm, 10)),
        "p90_rpm": float(np.percentile(rpm, 90)),
        "max_rpm": float(np.max(rpm)),
        "series": series,
    }
