from __future__ import annotations

import numpy as np


def crest_factor(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x * x)) + 1e-12)
    peak = float(np.max(np.abs(x)) + 1e-12)
    return peak / rms


def impulsiveness_kurtosis(x: np.ndarray) -> float:
    m = float(np.mean(x))
    v = float(np.mean((x - m) ** 2) + 1e-12)
    k = float(np.mean((x - m) ** 4) / (v * v))
    return k


def is_clipped(x: np.ndarray, clip_frac_thresh: float = 0.01) -> bool:
    if x.size == 0:
        return False
    a = np.abs(x)
    near = float(np.mean(a >= 0.999))
    return bool(near >= clip_frac_thresh)
