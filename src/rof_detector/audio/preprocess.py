from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def highpass(x: np.ndarray, sr: int, cutoff_hz: float = 120.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * sr
    b, a = butter(order, cutoff_hz / nyq, btype="highpass")
    return filtfilt(b, a, x).astype(np.float32, copy=False)


def normalize(x: np.ndarray) -> np.ndarray:
    m = float(np.max(np.abs(x)) + 1e-12)
    return (x / m).astype(np.float32, copy=False)
