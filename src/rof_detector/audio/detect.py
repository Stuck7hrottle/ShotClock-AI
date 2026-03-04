from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks

from rof_detector.audio.preprocess import highpass, normalize
from rof_detector.audio.features import crest_factor, impulsiveness_kurtosis, is_clipped


def _frame_rms(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    n = max(1, (len(x) - frame) // hop + 1)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = i * hop
        w = x[s : s + frame]
        out[i] = np.sqrt(np.mean(w * w) + 1e-12)
    return out


def _onset_function(x: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    frame = int(0.010 * sr)   # 10ms
    hop = int(0.0025 * sr)    # 2.5ms
    rms = _frame_rms(x, frame, hop)
    diff = np.maximum(0.0, np.diff(rms, prepend=rms[0]))
    m = float(np.max(diff) + 1e-12)
    return (diff / m).astype(np.float32), hop


def _percentile_from_sensitivity(environment: str, sensitivity: float) -> float:
    s = float(np.clip(sensitivity, 0.1, 1.0))
    base = 98.5 if environment in ("indoor", "auto") else 99.0
    p = base + (0.5 - s) * 2.0
    return float(np.clip(p, 95.0, 99.8))


def detect_shots_audio(
    wav_path: Path,
    *,
    sensitivity: float = 0.5,
    min_separation_ms: int = 35,
    echo_window_ms: int = 60,
    environment: str = "auto",
) -> List[Dict]:
    """Return list of audio events: {t, audio_score, audio_features}."""
    sr, x = wavfile.read(str(wav_path))
    if x.ndim > 1:
        x = x[:, 0]

    if x.dtype.kind in ("i", "u"):
        maxv = np.iinfo(x.dtype).max
        x = (x.astype(np.float32) / maxv).astype(np.float32)
    else:
        x = x.astype(np.float32)

    x = highpass(x, sr, cutoff_hz=120.0)
    x = normalize(x)

    onset, hop = _onset_function(x, sr)

    pctl = _percentile_from_sensitivity(environment, sensitivity)
    thr = float(np.percentile(onset, pctl))
    min_sep_frames = max(1, int((min_separation_ms / 1000.0) * sr / hop))

    peaks, props = find_peaks(onset, height=thr, distance=min_sep_frames)

    events = []
    win = int(0.025 * sr)
    heights = props.get("peak_heights", np.zeros(len(peaks), dtype=float))
    for j, pk in enumerate(peaks):
        t = (pk * hop) / sr
        center = int(pk * hop)
        s0 = max(0, center - win)
        s1 = min(len(x), center + win)
        w = x[s0:s1]

        cf = crest_factor(w)
        k = impulsiveness_kurtosis(w)
        clipped = is_clipped(w)

        score = 0.0
        score += min(1.0, (cf / 10.0)) * (0.6 if not clipped else 0.3)
        score += min(1.0, (k / 50.0)) * 0.4
        score = float(np.clip(score, 0.0, 1.0))

        events.append(
            {
                "t": float(t),
                "audio_score": score,
                "audio_features": {
                    "crest_factor": float(cf),
                    "kurtosis": float(k),
                    "clipped": bool(clipped),
                    "onset_height": float(heights[j]) if j < len(heights) else None,
                    "threshold_percentile": float(pctl),
                },
            }
        )

    echo_window_s = echo_window_ms / 1000.0
    events.sort(key=lambda e: e["t"])
    merged = []
    for e in events:
        if not merged:
            merged.append(e)
            continue
        dt = e["t"] - merged[-1]["t"]
        if dt <= echo_window_s:
            prev = merged[-1]
            if (e["audio_score"] > prev["audio_score"] + 0.05) and (dt > 0.01):
                merged[-1] = e
        else:
            merged.append(e)

    return merged
