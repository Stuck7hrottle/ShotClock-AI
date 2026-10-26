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


def peak_amplitude(x: np.ndarray, sr: int, center: int, span_s: float = 0.015) -> float:
    """Absolute peak in the first `span_s` after the onset (file is peak-normalized)."""
    w = x[center : center + max(1, int(span_s * sr))]
    return float(np.max(np.abs(w))) if w.size else 0.0


def high_frequency_ratio(
    x: np.ndarray, sr: int, center: int, span_s: float = 0.008, split_hz: float = 3000.0
) -> float:
    """Fraction of onset energy above `split_hz` (of everything above 300 Hz).

    A direct muzzle blast has a sharp broadband crack. Reflections lose the top
    end (air absorption, soft surfaces), and bumps/thumps/speech never had it,
    so this is one of the better echo-vs-shot cues.
    """
    w = x[center : center + max(16, int(span_s * sr))]
    if w.size < 16:
        return 0.0
    spec = np.abs(np.fft.rfft(w * np.hanning(w.size))) ** 2
    freqs = np.fft.rfftfreq(w.size, 1.0 / sr)
    total = float(np.sum(spec[freqs >= 300.0])) + 1e-12
    return float(np.sum(spec[freqs >= split_hz]) / total)


def rise_time_ms(x: np.ndarray, sr: int, center: int, span_s: float = 0.015) -> float:
    """Milliseconds from the refined onset to the local absolute peak."""
    w = x[center : center + max(1, int(span_s * sr))]
    if not w.size:
        return 0.0
    return float(np.argmax(np.abs(w)) * 1000.0 / sr)


def add_context_features(candidates: list[dict], lookback_s: float = 0.15) -> None:
    """Annotate each candidate with how loud it is relative to what came just before.

    Echoes always trail a louder source by a few tens of milliseconds, so a
    candidate much quieter than a candidate shortly before it is suspect.
    Adds `rel_prev_amp` (capped at 2.0; 2.0 when nothing precedes it) and
    `louder_prev_lag_ms` (lag to that louder predecessor, 0 when none).
    """
    for i, c in enumerate(candidates):
        f = c["features"]
        amp = float(f["peak_amp"])
        best_amp, best_lag = 0.0, 0.0
        for j in range(i - 1, -1, -1):
            lag = float(c["t"]) - float(candidates[j]["t"])
            if lag > lookback_s:
                break
            prev_amp = float(candidates[j]["features"]["peak_amp"])
            if prev_amp > best_amp:
                best_amp, best_lag = prev_amp, lag
        rel = amp / best_amp if best_amp > 0 else 2.0
        f["rel_prev_amp"] = float(min(rel, 2.0))
        f["louder_prev_lag_ms"] = float(best_lag * 1000.0) if rel < 1.0 else 0.0


def spectral_flatness(x: np.ndarray, sr: int, center: int, span_s: float = 0.008) -> float:
    """Wiener entropy of the onset spectrum above 300 Hz, in [0, 1].

    A muzzle blast is close to broadband noise (high flatness); door thumps,
    speech pops and mic bumps are low-frequency and tonal, and mechanical
    clicks are band-limited, so both score lower.
    """
    w = x[center : center + max(16, int(span_s * sr))]
    if w.size < 16:
        return 0.0
    spec = np.abs(np.fft.rfft(w * np.hanning(w.size))) ** 2 + 1e-12
    freqs = np.fft.rfftfreq(w.size, 1.0 / sr)
    band = spec[(freqs >= 300.0) & (freqs <= 16000.0)]
    if band.size == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(band))) / np.mean(band))
