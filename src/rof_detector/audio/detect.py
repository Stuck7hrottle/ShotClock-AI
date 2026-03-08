from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, find_peaks

from rof_detector.audio.features import crest_factor, impulsiveness_kurtosis, is_clipped
from rof_detector.audio.preprocess import highpass, normalize


# Tuned replacement for ShotClock-AI's audio/detect.py.
# Goals:
# - cut reverberation / tail-energy false positives
# - make burst gaps cleaner so downstream burst splitting can work
# - remain drop-in compatible with the existing pipeline


def _frame_rms(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    n = max(1, (len(x) - frame) // hop + 1)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = i * hop
        w = x[s : s + frame]
        out[i] = np.sqrt(np.mean(w * w) + 1e-12)
    return out



def _bandpass(
    x: np.ndarray,
    sr: int,
    lo_hz: float = 250.0,
    hi_hz: float = 4200.0,
    order: int = 4,
) -> np.ndarray:
    nyq = 0.5 * sr
    if nyq <= 0:
        return x.astype(np.float32, copy=False)

    lo = lo_hz / nyq
    hi = hi_hz / nyq

    # Clamp safely into the valid digital filter range.
    lo = max(1e-5, min(lo, 0.95))
    hi = max(lo + 1e-4, min(hi, 0.999))

    # If sample rate is too low for a proper band-pass, fall back gracefully.
    if not (0.0 < lo < hi < 1.0):
        return x.astype(np.float32, copy=False)

    b, a = butter(order, [lo, hi], btype="bandpass")
    return filtfilt(b, a, x).astype(np.float32, copy=False)



def _onset_function(x: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """
    Energy-difference onset on a band-limited waveform.

    The original detector used a straight RMS-difference envelope on a high-passed
    signal. That is fast, but it tends to keep mechanism noise, reverberant tails,
    and broad low-information transients. A modest band-pass helps concentrate on
    the sharp muzzle-blast region without changing the overall architecture.
    """
    frame = int(0.008 * sr)  # 8 ms
    hop = int(0.0025 * sr)  # 2.5 ms
    x_bp = _bandpass(x, sr)
    rms = _frame_rms(x_bp, frame, hop)
    diff = np.maximum(0.0, np.diff(rms, prepend=rms[0]))
    m = float(np.max(diff) + 1e-12)
    return (diff / m).astype(np.float32), hop



def _percentile_from_sensitivity(environment: str, sensitivity: float) -> float:
    s = float(np.clip(sensitivity, 0.1, 1.0))
    base = 98.8 if environment in ("indoor", "auto") else 99.1
    p = base + (0.5 - s) * 2.0
    return float(np.clip(p, 96.0, 99.9))



def _k_from_sensitivity(environment: str, sensitivity: float) -> float:
    """
    Convert sensitivity in [0.1, 1.0] to a robust threshold multiplier k for MAD thresholding.
    Higher sensitivity => lower k => more detections.
    """
    s = float(np.clip(sensitivity, 0.1, 1.0))

    # More conservative than the stock file; this is deliberate.
    k = 11.0 - 7.5 * s
    if environment in ("indoor", "auto"):
        k += 0.75
    return float(np.clip(k, 3.2, 13.5))



def _mad_threshold(onset: np.ndarray, k: float) -> tuple[float, float, float]:
    onset = np.asarray(onset, dtype=np.float32)
    med = float(np.median(onset))
    mad = float(np.median(np.abs(onset - med)))
    mad = max(mad, 1e-6)
    thr = med + float(k) * mad
    return float(thr), float(med), float(mad)



def _prominence_threshold(onset: np.ndarray, thr: float, sensitivity: float) -> float:
    onset = np.asarray(onset, dtype=np.float32)
    pos = onset[onset > 0.0]
    if pos.size == 0:
        return 0.10

    q75 = float(np.percentile(pos, 75))
    q90 = float(np.percentile(pos, 90))
    s = float(np.clip(sensitivity, 0.1, 1.0))

    # Slightly relax prominence at very high sensitivity, but keep a firm floor.
    scale = 0.42 - 0.10 * s
    prom = max(0.10, scale * max(thr, q75), 0.22 * q90)
    return float(prom)



def _score_floor_from_sensitivity(sensitivity: float, environment: str) -> float:
    s = float(np.clip(sensitivity, 0.1, 1.0))
    floor = 0.39 - 0.12 * s
    if environment in ("indoor", "auto"):
        floor += 0.02
    return float(np.clip(floor, 0.28, 0.40))



def _event_strength(event: Dict) -> float:
    feats = event.get("audio_features", {})
    return float(
        event.get("audio_score", 0.0)
        + 0.28 * float(feats.get("onset_prominence") or 0.0)
        + 0.14 * float(feats.get("onset_height") or 0.0)
    )



def _merge_echoes(events: List[Dict], echo_window_ms: int) -> List[Dict]:
    if not events:
        return []

    echo_window_s = echo_window_ms / 1000.0
    events = sorted(events, key=lambda e: e["t"])
    merged = [events[0]]

    for e in events[1:]:
        prev = merged[-1]
        dt = e["t"] - prev["t"]
        if dt <= echo_window_s:
            if _event_strength(e) > _event_strength(prev):
                merged[-1] = e
        else:
            merged.append(e)

    return merged



def _prune_bridge_events(events: List[Dict]) -> List[Dict]:
    """
    Remove weak candidates that only serve to bridge gaps between real bursts.

    This is intentionally conservative:
    - never removes the first/last event
    - never removes strong events
    - only removes weak interior events when their local timing is non-burst-like
      or when they split a long gap into two mediocre gaps.
    """
    if len(events) < 3:
        return events

    kept = events[:]
    changed = True
    while changed and len(kept) >= 3:
        changed = False
        new_events = [kept[0]]

        for i in range(1, len(kept) - 1):
            prev_e = new_events[-1]
            cur_e = kept[i]
            next_e = kept[i + 1]

            left = float(cur_e["t"] - prev_e["t"])
            right = float(next_e["t"] - cur_e["t"])
            combined = float(next_e["t"] - prev_e["t"])
            strength = _event_strength(cur_e)
            score = float(cur_e.get("audio_score", 0.0))

            # Events with at least one close, burst-like neighbor are usually fine.
            has_close_neighbor = min(left, right) <= 0.115

            # Weak event that lives in the middle of a broad timing hole.
            broad_hole = (left >= 0.14 and right >= 0.14 and combined >= 0.34)

            # Weak splitter of a long gap into two mediocre pieces.
            gap_splitter = (combined >= 0.26 and left >= 0.09 and right >= 0.09)

            # Very asymmetrical weak bridge, often reverberation or tail energy.
            asymmetric_bridge = max(left, right) >= 0.17 and min(left, right) >= 0.07

            remove = False
            if strength < 0.34 and broad_hole:
                remove = True
            elif strength < 0.32 and gap_splitter and not has_close_neighbor:
                remove = True
            elif score < 0.31 and strength < 0.36 and asymmetric_bridge and combined >= 0.24:
                remove = True

            if remove:
                changed = True
                continue

            new_events.append(cur_e)

        new_events.append(kept[-1])
        kept = new_events

    return kept



def detect_shots_audio(
    wav_path: Path,
    *,
    sensitivity: float = 0.5,
    min_separation_ms: int = 50,
    echo_window_ms: int = 45,
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

    k = _k_from_sensitivity(environment, sensitivity)
    thr, med, mad = _mad_threshold(onset, k)

    pctl = _percentile_from_sensitivity(environment, sensitivity)
    thr_pctl = float(np.percentile(onset, pctl))

    min_sep_frames = max(1, int((min_separation_ms / 1000.0) * sr / hop))
    prom = _prominence_threshold(onset, thr, sensitivity)
    peaks, props = find_peaks(
        onset,
        height=max(thr, 0.80 * thr_pctl),
        distance=min_sep_frames,
        prominence=prom,
    )

    events: List[Dict] = []
    win = int(0.020 * sr)  # ±20 ms is enough for impulsive scoring
    heights = props.get("peak_heights", np.zeros(len(peaks), dtype=float))
    prominences = props.get("prominences", np.zeros(len(peaks), dtype=float))
    score_floor = _score_floor_from_sensitivity(sensitivity, environment)

    for j, pk in enumerate(peaks):
        t = (pk * hop) / sr
        center = int(pk * hop)
        s0 = max(0, center - win)
        s1 = min(len(x), center + win)
        w = x[s0:s1]

        cf = crest_factor(w)
        kurt = impulsiveness_kurtosis(w)
        clipped = is_clipped(w)
        peak_height = float(heights[j]) if j < len(heights) else 0.0
        peak_prom = float(prominences[j]) if j < len(prominences) else 0.0

        # Main feature score.
        score = 0.0
        score += min(1.0, cf / 10.0) * (0.55 if not clipped else 0.28)
        score += min(1.0, kurt / 55.0) * 0.25

        # Onset quality matters a lot for suppressing tail noise.
        score += min(0.22, peak_height * 0.16)
        score += min(0.18, peak_prom * 0.22)
        score = float(np.clip(score, 0.0, 1.0))

        if score < score_floor:
            continue

        events.append(
            {
                "t": float(t),
                "audio_score": score,
                "audio_features": {
                    "crest_factor": float(cf),
                    "kurtosis": float(kurt),
                    "clipped": bool(clipped),
                    "onset_height": peak_height,
                    "onset_prominence": peak_prom,
                    "threshold_method": "mad+prominence",
                    "threshold": float(thr),
                    "onset_median": float(med),
                    "onset_mad": float(mad),
                    "k": float(k),
                    "prominence_threshold": float(prom),
                    "score_floor": float(score_floor),
                    "threshold_percentile": float(pctl),
                    "threshold_percentile_value": float(thr_pctl),
                },
            }
        )

    events = _merge_echoes(events, echo_window_ms=echo_window_ms)
    events = _prune_bridge_events(events)
    return events
