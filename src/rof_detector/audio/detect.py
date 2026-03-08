from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, find_peaks

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
    lo = max(1e-5, min(lo, 0.95))
    hi = max(lo + 1e-4, min(hi, 0.999))

    if not (0.0 < lo < hi < 1.0):
        return x.astype(np.float32, copy=False)

    b, a = butter(order, [lo, hi], btype="bandpass")
    return filtfilt(b, a, x).astype(np.float32, copy=False)


def _onset_function(x: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    # Light band-pass emphasis helps suppress slow boom / room tail energy.
    x_bp = _bandpass(x, sr)

    frame = int(0.010 * sr)  # 10 ms
    hop = int(0.0025 * sr)  # 2.5 ms
    rms = _frame_rms(x_bp, frame, hop)
    diff = np.maximum(0.0, np.diff(rms, prepend=rms[0]))
    m = float(np.max(diff) + 1e-12)
    return (diff / m).astype(np.float32), hop


def _percentile_from_sensitivity(environment: str, sensitivity: float) -> float:
    s = float(np.clip(sensitivity, 0.1, 1.0))
    base = 98.5 if environment in ("indoor", "auto") else 99.0
    p = base + (0.5 - s) * 2.0
    return float(np.clip(p, 95.0, 99.8))


def _k_from_sensitivity(environment: str, sensitivity: float) -> float:
    s = float(np.clip(sensitivity, 0.1, 1.0))
    k = 10.3 - 7.3 * s
    if environment in ("indoor", "auto"):
        k += 0.60
    return float(np.clip(k, 3.0, 13.0))


def _mad_threshold(onset: np.ndarray, k: float) -> tuple[float, float, float]:
    onset = np.asarray(onset, dtype=np.float32)
    med = float(np.median(onset))
    mad = float(np.median(np.abs(onset - med)))
    mad = max(mad, 1e-6)
    thr = med + float(k) * mad
    return float(thr), float(med), float(mad)


def _prominence_threshold(onset: np.ndarray, thr: float) -> float:
    onset = np.asarray(onset, dtype=np.float32)
    positive = onset[onset > 0.0]
    if positive.size == 0:
        return 0.08

    q75 = float(np.percentile(positive, 75))
    q90 = float(np.percentile(positive, 90))
    prom = max(0.08, 0.33 * max(thr, q75), 0.18 * q90)
    return float(prom)


def _score_floor_from_sensitivity(sensitivity: float) -> float:
    s = float(np.clip(sensitivity, 0.1, 1.0))
    floor = 0.35 - 0.11 * s
    return float(np.clip(floor, 0.24, 0.34))


def _event_strength(e: Dict) -> float:
    f = e.get("audio_features", {})
    return float(
        e.get("audio_score", 0.0)
        + 0.30 * float(f.get("onset_prominence") or 0.0)
        + 0.12 * float(f.get("onset_height") or 0.0)
    )


def _cluster_events(events: List[Dict], gap_s: float) -> List[List[Dict]]:
    if not events:
        return []
    ev = sorted(events, key=lambda e: float(e["t"]))
    clusters: List[List[Dict]] = [[ev[0]]]
    for e in ev[1:]:
        if float(e["t"]) - float(clusters[-1][-1]["t"]) <= gap_s:
            clusters[-1].append(e)
        else:
            clusters.append([e])
    return clusters


def _insert_recovery_candidates(
    kept: List[Dict],
    borderline: List[Dict],
    *,
    min_sep_s: float,
    cluster_gap_s: float,
) -> List[Dict]:
    if len(kept) < 2 or not borderline:
        return kept

    recovered = list(sorted(kept, key=lambda e: float(e["t"])))
    clusters = _cluster_events(recovered, cluster_gap_s)

    # Use plausible automatic-fire intervals only.
    global_dt = np.diff(np.array([e["t"] for e in recovered], dtype=float))
    global_dt = global_dt[(global_dt >= 0.05) & (global_dt <= 0.16)]
    global_dt_med = float(np.median(global_dt)) if global_dt.size else 0.09

    candidates = sorted(borderline, key=lambda e: float(e["t"]))
    for cl in clusters:
        if len(cl) < 2:
            continue

        cl_times = np.array([e["t"] for e in cl], dtype=float)
        local_dt = np.diff(cl_times)
        local_dt = local_dt[(local_dt >= 0.05) & (local_dt <= 0.16)]
        target_dt = float(np.median(local_dt)) if local_dt.size else global_dt_med
        target_dt = float(np.clip(target_dt, 0.06, 0.14))

        win_lo = float(cl_times[0] - max(0.12, 1.5 * target_dt))
        win_hi = float(cl_times[-1] + max(0.12, 1.5 * target_dt))

        for cand in candidates:
            t = float(cand["t"])
            if not (win_lo <= t <= win_hi):
                continue

            times = np.array([e["t"] for e in recovered], dtype=float)
            if np.min(np.abs(times - t)) < min_sep_s:
                continue

            insert_at = int(np.searchsorted(times, t))
            prev_t = float(times[insert_at - 1]) if insert_at > 0 else None
            next_t = float(times[insert_at]) if insert_at < len(times) else None

            prev_gap = (t - prev_t) if prev_t is not None else None
            next_gap = (next_t - t) if next_t is not None else None
            bridged = (next_t - prev_t) if (prev_t is not None and next_t is not None) else None

            # Recover only if this candidate fills an unusually large hole and creates
            # more burst-like spacing on one or both sides.
            should_insert = False
            if bridged is not None and bridged >= 1.7 * target_dt:
                lhs_ok = prev_gap is not None and 0.55 * target_dt <= prev_gap <= 1.55 * target_dt
                rhs_ok = next_gap is not None and 0.55 * target_dt <= next_gap <= 1.55 * target_dt
                should_insert = lhs_ok or rhs_ok

            if should_insert:
                recovered.append(cand)
                recovered.sort(key=lambda e: float(e["t"]))

    return recovered


def _cleanup_burst_structure(
    events: List[Dict],
    borderline: List[Dict],
    *,
    min_sep_s: float,
) -> List[Dict]:
    if len(events) < 2:
        return events

    cluster_gap_s = 0.25
    events = sorted(events, key=lambda e: float(e["t"]))

    # First try to recover borderline candidates inside already burst-like regions.
    events = _insert_recovery_candidates(
        events,
        borderline,
        min_sep_s=min_sep_s,
        cluster_gap_s=cluster_gap_s,
    )

    clusters = _cluster_events(events, cluster_gap_s)
    multi_clusters = [cl for cl in clusters if len(cl) >= 2]
    if not multi_clusters:
        return events

    cleaned: List[Dict] = []
    for idx, cl in enumerate(clusters):
        if len(cl) >= 2:
            cleaned.extend(cl)
            continue

        e = cl[0]
        strength = _event_strength(e)
        t = float(e["t"])
        prev_multi_gap = None
        next_multi_gap = None

        for j in range(idx - 1, -1, -1):
            if len(clusters[j]) >= 2:
                prev_multi_gap = t - float(clusters[j][-1]["t"])
                break
        for j in range(idx + 1, len(clusters)):
            if len(clusters[j]) >= 2:
                next_multi_gap = float(clusters[j][0]["t"]) - t
                break

        # Keep only unusually strong singletons that are attached to a nearby real burst.
        near_real_burst = (prev_multi_gap is not None and prev_multi_gap <= 0.18) or (
            next_multi_gap is not None and next_multi_gap <= 0.18
        )
        if strength >= 0.92 and near_real_burst:
            cleaned.append(e)

    cleaned.sort(key=lambda e: float(e["t"]))
    return cleaned if cleaned else events


def detect_shots_audio(
    wav_path: Path,
    *,
    sensitivity: float = 0.48,
    min_separation_ms: int = 35,
    echo_window_ms: int = 30,
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
    prom = _prominence_threshold(onset, thr)
    peaks, props = find_peaks(
        onset,
        height=thr,
        distance=min_sep_frames,
        prominence=prom,
    )

    candidates: List[Dict] = []
    win = int(0.025 * sr)
    heights = props.get("peak_heights", np.zeros(len(peaks), dtype=float))
    prominences = props.get("prominences", np.zeros(len(peaks), dtype=float))
    score_floor = _score_floor_from_sensitivity(sensitivity)
    recovery_floor = max(0.20, score_floor - 0.07)

    for j, pk in enumerate(peaks):
        t = (pk * hop) / sr
        center = int(pk * hop)
        s0 = max(0, center - win)
        s1 = min(len(x), center + win)
        w = x[s0:s1]

        cf = crest_factor(w)
        kurt = impulsiveness_kurtosis(w)
        clipped = is_clipped(w)

        score = 0.0
        score += min(1.0, (cf / 10.0)) * (0.52 if not clipped else 0.24)
        score += min(1.0, (kurt / 50.0)) * 0.30

        peak_prom = float(prominences[j]) if j < len(prominences) else 0.0
        peak_height = float(heights[j]) if j < len(heights) else 0.0

        prom_ratio = peak_prom / max(prom, 1e-6)
        height_ratio = peak_height / max(thr, 1e-6)

        score += min(0.22, max(0.0, prom_ratio - 1.0) * 0.10)
        score += min(0.14, max(0.0, height_ratio - 1.0) * 0.05)

        # Penalize weak/transient nuisance events that barely clear threshold.
        if prom_ratio < 1.18:
            score -= 0.08
        if height_ratio < 1.10:
            score -= 0.05

        score = float(np.clip(score, 0.0, 1.0))

        event = {
            "t": float(t),
            "audio_score": score,
            "audio_features": {
                "crest_factor": float(cf),
                "kurtosis": float(kurt),
                "clipped": bool(clipped),
                "onset_height": float(heights[j]) if j < len(heights) else None,
                "onset_prominence": peak_prom,
                "threshold_method": "mad+prominence+cleanup",
                "threshold": float(thr),
                "onset_median": float(med),
                "onset_mad": float(mad),
                "k": float(k),
                "prominence_threshold": float(prom),
                "score_floor": float(score_floor),
                "recovery_floor": float(recovery_floor),
                "threshold_percentile": float(pctl),
                "threshold_percentile_value": float(thr_pctl),
            },
        }
        candidates.append(event)

    accepted = [e for e in candidates if float(e["audio_score"]) >= score_floor]
    borderline = [e for e in candidates if recovery_floor <= float(e["audio_score"]) < score_floor]

    echo_window_s = echo_window_ms / 1000.0
    accepted.sort(key=lambda e: e["t"])
    merged: List[Dict] = []
    for e in accepted:
        if not merged:
            merged.append(e)
            continue

        prev = merged[-1]
        dt = e["t"] - prev["t"]
        if dt <= echo_window_s:
            if _event_strength(e) > _event_strength(prev):
                merged[-1] = e
        else:
            merged.append(e)

    cleaned = _cleanup_burst_structure(
        merged,
        borderline,
        min_sep_s=float(min_separation_ms) / 1000.0,
    )

    # Final sort and de-dup pass.
    cleaned.sort(key=lambda e: e["t"])
    final_events: List[Dict] = []
    for e in cleaned:
        if not final_events or (e["t"] - final_events[-1]["t"]) > echo_window_s:
            final_events.append(e)
        elif _event_strength(e) > _event_strength(final_events[-1]):
            final_events[-1] = e

    return final_events
