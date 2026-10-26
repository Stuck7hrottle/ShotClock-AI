from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, find_peaks

from rof_detector.audio.features import (
    add_context_features,
    crest_factor,
    high_frequency_ratio,
    impulsiveness_kurtosis,
    is_clipped,
    peak_amplitude,
    rise_time_ms,
    spectral_flatness,
)
from rof_detector.audio.preprocess import highpass, normalize
from rof_detector.audio.scoring import score_candidate


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
    hi_hz: float = 12000.0,
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
    # Band-pass emphasis suppresses slow boom / room tail energy while keeping
    # the high-frequency crack that marks a direct muzzle blast.
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
    # The onset function is half-wave rectified, so more than half of it is
    # exactly zero and a plain median/MAD collapses to ~0 (which made the
    # threshold, and with it the sensitivity setting, a no-op). Measure the
    # noise floor over the rising frames only.
    rising = onset[onset > 0.0]
    if rising.size == 0:
        rising = onset
    med = float(np.median(rising))
    mad = float(np.median(np.abs(rising - med)))
    mad = max(mad, 1e-6)
    thr = med + float(k) * mad
    return float(thr), float(med), float(mad)


def _prominence_threshold(onset: np.ndarray, thr: float, sensitivity: float = 0.5) -> float:
    onset = np.asarray(onset, dtype=np.float32)
    # Minimum prominence relative to the loudest onset in the file: 0.08 at the
    # default sensitivity, roughly halving at 1.0 and doubling at 0.1.
    s = float(np.clip(sensitivity, 0.1, 1.0))
    floor = 0.08 * 2.0 ** ((0.5 - s) * 2.5)
    positive = onset[onset > 0.0]
    if positive.size == 0:
        return float(floor)

    q75 = float(np.percentile(positive, 75))
    q90 = float(np.percentile(positive, 90))
    prom = max(floor, 0.33 * max(thr, q75), 0.18 * q90)
    return float(prom)


def _score_floor_from_sensitivity(sensitivity: float) -> float:
    s = float(np.clip(sensitivity, 0.1, 1.0))
    floor = 0.35 - 0.11 * s
    return float(np.clip(floor, 0.24, 0.34))


def _event_strength(e: dict) -> float:
    f = e.get("audio_features", {})
    return float(
        e.get("audio_score", 0.0)
        + 0.30 * float(f.get("onset_prominence") or 0.0)
        + 0.12 * float(f.get("onset_height") or 0.0)
    )


def _cluster_events(events: list[dict], gap_s: float) -> list[list[dict]]:
    if not events:
        return []
    ev = sorted(events, key=lambda e: float(e["t"]))
    clusters: list[list[dict]] = [[ev[0]]]
    for e in ev[1:]:
        if float(e["t"]) - float(clusters[-1][-1]["t"]) <= gap_s:
            clusters[-1].append(e)
        else:
            clusters.append([e])
    return clusters


def _insert_recovery_candidates(
    kept: list[dict],
    borderline: list[dict],
    *,
    min_sep_s: float,
    cluster_gap_s: float,
) -> list[dict]:
    if len(kept) < 2 or not borderline:
        return kept

    recovered = sorted(kept, key=lambda e: float(e["t"]))
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
    events: list[dict],
    borderline: list[dict],
    *,
    min_sep_s: float,
) -> list[dict]:
    if len(events) < 2:
        return events

    cluster_gap_s = 0.25
    events = sorted(events, key=lambda e: float(e["t"]))

    # Recover borderline candidates that fill holes inside burst-like regions.
    events = _insert_recovery_candidates(
        events,
        borderline,
        min_sep_s=min_sep_s,
        cluster_gap_s=cluster_gap_s,
    )

    return sorted(events, key=lambda e: float(e["t"]))


def estimate_echo_lag_ms(candidates: list[dict]) -> float | None:
    """Find a recording's dominant reflection delay, if it has one.

    A wall or berm returns every shot after the same delay, so quieter
    candidates that trail a louder one by a *consistent* lag are echoes, even
    when that lag is longer than the echo merge window. Returns the median lag
    of the largest group of such candidates (within +/-4 ms of each other) when
    at least two candidates agree, else None.
    """
    lags = sorted(
        float(c["features"]["louder_prev_lag_ms"])
        for c in candidates
        if float(c["features"].get("rel_prev_amp", 2.0)) < 0.7
        and 15.0 <= float(c["features"].get("louder_prev_lag_ms", 0.0)) <= 150.0
    )
    best: list[float] = []
    for i, lag in enumerate(lags):
        group = [v for v in lags[i:] if v - lag <= 8.0]
        if len(group) > len(best):
            best = group
    return float(np.median(best)) if len(best) >= 2 else None


def _echo_rel_amp_limits(sensitivity: float) -> tuple[float, float]:
    """(always-echo, echo-if-lag-matches) relative-amplitude limits.

    A candidate quieter than `hard` times a candidate just before it is treated
    as an echo outright; one quieter than `soft` is an echo when it also sits
    at the recording's dominant echo lag. Higher sensitivity is more lenient.
    """
    s = float(np.clip(sensitivity, 0.1, 1.0))
    hard = float(np.clip(0.40 - 0.30 * (s - 0.5), 0.2, 0.55))
    soft = float(np.clip(0.70 - 0.20 * (s - 0.5), 0.55, 0.8))
    return hard, soft


def _is_echo(features: dict, echo_lag_ms: float | None, sensitivity: float) -> bool:
    rel = float(features.get("rel_prev_amp", 2.0))
    hard, soft = _echo_rel_amp_limits(sensitivity)
    if rel < hard:
        return True
    lag = float(features.get("louder_prev_lag_ms", 0.0))
    return echo_lag_ms is not None and rel < soft and abs(lag - echo_lag_ms) <= 5.0


def _quiet_outlier_floor(events: list[dict], sensitivity: float) -> float:
    """Peak-amplitude floor below which an event is too quiet to be one of the
    shots being analysed (bumps, clicks, neighbouring lanes). Relative to the
    median of the louder half of events, so a few nuisances can't drag it down."""
    amps = sorted((float(e["audio_features"]["peak_amp"]) for e in events), reverse=True)
    if len(amps) < 3:
        return 0.0
    ref = float(np.median(amps[: max(1, len(amps) // 2)]))
    s = float(np.clip(sensitivity, 0.1, 1.0))
    ratio = float(np.clip(0.30 - 0.25 * (s - 0.5), 0.15, 0.45))
    return ratio * ref


def _refine_onset_sample(x: np.ndarray, sr: int, center: int, frac: float = 0.3) -> int:
    """Snap a coarse onset-frame position to the first sample where the
    waveform reaches `frac` of its local peak.

    The onset function works on 10 ms frames with a 2.5 ms hop, so a peak in it
    only says "the shot starts somewhere in this frame". Muzzle blasts rise in
    well under a millisecond, so the first sample that gets near the local
    maximum is a much better (and consistent) estimate of when the shot happened.
    """
    s0 = max(0, center - int(0.004 * sr))
    s1 = min(len(x), center + int(0.016 * sr))
    if s1 <= s0:
        return center
    a = np.abs(x[s0:s1])
    peak = float(np.max(a))
    if peak <= 0.0:
        return center
    first = int(np.argmax(a >= frac * peak))
    return s0 + first


def _enforce_min_separation(
    peaks: list[tuple[int, float, float]], min_sep_samples: int
) -> list[tuple[int, float, float]]:
    """Greedy non-maximum suppression on (sample, height, prominence) peaks:
    strongest first, drop anything closer than `min_sep_samples` to a kept peak.
    Allows ~1 ms of slack so shots exactly at the separation limit survive."""
    min_sep = max(1, min_sep_samples - min_sep_samples // 50)
    kept: list[tuple[int, float, float]] = []
    for p in sorted(peaks, key=lambda p: p[2], reverse=True):
        if all(abs(p[0] - q[0]) >= min_sep for q in kept):
            kept.append(p)
    return sorted(kept, key=lambda p: p[0])


def _load_mono_float(wav_path: Path) -> tuple[int, np.ndarray]:
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
    return int(sr), x


def find_candidates(
    wav_path: Path,
    *,
    sensitivity: float = 0.48,
    min_separation_ms: int = 35,
    environment: str = "auto",
) -> list[dict]:
    """Return every onset-peak candidate with its raw acoustic features and
    threshold context, before any accept/reject scoring is applied.

    Shared by detect_shots_audio (which scores and filters these) and
    train_shot_classifier.py (which labels them against ground truth to
    train the scoring model in audio/scoring.py).
    """
    sr, x = _load_mono_float(wav_path)

    onset, hop = _onset_function(x, sr)
    k = _k_from_sensitivity(environment, sensitivity)
    thr, med, mad = _mad_threshold(onset, k)
    pctl = _percentile_from_sensitivity(environment, sensitivity)
    thr_pctl = float(np.percentile(onset, pctl))

    prom = _prominence_threshold(onset, thr, sensitivity)
    # Peak-pick with only a small spacing constraint, then enforce the real
    # minimum separation on the refined (sample-accurate) times below. Doing
    # it on the coarse onset grid suppresses real shots that follow a shot
    # whose onset frame peaks late (slow-rising "thump" profiles).
    peaks, props = find_peaks(
        onset,
        height=thr,
        distance=max(1, int(0.012 * sr / hop)),
        prominence=prom,
    )
    heights = props.get("peak_heights", np.zeros(len(peaks), dtype=float))
    prominences = props.get("prominences", np.zeros(len(peaks), dtype=float))

    refined = [
        (_refine_onset_sample(x, sr, int(pk * hop)), float(heights[j]), float(prominences[j]))
        for j, pk in enumerate(peaks)
    ]
    refined = _enforce_min_separation(refined, int((min_separation_ms / 1000.0) * sr))

    candidates: list[dict] = []
    win = int(0.025 * sr)
    for center, peak_height, peak_prom in refined:
        t = center / sr
        s0 = max(0, center - win)
        s1 = min(len(x), center + win)
        w = x[s0:s1]

        cf = crest_factor(w)
        kurt = impulsiveness_kurtosis(w)
        clipped = is_clipped(w)

        prom_ratio = peak_prom / max(prom, 1e-6)
        height_ratio = peak_height / max(thr, 1e-6)

        candidates.append(
            {
                "t": float(t),
                "features": {
                    "crest_factor": float(cf),
                    "kurtosis": float(kurt),
                    "clipped": bool(clipped),
                    "onset_height": peak_height,
                    "onset_prominence": peak_prom,
                    "prom_ratio": float(prom_ratio),
                    "height_ratio": float(height_ratio),
                    "peak_amp": peak_amplitude(x, sr, center),
                    "hf_ratio": high_frequency_ratio(x, sr, center),
                    "rise_ms": rise_time_ms(x, sr, center),
                    "flatness": spectral_flatness(x, sr, center),
                },
                "threshold_context": {
                    "threshold": float(thr),
                    "onset_median": float(med),
                    "onset_mad": float(mad),
                    "k": float(k),
                    "prominence_threshold": float(prom),
                    "threshold_percentile": float(pctl),
                    "threshold_percentile_value": float(thr_pctl),
                },
            }
        )

    add_context_features(candidates)
    return candidates


def detect_shots_audio(
    wav_path: Path,
    *,
    sensitivity: float = 0.48,
    min_separation_ms: int = 35,
    echo_window_ms: int = 30,
    environment: str = "auto",
) -> list[dict]:
    """Return list of audio events: {t, audio_score, audio_features}."""
    events, _ = detect_shots_audio_detailed(
        wav_path,
        sensitivity=sensitivity,
        min_separation_ms=min_separation_ms,
        echo_window_ms=echo_window_ms,
        environment=environment,
    )
    return events


def detect_shots_audio_detailed(
    wav_path: Path,
    *,
    sensitivity: float = 0.48,
    min_separation_ms: int = 35,
    echo_window_ms: int = 30,
    environment: str = "auto",
) -> tuple[list[dict], list[dict]]:
    """Like detect_shots_audio, but also return the candidates that were
    rejected, each with a `reject_reason`, so a UI can offer them back."""
    raw_candidates = find_candidates(
        wav_path,
        sensitivity=sensitivity,
        min_separation_ms=min_separation_ms,
        environment=environment,
    )

    score_floor = _score_floor_from_sensitivity(sensitivity)
    recovery_floor = max(0.20, score_floor - 0.07)

    candidates: list[dict] = []
    for c in raw_candidates:
        feats = c["features"]
        ctx = c["threshold_context"]
        score = score_candidate(feats)

        event = {
            "t": c["t"],
            "audio_score": score,
            "audio_features": {
                **feats,
                "threshold_method": "mad+prominence+cleanup",
                **ctx,
                "score_floor": float(score_floor),
                "recovery_floor": float(recovery_floor),
            },
        }
        candidates.append(event)

    echo_lag_ms = estimate_echo_lag_ms(raw_candidates)
    for e in candidates:
        e["audio_features"]["echo_lag_ms"] = echo_lag_ms
        if _is_echo(e["audio_features"], echo_lag_ms, sensitivity):
            e["reject_reason"] = "echo"
        elif float(e["audio_score"]) < score_floor:
            e["reject_reason"] = "low score"

    quiet_floor = _quiet_outlier_floor(
        [e for e in candidates if "reject_reason" not in e], sensitivity
    )
    for e in candidates:
        if "reject_reason" not in e and float(e["audio_features"]["peak_amp"]) < quiet_floor:
            e["reject_reason"] = "too quiet"

    accepted = [e for e in candidates if "reject_reason" not in e]
    borderline = [
        e
        for e in candidates
        if e.get("reject_reason") == "low score" and float(e["audio_score"]) >= recovery_floor
    ]

    echo_window_s = echo_window_ms / 1000.0
    accepted.sort(key=lambda e: e["t"])
    merged: list[dict] = []
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
    final_events: list[dict] = []
    for e in cleaned:
        if not final_events or (e["t"] - final_events[-1]["t"]) > echo_window_s:
            final_events.append(e)
        elif _event_strength(e) > _event_strength(final_events[-1]):
            final_events[-1] = e

    kept = {id(e) for e in final_events}
    rejected = []
    for e in candidates:
        if id(e) in kept:
            continue
        e.setdefault("reject_reason", "merged (echo window)")
        rejected.append(e)
    for e in final_events:
        e.pop("reject_reason", None)
    return final_events, rejected
