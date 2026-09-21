from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram

# Confidence tiers used to color-code events across the waveform and
# spectrogram panels, so a strong detection is visually distinct from a
# borderline/recovered one at a glance.
_STRONG_COLOR = "#1a9850"  # confidence >= 0.75
_MODERATE_COLOR = "#fdae61"  # 0.5 <= confidence < 0.75
_WEAK_COLOR = "#d73027"  # confidence < 0.5


def _event_score(e: dict) -> float:
    # `confidence` is only meaningful once video confirmation has blended in;
    # without it, fuse_scores caps confidence at 0.7 * audio_score, which
    # would make every audio-only detection look weak regardless of how
    # strong the impulse actually was. Prefer audio_score in that case.
    if e.get("video_score") is not None and e.get("confidence") is not None:
        return float(e["confidence"])
    for key in ("audio_score", "confidence"):
        v = e.get(key)
        if v is not None:
            return float(v)
    return 0.0


def _event_color(score: float) -> str:
    if score >= 0.75:
        return _STRONG_COLOR
    if score >= 0.5:
        return _MODERATE_COLOR
    return _WEAK_COLOR


def _load_mono_float(wav_path: Path) -> tuple[int, np.ndarray]:
    sr, x = wavfile.read(str(wav_path))
    if x.ndim > 1:
        x = x[:, 0]
    if x.dtype.kind in ("i", "u"):
        maxv = np.iinfo(x.dtype).max
        x = x.astype(np.float32) / maxv
    else:
        x = x.astype(np.float32)
    return int(sr), x


def plot_waveform_with_events(wav_path: Path, events: list[dict], out_png: Path) -> None:
    sr, x = _load_mono_float(wav_path)
    t = np.arange(len(x), dtype=np.float32) / float(sr)

    fig, (ax_wave, ax_spec) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1]}
    )

    ax_wave.plot(t, x, color="#4575b4", linewidth=0.6)
    ymax = float(np.max(np.abs(x))) if x.size else 1.0
    ax_wave.set_ylim(-ymax * 1.1, ymax * 1.1)

    for e in events:
        score = _event_score(e)
        color = _event_color(score)
        et = float(e["t"])
        ax_wave.axvline(et, color=color, linewidth=1.2, alpha=0.85)
        ax_wave.annotate(
            f"{score:.2f}",
            xy=(et, ymax * 1.02),
            xycoords="data",
            annotation_clip=False,
            fontsize=6,
            color=color,
            rotation=90,
            ha="center",
            va="bottom",
        )

    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_title("Waveform with detected events")

    strong = plt.Line2D([], [], color=_STRONG_COLOR, label="strong (>=0.75)")
    moderate = plt.Line2D([], [], color=_MODERATE_COLOR, label="moderate (0.5-0.75)")
    weak = plt.Line2D([], [], color=_WEAK_COLOR, label="weak/borderline (<0.5)")
    ax_wave.legend(handles=[strong, moderate, weak], loc="upper right", fontsize=7)

    nperseg = max(64, int(0.010 * sr))
    noverlap = int(nperseg * 0.75)
    freqs, times, sxx = spectrogram(x, fs=sr, nperseg=nperseg, noverlap=noverlap)
    sxx_db = 10.0 * np.log10(sxx + 1e-12)
    ax_spec.pcolormesh(times, freqs, sxx_db, shading="gouraud", cmap="magma")
    ax_spec.set_ylim(0, min(8000, sr / 2))
    for e in events:
        color = _event_color(_event_score(e))
        ax_spec.axvline(float(e["t"]), color=color, linewidth=1.0, alpha=0.8)

    ax_spec.set_xlabel("Time (s)")
    ax_spec.set_ylabel("Frequency (Hz)")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
