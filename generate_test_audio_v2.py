import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter

SR = 48_000
RNG = np.random.default_rng(1337)


@dataclass
class Event:
    time_s: float
    kind: str
    amplitude: float
    profile: str
    parent_index: int | None = None
    note: str | None = None


# ---------- Signal helpers ----------
def normalize(audio: np.ndarray, peak: float = 0.98) -> np.ndarray:
    max_val = float(np.max(np.abs(audio)))
    if max_val > 0:
        audio = audio * (peak / max_val)
    return audio.astype(np.float32)


def apply_fade(sig: np.ndarray, fade_ms: float = 2.0, sr: int = SR) -> np.ndarray:
    n = len(sig)
    fade_n = min(int(sr * fade_ms / 1000.0), max(1, n // 8))
    env = np.ones(n, dtype=np.float32)
    env[:fade_n] *= np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
    env[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
    return sig * env


def one_pole_lowpass(x: np.ndarray, cutoff_hz: float, sr: int = SR) -> np.ndarray:
    if cutoff_hz >= sr / 2:
        return x.copy()
    alpha = np.exp(-2.0 * np.pi * cutoff_hz / sr)
    y = np.zeros_like(x, dtype=np.float32)
    y[0] = (1.0 - alpha) * x[0]
    for i in range(1, len(x)):
        y[i] = alpha * y[i - 1] + (1.0 - alpha) * x[i]
    return y


def bandpass_noise(
    duration_s: float, low_hz: float, high_hz: float, sr: int = SR, scale: float = 1.0
) -> np.ndarray:
    n = int(duration_s * sr)
    x = RNG.normal(0.0, 1.0, n).astype(np.float32)
    b, a = butter(4, [low_hz / (sr / 2), high_hz / (sr / 2)], btype="band")
    y = lfilter(b, a, x).astype(np.float32)
    return y * scale


def colored_noise(
    duration_s: float, color: str = "white", sr: int = SR, scale: float = 1.0
) -> np.ndarray:
    n = int(duration_s * sr)
    white = RNG.normal(0.0, 1.0, n).astype(np.float32)

    if color == "white":
        y = white
    elif color == "pink":
        # Simple pink-ish approximation using multi-stage smoothing differences.
        y = (
            white
            + 0.7 * one_pole_lowpass(white, 8000, sr)
            + 0.4 * one_pole_lowpass(white, 2000, sr)
            + 0.2 * one_pole_lowpass(white, 300, sr)
        )
    elif color == "brown":
        y = np.cumsum(white).astype(np.float32)
        y /= np.max(np.abs(y)) + 1e-8
    elif color == "hvac":
        rumble = one_pole_lowpass(white, 180, sr)
        hiss = bandpass_noise(duration_s, 3000, 9000, sr, scale=0.2)
        tone = 0.12 * np.sin(2 * np.pi * 60 * np.arange(n) / sr, dtype=np.float32)
        y = rumble + hiss + tone.astype(np.float32)
    else:
        raise ValueError(f"Unknown noise color: {color}")

    y = y / (np.max(np.abs(y)) + 1e-8)
    return (y * scale).astype(np.float32)


# ---------- Shot / nuisance synthesis ----------
def generate_shot_impulse(
    profile: str = "crack", duration_s: float = 0.11, sr: int = SR
) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)

    if profile == "crack":
        env = np.where(t < 0.0015, t / 0.0015, np.exp(-(t - 0.0015) / 0.012))
        body = bandpass_noise(duration_s, 1200, 9000, sr, scale=1.0)
        tail = one_pole_lowpass(RNG.normal(0, 1, len(t)).astype(np.float32), 700, sr) * 0.15
        sig = body * env + tail * np.exp(-t / 0.05)
    elif profile == "thump":
        env = np.where(t < 0.003, t / 0.003, np.exp(-(t - 0.003) / 0.03))
        low = bandpass_noise(duration_s, 80, 1200, sr, scale=1.0)
        sig = low * env
    elif profile == "clipped":
        env = np.where(t < 0.001, t / 0.001, np.exp(-(t - 0.001) / 0.01))
        sig = bandpass_noise(duration_s, 1000, 10000, sr, scale=1.2) * env
        sig = np.tanh(2.5 * sig)
    elif profile == "distant":
        env = np.where(t < 0.004, t / 0.004, np.exp(-(t - 0.004) / 0.04))
        sig = one_pole_lowpass(bandpass_noise(duration_s, 250, 3500, sr, scale=1.0), 2400, sr) * env
    elif profile == "reverb":
        env = np.where(t < 0.002, t / 0.002, np.exp(-(t - 0.002) / 0.025))
        direct = bandpass_noise(duration_s, 800, 7000, sr, scale=1.0) * env
        sig = direct.copy()
        for delay_ms, att in [(18, 0.35), (42, 0.22), (73, 0.15)]:
            delay = int(delay_ms * sr / 1000.0)
            if delay < len(sig):
                sig[delay:] += one_pole_lowpass(direct[:-delay], 2500, sr) * att
    else:
        raise ValueError(f"Unknown profile: {profile}")

    sig = sig / (np.max(np.abs(sig)) + 1e-8)
    return sig.astype(np.float32)


def generate_nuisance(kind: str, duration_s: float, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    if kind == "metal_click":
        env = np.exp(-t / 0.01)
        sig = bandpass_noise(duration_s, 2500, 12000, sr) * env
    elif kind == "door_thump":
        env = np.where(t < 0.01, t / 0.01, np.exp(-(t - 0.01) / 0.08))
        sig = bandpass_noise(duration_s, 40, 600, sr) * env
    elif kind == "mic_bump":
        env = np.where(t < 0.004, t / 0.004, np.exp(-(t - 0.004) / 0.06))
        sig = one_pole_lowpass(RNG.normal(0, 1, len(t)).astype(np.float32), 220, sr) * env
    elif kind == "speech_pop":
        env = np.where(t < 0.015, t / 0.015, np.exp(-(t - 0.015) / 0.04))
        base = 0.8 * np.sin(2 * np.pi * 180 * t) + 0.25 * np.sin(2 * np.pi * 360 * t)
        sig = base.astype(np.float32) * env
    else:
        raise ValueError(f"Unknown nuisance kind: {kind}")

    sig = sig / (np.max(np.abs(sig)) + 1e-8)
    return apply_fade(sig, 1.5, sr)


def add_clip(audio: np.ndarray, clip: np.ndarray, start_s: float, amp: float = 1.0, sr: int = SR):
    start = int(start_s * sr)
    if start >= len(audio):
        return
    end = min(start + len(clip), len(audio))
    audio[start:end] += clip[: end - start] * amp


# ---------- Scenario rendering ----------
def render_scenario(
    out_wav: Path,
    shot_times: list[float],
    shot_profiles: list[str] | str = "crack",
    shot_amplitudes: list[float] | float = 0.8,
    base_noise: tuple[str, float] = ("white", 0.01),
    echo_chains: list[list[tuple[float, float, float]]] | None = None,
    interference: list[tuple[float, float, str]] | None = None,
    nuisance: list[tuple[float, float, str]] | None = None,
    sr: int = SR,
) -> dict:
    total_duration = max(shot_times) + 1.0 if shot_times else 2.0
    audio = colored_noise(total_duration, base_noise[0], sr, base_noise[1])
    events: list[Event] = []

    if isinstance(shot_profiles, str):
        shot_profiles = [shot_profiles] * len(shot_times)
    if isinstance(shot_amplitudes, (int, float)):
        shot_amplitudes = [float(shot_amplitudes)] * len(shot_times)

    for idx, t_s in enumerate(shot_times):
        profile = shot_profiles[idx]
        amp = float(shot_amplitudes[idx])
        clip = generate_shot_impulse(profile, sr=sr)
        add_clip(audio, clip, t_s, amp, sr)
        events.append(Event(t_s, "primary_shot", amp, profile, parent_index=idx))

        if echo_chains and idx < len(echo_chains) and echo_chains[idx]:
            for delay_ms, attenuation, lp_cutoff_hz in echo_chains[idx]:
                echo_time = t_s + delay_ms / 1000.0
                echo = one_pole_lowpass(clip, lp_cutoff_hz, sr)
                echo = apply_fade(echo, fade_ms=1.0, sr=sr)
                add_clip(audio, echo, echo_time, amp * attenuation, sr)
                events.append(
                    Event(
                        echo_time,
                        "echo",
                        amp * attenuation,
                        profile,
                        parent_index=idx,
                        note=f"lp={lp_cutoff_hz}Hz",
                    )
                )

    if interference:
        for t_s, amp, profile in interference:
            clip = generate_shot_impulse(profile, sr=sr)
            add_clip(audio, clip, t_s, amp, sr)
            events.append(Event(t_s, "interference_shot", amp, profile))

    if nuisance:
        for t_s, amp, kind in nuisance:
            clip = generate_nuisance(kind, 0.08 if kind != "door_thump" else 0.18, sr)
            add_clip(audio, clip, t_s, amp, sr)
            events.append(Event(t_s, "nuisance", amp, kind))

    audio = normalize(audio)
    wavfile.write(out_wav, sr, audio)

    truth = {
        "file": out_wav.name,
        "sample_rate": sr,
        "expected_primary_shots": shot_times,
        "events": [asdict(e) for e in sorted(events, key=lambda e: e.time_s)],
        "notes": {
            "primary_shot_count": len(shot_times),
            "contains_echoes": bool(echo_chains),
            "contains_interference": bool(interference),
            "contains_nuisance": bool(nuisance),
            "base_noise": {"color": base_noise[0], "scale": base_noise[1]},
        },
    }
    return truth


def repeating(start: float, gap: float, count: int) -> list[float]:
    return [start + i * gap for i in range(count)]


def jittered(base_times: Iterable[float], jitter_ms: list[float]) -> list[float]:
    return [float(t + j / 1000.0) for t, j in zip(base_times, jitter_ms)]


# ---------- Scenario definitions ----------
def build_scenarios(out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    truths = []

    scenarios = []

    scenarios.append(
        {
            "name": "01_steady_600_varied_profiles.wav",
            "shot_times": repeating(0.5, 0.1, 10),
            "shot_profiles": [
                "crack",
                "thump",
                "crack",
                "clipped",
                "crack",
                "distant",
                "crack",
                "thump",
                "reverb",
                "crack",
            ],
            "shot_amplitudes": [0.82, 0.76, 0.83, 0.8, 0.78, 0.7, 0.84, 0.79, 0.75, 0.81],
            "base_noise": ("pink", 0.012),
        }
    )

    scenarios.append(
        {
            "name": "02_fast_1000_mixed.wav",
            "shot_times": repeating(0.5, 0.06, 15),
            "shot_profiles": ["crack", "clipped", "crack", "crack", "thump"] * 3,
            "shot_amplitudes": [0.78 + 0.05 * np.sin(i) for i in range(15)],
            "base_noise": ("pink", 0.016),
        }
    )

    scenarios.append(
        {
            "name": "03_two_bursts_hvac.wav",
            "shot_times": repeating(0.5, 0.1, 5) + repeating(1.5, 0.1, 5),
            "shot_profiles": ["crack", "crack", "thump", "crack", "distant"] * 2,
            "base_noise": ("hvac", 0.028),
        }
    )

    scenarios.append(
        {
            "name": "04_jittery_cadence_varied.wav",
            "shot_times": [0.5, 0.618, 0.703, 0.846, 0.927, 1.056],
            "shot_profiles": ["crack", "distant", "crack", "clipped", "reverb", "crack"],
            "shot_amplitudes": [0.82, 0.68, 0.77, 0.75, 0.7, 0.79],
            "base_noise": ("pink", 0.014),
        }
    )

    scenarios.append(
        {
            "name": "05_interference_and_clicks.wav",
            "shot_times": repeating(0.5, 0.1, 10),
            "shot_profiles": ["crack"] * 10,
            "base_noise": ("pink", 0.02),
            "interference": [
                (0.548, 0.28, "distant"),
                (0.653, 0.2, "thump"),
                (0.756, 0.24, "reverb"),
            ],
            "nuisance": [(0.44, 0.32, "metal_click"), (1.11, 0.28, "speech_pop")],
        }
    )

    echo_chain = [(24.0, 0.24, 3200.0), (41.0, 0.16, 2100.0), (67.0, 0.10, 1700.0)]
    scenarios.append(
        {
            "name": "06_echo_chamber_filtered.wav",
            "shot_times": repeating(0.5, 0.15, 5),
            "shot_profiles": ["crack", "reverb", "crack", "clipped", "crack"],
            "echo_chains": [echo_chain] * 5,
            "base_noise": ("pink", 0.015),
        }
    )

    scenarios.append(
        {
            "name": "07_accelerating_drift.wav",
            "shot_times": [0.5, 0.652, 0.781, 0.892, 0.979, 1.052, 1.115, 1.168],
            "shot_profiles": [
                "distant",
                "crack",
                "crack",
                "thump",
                "crack",
                "clipped",
                "crack",
                "crack",
            ],
            "shot_amplitudes": [0.62, 0.68, 0.73, 0.75, 0.8, 0.82, 0.78, 0.76],
            "base_noise": ("hvac", 0.022),
        }
    )

    scenarios.append(
        {
            "name": "08_slow_fire_with_bumps.wav",
            "shot_times": [0.5, 2.0, 3.5, 5.0],
            "shot_profiles": ["thump", "crack", "distant", "reverb"],
            "base_noise": ("brown", 0.018),
            "nuisance": [(1.1, 0.35, "mic_bump"), (4.2, 0.22, "door_thump")],
        }
    )

    scenarios.append(
        {
            "name": "09_double_taps_boundary.wav",
            "shot_times": [0.5, 0.54, 1.0, 1.04, 1.5, 1.54, 2.0, 2.043],
            "shot_profiles": [
                "crack",
                "clipped",
                "crack",
                "clipped",
                "crack",
                "clipped",
                "distant",
                "crack",
            ],
            "shot_amplitudes": [0.82, 0.65, 0.8, 0.64, 0.81, 0.66, 0.63, 0.8],
            "base_noise": ("pink", 0.016),
            "nuisance": [(2.3, 0.18, "metal_click")],
        }
    )

    scenarios.append(
        {
            "name": "10_noisy_env_harsh.wav",
            "shot_times": repeating(0.5, 0.1, 10),
            "shot_profiles": [
                "distant",
                "crack",
                "distant",
                "crack",
                "distant",
                "crack",
                "distant",
                "crack",
                "distant",
                "crack",
            ],
            "shot_amplitudes": [0.18, 0.22, 0.16, 0.2, 0.17, 0.21, 0.16, 0.2, 0.18, 0.22],
            "base_noise": ("hvac", 0.08),
            "nuisance": [
                (0.43, 0.34, "speech_pop"),
                (0.87, 0.29, "metal_click"),
                (1.08, 0.31, "mic_bump"),
                (1.36, 0.27, "door_thump"),
            ],
        }
    )

    scenarios.append(
        {
            "name": "11_multi_burst_realistic.wav",
            "shot_times": repeating(0.5, 0.1, 3) + repeating(1.5, 0.1, 3) + repeating(2.5, 0.1, 3),
            "shot_profiles": [
                "crack",
                "crack",
                "thump",
                "distant",
                "crack",
                "reverb",
                "clipped",
                "crack",
                "distant",
            ],
            "base_noise": ("pink", 0.015),
            "nuisance": [(1.25, 0.2, "speech_pop")],
        }
    )

    scenarios.append(
        {
            "name": "12_extreme_1200_edge.wav",
            "shot_times": repeating(0.5, 0.05, 20),
            "shot_profiles": ["crack", "clipped", "crack", "thump"] * 5,
            "shot_amplitudes": [0.75 + 0.05 * np.cos(i / 2) for i in range(20)],
            "base_noise": ("pink", 0.018),
        }
    )

    scenarios.append(
        {
            "name": "13_echo_vs_doubletap_ambiguous.wav",
            "shot_times": [0.5, 0.85, 1.2, 1.24, 1.6],
            "shot_profiles": ["crack", "reverb", "crack", "clipped", "crack"],
            "echo_chains": [
                [(38.0, 0.28, 2800.0)],
                [(34.0, 0.24, 2200.0)],
                [],
                [],
                [(42.0, 0.2, 2000.0)],
            ],
            "base_noise": ("pink", 0.014),
            "nuisance": [(1.02, 0.22, "metal_click")],
        }
    )

    scenarios.append(
        {
            "name": "14_threshold_walkaway.wav",
            "shot_times": repeating(0.5, 0.11, 8),
            "shot_profiles": [
                "distant",
                "distant",
                "crack",
                "distant",
                "crack",
                "distant",
                "crack",
                "distant",
            ],
            "shot_amplitudes": [0.28, 0.24, 0.21, 0.19, 0.17, 0.15, 0.14, 0.13],
            "base_noise": ("hvac", 0.05),
            "nuisance": [(0.95, 0.24, "speech_pop"), (1.44, 0.22, "mic_bump")],
        }
    )

    for config in scenarios:
        out_wav = out_dir / config["name"]
        truth = render_scenario(out_wav=out_wav, **{k: v for k, v in config.items() if k != "name"})
        truths.append(truth)
        print(f"Generated: {out_wav}")

    with open(out_dir / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(truths, f, indent=2)
    print(f"Saved ground truth: {out_dir / 'ground_truth.json'}")

    return truths


if __name__ == "__main__":
    build_scenarios(Path("test_audio_v2"))
