"""Detector regression tests on small synthetic recordings.

Each test renders a scenario with generate_test_audio_v2's synthesis helpers
and checks the behaviour a past bug got wrong.
"""

from pathlib import Path

import pytest

from generate_test_audio_v2 import render_scenario, repeating
from rof_detector.audio.detect import (
    detect_shots_audio,
    detect_shots_audio_detailed,
    find_candidates,
)

TOL_S = 0.01


def _render(tmp_path: Path, name: str, **scenario) -> tuple[Path, list[float]]:
    wav = tmp_path / name
    render_scenario(out_wav=wav, **scenario)
    return wav, list(scenario["shot_times"])


def _matched(expected: list[float], detected: list[float]) -> int:
    return sum(1 for t in expected if any(abs(t - d) <= TOL_S for d in detected))


def test_isolated_shots_survive_next_to_a_burst(tmp_path):
    # Used to fail: any isolated shot was dropped whenever the file also
    # contained a burst.
    wav, expected = _render(
        tmp_path,
        "burst_then_singles.wav",
        shot_times=repeating(0.5, 0.1, 6) + [2.0, 3.2, 4.4],
        base_noise=("pink", 0.015),
    )
    detected = [e["t"] for e in detect_shots_audio(wav)]
    assert len(detected) == len(expected)
    assert _matched(expected, detected) == len(expected)


def test_timestamps_are_sample_accurate(tmp_path):
    wav, expected = _render(
        tmp_path, "steady.wav", shot_times=repeating(0.5, 0.1, 8), base_noise=("pink", 0.012)
    )
    detected = [e["t"] for e in detect_shots_audio(wav)]
    assert len(detected) == len(expected)
    for t, d in zip(expected, detected):
        assert abs(t - d) < 0.003


def test_fixed_delay_wall_echo_is_rejected(tmp_path):
    # A 58 ms reflection is longer than the echo merge window, so only the
    # echo-lag / relative-loudness logic can reject it.
    wav, expected = _render(
        tmp_path,
        "wall_echo.wav",
        shot_times=repeating(0.5, 0.12, 8),
        echo_chains=[[(58.0, 0.42, 3000.0)] for _ in range(8)],
        base_noise=("pink", 0.014),
    )
    events, rejected = detect_shots_audio_detailed(wav)
    detected = [e["t"] for e in events]
    assert len(detected) == len(expected)
    assert _matched(expected, detected) == len(expected)
    assert all("reject_reason" in r for r in rejected)


@pytest.mark.parametrize("gap", [0.05, 0.06])
def test_fast_cyclic_rates_are_resolved(tmp_path, gap):
    # 1200 and 1000 RPM.
    wav, expected = _render(
        tmp_path, f"fast_{gap}.wav", shot_times=repeating(0.5, gap, 12), base_noise=("pink", 0.015)
    )
    detected = [e["t"] for e in detect_shots_audio(wav)]
    assert _matched(expected, detected) >= len(expected) - 1
    assert len(detected) <= len(expected)


def test_sensitivity_changes_the_candidate_threshold(tmp_path):
    # Used to be a no-op: the MAD threshold collapsed to ~0 on the
    # half-wave-rectified onset curve.
    wav, _ = _render(
        tmp_path,
        "walkaway.wav",
        shot_times=repeating(0.5, 0.11, 8),
        shot_profiles="distant",
        shot_amplitudes=[0.28, 0.24, 0.21, 0.19, 0.17, 0.15, 0.14, 0.13],
        base_noise=("hvac", 0.05),
    )
    assert len(find_candidates(wav, sensitivity=0.1)) < len(find_candidates(wav, sensitivity=1.0))
