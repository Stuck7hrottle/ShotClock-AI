from rof_detector.metrics.bursts import segment_bursts, summarize_bursts
from rof_detector.metrics.rof import compute_rof


def test_compute_rof_basic():
    times = [0.0, 0.1, 0.2, 0.3]  # 10 Hz => 600 RPM
    r = compute_rof(times)
    assert r["n_shots"] == 4
    assert abs(r["mean_rpm"] - 600.0) < 1e-6
    assert abs(r["cyclic_rpm"] - 600.0) < 1e-6


def test_mean_rpm_is_not_biased_by_uneven_splits():
    # Splits of 0.08 s and 0.12 s: 2 intervals over 0.2 s is 600 RPM. Averaging
    # the per-split RPMs (750 and 500) would wrongly give 625.
    r = compute_rof([0.0, 0.08, 0.2])
    assert abs(r["mean_rpm"] - 600.0) < 1e-6


def test_cyclic_rpm_ignores_pauses_between_bursts():
    times = [0.0, 0.1, 0.2, 2.0, 2.1, 2.2]  # two 600 RPM bursts, 1.8 s apart
    r = compute_rof(times, burst_gap_s=0.25)
    assert abs(r["cyclic_rpm"] - 600.0) < 1e-6
    assert r["mean_rpm"] < 200.0  # whole-span rate includes the pause


def test_burst_mean_rpm_uses_span():
    times = [0.0, 0.08, 0.2]
    summary = summarize_bursts(times, segment_bursts(times, burst_gap_s=0.25))
    assert abs(summary[0]["mean_rpm"] - 600.0) < 1e-6
