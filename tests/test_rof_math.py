from rof_detector.metrics.rof import compute_rof


def test_compute_rof_basic():
    times = [0.0, 0.1, 0.2, 0.3]  # 10 Hz => 600 RPM
    r = compute_rof(times)
    assert r["n_shots"] == 4
    assert abs(r["mean_rpm"] - 600.0) < 1e-6
