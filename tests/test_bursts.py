from rof_detector.metrics.bursts import segment_bursts, summarize_bursts


def test_burst_segmentation():
    times = [0.0, 0.1, 0.2, 1.0, 1.1]
    bursts = segment_bursts(times, burst_gap_s=0.4)
    assert bursts == [{"start_index": 0, "end_index": 2}, {"start_index": 3, "end_index": 4}]
    summary = summarize_bursts(times, bursts)
    assert summary[0]["n_shots"] == 3
    assert summary[1]["n_shots"] == 2
