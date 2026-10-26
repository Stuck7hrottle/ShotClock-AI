"""Export hand-corrected shot lists as training data.

The output uses the same layout as the synthetic set from
generate_test_audio_v2.py (a directory holding WAV files plus a
ground_truth.json list), so train_shot_classifier.py can learn from real
recordings with `--extra <dir>`.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path


def ground_truth_entry(
    wav_name: str,
    sample_rate: int,
    shot_times: list[float],
    labeled_range: tuple[float, float] | None = None,
    source_name: str | None = None,
) -> dict:
    """One ground_truth.json item. `labeled_range` marks the part of the
    recording a person actually reviewed; training ignores candidates outside
    it, so unreviewed stretches don't become false labels."""
    entry: dict = {
        "file": wav_name,
        "sample_rate": int(sample_rate),
        "expected_primary_shots": sorted(float(t) for t in shot_times),
        "events": [],
        "notes": {"source": "manual_review", "original_file": source_name},
    }
    if labeled_range is not None:
        entry["labeled_range"] = [float(labeled_range[0]), float(labeled_range[1])]
    return entry


def build_training_zip(
    wav_path: Path,
    clip_name: str,
    shot_times: list[float],
    sample_rate: int,
    labeled_range: tuple[float, float] | None = None,
    source_name: str | None = None,
) -> bytes:
    """Zip a `<clip_name>/` folder with the audio and its ground_truth.json."""
    wav_name = f"{clip_name}.wav"
    entry = ground_truth_entry(wav_name, sample_rate, shot_times, labeled_range, source_name)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(wav_path, f"{clip_name}/{wav_name}")
        zf.writestr(f"{clip_name}/ground_truth.json", json.dumps([entry], indent=2))
    return buf.getvalue()
