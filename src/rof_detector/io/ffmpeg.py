from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _ensure_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg and try again.")
    return exe


def extract_audio_wav(video_path: Path, sr: int = 48000) -> Path:
    """Extract mono WAV audio from a video using ffmpeg."""
    ffmpeg = _ensure_ffmpeg()
    out = video_path.with_suffix(".rof.wav")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "wav",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[:4000]}")
    return out
