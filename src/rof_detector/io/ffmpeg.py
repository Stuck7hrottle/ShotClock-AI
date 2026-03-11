from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def _ensure_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg and try again.")
    return exe


def _ensure_ffprobe() -> str:
    exe = shutil.which("ffprobe")
    if not exe:
        raise RuntimeError("ffprobe not found on PATH. Install ffmpeg/ffprobe and try again.")
    return exe


def probe_media(video_path: Path, timeout: int = 15) -> dict:
    ffprobe = _ensure_ffprobe()
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr[:2000]}")

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Could not parse ffprobe output: {e}") from e


def validate_media_file(video_path: Path) -> dict:
    meta = probe_media(video_path, timeout=15)

    streams = meta.get("streams", [])
    format_info = meta.get("format", {})

    if not streams:
        raise RuntimeError("No media streams found in uploaded file.")

    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    if not has_video:
        raise RuntimeError("Uploaded file does not contain a video stream.")

    if not has_audio:
        raise RuntimeError("Uploaded video does not contain an audio stream.")

    duration_raw = format_info.get("duration")
    try:
        duration = float(duration_raw) if duration_raw is not None else None
    except (TypeError, ValueError):
        duration = None

    if duration is None or duration <= 0:
        raise RuntimeError("Could not determine media duration.")

    if duration > 600:
        raise RuntimeError("Video too long. Maximum allowed duration is 10 minutes.")

    return meta

def extract_audio_wav(video_path: Path, sr: int = 48000, timeout: int = 60) -> Path:
    """Extract mono WAV audio from a video using ffmpeg."""
    if video_path.suffix.lower() == ".wav":
        return video_path

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
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("ffmpeg timed out while extracting audio.") from e

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[:4000]}")

    return out
