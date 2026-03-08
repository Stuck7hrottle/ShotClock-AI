from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from rof_detector.io.ffmpeg import extract_audio_wav
from rof_detector.audio.detect import detect_shots_audio
from rof_detector.vision.flash_detect import confirm_shots_with_flash
from rof_detector.fusion.fuse import fuse_scores
from rof_detector.metrics.rof import compute_rof
from rof_detector.metrics.bursts import segment_bursts, summarize_bursts
from rof_detector.viz.plots import plot_waveform_with_events
from rof_detector.viz.annotate_video import annotate_video_with_events

app = typer.Typer(add_completion=False, help="Rate-of-fire estimation from video.")


@app.command()
def detect(
    input_path: Path = typer.Argument(..., exists=True, readable=True, help="Input video file."),
    out: Path = typer.Option(Path("results.json"), help="Output JSON report path."),
    csv: Optional[Path] = typer.Option(None, help="Optional CSV export path."),
    plot: Optional[Path] = typer.Option(None, help="Optional waveform plot PNG path."),
    annotate: Optional[Path] = typer.Option(None, help="Optional annotated video output path."),
    roi: Optional[str] = typer.Option(
        None, help='ROI "x,y,w,h" for muzzle region (video confirmation).'
    ),
    roi_interactive: bool = typer.Option(False, help="Interactively select ROI on first frame."),
    no_vision: bool = typer.Option(False, help="Disable vision confirmation even if ROI provided."),
    environment: str = typer.Option(
        "auto", help="auto|indoor|outdoor: adjusts echo handling defaults."
    ),
    sensitivity: float = typer.Option(
        0.48, help="Detection sensitivity (0.1 to 1.0). Higher => more detections."
    ),
    min_separation_ms: int = typer.Option(
        35, help="Minimum time between shots to avoid double counts."
    ),
    echo_window_ms: int = typer.Option(30, help="Cluster/merge window to suppress indoor echoes."),
    burst_gap_ms: int = typer.Option(250, help="Gap threshold for new burst segmentation."),
):
    """Detect shot events and compute ROF."""
    wav_path = extract_audio_wav(input_path)

    audio_events = detect_shots_audio(
        wav_path,
        sensitivity=sensitivity,
        min_separation_ms=min_separation_ms,
        echo_window_ms=echo_window_ms,
        environment=environment,
    )

    # Disable vision confirmation automatically for audio files
    is_audio = input_path.suffix.lower() == ".wav"
    video_events = None
    if not no_vision and not is_audio and (roi is not None or roi_interactive):
        video_events = confirm_shots_with_flash(
            video_path=input_path,
            audio_events=audio_events,
            roi=roi,
            roi_interactive=roi_interactive,
        )

    fused = fuse_scores(audio_events, video_events)

    rof = compute_rof([e["t"] for e in fused])
    bursts = segment_bursts([e["t"] for e in fused], burst_gap_s=burst_gap_ms / 1000.0)
    burst_summary = summarize_bursts([e["t"] for e in fused], bursts)

    report = {
        "input": {"video": str(input_path), "audio_wav": str(wav_path)},
        "params": {
            "environment": environment,
            "sensitivity": sensitivity,
            "min_separation_ms": min_separation_ms,
            "echo_window_ms": echo_window_ms,
            "burst_gap_ms": burst_gap_ms,
            "roi": roi,
            "roi_interactive": roi_interactive,
            "no_vision": no_vision,
        },
        "events": fused,
        "rof": rof,
        "bursts": bursts,
        "burst_summary": burst_summary,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(f"Wrote {out}")

    if csv:
        from rof_detector.viz.exports import export_events_csv

        export_events_csv(csv, fused)
        typer.echo(f"Wrote {csv}")

    if plot:
        plot_waveform_with_events(wav_path, fused, plot)
        typer.echo(f"Wrote {plot}")

    if annotate:
        annotate_video_with_events(input_path, fused, annotate)
        typer.echo(f"Wrote {annotate}")
