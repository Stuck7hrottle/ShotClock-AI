import json
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.io import wavfile

from rof_detector.audio.detect import detect_shots_audio
from rof_detector.fusion.fuse import fuse_scores
from rof_detector.io.ffmpeg import extract_audio_wav, validate_media_file
from rof_detector.metrics.bursts import segment_bursts, summarize_bursts
from rof_detector.metrics.rof import compute_rof
from rof_detector.vision.flash_detect import confirm_shots_with_flash

import re

st.set_page_config(page_title="ShotClock AI", layout="wide")
st.title("🎯 ShotClock AI: Rate of Fire Analyzer")
st.write(
    "Upload a video to detect shots, inspect bursts, narrow the analysis window, "
    "and manually correct timestamps when needed."
)

MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi"}

def sanitize_filename(name: str) -> str:
    base = Path(name).name
    return re.sub(r"[^A-Za-z0-9._-]", "_", base)

def validate_uploaded_file(uploaded_file) -> tuple[bool, str | None]:
    if uploaded_file is None:
        return False, "No file uploaded."

    filename = sanitize_filename(uploaded_file.name or "upload")
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type: {suffix or 'unknown'}"

    if getattr(uploaded_file, "size", None) is not None:
        if uploaded_file.size <= 0:
            return False, "Uploaded file is empty."
        if uploaded_file.size > MAX_UPLOAD_BYTES:
            return False, "File too large. Maximum allowed size is 200 MB."

    return True, None

def parse_roi(roi_str: str) -> tuple[int, int, int, int] | None:
    roi_str = roi_str.strip()
    if not roi_str:
        return None
    try:
        parts = [int(p.strip()) for p in roi_str.split(",")]
        if len(parts) != 4:
            return None
        x, y, w, h = parts
        if w <= 0 or h <= 0:
            return None
        return x, y, w, h
    except Exception:
        return None


def load_wav_display_data(wav_path: Path) -> tuple[int, np.ndarray, float]:
    sr, x = wavfile.read(str(wav_path))
    if x.ndim > 1:
        x = x[:, 0]
    if x.dtype.kind in ("i", "u"):
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float32)
    duration_s = len(x) / float(sr)
    return sr, x, duration_s


def downsample_for_plot(
    t: np.ndarray, x: np.ndarray, max_points: int = 25000
) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return t, x
    idx = np.linspace(0, len(x) - 1, num=max_points, dtype=int)
    return t[idx], x[idx]


def cadence_stats_from_splits(splits: np.ndarray) -> dict[str, float | None]:
    splits = np.asarray(splits, dtype=float)
    splits = splits[np.isfinite(splits)]
    splits = splits[splits > 0]
    if splits.size < 2:
        return {"mean_split_s": None, "std_split_s": None, "cv": None}
    mean = float(np.mean(splits))
    std = float(np.std(splits, ddof=1))
    cv = float(std / mean) if mean > 0 else None
    return {"mean_split_s": mean, "std_split_s": std, "cv": cv}


def events_to_table(events: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sorted_events = sorted(events, key=lambda e: float(e["t"]))
    for i, e in enumerate(sorted_events):
        t = float(e["t"])
        split = t - float(sorted_events[i - 1]["t"]) if i > 0 else None
        inst_rpm = (60.0 / split) if split and split > 0 else None
        rows.append(
            {
                "Keep": True,
                "Shot #": i + 1,
                "Timestamp (s)": round(t, 4),
                "Split (s)": round(split, 4) if split is not None else None,
                "Inst. RPM": int(round(inst_rpm)) if inst_rpm else None,
                "Confidence": round(float(e.get("confidence", 0.0)), 3),
                "Audio Score": round(float(e.get("audio_score", 0.0)), 3),
                "Video Score": (
                    round(float(e.get("video_score", 0.0)), 3)
                    if e.get("video_score") is not None
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def editable_events_from_dataframe(df: pd.DataFrame) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if df.empty:
        return events
    for _, row in df.iterrows():
        keep_val = row.get("Keep", True)
        if pd.isna(keep_val) or not bool(keep_val):
            continue
        t_val = row.get("Timestamp (s)")
        if pd.isna(t_val):
            continue
        event = {
            "t": float(t_val),
            "confidence": None if pd.isna(row.get("Confidence")) else float(row["Confidence"]),
            "audio_score": None if pd.isna(row.get("Audio Score")) else float(row["Audio Score"]),
            "video_score": None if pd.isna(row.get("Video Score")) else float(row["Video Score"]),
            "source": "manual" if row.get("Confidence") is None else "detected",
        }
        events.append(event)
    events.sort(key=lambda e: float(e["t"]))
    return events


def events_in_range(
    events: list[dict[str, Any]], start_s: float, end_s: float
) -> list[dict[str, Any]]:
    return [dict(e) for e in events if start_s <= float(e["t"]) <= end_s]


def build_burst_options(
    burst_summary: list[dict[str, Any]], duration_s: float
) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = [
        {
            "label": f"Full recording: 0.00s – {duration_s:.2f}s",
            "start": 0.0,
            "end": duration_s,
            "kind": "full",
        }
    ]
    for i, bs in enumerate(burst_summary):
        start_t = float(bs.get("start_t", 0.0))
        end_t = float(bs.get("end_t", start_t))
        options.append(
            {
                "label": f"Burst {i + 1}: {start_t:.2f}s – {end_t:.2f}s ({int(bs.get('n_shots', 0))} shots)",
                "start": start_t,
                "end": end_t,
                "kind": "burst",
                "index": i,
            }
        )
    return options


def summarize_from_events(
    events: list[dict[str, Any]], burst_gap_ms: int
) -> tuple[list[float], dict[str, Any], list[dict[str, int]], list[dict[str, Any]], dict[str, Any]]:
    times = [float(e["t"]) for e in sorted(events, key=lambda e: float(e["t"]))]
    rof = compute_rof(times)
    bursts = segment_bursts(times, burst_gap_s=float(burst_gap_ms) / 1000.0) if times else []
    burst_summary = summarize_bursts(times, bursts) if times else []
    splits = (
        np.diff(np.asarray(times, dtype=float)) if len(times) >= 2 else np.array([], dtype=float)
    )
    cadence = cadence_stats_from_splits(splits)
    return times, rof, bursts, burst_summary, cadence


def plot_waveform_window(
    wav_path: Path,
    events: list[dict[str, Any]],
    bursts: list[dict[str, int]],
    center_s: float,
    width_s: float,
    highlight_start: float | None = None,
    highlight_end: float | None = None,
) -> None:
    sr, x, dur_s = load_wav_display_data(wav_path)
    width_s = float(max(1.0, width_s))
    center_s = float(np.clip(center_s, 0.0, dur_s))
    start_s = max(0.0, center_s - width_s / 2.0)
    end_s = min(dur_s, center_s + width_s / 2.0)
    s0 = int(start_s * sr)
    s1 = int(end_s * sr)
    seg = x[s0:s1]
    t = (np.arange(len(seg), dtype=np.float32) + s0) / float(sr)
    t, seg = downsample_for_plot(t, seg)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, seg, linewidth=0.8)

    if highlight_start is not None and highlight_end is not None:
        hs = max(start_s, float(highlight_start))
        he = min(end_s, float(highlight_end))
        if hs < he:
            ax.axvspan(hs, he, alpha=0.20)

    if events:
        event_times = [float(e["t"]) for e in events if start_s <= float(e["t"]) <= end_s]
        for et in event_times:
            ax.axvline(et, linewidth=1)

    if bursts and events:
        times = [float(e["t"]) for e in sorted(events, key=lambda e: float(e["t"]))]
        for b in bursts:
            bi0 = int(b["start_index"])
            bi1 = int(b["end_index"])
            if 0 <= bi0 < len(times) and 0 <= bi1 < len(times):
                bs = times[bi0]
                be = times[bi1]
                if be >= start_s and bs <= end_s:
                    ax.axvspan(max(bs, start_s), min(be, end_s), alpha=0.10)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Waveform ({start_s:.2f}s to {end_s:.2f}s)")
    st.pyplot(fig, width="stretch")
    plt.close(fig)


with st.sidebar:
    st.header("Detection Settings")
    environment = st.selectbox("Environment", ["auto", "indoor", "outdoor"], index=0)

    sensitivity = st.slider(
        "Sensitivity",
        min_value=0.1,
        max_value=1.0,
        value=0.48,
        step=0.01,
    )

    min_sep_ms = st.number_input(
        "Min separation (ms)",
        min_value=10,
        max_value=200,
        value=35,
        step=5,
    )

    echo_ms = st.number_input(
        "Echo merge window (ms)",
        min_value=0,
        max_value=200,
        value=30,
        step=5,
    )

    st.divider()
    st.header("Burst Settings")
    burst_gap_ms = st.number_input(
        "Burst gap (ms)",
        min_value=50,
        max_value=2000,
        value=250,
        step=50,
    )
    st.caption("If the gap between shots exceeds this, a new burst starts.")

    st.divider()
    st.header("Vision Confirmation (optional)")
    use_vision = st.checkbox("Enable muzzle-flash confirmation", value=False)
    roi_str = st.text_input('ROI "x,y,w,h" (recommended)', value="")
    roi_interactive = False
    st.caption(
        "Interactive OpenCV ROI picking is disabled in the web app because it "
        "requires a desktop GUI. Enter ROI manually as x,y,w,h instead."
    )

    st.divider()
    st.header("Waveform View")

    auto_zoom_waveform = st.checkbox(
        "Auto zoom waveform to selected window",
        value=True,
        help="Automatically zoom the waveform based on the selected burst/time window.",
    )

    wave_window = st.number_input(
        "Max/manual window width (s)",
        min_value=1.0,
        value=10.0,
        step=1.0,
        help="Used as the maximum width when auto zoom is enabled, or the fixed width when auto zoom is disabled.",
    )


uploaded_file = st.file_uploader("Upload Video (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])

if uploaded_file:
    ok, err = validate_uploaded_file(uploaded_file)
    if not ok:
        st.error(err)
        st.stop()

    with tempfile.TemporaryDirectory() as td:
        safe_name = sanitize_filename(uploaded_file.name or "upload.mp4")
        suffix = Path(safe_name).suffix.lower() or ".mp4"
        tmp_video = Path(td) / f"upload{suffix}"
        tmp_video.write_bytes(uploaded_file.getbuffer())

        try:
            validate_media_file(tmp_video)
        except Exception as e:
            st.error(f"Invalid or unsupported media file. Details: {e}")
            st.stop()

        st.video(str(tmp_video))

        with st.spinner("Extracting audio and detecting shots…"):
            try:
                wav_path = extract_audio_wav(tmp_video)
            except Exception as e:
                st.error(
                    "Audio extraction failed. This project requires **ffmpeg** installed and on PATH.\n\n"
                    "On Ubuntu/Debian:\n"
                    "```bash\nsudo apt update && sudo apt install ffmpeg\n```\n\n"
                    f"Details: {e}"
                )
                st.stop()

            audio_events = detect_shots_audio(
                wav_path,
                sensitivity=float(sensitivity),
                min_separation_ms=int(min_sep_ms),
                echo_window_ms=int(echo_ms),
                environment=environment,
            )

            video_events = None
            roi_tuple = parse_roi(roi_str)

            if use_vision:
                if roi_tuple is None:
                    st.info(
                        "Vision confirmation is enabled, but ROI is missing or invalid. Use format: x,y,w,h"
                    )
                else:
                    try:
                        video_events = confirm_shots_with_flash(
                            video_path=tmp_video,
                            audio_events=audio_events,
                            roi=",".join(str(v) for v in roi_tuple),
                            roi_interactive=False,
                        )
                    except Exception as e:
                        st.warning(f"Vision confirmation unavailable: {e}")

            fused_events = fuse_scores(audio_events, video_events)

        if not fused_events:
            st.warning("No shots detected. Increase sensitivity or check audio quality.")
            st.stop()

        _, _, duration_s = load_wav_display_data(wav_path)
        full_times, full_rof, full_bursts, full_burst_summary, full_cadence = summarize_from_events(
            fused_events,
            int(burst_gap_ms),
        )

        st.subheader("Analysis Range")

        analysis_mode = st.radio(
            "Analysis target",
            ["Full recording", "Detected burst", "Manual window"],
            horizontal=True,
            help="Choose a burst first when possible, then fine-tune with the time window slider.",
        )

        pad_s = st.slider(
            "Burst padding (s)",
            min_value=0.0,
            max_value=0.5,
            value=0.10,
            step=0.01,
            help="Expands detected burst ranges slightly so the first/last shot is not clipped.",
        )

        default_start = 0.0
        default_end = float(duration_s)

        if analysis_mode == "Detected burst":
            if full_burst_summary:
                burst_options = build_burst_options(full_burst_summary, duration_s)
                burst_only_options = [opt for opt in burst_options if opt["kind"] == "burst"]

                burst_labels = [opt["label"] for opt in burst_only_options]

                selected_burst_label = st.selectbox(
                    "Choose burst",
                    options=burst_labels,
                    index=0,
                    help="Select the burst you want to analyze, then fine-tune below if needed.",
                )

                selected_burst = burst_only_options[burst_labels.index(selected_burst_label)]

                default_start = max(0.0, float(selected_burst["start"]) - pad_s)
                default_end = min(float(duration_s), float(selected_burst["end"]) + pad_s)
            else:
                st.info("No bursts detected. Falling back to full recording.")
                default_start = 0.0
                default_end = float(duration_s)

        elif analysis_mode == "Manual window":
            default_start = 0.0
            default_end = min(float(duration_s), 10.0)

        selected_start_s, selected_end_s = st.slider(
            "Refine selected time window (s)",
            min_value=0.0,
            max_value=float(duration_s),
            value=(float(default_start), float(default_end)),
            step=0.01,
            help="Use this slider to fine-tune the final analysis window.",
        )

        if selected_end_s <= selected_start_s:
            st.error("End time must be greater than start time.")
            st.stop()

        selected_events = events_in_range(fused_events, selected_start_s, selected_end_s)
        selected_times, selected_rof, selected_bursts, selected_burst_summary, selected_cadence = (
            summarize_from_events(
                selected_events,
                int(burst_gap_ms),
            )
        )

        st.subheader("Waveform Debug View")

        selected_width_s = float(selected_end_s - selected_start_s)

        # Keep midpoint centering as the default behavior.
        wave_center_default = (selected_start_s + selected_end_s) / 2.0

        wave_center = st.number_input(
            "Waveform center (s)",
            min_value=0.0,
            max_value=float(duration_s),
            value=float(wave_center_default),
            step=0.1,
        )

        if auto_zoom_waveform:
            computed_wave_width = max(
                2.0,  # minimum useful zoom
                min(float(wave_window), selected_width_s * 3.0),
            )
        else:
            computed_wave_width = float(wave_window)

        if selected_events:
            shot_choice = st.selectbox(
                "Jump waveform to shot #",
                options=[0] + list(range(1, len(selected_events) + 1)),
                index=0,
                help="Choose a shot number to center the waveform around that shot. 0 = manual center.",
            )
            if shot_choice != 0:
                wave_center = float(selected_events[shot_choice - 1]["t"])

        plot_waveform_window(
            wav_path=wav_path,
            events=fused_events,
            bursts=full_bursts,
            center_s=wave_center,
            width_s=float(computed_wave_width),
            highlight_start=float(selected_start_s),
            highlight_end=float(selected_end_s),
        )

        st.caption(
            "The highlighted region is the active analysis window. Event markers show all detected shots. "
            "Burst shading is based on the full-file pass. When auto zoom is enabled, the waveform zooms "
            "to the selected burst/window automatically."
        )

        if not selected_events:
            st.warning("No detected shots fall inside the selected window.")
            st.stop()

        st.subheader("Manual Shot Editing")
        st.write(
            "Use the editor below to correct individual detections after choosing the burst or time window. "
            "Uncheck false positives, adjust timestamps, or add missed shots. Metrics update from the edited list."
        )

        editable_df = events_to_table(selected_events)
        edited_df = st.data_editor(
            editable_df,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "Keep": st.column_config.CheckboxColumn(
                    "Keep",
                    help="Uncheck to exclude a detected shot from final metrics.",
                    default=True,
                ),
                "Timestamp (s)": st.column_config.NumberColumn(
                    "Timestamp (s)",
                    help="Edit or add shot timestamps in seconds.",
                    min_value=0.0,
                    step=0.001,
                    format="%.4f",
                ),
                "Confidence": st.column_config.NumberColumn("Confidence", format="%.3f"),
                "Audio Score": st.column_config.NumberColumn("Audio Score", format="%.3f"),
                "Video Score": st.column_config.NumberColumn("Video Score", format="%.3f"),
            },
            disabled=["Shot #", "Split (s)", "Inst. RPM"],
            key="event_editor",
        )

        final_events = editable_events_from_dataframe(edited_df)
        if not final_events:
            st.warning("All shots were removed from the final analysis.")
            st.stop()

        final_times, final_rof, final_bursts, final_burst_summary, final_cadence = (
            summarize_from_events(
                final_events,
                int(burst_gap_ms),
            )
        )

        final_rows = []
        for i, e in enumerate(final_events):
            t = float(e["t"])
            split = t - float(final_events[i - 1]["t"]) if i > 0 else None
            inst_rpm = (60.0 / split) if split and split > 0 else None
            final_rows.append(
                {
                    "Shot #": i + 1,
                    "Timestamp (s)": round(t, 4),
                    "Split (s)": round(split, 4) if split is not None else None,
                    "Inst. RPM": int(round(inst_rpm)) if inst_rpm else None,
                    "Confidence": None
                    if e.get("confidence") is None
                    else round(float(e["confidence"]), 3),
                    "Audio Score": None
                    if e.get("audio_score") is None
                    else round(float(e["audio_score"]), 3),
                    "Video Score": None
                    if e.get("video_score") is None
                    else round(float(e["video_score"]), 3),
                    "Source": e.get("source", "detected"),
                }
            )
        final_df = pd.DataFrame(final_rows)

        burst_rows = []
        for bi, b in enumerate(final_bursts):
            s = int(b["start_index"])
            e = int(b["end_index"])
            seg_times = np.array(final_times[s : e + 1], dtype=float)
            seg_splits = np.diff(seg_times) if seg_times.size >= 2 else np.array([], dtype=float)
            c = cadence_stats_from_splits(seg_splits)
            bs = final_burst_summary[bi] if bi < len(final_burst_summary) else {}
            burst_rows.append(
                {
                    "Burst #": bi + 1,
                    "Shots": int(bs.get("n_shots", seg_times.size)),
                    "Start (s)": round(float(bs.get("start_t", seg_times[0])), 3)
                    if seg_times.size
                    else None,
                    "End (s)": round(float(bs.get("end_t", seg_times[-1])), 3)
                    if seg_times.size
                    else None,
                    "Duration (s)": round(
                        float(
                            bs.get(
                                "duration_s",
                                (seg_times[-1] - seg_times[0]) if seg_times.size else 0.0,
                            )
                        ),
                        3,
                    )
                    if seg_times.size
                    else None,
                    "Mean RPM": int(round(float(bs["mean_rpm"])))
                    if bs.get("mean_rpm") is not None
                    else None,
                    "Median Split (s)": round(float(np.median(seg_splits)), 3)
                    if seg_splits.size
                    else None,
                    "Cadence CV": round(float(c["cv"]), 3) if c["cv"] is not None else None,
                }
            )
        burst_df = pd.DataFrame(burst_rows)

        csv_data = final_df.to_csv(index=False).encode("utf-8")
        report = {
            "input": {"video": uploaded_file.name},
            "params": {
                "environment": environment,
                "sensitivity": float(sensitivity),
                "min_separation_ms": int(min_sep_ms),
                "echo_window_ms": int(echo_ms),
                "burst_gap_ms": int(burst_gap_ms),
                "use_vision": bool(use_vision),
                "roi": roi_str.strip() if roi_str.strip() else None,
                "roi_interactive": bool(roi_interactive),
                "selected_start_s": float(selected_start_s),
                "selected_end_s": float(selected_end_s),
                "wave_center_s": float(wave_center),
                "wave_window_s": float(wave_window),
            },
            "initial_events": fused_events,
            "selected_events_before_edit": selected_events,
            "final_events": final_events,
            "rof": final_rof,
            "bursts": final_bursts,
            "burst_summary": final_burst_summary,
            "cadence": final_cadence,
        }
        json_data = json.dumps(report, indent=2).encode("utf-8")

        mean_rpm = final_rof.get("mean_rpm")
        max_rpm = final_rof.get("max_rpm")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Shots", f"{len(final_events)}")
        c2.metric("Avg RPM", f"{int(round(mean_rpm))}" if mean_rpm is not None else "—")
        c3.metric("Max RPM", f"{int(round(max_rpm))}" if max_rpm is not None else "—")
        c4.metric("Bursts", f"{len(final_bursts)}")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Final Shot Data")
            st.dataframe(final_df, use_container_width=True)

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_data,
                    file_name="shot_data.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with dl2:
                st.download_button(
                    label="⬇️ Download JSON",
                    data=json_data,
                    file_name="analysis.json",
                    mime="application/json",
                    use_container_width=True,
                )

        with col2:
            st.subheader("Split Consistency")
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_df = final_df.dropna(subset=["Split (s)"])
            if not plot_df.empty:
                ax.plot(plot_df["Shot #"], plot_df["Split (s)"], marker="o")
            ax.set_ylabel("Split Time (seconds)")
            ax.set_xlabel("Shot Number")
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Burst Summary")
        if final_cadence["cv"] is not None:
            st.info(
                f"Cadence stability (CV of splits): **{final_cadence['cv']:.3f}** "
                f"(mean split {final_cadence['mean_split_s']:.3f}s, "
                f"std {final_cadence['std_split_s']:.3f}s). Lower CV = steadier cadence."
            )
        else:
            st.info("Cadence stability requires at least 3 detected shots.")

        st.dataframe(burst_df, use_container_width=True)

        with st.expander("Full-file baseline"):
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Shots", f"{len(fused_events)}")
            b2.metric(
                "Avg RPM",
                f"{int(round(full_rof.get('mean_rpm')))}"
                if full_rof.get("mean_rpm") is not None
                else "—",
            )
            b3.metric(
                "Max RPM",
                f"{int(round(full_rof.get('max_rpm')))}"
                if full_rof.get("max_rpm") is not None
                else "—",
            )
            b4.metric("Bursts", f"{len(full_bursts)}")

        with st.expander("Details (JSON)"):
            st.json(report)
