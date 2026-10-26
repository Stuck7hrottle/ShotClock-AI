import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.io import wavfile

from rof_detector.audio.detect import detect_shots_audio_detailed
from rof_detector.fusion.fuse import fuse_scores
from rof_detector.io.ffmpeg import extract_audio_wav, validate_media_file
from rof_detector.io.labels import build_training_zip
from rof_detector.metrics.bursts import segment_bursts, summarize_bursts
from rof_detector.metrics.rof import compute_rof
from rof_detector.vision.flash_detect import confirm_shots_with_flash

st.set_page_config(page_title="ShotClock AI", layout="wide")
st.title("🎯 ShotClock AI: Rate of Fire Analyzer")
st.write(
    "Upload a video or WAV to detect shots, inspect bursts, narrow the analysis window, "
    "and manually correct timestamps when needed."
)

MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}
AUDIO_EXTENSIONS = {".wav"}
ALLOWED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS
WORK_ROOT = Path(tempfile.gettempdir()) / "shotclock_uploads"
MATCH_EPS_S = 0.002  # two events closer than this are the same event


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


def stage_upload(uploaded_file) -> tuple[str, Path]:
    """Write the upload to a per-file work dir once and reuse it on reruns.

    Streamlit re-executes the whole script on every widget change; without
    this every slider move re-copied the upload and re-ran ffmpeg. Older
    uploads' dirs are removed so the container's /tmp doesn't fill up.
    """
    data = uploaded_file.getbuffer()
    file_key = getattr(uploaded_file, "file_id", None) or hashlib.sha1(data).hexdigest()
    file_key = sanitize_filename(str(file_key))
    suffix = Path(sanitize_filename(uploaded_file.name or "upload.mp4")).suffix.lower() or ".mp4"

    work_dir = WORK_ROOT / file_key
    media_path = work_dir / f"upload{suffix}"
    if not media_path.exists():
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        for old in WORK_ROOT.iterdir():
            if old.is_dir() and old.name != file_key:
                shutil.rmtree(old, ignore_errors=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        media_path.write_bytes(data)
    return file_key, media_path


@st.cache_data(show_spinner=False, max_entries=4)
def validate_and_extract(media_path_str: str) -> str:
    media_path = Path(media_path_str)
    is_audio = media_path.suffix.lower() in AUDIO_EXTENSIONS
    validate_media_file(media_path, require_video=not is_audio)
    return str(extract_audio_wav(media_path))


@st.cache_data(show_spinner=False, max_entries=16)
def run_detection(
    wav_path_str: str,
    media_path_str: str,
    sensitivity: float,
    min_sep_ms: int,
    echo_ms: int,
    environment: str,
    roi: str | None,
) -> tuple[list[dict], list[dict], str | None]:
    """Detect shots (and optionally confirm with muzzle flash). Cached per
    file + settings, so only changing a detection setting re-runs it."""
    audio_events, rejected = detect_shots_audio_detailed(
        Path(wav_path_str),
        sensitivity=sensitivity,
        min_separation_ms=min_sep_ms,
        echo_window_ms=echo_ms,
        environment=environment,
    )
    video_events = None
    vision_warning = None
    if roi is not None:
        try:
            video_events = confirm_shots_with_flash(
                video_path=Path(media_path_str),
                audio_events=audio_events,
                roi=roi,
                roi_interactive=False,
            )
        except Exception as e:
            vision_warning = f"Vision confirmation unavailable: {e}"

    fused = fuse_scores(audio_events, video_events)
    for e in fused:
        e["source"] = "detected"
    rejected_rows = [
        {
            "t": float(r["t"]),
            "audio_score": float(r["audio_score"]),
            "reason": r.get("reject_reason", "rejected"),
        }
        for r in rejected
    ]
    return fused, rejected_rows, vision_warning


@st.cache_data(show_spinner=False, max_entries=4)
def load_wav_display_data(wav_path_str: str) -> tuple[int, np.ndarray, float]:
    sr, x = wavfile.read(wav_path_str)
    if x.ndim > 1:
        x = x[:, 0]
    if x.dtype.kind in ("i", "u"):
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float32)
    duration_s = len(x) / float(sr)
    return int(sr), x, duration_s


@st.cache_data(show_spinner=False, max_entries=4)
def first_video_frame(media_path_str: str) -> np.ndarray | None:
    try:
        import cv2
    except Exception:
        return None
    cap = cv2.VideoCapture(media_path_str)
    ok, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None


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


def _opt_round(v: Any, nd: int) -> float | None:
    return None if v is None or pd.isna(v) else round(float(v), nd)


def events_to_table(events: list[dict[str, Any]], include_keep: bool = True) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sorted_events = sorted(events, key=lambda e: float(e["t"]))
    for i, e in enumerate(sorted_events):
        t = float(e["t"])
        split = t - float(sorted_events[i - 1]["t"]) if i > 0 else None
        inst_rpm = (60.0 / split) if split and split > 0 else None
        row: dict[str, Any] = {"Keep": True} if include_keep else {}
        row.update(
            {
                "Shot #": i + 1,
                # Full precision: the editor formats it for display, and
                # rounding here would shift every shot when edits are applied.
                "Timestamp (s)": t,
                "Split (s)": round(split, 4) if split is not None else None,
                "Inst. RPM": round(inst_rpm) if inst_rpm else None,
                "Confidence": _opt_round(e.get("confidence"), 3),
                "Audio Score": _opt_round(e.get("audio_score"), 3),
                "Video Score": _opt_round(e.get("video_score"), 3),
                "Source": e.get("source", "detected"),
            }
        )
        rows.append(row)
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
        source = row.get("Source")
        events.append(
            {
                "t": float(t_val),
                "confidence": _opt_round(row.get("Confidence"), 6),
                "audio_score": _opt_round(row.get("Audio Score"), 6),
                "video_score": _opt_round(row.get("Video Score"), 6),
                # Rows added in the editor have no Source yet.
                "source": "manual" if source is None or pd.isna(source) else str(source),
            }
        )
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
                "label": (
                    f"Burst {i + 1}: {start_t:.2f}s – {end_t:.2f}s "
                    f"({int(bs.get('n_shots', 0))} shots)"
                ),
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
    burst_gap_s = float(burst_gap_ms) / 1000.0
    rof = compute_rof(times, burst_gap_s=burst_gap_s)
    bursts = segment_bursts(times, burst_gap_s=burst_gap_s) if times else []
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
    rejected: list[dict[str, Any]] | None = None,
) -> None:
    sr, x, dur_s = load_wav_display_data(str(wav_path))
    width_s = float(max(0.2, width_s))
    center_s = float(np.clip(center_s, 0.0, dur_s))
    start_s = max(0.0, center_s - width_s / 2.0)
    end_s = min(dur_s, center_s + width_s / 2.0)
    s0 = int(start_s * sr)
    s1 = int(end_s * sr)
    seg = x[s0:s1]
    t = (np.arange(len(seg), dtype=np.float32) + s0) / float(sr)
    t, seg = downsample_for_plot(t, seg)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, seg, linewidth=0.8, color="#4c78a8")

    if highlight_start is not None and highlight_end is not None:
        hs = max(start_s, float(highlight_start))
        he = min(end_s, float(highlight_end))
        if hs < he:
            ax.axvspan(hs, he, alpha=0.12, color="#f2cf5b")

    sorted_events = sorted(events, key=lambda e: float(e["t"]))
    times = [float(e["t"]) for e in sorted_events]
    if bursts:
        for b in bursts:
            bi0 = int(b["start_index"])
            bi1 = int(b["end_index"])
            if 0 <= bi0 < len(times) and 0 <= bi1 < len(times):
                bs = times[bi0]
                be = times[bi1]
                if be >= start_s and bs <= end_s:
                    ax.axvspan(max(bs, start_s), min(be, end_s), alpha=0.10, color="#54a24b")

    for r in rejected or []:
        rt = float(r["t"])
        if start_s <= rt <= end_s:
            ax.axvline(rt, linewidth=1, linestyle=":", color="#9d9d9d")

    for e in sorted_events:
        et = float(e["t"])
        if start_s <= et <= end_s:
            manual = e.get("source") in ("manual", "recovered")
            ax.axvline(et, linewidth=1.2, color="#e45756" if manual else "#d62728", alpha=0.9)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(
        f"Waveform ({start_s:.2f}s to {end_s:.2f}s) — solid: shots, dotted: rejected candidates"
    )
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def audio_snippet(wav_path: Path, center_s: float, half_width_s: float = 0.3):
    sr, x, _ = load_wav_display_data(str(wav_path))
    s0 = max(0, int((center_s - half_width_s) * sr))
    s1 = min(len(x), int((center_s + half_width_s) * sr))
    return x[s0:s1], sr


def bump_editor_version(state_key: str) -> None:
    st.session_state[f"{state_key}::ver"] = st.session_state.get(f"{state_key}::ver", 0) + 1


with st.sidebar:
    st.header("Detection Settings")
    environment = st.selectbox("Environment", ["auto", "indoor", "outdoor"], index=0)

    sensitivity = st.slider(
        "Sensitivity",
        min_value=0.1,
        max_value=1.0,
        value=0.48,
        step=0.01,
        help="Higher finds quieter shots but lets more echoes and noise through.",
    )

    min_sep_ms = st.number_input(
        "Min separation (ms)",
        min_value=10,
        max_value=200,
        value=35,
        step=5,
        help="Candidates closer together than this are treated as one event.",
    )

    echo_ms = st.number_input(
        "Echo merge window (ms)",
        min_value=0,
        max_value=200,
        value=30,
        step=5,
        help=(
            "Events this close are merged. Longer reflections are handled automatically "
            "by echo-delay detection, so this rarely needs raising."
        ),
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
    roi_str = st.text_input('ROI "x,y,w,h"', value="")
    roi_interactive = False
    st.caption("Enable the checkbox to see the first frame with pixel coordinates.")

    st.divider()
    st.header("Waveform View")

    auto_zoom_waveform = st.checkbox(
        "Auto zoom waveform to selected window",
        value=True,
        help="Automatically zoom the waveform based on the selected burst/time window.",
    )

    wave_window = st.number_input(
        "Max/manual window width (s)",
        min_value=0.2,
        value=10.0,
        step=0.5,
        help=(
            "Used as the maximum width when auto zoom is enabled, "
            "or the fixed width when auto zoom is disabled."
        ),
    )


uploaded_file = st.file_uploader(
    "Upload video (MP4, MOV, AVI) or audio (WAV)", type=["mp4", "mov", "avi", "wav"]
)

if uploaded_file:
    ok, err = validate_uploaded_file(uploaded_file)
    if not ok:
        st.error(err)
        st.stop()

    file_key, media_path = stage_upload(uploaded_file)
    is_audio_only = media_path.suffix.lower() in AUDIO_EXTENSIONS

    with st.spinner("Extracting audio…"):
        try:
            wav_path = Path(validate_and_extract(str(media_path)))
        except Exception as e:
            st.error(
                "Could not read this file. Audio extraction requires **ffmpeg** on PATH "
                "(Ubuntu/Debian: `sudo apt install ffmpeg`).\n\n"
                f"Details: {e}"
            )
            st.stop()

    with st.expander("Media", expanded=False):
        if is_audio_only:
            st.audio(str(media_path))
        else:
            st.video(str(media_path))

    roi_tuple = parse_roi(roi_str)
    if use_vision and not is_audio_only:
        frame = first_video_frame(str(media_path))
        with st.expander("Find muzzle ROI coordinates", expanded=roi_tuple is None):
            if frame is None:
                st.info("Install opencv-python to preview frames: `pip install -e '.[video]'`")
            else:
                fig, ax = plt.subplots(figsize=(8, 8 * frame.shape[0] / frame.shape[1]))
                ax.imshow(frame)
                ax.grid(True, color="white", alpha=0.35, linewidth=0.6)
                if roi_tuple is not None:
                    x0, y0, w0, h0 = roi_tuple
                    ax.add_patch(plt.Rectangle((x0, y0), w0, h0, fill=False, color="#ff4b4b", lw=2))
                ax.set_title("First frame — read x (left→right) and y (top→bottom) in pixels")
                st.pyplot(fig)
                plt.close(fig)
        if roi_tuple is None:
            st.info("Vision confirmation is enabled but the ROI is missing or invalid (x,y,w,h).")
    elif use_vision and is_audio_only:
        st.info("Vision confirmation needs a video; it is skipped for WAV uploads.")

    vision_roi = (
        ",".join(str(v) for v in roi_tuple)
        if use_vision and roi_tuple is not None and not is_audio_only
        else None
    )
    with st.spinner("Detecting shots…"):
        fused_events, rejected_rows, vision_warning = run_detection(
            str(wav_path),
            str(media_path),
            float(sensitivity),
            int(min_sep_ms),
            int(echo_ms),
            environment,
            vision_roi,
        )
    if vision_warning:
        st.warning(vision_warning)

    # The working shot list lives in session state so hand edits survive
    # changing the analysis window. It resets when the file or a detection
    # setting changes (those produce a different detection result).
    state_key = (
        "events::"
        + hashlib.sha1(
            json.dumps(
                [file_key, sensitivity, min_sep_ms, echo_ms, environment, vision_roi]
            ).encode()
        ).hexdigest()
    )
    if state_key not in st.session_state:
        st.session_state[state_key] = [dict(e) for e in fused_events]
    working_events: list[dict[str, Any]] = st.session_state[state_key]
    editor_ver = st.session_state.get(f"{state_key}::ver", 0)

    if not fused_events and not rejected_rows:
        st.warning("No shots detected. Increase sensitivity or check audio quality.")
        st.stop()

    _, _, duration_s = load_wav_display_data(str(wav_path))
    full_times, full_rof, full_bursts, full_burst_summary, full_cadence = summarize_from_events(
        working_events,
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

            largest = max(
                range(len(burst_only_options)),
                key=lambda i: int(full_burst_summary[i].get("n_shots", 0)),
            )
            selected_burst_label = st.selectbox(
                "Choose burst",
                options=burst_labels,
                index=largest,
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

    selected_events = events_in_range(working_events, selected_start_s, selected_end_s)
    selected_rejected = [
        r
        for r in rejected_rows
        if selected_start_s <= r["t"] <= selected_end_s
        and not any(abs(r["t"] - float(e["t"])) < MATCH_EPS_S for e in working_events)
    ]

    final_times, final_rof, final_bursts, final_burst_summary, final_cadence = (
        summarize_from_events(selected_events, int(burst_gap_ms))
    )

    cyclic_rpm = final_rof.get("cyclic_rpm")
    mean_rpm = final_rof.get("mean_rpm")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Shots", f"{len(selected_events)}")
    c2.metric(
        "Cyclic rate (RPM)",
        f"{round(cyclic_rpm)}" if cyclic_rpm is not None else "—",
        help="Rate within bursts only (splits ≤ burst gap). This is the gun's rate of fire.",
    )
    c3.metric(
        "Overall rate (RPM)",
        f"{round(mean_rpm)}" if mean_rpm is not None else "—",
        help="Shots per minute from first to last shot, including pauses between bursts.",
    )
    c4.metric("Bursts", f"{len(final_bursts)}")

    st.subheader("Waveform")

    selected_width_s = float(selected_end_s - selected_start_s)
    wave_center = (selected_start_s + selected_end_s) / 2.0
    if auto_zoom_waveform:
        computed_wave_width = max(0.5, min(float(wave_window), selected_width_s * 1.2))
    else:
        computed_wave_width = float(wave_window)

    wc1, wc2 = st.columns([1, 1])
    with wc1:
        jump_options = ["Window center"] + [
            f"Shot {i + 1} @ {float(e['t']):.3f}s" for i, e in enumerate(selected_events)
        ]
        jump = st.selectbox("Center waveform on", jump_options, index=0)
        if jump != "Window center":
            wave_center = float(selected_events[jump_options.index(jump) - 1]["t"])
            computed_wave_width = min(computed_wave_width, 1.0)
    with wc2:
        listen_options = [
            f"Shot {i + 1} @ {float(e['t']):.3f}s" for i, e in enumerate(selected_events)
        ]
        listen_options += [f"Rejected @ {r['t']:.3f}s ({r['reason']})" for r in selected_rejected]
        listen_times = [float(e["t"]) for e in selected_events] + [
            r["t"] for r in selected_rejected
        ]
        if listen_options:
            listen = st.selectbox(
                "Listen to (±0.3 s)",
                listen_options,
                help="Hearing a candidate is often the fastest way to tell a shot from an echo.",
            )
            snippet, snippet_sr = audio_snippet(
                wav_path, listen_times[listen_options.index(listen)]
            )
            st.audio(snippet, sample_rate=snippet_sr)

    plot_waveform_window(
        wav_path=wav_path,
        events=working_events,
        bursts=full_bursts,
        center_s=wave_center,
        width_s=float(computed_wave_width),
        highlight_start=float(selected_start_s),
        highlight_end=float(selected_end_s),
        rejected=rejected_rows,
    )
    st.caption(
        "Yellow: analysis window. Green: bursts. Red lines: shots. "
        "Dotted grey: candidates the detector rejected (echoes, low score, too quiet)."
    )

    st.subheader("Review & Correct Shots")
    st.write(
        "Uncheck false positives, adjust timestamps, or add rows for missed shots, then "
        "click **Apply edits**. Edits are kept while you move the analysis window."
    )

    with st.form(f"editor_form::{state_key}::{editor_ver}"):
        edited_df = st.data_editor(
            events_to_table(selected_events),
            width="stretch",
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
            disabled=["Shot #", "Split (s)", "Inst. RPM", "Source"],
            key=f"editor::{state_key}::{editor_ver}::{selected_start_s}::{selected_end_s}",
        )
        apply_edits = st.form_submit_button("Apply edits", type="primary")

    if apply_edits:
        edited_window = editable_events_from_dataframe(edited_df)
        outside = [
            e for e in working_events if not (selected_start_s <= float(e["t"]) <= selected_end_s)
        ]
        st.session_state[state_key] = sorted(outside + edited_window, key=lambda e: float(e["t"]))
        bump_editor_version(state_key)
        st.rerun()

    ac1, ac2 = st.columns([3, 1])
    with ac1:
        if selected_rejected:
            rejected_labels = [
                f"{r['t']:.3f}s — {r['reason']} (score {r['audio_score']:.2f})"
                for r in selected_rejected
            ]
            to_add = st.multiselect(
                "Add back rejected candidates",
                rejected_labels,
                help="Candidates the detector found but rejected. Listen to them above first.",
            )
            if to_add and st.button("Add selected"):
                added = [
                    {
                        "t": selected_rejected[rejected_labels.index(lbl)]["t"],
                        "confidence": None,
                        "audio_score": selected_rejected[rejected_labels.index(lbl)]["audio_score"],
                        "video_score": None,
                        "source": "recovered",
                    }
                    for lbl in to_add
                ]
                st.session_state[state_key] = sorted(
                    working_events + added, key=lambda e: float(e["t"])
                )
                bump_editor_version(state_key)
                st.rerun()
    with ac2:
        if st.button("Reset to detected", help="Discard all edits for this file and settings."):
            st.session_state[state_key] = [dict(e) for e in fused_events]
            bump_editor_version(state_key)
            st.rerun()

    if not selected_events:
        st.warning("No shots in the selected window.")
        st.stop()

    final_df = events_to_table(selected_events, include_keep=False)

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
                "Start (s)": round(float(seg_times[0]), 3),
                "End (s)": round(float(seg_times[-1]), 3),
                "Duration (s)": round(float(seg_times[-1] - seg_times[0]), 3),
                "Rate (RPM)": round(float(bs["mean_rpm"]))
                if bs.get("mean_rpm") is not None
                else None,
                "Median Split (s)": round(float(np.median(seg_splits)), 3)
                if seg_splits.size
                else None,
                "Cadence CV": round(float(c["cv"]), 3) if c["cv"] is not None else None,
            }
        )
    burst_df = pd.DataFrame(burst_rows)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Final Shot Data")
        st.dataframe(
            final_df,
            width="stretch",
            hide_index=True,
            column_config={"Timestamp (s)": st.column_config.NumberColumn(format="%.4f")},
        )

    with col2:
        st.subheader("Split Consistency")
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_df = final_df.dropna(subset=["Split (s)"])
        if not plot_df.empty:
            ax.plot(plot_df["Shot #"], plot_df["Split (s)"], marker="o")
            in_burst = plot_df[plot_df["Split (s)"] <= burst_gap_ms / 1000.0]
            if not in_burst.empty:
                ax.set_ylim(0, max(float(in_burst["Split (s)"].max()) * 1.5, 0.05))
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
        st.info("Cadence stability requires at least 3 shots.")

    st.dataframe(burst_df, width="stretch", hide_index=True)

    report = {
        "input": {"file": uploaded_file.name},
        "params": {
            "environment": environment,
            "sensitivity": float(sensitivity),
            "min_separation_ms": int(min_sep_ms),
            "echo_window_ms": int(echo_ms),
            "burst_gap_ms": int(burst_gap_ms),
            "use_vision": bool(use_vision),
            "roi": vision_roi,
            "roi_interactive": bool(roi_interactive),
            "selected_start_s": float(selected_start_s),
            "selected_end_s": float(selected_end_s),
        },
        "initial_events": fused_events,
        "rejected_candidates": rejected_rows,
        "final_events": selected_events,
        "rof": final_rof,
        "bursts": final_bursts,
        "burst_summary": final_burst_summary,
        "cadence": final_cadence,
    }

    st.subheader("Export")
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            label="⬇️ Shot data (CSV)",
            data=final_df.to_csv(index=False).encode("utf-8"),
            file_name="shot_data.csv",
            mime="text/csv",
            width="stretch",
        )
    with dl2:
        st.download_button(
            label="⬇️ Full report (JSON)",
            data=json.dumps(report, indent=2).encode("utf-8"),
            file_name="analysis.json",
            mime="application/json",
            width="stretch",
        )
    with dl3:
        clip_name = Path(sanitize_filename(uploaded_file.name or "clip")).stem or "clip"
        sr_wav, _, _ = load_wav_display_data(str(wav_path))
        st.download_button(
            label="⬇️ Training data (ZIP)",
            data=build_training_zip(
                wav_path,
                clip_name,
                [float(e["t"]) for e in selected_events],
                sr_wav,
                labeled_range=(float(selected_start_s), float(selected_end_s)),
                source_name=uploaded_file.name,
            ),
            file_name=f"{clip_name}_labels.zip",
            mime="application/zip",
            width="stretch",
            help=(
                "Audio + your corrected shot times for the selected window, in the format "
                "train_shot_classifier.py reads (`--extra <unzipped folder>`). Real labeled "
                "clips are the single biggest way to improve detection accuracy."
            ),
        )

    with st.expander("Full-file baseline"):
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Shots", f"{len(working_events)}")
        b2.metric(
            "Cyclic rate (RPM)",
            f"{round(full_rof['cyclic_rpm'])}" if full_rof.get("cyclic_rpm") else "—",
        )
        b3.metric(
            "Overall rate (RPM)",
            f"{round(full_rof['mean_rpm'])}" if full_rof.get("mean_rpm") else "—",
        )
        b4.metric("Bursts", f"{len(full_bursts)}")

    with st.expander("Details (JSON)"):
        st.json(report)
