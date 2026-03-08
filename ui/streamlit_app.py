# ui/streamlit_app.py
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.io import wavfile

from rof_detector.audio.detect import detect_shots_audio
from rof_detector.fusion.fuse import fuse_scores
from rof_detector.io.ffmpeg import extract_audio_wav
from rof_detector.metrics.bursts import segment_bursts, summarize_bursts
from rof_detector.metrics.rof import compute_rof
from rof_detector.vision.flash_detect import confirm_shots_with_flash

st.set_page_config(page_title="ShotClock AI", layout="wide")

st.title("🎯 ShotClock AI: Rate of Fire Analyzer")
st.write("Upload a video to detect shots (audio-first) and compute splits / ROF.")


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


def plot_waveform_window(
    wav_path: Path, events: list[dict], center_s: float, width_s: float
) -> None:
    sr, x = wavfile.read(str(wav_path))
    if x.ndim > 1:
        x = x[:, 0]

    # normalize for display
    if x.dtype.kind in ("i", "u"):
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float32)

    n = len(x)
    dur_s = n / float(sr)

    width_s = float(max(1.0, width_s))
    center_s = float(np.clip(center_s, 0.0, dur_s))

    start_s = max(0.0, center_s - width_s / 2.0)
    end_s = min(dur_s, center_s + width_s / 2.0)

    s0 = int(start_s * sr)
    s1 = int(end_s * sr)

    seg = x[s0:s1]
    t = (np.arange(len(seg), dtype=np.float32) + s0) / float(sr)

    fig, ax = plt.subplots()
    ax.plot(t, seg)

    for e in events:
        et = float(e["t"])
        if start_s <= et <= end_s:
            ax.axvline(et)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Waveform ({start_s:.2f}s to {end_s:.2f}s)")
    st.pyplot(fig, width="stretch")


def cadence_stats_from_splits(splits: np.ndarray) -> dict:
    """
    Cadence stability: coefficient of variation (std/mean) on split times.
    Lower is steadier cadence.
    """
    splits = np.asarray(splits, dtype=float)
    splits = splits[np.isfinite(splits)]
    splits = splits[splits > 0]
    if splits.size < 2:
        return {"mean_split_s": None, "std_split_s": None, "cv": None}

    mean = float(np.mean(splits))
    std = float(np.std(splits, ddof=1))
    cv = float(std / mean) if mean > 0 else None
    return {"mean_split_s": mean, "std_split_s": std, "cv": cv}


with st.sidebar:
    st.header("Detection Settings")
    environment = st.selectbox("Environment", ["auto", "indoor", "outdoor"], index=0)

    sensitivity = st.slider(
        "Sensitivity",
        min_value=0.1,
        max_value=1.0,
        value=0.45,
        step=0.05,
    )

    min_sep_ms = st.number_input(
        "Min separation (ms)",
        min_value=10,
        max_value=200,
        value=50,
        step=5,
    )

    echo_ms = st.number_input(
        "Echo merge window (ms)",
        min_value=0,
        max_value=200,
        value=45,
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

    # Disable desktop OpenCV ROI picker in Streamlit/server environments
    roi_interactive = False
    st.caption(
        "Interactive OpenCV ROI picking is disabled in the web app because it "
        "requires a desktop GUI. Enter ROI manually as x,y,w,h instead."
    )

    st.divider()
    st.header("Waveform View")
    wave_window = st.number_input("Window width (s)", min_value=1.0, value=10.0, step=1.0)
    wave_center_manual = st.number_input("Center time (s)", min_value=0.0, value=0.0, step=1.0)


uploaded_file = st.file_uploader("Upload Video (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])

if uploaded_file:
    with tempfile.TemporaryDirectory() as td:
        tmp_video = Path(td) / "upload.mp4"
        tmp_video.write_bytes(uploaded_file.getbuffer())

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
                        'Vision confirmation is enabled, but ROI is missing or invalid. '
                        'Use format: x,y,w,h'
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

            fused = fuse_scores(audio_events, video_events)
            times = [e["t"] for e in fused]
            rof = compute_rof(times)

            bursts = segment_bursts(times, burst_gap_s=float(burst_gap_ms) / 1000.0)
            burst_summary = summarize_bursts(times, bursts)

        # Waveform debug
        st.subheader("Waveform Debug View")

        wave_center = float(wave_center_manual)
        if fused:
            shot_choice = st.selectbox(
                "Jump waveform to shot #",
                options=[0] + list(range(1, len(fused) + 1)),
                index=0,
                help="Choose a shot number to center the waveform around that shot. 0 = manual center.",
            )
            if shot_choice != 0:
                wave_center = float(fused[shot_choice - 1]["t"])

        plot_waveform_window(wav_path, fused, center_s=wave_center, width_s=float(wave_window))

        if not fused:
            st.warning("No shots detected. Increase sensitivity or check audio quality.")
            st.stop()

        # Build shot table
        rows = []
        for i, e in enumerate(fused):
            t = float(e["t"])
            split = t - float(fused[i - 1]["t"]) if i > 0 else 0.0
            inst_rpm = (60.0 / split) if split > 0 else 0.0
            rows.append(
                {
                    "Shot #": i + 1,
                    "Timestamp (s)": round(t, 3),
                    "Split (s)": round(split, 3) if i > 0 else None,
                    "Inst. RPM": int(round(inst_rpm)) if inst_rpm else None,
                    "Confidence": round(float(e.get("confidence", 0.0)), 3),
                }
            )
        df = pd.DataFrame(rows)

        # Global cadence stats (exclude first split None)
        splits = df["Split (s)"].dropna().to_numpy(dtype=float)
        cadence = cadence_stats_from_splits(splits)

        # Burst cadence stats
        burst_rows = []
        for bi, b in enumerate(bursts):
            s = int(b["start_index"])
            e = int(b["end_index"])
            seg_times = np.array(times[s : e + 1], dtype=float)
            seg_splits = np.diff(seg_times) if seg_times.size >= 2 else np.array([], dtype=float)
            c = cadence_stats_from_splits(seg_splits)

            bs = burst_summary[bi] if bi < len(burst_summary) else {}
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

        # Build downloadable artifacts
        csv_data = df.to_csv(index=False).encode("utf-8")

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
                "wave_center_s": float(wave_center),
                "wave_window_s": float(wave_window),
            },
            "events": fused,
            "rof": rof,
            "bursts": bursts,
            "burst_summary": burst_summary,
            "cadence": cadence,
        }
        json_data = json.dumps(report, indent=2).encode("utf-8")

        # Top-level summary cards
        mean_rpm = rof.get("mean_rpm")
        median_rpm = rof.get("median_rpm")
        max_rpm = rof.get("max_rpm")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Shots", f"{len(fused)}")
        c2.metric("Avg RPM", f"{int(round(mean_rpm))}" if mean_rpm is not None else "—")
        c3.metric("Max RPM", f"{int(round(max_rpm))}" if max_rpm is not None else "—")
        c4.metric("Bursts", f"{len(bursts)}")

        # Layout: shot table + split chart
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Shot Data")
            st.dataframe(df, use_container_width=True)

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
            fig, ax = plt.subplots()
            plot_df = df.dropna(subset=["Split (s)"])
            ax.plot(plot_df["Shot #"], plot_df["Split (s)"], marker="o")
            ax.set_ylabel("Split Time (seconds)")
            ax.set_xlabel("Shot Number")
            st.pyplot(fig)

        # Burst summary section
        st.subheader("Burst Summary")

        # Global cadence
        if cadence["cv"] is not None:
            st.info(
                f"Cadence stability (CV of splits): **{cadence['cv']:.3f}** "
                f"(mean split {cadence['mean_split_s']:.3f}s, std {cadence['std_split_s']:.3f}s). "
                "Lower CV = steadier cadence."
            )
        else:
            st.info("Cadence stability requires at least 3 detected shots.")

        st.dataframe(burst_df, use_container_width=True)

        with st.expander("Details (JSON)"):
            st.json(report)
