# ui/streamlit_app.py
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from rof_detector.audio.detect import detect_shots_audio
from rof_detector.fusion.fuse import fuse_scores
from rof_detector.io.ffmpeg import extract_audio_wav
from rof_detector.metrics.rof import compute_rof
from rof_detector.vision.flash_detect import confirm_shots_with_flash

st.set_page_config(page_title="ShotClock AI (rof-from-video)", layout="wide")

st.title("🎯 ShotClock AI: Rate of Fire Analyzer")
st.write("Upload a video to detect shots (audio-first) and compute splits / ROF.")

with st.sidebar:
    st.header("Detection Settings")
    environment = st.selectbox("Environment", ["auto", "indoor", "outdoor"], index=0)
    sensitivity = st.slider("Sensitivity", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
    min_sep_ms = st.number_input("Min separation (ms)", min_value=10, max_value=200, value=35, step=5)
    echo_ms = st.number_input("Echo merge window (ms)", min_value=0, max_value=200, value=60, step=5)

    st.divider()
    st.header("Vision Confirmation (optional)")
    use_vision = st.checkbox("Enable muzzle-flash confirmation", value=False)
    roi_str = st.text_input('ROI "x,y,w,h" (recommended)', value="")
    roi_interactive = st.checkbox("Pick ROI interactively (desktop)", value=False)
    st.caption("Vision requires opencv-python installed (pip install -e '.[dev,ui,video]').")

uploaded_file = st.file_uploader("Upload Video (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])

if uploaded_file:
    with tempfile.TemporaryDirectory() as td:
        tmp_video = Path(td) / "upload.mp4"
        tmp_video.write_bytes(uploaded_file.getbuffer())

        st.video(str(tmp_video))

        with st.spinner("Extracting audio and detecting shots…"):
            wav_path = extract_audio_wav(tmp_video)

            audio_events = detect_shots_audio(
                wav_path,
                sensitivity=float(sensitivity),
                min_separation_ms=int(min_sep_ms),
                echo_window_ms=int(echo_ms),
                environment=environment,
            )

            video_events = None
            if use_vision and (roi_str.strip() or roi_interactive):
                try:
                    video_events = confirm_shots_with_flash(
                        video_path=tmp_video,
                        audio_events=audio_events,
                        roi=roi_str.strip() if roi_str.strip() else None,
                        roi_interactive=bool(roi_interactive),
                    )
                except Exception as e:
                    st.warning(f"Vision confirmation unavailable: {e}")

            fused = fuse_scores(audio_events, video_events)
            times = [e["t"] for e in fused]
            rof = compute_rof(times)

        if not fused:
            st.warning("No shots detected. Increase sensitivity or check audio quality.")
        else:
            # Table (classic “shot clock” style) + confidence
            rows = []
            for i, e in enumerate(fused):
                t = float(e["t"])
                split = t - float(fused[i - 1]["t"]) if i > 0 else 0.0
                inst_rpm = (60.0 / split) if split > 0 else 0.0
                rows.append(
                    {
                        "Shot #": i + 1,
                        "Timestamp (s)": round(t, 3),
                        "Split (s)": round(split, 3),
                        "Inst. RPM": int(round(inst_rpm)) if inst_rpm else 0,
                        "Confidence": round(float(e.get("confidence", 0.0)), 3),
                    }
                )
            df = pd.DataFrame(rows)

            # Build downloadable artifacts
            csv_data = df.to_csv(index=False).encode("utf-8")

            report = {
                "input": {"video": uploaded_file.name},
                "params": {
                    "environment": environment,
                    "sensitivity": float(sensitivity),
                    "min_separation_ms": int(min_sep_ms),
                    "echo_window_ms": int(echo_ms),
                    "use_vision": bool(use_vision),
                    "roi": roi_str.strip() if roi_str.strip() else None,
                    "roi_interactive": bool(roi_interactive),
                },
                "events": fused,
                "rof": rof,
            }
            json_data = json.dumps(report, indent=2).encode("utf-8")

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
                ax.plot(df["Shot #"], df["Split (s)"], marker="o")
                ax.set_ylabel("Split Time (seconds)")
                ax.set_xlabel("Shot Number")
                st.pyplot(fig)

            mean_rpm = rof.get("mean_rpm")
            if mean_rpm is not None:
                st.success(f"Average Rate of Fire: {int(round(mean_rpm))} RPM")
            else:
                st.success("Detected 1 shot.")

            with st.expander("Details (JSON)"):
                st.json(report)
