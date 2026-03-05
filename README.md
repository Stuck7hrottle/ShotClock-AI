🎯 ShotClock-AI
Precise Firearm Rate-of-Fire (ROF) Estimation from Video

ShotClock-AI uses advanced audio impulse detection and optional muzzle flash verification to analyze firearm performance. Whether you are a competitive shooter tracking split times or a hobbyist testing equipment, this tool provides laboratory-grade cadence metrics from standard video footage.

🚀 Key Features
Hybrid Detection Engine: Uses adaptive onset detection for audio, with optional OpenCV-powered muzzle flash confirmation.

Advanced Echo Suppression: Indoor-friendly clustering prevents double-counting shots caused by range reverberations.

Rich Analytics: Generates detailed metrics including split times, instantaneous RPM, and burst segmentation.

Interactive Web UI: A full-featured Streamlit dashboard for visual waveform analysis and data export.

Professional Exports: Save your data in JSON or CSV formats, or generate annotated videos showing exactly where shots were detected.

🛠 Installation
1. System Requirement: FFmpeg
This project relies on ffmpeg for high-fidelity audio extraction.

Bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg
2. Python Setup
Bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ui,video]"
💻 Usage
Interactive Web Dashboard
The easiest way to analyze footage is through the Streamlit UI:

Bash
streamlit run ui/streamlit_app.py
Command Line Interface
For power users and batch processing:

Bash
# Basic audio-first detection
rof detect input.mp4 --out results.json

# Advanced detection with video confirmation
rof detect input.mp4 --roi "500,400,100,100" --annotate output.mp4
📊 Analytics & Metrics
ShotClock-AI provides comprehensive feedback for every session:

Shots & Bursts: Automatic grouping of shots into distinct strings of fire.

Cadence Stability: Coefficient of Variation (CV) tracking to measure how steady your rhythm is.

Split Consistency: Visual charts in the UI to identify performance trends.

⚖️ License
Distributed under the MIT License. See LICENSE for more information.

⚠️ Disclaimer: This tool is for analyzing existing footage only. It does not provide instructions for the use or handling of firearms.
