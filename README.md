# ShotClock-AI

Estimate a firearm's **rate of fire (ROF)** from video by detecting **shot events** using audio impulse detection with optional **muzzle flash confirmation** from video.

This tool is useful for analyzing training footage, competition shooting, testing videos, and other recorded firearm activity.

> ⚠️ This project analyzes **existing footage only**. It does **not** provide instructions for firearm use.

---

# Features

- Extract audio from video using **ffmpeg**
- Detect shot-like **impulses** using adaptive onset detection
- Indoor-friendly **echo clustering** to avoid double-counting
- Adjustable **sensitivity control** (0.1–1.0)
- Optional **video confirmation** using muzzle flash detection
- Burst segmentation and cadence analysis

### Metrics Generated

- Shot timestamps
- Split times between shots
- Instantaneous ROF
- Mean ROF
- Burst segmentation
- Burst statistics

### Export Options

- JSON report
- CSV event data
- Waveform visualization
- Annotated video with detected shots

---

# Installation

## System Dependencies (Linux)

This project requires **ffmpeg**.

Install on Debian/Ubuntu:

```bash
sudo apt update
sudo apt install ffmpeg

## Verify installation:

ffmpeg -version
Python Setup

## Create a virtual environment:

python -m venv .venv
source .venv/bin/activate

## Install the project:

pip install -e ".[dev,ui]"

Optional (for video confirmation features):

pip install -e ".[dev,ui,video]"
Running the Web UI

## Launch the Streamlit interface:

streamlit run ui/streamlit_app.py

## Then open:

http://localhost:8501
UI Capabilities

Upload video files

Adjust detection sensitivity

Tune echo suppression

View shot tables and split charts

Export CSV / JSON reports

## Optional:

Enable muzzle flash confirmation

Provide ROI coordinates (x,y,w,h)

Command Line Usage

Basic analysis:

rof detect input.mp4 \
  --out results.json \
  --csv results.csv \
  --plot waveform.png

With muzzle flash confirmation:

rof detect input.mp4 \
  --roi "x,y,w,h" \
  --annotate annotated.mp4

Interactive ROI selection:

rof detect input.mp4 --roi-interactive
Example Output

Example summary:

Shots detected: 14
Average ROF: 742 RPM
Median ROF: 735 RPM
Burst count: 2

Example CSV:

Shot	Timestamp	Split	Inst RPM
1	0.212	—	—
2	0.335	0.123	487
3	0.458	0.123	487
Development

Install development environment:

pip install -e ".[dev,ui,video]"

Run tests:

pytest -q

Lint code:

ruff check .
ruff format .
Project Structure
src/
  rof_detector/
    audio/        # shot detection
    vision/       # flash detection
    fusion/       # audio + video fusion
    metrics/      # ROF calculations
    viz/          # plots / video annotation
    io/           # ffmpeg utilities

ui/
  streamlit_app.py

tests/
  unit tests
Roadmap

Future improvements planned:

automatic muzzle detection

improved flash detection

waveform visualization in UI

automatic sensitivity calibration

batch video processing

cadence stability metrics

License

MIT
