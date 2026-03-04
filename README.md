# rof-from-video

Estimate a firearm's **rate of fire** from a video by detecting **shot events** using **audio-first detection** with optional **muzzle flash confirmation** from video.

> ⚠️ This project is intended for **measurement and analysis of existing footage** (e.g., training, testing, research). It does not provide instructions for weapon use.

## Features (v0.2)
- Extract audio from video via `ffmpeg`
- Detect shot-like **impulses** in audio (adaptive onset + peak picking)
- Indoor-friendly **echo clustering** to reduce double-counting
- **Sensitivity control** (0.1–1.0) to tune detection threshold
- Optional video ROI-based **flash confirmation**
- Compute:
  - shot timestamps
  - inter-shot intervals
  - instantaneous ROF (shots/sec, RPM)
  - burst segmentation + burst statistics
- Export:
  - JSON report
  - CSV of events
  - waveform plot with detected shots
  - optional annotated video

## Install
### Requirements
- Python 3.10+
- `ffmpeg` available on PATH
- (Optional) `opencv-python` for video confirmation/annotation

### Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Quickstart
Audio-only:
```bash
rof detect input.mp4 --out results.json --csv results.csv --plot waveform.png
```

Audio + video confirmation (ROI):
```bash
rof detect input.mp4 --roi "x,y,w,h" --out results.json --annotate annotated.mp4
```

Interactive ROI selection:
```bash
rof detect input.mp4 --roi-interactive --out results.json
```

## Web UI (Streamlit)

Install with UI extras:
```bash
pip install -e ".[dev,ui]"
# for vision confirmation in the UI:
pip install -e ".[dev,ui,video]"
```

Run:
```bash
streamlit run ui/streamlit_app.py
```

## Development
Run tests:
```bash
pytest -q
```

Lint:
```bash
ruff check .
ruff format .
```

## License
MIT
