# ShotClock-AI

Estimate a firearm’s **rate of fire (ROF)** from video by detecting **shot events** using audio impulse detection with optional **muzzle flash confirmation** from video.

> ⚠️ This project analyzes **existing footage only**. It does **not** provide instructions for firearm use.

---

## Features

### Detection
- Extract audio from video via **ffmpeg**
- Detect shot-like impulses using an adaptive onset detector
- Indoor-friendly **echo clustering** to reduce double-counting
- Adjustable **sensitivity** (0.1–1.0)

### Optional video confirmation
- ROI-based **muzzle flash detection**
- Optional interactive ROI selection (desktop OpenCV window)

### Analysis metrics
- Shot timestamps
- Split times between shots
- Instantaneous ROF
- Mean / median / max RPM
- **Burst segmentation**
- **Burst summary metrics**
- **Cadence stability** (split consistency via coefficient of variation)

### Visualization
- Waveform debug view with shot markers
- Jump-to-shot waveform navigation
- Split consistency chart

### Exports
- CSV shot table
- JSON analysis report
- Optional annotated video (CLI)

---

## Installation (Linux)

### System dependencies
This project requires **ffmpeg**.

Debian/Ubuntu:
```bash
sudo apt update
sudo apt install ffmpeg
```

Verify:
```bash
ffmpeg -version
```

### Python environment
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Install (CLI + dev tools)
```bash
pip install -e ".[dev]"
```

### Install (Web UI)
```bash
pip install -e ".[dev,ui]"
```

### Install (Web UI + vision confirmation)
```bash
pip install -e ".[dev,ui,video]"
```

---

## Run the Web UI (Streamlit)

```bash
streamlit run ui/streamlit_app.py
```

Then open:
- http://localhost:8501

UI capabilities:
- Upload videos
- Tune sensitivity / echo suppression
- Configure burst gap
- Optional muzzle flash confirmation (ROI)
- Waveform debug view + jump-to-shot
- Download CSV + JSON reports

---

## Command line usage

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

---

## Example output

Example summary:
- Shots detected: 14  
- Average ROF: 742 RPM  
- Max ROF: 801 RPM  
- Bursts: 2  

Burst summary example:

| Burst | Shots | Duration (s) | Mean RPM | Cadence CV |
|------:|------:|-------------:|---------:|-----------:|
| 1 | 7 | 0.54 | 778 | 0.032 |
| 2 | 7 | 0.56 | 750 | 0.041 |

---

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

---

## Project structure

```
src/rof_detector/
  audio/        # shot detection
  vision/       # flash detection
  fusion/       # audio + video fusion
  metrics/      # ROF, bursts, cadence
  viz/          # plots / annotation
  io/           # ffmpeg helpers

ui/
  streamlit_app.py

tests/
```

---

## Roadmap

Planned improvements:
- Automatic threshold calibration (noise-floor aware)
- Better indoor echo suppression
- Improved flash scoring and ROI guidance
- Batch processing
- Optional “labeling mode” for precision/recall evaluation

---

## License
MIT
