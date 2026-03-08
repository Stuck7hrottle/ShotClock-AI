# ShotClock-AI

**ShotClock-AI** is a high-precision acoustic rate-of-fire (ROF) analysis tool designed to detect impulsive events such as firearm shots from **audio or video sources**.

It extracts shot timestamps, groups bursts, and calculates rate-of-fire statistics while suppressing echoes and acoustic artifacts.

The system is designed around **robust signal-processing techniques** rather than simple peak detection, enabling it to operate reliably in challenging acoustic environments.

---

# Features

### Shot Detection

ShotClock-AI identifies impulsive acoustic events using multiple signal features:

* Adaptive amplitude thresholding
* Crest factor analysis
* Kurtosis analysis
* Peak prominence detection
* Echo suppression

### Burst Detection

Automatically groups shots into bursts based on timing gaps.

### ROF Statistics

Computes detailed rate-of-fire metrics:

* Mean RPM
* Median RPM
* Percentile RPM
* Maximum RPM
* Burst-level ROF statistics

### Robustness Improvements

The detector includes specialized logic to handle:

* Echo reflections
* Close-interval shots (double taps)
* Mechanical clicks
* Microphone bumps
* Background noise

---

# Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/ShotClock-AI.git
cd ShotClock-AI
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the package:

```bash
pip install -e .
```

The CLI command will now be available:

```bash
rof
```

---

# Basic CLI Usage

Run detection on an audio file:

```bash
rof input.wav
```

Run detection on a video file:

```bash
rof input.mp4
```

Save JSON results:

```bash
rof input.wav --out results.json
```

Export CSV:

```bash
rof input.wav --csv results.csv
```

Generate waveform plot:

```bash
rof input.wav --plot waveform.png
```

---

# CLI Parameters

| Parameter             | Description                        |
| --------------------- | ---------------------------------- |
| `--sensitivity`       | Detection sensitivity              |
| `--min-separation-ms` | Minimum allowed time between shots |
| `--echo-window-ms`    | Echo suppression window            |
| `--burst-gap-ms`      | Gap used to split bursts           |
| `--environment`       | Optional preset tuning             |

Example:

```bash
rof audio.wav \
    --sensitivity 0.48 \
    --min-separation-ms 35 \
    --echo-window-ms 30
```

---

# Default Detection Tuning

Production defaults:

```
sensitivity = 0.48
min_separation_ms = 35
echo_window_ms = 30
burst_gap_ms = 250
```

These values were derived using the synthetic regression dataset included with the project.

They balance:

* echo rejection
* close-shot detection
* transient noise suppression

---

# Output Format

Example JSON output:

```json
{
  "events": [
    {
      "t": 0.495,
      "confidence": 0.52,
      "audio_score": 0.74
    }
  ],
  "rof": {
    "mean_rpm": 720.4,
    "max_rpm": 1040.2
  },
  "bursts": [...]
}
```

---

# Streamlit Web Interface

ShotClock-AI includes an optional **Streamlit interface** for interactive analysis.

Run the web UI:

```bash
streamlit run streamlit_app.py
```

The interface allows you to:

* Upload audio or video files
* Run detection interactively
* Visualize detected shots
* Display ROF statistics
* Inspect waveform plots

This interface is useful for **manual verification and quick analysis** without using the CLI.

---

# Project Structure

```
ShotClock-AI
│
├── rof_detector
│   ├── cli.py
│   ├── detect.py
│   ├── audio.py
│
├── test_audio
├── test_audio_v2
│
├── generate_test_audio.py
├── generate_test_audio_v2.py
│
├── batch_analyze.py
├── batch_analyze_v2.py
│
├── streamlit_app.py
│
└── README.md
```

---

# Tuning the Detector

ShotClock-AI includes tools for **systematic tuning and regression testing**.

Two scripts generate synthetic audio datasets:

```
generate_test_audio.py
generate_test_audio_v2.py
```

The second version produces more realistic conditions:

* filtered echoes
* impulse variation
* nuisance transients
* environmental noise

---

# Generating Test Audio

Basic dataset:

```bash
python generate_test_audio.py
```

Advanced dataset:

```bash
python generate_test_audio_v2.py
```

This produces:

```
test_audio_v2/
    *.wav
    ground_truth.json
```

The ground truth file contains expected shot timestamps.

---

# Batch Analysis

Evaluate detection performance:

```bash
python batch_analyze_v2.py
```

Example output:

```
Expected shots: 128
Detected events: 135

True positives: 125
False positives: 10
False negatives: 3

Precision: 92.6%
Recall: 97.6%
F1 score: 95.0%
```

The script also produces a detailed JSON regression report.

---

# Important Test Scenarios

| Scenario        | Purpose                  |
| --------------- | ------------------------ |
| steady cadence  | baseline accuracy        |
| fast burst      | high RPM                 |
| echo chamber    | echo suppression         |
| jittery cadence | irregular firing         |
| interference    | transient noise          |
| slow fire       | isolated shots           |
| double taps     | close-interval detection |

---

# Recommended Tuning Workflow

1. Generate advanced test audio

```bash
python generate_test_audio_v2.py
```

2. Run regression analysis

```bash
python batch_analyze_v2.py
```

3. Adjust parameters

```
sensitivity
min_separation_ms
echo_window_ms
```

4. Re-run regression

Focus on these edge scenarios:

```
05_interference_and_clicks
08_slow_fire_with_bumps
09_double_taps_boundary
```

---

# Performance Benchmarks

Using the included synthetic regression dataset:

| Metric    | Result  |
| --------- | ------- |
| Precision | ~92–94% |
| Recall    | ~97–98% |
| F1 Score  | ~95%    |

The system maintains high recall while preserving accurate burst timing and ROF statistics.

---

# Real-World Audio Limitations

ShotClock-AI is designed for impulsive acoustic events but performance depends on recording conditions.

Performance may degrade when:

* microphone clipping occurs
* heavy compression is applied to audio
* strong mechanical impacts occur near the microphone
* echoes arrive very close to the direct impulse
* heavy crowd noise is present

For best results:

* use uncompressed audio
* avoid automatic gain control
* keep microphones within reasonable distance of the source

---

# Development Goals

Planned improvements:

* environment presets
* improved transient classification
* multi-channel microphone support
* GPU-accelerated signal analysis
* optional vision-based shot confirmation

---

# Contributing

Pull requests and improvements are welcome.

When submitting tuning changes please include:

* updated regression results
* batch analysis output
* explanation of parameter changes

---

# License

MIT License
