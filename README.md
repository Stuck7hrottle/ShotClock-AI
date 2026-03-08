# ShotClock-AI

**ShotClock-AI** is a high-precision audio-driven rate-of-fire (ROF) detection tool designed to identify impulsive events (such as firearm shots) from audio or video sources. It detects shot timing, groups bursts, and computes rate-of-fire statistics while suppressing echoes and common acoustic artifacts.

The project focuses on **robust acoustic detection**, including:

* Echo suppression
* Close-interval shot detection (double taps)
* Burst segmentation
* Noise and interference resilience
* Accurate ROF statistics

ShotClock-AI can operate on **audio files or videos** and outputs structured results suitable for analysis or downstream processing.

---

# Features

### Shot Detection

Detects impulsive acoustic events using multiple signal features:

* Adaptive amplitude thresholding
* Crest factor analysis
* Kurtosis analysis
* Prominence detection
* Echo suppression

### Burst Detection

Automatically groups shots into bursts using configurable timing gaps.

### ROF Statistics

Provides rate-of-fire analytics including:

* Mean RPM
* Median RPM
* Percentile RPM
* Maximum RPM
* Burst statistics

### Robustness Improvements

The detector includes logic for handling:

* Echo reflections
* Close-spaced shots
* Mechanical clicks
* Microphone bumps
* Background noise

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/ShotClock-AI.git
cd ShotClock-AI
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -e .
```

The CLI command will now be available:

```bash
rof
```

---

# Basic Usage

Run detection on an audio file:

```bash
rof input.wav
```

Run detection on a video file:

```bash
rof input.mp4
```

Save results:

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
| `--echo-window-ms`    | Window used for echo suppression   |
| `--burst-gap-ms`      | Time gap used to split bursts      |
| `--environment`       | Optional preset environment tuning |

Example:

```bash
rof audio.wav \
    --sensitivity 0.48 \
    --min-separation-ms 35 \
    --echo-window-ms 30
```

---

# Default Detection Tuning

The current production defaults are tuned for realistic firearm-like acoustic signals.

```
sensitivity = 0.48
min_separation_ms = 35
echo_window_ms = 30
burst_gap_ms = 250
```

These values balance:

* echo rejection
* close-shot detection
* transient noise suppression

They were derived using the included synthetic regression dataset.

---

# Output Format

Detection results are returned as structured JSON:

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
│
├── test_audio_v2
│
├── generate_test_audio.py
├── generate_test_audio_v2.py
│
├── batch_analyze.py
├── batch_analyze_v2.py
│
└── README.md
```

---

# Tuning the Detector

ShotClock-AI includes tools for **systematic parameter tuning** using synthetic datasets.

Two scripts generate controlled audio scenarios:

```
generate_test_audio.py
generate_test_audio_v2.py
```

The second version produces **more realistic acoustic conditions**, including:

* filtered echoes
* impulse variation
* nuisance transients
* environmental noise

---

# Generating Test Audio

Generate the base dataset:

```bash
python generate_test_audio.py
```

Generate the advanced dataset:

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

Evaluate detector performance against the synthetic dataset:

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

The script also produces a detailed JSON report for each scenario.

---

# Important Test Scenarios

The synthetic dataset includes scenarios targeting specific failure modes.

| Scenario        | Purpose                  |
| --------------- | ------------------------ |
| steady cadence  | baseline accuracy        |
| fast burst      | high RPM detection       |
| echo chamber    | echo suppression         |
| jittery cadence | irregular firing         |
| interference    | transient noise          |
| slow fire       | isolated shots           |
| double taps     | close-interval detection |

These tests help verify that tuning changes do not introduce regressions.

---

# Recommended Tuning Workflow

1️⃣ Generate test audio

```bash
python generate_test_audio_v2.py
```

2️⃣ Run batch analysis

```bash
python batch_analyze_v2.py
```

3️⃣ Adjust parameters

```
sensitivity
min_separation_ms
echo_window_ms
```

4️⃣ Re-run batch analysis

Compare:

* precision
* recall
* F1 score

5️⃣ Verify edge scenarios

Particularly:

```
05_interference_and_clicks
08_slow_fire_with_bumps
09_double_taps_boundary
```

These scenarios expose most detection weaknesses.

---

# Tuning Guidelines

### Echo suppression

```
echo_window_ms
```

Typical range:

```
25–45 ms
```

Lower values preserve close shots but risk echo duplication.

---

### Shot separation

```
min_separation_ms
```

Typical range:

```
30–50 ms
```

Lower values allow faster firing detection.

---

### Sensitivity

```
sensitivity
```

Typical range:

```
0.45–0.55
```

Lower values increase recall but may increase false positives.

---

# Development Goals

Planned improvements:

* adaptive environment presets
* improved transient classification
* multi-channel microphone support
* GPU-accelerated signal analysis
* optional vision-based shot confirmation

---

# License

MIT License

---

# Contributing

Pull requests and improvements are welcome.

If contributing tuning improvements, please include:

* updated regression results
* batch analysis output
* explanation of parameter changes

---

# Acknowledgments

ShotClock-AI was built to explore high-accuracy acoustic ROF detection using signal processing techniques and synthetic regression testing.
