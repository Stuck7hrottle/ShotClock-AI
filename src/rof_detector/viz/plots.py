from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt


def plot_waveform_with_events(wav_path: Path, events: List[Dict], out_png: Path) -> None:
    sr, x = wavfile.read(str(wav_path))
    if x.ndim > 1:
        x = x[:, 0]
    if x.dtype.kind in ("i", "u"):
        maxv = np.iinfo(x.dtype).max
        x = x.astype(np.float32) / maxv
    else:
        x = x.astype(np.float32)

    t = np.arange(len(x), dtype=np.float32) / float(sr)

    plt.figure()
    plt.plot(t, x)
    for e in events:
        plt.axvline(float(e["t"]))
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform with detected events")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
