#!/usr/bin/env python3
"""Train the logistic-regression shot scorer used by audio/scoring.py.

Replaces the hand-tuned score formula in audio/detect.py with weights
learned from the synthetic regression dataset (generate_test_audio_v2.py +
ground_truth.json), which already carries labeled true-shot timestamps.

Usage:
    python generate_test_audio_v2.py   # if test_audio_v2/ doesn't exist yet
    python train_shot_classifier.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rof_detector.audio.detect import find_candidates
from rof_detector.audio.scoring import FEATURE_NAMES, MODEL_PATH, feature_vector

# Scenarios held out of training entirely, used only to report honest
# validation metrics. Chosen from the README's "important test scenarios"
# table to cover echo, double-tap, and noisy-environment edge cases.
VAL_SCENARIOS = {
    "08_slow_fire_with_bumps.wav",
    "09_double_taps_boundary.wav",
    "10_noisy_env_harsh.wav",
    "13_echo_vs_doubletap_ambiguous.wav",
}

# Sample candidates at a spread of sensitivities so the classifier sees
# threshold contexts similar to what real users will run with, rather than
# overfitting to a single operating point.
TRAIN_SENSITIVITIES = [0.35, 0.48, 0.6, 0.75]

MATCH_TOLERANCE_S = 0.035


def _label_candidates(
    candidates: list[dict], expected: list[float], tolerance_s: float
) -> list[tuple[dict, int]]:
    expected_sorted = sorted(expected)
    used = [False] * len(expected_sorted)
    labeled = []
    for c in sorted(candidates, key=lambda c: c["t"]):
        t = c["t"]
        best_i, best_err = None, None
        for i, et in enumerate(expected_sorted):
            if used[i]:
                continue
            err = abs(t - et)
            if err <= tolerance_s and (best_err is None or err < best_err):
                best_i, best_err = i, err
        label = 0
        if best_i is not None:
            used[best_i] = True
            label = 1
        labeled.append((c, label))
    return labeled


def build_dataset(
    audio_dir: Path, truth_path: Path
) -> tuple[dict[str, list[tuple[np.ndarray, int]]], list[str]]:
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    by_scenario: dict[str, list[tuple[np.ndarray, int]]] = {}
    feature_names = FEATURE_NAMES

    for item in truth:
        fname = item["file"]
        wav_path = audio_dir / fname
        if not wav_path.exists():
            continue
        expected = [float(t) for t in item.get("expected_primary_shots", [])]

        examples: list[tuple[np.ndarray, int]] = []
        for sens in TRAIN_SENSITIVITIES:
            candidates = find_candidates(wav_path, sensitivity=sens)
            for c, label in _label_candidates(candidates, expected, MATCH_TOLERANCE_S):
                examples.append((feature_vector(c["features"]), label))
        by_scenario[fname] = examples

    return by_scenario, feature_names


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 0.5,
    lr: float = 0.3,
    iters: int = 4000,
    max_neg_weight_ratio: float = 3.0,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    Xz = (X - mean) / std

    n, d = Xz.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0

    # Real shots vastly outnumber false candidates in this dataset (most
    # scenarios are clean), so give false candidates a modest weight boost
    # to make sure the small number of hard negatives (echoes, clicks,
    # bumps) actually influence the fit. Cap the ratio: fully balancing to
    # 50/50 overcorrects and starts rejecting true positives that resemble
    # the (few, noisy) negative examples.
    w_pos = 1.0
    n_pos = max(1, int(y.sum()))
    n_neg = max(1, int((1 - y).sum()))
    w_neg = min(max_neg_weight_ratio, n_pos / n_neg)
    sample_w = np.where(y == 1, w_pos, w_neg)

    for _ in range(iters):
        z = Xz @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        grad_z = (p - y) * sample_w
        grad_w = (Xz.T @ grad_z) / n + l2 * w / n
        grad_b = float(np.sum(grad_z)) / n
        w -= lr * grad_w
        b -= lr * grad_b

    return w, b, mean, std


def evaluate(
    w: np.ndarray, b: float, mean: np.ndarray, std: np.ndarray, X: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    if len(y) == 0:
        return {"n": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    Xz = (X - mean) / std
    p = 1.0 / (1.0 + np.exp(-(Xz @ w + b)))
    pred = (p >= 0.5).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"n": len(y), "precision": precision, "recall": recall, "f1": f1}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the audio shot classifier.")
    parser.add_argument("--audio-dir", default="test_audio_v2")
    parser.add_argument("--truth", default="test_audio_v2/ground_truth.json")
    parser.add_argument("--out", default=str(MODEL_PATH))
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    truth_path = Path(args.truth)
    if not truth_path.exists():
        raise SystemExit(f"{truth_path} not found. Run `python generate_test_audio_v2.py` first.")

    by_scenario, feature_names = build_dataset(audio_dir, truth_path)

    train_X, train_y, val_X, val_y = [], [], [], []
    for fname, examples in by_scenario.items():
        is_val = fname in VAL_SCENARIOS
        for x, label in examples:
            (val_X if is_val else train_X).append(x)
            (val_y if is_val else train_y).append(label)

    train_X = np.array(train_X, dtype=np.float64)
    train_y = np.array(train_y, dtype=np.float64)
    val_X = np.array(val_X, dtype=np.float64) if val_X else np.zeros((0, len(feature_names)))
    val_y = np.array(val_y, dtype=np.float64)

    print(f"Train examples: {len(train_y)} (positives={int(train_y.sum())})")
    print(f"Val examples:   {len(val_y)} (positives={int(val_y.sum())}) [{sorted(VAL_SCENARIOS)}]")

    w, b, mean, std = train_logistic_regression(train_X, train_y)

    train_metrics = evaluate(w, b, mean, std, train_X, train_y)
    val_metrics = evaluate(w, b, mean, std, val_X, val_y)
    print(
        f"Train: precision={train_metrics['precision']:.3f} recall={train_metrics['recall']:.3f} "
        f"f1={train_metrics['f1']:.3f}"
    )
    print(
        f"Val:   precision={val_metrics['precision']:.3f} recall={val_metrics['recall']:.3f} "
        f"f1={val_metrics['f1']:.3f}"
    )

    out_path = Path(args.out)
    out_path.write_text(
        json.dumps(
            {
                "feature_names": feature_names,
                "weights": w.tolist(),
                "bias": b,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "val_scenarios": sorted(VAL_SCENARIOS),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved model: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
