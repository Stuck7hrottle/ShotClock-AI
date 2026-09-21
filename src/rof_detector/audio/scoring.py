from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# Trained by train_shot_classifier.py against the synthetic regression
# dataset (generate_test_audio_v2.py + ground_truth.json). Ships as package
# data; detection falls back to the hand-tuned heuristic below if missing.
MODEL_PATH = Path(__file__).parent / "model.json"

FEATURE_NAMES = [
    "crest_factor",
    "kurtosis",
    "clipped",
    "onset_height",
    "onset_prominence",
    "prom_ratio",
    "height_ratio",
]


def feature_vector(features: Dict) -> np.ndarray:
    # prom_ratio/height_ratio are peak-height-over-threshold ratios. On
    # near-silent synthetic audio the threshold can be almost zero, so raw
    # ratios can reach 1e5+; on real audio with a healthy noise floor the
    # same "clearly above threshold" candidate might only score 2-10. log1p
    # compresses that unbounded range onto a scale where both look similar,
    # instead of every real-world candidate reading as a training-set outlier.
    return np.array(
        [
            min(float(features["crest_factor"]), 40.0),
            min(float(features["kurtosis"]), 200.0),
            1.0 if features["clipped"] else 0.0,
            float(features["onset_height"]),
            float(features["onset_prominence"]),
            np.log1p(max(0.0, float(features["prom_ratio"]))),
            np.log1p(max(0.0, float(features["height_ratio"]))),
        ],
        dtype=np.float64,
    )


class ShotScorer:
    """Logistic-regression scorer: P(real shot | features)."""

    def __init__(self, weights, bias, mean, std):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.bias = float(bias)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)

    @classmethod
    def load(cls, path: Path) -> "ShotScorer":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["weights"], data["bias"], data["mean"], data["std"])

    def score(self, features: Dict) -> float:
        z = (feature_vector(features) - self.mean) / self.std
        logit = float(np.dot(self.weights, z) + self.bias)
        return float(1.0 / (1.0 + np.exp(-logit)))


def heuristic_score(features: Dict) -> float:
    """Original hand-tuned scoring formula, kept as a fallback for
    environments where the trained model.json isn't available."""
    cf = float(features["crest_factor"])
    kurt = float(features["kurtosis"])
    clipped = bool(features["clipped"])
    prom_ratio = float(features["prom_ratio"])
    height_ratio = float(features["height_ratio"])

    score = 0.0
    score += min(1.0, (cf / 10.0)) * (0.52 if not clipped else 0.24)
    score += min(1.0, (kurt / 50.0)) * 0.30
    score += min(0.22, max(0.0, prom_ratio - 1.0) * 0.10)
    score += min(0.14, max(0.0, height_ratio - 1.0) * 0.05)

    if prom_ratio < 1.18:
        score -= 0.08
    if height_ratio < 1.10:
        score -= 0.05

    return float(np.clip(score, 0.0, 1.0))


_scorer: Optional[ShotScorer] = None
_scorer_loaded = False


def _get_scorer() -> Optional[ShotScorer]:
    global _scorer, _scorer_loaded
    if not _scorer_loaded:
        _scorer_loaded = True
        if MODEL_PATH.exists():
            _scorer = ShotScorer.load(MODEL_PATH)
    return _scorer


def score_candidate(features: Dict) -> float:
    scorer = _get_scorer()
    if scorer is not None:
        return scorer.score(features)
    return heuristic_score(features)
