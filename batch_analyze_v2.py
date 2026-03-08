#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass
class MatchResult:
    tp: int
    fp: int
    fn: int
    matched_pairs: list[tuple[float, float]]  # (expected, detected)
    unmatched_expected: list[float]
    unmatched_detected: list[float]


def load_ground_truth(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("ground_truth.json must contain a list of scenario objects")
    return data


def greedy_match(expected: list[float], detected: list[float], tolerance_s: float) -> MatchResult:
    expected = sorted(float(x) for x in expected)
    detected = sorted(float(x) for x in detected)
    used = [False] * len(detected)
    matched_pairs: list[tuple[float, float]] = []
    unmatched_expected: list[float] = []

    for exp_t in expected:
        best_idx = None
        best_err = None
        for i, det_t in enumerate(detected):
            if used[i]:
                continue
            err = abs(det_t - exp_t)
            if err <= tolerance_s and (best_err is None or err < best_err):
                best_err = err
                best_idx = i
        if best_idx is None:
            unmatched_expected.append(exp_t)
        else:
            used[best_idx] = True
            matched_pairs.append((exp_t, detected[best_idx]))

    unmatched_detected = [detected[i] for i, was_used in enumerate(used) if not was_used]
    tp = len(matched_pairs)
    fp = len(unmatched_detected)
    fn = len(unmatched_expected)
    return MatchResult(tp, fp, fn, matched_pairs, unmatched_expected, unmatched_detected)


def run_detector(
    wav_path: Path,
    rof_cmd: str,
    environment: str,
    sensitivity: float,
    min_separation_ms: int,
    echo_window_ms: int,
    burst_gap_ms: int,
) -> dict[str, Any]:
    rof_exe = shutil.which(rof_cmd)
    if rof_exe is None:
        raise FileNotFoundError(f"Could not find CLI executable '{rof_cmd}' on PATH")

    with tempfile.TemporaryDirectory(prefix="rof_batch_") as tmpdir:
        out_path = Path(tmpdir) / f"{wav_path.stem}.json"
        cmd = [
            rof_exe,
            str(wav_path),
            "--out",
            str(out_path),
            "--environment",
            environment,
            "--sensitivity",
            str(sensitivity),
            "--min-separation-ms",
            str(min_separation_ms),
            "--echo-window-ms",
            str(echo_window_ms),
            "--burst-gap-ms",
            str(burst_gap_ms),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Detector failed for {wav_path.name}\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )
        return json.loads(out_path.read_text(encoding="utf-8"))


def safe_pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    events = report.get("events", []) or []
    rof = report.get("rof", {}) or {}
    bursts = report.get("bursts", []) or []
    return {
        "detected_times": [float(e["t"]) for e in events if "t" in e],
        "detected_count": len(events),
        "avg_rpm": rof.get("avg_rpm"),
        "max_rpm": rof.get("max_rpm"),
        "burst_count": len(bursts),
    }


def fmt_ms(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    return f"{seconds * 1000.0:.1f}"


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "Scenario",
        "Exp",
        "Det",
        "TP",
        "FP",
        "FN",
        "Prec%",
        "Rec%",
        "F1%",
        "MeanErrMs",
        "Avg RPM",
        "Bursts",
        "Status",
    ]
    widths = {h: len(h) for h in headers}
    for r in rows:
        widths["Scenario"] = max(widths["Scenario"], len(str(r["scenario"])))
        widths["Exp"] = max(widths["Exp"], len(str(r["expected_count"])))
        widths["Det"] = max(widths["Det"], len(str(r["detected_count"])))
        widths["TP"] = max(widths["TP"], len(str(r["tp"])))
        widths["FP"] = max(widths["FP"], len(str(r["fp"])))
        widths["FN"] = max(widths["FN"], len(str(r["fn"])))
        widths["Prec%"] = max(widths["Prec%"], len(f"{r['precision_pct']:.1f}"))
        widths["Rec%"] = max(widths["Rec%"], len(f"{r['recall_pct']:.1f}"))
        widths["F1%"] = max(widths["F1%"], len(f"{r['f1_pct']:.1f}"))
        widths["MeanErrMs"] = max(widths["MeanErrMs"], len(fmt_ms(r["mean_abs_err_s"])))
        widths["Avg RPM"] = max(widths["Avg RPM"], len("-" if r["avg_rpm"] is None else f"{r['avg_rpm']:.1f}"))
        widths["Bursts"] = max(widths["Bursts"], len(str(r["burst_count"])))
        widths["Status"] = max(widths["Status"], len(r["status"]))

    def render_row(cells: dict[str, str]) -> str:
        return "  ".join(str(cells[h]).rjust(widths[h]) if h != "Scenario" else str(cells[h]).ljust(widths[h]) for h in headers)

    print(render_row({h: h for h in headers}))
    print(render_row({h: "-" * widths[h] for h in headers}))
    for r in rows:
        print(
            render_row(
                {
                    "Scenario": r["scenario"],
                    "Exp": str(r["expected_count"]),
                    "Det": str(r["detected_count"]),
                    "TP": str(r["tp"]),
                    "FP": str(r["fp"]),
                    "FN": str(r["fn"]),
                    "Prec%": f"{r['precision_pct']:.1f}",
                    "Rec%": f"{r['recall_pct']:.1f}",
                    "F1%": f"{r['f1_pct']:.1f}",
                    "MeanErrMs": fmt_ms(r["mean_abs_err_s"]),
                    "Avg RPM": "-" if r["avg_rpm"] is None else f"{r['avg_rpm']:.1f}",
                    "Bursts": str(r["burst_count"]),
                    "Status": r["status"],
                }
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-evaluate ShotClock-AI results against ground_truth.json")
    parser.add_argument("--audio-dir", default="test_audio_v2", help="Directory containing generated WAV files")
    parser.add_argument("--truth", default="test_audio_v2/ground_truth.json", help="Path to ground_truth.json")
    parser.add_argument("--rof-cmd", default="rof", help="CLI executable to invoke")
    parser.add_argument("--environment", default="auto", choices=["auto", "indoor", "outdoor"])
    parser.add_argument("--sensitivity", type=float, default=0.5)
    parser.add_argument("--min-separation-ms", type=int, default=50)
    parser.add_argument("--echo-window-ms", type=int, default=45)
    parser.add_argument("--burst-gap-ms", type=int, default=250)
    parser.add_argument("--match-tolerance-ms", type=float, default=35.0, help="Max timing error for a true-positive match")
    parser.add_argument("--save-json", default="batch_analysis_v2_results.json", help="Where to save the detailed evaluation JSON")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    truth_path = Path(args.truth)
    if not audio_dir.exists():
        print(f"Audio directory not found: {audio_dir}", file=sys.stderr)
        return 2
    if not truth_path.exists():
        print(f"Ground-truth file not found: {truth_path}", file=sys.stderr)
        return 2

    truth_items = load_ground_truth(truth_path)
    scenario_rows: list[dict[str, Any]] = []
    detailed: list[dict[str, Any]] = []

    print(f"🚀 Starting scored batch analysis of {len(truth_items)} files...\n")

    for item in truth_items:
        wav_path = audio_dir / item["file"]
        if not wav_path.exists():
            print(f"Skipping missing WAV: {wav_path.name}")
            continue

        report = run_detector(
            wav_path=wav_path,
            rof_cmd=args.rof_cmd,
            environment=args.environment,
            sensitivity=args.sensitivity,
            min_separation_ms=args.min_separation_ms,
            echo_window_ms=args.echo_window_ms,
            burst_gap_ms=args.burst_gap_ms,
        )
        summary = summarize_report(report)
        expected = [float(x) for x in item.get("expected_primary_shots", [])]
        detected = summary["detected_times"]
        match = greedy_match(expected, detected, tolerance_s=args.match_tolerance_ms / 1000.0)

        precision = match.tp / (match.tp + match.fp) if (match.tp + match.fp) else 0.0
        recall = match.tp / (match.tp + match.fn) if (match.tp + match.fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        mean_abs_err = mean(abs(det - exp) for exp, det in match.matched_pairs) if match.matched_pairs else None

        if match.fp == 0 and match.fn == 0:
            status = "✅ Exact"
        elif recall >= 0.9 and precision >= 0.9:
            status = "🟨 Close"
        else:
            status = "❌ Needs work"

        row = {
            "scenario": wav_path.stem,
            "expected_count": len(expected),
            "detected_count": summary["detected_count"],
            "tp": match.tp,
            "fp": match.fp,
            "fn": match.fn,
            "precision_pct": precision * 100.0,
            "recall_pct": recall * 100.0,
            "f1_pct": f1 * 100.0,
            "mean_abs_err_s": mean_abs_err,
            "avg_rpm": summary["avg_rpm"],
            "max_rpm": summary["max_rpm"],
            "burst_count": summary["burst_count"],
            "status": status,
        }
        scenario_rows.append(row)

        detailed.append(
            {
                "scenario": wav_path.name,
                "expected_primary_shots": expected,
                "detected_times": detected,
                "metrics": row,
                "matched_pairs": [
                    {
                        "expected_time_s": exp,
                        "detected_time_s": det,
                        "abs_error_ms": abs(det - exp) * 1000.0,
                    }
                    for exp, det in match.matched_pairs
                ],
                "missed_expected_times_s": match.unmatched_expected,
                "extra_detected_times_s": match.unmatched_detected,
                "raw_report": report,
            }
        )

        print(f"Finished: {wav_path.name}")

    print("\n--- SCORED BATCH SUMMARY REPORT ---")
    print_table(scenario_rows)

    total_tp = sum(r["tp"] for r in scenario_rows)
    total_fp = sum(r["fp"] for r in scenario_rows)
    total_fn = sum(r["fn"] for r in scenario_rows)
    total_expected = sum(r["expected_count"] for r in scenario_rows)
    total_detected = sum(r["detected_count"] for r in scenario_rows)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print("\n--- OVERALL ---")
    print(f"Expected primary shots: {total_expected}")
    print(f"Detected events:         {total_detected}")
    print(f"True positives:          {total_tp}")
    print(f"False positives:         {total_fp}")
    print(f"False negatives:         {total_fn}")
    print(f"Precision:               {precision * 100.0:.2f}%")
    print(f"Recall:                  {recall * 100.0:.2f}%")
    print(f"F1 score:                {f1 * 100.0:.2f}%")

    payload = {
        "params": {
            "audio_dir": str(audio_dir),
            "truth": str(truth_path),
            "rof_cmd": args.rof_cmd,
            "environment": args.environment,
            "sensitivity": args.sensitivity,
            "min_separation_ms": args.min_separation_ms,
            "echo_window_ms": args.echo_window_ms,
            "burst_gap_ms": args.burst_gap_ms,
            "match_tolerance_ms": args.match_tolerance_ms,
        },
        "overall": {
            "expected_primary_shots": total_expected,
            "detected_events": total_detected,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision_pct": precision * 100.0,
            "recall_pct": recall * 100.0,
            "f1_pct": f1 * 100.0,
        },
        "scenarios": detailed,
    }
    out_path = Path(args.save_json)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved detailed report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
