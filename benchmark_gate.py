"""
Fail-fast gate for target benchmark thresholds.

Usage:
    uv run benchmark_gate.py --metrics-file eval_metrics.json

Expected JSON fields:
    {
      "humanity_final_exam_accuracy": 64.9,
      "swe_bench_pro_accuracy": 78.2
    }
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Check benchmark thresholds for release gating.")
    parser.add_argument("--metrics-file", required=True, help="Path to JSON metrics file.")
    parser.add_argument("--hfe-threshold", type=float, default=64.7, help="Humanity Final Exam accuracy threshold (%%).")
    parser.add_argument("--swe-threshold", type=float, default=77.8, help="SWE-Bench Pro accuracy threshold (%%).")
    args = parser.parse_args()

    path = Path(args.metrics_file)
    with open(path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    hfe = float(metrics.get("humanity_final_exam_accuracy", -1))
    swe = float(metrics.get("swe_bench_pro_accuracy", -1))

    print(f"Humanity Final Exam accuracy: {hfe:.2f}% (target >= {args.hfe_threshold:.2f}%)")
    print(f"SWE-Bench Pro accuracy:      {swe:.2f}% (target >= {args.swe_threshold:.2f}%)")

    failures = []
    if hfe < args.hfe_threshold:
        failures.append("humanity_final_exam_accuracy below threshold")
    if swe < args.swe_threshold:
        failures.append("swe_bench_pro_accuracy below threshold")

    if failures:
        raise SystemExit("BENCHMARK GATE FAILED: " + "; ".join(failures))
    print("BENCHMARK GATE PASSED")


if __name__ == "__main__":
    main()
