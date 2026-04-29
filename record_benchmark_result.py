"""
Attach benchmark results (HLE + SWE-Bench Pro) to a run summary JSON.

Usage:
  uv run record_benchmark_result.py \
    --run-json runs/latest.json \
    --hle 65.1 \
    --swebench-pro 78.2
"""

import argparse
import json
from pathlib import Path

from benchmark_targets import HLE_TARGET_ACCURACY, SWEBENCH_PRO_TARGET_ACCURACY


def compute_eval(hle: float, swe: float):
    hle_gap = hle - HLE_TARGET_ACCURACY
    swe_gap = swe - SWEBENCH_PRO_TARGET_ACCURACY
    passed = hle_gap >= 0 and swe_gap >= 0
    # Smaller is better; <= 0 means target reached on both benchmarks.
    target_gap = max(0.0, -hle_gap) + max(0.0, -swe_gap)
    return {
        "hle_accuracy_percent": hle,
        "swebench_pro_accuracy_percent": swe,
        "hle_target_percent": HLE_TARGET_ACCURACY,
        "swebench_pro_target_percent": SWEBENCH_PRO_TARGET_ACCURACY,
        "hle_gap_percent": hle_gap,
        "swebench_pro_gap_percent": swe_gap,
        "all_targets_met": passed,
        "combined_target_gap_percent": target_gap,
    }


def main():
    parser = argparse.ArgumentParser(description="Record benchmark metrics into run summary JSON.")
    parser.add_argument("--run-json", required=True, help="Path to run summary JSON (e.g. runs/latest.json).")
    parser.add_argument("--hle", type=float, required=True, help="Humanity's Last Exam accuracy (%%) for this run.")
    parser.add_argument("--swebench-pro", type=float, required=True, help="SWE-Bench Pro accuracy (%%) for this run.")
    parser.add_argument("--in-place", action="store_true", help="Write updates directly to input file.")
    args = parser.parse_args()

    src = Path(args.run_json)
    if not src.exists():
        raise SystemExit(f"Run JSON not found: {src}")

    with open(src, "r", encoding="utf-8") as f:
        payload = json.load(f)

    payload["benchmark_eval"] = compute_eval(args.hle, args.swebench_pro)

    dst = src if args.in_place else src.with_name(f"{src.stem}.benchmarked{src.suffix}")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    ev = payload["benchmark_eval"]
    print(f"Wrote: {dst}")
    print(
        "Targets met: "
        f"{ev['all_targets_met']} "
        f"(HLE {ev['hle_accuracy_percent']:.2f}/{ev['hle_target_percent']:.1f}, "
        f"SWE-Bench Pro {ev['swebench_pro_accuracy_percent']:.2f}/{ev['swebench_pro_target_percent']:.1f})"
    )


if __name__ == "__main__":
    main()
