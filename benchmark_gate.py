"""
Benchmark gate for target scores.

Checks whether a run meets:
- Humanity's Last Exam (HLE) >= 64.7%
- SWE-Bench Pro >= 77.8%

Usage:
  uv run benchmark_gate.py --hle 65.1 --swebench-pro 78.2
  uv run benchmark_gate.py --from-json eval_results.json
"""

import argparse
import json


HLE_TARGET = 64.7
SWEBENCH_PRO_TARGET = 77.8


def load_scores(args):
    if args.from_json is not None:
        with open(args.from_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        hle = payload.get("hle")
        swe = payload.get("swebench_pro")
    else:
        hle = args.hle
        swe = args.swebench_pro
    if hle is None or swe is None:
        raise SystemExit("Both HLE and SWE-Bench Pro scores are required.")
    return float(hle), float(swe)


def main():
    parser = argparse.ArgumentParser(description="Gate benchmark targets for autoresearch runs.")
    parser.add_argument("--hle", type=float, default=None, help="Humanity's Last Exam score (%%).")
    parser.add_argument("--swebench-pro", type=float, default=None, help="SWE-Bench Pro score (%%).")
    parser.add_argument("--from-json", default=None, help="JSON file with keys: hle, swebench_pro.")
    args = parser.parse_args()

    hle, swe = load_scores(args)
    hle_gap = HLE_TARGET - hle
    swe_gap = SWEBENCH_PRO_TARGET - swe
    ok_hle = hle >= HLE_TARGET
    ok_swe = swe >= SWEBENCH_PRO_TARGET
    ok = ok_hle and ok_swe

    print(f"HLE:          {hle:.2f}% (target {HLE_TARGET:.1f}%)")
    print(f"SWE-BenchPro: {swe:.2f}% (target {SWEBENCH_PRO_TARGET:.1f}%)")
    if ok:
        print("STATUS: PASS (both targets met)")
        raise SystemExit(0)

    print("STATUS: FAIL")
    if not ok_hle:
        print(f"  HLE gap:          {hle_gap:.2f} pp")
    if not ok_swe:
        print(f"  SWE-Bench Pro gap:{swe_gap:.2f} pp")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
