"""
Run multiple autoresearch experiments with environment-variable overrides.

Usage:
    uv run run_sweep.py --spec sweep.example.json
    uv run run_sweep.py --spec my_sweep.json --max-runs 5
    uv run run_sweep.py --spec my_sweep.json --target-hle 64.7 --target-swepro 77.8
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def load_spec(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "runs" not in payload or not isinstance(payload["runs"], list):
        raise ValueError("Sweep spec must contain a top-level 'runs' list.")
    return payload


def run_once(index: int, run_cfg: dict, base_env: dict):
    env = base_env.copy()
    env["AUTORESEARCH_RUN_DIR"] = run_cfg.get("run_dir", "runs")
    overrides = run_cfg.get("overrides", {})
    for key, value in overrides.items():
        env[key] = str(value)

    name = run_cfg.get("name", f"run_{index:03d}")
    print(f"\n=== [{index}] {name} ===")
    if overrides:
        print("overrides:")
        for key in sorted(overrides):
            print(f"  {key}={overrides[key]}")
    else:
        print("overrides: (none)")
    print(f"run_dir: {env['AUTORESEARCH_RUN_DIR']}")

    subprocess.run([sys.executable, "train.py"], env=env, check=True)


def read_latest_metrics(run_dir: str):
    latest_path = Path(run_dir) / "latest.json"
    if not latest_path.exists():
        return {}
    with open(latest_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("metrics", {})


def main():
    parser = argparse.ArgumentParser(description="Execute a sequence of autoresearch runs.")
    parser.add_argument("--spec", required=True, help="Path to JSON sweep spec.")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap for number of runs.")
    parser.add_argument("--target-hle", type=float, default=None, help="Stop early when HLE accuracy >= this.")
    parser.add_argument("--target-swepro", type=float, default=None, help="Stop early when SWE-Bench Pro accuracy >= this.")
    args = parser.parse_args()

    spec_path = Path(args.spec)
    payload = load_spec(spec_path)
    runs = payload["runs"]
    if args.max_runs is not None:
        runs = runs[: args.max_runs]

    if not runs:
        raise SystemExit("No runs to execute after applying filters.")

    base_env = os.environ.copy()
    for i, run_cfg in enumerate(runs, start=1):
        run_once(i, run_cfg, base_env)
        run_dir = run_cfg.get("run_dir", "runs")
        metrics = read_latest_metrics(run_dir)
        hle = metrics.get("hle_accuracy")
        swe = metrics.get("swebench_pro_accuracy")
        print(f"post-run metrics: hle_accuracy={hle} swebench_pro_accuracy={swe}")
        if args.target_hle is not None and args.target_swepro is not None:
            if hle is not None and swe is not None and hle >= args.target_hle and swe >= args.target_swepro:
                print(
                    "Targets reached; stopping early "
                    f"(HLE {hle:.3f} >= {args.target_hle}, SWE-Pro {swe:.3f} >= {args.target_swepro})."
                )
                break


if __name__ == "__main__":
    main()
