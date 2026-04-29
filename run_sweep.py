"""
Run multiple autoresearch experiments with environment-variable overrides.

Usage:
    uv run run_sweep.py --spec sweep.example.json
    uv run run_sweep.py --spec my_sweep.json --max-runs 5
    uv run run_sweep.py --spec my_sweep.json --autostop --eval-command "python external_eval.py --out eval.json" --eval-json eval.json
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from benchmark_gate import HLE_TARGET, SWEBENCH_PRO_TARGET, evaluate_scores


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
    return env["AUTORESEARCH_RUN_DIR"]


def resolve_autostop_config(payload, args):
    cfg = payload.get("autostop", {})
    enabled = args.autostop or bool(cfg.get("enabled", False))
    eval_command = args.eval_command or cfg.get("eval_command")
    eval_json = args.eval_json or cfg.get("eval_json")
    max_total_runs = args.max_total_runs if args.max_total_runs is not None else cfg.get("max_total_runs")
    cycle_runs = bool(cfg.get("cycle_runs", True))
    return {
        "enabled": enabled,
        "eval_command": eval_command,
        "eval_json": eval_json,
        "max_total_runs": max_total_runs,
        "cycle_runs": cycle_runs,
    }


def run_external_eval(eval_command: str, env: dict):
    subprocess.run(eval_command, env=env, shell=True, check=True)


def load_eval_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    hle = payload.get("hle")
    swe = payload.get("swebench_pro")
    if hle is None or swe is None:
        raise ValueError(f"Eval JSON must contain 'hle' and 'swebench_pro'. File: {path}")
    return float(hle), float(swe)


def main():
    parser = argparse.ArgumentParser(description="Execute a sequence of autoresearch runs.")
    parser.add_argument("--spec", required=True, help="Path to JSON sweep spec.")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap for number of runs.")
    parser.add_argument("--autostop", action="store_true", help="Enable autostop loop until benchmark targets are met.")
    parser.add_argument("--eval-command", default=None, help="External eval command executed after each train run in autostop mode.")
    parser.add_argument("--eval-json", default=None, help="Path to JSON output from eval command containing keys: hle, swebench_pro.")
    parser.add_argument("--max-total-runs", type=int, default=None, help="Hard cap for autostop mode to prevent infinite looping.")
    args = parser.parse_args()

    spec_path = Path(args.spec)
    payload = load_spec(spec_path)
    runs = payload["runs"][:]
    if args.max_runs is not None and not args.autostop:
        runs = runs[: args.max_runs]

    if not runs:
        raise SystemExit("No runs to execute after applying filters.")

    autostop = resolve_autostop_config(payload, args)
    eval_json_path = None
    if autostop["eval_json"] is not None:
        eval_json_path = (spec_path.parent / autostop["eval_json"]).resolve()

    if autostop["enabled"]:
        if not autostop["eval_command"] or eval_json_path is None:
            raise SystemExit("Autostop mode requires eval command and eval json path.")
        print("Autostop mode enabled.")
        print(f"Targets: HLE >= {HLE_TARGET}, SWE-Bench Pro >= {SWEBENCH_PRO_TARGET}")

    base_env = os.environ.copy()
    run_counter = 0
    while True:
        for run_cfg in runs:
            run_counter += 1
            run_dir = run_once(run_counter, run_cfg, base_env)
            if not autostop["enabled"]:
                continue

            if autostop["max_total_runs"] is not None and run_counter > int(autostop["max_total_runs"]):
                raise SystemExit(f"Reached max_total_runs={autostop['max_total_runs']} without meeting targets.")

            eval_env = base_env.copy()
            eval_env["AUTORESEARCH_LAST_RUN_DIR"] = str(run_dir)
            eval_env["AUTORESEARCH_LAST_RUN_INDEX"] = str(run_counter)
            print(f"[autostop] running eval command: {autostop['eval_command']}")
            run_external_eval(autostop["eval_command"], env=eval_env)
            hle, swe = load_eval_json(eval_json_path)
            evaluation = evaluate_scores(hle, swe)
            print(f"[autostop] HLE={hle:.2f}% SWE-BenchPro={swe:.2f}%")
            if evaluation["ok"]:
                print("[autostop] Target reached. Stopping sweep.")
                return
            print("[autostop] Target not yet reached; continuing.")

        if not autostop["enabled"] or not autostop["cycle_runs"]:
            break


if __name__ == "__main__":
    main()
