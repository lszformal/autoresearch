"""
Run multiple autoresearch experiments with environment-variable overrides.

Usage:
    uv run run_sweep.py --spec sweep.example.json
    uv run run_sweep.py --spec my_sweep.json --max-runs 5
    uv run run_sweep.py --spec my_sweep.json --autostop --eval-command "python external_eval.py --run {latest_summary} --out eval.json" --eval-result-json eval.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_HLE_TARGET = 64.7
DEFAULT_SWEBENCH_PRO_TARGET = 77.8


def load_spec(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "runs" not in payload or not isinstance(payload["runs"], list):
        raise ValueError("Sweep spec must contain a top-level 'runs' list.")
    return payload


def render_template(template: str, values: dict):
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def run_once(index: int, run_cfg: dict, base_env: dict, train_command: str | None = None):
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

    if train_command is None:
        subprocess.run([sys.executable, "train.py"], env=env, check=True)
    else:
        rendered = render_template(train_command, {
            "run_dir": env["AUTORESEARCH_RUN_DIR"],
            "run_name": name,
            "run_index": index,
        })
        subprocess.run(rendered, env=env, shell=True, check=True)

    latest_summary = Path(env["AUTORESEARCH_RUN_DIR"]) / "latest.json"
    if not latest_summary.exists():
        raise SystemExit(f"Expected latest run summary at {latest_summary}, but file is missing.")
    return {
        "run_dir": env["AUTORESEARCH_RUN_DIR"],
        "run_name": name,
        "latest_summary": latest_summary,
    }


def load_eval_scores(eval_result_json: Path):
    with open(eval_result_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    hle = payload.get("hle")
    swebench_pro = payload.get("swebench_pro")
    if hle is None or swebench_pro is None:
        raise SystemExit(
            f"Eval JSON {eval_result_json} must contain keys 'hle' and 'swebench_pro'. "
            f"Got keys: {sorted(payload.keys())}"
        )
    return float(hle), float(swebench_pro), payload


def evaluate_and_check_targets(
    run_info: dict,
    eval_command: str,
    eval_result_json: Path,
    hle_target: float,
    swebench_target: float,
):
    rendered = render_template(eval_command, {
        "run_dir": run_info["run_dir"],
        "run_name": run_info["run_name"],
        "latest_summary": str(run_info["latest_summary"]),
        "eval_json": str(eval_result_json),
    })
    subprocess.run(rendered, shell=True, check=True)
    hle, swebench_pro, payload = load_eval_scores(eval_result_json)
    ok = (hle >= hle_target) and (swebench_pro >= swebench_target)
    print(
        f"[autostop-eval] hle={hle:.2f}% (target {hle_target:.1f}%) | "
        f"swebench_pro={swebench_pro:.2f}% (target {swebench_target:.1f}%) | pass={ok}"
    )
    if not ok:
        print(
            f"[autostop-eval] gaps => "
            f"hle_gap={max(0.0, hle_target - hle):.2f}pp, "
            f"swebench_gap={max(0.0, swebench_target - swebench_pro):.2f}pp"
        )
    return ok, payload


def main():
    parser = argparse.ArgumentParser(description="Execute a sequence of autoresearch runs.")
    parser.add_argument("--spec", required=True, help="Path to JSON sweep spec.")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap for number of runs.")
    parser.add_argument("--train-command", default=None, help="Optional shell command override for each train run.")
    parser.add_argument("--autostop", action="store_true", help="Loop sweep until benchmark targets are met.")
    parser.add_argument(
        "--eval-command",
        default=None,
        help=(
            "Shell command to run external evaluation after each run. "
            "Template variables: {run_dir}, {run_name}, {latest_summary}, {eval_json}."
        ),
    )
    parser.add_argument(
        "--eval-result-json",
        default=None,
        help="Path to evaluator output JSON with keys: hle, swebench_pro.",
    )
    parser.add_argument("--hle-target", type=float, default=DEFAULT_HLE_TARGET, help="HLE target percentage.")
    parser.add_argument("--swebench-pro-target", type=float, default=DEFAULT_SWEBENCH_PRO_TARGET, help="SWE-Bench Pro target percentage.")
    parser.add_argument("--sleep-between-cycles", type=float, default=0.0, help="Optional sleep seconds between full sweep cycles in autostop mode.")
    parser.add_argument(
        "--max-total-runs",
        type=int,
        default=100,
        help=(
            "Hard cap on total launched runs in --autostop mode across all cycles. "
            "Set to a positive integer to bound compute usage."
        ),
    )
    args = parser.parse_args()

    spec_path = Path(args.spec)
    payload = load_spec(spec_path)
    runs = payload["runs"]
    if args.max_runs is not None:
        runs = runs[: args.max_runs]

    if not runs:
        raise SystemExit("No runs to execute after applying filters.")

    base_env = os.environ.copy()
    if args.autostop:
        if args.eval_command is None or args.eval_result_json is None:
            raise SystemExit("--autostop requires both --eval-command and --eval-result-json.")
        if args.max_total_runs is None or args.max_total_runs <= 0:
            raise SystemExit("--max-total-runs must be a positive integer in --autostop mode.")
        eval_result_json = Path(args.eval_result_json)
        cycle = 0
        total_runs = 0
        while True:
            cycle += 1
            print(f"\n######## AUTOSTOP SWEEP CYCLE {cycle} ########")
            for i, run_cfg in enumerate(runs, start=1):
                if total_runs >= args.max_total_runs:
                    raise SystemExit(
                        "AUTOSTOP: reached --max-total-runs limit "
                        f"({args.max_total_runs}) without meeting benchmark targets."
                    )
                run_info = run_once(i, run_cfg, base_env, train_command=args.train_command)
                total_runs += 1
                ok, _payload = evaluate_and_check_targets(
                    run_info=run_info,
                    eval_command=args.eval_command,
                    eval_result_json=eval_result_json,
                    hle_target=args.hle_target,
                    swebench_target=args.swebench_pro_target,
                )
                if ok:
                    print("\nAUTOSTOP: benchmark targets reached, stopping sweep.")
                    return
            if args.sleep_between_cycles > 0:
                print(f"Cycle completed without meeting targets; sleeping {args.sleep_between_cycles}s.")
                time.sleep(args.sleep_between_cycles)
    else:
        for i, run_cfg in enumerate(runs, start=1):
            run_once(i, run_cfg, base_env, train_command=args.train_command)


if __name__ == "__main__":
    main()
