"""
Run multiple autoresearch experiments with environment-variable overrides.

Usage:
    uv run run_sweep.py --spec sweep.example.json
    uv run run_sweep.py --spec my_sweep.json --max-runs 5
    uv run run_sweep.py --spec my_sweep.json --autostop --eval-cmd "python external_eval.py --run {latest_json}"
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from benchmark_gate import HLE_TARGET, SWEBENCH_PRO_TARGET

def load_spec(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "runs" not in payload or not isinstance(payload["runs"], list):
        raise ValueError("Sweep spec must contain a top-level 'runs' list.")
    return payload


def save_autostop_history(history_path: Path, payload: dict):
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True))
        f.write("\n")


def run_external_eval(eval_cmd_template: str, latest_json: Path, run_dir: str, run_name: str, run_index: int):
    cmd = eval_cmd_template.format(
        latest_json=str(latest_json),
        run_dir=run_dir,
        run_name=run_name,
        run_index=run_index,
    )
    print(f"external_eval_cmd: {cmd}")
    proc = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"External evaluation failed (exit={proc.returncode}).\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("External evaluation produced empty stdout; expected JSON on stdout.")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as e:
        raise RuntimeError(
            "Failed to parse external evaluation JSON from last stdout line.\n"
            f"Last line: {lines[-1]!r}"
        ) from e

    if "hle" not in payload or "swebench_pro" not in payload:
        raise RuntimeError("External evaluation JSON must contain keys: hle, swebench_pro")
    return float(payload["hle"]), float(payload["swebench_pro"]), payload


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
    latest_json = Path(env["AUTORESEARCH_RUN_DIR"]) / "latest.json"
    return {
        "name": name,
        "run_dir": env["AUTORESEARCH_RUN_DIR"],
        "latest_json": latest_json,
    }


def main():
    parser = argparse.ArgumentParser(description="Execute a sequence of autoresearch runs.")
    parser.add_argument("--spec", required=True, help="Path to JSON sweep spec.")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap for number of runs.")
    parser.add_argument("--autostop", action="store_true", help="Loop sweep until benchmark targets are met.")
    parser.add_argument(
        "--eval-cmd",
        default=None,
        help=(
            "External evaluator command template (required with --autostop). "
            "Must print JSON with keys {hle, swebench_pro} on the last stdout line. "
            "Template fields: {latest_json}, {run_dir}, {run_name}, {run_index}"
        ),
    )
    parser.add_argument("--max-cycles", type=int, default=None, help="Optional cap on sweep cycles in --autostop mode.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Sleep between runs (seconds).")
    parser.add_argument(
        "--history-file",
        default="runs/autostop_history.jsonl",
        help="Where to append autostop evaluation records.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs and exit without training.")
    args = parser.parse_args()

    if args.autostop and not args.eval_cmd:
        raise SystemExit("--autostop requires --eval-cmd.")

    spec_path = Path(args.spec)
    payload = load_spec(spec_path)
    runs = payload["runs"]
    if args.max_runs is not None:
        runs = runs[: args.max_runs]

    if not runs:
        raise SystemExit("No runs to execute after applying filters.")

    base_env = os.environ.copy()
    if args.dry_run:
        print("Dry run plan:")
        for i, run_cfg in enumerate(runs, start=1):
            print(f"  [{i}] {run_cfg.get('name', f'run_{i:03d}')} run_dir={run_cfg.get('run_dir', 'runs')}")
        return

    if not args.autostop:
        for i, run_cfg in enumerate(runs, start=1):
            run_once(i, run_cfg, base_env)
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
        return

    history_path = Path(args.history_file)
    cycle = 0
    while True:
        cycle += 1
        print(f"\n=== autostop cycle {cycle} ===")
        for i, run_cfg in enumerate(runs, start=1):
            result = run_once(i, run_cfg, base_env)
            hle, swe, eval_payload = run_external_eval(
                eval_cmd_template=args.eval_cmd,
                latest_json=result["latest_json"],
                run_dir=result["run_dir"],
                run_name=result["name"],
                run_index=i,
            )
            passed = (hle >= HLE_TARGET) and (swe >= SWEBENCH_PRO_TARGET)
            print(
                f"autostop_eval: run={result['name']} cycle={cycle} "
                f"hle={hle:.2f}/{HLE_TARGET:.1f} swebench_pro={swe:.2f}/{SWEBENCH_PRO_TARGET:.1f} "
                f"passed={passed}"
            )
            save_autostop_history(history_path, {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "cycle": cycle,
                "run_index": i,
                "run_name": result["name"],
                "run_dir": result["run_dir"],
                "latest_json": str(result["latest_json"]),
                "hle": hle,
                "swebench_pro": swe,
                "passed": passed,
                "targets": {"hle": HLE_TARGET, "swebench_pro": SWEBENCH_PRO_TARGET},
                "eval_payload": eval_payload,
            })
            if passed:
                print("AUTOSTOP: target achieved in external evaluation. Stopping sweep.")
                return
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
        if args.max_cycles is not None and cycle >= args.max_cycles:
            raise SystemExit(
                f"AUTOSTOP: reached max cycles ({args.max_cycles}) without meeting targets "
                f"(HLE>={HLE_TARGET}, SWE-Bench Pro>={SWEBENCH_PRO_TARGET})."
            )


if __name__ == "__main__":
    main()
