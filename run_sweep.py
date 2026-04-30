"""
Run multiple autoresearch experiments with environment-variable overrides.

Usage:
    uv run run_sweep.py --spec sweep.example.json
    uv run run_sweep.py --spec my_sweep.json --max-runs 5
    uv run run_sweep.py --spec my_sweep.json --autostop --loop \
      --eval-cmd "python external_eval.py --run-dir {run_dir} --run-name {run_name}"
"""

import argparse
import json
import os
import shlex
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
    return {"run_name": name, "run_dir": env["AUTORESEARCH_RUN_DIR"], "index": index}


def _parse_eval_stdout(stdout: str):
    text = stdout.strip()
    if not text:
        raise ValueError("External evaluator returned empty stdout.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            raise ValueError("External evaluator stdout had no parseable content.")
        payload = json.loads(lines[-1])
    hle = payload.get("hle")
    swebench_pro = payload.get("swebench_pro")
    if hle is None or swebench_pro is None:
        raise ValueError("External evaluator output JSON must contain keys: hle, swebench_pro.")
    return float(hle), float(swebench_pro), payload


def run_external_eval(eval_cmd_template: str, run_meta: dict):
    cmd = eval_cmd_template.format(
        run_dir=run_meta["run_dir"],
        run_name=run_meta["run_name"],
        run_index=run_meta["index"],
    )
    print(f"[eval] running: {cmd}")
    proc = subprocess.run(
        shlex.split(cmd),
        check=True,
        text=True,
        capture_output=True,
    )
    if proc.stderr.strip():
        print(f"[eval] stderr:\n{proc.stderr.strip()}")
    hle, swebench_pro, payload = _parse_eval_stdout(proc.stdout)
    return hle, swebench_pro, payload


def main():
    parser = argparse.ArgumentParser(description="Execute a sequence of autoresearch runs.")
    parser.add_argument("--spec", required=True, help="Path to JSON sweep spec.")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap for number of runs.")
    parser.add_argument("--autostop", action="store_true", help="Stop only when both benchmark targets are met.")
    parser.add_argument("--loop", action="store_true", help="Loop over sweep spec repeatedly (use with --autostop).")
    parser.add_argument("--eval-cmd", default=None, help=(
        "External evaluator command template. Must print JSON with keys "
        "'hle' and 'swebench_pro'. Placeholders: {run_dir}, {run_name}, {run_index}."
    ))
    parser.add_argument("--max-total-runs", type=int, default=200, help="Safety cap for total runs in loop mode.")
    args = parser.parse_args()

    spec_path = Path(args.spec)
    payload = load_spec(spec_path)
    runs = payload["runs"]
    if args.max_runs is not None:
        runs = runs[: args.max_runs]

    if not runs:
        raise SystemExit("No runs to execute after applying filters.")

    if args.autostop and not args.eval_cmd:
        raise SystemExit("--autostop requires --eval-cmd so benchmark scores can be checked externally.")

    base_env = os.environ.copy()
    total_runs = 0
    sweep_round = 0
    while True:
        sweep_round += 1
        print(f"\n========== Sweep round {sweep_round} ==========")
        for i, run_cfg in enumerate(runs, start=1):
            total_runs += 1
            run_meta = run_once(total_runs, run_cfg, base_env)
            if args.autostop:
                hle, swebench_pro, payload = run_external_eval(args.eval_cmd, run_meta)
                result = evaluate_scores(hle, swebench_pro)
                print(
                    f"[eval] hle={hle:.2f}% (target {HLE_TARGET:.1f}%), "
                    f"swebench_pro={swebench_pro:.2f}% (target {SWEBENCH_PRO_TARGET:.1f}%)"
                )
                if result["ok"]:
                    print("[autostop] Targets met. Stopping sweep.")
                    return
                print(
                    "[autostop] Targets not met yet. "
                    f"Gaps: HLE={result['hle_gap']:.2f}pp, SWE-Bench Pro={result['swebench_pro_gap']:.2f}pp"
                )
                print(f"[autostop] Raw evaluator payload: {json.dumps(payload, ensure_ascii=False)}")

            if args.max_total_runs is not None and total_runs >= args.max_total_runs:
                raise SystemExit(
                    f"Reached --max-total-runs={args.max_total_runs} before meeting benchmark targets."
                )

        if not args.loop:
            break

    if args.autostop:
        raise SystemExit("Sweep completed but benchmark targets were not met.")


if __name__ == "__main__":
    main()
