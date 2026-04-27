"""
Analyze autoresearch JSON run summaries and print a leaderboard.

Usage:
    uv run analyze_runs.py
    uv run analyze_runs.py --limit 20 --json
"""

import argparse
import json
from pathlib import Path


def load_runs(run_dir: Path):
    runs = []
    for path in sorted(run_dir.glob("run_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if "metrics" not in payload or "val_bpb" not in payload["metrics"]:
            continue
        payload["_path"] = str(path)
        runs.append(payload)
    return runs


def print_table(rows):
    headers = ["rank", "val_bpb", "delta_best%", "steps", "mfu%", "gpu", "depth", "window", "file"]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, col in enumerate(row):
            widths[i] = max(widths[i], len(col))
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * widths[i] for i in range(len(headers)))
    print(header_line)
    print(sep_line)
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def main():
    parser = argparse.ArgumentParser(description="Analyze autoresearch run summaries.")
    parser.add_argument("--run-dir", default="runs", help="Directory containing run_*.json files.")
    parser.add_argument("--limit", type=int, default=10, help="How many top runs to print.")
    parser.add_argument("--json", action="store_true", help="Print top runs in JSON format.")
    parser.add_argument("--newest", action="store_true", help="Sort by newest timestamp instead of val_bpb.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    runs = load_runs(run_dir)
    if not runs:
        raise SystemExit(f"No run_*.json files found in {run_dir}")

    if args.newest:
        ranked = sorted(runs, key=lambda r: r.get("timestamp_utc", ""), reverse=True)
    else:
        ranked = sorted(runs, key=lambda r: r["metrics"]["val_bpb"])
    top = ranked[: args.limit]
    best_bpb = min(r["metrics"]["val_bpb"] for r in runs)

    if args.json:
        print(json.dumps(top, indent=2))
        return

    rows = []
    for rank, run in enumerate(top, start=1):
        metrics = run["metrics"]
        model = run["model"]
        rows.append([
            str(rank),
            f'{metrics["val_bpb"]:.6f}',
            f'{100 * (metrics["val_bpb"] / best_bpb - 1):+.2f}',
            str(metrics["num_steps"]),
            f'{metrics["mfu_percent"]:.2f}',
            run.get("hardware", {}).get("cuda_device_name", "unknown"),
            str(model["depth"]),
            str(model["window_pattern"]),
            Path(run["_path"]).name,
        ])

    print(f"Loaded {len(runs)} runs from {run_dir}")
    print_table(rows)


if __name__ == "__main__":
    main()
