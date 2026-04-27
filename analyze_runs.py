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
        # Keep metadata JSON-serializable so `--json` output never fails.
        payload["_path"] = str(path)
        runs.append(payload)
    return runs


def print_table(rows):
    headers = ["rank", "val_bpb", "steps", "mfu%", "tokens_M", "depth", "window", "commit", "file"]
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
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    runs = load_runs(run_dir)
    if not runs:
        raise SystemExit(f"No run_*.json files found in {run_dir}")

    ranked = sorted(runs, key=lambda r: r["metrics"]["val_bpb"])
    top = ranked[: args.limit]

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
            str(metrics["num_steps"]),
            f'{metrics["mfu_percent"]:.2f}',
            f'{metrics["total_tokens_M"]:.1f}',
            str(model["depth"]),
            str(model["window_pattern"]),
            str(run.get("git_commit", "unknown")),
            Path(run["_path"]).name,
        ])

    print(f"Loaded {len(runs)} runs from {run_dir}")
    print_table(rows)


if __name__ == "__main__":
    main()
