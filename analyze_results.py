"""
Utilities for autonomous experiment bookkeeping.

Features:
- Parse training summary metrics from a run log.
- Append a new row to results.tsv with consistent formatting.
- Print a compact leaderboard and trend stats from existing results.tsv.

Examples:
    uv run analyze_results.py parse --log run.log
    uv run analyze_results.py append --log run.log --description "baseline" --status keep
    uv run analyze_results.py leaderboard --top 10
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

RESULTS_HEADER = ["commit", "val_bpb", "memory_gb", "status", "description"]
SUMMARY_KEYS = ("val_bpb", "peak_vram_mb", "training_seconds", "num_steps")


@dataclass
class RunSummary:
    val_bpb: float
    peak_vram_mb: float
    training_seconds: float | None = None
    num_steps: int | None = None

    @property
    def memory_gb(self) -> float:
        return round(self.peak_vram_mb / 1024.0, 1)


def parse_run_log(log_path: Path) -> RunSummary:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    values: dict[str, str] = {}
    for key in SUMMARY_KEYS:
        m = re.search(rf"^{re.escape(key)}:\s*([^\n]+)$", text, flags=re.MULTILINE)
        if m:
            values[key] = m.group(1).strip()

    if "val_bpb" not in values or "peak_vram_mb" not in values:
        raise ValueError(
            f"Could not find required keys in {log_path}: val_bpb and peak_vram_mb."
        )

    return RunSummary(
        val_bpb=float(values["val_bpb"]),
        peak_vram_mb=float(values["peak_vram_mb"]),
        training_seconds=float(values["training_seconds"]) if "training_seconds" in values else None,
        num_steps=int(float(values["num_steps"])) if "num_steps" in values else None,
    )


def get_short_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def ensure_results_file(path: Path) -> None:
    if not path.exists():
        path.write_text("\t".join(RESULTS_HEADER) + "\n", encoding="utf-8")
        return

    first_line = path.read_text(encoding="utf-8").splitlines()[:1]
    header = first_line[0].split("\t") if first_line else []
    if header != RESULTS_HEADER:
        raise ValueError(
            f"Unexpected header in {path}: {header!r} (expected {RESULTS_HEADER!r})"
        )


def append_result(
    path: Path,
    summary: RunSummary | None,
    status: str,
    description: str,
    commit: str,
) -> None:
    ensure_results_file(path)
    if "\t" in description:
        raise ValueError("Description cannot contain tab characters.")

    val_bpb = f"{summary.val_bpb:.6f}" if summary is not None else ""
    memory_gb = f"{summary.memory_gb:.1f}" if summary is not None else ""

    row = [
        commit,
        val_bpb,
        memory_gb,
        status,
        description,
    ]
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(row)


def read_results(path: Path) -> list[dict[str, str]]:
    ensure_results_file(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    return rows


def fmt_rows(rows: Iterable[dict[str, str]]) -> str:
    rows = list(rows)
    if not rows:
        return "(no rows)"
    cols = RESULTS_HEADER
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    lines = []
    lines.append("  ".join(c.ljust(widths[c]) for c in cols))
    lines.append("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        lines.append("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    return "\n".join(lines)


def leaderboard(path: Path, top: int) -> str:
    rows = [r for r in read_results(path) if r.get("status") == "keep"]
    rows.sort(key=lambda r: float(r["val_bpb"]))
    return fmt_rows(rows[:top])


def trend(path: Path) -> str:
    rows = [r for r in read_results(path) if r.get("status") == "keep"]
    if not rows:
        return "No kept runs yet."

    best = min(rows, key=lambda r: float(r["val_bpb"]))
    first = rows[0]
    delta = float(best["val_bpb"]) - float(first["val_bpb"])
    direction = "improved" if delta < 0 else "worsened"
    return (
        f"kept_runs={len(rows)} | best={best['val_bpb']} ({best['commit']}) | "
        f"start={first['val_bpb']} -> {direction} by {delta:.6f} bpb"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Autoresearch results utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_parse = sub.add_parser("parse", help="Parse summary metrics from a log")
    p_parse.add_argument("--log", type=Path, default=Path("run.log"))

    p_append = sub.add_parser("append", help="Append one run to results.tsv")
    p_append.add_argument("--log", type=Path, default=Path("run.log"))
    p_append.add_argument("--results", type=Path, default=Path("results.tsv"))
    p_append.add_argument("--status", choices=["keep", "discard", "crash"], required=True)
    p_append.add_argument("--description", required=True)
    p_append.add_argument("--commit", default=None, help="Defaults to current HEAD short hash")

    p_lead = sub.add_parser("leaderboard", help="Print best kept runs")
    p_lead.add_argument("--results", type=Path, default=Path("results.tsv"))
    p_lead.add_argument("--top", type=int, default=10)

    p_trend = sub.add_parser("trend", help="Print trend summary over kept runs")
    p_trend.add_argument("--results", type=Path, default=Path("results.tsv"))

    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.cmd == "parse":
        s = parse_run_log(args.log)
        print(f"val_bpb={s.val_bpb:.6f}")
        print(f"peak_vram_mb={s.peak_vram_mb:.1f}")
        print(f"memory_gb={s.memory_gb:.1f}")
        if s.training_seconds is not None:
            print(f"training_seconds={s.training_seconds:.1f}")
        if s.num_steps is not None:
            print(f"num_steps={s.num_steps}")
        return

    if args.cmd == "append":
        commit = args.commit or get_short_commit()
        s: RunSummary | None = None
        if args.status != "crash":
            s = parse_run_log(args.log)

        append_result(args.results, s, args.status, args.description, commit)
        if s is None:
            print(f"Appended {commit} | status=crash | metrics=unavailable")
        else:
            print(f"Appended {commit} | val_bpb={s.val_bpb:.6f} | mem={s.memory_gb:.1f} GB")
        return

    if args.cmd == "leaderboard":
        print(leaderboard(args.results, args.top))
        return

    if args.cmd == "trend":
        print(trend(args.results))
        return

    raise RuntimeError(f"Unhandled command: {args.cmd}")


if __name__ == "__main__":
    main()
