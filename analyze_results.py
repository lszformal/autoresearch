"""Summarize autoresearch experiment logs from results.tsv.

Usage:
    uv run python analyze_results.py
    uv run python analyze_results.py --path results.tsv --top 5
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


@dataclass
class ResultRow:
    commit: str
    val_bpb: float
    memory_gb: float
    status: str
    description: str


def load_results(path: Path) -> list[ResultRow]:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    rows: list[ResultRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        expected = {"commit", "val_bpb", "memory_gb", "status", "description"}
        if not reader.fieldnames or set(reader.fieldnames) != expected:
            raise ValueError(
                "results.tsv has unexpected columns. "
                f"Expected {sorted(expected)}, got {reader.fieldnames}"
            )

        for row in reader:
            rows.append(
                ResultRow(
                    commit=row["commit"],
                    val_bpb=float(row["val_bpb"]),
                    memory_gb=float(row["memory_gb"]),
                    status=row["status"].strip().lower(),
                    description=row["description"].strip(),
                )
            )
    return rows


def summarize(rows: list[ResultRow], top_n: int) -> str:
    if not rows:
        return "No experiment rows found in results.tsv yet."

    keeps = [r for r in rows if r.status == "keep"]
    discards = [r for r in rows if r.status == "discard"]
    crashes = [r for r in rows if r.status == "crash"]

    lines: list[str] = []
    lines.append(f"Total experiments: {len(rows)}")
    lines.append(
        "Status counts: "
        f"keep={len(keeps)}, discard={len(discards)}, crash={len(crashes)}"
    )

    if crashes:
        crash_rate = len(crashes) / len(rows)
        lines.append(f"Crash rate: {crash_rate:.1%}")

    successful = [r for r in rows if r.status in {"keep", "discard"} and r.val_bpb > 0]
    if successful:
        best = min(successful, key=lambda r: r.val_bpb)
        avg_bpb = mean(r.val_bpb for r in successful)
        avg_mem = mean(r.memory_gb for r in successful)
        lines.append(
            f"Best val_bpb: {best.val_bpb:.6f} ({best.commit}, {best.description})"
        )
        lines.append(f"Average val_bpb (non-crash): {avg_bpb:.6f}")
        lines.append(f"Average memory_gb (non-crash): {avg_mem:.1f}")

        ranked = sorted(successful, key=lambda r: r.val_bpb)[:top_n]
        lines.append("")
        lines.append(f"Top {len(ranked)} runs by val_bpb:")
        for i, r in enumerate(ranked, start=1):
            lines.append(
                f"{i:>2}. {r.val_bpb:.6f} | {r.memory_gb:>4.1f} GB | {r.status:<7} "
                f"| {r.commit} | {r.description}"
            )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("results.tsv"),
        help="Path to tab-separated results file (default: results.tsv)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of best runs to print (default: 10)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_results(args.path)
    print(summarize(rows, top_n=max(1, args.top)))


if __name__ == "__main__":
    main()
