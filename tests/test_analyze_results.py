import csv
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "analyze_results.py"


class AnalyzeResultsCrashAppendTests(unittest.TestCase):
    def test_append_crash_succeeds_without_summary_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "run.log"
            results_path = tmp_path / "results.tsv"

            # Simulate a traceback-only failed run log.
            log_path.write_text(
                "Traceback (most recent call last):\n"
                "RuntimeError: out of memory\n",
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    "python",
                    str(SCRIPT),
                    "append",
                    "--log",
                    str(log_path),
                    "--results",
                    str(results_path),
                    "--status",
                    "crash",
                    "--description",
                    "OOM during eval",
                    "--commit",
                    "abc1234",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("status=crash", proc.stdout)

            with results_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f, delimiter="\t"))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["commit"], "abc1234")
            self.assertEqual(rows[0]["status"], "crash")
            self.assertEqual(rows[0]["description"], "OOM during eval")
            self.assertEqual(rows[0]["val_bpb"], "")
            self.assertEqual(rows[0]["memory_gb"], "")


if __name__ == "__main__":
    unittest.main()
