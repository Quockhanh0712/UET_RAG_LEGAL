"""
Quick benchmark wrapper for the current research benchmark entrypoint.

Usage:
    python scripts/research/quick_test_benchmark.py
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

cmd = [
    str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"),
    str(PROJECT_ROOT / "scripts" / "research" / "run_research_benchmark_v2.py"),
    "--queries-path", str(PROJECT_ROOT / "data" / "gold" / "research" / "query_set_v1.csv"),
    "--labels-path", str(PROJECT_ROOT / "data" / "gold" / "research" / "eval_pairs_template.csv"),
    "--output-path", str(PROJECT_ROOT / "data" / "gold" / "research" / "benchmark_results_quick.csv"),
    "--max-queries", "10",
]

print("=" * 60)
print("Quick Test Benchmark")
print("=" * 60)
print(f"Running: {' '.join(cmd[1:])}")
print()

subprocess.run(cmd, check=False)
