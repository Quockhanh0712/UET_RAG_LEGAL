"""
Quick test benchmark với 10 mẫu để verify system

Usage:
    python scripts/quick_test_benchmark.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Run benchmark with Vietnamese Reranker
import subprocess

cmd = [
    str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"),
    str(PROJECT_ROOT / "scripts" / "benchmark_alqac.py"),
    "--csv", "ALQAC.csv",
    "--sample", "10",
    "--use-reranker",  # Enable Vietnamese Reranker
    "--no-bert-score"  # Skip BERTScore for speed
]

print("=" * 60)
print("🚀 Quick Test Benchmark")
print("=" * 60)
print(f"Running: {' '.join(cmd[1:])}")
print()

subprocess.run(cmd)
