"""Standalone Multi-Step Forecasting Evaluation Script."""

from __future__ import annotations
import sys
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_task_3_3 import run_task_3_3_experiments

if __name__ == "__main__":
    run_task_3_3_experiments()
