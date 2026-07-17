"""Expand manifest into eval jobs (prompts + controls). Run after 01.

  .venv/bin/python scripts/02_build_jobs.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from musicprobe.jobs import build_jobs

if __name__ == "__main__":
    build_jobs()
