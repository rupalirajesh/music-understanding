"""The stimulus manifest: one row per audio stimulus, the single source of
truth every downstream step (eval harness, L1 baselines, probing) reads.

Schema (columns):
  stimulus_id   str   stable unique id, e.g. "pitch_note_id/000123"
  task          str   task name from TASKS.md
  tier          int   1/2/3
  audio_path    str   relative to experiments/ root
  ground_truth  str   canonical answer string (scoring joins on this)
  factors       str   JSON dict of every controlled factor (key, program,
                      soundfont, register, tempo, delta_cents, ...) — used for
                      psychometric curves, confusion slicing, held-out splits
  duration_s    float
  seed          int   RNG seed that produced this row (full reproducibility)
"""
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from .config import MANIFEST_PATH, EXP_ROOT


@dataclass
class StimulusRow:
    stimulus_id: str
    task: str
    tier: int
    audio_path: str  # relative to EXP_ROOT
    ground_truth: str
    factors: dict
    duration_s: float
    seed: int

    def to_record(self) -> dict:
        d = asdict(self)
        d["factors"] = json.dumps(d["factors"], sort_keys=True)
        return d


def save_rows(rows: list[StimulusRow], task: str):
    """Replace all rows for `task` in the manifest (idempotent regeneration)."""
    new = pd.DataFrame([r.to_record() for r in rows])
    if MANIFEST_PATH.exists():
        old = pd.read_parquet(MANIFEST_PATH)
        old = old[old["task"] != task]
        new = pd.concat([old, new], ignore_index=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(MANIFEST_PATH, index=False)
    print(f"[manifest] {task}: {len(rows)} rows (manifest total {len(new)})")


def load_manifest(tasks: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_parquet(MANIFEST_PATH)
    if tasks:
        df = df[df["task"].isin(tasks)].reset_index(drop=True)
    df["factors"] = df["factors"].apply(json.loads)
    return df


def audio_abspath(rel: str) -> Path:
    return EXP_ROOT / rel
