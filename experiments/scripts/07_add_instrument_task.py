"""One-off: add the instrument_id task over the EXISTING pitch recordings.

  .venv/bin/python scripts/07_add_instrument_task.py

Regenerates the pitch manifest rows (same seeds -> identical audio, now with a
third row per clip asking "what instrument is this?") and APPENDS the new jobs
to jobs.parquet. Existing job rows pass through byte-identical, so the
already-run models (Qwen2-Audio, Gemini) just rerun their runner and pick up
only the new instrument_id jobs (runners are resumable by job_id).

Idempotent — safe to rerun.
"""
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from musicprobe.config import GLOBAL_SEED, available_soundfonts
from musicprobe.manifest import save_rows
from musicprobe.generators import pitch
from musicprobe.jobs import append_jobs

if __name__ == "__main__":
    assert available_soundfonts(), "no soundfonts in assets/soundfonts/"
    # exact same seeding as scripts/01_generate_stimuli.py for the pitch family
    rng = np.random.default_rng(GLOBAL_SEED + zlib.crc32(b"pitch") % 1000)
    rows = pitch.generate(rng)
    for task in ("pitch_note_id", "octave_id", "instrument_id"):
        save_rows([r for r in rows if r.task == task], task)
    append_jobs(["instrument_id"])
