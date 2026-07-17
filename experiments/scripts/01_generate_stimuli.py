"""Generate all synthetic stimuli + manifest.

  .venv/bin/python scripts/01_generate_stimuli.py --quick    # smoke test (~1 min)
  .venv/bin/python scripts/01_generate_stimuli.py            # full battery
  .venv/bin/python scripts/01_generate_stimuli.py --tasks cents key_mode
"""
import argparse
import sys
import zlib
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from musicprobe.config import GLOBAL_SEED, available_soundfonts
from musicprobe.manifest import save_rows
from musicprobe.generators import (pitch, intervals, cents, tempo_meter,
                                   key_mode, chords, quantization)

GENERATORS = {
    "pitch": (pitch.generate, ["pitch_note_id", "octave_id"]),
    "intervals": (intervals.generate, ["interval_id"]),
    "cents": (cents.generate, ["cents_discrimination"]),
    "tempo_meter": (tempo_meter.generate, ["tempo_bpm", "beats_per_bar"]),
    "key_mode": (key_mode.generate, ["key_id", "mode_id"]),
    "chords": (chords.generate, ["chord_quality", "progression_id", "note_count"]),
    "quantization": (quantization.generate, ["tuning_judgment"]),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test battery")
    ap.add_argument("--tasks", nargs="*", default=list(GENERATORS),
                    choices=list(GENERATORS))
    args = ap.parse_args()

    sfs = available_soundfonts()
    assert sfs, "no soundfonts found in assets/soundfonts/"
    print(f"soundfonts: {list(sfs)}")

    for name in args.tasks:
        gen, task_names = GENERATORS[name]
        t0 = time.time()
        rows = gen(np.random.default_rng(GLOBAL_SEED + zlib.crc32(name.encode()) % 1000), args.quick)
        for task in task_names:
            save_rows([r for r in rows if r.task == task], task)
        print(f"  {name}: {len(rows)} rows in {time.time() - t0:.0f}s")
