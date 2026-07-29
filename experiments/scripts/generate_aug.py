"""Generate a LARGE extra training set for the two microtone tasks (cents,
tuning) to test whether Track-F pitch-stream fusion was a data-size failure.

Pure numpy synthesis (no soundfont) -> cheap. Distinct IDs (augc_*/augt_*) so it
never collides with the frozen v1 battery. Writes:
  manifests/aug_train_jobs.parquet  (stimulus_id, task, prompt, ground_truth,
                                     audio_path, image_condition='image', format)
These are TRAINING-ONLY rows; eval stays on the frozen v1 held-out split, so
results remain comparable to the small-data run.

  python scripts/generate_aug.py --cents-per-cell 140 --tuning-per-cell 160
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from musicprobe.config import STIMULI_DIR, EXP_ROOT, MANIFEST_DIR
from musicprobe.synth import tone_pair, harmonic_tone, write_wav
from musicprobe.theory import midi_to_freq
from musicprobe.prompts import build_prompt

CENTS_DELTAS = [5, 10, 25, 50, 100]
TUNE_DETUNES = [0, 5, 10, 20, 35, 50]
AUG_JOBS_PATH = MANIFEST_DIR / "aug_train_jobs.parquet"


def _row(sid, task, audio_rel, truth, rng):
    return {"stimulus_id": sid, "task": task,
            "prompt": build_prompt(task, int(rng.integers(3)), "open", None),
            "ground_truth": truth, "audio_path": audio_rel,
            "image_condition": "image", "format": "open"}


def gen_cents(per_cell, same_n, rng):
    rows = []
    cells = [(d, s) for d in CENTS_DELTAS for s in (+1, -1)] * per_cell + [(0, 0)] * same_n
    for i, (delta, sign) in enumerate(cells):
        base_midi = rng.uniform(52, 76)
        f1 = midi_to_freq(base_midi)
        f2 = f1 * 2 ** (sign * delta / 1200)
        sid = f"cents_discrimination/augc_{i:05d}"
        rel = f"stimuli/cents/augc_{i:05d}.wav"
        write_wav(EXP_ROOT / rel, tone_pair(f1, f2))
        truth = "same" if delta == 0 else ("higher" if sign > 0 else "lower")
        rows.append(_row(sid, "cents_discrimination", rel, truth, rng))
    return rows


def gen_tuning(per_cell, rng):
    rows = []
    i = 0
    for detune in TUNE_DETUNES:
        n = per_cell * 3 if detune == 0 else per_cell     # keep 2AFC balanced
        for _ in range(n):
            base = int(rng.integers(52, 77))
            sign = 1 if rng.random() < 0.5 else -1
            midi_exact = base + sign * detune / 100
            sid = f"tuning_judgment/augt_{i:05d}"
            rel = f"stimuli/tuning/augt_{i:05d}.wav"
            write_wav(EXP_ROOT / rel, harmonic_tone(midi_to_freq(midi_exact), 2.0))
            truth = "in tune" if detune == 0 else "out of tune"
            rows.append(_row(sid, "tuning_judgment", rel, truth, rng))
            i += 1
    return rows


def main(cents_per_cell, tuning_per_cell, same_n, seed):
    rng = np.random.default_rng(seed)
    rows = gen_cents(cents_per_cell, same_n, rng) + gen_tuning(tuning_per_cell, rng)
    df = pd.DataFrame(rows)
    df.to_parquet(AUG_JOBS_PATH, index=False)
    print(f"[aug] {len(df)} training stimuli "
          f"({(df.task=='cents_discrimination').sum()} cents, "
          f"{(df.task=='tuning_judgment').sum()} tuning) -> {AUG_JOBS_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cents-per-cell", type=int, default=140)   # x10 cells -> 1400
    ap.add_argument("--tuning-per-cell", type=int, default=160)  # ~960
    ap.add_argument("--same-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    main(a.cents_per_cell, a.tuning_per_cell, a.same_n, a.seed)
