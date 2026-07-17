"""Tier 1: microtonal pitch discrimination — the psychometric-curve task.

Two harmonic complex tones; the second is higher / lower / same by
delta ∈ {5, 10, 25, 50, 100} cents. Direct numpy synthesis (not MIDI):
the two tones differ in nothing but f0. Base pitch is randomized *and*
given a random detune off the 12-TET grid, so a model with an internal
semitone grid gains no anchor.

Analysis: accuracy vs |delta| -> threshold where the curve crosses 62.5%
(midpoint between 3AFC chance 33% and perfect). Trained humans: 5-10 cents.
"""
import numpy as np

from ..config import STIMULI_DIR, EXP_ROOT
from ..manifest import StimulusRow
from ..synth import tone_pair, write_wav
from ..theory import midi_to_freq

DELTAS = [5, 10, 25, 50, 100]
TRIALS_PER_CELL = 15   # x (5 deltas x 2 directions) + same-trials = 180 total
SAME_TRIALS = 30


def _make(rng, delta_cents: float) -> tuple[float, float, dict]:
    base_midi = rng.uniform(52, 76)          # ~E3..E5, continuous (off-grid)
    f1 = midi_to_freq(base_midi)
    f2 = f1 * 2 ** (delta_cents / 1200)
    return f1, f2, {"base_midi": round(base_midi, 3)}


def generate(rng: np.random.Generator, quick: bool = False) -> list[StimulusRow]:
    deltas = DELTAS if not quick else [10, 100]
    per_cell = TRIALS_PER_CELL if not quick else 2
    same_n = SAME_TRIALS if not quick else 2

    cells = [(d, s) for d in deltas for s in (+1, -1)] * per_cell
    cells += [(0, 0)] * same_n
    rows = []
    for i, (delta, sign) in enumerate(cells):
        seed = int(rng.integers(2**31))
        r = np.random.default_rng(seed)
        f1, f2, extra = _make(r, sign * delta)
        sid = f"trial_{i:04d}"
        out = STIMULI_DIR / "cents" / f"{sid}.wav"
        write_wav(out, tone_pair(f1, f2))
        truth = "same" if delta == 0 else ("higher" if sign > 0 else "lower")
        rows.append(StimulusRow(
            f"cents_discrimination/{sid}", "cents_discrimination", 1,
            str(out.relative_to(EXP_ROOT)), truth,
            {"delta_cents": delta, "direction": truth,
             "f1_hz": round(f1, 3), "f2_hz": round(f2, 3), **extra},
            2.5, seed))
    return rows
