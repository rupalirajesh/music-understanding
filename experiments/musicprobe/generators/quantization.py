"""The 12-TET grid probe (behavioral half).

Stimuli: single harmonic tones detuned 0..50 cents off the semitone grid.
Task `tuning_judgment`: "is this note in tune or out of tune?" (2AFC),
detuning in {0, 5, 10, 20, 35, 50} cents -> a tuning-detection psychometric
curve. A model whose pitch representation snaps to 12-TET should call
everything "in tune" (the snap destroys the evidence).

The same stimuli carry `detune_cents` + fractional `midi_exact` in factors,
which is the *probe* target for the representational half (Track B): probe
error vs distance-from-nearest-semitone — scalloped error = 12-TET-quantized
representation (RESEARCH_PLAN §5.3).
"""
import numpy as np

from ..config import STIMULI_DIR, EXP_ROOT
from ..manifest import StimulusRow
from ..synth import harmonic_tone, write_wav
from ..theory import midi_to_freq, midi_to_name

DETUNES = [0, 5, 10, 20, 35, 50]
TRIALS_PER_CELL = 15
DUR = 2.0


def generate(rng: np.random.Generator, quick: bool = False) -> list[StimulusRow]:
    detunes = DETUNES if not quick else [0, 50]
    rows = []
    i = 0
    for detune in detunes:
        # 3x trials for the in-tune class so the 2AFC isn't 15-vs-75 imbalanced
        per_cell = (TRIALS_PER_CELL * 3 if detune == 0 else TRIALS_PER_CELL) \
            if not quick else 2
        for _ in range(per_cell):
            seed = int(rng.integers(2**31))
            r = np.random.default_rng(seed)
            base = int(r.integers(52, 77))                # E3..E5, ON the grid
            sign = 1 if r.random() < 0.5 else -1
            midi_exact = base + sign * detune / 100
            y = harmonic_tone(midi_to_freq(midi_exact), DUR)
            sid = f"tune_{i:04d}"
            i += 1
            out = STIMULI_DIR / "tuning" / f"{sid}.wav"
            write_wav(out, y)
            truth = "in tune" if detune == 0 else "out of tune"
            rows.append(StimulusRow(
                f"tuning_judgment/{sid}", "tuning_judgment", 1,
                str(out.relative_to(EXP_ROOT)), truth,
                {"detune_cents": detune, "sign": sign,
                 "base_midi": base, "base_name": midi_to_name(base),
                 "midi_exact": round(midi_exact, 4)},
                DUR, seed))
    return rows
