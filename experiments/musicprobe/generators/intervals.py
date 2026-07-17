"""Tier 1: interval identification — melodic (asc/desc) and harmonic.

Root note is randomized per item so absolute-pitch memorization can't help;
only the *relation* between notes answers the question.
"""
import numpy as np

from ..config import STIMULI_DIR, EXP_ROOT, GM_PROGRAMS, available_soundfonts
from ..manifest import StimulusRow
from ..synth import midi_notes, render_midi
from ..theory import INTERVALS, midi_to_name


def generate(rng: np.random.Generator, quick: bool = False) -> list[StimulusRow]:
    sfs = list(available_soundfonts().items())
    progs = list(GM_PROGRAMS.items())
    semis = list(INTERVALS) if not quick else [4, 7, 12]
    presentations = ["melodic_asc", "melodic_desc", "harmonic"]
    n_roots = 4 if not quick else 1

    rows = []
    for semi in semis:
        short, spoken = INTERVALS[semi]
        for pres in presentations:
            for i in range(n_roots):
                seed = int(rng.integers(2**31))
                r = np.random.default_rng(seed)
                root = int(r.integers(48, 72))  # C3..B4 keeps top note <= B5
                lo, hi = root, root + semi
                a, b = (hi, lo) if pres == "melodic_desc" else (lo, hi)
                if pres == "harmonic":
                    notes = [(lo, 0.25, 2.25), (hi, 0.25, 2.25)]
                else:
                    notes = [(a, 0.25, 1.25), (b, 1.5, 2.5)]
                prog_name, prog = progs[r.integers(len(progs))]
                sf_name, sf_path = sfs[r.integers(len(sfs))]
                sid = f"{short}_{pres}_{root}_{i}"
                out = STIMULI_DIR / "intervals" / f"{sid}.wav"
                render_midi(midi_notes(notes, program=prog), sf_path, out)
                rows.append(StimulusRow(
                    f"interval_id/{sid}", "interval_id", 1,
                    str(out.relative_to(EXP_ROOT)), spoken,
                    {"semitones": semi, "interval_short": short,
                     "presentation": pres, "root": root,
                     "root_name": midi_to_name(root),
                     "program": prog_name, "soundfont": sf_name},
                    2.75, seed))
    return rows
