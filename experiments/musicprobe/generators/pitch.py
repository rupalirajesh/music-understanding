"""Tier 1: single-note pitch identification + octave placement + instrument ID.

Same audio serves three tasks (pitch class, octave, instrument) as separate
manifest rows. Factors crossed: pitch class x octave x instrument x soundfont.
"""
import numpy as np

from ..config import STIMULI_DIR, EXP_ROOT, GM_PROGRAMS, available_soundfonts
from ..manifest import StimulusRow
from ..synth import midi_notes, render_midi
from ..theory import midi_to_pitch_class, midi_to_name, INSTRUMENT_SPOKEN

DUR = 2.0


def generate(rng: np.random.Generator, quick: bool = False) -> list[StimulusRow]:
    sfs = list(available_soundfonts().items())
    progs = list(GM_PROGRAMS.items())
    octaves = [3, 4, 5]  # C3..B5 — comfortably inside every GM instrument's range
    pcs = range(12) if not quick else range(0, 12, 4)
    n_timbres = 2 if not quick else 1

    rows = []
    for pc in pcs:
        for octave in octaves:
            midi = 12 * (octave + 1) + pc
            for rep in range(n_timbres):
                seed = int(rng.integers(2**31))
                r = np.random.default_rng(seed)
                prog_name, prog = progs[r.integers(len(progs))]
                sf_name, sf_path = sfs[r.integers(len(sfs))]
                sid = f"pitch/{midi:03d}_{prog_name}_{sf_name}_{rep}"
                out = STIMULI_DIR / "pitch" / f"{sid.split('/')[1]}.wav"
                pm = midi_notes([(midi, 0.25, 0.25 + DUR)], program=prog)
                render_midi(pm, sf_path, out)
                factors = {"midi": midi, "pitch_class": midi_to_pitch_class(midi),
                           "octave": octave, "note_name": midi_to_name(midi),
                           "program": prog_name, "soundfont": sf_name}
                rel = str(out.relative_to(EXP_ROOT))
                rows.append(StimulusRow(f"pitch_note_id/{sid}", "pitch_note_id", 1,
                                        rel, midi_to_pitch_class(midi), factors,
                                        DUR + 0.5, seed))
                rows.append(StimulusRow(f"octave_id/{sid}", "octave_id", 1,
                                        rel, str(octave), factors, DUR + 0.5, seed))
                rows.append(StimulusRow(f"instrument_id/{sid}", "instrument_id", 1,
                                        rel, INSTRUMENT_SPOKEN[prog_name], factors,
                                        DUR + 0.5, seed))
    return rows
