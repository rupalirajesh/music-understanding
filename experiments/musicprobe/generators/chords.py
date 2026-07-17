"""Tier 2: chord quality, chord progressions, and polyphony (note counting).

Chord quality is presented block vs arpeggiated (arpeggiation serializes the
notes — if a model succeeds arpeggiated but fails block, simultaneity is the
bottleneck, not interval knowledge). Inversion is a controlled factor.
"""
import numpy as np

from ..config import STIMULI_DIR, EXP_ROOT, GM_PROGRAMS, available_soundfonts
from ..manifest import StimulusRow
from ..synth import midi_notes, render_midi
from ..theory import CHORDS, CHORD_SPOKEN, PROGRESSIONS, NOTE_NAMES


def _voice(root_midi: int, quality: str, inversion: int) -> list[int]:
    ivs = CHORDS[quality]
    pitches = [root_midi + iv for iv in ivs]
    for _ in range(inversion):
        pitches = pitches[1:] + [pitches[0] + 12]
    return pitches


def generate(rng: np.random.Generator, quick: bool = False) -> list[StimulusRow]:
    sfs = list(available_soundfonts().items())
    progs = list(GM_PROGRAMS.items())
    rows = []

    # --- chord_quality ---
    qualities = list(CHORDS) if not quick else ["major", "minor", "dominant7"]
    n_items = 6 if not quick else 1
    for quality in qualities:
        for pres in ["block", "arpeggio"]:
            for i in range(n_items):
                seed = int(rng.integers(2**31))
                r = np.random.default_rng(seed)
                root = int(r.integers(48, 65))
                inversion = int(r.integers(0, 3 if len(CHORDS[quality]) == 3 else 4)) \
                    if r.random() < 0.4 else 0
                pitches = _voice(root, quality, inversion)
                if pres == "block":
                    notes = [(p, 0.25, 2.75) for p in pitches]
                else:
                    notes = [(p, 0.25 + j * 0.45, 0.25 + j * 0.45 + 0.9)
                             for j, p in enumerate(pitches)]
                    notes += [(p, 0.25 + len(pitches) * 0.45, 2.2 + len(pitches) * 0.45)
                              for p in pitches]  # arpeggio then block restatement
                prog_name, prog = progs[r.integers(len(progs))]
                sf_name, sf_path = sfs[r.integers(len(sfs))]
                sid = f"{quality}_{pres}_{root}_{i}"
                out = STIMULI_DIR / "chords" / f"{sid}.wav"
                render_midi(midi_notes(notes, program=prog), sf_path, out)
                dur = max(e for _, _, e in notes) + 0.25
                rows.append(StimulusRow(
                    f"chord_quality/{sid}", "chord_quality", 2,
                    str(out.relative_to(EXP_ROOT)), CHORD_SPOKEN[quality],
                    {"quality": quality, "presentation": pres, "root": root,
                     "root_name": NOTE_NAMES[root % 12], "inversion": inversion,
                     "program": prog_name, "soundfont": sf_name}, dur, seed))

    # --- progression_id ---
    prog_names_list = ["I-IV-V-I", "I-V-vi-IV", "ii-V-I", "12-bar-blues"] \
        if not quick else ["I-IV-V-I", "12-bar-blues"]
    n_keys = 8 if not quick else 1
    for pname in prog_names_list:
        for i in range(n_keys):
            seed = int(rng.integers(2**31))
            r = np.random.default_rng(seed)
            tonic = int(r.integers(53, 65))
            chord_dur = 1.5 if pname != "12-bar-blues" else 0.9
            notes, t = [], 0.25
            for root_off, quality in PROGRESSIONS[pname]:
                for p in _voice(tonic + root_off - 12, quality, 0):
                    notes.append((p, t, t + chord_dur))
                notes.append((tonic + root_off, t, t + chord_dur))  # doubled root on top
                t += chord_dur
            prog_name, prog = progs[r.integers(len(progs))]
            sf_name, sf_path = sfs[r.integers(len(sfs))]
            sid = f"{pname}_{tonic}_{i}"
            out = STIMULI_DIR / "progressions" / f"{sid}.wav"
            render_midi(midi_notes(notes, program=prog), sf_path, out)
            rows.append(StimulusRow(
                f"progression_id/{sid}", "progression_id", 2,
                str(out.relative_to(EXP_ROOT)), pname,
                {"progression": pname, "tonic": tonic,
                 "program": prog_name, "soundfont": sf_name}, t + 0.25, seed))

    # --- note_count (polyphony) ---
    counts = range(1, 6) if not quick else [1, 4]
    per_count = 20 if not quick else 1
    for n in counts:
        for i in range(per_count):
            seed = int(rng.integers(2**31))
            r = np.random.default_rng(seed)
            base = int(r.integers(48, 60))
            # random distinct pitches, min 2 semitones apart to avoid beating fusion
            pool = list(range(base, base + 24))
            pitches, tries = [], 0
            while len(pitches) < n and tries < 200:
                c = int(r.choice(pool)); tries += 1
                if all(abs(c - p) >= 2 for p in pitches):
                    pitches.append(c)
            notes = [(p, 0.25, 2.75) for p in pitches]
            prog_name, prog = progs[r.integers(len(progs))]
            sf_name, sf_path = sfs[r.integers(len(sfs))]
            sid = f"count{n}_{i:02d}"
            out = STIMULI_DIR / "polyphony" / f"{sid}.wav"
            render_midi(midi_notes(notes, program=prog), sf_path, out)
            rows.append(StimulusRow(
                f"note_count/{sid}", "note_count", 1,
                str(out.relative_to(EXP_ROOT)), str(n),
                {"n_notes": n, "pitches": pitches,
                 "program": prog_name, "soundfont": sf_name}, 3.0, seed))
    return rows
