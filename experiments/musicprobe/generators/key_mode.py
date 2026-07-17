"""Tier 2: key identification (24 keys) and scale/mode identification.

Key stimuli come in two forms of increasing ecological difficulty:
  scale    — ascending+descending scale run (pure pitch-set matching)
  progression  — I-IV-V-I (or i-iv-v-i) progression with a diatonic melody on top
             (requires hearing tonality from harmony, not enumerating notes)

Mode stimuli likewise: bare scale vs melody-over-tonic-drone. The drone pins
the tonic so the question isolates *mode* perception from tonic finding.
"""
import numpy as np

from ..config import STIMULI_DIR, EXP_ROOT, GM_PROGRAMS, available_soundfonts
from ..manifest import StimulusRow
from ..synth import midi_notes, render_midi
from ..theory import MODES, MODE_SPOKEN, NOTE_NAMES, PROGRESSIONS, CHORDS, key_name


def _scale_notes(tonic_midi: int, degrees: list[int], note_dur=0.32) -> list:
    up = [tonic_midi + d for d in degrees] + [tonic_midi + 12]
    seq = up + up[-2::-1]  # ascending then descending
    return [(p, 0.25 + i * note_dur, 0.25 + (i + 1) * note_dur) for i, p in enumerate(seq)]


def _progression_notes(tonic_midi: int, mode: str, rng) -> list:
    prog = PROGRESSIONS["I-IV-V-I"] if mode == "major" else PROGRESSIONS["i-iv-v-i"]
    degrees = MODES["major" if mode == "major" else "natural_minor"]
    notes, t, chord_dur = [], 0.25, 1.4
    for root_off, quality in prog:
        for iv in CHORDS[quality]:
            notes.append((tonic_midi - 12 + root_off + iv, t, t + chord_dur))
        # diatonic melody note on top
        mel = tonic_midi + 12 + degrees[int(rng.integers(len(degrees)))]
        notes.append((mel, t, t + chord_dur * 0.6))
        t += chord_dur
    return notes


def generate(rng: np.random.Generator, quick: bool = False) -> list[StimulusRow]:
    sfs = list(available_soundfonts().items())
    progs = list(GM_PROGRAMS.items())
    rows = []

    # --- key_id: 24 keys x {scale, progression} x reps ---
    tonics = range(12) if not quick else [0, 7]
    reps = 2 if not quick else 1
    for tonic_pc in tonics:
        for mode in ["major", "minor"]:
            for form in ["scale", "progression"]:
                for rep in range(reps):
                    seed = int(rng.integers(2**31))
                    r = np.random.default_rng(seed)
                    tonic_midi = 60 + tonic_pc - (12 if tonic_pc > 6 else 0)
                    if form == "scale":
                        deg = MODES["major" if mode == "major" else "natural_minor"]
                        notes = _scale_notes(tonic_midi, deg)
                    else:
                        notes = _progression_notes(tonic_midi, mode, r)
                    prog_name, prog = progs[r.integers(len(progs))]
                    sf_name, sf_path = sfs[r.integers(len(sfs))]
                    sid = f"{NOTE_NAMES[tonic_pc].replace('#','s')}_{mode}_{form}_{rep}"
                    out = STIMULI_DIR / "key" / f"{sid}.wav"
                    render_midi(midi_notes(notes, program=prog), sf_path, out)
                    dur = max(e for _, _, e in notes) + 0.25
                    rows.append(StimulusRow(
                        f"key_id/{sid}", "key_id", 2,
                        str(out.relative_to(EXP_ROOT)), key_name(tonic_pc, mode),
                        {"tonic_pc": tonic_pc, "mode": mode, "form": form,
                         "program": prog_name, "soundfont": sf_name}, dur, seed))

    # --- mode_id: all modes x {bare_scale, melody_drone} x tonics ---
    modes = list(MODES) if not quick else ["major", "dorian", "blues"]
    mode_tonics = [57, 60, 62, 64] if not quick else [60]  # A3, C4, D4, E4
    for mode in modes:
        for tonic_midi in mode_tonics:
            for form in ["bare_scale", "melody_drone"]:
                seed = int(rng.integers(2**31))
                r = np.random.default_rng(seed)
                if form == "bare_scale":
                    notes = _scale_notes(tonic_midi, MODES[mode])
                else:
                    # random diatonic melody over a two-octave tonic drone
                    deg = MODES[mode]
                    steps = 16
                    mel, t = [], 0.25
                    for _ in range(steps):
                        p = tonic_midi + deg[int(r.integers(len(deg)))] + \
                            12 * int(r.integers(0, 2))
                        d = float(r.choice([0.25, 0.5, 0.5, 0.75]))
                        mel.append((p, t, t + d * 0.95))
                        t += d
                    drone = [(tonic_midi - 12, 0.25, t), (tonic_midi - 24, 0.25, t)]
                    notes = mel + drone
                prog_name, prog = progs[r.integers(len(progs))]
                sf_name, sf_path = sfs[r.integers(len(sfs))]
                sid = f"{mode}_{tonic_midi}_{form}"
                out = STIMULI_DIR / "mode" / f"{sid}.wav"
                render_midi(midi_notes(notes, program=prog), sf_path, out)
                dur = max(e for _, _, e in notes) + 0.25
                rows.append(StimulusRow(
                    f"mode_id/{sid}", "mode_id", 2,
                    str(out.relative_to(EXP_ROOT)), MODE_SPOKEN[mode],
                    {"mode": mode, "tonic_midi": tonic_midi, "form": form,
                     "program": prog_name, "soundfont": sf_name}, dur, seed))
    return rows
