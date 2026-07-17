"""Tier 1: tempo estimation (open-ended BPM) and meter identification.

Tempo uses sample-accurate numpy click tracks (fluidsynth timing depends on
soundfont envelopes). BPM is drawn continuously so round-number priors
("probably 120") are detectable: error distribution vs a no-audio control.
Meter stimuli put the accent structure in both loudness and pitch of clicks.
"""
import numpy as np

from ..config import STIMULI_DIR, EXP_ROOT
from ..manifest import StimulusRow
from ..synth import click_track, write_wav
from ..theory import METERS


def generate(rng: np.random.Generator, quick: bool = False) -> list[StimulusRow]:
    rows = []

    # --- tempo_bpm ---
    n_tempo = 60 if not quick else 4
    for i in range(n_tempo):
        seed = int(rng.integers(2**31))
        r = np.random.default_rng(seed)
        bpm = float(np.round(r.uniform(55, 185), 1))
        n_bars = max(4, int(np.ceil(bpm / 60 * 10 / 4)))  # ~>=10 s of audio
        y = click_track(bpm, METERS["4/4"][1], n_bars)
        sid = f"tempo_{i:03d}"
        out = STIMULI_DIR / "tempo" / f"{sid}.wav"
        write_wav(out, y)
        rows.append(StimulusRow(
            f"tempo_bpm/{sid}", "tempo_bpm", 1,
            str(out.relative_to(EXP_ROOT)), str(bpm),
            {"bpm": bpm, "n_bars": n_bars, "meter": "4/4"},
            n_bars * 4 * 60 / bpm, seed))

    # --- beats_per_bar ---
    # Audio only conveys the cycle length and accent grouping, NOT the notated
    # denominator (4/4 vs 4/8 sound identical). So the question is the audible
    # one: how many beats per cycle. The notated meter stays in factors for
    # slicing; a 3/4-vs-6/8 *grouping* task (2x3 vs 3x2) is specced separately.
    per_meter = 20 if not quick else 1
    for meter, (beats, accents) in METERS.items():
        for i in range(per_meter):
            seed = int(rng.integers(2**31))
            r = np.random.default_rng(seed)
            bpm = float(np.round(r.uniform(90, 150), 1))
            n_bars = max(4, int(np.ceil(bpm / 60 * 10 / beats)))
            y = click_track(bpm, accents, n_bars)
            sid = f"meter_{meter.replace('/', '-')}_{i:02d}"
            out = STIMULI_DIR / "meter" / f"{sid}.wav"
            write_wav(out, y)
            rows.append(StimulusRow(
                f"beats_per_bar/{sid}", "beats_per_bar", 1,
                str(out.relative_to(EXP_ROOT)), str(beats),
                {"meter": meter, "beats": beats, "bpm": bpm, "n_bars": n_bars},
                n_bars * beats * 60 / bpm, seed))
    return rows
