"""Audio rendering: (a) MIDI -> fluidsynth for realistic instrument timbres,
(b) direct numpy synthesis for stimuli needing sub-semitone pitch control
(cents psychometrics) where MIDI's pitch grid is the wrong tool.

All output: mono WAV at config.SAMPLE_RATE, peak-normalized to -3 dBFS so
loudness never differs systematically between stimulus arms (pitfall §9).
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

from .config import SAMPLE_RATE, RENDER_GAIN

PEAK_DBFS = -3.0


def _normalize(y: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(y))
    if peak < 1e-9:
        raise ValueError("rendered audio is silent")
    return y * (10 ** (PEAK_DBFS / 20) / peak)


def write_wav(path: Path, y: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, _normalize(y).astype(np.float32), SAMPLE_RATE)


# ---------------------------------------------------------------- MIDI path

def render_midi(pm: pretty_midi.PrettyMIDI, soundfont: Path, out_path: Path):
    """Render a PrettyMIDI object to WAV via the fluidsynth CLI."""
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        midi_path = Path(f.name)
    pm.write(str(midi_path))
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            raw_path = Path(f.name)
        cmd = [
            "fluidsynth", "-ni", "-g", str(RENDER_GAIN),
            "-F", str(raw_path), "-r", str(SAMPLE_RATE),
            str(soundfont), str(midi_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if res.returncode != 0:
            raise RuntimeError(f"fluidsynth failed: {res.stderr[-500:]}")
        y, sr = sf.read(raw_path)
        if y.ndim == 2:
            y = y.mean(axis=1)
        write_wav(out_path, y)
    finally:
        midi_path.unlink(missing_ok=True)
        raw_path.unlink(missing_ok=True)


def midi_notes(notes: list[tuple[int, float, float]], program: int = 0,
               velocity: int = 96, is_drum: bool = False) -> pretty_midi.PrettyMIDI:
    """Build a PrettyMIDI from (midi_pitch, start_s, end_s) triples."""
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program, is_drum=is_drum)
    for pitch, start, end in notes:
        inst.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch,
                                           start=start, end=end))
    pm.instruments.append(inst)
    return pm


# ------------------------------------------------------------- numpy path

def harmonic_tone(freq: float, dur: float, n_harmonics: int = 8,
                  rolloff: float = 0.75, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Sustained harmonic complex tone with gentle attack/release.

    Used for cents-discrimination stimuli: identical spectra except for f0,
    so nothing but pitch differs between the two intervals of a trial.
    """
    t = np.arange(int(dur * sr)) / sr
    y = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        if freq * k > sr / 2:
            break
        y += (rolloff ** (k - 1)) * np.sin(2 * np.pi * freq * k * t)
    # 20 ms raised-cosine attack and release: no clicks, no spectral splatter
    ramp = int(0.020 * sr)
    env = np.ones_like(y)
    env[:ramp] = 0.5 * (1 - np.cos(np.pi * np.arange(ramp) / ramp))
    env[-ramp:] = env[:ramp][::-1]
    return y * env


def tone_pair(f1: float, f2: float, tone_dur: float = 1.0,
              gap: float = 0.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    a = harmonic_tone(f1, tone_dur, sr=sr)
    b = harmonic_tone(f2, tone_dur, sr=sr)
    return np.concatenate([a, np.zeros(int(gap * sr)), b])


def click_track(bpm: float, meter_accents: list[float], n_bars: int,
                sr: int = SAMPLE_RATE) -> np.ndarray:
    """Percussive click pattern with metrical accents (for tempo/meter tasks).

    Strong beats: louder, lower-pitched click. Rendered directly so tempo is
    sample-accurate (fluidsynth drum timing depends on the soundfont).
    """
    beat_s = 60.0 / bpm
    total = int(n_bars * len(meter_accents) * beat_s * sr) + sr // 10
    y = np.zeros(total)
    click_hi = harmonic_tone(1500, 0.03, n_harmonics=1, sr=sr)
    click_lo = harmonic_tone(800, 0.05, n_harmonics=3, sr=sr)
    for bar in range(n_bars):
        for i, accent in enumerate(meter_accents):
            pos = int((bar * len(meter_accents) + i) * beat_s * sr)
            c = click_lo * (0.6 + 0.4 * accent) if accent >= 1 else click_hi * (0.35 + 0.4 * accent)
            y[pos:pos + len(c)] += c
    return y
