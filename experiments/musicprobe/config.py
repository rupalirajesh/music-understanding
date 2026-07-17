"""Central configuration: paths, rendering constants, instrument/soundfont rosters."""
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parent.parent
STIMULI_DIR = EXP_ROOT / "stimuli"
MANIFEST_DIR = EXP_ROOT / "manifests"
RESULTS_DIR = EXP_ROOT / "results"
SOUNDFONT_DIR = EXP_ROOT / "assets" / "soundfonts"

MANIFEST_PATH = MANIFEST_DIR / "stimuli.parquet"

SAMPLE_RATE = 44_100  # render at full quality; models resample themselves (documented bottleneck)
RENDER_GAIN = 0.7

# GM programs chosen to span timbre families: harmonic-rich percussive, bowed
# sustain, near-sinusoidal, and synthetic. Timbre is a *factor*, never held fixed.
GM_PROGRAMS = {
    "piano": 0,
    "violin": 40,
    "flute": 73,
    "synth_lead": 80,
}

# Multiple soundfonts so probe/eval splits can hold out a whole soundfont
# (leakage guard: never split by clip alone).
SOUNDFONTS = {
    "timgm": SOUNDFONT_DIR / "TimGM6mb.sf2",
    "fluidr3": SOUNDFONT_DIR / "FluidR3 GM2-2.SF2",
    "musescore": SOUNDFONT_DIR / "MuseScore_General.sf3",
}

GLOBAL_SEED = 20260717


def available_soundfonts() -> dict:
    return {k: v for k, v in SOUNDFONTS.items() if v.exists()}
