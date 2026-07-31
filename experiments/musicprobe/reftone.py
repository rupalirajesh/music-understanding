"""In-AUDIO reference tone for tuning_judgment (novel track, 2026-07-31).

Track D-zoom's key finding: absolute tuning needs an explicit REFERENCE to
judge the target pitch against — a bare F0 value (Track E, text) doesn't fix
it, only a reference-line-anchored image does. This track asks whether that
same ingredient works delivered in-AUDIO instead of switching modality at
all: play a short reference tone at the nominal (exactly-in-tune) pitch,
then the target tone — the way a musician tunes against a reference note.
No external renderer, no vision tower — built from synth.harmonic_tone(), the
same primitive cents_discrimination's two-tone stimuli use (via tone_pair()),
just with independent per-tone durations so the target tone's length always
matches the plain stimulus exactly (tone_pair forces both tones to one length).

Two new audio variants per tuning_judgment stimulus (from the SAME target
tone, so ground truth — is the TARGET in tune? — never changes):
  reftone        reference tone (nominal, i.e. exactly-in-tune pitch for this
                 stimulus's `base_midi`) + gap + the ORIGINAL target tone.
  wrong_reftone  reference tone shifted a few semitones away from the correct
                 nominal pitch + gap + the SAME target tone — the mechanism
                 control: if the model is genuinely comparing target-to-
                 reference, a wrong reference should mislead it; if it's just
                 reacting to "there are two tones now," wrong_reftone should
                 score the same as reftone (same logic as wrong_image/
                 wrong_audio elsewhere in this project).
"""
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE
from .generators.quantization import DUR as TARGET_DUR  # 2.0s -- the ORIGINAL
# tuning_judgment target-tone duration; the reftone variant must reuse this
# exactly, not shorten the target tone, or clip length becomes a confound.
from .synth import harmonic_tone
from .theory import midi_to_freq

REF_TONE_DUR = 1.0     # shorter than the target tone -- clearly a cue, not a second target
GAP = 0.4
WRONG_OFFSET_SEMITONES = (-4, -3, -2, -1, 1, 2, 3, 4)  # audibly different nominal pitch,
# still a plausible clean reference tone (not a random frequency)


def _two_tone(ref_freq: float, target_freq: float) -> np.ndarray:
    """Like synth.tone_pair, but the two tones can have DIFFERENT durations
    (tone_pair forces tone_dur to apply to both) -- the target tone here must
    stay at TARGET_DUR, matching the plain (no-reference) stimulus exactly, so
    clip length isn't a confound between conditions."""
    ref = harmonic_tone(ref_freq, REF_TONE_DUR)
    target = harmonic_tone(target_freq, TARGET_DUR)
    return np.concatenate([ref, np.zeros(int(GAP * SAMPLE_RATE)), target])


def reftone_path(audio_path: str) -> str:
    p = Path(audio_path)
    assert p.parts[0] == "stimuli", f"unexpected audio_path shape: {audio_path}"
    return str(Path("stimuli") / "reftone" / Path(*p.parts[1:]))


def wrong_reftone_path(audio_path: str) -> str:
    p = Path(audio_path)
    assert p.parts[0] == "stimuli", f"unexpected audio_path shape: {audio_path}"
    return str(Path("stimuli") / "reftone_wrong" / Path(*p.parts[1:]))


def build_reftone_audio(base_midi: int, midi_exact: float) -> np.ndarray:
    """Correct-reference variant: reference tone at the nominal (integer,
    exactly-in-tune) base_midi, then the actual target tone at midi_exact."""
    return _two_tone(midi_to_freq(base_midi), midi_to_freq(midi_exact))


def build_wrong_reftone_audio(base_midi: int, midi_exact: float, seed: int) -> np.ndarray:
    """Wrong-reference control: same target tone, reference shifted a few
    semitones away (deterministic per-stimulus draw, so reruns are stable)."""
    r = np.random.default_rng(seed)
    offset = WRONG_OFFSET_SEMITONES[int(r.integers(len(WRONG_OFFSET_SEMITONES)))]
    return _two_tone(midi_to_freq(base_midi + offset), midi_to_freq(midi_exact))
