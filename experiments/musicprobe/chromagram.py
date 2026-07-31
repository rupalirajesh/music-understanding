"""Where a stimulus's CHROMAGRAM image lives (parallel to f0_contour.py).

Track D/E/F all targeted the pitch-precision shortlist (octave_id, note_count,
tuning_judgment, cents_discrimination). This is the first visual front-end
aimed at the OTHER cluster Track B flagged (L3 > generic-encoder L2, never
re-tested causally): key_id, mode_id, chord_quality, interval_id — harmonic/
relational tasks where the relevant structure is pitch-CLASS content over
time, not absolute pitch height. A 12-row (pitch class) x time heatmap is the
harmonic analogue of the F0-contour's "pitch as vertical position" trick:
same abstraction level as the audio (a re-rendering of the signal via a
standard chroma transform, not a symbolic answer) — no tonic/root/quality is
ever highlighted or labelled beyond the fixed pitch-class axis, so nothing
about which pitch class matters for THIS stimulus leaks into the image.
"""
from pathlib import Path


def chromagram_path(audio_path: str) -> str:
    p = Path(audio_path)
    assert p.parts[0] == "stimuli", f"unexpected audio_path shape: {audio_path}"
    return str(Path("stimuli") / "chromagrams" / Path(*p.parts[1:]).with_suffix(".png"))
