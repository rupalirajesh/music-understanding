"""Where a stimulus's RHYTHM-cluster front-end images live (parallel to
harmony_repr.py) -- Tracks R/S/T/U/V/W, the sequential representation ladder
for tempo_bpm/beats_per_bar (PROJECT_STATE.md next action 14). First causal
fine-tuning of any kind attempted on this cluster.

All six are re-renderings of the AUDIO signal via a real DSP transform
(librosa onset/tempogram functions), never from the ground-truth BPM or
beats-per-bar label -- same non-leakage rule as harmony_repr.py. Tracks V/W
need particular care here: the metrical grid / circle they draw must come
from a DETECTED periodicity, not from the ground-truth beats-per-bar count,
or the number of grid divisions/dots would hand over the answer directly.
See scripts/render_rhythm_repr.py for the rendering code.
"""
from pathlib import Path


def _repr_path(audio_path: str, subdir: str) -> str:
    p = Path(audio_path)
    assert p.parts[0] == "stimuli", f"unexpected audio_path shape: {audio_path}"
    return str(Path("stimuli") / subdir / Path(*p.parts[1:]).with_suffix(".png"))


def tempogram_path(audio_path: str) -> str:
    """Track R: librosa tempogram -- periodicity-vs-time heatmap, the
    rhythm analogue of chroma_cqt (pitch-class-vs-time)."""
    return _repr_path(audio_path, "tempogram")


def tempogram_picked_path(audio_path: str) -> str:
    """Track S: peak-picked (binarized) tempogram, same fix as Track L
    applied to periodicity instead of pitch class."""
    return _repr_path(audio_path, "tempogram_picked")


def onset_line_path(audio_path: str) -> str:
    """Track T: onset-strength envelope as a single 1-D curve over time --
    the direct rhythm analogue of f0_contour.py's pitch-over-time line."""
    return _repr_path(audio_path, "onset_line")


def onset_line_zoom_path(audio_path: str) -> str:
    """Track U: same onset-strength curve, zoomed (finer time resolution)."""
    return _repr_path(audio_path, "onset_line_zoom")


def rhythm_roll_path(audio_path: str) -> str:
    """Track V: onset markers against a metrical grid inferred from the
    audio's OWN detected tempo (not the ground-truth beats-per-bar count)."""
    return _repr_path(audio_path, "rhythm_roll")


def rhythm_necklace_path(audio_path: str) -> str:
    """Track W: onsets folded modulo one detected cycle length, plotted as
    a circular "necklace" (Toussaint, The Geometry of Musical Rhythm)."""
    return _repr_path(audio_path, "rhythm_necklace")


def rhythm_roll_zoom_path(audio_path: str) -> str:
    """Track Y (2026-08-06): Track V's onset-vs-detected-pulse-grid rhythm
    roll, rendered at Track U's finer time resolution (HOP_ZOOM) instead of
    the default hop. Tracks R-W tested zoom (T/U) and an explicit detected-
    pulse reference (V) separately but never combined them -- this is that
    missing combination, the rhythm analogue of Track X above and of what
    actually fixed pitch (Track D-zoom = zoom + reference, not either alone).
    Grid spacing still comes from _detect_click_period (audio-derived median
    inter-onset interval), never from the ground-truth beats-per-bar count --
    same leakage rule as Track V."""
    return _repr_path(audio_path, "rhythm_roll_zoom")
