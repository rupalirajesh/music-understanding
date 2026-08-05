"""Where a stimulus's HARMONY-cluster front-end images live (parallel to
chromagram.py/f0_contour.py) -- Tracks L/M/N/O/P/Q, the sequential
representation ladder for key_id/mode_id/chord_quality/interval_id
(RESEARCH_PLAN.md S:12.7, PROJECT_STATE.md next action 13).

All six are re-renderings of the AUDIO signal via a real DSP transform
(librosa), never from MIDI/factors ground truth -- same non-leakage rule as
chromagram.py's chroma_cqt. See scripts/render_harmony_repr.py for the
actual rendering code; this module only defines where each representation's
PNG lives, same split as every other *_path.py module in this package.
"""
from pathlib import Path


def _repr_path(audio_path: str, subdir: str) -> str:
    p = Path(audio_path)
    assert p.parts[0] == "stimuli", f"unexpected audio_path shape: {audio_path}"
    return str(Path("stimuli") / subdir / Path(*p.parts[1:]).with_suffix(".png"))


def chroma_picked_path(audio_path: str) -> str:
    """Track L: peak-picked (binarized) chroma -- top-K active pitch-class
    bins per time-frame, bright block / dark rest, instead of the raw
    continuous energy heatmap Track G used."""
    return _repr_path(audio_path, "chroma_picked")


def chroma_picked_zoom_path(audio_path: str) -> str:
    """Track M: same binarization as Track L, finer time resolution
    (smaller hop) and a wider figure -- the "zoom" half of D-zoom that
    Track G's flat chroma never tested."""
    return _repr_path(audio_path, "chroma_picked_zoom")


def harmony_line_path(audio_path: str) -> str:
    """Track N: multi-pitch trajectory chart -- audio-derived (librosa
    piptrack) candidate pitches per frame, plotted as points/short
    connected runs over a log-Hz axis. The polyphonic generalization of
    f0_contour.py's single-line pitch chart."""
    return _repr_path(audio_path, "harmony_line")


def harmony_line_zoom_path(audio_path: str) -> str:
    """Track O: same chart as Track N, y-axis zoomed to the stimulus's own
    active pitch range (mirrors f0_contour.f0_zoom_path)."""
    return _repr_path(audio_path, "harmony_line_zoom")


def piano_roll_path(audio_path: str) -> str:
    """Track P: full piano-roll -- absolute pitch height x time, one block
    per audio-detected note (onset-segmented, piptrack pitch estimate).
    Chords show as vertically-stacked blocks at one time-coordinate."""
    return _repr_path(audio_path, "piano_roll")


def tonnetz_path(audio_path: str) -> str:
    """Track Q: tonal-centroid / Tonnetz -- librosa.feature.tonnetz applied
    to the same chroma_cqt Track G used, a 6-D space where harmonically
    close pitch relations (fifths, thirds) map to small distances (Harte,
    Sandler & Gasser 2006)."""
    return _repr_path(audio_path, "tonnetz")
