"""Where a stimulus's F0-CONTOUR image lives (parallel to spectrograms.py).

The F0-contour plot is the "readable" alternative image for Track D's
make-the-model-use-it experiments: instead of a spectrogram TEXTURE (which a
natural-image vision tower has never seen), we hand it a LINE CHART of pitch
over time — pitch as literal vertical position on a log-Hz axis, the exact
thing chart-reading VLMs are good at, and the exact thing (fine F0) the audio
encoder discards. This is the "L2 fix" delivered through the vision channel.
"""
from pathlib import Path


def f0_contour_path(audio_path: str) -> str:
    p = Path(audio_path)
    assert p.parts[0] == "stimuli", f"unexpected audio_path shape: {audio_path}"
    return str(Path("stimuli") / "f0contours" / Path(*p.parts[1:]).with_suffix(".png"))
