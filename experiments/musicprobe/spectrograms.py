"""Single source of truth for where a stimulus's spectrogram image lives.
Both scripts/10_render_spectrograms.py (writes them) and image_jobs.py
(reads them) import this — don't reimplement the path derivation elsewhere.
"""
from pathlib import Path


def spectrogram_path(audio_path: str) -> str:
    """Deterministic derivation: stimuli/<task>/<name>.wav ->
    stimuli/spectrograms/<task>/<name>.png. Not stored as a manifest column
    on purpose — manifests/stimuli.parquet is frozen (PROJECT_STATE.md
    decision 10); adding a column would be a schema change."""
    p = Path(audio_path)
    assert p.parts[0] == "stimuli", f"unexpected audio_path shape: {audio_path}"
    return str(Path("stimuli") / "spectrograms" / Path(*p.parts[1:]).with_suffix(".png"))
