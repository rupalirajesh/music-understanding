"""Pitch-tracker-as-TEXT feature — the scalable, deployable alternative to the
image (Track E). Instead of rendering a chart, run pyin and hand the model the
measured fundamental frequencies as a short text string in the prompt. One
forward pass, no vision tower, works on any audio the tracker handles — the
honest "add a pitch-aware front-end" path (vs the hand-rendered zoom image).

The string is the RAW tracker output (sustained-segment medians in Hz); the
model still has to reason over the numbers to answer, so this tests whether the
LM can USE a pitch front-end, not whether we handed it the answer in words.
"""
import json

import numpy as np
import soundfile as sf
import librosa

from .config import EXP_ROOT, MANIFEST_DIR, MANIFEST_PATH
from .image_jobs import DEFAULT_TASKS

F0_TEXT_PATH = MANIFEST_DIR / "f0_text.json"
TARGET_SR, FMIN, FMAX = 16000, 100.0, 1200.0


def f0_segments(wav_path) -> list[float]:
    """Median Hz of each sustained (voiced) segment, in time order."""
    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
    f0, _, _ = librosa.pyin(y, sr=TARGET_SR, fmin=FMIN, fmax=FMAX,
                            frame_length=2048, hop_length=160)
    voiced = np.isfinite(f0)
    segs, cur = [], []
    for i, v in enumerate(voiced):
        if v:
            cur.append(f0[i])
        elif len(cur) >= 5:
            segs.append(round(float(np.median(cur)), 1)); cur = []
        else:
            cur = []
    if len(cur) >= 5:
        segs.append(round(float(np.median(cur)), 1))
    return segs


def text_for_segments(segs) -> str:
    if not segs:
        return "Pitch-tracker readout: no clear sustained pitch detected."
    return ("Pitch-tracker readout (measured sustained fundamental frequencies "
            "in Hz, in time order): " + ", ".join(f"{h:.1f}" for h in segs) + ".")


def build_map(tasks=DEFAULT_TASKS) -> dict:
    import pandas as pd
    man = pd.read_parquet(MANIFEST_PATH)
    man = man[man.task.isin(tasks)]
    out = {}
    for i, r in enumerate(man.itertuples(), 1):
        try:
            out[r.audio_path] = text_for_segments(f0_segments(EXP_ROOT / r.audio_path))
        except Exception as e:
            out[r.audio_path] = "Pitch-tracker readout: no clear sustained pitch detected."
        if i % 100 == 0:
            print(f"  f0_text {i}/{len(man)}")
    F0_TEXT_PATH.write_text(json.dumps(out))
    print(f"[f0_text] wrote {F0_TEXT_PATH} ({len(out)} stimuli)")
    return out


def load_map() -> dict:
    if F0_TEXT_PATH.exists():
        return json.loads(F0_TEXT_PATH.read_text())
    return build_map()


if __name__ == "__main__":
    build_map()
