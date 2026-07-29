"""Frame-level pitch features for the pitch-STREAM fusion experiment (Track F).

Unlike f0_text (a summary string) or the rendered image, this is a task-agnostic
FRAME-LEVEL contour fed straight into the LM's embedding space via a learned
projector — the scalable, end-to-end 'add a pitch channel' fix. Each stimulus ->
a fixed (K, 2) array: [normalised log-pitch, voiced flag] over K time bins.

Cached to manifests/pitch_feats.npz so training/eval don't re-run pyin.
"""
import numpy as np
import soundfile as sf
import librosa

from .config import EXP_ROOT, MANIFEST_DIR, MANIFEST_PATH
from .image_jobs import DEFAULT_TASKS

PITCH_FEATS_PATH = MANIFEST_DIR / "pitch_feats.npz"
TARGET_SR, FMIN, FMAX = 16000, 100.0, 1200.0
K = 24                       # number of pitch tokens per stimulus
REF_HZ = 220.0               # normalise cents relative to A3


def pitch_feature(wav_path) -> np.ndarray:
    """(K, 2) float32: [log-pitch in octaves rel. 220 Hz (0 if unvoiced), voiced]."""
    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
    f0, _, _ = librosa.pyin(y, sr=TARGET_SR, fmin=FMIN, fmax=FMAX,
                            frame_length=2048, hop_length=160)
    n = len(f0)
    out = np.zeros((K, 2), np.float32)
    if n == 0:
        return out
    edges = np.linspace(0, n, K + 1).astype(int)
    for i in range(K):
        seg = f0[edges[i]:max(edges[i] + 1, edges[i + 1])]
        v = seg[np.isfinite(seg)]
        if len(v):
            out[i, 0] = np.log2(np.median(v) / REF_HZ)   # octaves rel. 220 Hz
            out[i, 1] = 1.0
    return out


def build_map(tasks=DEFAULT_TASKS) -> dict:
    import pandas as pd
    man = pd.read_parquet(MANIFEST_PATH)
    man = man[man.task.isin(tasks)]
    out = {}
    for i, r in enumerate(man.itertuples(), 1):
        try:
            out[r.audio_path] = pitch_feature(EXP_ROOT / r.audio_path)
        except Exception:
            out[r.audio_path] = np.zeros((K, 2), np.float32)
        if i % 100 == 0:
            print(f"  pitch_feats {i}/{len(man)}")
    np.savez_compressed(PITCH_FEATS_PATH, **{k.replace("/", "__"): v for k, v in out.items()})
    print(f"[pitch_feats] wrote {PITCH_FEATS_PATH} ({len(out)} stimuli, K={K})")
    return out


def load_map() -> dict:
    if not PITCH_FEATS_PATH.exists():
        return build_map()
    z = np.load(PITCH_FEATS_PATH)
    return {k.replace("__", "/"): z[k] for k in z.files}


if __name__ == "__main__":
    build_map()
