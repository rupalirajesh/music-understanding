"""Track D, Phase 1 (RESEARCH_PLAN.md §12.2): render a log-mel spectrogram
PNG for every stimulus in the manifest — the input image for the
audio+spectrogram-image experiment.

Runs on the laptop, no GPU needed (same category as 01_generate_stimuli.py:
local, deterministic, CPU-only). Local run confirmed working 2026-07-24.

  python scripts/10_render_spectrograms.py

Deliberately uses the SAME log-mel parameters as Whisper's front end (§0.2 /
§1.1: 128 mel channels, 25ms window, 10ms hop) rather than inventing new
ones — the point of Phase 1 is testing whether a *different pretrained
pathway* (vision) reads the same transform better, not handing the model a
different transform altogether.

Output path is DERIVED from audio_path via musicprobe.spectrograms
.spectrogram_path() (single source of truth, also used by image_jobs.py) —
not stored as a new manifest column, since manifests/stimuli.parquet is
frozen (PROJECT_STATE.md decision 10) and adding a column would be a schema
change.

Images live in stimuli/spectrograms/ (gitignored, same as stimuli/ itself —
regenerable from the WAVs, which are themselves regenerable from seeded
synthesis).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")  # headless — no display needed, just write PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from musicprobe.spectrograms import spectrogram_path

N_MELS = 128
WIN_MS, HOP_MS = 25, 10
TARGET_SR = 16000  # matches what these models actually resample to (§0.2/§1.1)


def render_one(wav_path: Path, out_path: Path) -> None:
    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)

    win_length = int(TARGET_SR * WIN_MS / 1000)
    hop_length = int(TARGET_SR * HOP_MS / 1000)
    n_fft = 1 << (win_length - 1).bit_length()  # next pow2 >= win_length
    mel = librosa.feature.melspectrogram(
        y=y, sr=TARGET_SR, n_fft=n_fft, win_length=win_length,
        hop_length=hop_length, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
    librosa.display.specshow(log_mel, sr=TARGET_SR, hop_length=hop_length,
                             x_axis="time", y_axis="mel", ax=ax, cmap="magma")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main(manifest: str, exp_root: str, force: bool, limit: int | None = None):
    root = Path(exp_root)
    man = pd.read_parquet(manifest)
    if limit:
        man = man.head(limit)
    done = skipped = failed = 0
    for i, row in enumerate(man.itertuples(), 1):
        out_rel = spectrogram_path(row.audio_path)
        out_path = root / out_rel
        if out_path.exists() and not force:
            skipped += 1
            continue
        try:
            render_one(root / row.audio_path, out_path)
            done += 1
        except Exception as e:
            failed += 1
            print(f"!! failed on {row.stimulus_id}: {type(e).__name__}: {e}")
        if i % 100 == 0:
            print(f"{i}/{len(man)} (rendered {done}, skipped {skipped}, failed {failed})")
    print(f"done: rendered {done}, skipped {skipped} (already existed), failed {failed}, "
          f"out of {len(man)} stimuli")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifests/stimuli.parquet")
    ap.add_argument("--exp-root", default=".")
    ap.add_argument("--force", action="store_true", help="re-render even if PNG exists")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(args.manifest, args.exp_root, args.force, args.limit)
