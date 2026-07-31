"""Render a CHROMAGRAM heatmap PNG per stimulus — the visual front-end for the
harmonic-task cluster (key_id/mode_id/chord_quality/interval_id) that Tracks
C-F never targeted (see musicprobe/chromagram.py).

12 pitch-class rows x time, energy from librosa's CQT-based chroma (musically
aligned: equal resolution per semitone, same reasoning that makes MERT/CQT the
L2 upper bound elsewhere in this project — see RESEARCH_PLAN.md S:1.2). Axis
is labelled with pitch-class NAMES (C, C#, D, ...) — fixed for every stimulus,
not tied to this stimulus's actual tonic/root/quality, so labelling the axis
does not leak the answer (same non-leakage argument as the F0-contour's
semitone gridlines).

  python scripts/render_chromagrams.py
  python scripts/render_chromagrams.py --limit 20 --force

CPU-only, deterministic (same category as render_f0_contours.py).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import pandas as pd

from musicprobe.config import EXP_ROOT, MANIFEST_PATH
from musicprobe.chromagram import chromagram_path

TARGET_SR = 22050    # librosa's CQT default; plenty for our E2..E6-ish stimuli
HOP = 512
PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def render_one(wav_path: Path, out_path: Path) -> None:
    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
    chroma = librosa.feature.chroma_cqt(y=y, sr=TARGET_SR, hop_length=HOP)
    t = librosa.times_like(chroma, sr=TARGET_SR, hop_length=HOP)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
    ax.imshow(chroma, aspect="auto", origin="lower", cmap="magma",
              extent=[0, t[-1] if len(t) else 1, -0.5, 11.5], vmin=0, vmax=1)
    ax.set_yticks(range(12))
    ax.set_yticklabels(PC_NAMES, fontsize=7)
    ax.set_ylabel("pitch class")
    ax.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(manifest, exp_root, force, limit, tasks):
    root = Path(exp_root)
    man = pd.read_parquet(manifest)
    if tasks:
        man = man[man.task.isin(tasks)]
    if limit:
        man = man.head(limit)
    done = skipped = failed = 0
    for i, row in enumerate(man.itertuples(), 1):
        out_path = root / chromagram_path(row.audio_path)
        if out_path.exists() and not force:
            skipped += 1
            continue
        try:
            render_one(root / row.audio_path, out_path)
            done += 1
        except Exception as e:
            failed += 1
            print(f"!! failed on {row.stimulus_id}: {type(e).__name__}: {e}")
        if i % 50 == 0:
            print(f"{i}/{len(man)} (rendered {done}, skipped {skipped}, failed {failed})")
    print(f"done: rendered {done}, skipped {skipped}, failed {failed}, out of {len(man)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    ap.add_argument("--exp-root", default=str(EXP_ROOT))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tasks", nargs="*",
                    default=["key_id", "mode_id", "chord_quality", "interval_id"])
    args = ap.parse_args()
    main(args.manifest, args.exp_root, args.force, args.limit, args.tasks)
