"""Render an F0-CONTOUR line-chart PNG per stimulus — the "readable" image for
Track D's make-the-model-use-the-image experiments (see musicprobe/f0_contour.py).

Pitch is extracted with librosa.pyin (monophonic) and drawn as pitch-over-time
on a LOG-frequency y-axis, so pitch = vertical position — what a chart-reading
vision encoder handles well, unlike a spectrogram texture. Light semitone
gridlines give an absolute reference (the thing the audio path can't recover).

  python scripts/render_f0_contours.py
  python scripts/render_f0_contours.py --limit 20 --force

CPU-only, deterministic (same category as render_spectrograms.py).
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
from musicprobe.f0_contour import f0_contour_path

TARGET_SR = 16000
FMIN, FMAX = 100.0, 1200.0        # covers our E3..E5 stimuli with headroom
YLIM = (90.0, 1300.0)             # fixed axis -> absolute pitch is comparable across images


def render_one(wav_path: Path, out_path: Path) -> None:
    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
    f0, voiced, _ = librosa.pyin(y, sr=TARGET_SR, fmin=FMIN, fmax=FMAX,
                                 frame_length=2048, hop_length=160)
    t = librosa.times_like(f0, sr=TARGET_SR, hop_length=160)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
    # semitone gridlines from FMIN..FMAX as an absolute pitch reference
    m0, m1 = int(librosa.hz_to_midi(FMIN)), int(librosa.hz_to_midi(FMAX))
    for m in range(m0, m1 + 1):
        ax.axhline(librosa.midi_to_hz(m), color="0.9", lw=0.5, zorder=0)
    ax.plot(t, f0, color="#0a4fd6", lw=2.4, solid_capstyle="round")
    ax.set_yscale("log")
    ax.set_ylim(*YLIM)
    ax.set_xlim(0, t[-1] if len(t) else 1)
    ax.set_ylabel("pitch (Hz)")
    ax.set_xlabel("time (s)")
    ax.set_yticks([110, 220, 440, 880])
    ax.set_yticklabels(["110", "220", "440", "880"])
    ax.grid(False)
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
        out_path = root / f0_contour_path(row.audio_path)
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
    print(f"done: rendered {done}, skipped {skipped}, failed {failed}, out of {len(man)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    ap.add_argument("--exp-root", default=str(EXP_ROOT))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tasks", nargs="*",
                    default=["octave_id", "tuning_judgment", "cents_discrimination", "note_count"])
    args = ap.parse_args()
    main(args.manifest, args.exp_root, args.force, args.limit, args.tasks)
