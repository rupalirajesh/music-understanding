"""Render the Tracks R/S/T/U/V/W rhythm-cluster front-end images
(musicprobe/rhythm_repr.py). All six are computed from the AUDIO via
librosa onset/tempogram functions, never from the ground-truth BPM or
beats-per-bar label.

  python scripts/render_rhythm_repr.py --kind tempogram
  python scripts/render_rhythm_repr.py --kind all --limit 20 --force

CPU-only, deterministic (same category as render_harmony_repr.py).
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
from musicprobe.rhythm_repr import (tempogram_path, tempogram_picked_path, onset_line_path,
                                     onset_line_zoom_path, rhythm_roll_path, rhythm_necklace_path)

TARGET_SR = 22050
HOP = 512
HOP_ZOOM = 128
TOP_K = 3


def _load(wav_path: Path, sr=TARGET_SR):
    y, orig_sr = sf.read(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return librosa.resample(y.astype(np.float32), orig_sr=orig_sr, target_sr=sr)


def _onset_env(y, sr, hop):
    return librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)


def render_tempogram(wav_path: Path, out_path: Path) -> None:
    y = _load(wav_path)
    env = _onset_env(y, TARGET_SR, HOP)
    tg = librosa.feature.tempogram(onset_envelope=env, sr=TARGET_SR, hop_length=HOP)
    t = librosa.times_like(tg, sr=TARGET_SR, hop_length=HOP)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
    ax.imshow(tg, aspect="auto", origin="lower", cmap="magma",
              extent=[t[0] if len(t) else 0, t[-1] if len(t) else 1, 0, tg.shape[0]])
    ax.set_xlabel("time (s)"); ax.set_ylabel("lag (frames)")
    ax.set_title("tempogram", fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def render_tempogram_picked(wav_path: Path, out_path: Path, silence_frac: float = 0.02) -> None:
    y = _load(wav_path)
    env = _onset_env(y, TARGET_SR, HOP)
    tg = librosa.feature.tempogram(onset_envelope=env, sr=TARGET_SR, hop_length=HOP)
    picked = np.zeros_like(tg)
    k = min(TOP_K, tg.shape[0])
    frame_energy = tg.sum(axis=0)
    active = frame_energy > frame_energy.max() * silence_frac if frame_energy.max() > 0 else \
        np.zeros(tg.shape[1], dtype=bool)
    idx = np.argsort(tg, axis=0)[-k:, :]
    for col in np.where(active)[0]:
        picked[idx[:, col], col] = 1.0
    t = librosa.times_like(tg, sr=TARGET_SR, hop_length=HOP)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
    ax.imshow(picked, aspect="auto", origin="lower", cmap="gray_r",
              extent=[t[0] if len(t) else 0, t[-1] if len(t) else 1, 0, tg.shape[0]],
              vmin=0, vmax=1)
    ax.set_xlabel("time (s)"); ax.set_ylabel("lag (frames)")
    ax.set_title("peak-picked tempogram", fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def render_onset_line(wav_path: Path, out_path: Path, hop: int) -> None:
    y = _load(wav_path)
    env = _onset_env(y, TARGET_SR, hop)
    t = librosa.times_like(env, sr=TARGET_SR, hop_length=hop)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 2.5) if hop == HOP_ZOOM else (6, 2.5), dpi=150)
    ax.plot(t, env, color="#1a5276", lw=1)
    ax.set_xlabel("time (s)"); ax.set_ylabel("onset strength")
    ax.set_title("onset-strength envelope" + (" -- zoomed" if hop == HOP_ZOOM else ""),
                fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def _detect_click_period(y, sr, hop=HOP):
    """Median inter-onset interval from librosa's own onset detector.
    Tried onset-envelope autocorrelation first (same method as
    musicprobe.l1_baselines.tempo_estimate) plus an octave-error guard, but
    debugging on real beats_per_bar stimuli showed it still regularly locks
    onto a sub-harmonic of the true click rate (alternating accents create
    a strong periodicity at 2x/3x the real interval that can beat the
    fundamental in the raw autocorrelation). Onset detection itself is
    reliable on these clean click stimuli (verified: 24/24 correctly and
    evenly spaced onsets on a real beats=5 example) -- the median gap
    between DETECTED onsets is a far more direct, robust estimate of the
    click period than re-deriving it from the envelope."""
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, units="time")
    if len(onsets) < 2:
        return None
    return float(np.median(np.diff(onsets)))


def render_rhythm_roll(wav_path: Path, out_path: Path) -> None:
    y = _load(wav_path)
    onset_frames = librosa.onset.onset_detect(y=y, sr=TARGET_SR, hop_length=HOP)
    onset_times = librosa.frames_to_time(onset_frames, sr=TARGET_SR, hop_length=HOP)
    period = _detect_click_period(y, TARGET_SR)
    duration = len(y) / TARGET_SR
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=150)
    if len(onset_times):
        ax.vlines(onset_times, 0, 1, color="#1a5276", lw=1.5)
    if period and period > 0:
        grid = np.arange(onset_times[0] if len(onset_times) else 0, duration, period)
        ax.vlines(grid, 0, 1, color="#c0392b", lw=0.5, alpha=0.4, linestyle="--")
    ax.set_xlabel("time (s)"); ax.set_yticks([])
    ax.set_title("rhythm-roll (onsets vs. detected pulse grid)", fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def render_rhythm_necklace(wav_path: Path, out_path: Path) -> None:
    y = _load(wav_path)
    onset_frames = librosa.onset.onset_detect(y=y, sr=TARGET_SR, hop_length=HOP)
    onset_times = librosa.frames_to_time(onset_frames, sr=TARGET_SR, hop_length=HOP)
    click_period = _detect_click_period(y, TARGET_SR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150, subplot_kw={"projection": "polar"})
    if click_period and len(onset_times) >= 2:
        env = _onset_env(y, TARGET_SR, HOP)
        env_t = librosa.times_like(env, sr=TARGET_SR, hop_length=HOP)
        strength = np.interp(onset_times, env_t, env)  # per-onset accent strength
        # candidate bar (cycle) lengths: small integer multiples of the click
        # period (TASKS.md 1.6's stimuli cycle every 3-7 clicks). A uniform
        # click train folded mod ANY integer multiple looks equally
        # "concentrated" in plain onset-timing terms, so timing alone can't
        # tell n=3 from n=6 -- pick the n whose phase distribution is most
        # concentrated when WEIGHTED by accent strength instead (a real
        # downbeat accent should land at a consistent phase across bars;
        # a wrong n scatters the loud onsets across different phases). This
        # is a weighted mean-resultant-length statistic (circular stats),
        # not just "which n gives evenly spaced points."
        best_n, best_score, best_cycle = 4, -1.0, click_period * 4
        for n in range(3, 8):
            cycle = click_period * n
            phases = 2 * np.pi * (onset_times % cycle) / cycle
            w = strength / strength.sum()
            score = np.hypot((w * np.cos(phases)).sum(), (w * np.sin(phases)).sum())
            if score > best_score:
                best_score, best_n, best_cycle = score, n, cycle
        theta = 2 * np.pi * (onset_times % best_cycle) / best_cycle
        sizes = 30 + 150 * (strength - strength.min()) / (np.ptp(strength) + 1e-9)
        ax.scatter(theta, np.ones_like(theta), s=sizes, color="#1a5276", alpha=0.8)
        ax.plot(np.append(theta, theta[0]), np.append(np.ones_like(theta), 1),
               color="#1a5276", lw=1, alpha=0.4)
    ax.set_yticks([]); ax.set_title("rhythm necklace", fontsize=9, pad=15)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


RENDERERS = {
    "tempogram": render_tempogram,
    "tempogram_picked": render_tempogram_picked,
    "onset_line": lambda w, o: render_onset_line(w, o, HOP),
    "onset_line_zoom": lambda w, o: render_onset_line(w, o, HOP_ZOOM),
    "rhythm_roll": render_rhythm_roll,
    "rhythm_necklace": render_rhythm_necklace,
}
PATH_FNS = {
    "tempogram": tempogram_path, "tempogram_picked": tempogram_picked_path,
    "onset_line": onset_line_path, "onset_line_zoom": onset_line_zoom_path,
    "rhythm_roll": rhythm_roll_path, "rhythm_necklace": rhythm_necklace_path,
}
RHYTHM_TASKS = ("tempo_bpm", "beats_per_bar")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="all", choices=list(RENDERERS) + ["all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--scope", default="battery", choices=["battery", "rhythm-only"],
                    help="'battery' (default) renders EVERY stimulus, not just the 2 "
                         "rhythm tasks -- required so build_image_jobs' wrong_image "
                         "control (drawn from the whole battery, image_jobs.py) doesn't "
                         "hit a missing-file error. 'rhythm-only' is for quick local "
                         "smoke-testing only, not for building a real training manifest.")
    args = ap.parse_args()

    man = pd.read_parquet(MANIFEST_PATH)
    if args.scope == "rhythm-only":
        man = man[man.task.isin(RHYTHM_TASKS)]
    if args.limit:
        man = man.head(args.limit)

    kinds = list(RENDERERS) if args.kind == "all" else [args.kind]
    for kind in kinds:
        render_fn, path_fn = RENDERERS[kind], PATH_FNS[kind]
        n_done = n_skip = n_err = 0
        for row in man.itertuples():
            out_path = EXP_ROOT / path_fn(row.audio_path)
            if out_path.exists() and not args.force:
                n_skip += 1
                continue
            try:
                render_fn(EXP_ROOT / row.audio_path, out_path)
                n_done += 1
            except Exception as e:
                n_err += 1
                print(f"  [{kind}] ERROR on {row.audio_path}: {e}")
        print(f"[{kind}] rendered {n_done}, skipped {n_skip}, errors {n_err} "
              f"(of {len(man)} rhythm stimuli)")


if __name__ == "__main__":
    main()
